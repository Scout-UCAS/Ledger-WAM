# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import os
import sys
from pathlib import Path
import wandb

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from safetensors.torch import save_file
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model, apply_ac
from distributed.util import (
    _configure_model, 
    init_distributed, 
    dist_mean, 
    dist_max
)
from einops import rearrange
from modules.utils import (
    load_transformer,
)
from utils import (
    init_logger, 
    logger, 
    get_mesh_id, 
    sample_timestep_id,
    data_seq_to_patch,
    warmup_constant_lambda,
    FlowMatchScheduler
)

from dataset import MultiLatentLeRobotDataset
import gc

try:
    from wan_va.ledger.neural import (
        build_repair_debt_reward_targets,
        compute_ledger_losses,
    )
    from wan_va.ledger.protocol import validate_repair_catalog
except ImportError:  # Script-style execution with ``wan_va`` on sys.path.
    from ledger.neural import (
        build_repair_debt_reward_targets,
        compute_ledger_losses,
    )
    from ledger.protocol import validate_repair_catalog


class Trainer:
    def __init__(self, config):
        if config.enable_wandb and config.rank == 0:
            wandb.login(host=os.environ['WANDB_BASE_URL'], key=os.environ['WANDB_API_KEY'])
            self.wandb = wandb
            self.wandb.init(
                entity=os.environ["WANDB_TEAM_NAME"],
                project=os.getenv("WANDB_PROJECT", "va_robotwin"),
                # dir=log_dir,
                config=config,
                mode="online",
                name='test_lln'
                # name=os.path.basename(os.path.normpath(job_config.job.dump_folder))
            )
            logger.info("WandB logging enabled")
        self.step = 0
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size
        self.ledger_enabled = bool(getattr(config, "ledger_enabled", False))
        if self.ledger_enabled:
            validate_repair_catalog(tuple(config.ledger_repair_catalog))
        if self.ledger_enabled and len(config.ledger_rollback_stage_ontology) != int(
            config.ledger_max_rollback_stages):
            raise ValueError(
                "ledger_rollback_stage_ontology length must equal "
                "ledger_max_rollback_stages")

        # Load models
        logger.info("Loading models...")

        # Load and shard transformer with FSDP
        logger.info("Loading transformer...")

        if hasattr(config, 'resume_from') and config.resume_from:
            transformer_path = os.path.join(config.resume_from, 'transformer')
            if config.rank == 0:
                logger.info(f"Resuming from checkpoint: {transformer_path}")
        else:
            transformer_path = os.path.join(config.wan22_pretrained_model_name_or_path, 'transformer')

        transformer_overrides = {"attn_mode": "flex"}
        if self.ledger_enabled:
            repair_catalog = getattr(config, "ledger_repair_catalog", ())
            transformer_overrides.update({
                "enable_ledger": True,
                # Diffusers cannot materialize newly added parameters from a
                # legacy checkpoint while low-memory/meta loading is active.
                "low_cpu_mem_usage": False,
                "ledger_hidden_dim": int(config.ledger_hidden_dim),
                "ledger_num_claim_slots": int(config.ledger_max_claims),
                "ledger_num_relations": int(config.ledger_num_relations),
                "ledger_max_rollback_steps": int(
                    config.ledger_max_rollback_stages),
                "ledger_num_repair_actions": len(repair_catalog),
                "ledger_delta_dim": int(config.ledger_delta_dim),
                "ledger_num_heads": int(config.ledger_num_heads),
                "ledger_dropout": float(config.ledger_dropout),
                "ledger_num_claim_types": int(config.ledger_num_claim_types),
                "ledger_num_subjects": int(config.ledger_num_subjects),
                "ledger_num_objects": int(config.ledger_num_objects),
                "ledger_num_preconditions": int(
                    config.ledger_num_preconditions),
                "ledger_num_effects": int(config.ledger_num_effects),
            })

        self.transformer = load_transformer(
            transformer_path,
            torch_dtype=torch.float32,
            torch_device='cpu',
            **transformer_overrides,
        )

        logger.info("Setting up activation checkpointing ...")
        apply_ac(self.transformer)

        logger.info("Setting up FSDP...")
        shard_fn = shard_model
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_fn,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )
        self.transformer.train()
        self.transformer.requires_grad_(True)

        # Optimizer.  Ledger parameters use their own group so the optional
        # head-only warm-up can hold the large WAM backbone at zero learning
        # rate without removing parameters from the optimizer/checkpoint.
        trainable_named_params = [
            (name, parameter)
            for name, parameter in self.transformer.named_parameters()
            if parameter.requires_grad
        ]
        if self.ledger_enabled:
            ledger_params = [
                parameter
                for name, parameter in trainable_named_params
                if "ledger_head" in name or "ledger_condition_proj" in name
            ]
            if not ledger_params:
                raise RuntimeError(
                    "ledger_enabled=True but the transformer has no Ledger-WAM "
                    "parameters; verify the checkpoint/model overrides"
                )
            ledger_param_ids = {id(parameter) for parameter in ledger_params}
            backbone_params = [
                parameter
                for _name, parameter in trainable_named_params
                if id(parameter) not in ledger_param_ids
            ]
            optimizer_params = [
                {"params": backbone_params},
                {"params": ledger_params},
            ]
        else:
            optimizer_params = [
                parameter for _name, parameter in trainable_named_params
            ]

        self.optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
            foreach=False,
        )

        def base_lr_lambda(step):
            return warmup_constant_lambda(
                step, warmup_steps=config.warmup_steps)
        if self.ledger_enabled:
            ledger_warmup_steps = int(
                getattr(config, "ledger_head_warmup_steps", 0))
            freeze_backbone = bool(getattr(
                config, "ledger_freeze_backbone_during_warmup", False))

            def backbone_lr_lambda(step):
                if freeze_backbone and step < ledger_warmup_steps:
                    return 0.0
                return base_lr_lambda(step)

            lr_lambda = [backbone_lr_lambda, base_lr_lambda]
        else:
            lr_lambda = base_lr_lambda
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda)

        # Setup dataloaders
        logger.info("Setting up datasets...")
        train_dataset = MultiLatentLeRobotDataset(config=config)
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=config.world_size,
            rank=config.rank,
            shuffle=True,
            seed=42
        ) if config.world_size > 1 else None
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None), 
            num_workers=config.load_worker,
            sampler=train_sampler,
        )

        self.train_scheduler_latent = FlowMatchScheduler(shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_action.set_timesteps(1000, training=True)

        self.save_dir = Path(config.save_root) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        self.train_loader_iter = None
        if hasattr(config, 'resume_from') and config.resume_from:
            self._load_training_state(config.resume_from)
    
    def _get_next_batch(self):
        """Get next batch from iterator, reset if epoch is finished."""
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)
        
        try:
            batch = next(self.train_loader_iter)
        except StopIteration:
            # Reset sampler and iterator when epoch finishes
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(self.train_loader.sampler.epoch + 1)
            self.train_loader_iter = iter(self.train_loader)
            batch = next(self.train_loader_iter)
        
        return batch

    @torch.no_grad()
    def _add_noise(self, latent, train_scheduler, action_mask=False, action_mode=False, noisy_cond_prob=0.):
        B, C, F, H, W = latent.shape

        timestep_ids = sample_timestep_id(batch_size=F, num_train_timesteps=train_scheduler.num_train_timesteps)
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noisy_latents =train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets =train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1
        
        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,  # F
            latent.shape[-2] // patch_h,  # H
            latent.shape[-1] // patch_w,  # W
            t=1 if action_mode else 0,  # 1 for action mode (0 for latent), not used
            f_w=1,
            f_shift=0,
            action=action_mode
        ).to(self.device)  # shape: [4, seq_len]
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                    batch_size=F,
                    min_timestep_bd=0.5, 
                    max_timestep_bd=1.0, 
                    num_train_timesteps=train_scheduler.num_train_timesteps,
                )
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(device=self.device)
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()

        return dict(
            timesteps=timesteps[None].repeat(B, 1),
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps[None].repeat(B, 1),
            grid_id=latent_grid_id,
        )

    @torch.no_grad()
    def _prepare_counterfactual_actions(self, batch_dict, observed_actions):
        """Choose one annotated counterfactual, or synthesize a safe negative."""
        batch_size, action_dim = observed_actions.shape[:2]
        selected_indices = torch.zeros(
            batch_size, dtype=torch.long, device=observed_actions.device)
        explicit = torch.zeros(
            batch_size, dtype=torch.bool, device=observed_actions.device)
        counterfactual = observed_actions.clone()

        raw_actions = batch_dict.get("ledger_counterfactual_actions")
        raw_action_mask = batch_dict.get("ledger_counterfactual_action_mask")
        raw_candidate_mask = batch_dict.get("ledger_counterfactual_mask")
        if (
            torch.is_tensor(raw_actions)
            and raw_actions.ndim == 3
            and raw_actions.shape[1] > 0
            and raw_actions.shape[2] == action_dim
        ):
            if raw_actions.shape[0] != batch_size:
                raise ValueError(
                    "ledger_counterfactual_actions batch size must match observed "
                    "actions"
                )
            if raw_action_mask is None:
                raw_action_mask = torch.isfinite(raw_actions) & raw_actions.ne(
                    float(getattr(self.config, "ledger_ignore_index", -100))
                )
            elif raw_action_mask.shape != raw_actions.shape:
                raise ValueError(
                    "ledger_counterfactual_action_mask must match "
                    "ledger_counterfactual_actions"
                )
            if (
                raw_candidate_mask is not None
                and raw_candidate_mask.shape[:2] != raw_actions.shape[:2]
            ):
                raise ValueError(
                    "ledger_counterfactual_mask must have shape [batch, candidates]"
                )
            candidate_valid = raw_action_mask.any(dim=-1)
            if raw_candidate_mask is not None:
                candidate_valid = candidate_valid & raw_candidate_mask.bool()
            explicit = candidate_valid.any(dim=1)
            selected_indices = candidate_valid.float().argmax(dim=1)
            batch_indices = torch.arange(batch_size, device=raw_actions.device)
            selected_vector = raw_actions[batch_indices, selected_indices]
            selected_mask = raw_action_mask[
                batch_indices, selected_indices].bool()
            observed_vector = observed_actions[:, :, -1, -1, 0]
            selected_vector = torch.where(
                selected_mask, selected_vector, observed_vector)
            counterfactual[:, :, -1, -1, 0] = torch.where(
                explicit[:, None], selected_vector, observed_vector)

        sample_probability = float(getattr(
            self.config, "ledger_counterfactual_sample_probability", 0.0))
        synthesize = (~explicit) & (
            torch.rand(batch_size, device=observed_actions.device)
            < sample_probability)
        current_action = observed_actions[:, :, -1, -1, 0]
        action_mask = batch_dict.get("actions_mask")
        if action_mask is None:
            current_action_mask = torch.ones_like(current_action, dtype=torch.bool)
        else:
            if (
                action_mask.ndim != observed_actions.ndim
                or action_mask.shape[0] != batch_size
                or action_mask.shape[1] != action_dim
                or action_mask.shape[-1] != observed_actions.shape[-1]
            ):
                raise ValueError(
                    "actions_mask must have shape [batch, action_dim, frames, "
                    "actions_per_frame, 1]"
                )
            current_action_mask = action_mask[:, :, -1, -1, 0].bool()
        synthetic_action = torch.where(
            current_action_mask,
            torch.clamp(-current_action + 0.05, -1.5, 1.5),
            current_action,
        )
        counterfactual[:, :, -1, -1, 0] = torch.where(
            synthesize[:, None],
            synthetic_action,
            counterfactual[:, :, -1, -1, 0],
        )
        return counterfactual, explicit, synthesize, selected_indices

    @staticmethod
    def _first_valid_label(values, mask, ignore_index=-100):
        valid = mask.bool()
        indices = valid.float().argmax(dim=1)
        selected = values.gather(1, indices[:, None]).squeeze(1)
        fill = torch.full_like(selected, ignore_index)
        return torch.where(valid.any(dim=1), selected, fill), valid.any(dim=1)

    @torch.no_grad()
    def _prepare_ledger_targets(
        self,
        batch_dict,
        explicit_counterfactual,
        synthetic_counterfactual,
        counterfactual_indices,
    ):
        targets = {}
        masks = {}
        claim_slot_mask = batch_dict.get("ledger_claim_mask")
        ledger_available = batch_dict.get("ledger_available")
        if claim_slot_mask is not None:
            targets["presence"] = claim_slot_mask.float()
            if ledger_available is None:
                masks["presence"] = torch.ones_like(
                    claim_slot_mask, dtype=torch.bool)
            else:
                masks["presence"] = ledger_available.bool()[:, None].expand_as(
                    claim_slot_mask)
        task_stems = (
            "claim",
            "claim_type",
            "subject",
            "object",
            "precondition",
            "effect",
            "evidence",
            "dependency",
            "debt",
            "relation",
            "rollback",
            "uncertainty",
            "repair_cost",
            "observability",
            "importance",
        )
        categorical_bounds = {
            "claim_type": int(self.config.ledger_num_claim_types),
            "subject": int(self.config.ledger_num_subjects),
            "object": int(self.config.ledger_num_objects),
            "relation": int(self.config.ledger_num_relations),
            "precondition": int(self.config.ledger_num_preconditions),
            "effect": int(self.config.ledger_num_effects),
            "rollback": int(self.config.ledger_max_rollback_stages),
        }
        for stem in task_stems:
            value = batch_dict.get("ledger_{}_labels".format(stem))
            mask = batch_dict.get("ledger_{}_mask".format(stem))
            if value is not None and mask is not None:
                target_value = value
                if stem == "repair_cost":
                    cost_scale = float(getattr(
                        self.config, "ledger_repair_cost_scale", 1.0))
                    if cost_scale <= 0:
                        raise ValueError("ledger_repair_cost_scale must be positive")
                    target_value = (value.float() / cost_scale).clamp(0.0, 1.0)
                targets[stem] = target_value
                masks[stem] = mask.bool()
                if stem in categorical_bounds:
                    invalid = masks[stem] & (
                        value.lt(0) | value.ge(categorical_bounds[stem]))
                    if invalid.any() and bool(getattr(
                        self.config, "ledger_strict", False)):
                        raise ValueError(
                            "Ledger label {!r} is outside [0, {})".format(
                                stem, categorical_bounds[stem]))
                    masks[stem] = masks[stem] & ~invalid

        dependency_matrix = batch_dict.get("ledger_dependency_matrix")
        dependency_matrix_mask = batch_dict.get(
            "ledger_dependency_matrix_mask")
        if dependency_matrix is not None and dependency_matrix_mask is not None:
            targets["dependency_matrix"] = dependency_matrix
            masks["dependency_matrix"] = dependency_matrix_mask.bool()

        repair_catalog = tuple(getattr(
            self.config, "ledger_repair_catalog", ()))
        repair_action_cost_target = None
        repair_task_risk_target = None
        if repair_catalog:
            action_cost = torch.tensor(
                [float(entry.get("cost", 0.0)) for entry in repair_catalog],
                dtype=batch_dict["latents"].dtype,
                device=batch_dict["latents"].device,
            ).clamp(0.0, 1.0)
            action_cost = action_cost[None].expand(
                batch_dict["latents"].shape[0], -1)
            task_risk = torch.tensor(
                [float(entry.get("risk", 0.0)) for entry in repair_catalog],
                dtype=batch_dict["latents"].dtype,
                device=batch_dict["latents"].device,
            ).clamp(0.0, 1.0)
            task_risk = task_risk[None].expand(
                batch_dict["latents"].shape[0], -1)
            repair_action_cost_target = action_cost
            repair_task_risk_target = task_risk
            targets["action_cost"] = action_cost
            targets["repair_task_risk"] = task_risk
            masks["action_cost"] = torch.ones_like(
                action_cost, dtype=torch.bool)

        # The repair policy is ledger-level.  Sidecars annotate repair skills
        # per claim, so select the action attached to the highest-debt valid
        # claim (falling back to the first valid label).
        repair_labels = batch_dict.get("ledger_repair_action_labels")
        repair_mask = batch_dict.get("ledger_repair_action_mask")
        num_repair_actions = len(getattr(
            self.config, "ledger_repair_catalog", ()))
        selected_repair = None
        selected_repair_valid = None
        if repair_labels is not None and repair_mask is not None:
            label_valid = repair_mask.bool()
            label_valid = label_valid & repair_labels.ge(0) & repair_labels.lt(
                num_repair_actions)
            priority = batch_dict.get("ledger_debt_labels")
            priority_mask = batch_dict.get("ledger_debt_mask")
            if priority is None:
                priority = torch.zeros_like(repair_labels, dtype=torch.float32)
                priority_valid = torch.zeros_like(label_valid)
            else:
                priority = priority.float()
                priority_valid = label_valid & (
                    torch.ones_like(label_valid)
                    if priority_mask is None else priority_mask.bool()
                )
            ranked_priority = priority.masked_fill(
                ~priority_valid, float("-inf"))
            priority_claim = ranked_priority.argmax(dim=1)
            fallback_claim = label_valid.float().argmax(dim=1)
            selected_claim = torch.where(
                priority_valid.any(dim=1), priority_claim, fallback_claim)
            selected_repair = repair_labels.gather(
                1, selected_claim[:, None]).squeeze(1)
            selected_repair_valid = label_valid.any(dim=1)
            selected_repair = torch.where(
                selected_repair_valid,
                selected_repair,
                torch.full_like(selected_repair, -100),
            )
            targets["repair"] = selected_repair
            masks["repair"] = selected_repair_valid

        post_debt = batch_dict.get("ledger_post_repair_debt_labels")
        post_debt_mask = batch_dict.get("ledger_post_repair_debt_mask")
        current_debt = batch_dict.get("ledger_debt_labels")
        current_debt_mask = batch_dict.get("ledger_debt_mask")
        if (
            post_debt is not None
            and post_debt_mask is not None
            and repair_labels is not None
            and repair_mask is not None
            and num_repair_actions > 0
        ):
            batch_size, num_claims = post_debt.shape
            repair_world = torch.full(
                (batch_size, num_repair_actions, num_claims),
                -100.0,
                dtype=post_debt.dtype,
                device=post_debt.device,
            )
            repair_world_mask = torch.zeros_like(
                repair_world, dtype=torch.bool)
            valid = (
                post_debt_mask.bool()
                & repair_mask.bool()
                & repair_labels.ge(0)
                & repair_labels.lt(num_repair_actions)
            )
            for batch_index, claim_index in valid.nonzero(as_tuple=False):
                repair_index = int(repair_labels[batch_index, claim_index])
                repair_world[batch_index, repair_index, claim_index] = post_debt[
                    batch_index, claim_index]
                repair_world_mask[
                    batch_index, repair_index, claim_index] = True
            targets["repair_post_debt"] = repair_world
            masks["repair_world"] = repair_world_mask

            if current_debt is not None and current_debt_mask is not None:
                # Supervise the same signed, importance-normalized global-risk
                # change predicted by ``repair_debt_reduction``.  Claims not
                # targeted by a repair retain their current debt; annotated
                # targets replace only the affected claims.  This permits a
                # failed repair to have a negative reward.
                debt_reward, debt_reward_mask = (
                    build_repair_debt_reward_targets(
                        current_debt=current_debt,
                        current_debt_mask=current_debt_mask,
                        post_repair_debt=post_debt,
                        post_repair_mask=post_debt_mask,
                        repair_labels=repair_labels,
                        repair_label_mask=repair_mask,
                        num_repair_actions=num_repair_actions,
                        claim_mask=batch_dict.get("ledger_claim_mask"),
                        importance=batch_dict.get("ledger_importance_labels"),
                        importance_mask=batch_dict.get(
                            "ledger_importance_mask"),
                        action_cost=repair_action_cost_target,
                        task_risk=repair_task_risk_target,
                        cost_weight=float(getattr(
                            self.config, "ledger_repair_cost_weight", 0.0)),
                        risk_weight=float(getattr(
                            self.config, "ledger_repair_risk_weight", 0.0)),
                    )
                )
                targets["debt_reward"] = debt_reward
                masks["debt_reward"] = debt_reward_mask

        # Counterfactual sidecars store one scalar ledger delta per claim.  The
        # neural objective uses it as an action-effect change indicator.
        claim_mask = batch_dict.get("ledger_claim_mask")
        if claim_mask is not None:
            cf_target = torch.ones_like(claim_mask, dtype=torch.float32)
            effect_mask = batch_dict.get("ledger_effect_mask")
            if effect_mask is None:
                effect_mask = torch.zeros_like(claim_mask, dtype=torch.bool)
            # A synthesized action perturbation is not evidence that every
            # belief changes.  Supervise it only for claims explicitly labeled
            # as action effects; explicit counterfactual deltas below retain
            # their own per-claim masks and zero/non-zero targets.
            cf_mask = (
                synthetic_counterfactual[:, None]
                & claim_mask.bool()
                & effect_mask.bool()
            )
            cf_global_target = torch.ones(
                claim_mask.shape[0],
                dtype=torch.float32,
                device=claim_mask.device,
            )
            cf_global_mask = (
                explicit_counterfactual & claim_mask.bool().any(dim=1)
            ) | (
                synthetic_counterfactual
                & (claim_mask.bool() & effect_mask.bool()).any(dim=1)
            )
            masks["cf_global_slot"] = torch.where(
                explicit_counterfactual[:, None],
                claim_mask.bool(),
                claim_mask.bool() & effect_mask.bool(),
            )
            raw_deltas = batch_dict.get("ledger_counterfactual_deltas")
            raw_delta_mask = batch_dict.get("ledger_counterfactual_delta_mask")
            if (
                raw_deltas is not None
                and raw_delta_mask is not None
                and raw_deltas.shape[1] > 0
            ):
                batch_indices = torch.arange(
                    raw_deltas.shape[0], device=raw_deltas.device)
                selected_delta = raw_deltas[
                    batch_indices, counterfactual_indices]
                selected_delta_mask = raw_delta_mask[
                    batch_indices, counterfactual_indices].bool()
                cf_target = torch.where(
                    explicit_counterfactual[:, None],
                    selected_delta.abs().gt(1e-6).float(),
                    cf_target,
                )
                cf_mask = cf_mask | (
                    explicit_counterfactual[:, None]
                    & selected_delta_mask
                    & claim_mask.bool()
                )
                has_delta_annotation = selected_delta_mask.any(dim=1)
                annotated_global_change = (
                    selected_delta.abs().gt(1e-6) & selected_delta_mask
                ).any(dim=1).float()
                cf_global_target = torch.where(
                    explicit_counterfactual & has_delta_annotation,
                    annotated_global_change,
                    cf_global_target,
                )
            targets["cf"] = cf_target
            masks["cf"] = cf_mask
            # Explicit alternative actions remain useful even without dense
            # per-claim deltas: they supervise a shared transition head at the
            # whole-ledger level instead of pretending every claim changed.
            targets["cf_global"] = cf_global_target
            masks["cf_global"] = cf_global_mask
        return targets, masks

    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict):
        """Prepare input dict following infer code pattern from wan_va_server.py."""
        # Generate grid_id following infer code (no batch dimension yet)
        # For action mode: get_mesh_id(shape[-3], shape[-2], shape[-1], t=1, f_w=1, f_shift, action=True)
        latent_dict = self._add_noise(
            latent=batch_dict['latents'], 
            train_scheduler=self.train_scheduler_latent, 
            action_mask=None, 
            action_mode=False,
            noisy_cond_prob=0.5)
        
        action_dict = self._add_noise(
            latent=batch_dict['actions'], 
            train_scheduler=self.train_scheduler_action, 
            action_mask=batch_dict['actions_mask'], 
            action_mode=True,
            noisy_cond_prob=0.0)

        latent_dict['text_emb'] = batch_dict['text_emb']
        action_dict['text_emb'] = batch_dict['text_emb']
        action_dict['actions_mask'] = batch_dict['actions_mask']

        input_dict = {
            'latent_dict': latent_dict,
            'action_dict': action_dict,
            'chunk_size': torch.randint(1, 5, (1,)).item(),
            'window_size': torch.randint(4, 65, (1,)).item(),
        }
        if self.ledger_enabled:
            history_horizon = max(1, int(getattr(
                self.config, "ledger_history_horizon", 1)))
            has_history_latents = "history_latents" in batch_dict
            has_history_actions = "history_actions" in batch_dict
            if has_history_latents != has_history_actions:
                raise ValueError(
                    "history_latents and history_actions must be supplied together"
                )
            if (
                has_history_latents
                and batch_dict["history_latents"].shape[2]
                != batch_dict["history_actions"].shape[2]
            ):
                raise ValueError(
                    "history_latents and history_actions must have equal horizons"
                )
            # Only tensors explicitly named as history may expose multiple
            # frames.  The legacy ``latents/actions`` tensors are the target
            # future segment, so taking more than their first frame would leak
            # future information around the backbone's causal attention mask.
            if has_history_latents:
                observed_latents = batch_dict["history_latents"][
                    :, :, -history_horizon:]
            else:
                observed_latents = batch_dict["latents"][:, :, :1]
            if has_history_actions:
                observed_actions = batch_dict["history_actions"][
                    :, :, -history_horizon:]
            else:
                observed_actions = batch_dict["actions"][:, :, :1]
            (
                counterfactual_actions,
                explicit_counterfactual,
                synthetic_counterfactual,
                counterfactual_indices,
            ) = self._prepare_counterfactual_actions(
                batch_dict, observed_actions)

            available = batch_dict.get("ledger_available")

            input_dict["ledger_dict"] = {
                "latents": observed_latents,
                "actions": observed_actions,
                "text_emb": batch_dict["text_emb"],
                "counterfactual_actions": counterfactual_actions,
                # Claim presence is a prediction target, not a model input.
                # Feeding ledger_claim_mask here would leak annotations into
                # global risk, action conditioning, and recurrent matching.
                "masks": {},
                "context_mask": (
                    torch.ones(observed_latents.shape[0], device=self.device)
                    if available is None else available.float()
                ),
            }
            if (
                not has_history_latents
                and batch_dict["latents"].shape[2] > 1
                and batch_dict["actions"].shape[2] > 1
            ):
                # Default LeRobot segments do not expose a named past-history
                # tensor.  Their second frame is therefore reserved for a
                # self-distilled recurrent-update objective only.  It is not
                # used by ledger_context or the current action prediction.
                input_dict["ledger_dict"]["recurrent_next"] = {
                    "latents": batch_dict["latents"][:, :, 1:2],
                    "actions": batch_dict["actions"][:, :, 1:2],
                    "text_emb": batch_dict["text_emb"],
                    "counterfactual_actions": None,
                    "masks": {},
                }
            targets, masks = self._prepare_ledger_targets(
                batch_dict,
                explicit_counterfactual,
                synthetic_counterfactual,
                counterfactual_indices,
            )
            input_dict["ledger_targets"] = targets
            input_dict["ledger_loss_masks"] = masks
        return input_dict

    def convert_input_format(self, input_dict):
        """Move a nested batch to the training device.

        Ledger annotations contain nested dictionaries and may also carry
        string metadata.  The original implementation assumed every top-level
        value was a tensor, which made those batches impossible to collate.
        """

        def move(value):
            if torch.is_tensor(value):
                return value.to(self.device)
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            if isinstance(value, list):
                return [move(item) for item in value]
            if isinstance(value, tuple):
                return tuple(move(item) for item in value)
            return value

        return move(input_dict)

    def compute_loss(self,
        input_dict,
        pred
    ):
        latent_pred, action_pred = pred[:2]
        action_pred = rearrange(action_pred, 'b (f n) c -> b c f n 1', f=input_dict['action_dict']['targets'].shape[-3])
        latent_pred = data_seq_to_patch(
                        self.patch_size, latent_pred,
                        input_dict['latent_dict']['targets'].shape[-3], input_dict['latent_dict']['targets'].shape[-2],
                        input_dict['latent_dict']['targets'].shape[-1], batch_size=latent_pred.shape[0])
        Bn, Fn = input_dict['latent_dict']['timesteps'].shape
        latent_loss_weight = self.train_scheduler_latent.training_weight(input_dict['latent_dict']['timesteps'].flatten()).reshape(Bn, Fn)
        action_loss_weight = self.train_scheduler_action.training_weight(input_dict['action_dict']['timesteps'].flatten()).reshape(Bn, Fn)

        # Frame-wise video loss calculation
        latent_loss = F.mse_loss(latent_pred.float(), input_dict['latent_dict']['targets'].float().detach(), reduction='none')
        latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        latent_loss = latent_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and compute mask per frame
        latent_loss_per_frame = latent_loss.sum(dim=1)  # (B*F,)
        latent_mask_per_frame = torch.ones_like(latent_loss).sum(dim=1)  # (B*F,)
        latent_loss = (latent_loss_per_frame / (latent_mask_per_frame + 1e-6)).mean()

        # Frame-wise action loss calculation
        action_loss = F.mse_loss(action_pred.float(), input_dict['action_dict']['targets'].float().detach(), reduction='none')
        action_loss = action_loss * action_loss_weight[:, None, :, None, None]
        action_loss = action_loss * input_dict['action_dict']['actions_mask'].float()
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        action_loss = action_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_mask = input_dict['action_dict']['actions_mask'].float().permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_loss = action_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        action_mask = action_mask.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and normalize by mask per frame
        action_loss_per_frame = action_loss.sum(dim=1)  # (B*F,)
        action_mask_per_frame = action_mask.sum(dim=1)  # (B*F,)
        action_loss = (action_loss_per_frame / (action_mask_per_frame + 1e-6)).mean()

        return latent_loss / self.gradient_accumulation_steps, action_loss / self.gradient_accumulation_steps

    def _ledger_loss_weights(self):
        configured = dict(getattr(self.config, "ledger_loss_weights", {}))
        aliases = {
            "cf": "counterfactual",
            "repair": "repair_action",
            "repair_world": "post_repair_debt",
        }
        names = (
            "claim",
            "presence",
            "claim_type",
            "subject",
            "object",
            "precondition",
            "effect",
            "evidence",
            "uncertainty",
            "dependency",
            "dependency_matrix",
            "debt",
            "repair_cost",
            "observability",
            "importance",
            "relation",
            "rollback",
            "cf",
            "repair",
            "action_cost",
            "repair_world",
            "debt_reward",
            "recurrent",
        )
        return {
            name: float(configured.get(
                name, configured.get(aliases.get(name, ""), 0.0)))
            for name in names
        }

    def _train_step(self, batch, batch_idx):
        """Train a single batch, returns losses for logging."""
        batch = self.convert_input_format(batch)
        input_dict = self._prepare_input_dict(batch)
        
        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        
        if not should_sync:
            self.transformer.set_requires_gradient_sync(False)
        else:
            self.transformer.set_requires_gradient_sync(True)

        output = self.transformer(input_dict, train_mode=True)
        latent_loss, action_loss = self.compute_loss(input_dict, output)
        base_weights = dict(getattr(
            self.config, "ledger_loss_weights", {})) if self.ledger_enabled else {}
        loss = (
            float(base_weights.get("video", 1.0)) * latent_loss
            + float(base_weights.get("action", 1.0)) * action_loss
        )

        ledger_losses = None
        if self.ledger_enabled:
            if len(output) < 3:
                raise RuntimeError(
                    "Ledger-WAM training expected neural ledger outputs")
            ledger_losses = compute_ledger_losses(
                output[2],
                targets=input_dict.get("ledger_targets"),
                masks=input_dict.get("ledger_loss_masks"),
                weights=self._ledger_loss_weights(),
                cf_margin=float(getattr(
                    self.config, "ledger_counterfactual_margin", 1.0)),
                repair_action_cost_weight=float(getattr(
                    self.config, "ledger_repair_cost_weight", 0.0)),
                repair_task_risk_weight=float(getattr(
                    self.config, "ledger_repair_risk_weight", 0.0)),
            )
            loss = loss + (
                ledger_losses["total"] / self.gradient_accumulation_steps)

        loss.backward()

        losses = {'latent_loss': latent_loss.detach(), 'action_loss': action_loss.detach()}
        if ledger_losses is not None:
            for name, value in ledger_losses.items():
                losses["ledger_{}_loss".format(name)] = (
                    value.detach() / self.gradient_accumulation_steps)
            losses["ledger_global_risk"] = output[2][
                "global_risk"].detach().mean()
        
        # Only update weights after accumulating gradients
        if should_sync:
            total_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 2.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            losses['total_norm'] = total_norm
            losses['should_log'] = True
        else:
            losses['should_log'] = False

        return losses

    def save_checkpoint(self,):
        """Save model checkpoint in the same format as pretrained model."""
        try:
            state_dict = get_model_state_dict(
                self.transformer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
            optim_state = get_optimizer_state_dict(
                    self.transformer, self.optimizer,
                    options=StateDictOptions(full_state_dict=True, cpu_offload=True),
                )

            # Only rank 0 saves the checkpoint
            if self.config.rank == 0:
                checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                # Save transformer in the same format as pretrained model
                transformer_dir = checkpoint_dir / "transformer"
                transformer_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"Saving transformer to {transformer_dir}")

                # Manually save in diffusers format (outside FSDP context to avoid deadlock)
                # Save model weights
                model_file = transformer_dir / "diffusion_pytorch_model.safetensors"
                save_file(state_dict_bf16, model_file)

                # Save config (copy from original transformer config and update _name_or_path)
                config_file = transformer_dir / "config.json"
                config_dict = dict(self.transformer.config)
                config_dict.pop('_name_or_path', None)
                with open(config_file, 'w') as f:
                    json.dump(config_dict, f, indent=2)

                # Preserve optimizer state and Ledger schema/config metadata so
                # staged head warm-up can resume without changing ontologies.
                training_state_path = checkpoint_dir / "training_state.pt"
                logger.info(f"Saving training state to {training_state_path}")
                torch.save({
                    'step': self.step,
                    'optimizer_state_dict': optim_state,
                    'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                    'config': dict(self.config),
                }, training_state_path)

                logger.info(f"Checkpoint saved successfully at step {self.step}")

            # Synchronize all processes after saving
            if dist.is_initialized():
                dist.barrier()

        except Exception as e:
            if self.config.rank == 0:
                logger.error(f"Failed to save checkpoint: {e}")
                import traceback
                logger.error(traceback.format_exc())
            # Ensure all processes stay synchronized even on error
            if dist.is_initialized():
                dist.barrier()

    def _load_training_state(self, checkpoint_path):
        """Load training state (optimizer + step) after FSDP and optimizer creation."""
        checkpoint_dir = Path(checkpoint_path)
        training_state_path = checkpoint_dir / "training_state.pt"

        if not training_state_path.exists():
            if self.config.rank == 0:
                logger.warning(f"Training state not found: {training_state_path}, starting from step 0")
            return

        if self.config.rank == 0:
            logger.info(f"Loading training state from {training_state_path}")

        # All ranks load the training state directly
        training_state = torch.load(training_state_path, map_location='cpu', weights_only=False)

        if self.ledger_enabled:
            saved_config = training_state.get('config', {})
            for key in (
                'ledger_max_claims',
                'ledger_num_relations',
                'ledger_max_rollback_stages',
                'ledger_repair_catalog',
            ):
                current_value = getattr(self.config, key, None)
                if key in saved_config and saved_config[key] != current_value:
                    raise ValueError(
                        f"Ledger checkpoint config mismatch for {key}: "
                        f"{saved_config[key]!r} != {current_value!r}")

        # All ranks load optimizer state (required for FSDP)
        set_optimizer_state_dict(
            self.transformer, self.optimizer,
            optim_state_dict=training_state['optimizer_state_dict'],
            options=StateDictOptions(full_state_dict=True, strict=False)
        )
        if 'lr_scheduler_state_dict' in training_state:
            self.lr_scheduler.load_state_dict(
                training_state['lr_scheduler_state_dict'])
        self.step = training_state.get('step', 0)

        if self.config.rank == 0:
            logger.info(f"Training state loaded, resuming from step {self.step}")

        # Synchronize all ranks
        if dist.is_initialized():
            dist.barrier()

    def train(self):
        """Main training loop - train by steps instead of epochs."""
        logger.info(f"Starting training for {self.config.num_steps} steps...")
        self.transformer.train()

        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="Training",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step
        )

        self.optimizer.zero_grad()
        accumulated_latent_losses = []
        accumulated_action_losses = []
        accumulated_ledger_metrics = {}
        step_in_accumulation = 0

        while self.step < self.config.num_steps:
            # Get next batch (handles epoch reset automatically)
            batch = self._get_next_batch()
            
            losses = self._train_step(batch, step_in_accumulation)
            
            # Accumulate losses for logging
            accumulated_latent_losses.append(losses['latent_loss'])
            accumulated_action_losses.append(losses['action_loss'])
            for name, value in losses.items():
                if name.startswith("ledger_"):
                    accumulated_ledger_metrics.setdefault(name, []).append(value)
            step_in_accumulation += 1

            # Log and checkpoint when optimizer steps
            if losses['should_log']:
                lr = self.lr_scheduler.get_last_lr()[0]

                # Average accumulated losses
                latent_loss_show = dist_mean(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                action_loss_show = dist_mean(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                max_latent_loss_show = dist_max(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                max_action_loss_show = dist_max(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                ledger_metrics_show = {
                    name: dist_mean(torch.stack(values).sum()).detach().cpu().item()
                    for name, values in accumulated_ledger_metrics.items()
                }

                # Clear accumulated losses
                accumulated_latent_losses = []
                accumulated_action_losses = []
                accumulated_ledger_metrics = {}
                step_in_accumulation = 0

                torch.cuda.synchronize()
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.config.rank == 0:
                    total_norm = losses['total_norm']
                    progress_bar.n += 1
                    postfix = {
                        'latent_loss': f'{latent_loss_show:.4f}',
                        'action_loss': f'{action_loss_show:.4f}',
                        'step': self.step,
                        'grad_norm': f'{total_norm.item():.2f}',
                        'lr': f'{lr:.2e}'
                    }
                    if "ledger_total_loss" in ledger_metrics_show:
                        postfix["ledger_loss"] = (
                            f"{ledger_metrics_show['ledger_total_loss']:.4f}")
                    if "ledger_global_risk" in ledger_metrics_show:
                        postfix["ledger_risk"] = (
                            f"{ledger_metrics_show['ledger_global_risk']:.3f}")
                    progress_bar.set_postfix(postfix)
                    if self.config.enable_wandb:
                        wandb_metrics = {
                            'loss_metrics/global_avg_video_loss': latent_loss_show,
                            'loss_metrics/global_avg_action_loss': action_loss_show,
                            'loss_metrics/global_max_video_loss': max_latent_loss_show,
                            'loss_metrics/global_max_action_loss': max_action_loss_show,
                            'grad_norm': total_norm.item(),
                            'lr': lr,
                        }
                        wandb_metrics.update({
                            "ledger/{}".format(name.removeprefix("ledger_")):
                            value
                            for name, value in ledger_metrics_show.items()
                        })
                        self.wandb.log(wandb_metrics, step=self.step)
                
                self.step += 1
                
                if self.step % self.config.save_interval == 0:
                    if self.config.rank == 0:
                        logger.info(f"Starting save model at step {self.step}")
                    self.save_checkpoint()

            if dist.is_initialized():
                dist.barrier()

        progress_bar.close()
        logger.info("Training completed!")


def run(args):
    """Main entry point."""
    config = VA_CONFIGS[args.config_name]

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_distributed(world_size, local_rank, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    if args.save_root is not None:
        config.save_root = args.save_root
    if args.ledger_annotation_path is not None:
        config.ledger_annotation_path = args.ledger_annotation_path
    if args.ledger_strict:
        config.ledger_strict = True

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")

    trainer = Trainer(config)
    trainer.train()


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train WAN model for robotics")
    parser.add_argument(
        "--config-name",
        type=str,
        default='robotwin_train',
        help="Config name",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="Root directory for saving checkpoints",
    )
    parser.add_argument(
        "--ledger-annotation-path",
        type=str,
        default=None,
        help="Ledger-WAM JSON/JSONL sidecar (overrides the config)",
    )
    parser.add_argument(
        "--ledger-strict",
        action="store_true",
        help="Fail if a Ledger-WAM sidecar or required field is invalid",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
