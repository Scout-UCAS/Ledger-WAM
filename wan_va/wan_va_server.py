# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import json
import os
import sys
import time
from PIL import Image
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model
from distributed.util import _configure_model, init_distributed
from modules.utils import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
)
from utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    logger,
    run_async_server_mode,
    save_async,
)

try:
    from wan_va.ledger import (
        CausalBeliefLedger,
        CausalClaim,
        ClaimStatus,
        DebtWeights,
        Evidence,
        LedgerRuntimeState,
        LogicalPlanState,
        PlannerDecisionType,
        PlanningCheckpoint,
        RepairCandidate,
        RepairExecutionTracker,
        SelfHealingPlanner,
        SelfHealingPlannerConfig,
        validate_repair_action_chunk,
        validate_repair_catalog,
    )
except ImportError:  # Script-style execution with ``wan_va`` on sys.path.
    from ledger import (
        CausalBeliefLedger,
        CausalClaim,
        ClaimStatus,
        DebtWeights,
        Evidence,
        LedgerRuntimeState,
        LogicalPlanState,
        PlannerDecisionType,
        PlanningCheckpoint,
        RepairCandidate,
        RepairExecutionTracker,
        SelfHealingPlanner,
        SelfHealingPlannerConfig,
        validate_repair_action_chunk,
        validate_repair_catalog,
    )


class VA_Server:

    def __init__(self, job_config):
        self.cache_name = 'pos'
        self.job_config = job_config
        self.save_root = job_config.save_root
        self.dtype = job_config.param_dtype
        self.device = torch.device(f"cuda:{job_config.local_rank}")
        self.enable_offload = getattr(job_config, 'enable_offload', True)  # offload vae & text_encoder to save vram
        self.ledger_enabled = bool(getattr(job_config, "ledger_enabled", False))
        if self.ledger_enabled:
            validate_repair_catalog(tuple(getattr(
                job_config, "ledger_repair_catalog", ())))
        if self.ledger_enabled and len(
            job_config.ledger_rollback_stage_ontology) != int(
                job_config.ledger_max_rollback_stages):
            raise ValueError(
                "ledger_rollback_stage_ontology length must equal "
                "ledger_max_rollback_stages")

        self.scheduler = FlowMatchScheduler(shift=self.job_config.snr_shift,
                                            sigma_min=0.0,
                                            extra_one_step=True)
        self.action_scheduler = FlowMatchScheduler(
            shift=self.job_config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)

        self.vae = load_vae(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'vae'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)

        self.tokenizer = load_tokenizer(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'tokenizer'), )

        self.text_encoder = load_text_encoder(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'text_encoder'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )

        transformer_overrides = {"attn_mode": "torch"}
        if self.ledger_enabled:
            repair_catalog = getattr(job_config, "ledger_repair_catalog", ())
            transformer_overrides.update({
                "enable_ledger": True,
                "low_cpu_mem_usage": False,
                "ledger_hidden_dim": int(job_config.ledger_hidden_dim),
                "ledger_num_claim_slots": int(job_config.ledger_max_claims),
                "ledger_num_relations": int(job_config.ledger_num_relations),
                "ledger_max_rollback_steps": int(
                    job_config.ledger_max_rollback_stages),
                "ledger_num_repair_actions": len(repair_catalog),
                "ledger_delta_dim": int(job_config.ledger_delta_dim),
                "ledger_num_heads": int(job_config.ledger_num_heads),
                "ledger_dropout": float(job_config.ledger_dropout),
                "ledger_num_claim_types": int(
                    job_config.ledger_num_claim_types),
                "ledger_num_subjects": int(job_config.ledger_num_subjects),
                "ledger_num_objects": int(job_config.ledger_num_objects),
                "ledger_num_preconditions": int(
                    job_config.ledger_num_preconditions),
                "ledger_num_effects": int(job_config.ledger_num_effects),
            })
        transformer_path = os.path.join(
            job_config.wan22_pretrained_model_name_or_path, 'transformer')
        if self.ledger_enabled and not bool(getattr(
            job_config, "ledger_allow_random_head", False)):
            transformer_config_path = os.path.join(
                transformer_path, "config.json")
            try:
                with open(transformer_config_path, "r") as handle:
                    saved_transformer_config = json.load(handle)
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    "Ledger-WAM server requires a trained transformer config; "
                    "set ledger_allow_random_head=True only for debugging"
                ) from exc
            if not saved_transformer_config.get("enable_ledger", False):
                raise RuntimeError(
                    "The selected checkpoint has no trained Ledger-WAM head. "
                    "Run ledger_*_train first, or explicitly set "
                    "ledger_allow_random_head=True for debugging."
                )
        self.transformer = load_transformer(
            transformer_path,
            torch_dtype=self.dtype,
            torch_device=self.device,
            **transformer_overrides,
        )
        shard_fn = shard_model
        self.transformer = _configure_model(model=self.transformer,
                                            shard_fn=shard_fn,
                                            param_dtype=self.dtype,
                                            device=self.device,
                                            eval_mode=True,
                                            )

        self.env_type = job_config.env_type
        self.streaming_vae_half = None
        if self.env_type == 'robotwin_tshape':
            vae_half = load_vae(
                os.path.join(job_config.wan22_pretrained_model_name_or_path,
                             'vae'),
                torch_dtype=self.dtype,
                torch_device='cpu' if self.enable_offload else self.device,
            )
            self.streaming_vae_half = WanVAEStreamingWrapper(vae_half)

        self.ledger_runtime = None
        self.ledger_planner = None
        self.ledger_context = None
        self.ledger_prediction = None
        self.ledger_repair_candidates = ()
        self.runtime_debt_weights = None
        self.last_executed_action = None
        self.pending_repair_claim_ids = ()
        self.repair_execution_tracker = RepairExecutionTracker()
        self.previous_claim_slots = None
        self.previous_claim_mask = None
        self.current_prompt = None
        if self.ledger_enabled:
            debt_config = dict(getattr(job_config, "ledger_debt_weights", {}))
            debt_weights = DebtWeights(
                confidence_gap=float(debt_config.get(
                    "lack_of_confidence", 1.0)),
                uncertainty=float(debt_config.get("uncertainty", 1.0)),
                dependency=float(debt_config.get("dependency", 1.0)),
                repair_cost=float(debt_config.get("repair_cost", 1.0)),
                unobservability=float(debt_config.get(
                    "lack_of_observability", 1.0)),
                bias=float(debt_config.get("bias", -2.0)),
            )
            planner_config = SelfHealingPlannerConfig(
                global_risk_threshold=float(
                    job_config.ledger_global_risk_threshold),
                claim_debt_threshold=float(job_config.ledger_debt_threshold),
                min_normalized_importance=float(
                    job_config.ledger_importance_threshold),
                cost_weight=float(job_config.ledger_repair_cost_weight),
                risk_weight=float(job_config.ledger_repair_risk_weight),
                policy_weight=float(getattr(
                    job_config, "ledger_repair_policy_prior_weight", 0.0)),
                minimum_repair_score=float(getattr(
                    job_config, "ledger_minimum_repair_score", 0.0)),
                debt_weights=debt_weights,
                # Neural per-claim debt is directly calibrated by L_debt and
                # shares a scale with predicted post-repair risk.
                recompute_debt=False,
            )
            self.ledger_planner = SelfHealingPlanner(planner_config)
            self.runtime_debt_weights = debt_weights
            self.ledger_runtime = LedgerRuntimeState(
                ledger=CausalBeliefLedger(),
                plan_state=LogicalPlanState(),
            )

    def _get_t5_prompt_embeds(
        self,
        prompt=None,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        text_encoder_device = next(self.text_encoder.parameters()).device
        prompt_embeds = self.text_encoder(text_input_ids.to(text_encoder_device),
                                          mask.to(text_encoder_device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack([
            torch.cat(
                [u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
                                    dim=0)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt,
                                           seq_len, -1)

        return prompt_embeds.to(device)

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=226,
        device=None,
        dtype=None,
    ):
        r"""
        TODO
        """
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(
                negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(
                    negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}.")
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`.")

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        return prompt_embeds, negative_prompt_embeds

    def normalize_latents(
        self,
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1,
                                         1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1,
                                       1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    def preprocess_action(self, action):
        action_model_input = torch.from_numpy(action)
        CA, FA, HA = action_model_input.shape  # C, F, H
        action_model_input_paded = F.pad(action_model_input,
                                         [0, 0, 0, 0, 0, 1],
                                         mode='constant',
                                         value=0)

        action_model_input = action_model_input_paded[
            self.job_config.inverse_used_action_channel_ids]

        if self.action_norm_method == 'quantiles':
            action_model_input = (action_model_input - self.actions_q01) / (
                self.actions_q99 - self.actions_q01 + 1e-6) * 2. - 1.
        else:
            raise ValueError(
                "Unsupported action_norm_method {!r}; expected 'quantiles'".format(
                    self.action_norm_method
                )
            )
        return action_model_input.unsqueeze(0).unsqueeze(-1)  # B, C, F, H, W

    def postprocess_action(self, action):
        action = action.cpu()  # B, C, F, H, W

        action = action[0, ..., 0]  #C, F, H
        if self.action_norm_method == 'quantiles':
            action = (action + 1) / 2 * (self.actions_q99 - self.actions_q01 +
                                         1e-6) + self.actions_q01
        else:
            raise ValueError(
                "Unsupported action_norm_method {!r}; expected 'quantiles'".format(
                    self.action_norm_method
                )
            )
        action = action.squeeze(0).detach().cpu().numpy()
        return action[self.job_config.used_action_channel_ids]

    def _reset_ledger_runtime(self, serialized_state=None, checkpoints=()):
        if not self.ledger_enabled:
            return
        if serialized_state is None:
            runtime = LedgerRuntimeState(
                ledger=CausalBeliefLedger(),
                plan_state=LogicalPlanState(),
            )
        elif isinstance(serialized_state, str):
            runtime = LedgerRuntimeState.from_json(serialized_state)
        elif isinstance(serialized_state, dict):
            runtime = LedgerRuntimeState.from_dict(serialized_state)
        else:
            raise TypeError("ledger_state must be a JSON string or dictionary")

        for checkpoint_data in checkpoints or ():
            checkpoint = PlanningCheckpoint.from_dict(checkpoint_data)
            if checkpoint.checkpoint_id not in runtime.plan_state.checkpoints:
                runtime.plan_state.add_checkpoint(checkpoint)
        if "task_start" not in runtime.plan_state.checkpoints:
            runtime.plan_state.add_checkpoint(
                PlanningCheckpoint(
                    checkpoint_id="task_start",
                    cursor=0,
                    subgoal="task_start",
                )
            )
        self.ledger_runtime = runtime
        self.ledger_context = None
        self.ledger_prediction = None
        self.ledger_repair_candidates = ()
        self.pending_repair_claim_ids = ()
        self.repair_execution_tracker.reset()
        self.previous_claim_slots = None
        self.previous_claim_mask = None

    def _apply_external_ledger_update(self, request):
        """Apply simulator/event-parser evidence supplied by the controller."""
        if not self.ledger_enabled:
            return
        ledger = self.ledger_runtime.ledger
        plan_state = self.ledger_runtime.plan_state
        for claim_data in request.get("ledger_claims", ()) or ():
            claim = CausalClaim.from_dict(claim_data)
            ledger.add_claim(
                claim, replace=claim.claim_id in ledger.claims)
        for prerequisite_id, dependent_id in (
            request.get("ledger_dependencies", ()) or ()):
            ledger.add_dependency(str(prerequisite_id), str(dependent_id))
        for evidence_update in request.get("ledger_evidence", ()) or ():
            update = dict(evidence_update)
            claim_id = update.pop("claim_id")
            evidence_data = update.pop("evidence", update)
            ledger.record_evidence(
                str(claim_id), Evidence.from_dict(evidence_data))
            claim = ledger.get_claim(str(claim_id))
            claim.debt = self._calibrated_debt_value(
                confidence=claim.confidence,
                uncertainty=claim.uncertainty,
                dependency=claim.dependency,
                repair_cost=claim.repair_cost,
                observability=claim.observability,
            )
        for checkpoint_data in request.get("planning_checkpoints", ()) or ():
            checkpoint = PlanningCheckpoint.from_dict(checkpoint_data)
            if checkpoint.checkpoint_id not in plan_state.checkpoints:
                plan_state.add_checkpoint(checkpoint)
        if request.get("plan_cursor") is not None:
            cursor = int(request["plan_cursor"])
            if cursor < 0:
                raise ValueError("plan_cursor must be non-negative")
            plan_state.cursor = cursor
        if request.get("active_subgoal") is not None:
            plan_state.active_subgoal = str(request["active_subgoal"])
        if self.ledger_prediction is not None:
            self._refresh_neural_repair_candidates()

    def _calibrated_debt_value(
        self,
        *,
        confidence,
        uncertainty,
        dependency,
        repair_cost,
        observability,
    ):
        """Re-evaluate debt after external evidence changes confidence."""
        prediction = self.ledger_prediction
        if prediction is not None and "debt_weights" in prediction:
            weights = prediction["debt_weights"].detach().float().cpu()
            bias = prediction["debt_bias"].detach().float().cpu()
            features = torch.tensor(
                [
                    1.0 - float(confidence),
                    float(uncertainty),
                    float(dependency),
                    float(repair_cost),
                    1.0 - float(observability),
                ],
                dtype=weights.dtype,
            )
            return float(torch.sigmoid((features * weights).sum() + bias))
        if self.runtime_debt_weights is None:
            return 0.0
        weights = self.runtime_debt_weights
        logit = (
            weights.bias
            + weights.confidence_gap * (1.0 - float(confidence))
            + weights.uncertainty * float(uncertainty)
            + weights.dependency * float(dependency)
            + weights.repair_cost * float(repair_cost)
            + weights.unobservability * (1.0 - float(observability))
        )
        return float(torch.sigmoid(torch.tensor(logit)))

    def _apply_repair_execution_ack(self, request, allow_implicit=False):
        """Apply execution evidence before validating claims on new observations."""
        acknowledgement = self.repair_execution_tracker.acknowledge(
            request.get("repair_execution_ack"),
            implicit_success=(
                allow_implicit
                and request.get("repair_execution_ack") is None
                and request.get("state") is not None
            ),
        )
        if acknowledgement is not None and acknowledgement.success:
            self.pending_repair_claim_ids = tuple(sorted(
                set(self.pending_repair_claim_ids)
                | set(acknowledgement.target_claim_ids)
            ))
        return acknowledgement

    def _checkpoint_for_rollback_index(self, rollback_index):
        ontology = tuple(getattr(
            self.job_config, "ledger_rollback_stage_ontology", ()))
        index = int(rollback_index)
        if index < 0 or index >= len(ontology):
            return "task_start"
        stage_id = str(ontology[index])
        plan_state = self.ledger_runtime.plan_state
        if stage_id in plan_state.checkpoints:
            return stage_id
        candidates = [
            checkpoint
            for checkpoint in plan_state.checkpoints.values()
            if checkpoint.subgoal == stage_id
            or checkpoint.metadata.get("stage_id") == stage_id
        ]
        if candidates:
            candidates.sort(key=lambda checkpoint: checkpoint.cursor)
            return candidates[-1].checkpoint_id
        if stage_id == "current_subgoal" and plan_state.checkpoints:
            return max(
                plan_state.checkpoints.values(),
                key=lambda checkpoint: checkpoint.cursor,
            ).checkpoint_id
        # Fixed-class semantics remain stable even when an environment adapter
        # has not supplied the requested stage checkpoint.
        return "task_start"

    @torch.no_grad()
    def _update_neural_ledger(self, observed_latents, observed_actions=None):
        """Update runtime slots and repair predictions from a real observation."""
        if not self.ledger_enabled or observed_latents is None:
            return ()
        if self.prompt_embeds is None:
            raise RuntimeError("Ledger-WAM requires a language prompt")
        history_horizon = max(1, int(getattr(
            self.job_config, "ledger_history_horizon", 1)))
        observed_latents = observed_latents[:, :, -history_horizon:]
        if observed_actions is None:
            observed_actions = torch.zeros(
                1,
                self.job_config.action_dim,
                observed_latents.shape[2],
                self.job_config.action_per_frame,
                1,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            observed_actions = observed_actions[:, :, -history_horizon:]
        prediction = self.transformer(
            {
                "latents": observed_latents.to(self.dtype),
                "actions": observed_actions.to(self.dtype),
                "text_emb": self.prompt_embeds.to(self.dtype),
                "previous_claim_slots": self.previous_claim_slots,
                "previous_claim_mask": self.previous_claim_mask,
            },
            ledger_mode=True,
        )
        self.ledger_prediction = prediction
        self.ledger_context = prediction["ledger_context"].detach()
        self.previous_claim_slots = prediction["claim_slots"].detach()
        self.previous_claim_mask = prediction["active_claim_mask"].detach()

        relation_names = tuple(getattr(
            self.job_config,
            "ledger_relation_ontology",
            tuple("relation_{}".format(index) for index in range(
                int(self.job_config.ledger_num_relations))),
        ))
        confidence = prediction["confidence"][0].float().cpu()
        uncertainty = prediction["uncertainty"][0].float().cpu()
        dependency = prediction["dependency"][0].float().cpu()
        repair_cost = prediction["repair_cost"][0].float().cpu()
        observability = prediction["observability"][0].float().cpu()
        importance = prediction["importance"][0].float().cpu()
        presence = prediction.get("presence")
        if presence is None:
            presence = torch.ones_like(prediction["importance"])
        presence = presence[0].float().cpu()
        effective_importance = importance * presence
        neural_debt = prediction["debt"][0].float().cpu()
        relation_ids = prediction["relation_logits"][0].argmax(dim=-1).cpu()
        rollback_ids = prediction["rollback_logits"][0].argmax(dim=-1).cpu()
        precondition_ids = prediction.get("precondition_logits")
        if precondition_ids is not None:
            precondition_ids = precondition_ids[0].argmax(dim=-1).cpu()
        effect_ids = prediction.get("effect_logits")
        if effect_ids is not None:
            effect_ids = effect_ids[0].argmax(dim=-1).cpu()
        claim_type_ids = prediction.get("claim_type_logits")
        if claim_type_ids is not None:
            claim_type_ids = claim_type_ids[0].argmax(dim=-1).cpu()
        evidence_scores = prediction.get("evidence")
        if evidence_scores is not None:
            evidence_scores = evidence_scores[0].float().cpu()
        claim_mask = prediction.get(
            "active_claim_mask", prediction.get("claim_mask"))
        if claim_mask is None:
            claim_mask = torch.ones_like(
                prediction["confidence"], dtype=torch.bool)
        claim_mask = claim_mask[0].bool().cpu()

        ledger = self.ledger_runtime.ledger
        verification_threshold = float(getattr(
            self.job_config, "ledger_verification_threshold", 0.8))
        conflict_threshold = float(getattr(
            self.job_config, "ledger_confidence_threshold", 0.3))
        debt_threshold = float(self.job_config.ledger_debt_threshold)
        pending_repair = set(self.pending_repair_claim_ids)
        resolved_pending = set()
        for slot_index in range(confidence.shape[0]):
            claim_id = "neural_slot_{}".format(slot_index)
            if not bool(claim_mask[slot_index]):
                stale_claim = ledger.claims.get(claim_id)
                externally_grounded = (
                    stale_claim is not None
                    and (bool(stale_claim.evidence) or stale_claim.metadata.get(
                        "source") != "neural_ledger")
                )
                if stale_claim is not None and not externally_grounded:
                    stale_claim.status = ClaimStatus.INVALIDATED
                    stale_claim.importance = 0.0
                # A targeted claim disappearing after an acknowledged repair
                # is a completed observation outcome, not a permanent pending
                # validation that may poison this slot when it is later reused.
                if claim_id in pending_repair:
                    resolved_pending.add(claim_id)
                continue
            previous = ledger.claims.get(claim_id)
            neural_confidence = float(confidence[slot_index])
            fused_confidence = neural_confidence
            external_authoritative = False
            if previous is not None:
                external_authoritative = bool(previous.evidence) or (
                    previous.metadata.get("source") != "neural_ledger")
                if external_authoritative:
                    external_weight = float(getattr(
                        self.job_config,
                        "ledger_external_evidence_weight",
                        0.7,
                    ))
                    fused_confidence = (
                        external_weight * previous.confidence
                        + (1.0 - external_weight) * neural_confidence
                    )
            calibrated_debt = self._calibrated_debt_value(
                confidence=fused_confidence,
                uncertainty=float(uncertainty[slot_index]),
                dependency=float(dependency[slot_index]),
                repair_cost=float(repair_cost[slot_index]),
                observability=float(observability[slot_index]),
            )
            if (
                claim_id in pending_repair
                and fused_confidence < conflict_threshold
                and calibrated_debt >= debt_threshold
            ):
                # Low confidence alone is not a contradiction.  It becomes a
                # refutation only after a targeted validation/repair action.
                status = ClaimStatus.REFUTED
            elif fused_confidence >= verification_threshold:
                status = ClaimStatus.VERIFIED
            elif previous is not None and previous.status in (
                ClaimStatus.REFUTED, ClaimStatus.INVALIDATED):
                status = previous.status
            else:
                status = ClaimStatus.HYPOTHESIZED
            relation_index = int(relation_ids[slot_index])
            relation = (
                relation_names[relation_index]
                if relation_index < len(relation_names)
                else "relation_{}".format(relation_index)
            )
            entities = ("slot_{}".format(slot_index),)
            if "subject_logits" in prediction and "object_logits" in prediction:
                subject = int(prediction["subject_logits"][
                    0, slot_index].argmax())
                object_id = int(prediction["object_logits"][
                    0, slot_index].argmax())
                entities = (
                    "entity_{}".format(subject),
                    "entity_{}".format(object_id),
                )
            rollback_checkpoint = self._checkpoint_for_rollback_index(
                int(rollback_ids[slot_index]))
            neural_preconditions = (
                () if precondition_ids is None else (
                    "precondition_{}".format(int(
                        precondition_ids[slot_index])),)
            )
            neural_effects = (
                () if effect_ids is None else (
                    "effect_{}".format(int(effect_ids[slot_index])),)
            )
            if external_authoritative:
                entities = previous.entities
                relation = previous.relation
                preconditions = previous.preconditions or neural_preconditions
                effects = previous.effects or neural_effects
                rollback_checkpoint = (
                    previous.rollback_checkpoint or rollback_checkpoint)
            else:
                preconditions = neural_preconditions
                effects = neural_effects
            metadata = (
                {} if previous is None else dict(previous.metadata))
            metadata.update({
                "source": (
                    metadata.get("source", "neural_ledger")
                    if external_authoritative else "neural_ledger"),
                "slot": slot_index,
                "neural_confidence": neural_confidence,
                "presence": float(presence[slot_index]),
                "neural_debt": float(neural_debt[slot_index]),
                "claim_type": (
                    None if claim_type_ids is None
                    else int(claim_type_ids[slot_index])),
                "evidence_score": (
                    None if evidence_scores is None
                    else float(evidence_scores[slot_index])),
            })
            claim = CausalClaim(
                claim_id=claim_id,
                entities=entities,
                relation=relation,
                preconditions=preconditions,
                effects=effects,
                evidence=([] if previous is None else list(previous.evidence)),
                confidence=fused_confidence,
                uncertainty=float(uncertainty[slot_index]),
                dependency=float(dependency[slot_index]),
                repair_cost=float(repair_cost[slot_index]),
                observability=float(observability[slot_index]),
                importance=float(effective_importance[slot_index]),
                debt=calibrated_debt,
                rollback_checkpoint=rollback_checkpoint,
                status=status,
                created_at=(
                    int(self.frame_st_id)
                    if previous is None else previous.created_at),
                updated_at=int(self.frame_st_id),
                metadata=metadata,
            )
            ledger.add_claim(claim, replace=previous is not None)
            if claim_id in pending_repair:
                resolved_pending.add(claim_id)
            if status is ClaimStatus.REFUTED:
                ledger.invalidate_descendants(claim_id)
        self.pending_repair_claim_ids = tuple(sorted(
            pending_repair - resolved_pending))

        dependency_matrix = prediction.get("dependency_matrix")
        if dependency_matrix is not None:
            edge_threshold = float(getattr(
                self.job_config, "ledger_dependency_edge_threshold", 0.5))
            matrix = dependency_matrix[0].float().cpu()
            neural_claim_ids = {
                claim_id for claim_id in ledger.claims
                if claim_id.startswith("neural_slot_")}
            ledger.clear_dependencies(neural_claim_ids)
            directed_edges = []
            for source in range(matrix.shape[0]):
                for target in range(matrix.shape[1]):
                    if source == target:
                        continue
                    probability = float(matrix[source, target])
                    if probability >= edge_threshold:
                        directed_edges.append((probability, source, target))
            for _probability, source, target in sorted(
                directed_edges, reverse=True):
                source_id = "neural_slot_{}".format(source)
                target_id = "neural_slot_{}".format(target)
                if source_id not in ledger.claims or target_id not in ledger.claims:
                    continue
                try:
                    ledger.add_dependency(source_id, target_id)
                except ValueError:
                    # Skip the lower-scored edge when pairwise predictions
                    # would otherwise create a dependency cycle.
                    continue

        return self._refresh_neural_repair_candidates()

    def _refresh_neural_repair_candidates(self):
        """Recompose repair utility from cached transitions and live beliefs."""
        prediction = self.ledger_prediction
        if prediction is None:
            self.ledger_repair_candidates = ()
            return self.ledger_repair_candidates
        ledger = self.ledger_runtime.ledger
        neural_debt = prediction["debt"][0].float().cpu()
        repair_catalog = tuple(getattr(
            self.job_config, "ledger_repair_catalog", ()))
        repair_reduction = prediction["repair_reduction"][0].float().cpu()
        runtime_debt = torch.tensor(
            [
                ledger.claims.get("neural_slot_{}".format(slot_index)).debt
                if "neural_slot_{}".format(slot_index) in ledger.claims
                else float(neural_debt[slot_index])
                for slot_index in range(neural_debt.shape[0])
            ],
            dtype=repair_reduction.dtype,
        )
        runtime_importance = torch.tensor(
            [
                ledger.claims["neural_slot_{}".format(slot_index)].importance
                if "neural_slot_{}".format(slot_index) in ledger.claims
                else 0.0
                for slot_index in range(neural_debt.shape[0])
            ],
            dtype=repair_reduction.dtype,
        )
        runtime_active = torch.tensor(
            [
                (
                    "neural_slot_{}".format(slot_index) in ledger.claims
                    and ledger.claims[
                        "neural_slot_{}".format(slot_index)
                    ].status is not ClaimStatus.INVALIDATED
                )
                for slot_index in range(neural_debt.shape[0])
            ],
            dtype=torch.bool,
        )
        current_debt = runtime_debt.unsqueeze(0)
        post_debt_per_claim = torch.where(
            repair_reduction >= 0,
            current_debt * (1.0 - repair_reduction),
            current_debt + (-repair_reduction) * (1.0 - current_debt),
        )
        # Match CausalBeliefLedger.global_risk exactly: invalidated claims do
        # not contribute to either current or predicted post-repair risk.
        aggregation_weight = runtime_importance * runtime_active.float()
        external_active_claims = [
            claim
            for claim_id, claim in ledger.claims.items()
            if not claim_id.startswith("neural_slot_")
            and claim.status is not ClaimStatus.INVALIDATED
        ]
        external_weight = sum(
            claim.importance for claim in external_active_claims)
        external_weighted_debt = sum(
            claim.importance * claim.debt for claim in external_active_claims)
        denominator = aggregation_weight.sum() + external_weight
        post_risk = ((
            post_debt_per_claim * aggregation_weight[None]
        ).sum(dim=-1) + external_weighted_debt) / denominator.clamp_min(1e-6)
        predicted_cost = prediction["repair_action_cost"][0].float().cpu()
        repair_log_probability = prediction[
            "repair_logits"][0].float().log_softmax(dim=-1).cpu()
        neural_repair_scores = prediction[
            "repair_scores"][0].float().cpu()
        candidates = []
        for repair_index, entry in enumerate(repair_catalog):
            if repair_index >= post_risk.shape[0]:
                break
            target_score = (
                repair_reduction[repair_index]
                * runtime_debt
                * runtime_importance
                * runtime_active.float()
            )
            target_claim_ids = ()
            if target_score.numel() and float(target_score.max()) > 0:
                target_claim_ids = (
                    "neural_slot_{}".format(int(target_score.argmax())),)
            candidates.append(RepairCandidate(
                action_id=str(entry["name"]),
                target_claim_ids=target_claim_ids,
                expected_global_risk=float(post_risk[repair_index].clamp(0, 1)),
                action_cost=float(predicted_cost[repair_index].clamp(0, 1)),
                task_risk=float(entry.get("risk", 0.0)),
                policy_log_probability=float(
                    repair_log_probability[repair_index]),
                metadata={
                    "repair_index": repair_index,
                    "predicted_action_cost": float(
                        predicted_cost[repair_index]),
                    "catalog_action_cost": float(entry.get("cost", 0.0)),
                    "neural_repair_score": float(
                        neural_repair_scores[repair_index]),
                },
            ))
        self.ledger_repair_candidates = tuple(candidates)
        return self.ledger_repair_candidates

    def _external_repair_candidates(self, request):
        return tuple(
            RepairCandidate.from_dict(candidate)
            for candidate in (request.get("repair_candidates", ()) or ())
        )

    def _planner_decision(self, request):
        explicit = self._external_repair_candidates(request)
        explicit_ids = {candidate.action_id for candidate in explicit}
        candidates = explicit + tuple(
            candidate for candidate in self.ledger_repair_candidates
            if candidate.action_id not in explicit_ids)
        return self.ledger_planner.decide(
            ledger=self.ledger_runtime.ledger,
            plan_state=self.ledger_runtime.plan_state,
            task_action_id=str(request.get("task_action_id", "task_chunk")),
            repair_candidates=candidates,
        )

    def _planner_response(self, decision):
        response = {
            "planner": decision.to_dict(),
            "ledger": self.ledger_runtime.ledger.to_dict(),
            "logical_plan": self.ledger_runtime.plan_state.to_dict(),
        }
        outstanding = self.repair_execution_tracker.outstanding
        if outstanding is not None:
            response.update({
                "action": None,
                "requires_repair_execution_ack": True,
                "repair_execution_ack_required": True,
                "repair_action_id": outstanding.action_id,
                "repair_execution": outstanding.to_dict(),
            })
        return response

    def _awaiting_repair_response(self):
        outstanding = self.repair_execution_tracker.outstanding
        if outstanding is None:
            raise RuntimeError("there is no outstanding repair action")
        return {
            "planner": {
                "decision_type": "awaiting_repair_ack",
                "action_id": outstanding.action_id,
                "reason": (
                    "a dispatched physical repair must be acknowledged before "
                    "another planner decision"
                ),
                "global_risk": self.ledger_runtime.ledger.global_risk(),
                "target_claim_ids": list(outstanding.target_claim_ids),
                "repair_score": None,
                "rollback_event": None,
                "physical_state_rolled_back": False,
            },
            "ledger": self.ledger_runtime.ledger.to_dict(),
            "logical_plan": self.ledger_runtime.plan_state.to_dict(),
            "action": None,
            "requires_repair_execution_ack": True,
            "repair_execution_ack_required": True,
            "repair_action_id": outstanding.action_id,
            "repair_execution": outstanding.to_dict(),
        }

    def _repair_instruction(self, decision):
        if decision.decision_type is PlannerDecisionType.REPAIR:
            for entry in getattr(
                self.job_config, "ledger_repair_catalog", ()):
                if str(entry["name"]) == decision.action_id:
                    return (
                        "Before continuing the task, execute this local recovery "
                        "action: {}".format(entry.get(
                            "description", decision.action_id)))
            return "Execute local repair action: {}".format(decision.action_id)
        checkpoint = (
            None if decision.rollback_event is None
            else decision.rollback_event.checkpoint_id)
        return (
            "Recover locally from the current physical state and resume the "
            "subplan at checkpoint {}. Re-check claims: {}".format(
                checkpoint or "nearest safe stage",
                ", ".join(decision.target_claim_ids) or "unknown",
            )
        )

    def _supplied_repair_action(self, request, action_id):
        action_chunks = request.get("repair_action_chunks")
        if not isinstance(action_chunks, dict) or action_id not in action_chunks:
            return None
        return self._validate_repair_action(action_chunks[action_id])

    def _validate_repair_action(self, action):
        return validate_repair_action_chunk(
            action,
            expected_channels=len(self.job_config.used_action_channel_ids),
            expected_frames=self.job_config.frame_chunk_size,
            actions_per_frame=self.job_config.action_per_frame,
        )
    
    def _repeat_input_for_cfg(self, input_dict):
        if self.use_cfg:
            input_dict['noisy_latents'] = input_dict['noisy_latents'].repeat(2, 1, 1, 1, 1)
            input_dict['text_emb'] = torch.cat([self.prompt_embeds.to(self.dtype).clone(), self.negative_prompt_embeds.to(self.dtype).clone()], dim=0)
            input_dict['grid_id'] = input_dict['grid_id'][None].repeat(2, 1, 1)
            input_dict['timesteps'] = input_dict['timesteps'][None].repeat(2, 1)
        else:
            input_dict['grid_id'] = input_dict['grid_id'][None]
            input_dict['timesteps'] = input_dict['timesteps'][None]
        return input_dict

    def _prepare_latent_input(self,
                              latent_model_input,
                              action_model_input,
                              latent_t=0,
                              action_t=0,
                              latent_cond=None,
                              action_cond=None,
                              frame_st_id=0,
                              patch_size=(1, 2, 2)):
        logger.info(f"FRAME START ID: {frame_st_id}")
        input_dict = dict()
        if latent_model_input is not None:
            input_dict['latent_res_lst'] = {
                'noisy_latents':
                latent_model_input,
                'timesteps':
                torch.ones([latent_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * latent_t,
                'grid_id':
                get_mesh_id(latent_model_input.shape[-3] // patch_size[0],
                            latent_model_input.shape[-2] // patch_size[1],
                            latent_model_input.shape[-1] // patch_size[2], 0,
                            1, frame_st_id).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                input_dict['latent_res_lst'][
                    'noisy_latents'][:, :, 0:1] = latent_cond[:, :, 0:1]
                input_dict['latent_res_lst']['timesteps'][0:1] *= 0

        if action_model_input is not None:
            input_dict['action_res_lst'] = {
                'noisy_latents':
                action_model_input,
                'timesteps':
                torch.ones([action_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * action_t,
                'grid_id':
                get_mesh_id(action_model_input.shape[-3],
                            action_model_input.shape[-2],
                            action_model_input.shape[-1],
                            1,
                            1,
                            frame_st_id,
                            action=True).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }

            if action_cond is not None:
                input_dict['action_res_lst'][
                    'noisy_latents'][:, :, 0:1] = action_cond[:, :, 0:1]
                input_dict['action_res_lst']['timesteps'][0:1] *= 0
            input_dict['action_res_lst']['noisy_latents'][:, ~self.
                                                          action_mask] *= 0
            if self.ledger_enabled and self.ledger_context is not None:
                input_dict['action_res_lst']['ledger_context'] = (
                    self.ledger_context)
        return input_dict

    def _encode_obs(self, obs):
        images = obs['obs']
        if not isinstance(images, list):
            images = [images]
        if len(images) < 1:
            return None
        videos = []
        for k_i, k in enumerate(self.job_config.obs_cam_keys):
            if self.env_type == 'robotwin_tshape':
                if k_i == 0:  # camera high
                    height_i, width_i = self.height, self.width
                else:
                    height_i, width_i = self.height // 2, self.width // 2
            else:
                height_i, width_i = self.height, self.width

            history_video_k = torch.from_numpy(
                np.stack([each[k]
                          for each in images])).float().permute(3, 0, 1, 2)
            history_video_k = F.interpolate(history_video_k,
                                            size=(height_i, width_i),
                                            mode='bilinear',
                                            align_corners=False).unsqueeze(0)
            videos.append(history_video_k)

        if self.env_type == 'robotwin_tshape':
            videos_high = videos[0] / 255.0 * 2.0 - 1.0
            videos_left_and_right = torch.cat(videos[1:],
                                              dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            enc_out_high = self.streaming_vae.encode_chunk(
                videos_high.to(vae_device).to(self.dtype))
            enc_out_left_and_right = self.streaming_vae_half.encode_chunk(
                videos_left_and_right.to(vae_device).to(self.dtype))
            enc_out = torch.cat([
                torch.cat(enc_out_left_and_right.split(1, dim=0), dim=-1),
                enc_out_high
            ],
                                dim=-2)
        else:
            videos = torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            videos_chunk = videos.to(vae_device).to(self.dtype)
            enc_out = self.streaming_vae.encode_chunk(videos_chunk)

        mu, logvar = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(self.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self.vae.config.latents_std).to(mu.device)
        mu_norm = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
        video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)
        return video_latent.to(self.device)

    def _reset(self, prompt=None, ledger_state=None, planning_checkpoints=()):
        logger.info('Reset.')
        self.use_cfg = (self.job_config.guidance_scale > 1) or (self.job_config.action_guidance_scale > 1)
        #### Reset all parameters
        self.frame_st_id = 0
        self.init_latent = None
        self.current_prompt = prompt
        self.last_executed_action = None
        #### clean vae and transformer cache
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()

        self.action_per_frame = self.job_config.action_per_frame
        self.height, self.width = self.job_config.height, self.job_config.width

        if self.env_type == 'robotwin_tshape':
            self.latent_height, self.latent_width = (
                (self.height // 16) * 3) // 2, self.width // 16
            self.streaming_vae_half.clear_cache()
        else:
            self.latent_height, self.latent_width = self.height // 16, self.width // 16 * len(
                self.job_config.obs_cam_keys)

        patch_size = self.job_config.patch_size
        latent_token_per_chunk = (self.job_config.frame_chunk_size *
                                  self.latent_height * self.latent_width) // (
                                      patch_size[0] * patch_size[1] *
                                      patch_size[2])
        action_token_per_chunk = self.job_config.frame_chunk_size * self.action_per_frame
        self.transformer.create_empty_cache(self.cache_name,
                                            self.job_config.attn_window,
                                            latent_token_per_chunk,
                                            action_token_per_chunk,
                                            dtype=self.dtype,
                                            device=self.device,
                                            batch_size = 2 if self.use_cfg else 1
                                            )

        self.action_mask = torch.zeros([self.job_config.action_dim]).bool()
        self.action_mask[self.job_config.used_action_channel_ids] = True

        self.actions_q01 = torch.tensor(self.job_config.norm_stat['q01'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.actions_q99 = torch.tensor(self.job_config.norm_stat['q99'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.action_norm_method = self.job_config.action_norm_method

        ##### get prompt
        if prompt is None:
            self.prompt_embeds = self.negative_prompt_embeds = None
        else:
            self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=None,
                do_classifier_free_guidance=self.use_cfg,
                num_videos_per_prompt=1,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                max_sequence_length=512,
                device=self.device,
                dtype=self.dtype,
            )

        self._reset_ledger_runtime(
            serialized_state=ledger_state,
            checkpoints=planning_checkpoints,
        )

        self.exp_name = f"{prompt}_{time.strftime('%Y%m%d_%H%M%S')}" if prompt else "default"
        self.exp_save_root = os.path.join(self.save_root, 'real', self.exp_name)
        os.makedirs(self.exp_save_root, exist_ok=True)
        torch.cuda.empty_cache()

    def _infer(self, obs, frame_st_id=0, initial_latent=None):
        frame_chunk_size = self.job_config.frame_chunk_size
        if frame_st_id == 0:
            init_latent = (
                initial_latent
                if initial_latent is not None
                else self._encode_obs(obs))
            self.init_latent = init_latent

        latents = torch.randn(1,
                              48,
                              frame_chunk_size,
                              self.latent_height,
                              self.latent_width,
                              device=self.device,
                              dtype=self.dtype)
        actions = torch.randn(1,
                              self.job_config.action_dim,
                              frame_chunk_size,
                              self.action_per_frame,
                              1,
                              device=self.device,
                              dtype=self.dtype)

        video_inference_step = self.job_config.num_inference_steps
        action_inference_step = self.job_config.action_num_inference_steps
        video_step = self.job_config.video_exec_step

        self.scheduler.set_timesteps(video_inference_step)
        self.action_scheduler.set_timesteps(action_inference_step)
        timesteps = self.scheduler.timesteps
        action_timesteps = self.action_scheduler.timesteps

        timesteps = F.pad(timesteps, (0, 1), mode='constant', value=0)

        if video_step != -1:
            timesteps = timesteps[:video_step]

        action_timesteps = F.pad(
            action_timesteps,
            (0,
             1),  # pad 1 element at the end (right side) of the last dimension
            mode='constant',
            value=0)

        with (
                torch.no_grad(),
        ):
            # 1. Video Generation Loop
            for i, t in enumerate(tqdm(timesteps)):
                last_step = i == len(timesteps) - 1
                latent_cond = init_latent[:, :, 0:1].to(
                    self.dtype) if frame_st_id == 0 else None
                input_dict = self._prepare_latent_input(
                    latents,
                    None,
                    t,
                    t,
                    latent_cond,
                    None,
                    frame_st_id=frame_st_id)

                video_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=False)

                if not last_step or video_step != -1:
                    video_noise_pred = data_seq_to_patch(
                        self.job_config.patch_size, video_noise_pred,
                        frame_chunk_size, self.latent_height,
                        self.latent_width, batch_size=2 if self.use_cfg else 1)
                    if self.job_config.guidance_scale > 1:
                        video_noise_pred = video_noise_pred[1:] + self.job_config.guidance_scale * (video_noise_pred[:1] - video_noise_pred[1:])
                    else:
                        video_noise_pred = video_noise_pred[:1]
                    latents = self.scheduler.step(video_noise_pred,
                                                  t,
                                                  latents,
                                                  return_dict=False)

                latents[:, :, 0:1] = latent_cond if frame_st_id == 0 else latents[:, :, 0:1]

            for i, t in enumerate(tqdm(action_timesteps)):
                last_step = i == len(action_timesteps) - 1
                action_cond = torch.zeros(
                    [
                        1, self.job_config.action_dim, 1,
                        self.action_per_frame, 1
                    ],
                    device=self.device,
                    dtype=self.dtype) if frame_st_id == 0 else None

                input_dict = self._prepare_latent_input(
                    None,
                    actions,
                    t,
                    t,
                    None,
                    action_cond,
                    frame_st_id=frame_st_id)
                action_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['action_res_lst']),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=True)

                if not last_step:
                    action_noise_pred = rearrange(action_noise_pred,
                                                  'b (f n) c -> b c f n 1',
                                                  f=frame_chunk_size)
                    if self.job_config.action_guidance_scale > 1:
                        action_noise_pred = action_noise_pred[1:] + self.job_config.action_guidance_scale * (action_noise_pred[:1] - action_noise_pred[1:])
                    else:
                        action_noise_pred = action_noise_pred[:1]
                    actions = self.action_scheduler.step(action_noise_pred,
                                                         t,
                                                         actions,
                                                         return_dict=False)

                actions[:, :, 0:1] = action_cond if frame_st_id == 0 else actions[:, :, 0:1]

        actions[:, ~self.action_mask] *= 0

        save_async(latents, os.path.join(self.exp_save_root, f'latents_{frame_st_id}.pt'))
        save_async(actions, os.path.join(self.exp_save_root, f'actions_{frame_st_id}.pt'))

        actions = self.postprocess_action(actions)
        torch.cuda.empty_cache()
        return actions, latents

    def _compute_kv_cache(self, obs):
        ### optional async save obs for debug
        self.transformer.clear_pred_cache(self.cache_name)
        save_async(obs['obs'], os.path.join(self.exp_save_root, f'obs_data_{self.frame_st_id}.pt'))
        latent_model_input = self._encode_obs(obs)
        if self.frame_st_id == 0:
            latent_model_input = torch.cat(
                [self.init_latent, latent_model_input],
                dim=2) if latent_model_input is not None else self.init_latent

        action_model_input = self.preprocess_action(obs['state'])
        action_model_input = action_model_input.to(latent_model_input)
        self.last_executed_action = np.asarray(obs['state']).copy()
        if self.ledger_enabled:
            self._update_neural_ledger(
                latent_model_input, action_model_input)
        logger.info(
            f"get KV cache obs: {latent_model_input.shape} {action_model_input.shape}"
        )
        input_dict = self._prepare_latent_input(latent_model_input,
                                                action_model_input,
                                                frame_st_id=self.frame_st_id)

        with (
                torch.no_grad(),
        ):
            self.transformer(self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                             update_cache=2,
                             cache_name=self.cache_name,
                             action_mode=False)

            self.transformer(self._repeat_input_for_cfg(input_dict['action_res_lst']),
                             update_cache=2,
                             cache_name=self.cache_name,
                             action_mode=True)
        torch.cuda.empty_cache()
        self.frame_st_id += latent_model_input.shape[2]
        if self.ledger_enabled:
            plan_state = self.ledger_runtime.plan_state
            plan_state.cursor += 1
            checkpoint_id = "observed_chunk_{}".format(plan_state.cursor)
            if checkpoint_id not in plan_state.checkpoints:
                plan_state.add_checkpoint(PlanningCheckpoint(
                    checkpoint_id=checkpoint_id,
                    cursor=plan_state.cursor,
                    subgoal=plan_state.active_subgoal,
                ))

    def _infer_recovery_chunk(self, obs, decision, initial_latent=None):
        original_prompt_embeds = self.prompt_embeds
        original_negative_embeds = self.negative_prompt_embeds
        instruction = self._repair_instruction(decision)
        recovery_prompt = "{}. {}".format(
            self.current_prompt or "Complete the manipulation task",
            instruction,
        )
        try:
            self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(
                prompt=recovery_prompt,
                negative_prompt=None,
                do_classifier_free_guidance=self.use_cfg,
                num_videos_per_prompt=1,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                max_sequence_length=512,
                device=self.device,
                dtype=self.dtype,
            )
            return self._infer(
                obs,
                frame_st_id=self.frame_st_id,
                initial_latent=initial_latent,
            )
        finally:
            self.prompt_embeds = original_prompt_embeds
            self.negative_prompt_embeds = original_negative_embeds

    @torch.no_grad()
    def infer(self, obs):
        reset = obs.get('reset', False)
        prompt = obs.get('prompt', None)
        compute_kv_cache = obs.get('compute_kv_cache', False)

        if reset:
            logger.info("******************* Reset server ******************")
            self._reset(
                prompt=prompt,
                ledger_state=obs.get("ledger_state"),
                planning_checkpoints=obs.get("planning_checkpoints", ()),
            )
            if self.ledger_enabled:
                # Reset/load is state initialization, not an executable planner
                # transaction.  In particular, restored refutations must stay
                # unresolved until the next request can provide a concrete
                # recovery action through the normal handshake.
                return {
                    "planner": {
                        "decision_type": "state_loaded",
                        "action_id": None,
                        "reason": "ledger runtime initialized without planning",
                        "global_risk": (
                            self.ledger_runtime.ledger.global_risk()),
                        "target_claim_ids": [],
                        "repair_score": None,
                        "rollback_event": None,
                        "physical_state_rolled_back": False,
                    },
                    "ledger": self.ledger_runtime.ledger.to_dict(),
                    "logical_plan": self.ledger_runtime.plan_state.to_dict(),
                }
            return dict()
        elif compute_kv_cache:
            logger.info(
                "################# Compute KV Cache #################")
            acknowledgement = None
            if self.ledger_enabled:
                acknowledgement = self._apply_repair_execution_ack(
                    obs, allow_implicit=True)
            self._apply_external_ledger_update(obs)
            self._compute_kv_cache(obs)
            if self.ledger_enabled:
                response = {
                    "ledger": self.ledger_runtime.ledger.to_dict(),
                    "logical_plan": self.ledger_runtime.plan_state.to_dict(),
                }
                if acknowledgement is not None:
                    response["repair_execution_ack"] = (
                        acknowledgement.to_dict())
                return response
            return dict()
        else:
            logger.info("################# Infer One Chunk #################")
            if not self.ledger_enabled:
                action, _ = self._infer(obs, frame_st_id=self.frame_st_id)
                return dict(action=action)

            if obs.get("repair_execution_ack") is not None:
                raise ValueError(
                    "repair_execution_ack must be sent with "
                    "compute_kv_cache=True so the post-repair observation is "
                    "committed to both the Ledger and Transformer KV cache"
                )
            self._apply_external_ledger_update(obs)
            initial_latent = None
            if self.ledger_context is None:
                initial_latent = self._encode_obs(obs)
                if self.frame_st_id == 0:
                    self.init_latent = initial_latent
                observed_actions = None
                if obs.get("state") is not None:
                    observed_actions = self.preprocess_action(obs["state"])
                    observed_actions = observed_actions.to(initial_latent)
                self._update_neural_ledger(
                    initial_latent, observed_actions)

            if self.repair_execution_tracker.outstanding is not None:
                # Do not call the planner here: rollback decisions mutate the
                # logical ledger, so even a discarded second decision would be
                # an observable state change while the first action is pending.
                return self._awaiting_repair_response()

            pre_decision_runtime = self.ledger_runtime.to_dict()
            decision = self._planner_decision(obs)
            response = self._planner_response(decision)
            if decision.decision_type is PlannerDecisionType.TASK:
                action, _ = self._infer(
                    obs,
                    frame_st_id=self.frame_st_id,
                    initial_latent=initial_latent,
                )
            else:
                recovery_action_id = (
                    decision.action_id
                    if decision.action_id is not None else "local_rollback")
                try:
                    supplied_action = self._supplied_repair_action(
                        obs, recovery_action_id)
                    if supplied_action is not None:
                        action = supplied_action
                        repair_source = "supplied_chunk"
                    elif not bool(getattr(
                        self.job_config,
                        "ledger_allow_prompt_repair_fallback",
                        False,
                    )):
                        if decision.decision_type is PlannerDecisionType.ROLLBACK:
                            # Planner rollback is a proposal until a concrete
                            # recovery action can be issued.  Restore the
                            # logical transaction when execution is unavailable.
                            self.ledger_runtime = LedgerRuntimeState.from_dict(
                                pre_decision_runtime)
                            response = self._planner_response(decision)
                        response["action"] = None
                        response["requires_repair_action"] = True
                        response["repair_action_id"] = recovery_action_id
                        response["repair_instruction"] = self._repair_instruction(
                            decision)
                        return response
                    else:
                        if decision.decision_type is PlannerDecisionType.ROLLBACK:
                            self.transformer.clear_pred_cache(self.cache_name)
                        action, _ = self._infer_recovery_chunk(
                            obs, decision, initial_latent=initial_latent)
                        action = self._validate_repair_action(action)
                        repair_source = "prompt_recovery"
                    issued_repair = self.repair_execution_tracker.issue(
                        recovery_action_id,
                        decision.target_claim_ids,
                        source=repair_source,
                        issued_at=self.frame_st_id,
                    )
                except Exception:
                    if decision.decision_type is PlannerDecisionType.ROLLBACK:
                        self.ledger_runtime = LedgerRuntimeState.from_dict(
                            pre_decision_runtime)
                    raise
                response.update({
                    "repair_action_id": recovery_action_id,
                    "repair_execution_ack_required": True,
                    "requires_repair_execution_ack": True,
                    "repair_execution": issued_repair.to_dict(),
                })
            response["action"] = action
            return response
    
    def decode_one_video(self, latents, output_type):
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
        return video
    
    def load_init_obs(self):
        imf_dict = {v: np.array(Image.open(os.path.join(self.job_config.input_img_path, f"{v}.png")).convert("RGB")) for v in self.job_config.obs_cam_keys}
        init_obs = {}
        init_obs['obs'] = [imf_dict]
        return init_obs
    
    @torch.no_grad()
    def generate(self):
        self.video_processor = VideoProcessor(vae_scale_factor=1)
        self._reset(self.job_config.prompt)
        init_obs = self.load_init_obs()
        pred_latent_lst = []
        for chunk_id in range(self.job_config.num_chunks_to_infer):
            _actions, latents = self._infer(init_obs, frame_st_id=(chunk_id * self.job_config.frame_chunk_size))
            pred_latent_lst.append(latents)
        pred_latent = torch.cat(pred_latent_lst, dim=2)
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()
        if self.streaming_vae_half:
            self.streaming_vae_half.clear_cache()
        del self.transformer
        del self.streaming_vae_half
        del self.text_encoder
        torch.cuda.empty_cache()
        
        # Move VAE to GPU for decoding
        if self.enable_offload:
            self.vae = self.vae.to(self.device).to(self.dtype)
        
        decoded_video = self.decode_one_video(pred_latent, 'np')[0]
        export_to_video(decoded_video, os.path.join(self.save_root, "demo.mp4"), fps=10)

def run(args):    
    
    config = VA_CONFIGS[args.config_name]
    port = config.port if args.port is None else args.port
    if args.save_root is not None:
        config.save_root = args.save_root
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    model = VA_Server(config)
    if config.infer_mode == 'i2va':
        logger.info("******************************USE I2AV mode******************************")
        model.generate()
    elif config.infer_mode == 'server':
        logger.info("******************************USE Server mode******************************")
        run_async_server_mode(model, local_rank, config.host, port)
    else:
        raise ValueError(f"Unknown infer mode: {config.infer_mode}")

def main():
    """
    TODO
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        type=str,
        required=False,
        default='robotwin',
        help="config name.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help='(start) port'
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=None,
        help='save root'
    )
    args = parser.parse_args()
    run(args)
    logger.info("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    init_logger()
    main()
