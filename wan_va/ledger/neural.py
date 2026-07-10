"""Pure-PyTorch neural causal-belief ledger components.

The module intentionally has no dependency on the Wan runtime, diffusers, or
flash-attn.  It can therefore be trained and tested independently, then used as
a side head on top of the existing World Action Model.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor

LEDGER_LOSS_NAMES: Tuple[str, ...] = (
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


def _positive(raw_value: Tensor) -> Tensor:
    """Map learned scalar/vector parameters to strictly positive values."""

    return F.softplus(raw_value) + torch.finfo(raw_value.dtype).eps


def _flatten_valid_mask(
    mask: Optional[Tensor],
    batch_size: int,
    token_count: int,
    device: torch.device,
    name: str,
) -> Tensor:
    """Return a boolean ``[B, token_count]`` mask where True means valid."""

    if mask is None:
        return torch.ones(batch_size, token_count, dtype=torch.bool, device=device)
    if mask.shape[0] != batch_size:
        raise ValueError(
            "%s mask batch size %d does not match input batch size %d"
            % (name, mask.shape[0], batch_size)
        )
    mask = mask.reshape(batch_size, -1).to(device=device, dtype=torch.bool)
    if mask.shape[1] != token_count:
        raise ValueError(
            "%s mask has %d tokens, expected %d" % (name, mask.shape[1], token_count)
        )
    return mask


def _masked_token_mean(tokens: Tensor, valid_mask: Tensor) -> Tensor:
    weights = valid_mask.to(dtype=tokens.dtype).unsqueeze(-1)
    numerator = (tokens * weights).sum(dim=1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return numerator / denominator


class NeuralLedgerHead(nn.Module):
    """Generate a fixed set of structured causal claim slots.

    Args:
        latent_channels: Channel dimension of ``latents``.
        action_dim: Channel dimension of ``actions``.
        text_dim: Last dimension of the input text embeddings.
        hidden_dim: Internal claim-slot and context width.
        num_claim_slots: Fixed number of causal claims in the ledger.
        num_relations: Size of the relation ontology.
        num_rollback_steps: Number of discrete rollback locations.
        num_repair_actions: Number of repair candidates/skills.
        delta_dim: Width of factual, counterfactual, and repair-world deltas.
        num_heads: Number of attention heads.
        dropout: Attention/MLP dropout probability.
        num_claim_types: Size of the causal-claim type ontology.
        num_subjects: Size of the subject/entity vocabulary.
        num_objects: Size of the object/entity vocabulary.
        num_preconditions: Size of the action-precondition ontology.
        num_effects: Size of the action-effect ontology.

    Inputs:
        latents: ``[B, C, F, H, W]`` dense visual latents.
        actions: ``[B, A, F, N, 1]`` action/state trajectories.
        text_emb: ``[B, L, T]`` language embeddings.

    Optional ``masks`` entries are named ``latent``, ``action``, and ``text``.
    They use True for a valid token and may be supplied in flattened or native
    spatial shape.  An explicit boolean ``claim`` mask is treated as the hard
    valid-slot set.  Without it, learned presence probabilities softly weight
    all ledger-level aggregations.  ``counterfactual_actions`` has the same
    layout as actions.  ``previous_claim_slots``, when supplied, has shape
    ``[B, num_claim_slots, hidden_dim]`` and anchors the output slot identity to
    the preceding ledger state. ``previous_claim_mask`` optionally marks which
    historical slots own an identity; unused slots are initialized from the
    current candidates instead of copying historical padding.
    """

    _DEBT_COMPONENTS: Tuple[str, ...] = (
        "lack_of_confidence",
        "uncertainty",
        "dependency",
        "repair_cost",
        "lack_of_observability",
    )

    def __init__(
        self,
        latent_channels: int = 48,
        action_dim: int = 30,
        text_dim: int = 4096,
        hidden_dim: int = 256,
        num_claim_slots: int = 16,
        num_relations: int = 12,
        num_rollback_steps: int = 16,
        num_repair_actions: int = 8,
        delta_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.0,
        num_claim_types: int = 8,
        num_subjects: int = 64,
        num_objects: int = 64,
        num_preconditions: int = 16,
        num_effects: int = 16,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        for name, value in (
            ("num_claim_slots", num_claim_slots),
            ("num_relations", num_relations),
            ("num_rollback_steps", num_rollback_steps),
            ("num_repair_actions", num_repair_actions),
            ("num_claim_types", num_claim_types),
            ("num_subjects", num_subjects),
            ("num_objects", num_objects),
            ("num_preconditions", num_preconditions),
            ("num_effects", num_effects),
        ):
            if value <= 0:
                raise ValueError("%s must be positive" % name)

        self.latent_channels = latent_channels
        self.action_dim = action_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_claim_slots = num_claim_slots
        self.num_relations = num_relations
        self.num_rollback_steps = num_rollback_steps
        self.num_repair_actions = num_repair_actions
        self.num_claim_types = num_claim_types
        self.num_subjects = num_subjects
        self.num_objects = num_objects
        self.num_preconditions = num_preconditions
        self.num_effects = num_effects
        self.delta_dim = hidden_dim if delta_dim is None else delta_dim

        self.latent_projection = nn.Sequential(
            nn.LayerNorm(latent_channels), nn.Linear(latent_channels, hidden_dim)
        )
        self.action_projection = nn.Sequential(
            nn.LayerNorm(action_dim), nn.Linear(action_dim, hidden_dim)
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden_dim)
        )

        self.modality_embedding = nn.Parameter(torch.empty(3, hidden_dim))
        self.null_context = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.claim_queries = nn.Parameter(torch.empty(num_claim_slots, hidden_dim))
        self.global_seed = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )

        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.slot_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.slot_norm = nn.LayerNorm(hidden_dim)
        self.slot_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

        self.claim_head = nn.Linear(hidden_dim, 1)
        self.presence_head = nn.Linear(hidden_dim, 1)
        self.claim_type_head = nn.Linear(hidden_dim, num_claim_types)
        self.subject_head = nn.Linear(hidden_dim, num_subjects)
        self.object_head = nn.Linear(hidden_dim, num_objects)
        self.precondition_head = nn.Linear(hidden_dim, num_preconditions)
        self.effect_head = nn.Linear(hidden_dim, num_effects)
        self.evidence_head = nn.Linear(hidden_dim, 1)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)
        self.dependency_head = nn.Linear(hidden_dim, 1)
        # Directed pairwise scores model whether claim ``source`` is a
        # prerequisite of claim ``target``.  Separate projections intentionally
        # avoid forcing the dependency graph to be symmetric.
        self.dependency_source_projection = nn.Linear(
            hidden_dim, hidden_dim, bias=False
        )
        self.dependency_target_projection = nn.Linear(
            hidden_dim, hidden_dim, bias=False
        )
        self.dependency_matrix_bias = nn.Parameter(torch.zeros(()))
        self.repair_cost_head = nn.Linear(hidden_dim, 1)
        self.observability_head = nn.Linear(hidden_dim, 1)
        self.importance_head = nn.Linear(hidden_dim, 1)
        self.relation_head = nn.Linear(hidden_dim, num_relations)
        self.rollback_head = nn.Linear(hidden_dim, num_rollback_steps)

        # softplus(raw_debt_weights) makes every derivative with respect to a
        # risk component non-negative.  Confidence and observability enter as
        # their complements, so increasing either cannot increase debt.
        self.raw_debt_weights = nn.Parameter(torch.zeros(len(self._DEBT_COMPONENTS)))
        self.debt_bias = nn.Parameter(torch.tensor(-2.0))

        # Factual and counterfactual actions must pass through exactly the same
        # transition function.  Separate heads create a trivial shortcut where
        # different biases satisfy the contrastive margin without learning any
        # action-conditioned causal effect.
        self.action_delta_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.delta_dim),
        )

        self.repair_queries = nn.Parameter(torch.empty(num_repair_actions, hidden_dim))
        self.repair_fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.repair_claim_fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU()
        )
        self.repair_reduction_head = nn.Linear(hidden_dim, 1)
        self.repair_logit_head = nn.Linear(hidden_dim, 1)
        self.repair_action_cost_head = nn.Linear(hidden_dim, 1)
        self.repair_world_head = nn.Linear(hidden_dim, self.delta_dim)

        self._reset_parameters()

        # Appending the state updater after the legacy parameter reset keeps
        # the no-history path and its initialization independent of this
        # optional recurrent update.  Previous slots query the new candidates,
        # so every result position retains the previous ledger's ordering.
        self.previous_slot_norm = nn.LayerNorm(hidden_dim)
        self.slot_update_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.slot_update_gate_head = nn.Linear(2 * hidden_dim, hidden_dim)
        self.slot_update_norm = nn.LayerNorm(hidden_dim)
        nn.init.zeros_(self.slot_update_gate_head.weight)
        nn.init.constant_(self.slot_update_gate_head.bias, -1.0)

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.modality_embedding, std=0.02)
        nn.init.normal_(self.null_context, std=0.02)
        nn.init.normal_(self.claim_queries, std=0.02)
        nn.init.normal_(self.repair_queries, std=0.02)

    def _validate_inputs(
        self, latents: Tensor, actions: Tensor, text_emb: Tensor
    ) -> int:
        if latents.ndim != 5:
            raise ValueError("latents must have shape [B, C, F, H, W]")
        if actions.ndim != 5 or actions.shape[-1] != 1:
            raise ValueError("actions must have shape [B, A, F, N, 1]")
        if text_emb.ndim != 3:
            raise ValueError("text_emb must have shape [B, L, T]")
        batch_size = latents.shape[0]
        if actions.shape[0] != batch_size or text_emb.shape[0] != batch_size:
            raise ValueError("all ledger inputs must have the same batch size")
        if latents.shape[1] != self.latent_channels:
            raise ValueError(
                "latents have %d channels, expected %d"
                % (latents.shape[1], self.latent_channels)
            )
        if actions.shape[1] != self.action_dim:
            raise ValueError(
                "actions have %d channels, expected %d"
                % (actions.shape[1], self.action_dim)
            )
        if text_emb.shape[-1] != self.text_dim:
            raise ValueError(
                "text embeddings have width %d, expected %d"
                % (text_emb.shape[-1], self.text_dim)
            )
        return batch_size

    def _encode_action_tokens(self, actions: Tensor) -> Tensor:
        tokens = (
            actions.squeeze(-1)
            .permute(0, 2, 3, 1)
            .reshape(actions.shape[0], -1, self.action_dim)
        )
        projected: Tensor = self.action_projection(tokens)
        encoded: Tensor = projected + self.modality_embedding[1]
        return encoded

    def compute_debt(
        self,
        confidence: Tensor,
        uncertainty: Tensor,
        dependency: Tensor,
        repair_cost: Tensor,
        observability: Tensor,
        importance: Tensor,
        claim_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute monotone per-claim debt and global risk.

        All supplied components should be probabilities in ``[0, 1]``.  The
        learned component weights are strictly positive.  Consequently debt is
        monotonically increasing in uncertainty, dependency, and repair cost,
        and monotonically decreasing in confidence/observability.  Importance
        is deliberately kept out of per-claim debt and instead supplies the
        normalized, non-negative weights of the global risk average.
        """

        reference_shape = confidence.shape
        components = (
            uncertainty,
            dependency,
            repair_cost,
            observability,
            importance,
        )
        if any(value.shape != reference_shape for value in components):
            raise ValueError("all debt components must have the same shape")

        debt_features = torch.stack(
            (
                1.0 - confidence,
                uncertainty,
                dependency,
                repair_cost,
                1.0 - observability,
            ),
            dim=-1,
        )
        debt_weights = _positive(self.raw_debt_weights)
        debt_logits = (debt_features * debt_weights).sum(dim=-1) + self.debt_bias
        debt = torch.sigmoid(debt_logits)
        if claim_mask is None:
            claim_weight = torch.ones_like(debt)
        else:
            if claim_mask.shape != debt.shape:
                raise ValueError("claim_mask must have the same shape as debt")
            claim_weight = claim_mask.to(device=debt.device, dtype=debt.dtype)
        normalized_weight = importance * claim_weight
        global_risk = (debt * normalized_weight).sum(dim=-1) / (
            normalized_weight.sum(dim=-1).clamp_min(1e-6)
        )
        return {
            "debt_features": debt_features,
            "debt_weights": debt_weights,
            "debt_bias": self.debt_bias,
            "debt_logits": debt_logits,
            "debt": debt,
            "global_risk": global_risk,
        }

    def forward(
        self,
        latents: Tensor,
        actions: Tensor,
        text_emb: Tensor,
        counterfactual_actions: Optional[Tensor] = None,
        masks: Optional[Mapping[str, Tensor]] = None,
        previous_claim_slots: Optional[Tensor] = None,
        previous_claim_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        batch_size = self._validate_inputs(latents, actions, text_emb)
        masks = {} if masks is None else masks
        claim_mask_is_explicit = masks.get("claim") is not None

        latent_tokens = latents.permute(0, 2, 3, 4, 1).reshape(
            batch_size, -1, self.latent_channels
        )
        latent_tokens = (
            self.latent_projection(latent_tokens) + self.modality_embedding[0]
        )
        action_tokens = self._encode_action_tokens(actions)
        text_tokens = self.text_projection(text_emb) + self.modality_embedding[2]

        latent_valid = _flatten_valid_mask(
            masks.get("latent"),
            batch_size,
            latent_tokens.shape[1],
            latents.device,
            "latent",
        )
        action_valid = _flatten_valid_mask(
            masks.get("action"),
            batch_size,
            action_tokens.shape[1],
            actions.device,
            "action",
        )
        text_valid = _flatten_valid_mask(
            masks.get("text"),
            batch_size,
            text_tokens.shape[1],
            text_emb.device,
            "text",
        )
        claim_valid = _flatten_valid_mask(
            masks.get("claim"),
            batch_size,
            self.num_claim_slots,
            latents.device,
            "claim",
        )

        latent_summary = _masked_token_mean(latent_tokens, latent_valid)
        action_summary = _masked_token_mean(action_tokens, action_valid)
        text_summary = _masked_token_mean(text_tokens, text_valid)
        seed = self.global_seed(
            torch.cat((latent_summary, action_summary, text_summary), dim=-1)
        )

        # A permanently valid null token prevents NaNs even if callers mask all
        # tokens of every real modality.
        null_context = self.null_context.expand(batch_size, -1, -1)
        context_tokens = torch.cat(
            (null_context, latent_tokens, action_tokens, text_tokens), dim=1
        )
        null_valid = torch.ones(batch_size, 1, dtype=torch.bool, device=latents.device)
        context_valid = torch.cat(
            (null_valid, latent_valid, action_valid, text_valid), dim=1
        )

        claim_slots = self.claim_queries.unsqueeze(0).expand(batch_size, -1, -1)
        claim_slots = claim_slots + seed.unsqueeze(1)
        cross_output, _ = self.cross_attention(
            claim_slots,
            context_tokens,
            context_tokens,
            key_padding_mask=~context_valid,
            need_weights=False,
        )
        claim_slots = self.cross_norm(claim_slots + cross_output)
        slot_output, _ = self.slot_attention(
            claim_slots, claim_slots, claim_slots, need_weights=False
        )
        claim_slots = self.slot_norm(claim_slots + slot_output)
        claim_slots = self.output_norm(claim_slots + self.slot_mlp(claim_slots))

        slot_update_output: Dict[str, Tensor] = {}
        if previous_claim_slots is not None:
            expected_previous_shape = (
                batch_size,
                self.num_claim_slots,
                self.hidden_dim,
            )
            if (
                not torch.is_tensor(previous_claim_slots)
                or tuple(previous_claim_slots.shape) != expected_previous_shape
            ):
                actual_shape = (
                    tuple(previous_claim_slots.shape)
                    if torch.is_tensor(previous_claim_slots)
                    else type(previous_claim_slots).__name__
                )
                raise ValueError(
                    "previous_claim_slots must have shape %s, got %s"
                    % (expected_previous_shape, actual_shape)
                )
            claim_slot_candidates = claim_slots
            previous_claim_slots = previous_claim_slots.to(
                device=claim_slots.device, dtype=claim_slots.dtype
            )
            normalized_previous_slots = self.previous_slot_norm(previous_claim_slots)

            # Explicitly annotated null/padding candidates must not attract a
            # previous claim during identity matching.  Keep one temporary key
            # available for an all-empty record so MultiheadAttention remains
            # finite; the public claim mask still marks every slot invalid.
            attention_claim_valid = claim_valid.clone()
            empty_candidate_rows = ~attention_claim_valid.any(dim=-1)
            if empty_candidate_rows.any():
                attention_claim_valid[empty_candidate_rows, 0] = True
            matched_claim_slots, slot_matching_weights = self.slot_update_attention(
                normalized_previous_slots,
                claim_slot_candidates,
                claim_slot_candidates,
                key_padding_mask=~attention_claim_valid,
                need_weights=True,
                average_attn_weights=True,
            )
            slot_update_gate = torch.sigmoid(
                self.slot_update_gate_head(
                    torch.cat((normalized_previous_slots, matched_claim_slots), dim=-1)
                )
            )
            updated_claim_slots = self.slot_update_norm(
                (1.0 - slot_update_gate) * normalized_previous_slots
                + slot_update_gate * matched_claim_slots
            )
            if previous_claim_mask is None:
                previous_valid = claim_valid
            else:
                previous_valid = _flatten_valid_mask(
                    previous_claim_mask,
                    batch_size,
                    self.num_claim_slots,
                    claim_slots.device,
                    "previous claim",
                )
            # An unused historical slot may become active again.  In that case
            # use the current candidate instead of forcing it to copy a null
            # previous slot; valid previous slots retain their stable order.
            claim_slots = torch.where(
                previous_valid.unsqueeze(-1),
                updated_claim_slots,
                claim_slot_candidates,
            )
            slot_update_output = {
                "claim_slot_candidates": claim_slot_candidates,
                "slot_matching_weights": slot_matching_weights,
                "slot_update_gate": slot_update_gate,
                "slot_update_delta": claim_slots - normalized_previous_slots,
                "previous_claim_mask": previous_valid,
            }

        presence_logits = self.presence_head(claim_slots).squeeze(-1)
        presence = torch.sigmoid(presence_logits)
        active_claim_mask = presence.ge(0.5) & claim_valid
        if claim_mask_is_explicit:
            aggregation_claim_weight = claim_valid.to(dtype=claim_slots.dtype)
        else:
            aggregation_claim_weight = presence

        claim_logits = self.claim_head(claim_slots).squeeze(-1)
        confidence = torch.sigmoid(claim_logits)
        evidence_logits = self.evidence_head(claim_slots).squeeze(-1)
        evidence = torch.sigmoid(evidence_logits)
        uncertainty_logits = self.uncertainty_head(claim_slots).squeeze(-1)
        uncertainty = torch.sigmoid(uncertainty_logits)
        dependency_logits = self.dependency_head(claim_slots).squeeze(-1)
        dependency = torch.sigmoid(dependency_logits)
        dependency_source = self.dependency_source_projection(claim_slots)
        dependency_target = self.dependency_target_projection(claim_slots)
        dependency_matrix_logits = (
            torch.matmul(dependency_source, dependency_target.transpose(-1, -2))
            * (self.hidden_dim**-0.5)
            + self.dependency_matrix_bias
        )
        dependency_matrix = torch.sigmoid(dependency_matrix_logits)
        repair_cost_logits = self.repair_cost_head(claim_slots).squeeze(-1)
        repair_cost = torch.sigmoid(repair_cost_logits)
        observability_logits = self.observability_head(claim_slots).squeeze(-1)
        observability = torch.sigmoid(observability_logits)
        importance_logits = self.importance_head(claim_slots).squeeze(-1)
        importance = torch.sigmoid(importance_logits)

        debt_output = self.compute_debt(
            confidence,
            uncertainty,
            dependency,
            repair_cost,
            observability,
            importance,
            claim_mask=aggregation_claim_weight,
        )

        context_weight = confidence * importance * aggregation_claim_weight
        context_denominator = context_weight.sum(dim=1, keepdim=True)
        pooled_claim_context = (claim_slots * context_weight.unsqueeze(-1)).sum(
            dim=1
        ) / context_denominator.clamp_min(1e-6)
        # If all claim slots are masked, retain the multimodal seed rather than
        # returning an arbitrary or non-finite weighted average.
        ledger_context = torch.where(
            context_denominator.gt(0), pooled_claim_context, seed
        )

        action_for_slots = action_summary.unsqueeze(1).expand(
            -1, self.num_claim_slots, -1
        )
        factual_delta = self.action_delta_head(
            torch.cat((claim_slots, action_for_slots), dim=-1)
        )

        if counterfactual_actions is None:
            counterfactual_summary = action_summary
        else:
            if counterfactual_actions.shape != actions.shape:
                raise ValueError(
                    "counterfactual_actions must have the same shape as actions"
                )
            counterfactual_tokens = self._encode_action_tokens(counterfactual_actions)
            counterfactual_summary = _masked_token_mean(
                counterfactual_tokens, action_valid
            )
        counterfactual_for_slots = counterfactual_summary.unsqueeze(1).expand(
            -1, self.num_claim_slots, -1
        )
        counterfactual_delta = self.action_delta_head(
            torch.cat((claim_slots, counterfactual_for_slots), dim=-1)
        )

        repair_queries = self.repair_queries.unsqueeze(0).expand(batch_size, -1, -1)
        ledger_for_repairs = ledger_context.unsqueeze(1).expand(
            -1, self.num_repair_actions, -1
        )
        repair_features = self.repair_fusion(
            torch.cat((repair_queries, ledger_for_repairs), dim=-1)
        )
        repair_logits = self.repair_logit_head(repair_features).squeeze(-1)
        repair_action_cost = torch.sigmoid(
            self.repair_action_cost_head(repair_features).squeeze(-1)
        )
        repair_world_delta = self.repair_world_head(repair_features)

        repair_for_claims = repair_features.unsqueeze(2).expand(
            -1, -1, self.num_claim_slots, -1
        )
        claims_for_repairs = claim_slots.unsqueeze(1).expand(
            -1, self.num_repair_actions, -1, -1
        )
        repair_claim_features = self.repair_claim_fusion(
            torch.cat((claims_for_repairs, repair_for_claims), dim=-1)
        )
        # The lightweight repair-world prediction is part of the structured
        # transition used to estimate post-repair debt, so it receives signal
        # even when a dataset only annotates debt rather than a dense delta.
        repair_reduction_logits = self.repair_reduction_head(
            repair_claim_features
        ).squeeze(-1) + repair_world_delta.mean(dim=-1).unsqueeze(-1)
        # A repair hypothesis may help or harm a claim.  Positive transition
        # effects reduce debt; negative effects increase it toward one.  This
        # bounded parameterization can therefore learn failed repairs while
        # keeping every predicted post-repair debt in [0, 1].
        repair_reduction = torch.tanh(repair_reduction_logits)
        current_debt = debt_output["debt"].unsqueeze(1)
        repair_post_debt_per_claim = torch.where(
            repair_reduction >= 0,
            current_debt * (1.0 - repair_reduction),
            current_debt + (-repair_reduction) * (1.0 - current_debt),
        )
        repair_debt_change_per_claim = current_debt - repair_post_debt_per_claim
        normalized_weight = importance * aggregation_claim_weight.to(
            dtype=repair_post_debt_per_claim.dtype
        )
        repair_post_debt = (
            repair_post_debt_per_claim * normalized_weight.unsqueeze(1)
        ).sum(dim=-1) / normalized_weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        repair_debt_reduction = (
            debt_output["global_risk"].unsqueeze(-1) - repair_post_debt
        )
        # Runtime combines this calibrated debt reduction with configured
        # physical action cost, task risk, and policy-prior coefficients.  Keep
        # the neural score free of a second learned cost scale so training and
        # online planning use the same explicit utility decomposition.
        repair_scores = repair_debt_reduction

        output = {
            "claim_slots": claim_slots,
            # Backward-compatible hard validity envelope: when no mask is
            # supplied every fixed query slot is valid, while presence carries
            # the learned active/null decision used by aggregation.
            "claim_mask": claim_valid,
            "active_claim_mask": active_claim_mask,
            "claim_aggregation_weight": aggregation_claim_weight,
            "presence_logits": presence_logits,
            "presence": presence,
            "claim_presence_logits": presence_logits,
            "claim_presence": presence,
            "claim_logits": claim_logits,
            "confidence": confidence,
            "claim_type_logits": self.claim_type_head(claim_slots),
            "subject_logits": self.subject_head(claim_slots),
            "object_logits": self.object_head(claim_slots),
            "precondition_logits": self.precondition_head(claim_slots),
            "effect_logits": self.effect_head(claim_slots),
            "evidence_logits": evidence_logits,
            "evidence": evidence,
            "uncertainty_logits": uncertainty_logits,
            "uncertainty": uncertainty,
            "dependency_logits": dependency_logits,
            "dependency": dependency,
            "dependency_matrix_logits": dependency_matrix_logits,
            "dependency_matrix": dependency_matrix,
            # Explicit aliases make it unambiguous that the schema-named
            # matrix stores probabilities, while preserving the concise key.
            "dependency_matrix_probability": dependency_matrix,
            "dependency_matrix_probabilities": dependency_matrix,
            "repair_cost_logits": repair_cost_logits,
            "repair_cost": repair_cost,
            "observability_logits": observability_logits,
            "observability": observability,
            "importance_logits": importance_logits,
            "importance": importance,
            "relation_logits": self.relation_head(claim_slots),
            "rollback_logits": self.rollback_head(claim_slots),
            "factual_delta": factual_delta,
            "counterfactual_delta": counterfactual_delta,
            "ledger_context": ledger_context,
            "repair_features": repair_features,
            "repair_logits": repair_logits,
            "repair_action_cost": repair_action_cost,
            "repair_reduction": repair_reduction,
            "repair_debt_change_per_claim": repair_debt_change_per_claim,
            "repair_post_debt_per_claim": repair_post_debt_per_claim,
            "repair_post_debt": repair_post_debt,
            "repair_debt_reduction": repair_debt_reduction,
            "repair_scores": repair_scores,
            "repair_world_delta": repair_world_delta,
        }
        output.update(slot_update_output)
        output.update(debt_output)
        return output


def _first_present(
    mapping: Mapping[str, Tensor], names: Sequence[str]
) -> Optional[Tensor]:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def _masked_mean(loss: Tensor, mask: Optional[Tensor]) -> Tensor:
    if mask is None:
        return loss.mean()
    mask = mask.to(device=loss.device, dtype=loss.dtype)
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    try:
        mask = mask.expand_as(loss)
    except RuntimeError as exc:
        raise ValueError(
            "loss mask shape %s cannot broadcast to loss shape %s"
            % (tuple(mask.shape), tuple(loss.shape))
        ) from exc
    denominator = mask.sum().clamp_min(1.0)
    # torch.where prevents ignored NaN/Inf labels from contaminating a masked
    # reduction (multiplying NaN by zero would still produce NaN).
    safe_loss = torch.where(mask.ne(0), loss, torch.zeros_like(loss))
    return (safe_loss * mask).sum() / denominator


def _combine_loss_masks(
    first: Optional[Tensor], second: Tensor, reference: Tensor
) -> Tensor:
    """Broadcast and combine two validity masks for ``reference``."""

    combined = second.to(device=reference.device, dtype=torch.bool)
    while combined.ndim < reference.ndim:
        combined = combined.unsqueeze(-1)
    try:
        combined = combined.expand_as(reference)
    except RuntimeError as exc:
        raise ValueError("validity mask cannot broadcast to loss") from exc
    if first is None:
        return combined
    external = first.to(device=reference.device, dtype=torch.bool)
    while external.ndim < reference.ndim:
        external = external.unsqueeze(-1)
    try:
        external = external.expand_as(reference)
    except RuntimeError as exc:
        raise ValueError("external mask cannot broadcast to loss") from exc
    return combined & external


def _probability_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Optional[Tensor],
    with_logits: bool,
    regression: bool = False,
) -> Tensor:
    """Loss for [0,1] labels with automatic -100/NaN sentinel masking."""

    target = target.to(prediction)
    if target.ndim == prediction.ndim + 1 and target.shape[-1] == 1:
        target = target.squeeze(-1)
    if target.shape != prediction.shape:
        raise ValueError(
            "probability target shape %s does not match prediction shape %s"
            % (tuple(target.shape), tuple(prediction.shape))
        )

    valid = torch.isfinite(target) & target.ge(0.0) & target.le(1.0)
    if mask is not None:
        external_mask = mask.to(device=prediction.device, dtype=torch.bool)
        while external_mask.ndim < valid.ndim:
            external_mask = external_mask.unsqueeze(-1)
        try:
            external_mask = external_mask.expand_as(valid)
        except RuntimeError as exc:
            raise ValueError("probability mask cannot broadcast to target") from exc
        valid = valid & external_mask
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    if regression:
        raw = F.smooth_l1_loss(prediction, safe_target, reduction="none")
    elif with_logits:
        raw = F.binary_cross_entropy_with_logits(
            prediction, safe_target, reduction="none"
        )
    else:
        raw = F.binary_cross_entropy(prediction, safe_target, reduction="none")
    return _masked_mean(raw, valid)


def _regression_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Optional[Tensor],
    non_negative: bool = False,
) -> Tensor:
    """Masked Smooth-L1 loss with automatic NaN/-100 sentinel handling."""

    target = target.to(prediction)
    if target.ndim == prediction.ndim + 1 and target.shape[-1] == 1:
        target = target.squeeze(-1)
    if target.shape != prediction.shape:
        raise ValueError(
            "regression target shape %s does not match prediction shape %s"
            % (tuple(target.shape), tuple(prediction.shape))
        )
    valid = torch.isfinite(target) & target.ne(-100.0)
    if non_negative:
        valid = valid & target.ge(0.0)
    if mask is not None:
        external_mask = mask.to(device=prediction.device, dtype=torch.bool)
        while external_mask.ndim < valid.ndim:
            external_mask = external_mask.unsqueeze(-1)
        try:
            external_mask = external_mask.expand_as(valid)
        except RuntimeError as exc:
            raise ValueError("regression mask cannot broadcast to target") from exc
        valid = valid & external_mask
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    raw = F.smooth_l1_loss(prediction, safe_target, reduction="none")
    return _masked_mean(raw, valid)


def _classification_loss(
    logits: Tensor, target: Tensor, mask: Optional[Tensor]
) -> Tensor:
    """Masked categorical loss supporting integer or soft targets."""

    if target.shape == logits.shape:
        loss = -(target.to(logits) * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        return _masked_mean(loss, mask)

    if target.ndim == logits.ndim and target.shape[-1] == 1:
        target = target.squeeze(-1)
    target = target.to(device=logits.device, dtype=torch.long)
    if target.shape != logits.shape[:-1]:
        raise ValueError(
            "classification target shape %s does not match logits shape %s"
            % (tuple(target.shape), tuple(logits.shape))
        )
    valid = target.ne(-100) & target.ge(0) & target.lt(logits.shape[-1])
    safe_target = target.masked_fill(~valid, 0)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        safe_target.reshape(-1),
        reduction="none",
    ).reshape_as(safe_target)
    valid_mask = valid if mask is None else valid & mask.to(valid.device, torch.bool)
    return _masked_mean(loss, valid_mask)


class NeuralLedgerLoss(nn.Module):
    """Masked multi-task objective for :class:`NeuralLedgerHead`.

    Every task is optional.  Missing targets and entirely masked targets return
    a differentiable zero, making mixed datasets safe.  Canonical target names
    are the names in ``LEDGER_LOSS_NAMES``; common ``*_labels``/``*_target``
    aliases are accepted.  Masks can live in ``targets`` or in the separate
    ``masks`` mapping under ``<task>_mask`` (or simply ``<task>``).
    """

    _TARGET_ALIASES: Dict[str, Tuple[str, ...]] = {
        "claim": (
            "claim",
            "claim_labels",
            "claim_target",
            "ledger_claim_labels",
        ),
        "presence": (
            "presence",
            "presence_labels",
            "presence_target",
            "claim_presence",
            "claim_presence_labels",
            "ledger_claim_mask",
        ),
        "claim_type": (
            "claim_type",
            "claim_type_labels",
            "claim_type_target",
            "type",
            "type_labels",
            "claim_type_id",
            "ledger_claim_type_labels",
        ),
        "subject": (
            "subject",
            "subject_labels",
            "subject_target",
            "subject_id",
            "subject_ids",
            "ledger_subject_labels",
        ),
        "object": (
            "object",
            "object_labels",
            "object_target",
            "object_id",
            "object_ids",
            "ledger_object_labels",
        ),
        "precondition": (
            "precondition",
            "precondition_labels",
            "precondition_target",
            "precondition_id",
            "ledger_precondition_labels",
        ),
        "effect": (
            "effect",
            "effect_labels",
            "effect_target",
            "effect_id",
            "ledger_effect_labels",
        ),
        "evidence": (
            "evidence",
            "evidence_labels",
            "evidence_target",
            "evidence_score",
            "ledger_evidence_labels",
        ),
        "uncertainty": (
            "uncertainty",
            "uncertainty_labels",
            "uncertainty_target",
            "uncertainty_score",
            "ledger_uncertainty_labels",
        ),
        "dependency": (
            "dependency",
            "dependency_labels",
            "dependency_target",
            "dependency_score",
            "downstream_dependency",
            "ledger_dependency_labels",
        ),
        "dependency_matrix": (
            "dependency_matrix",
            "dependency_matrix_labels",
            "dependency_matrix_target",
            "ledger_dependency_matrix",
        ),
        "debt": (
            "debt",
            "debt_labels",
            "debt_target",
            "causal_debt",
            "ledger_debt_labels",
        ),
        "repair_cost": (
            "repair_cost",
            "repair_cost_labels",
            "repair_cost_target",
            "recovery_cost",
            "ledger_repair_cost_labels",
        ),
        "observability": (
            "observability",
            "observability_labels",
            "observability_target",
            "observable",
            "observability_score",
            "ledger_observability_labels",
        ),
        "importance": (
            "importance",
            "importance_labels",
            "importance_target",
            "task_importance",
            "ledger_importance_labels",
        ),
        "relation": (
            "relation",
            "relation_labels",
            "relation_target",
            "relation_id",
            "ledger_relation_labels",
        ),
        "rollback": (
            "rollback",
            "rollback_labels",
            "rollback_target",
            "rollback_stage",
            "rollback_index",
            "ledger_rollback_labels",
        ),
        "cf": ("cf", "cf_labels", "cf_target"),
        "repair": (
            "repair",
            "repair_labels",
            "repair_target",
            "repair_action",
            "repair_action_labels",
            "ledger_repair_action_labels",
        ),
        "action_cost": (
            "action_cost",
            "action_cost_labels",
            "action_cost_target",
            "repair_action_cost",
        ),
        "repair_world": (
            "repair_post_debt",
            "post_repair_debt",
            "repair_world",
            "repair_world_labels",
            "repair_world_target",
            "repair_world_delta",
        ),
        "debt_reward": (
            "debt_reward",
            "debt_reward_labels",
            "debt_reward_target",
        ),
    }

    def __init__(
        self,
        weights: Optional[Mapping[str, float]] = None,
        cf_margin: float = 1.0,
        repair_action_cost_weight: float = 0.0,
        repair_task_risk_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if cf_margin < 0:
            raise ValueError("cf_margin must be non-negative")
        if repair_action_cost_weight < 0:
            raise ValueError("repair_action_cost_weight must be non-negative")
        if repair_task_risk_weight < 0:
            raise ValueError("repair_task_risk_weight must be non-negative")
        unknown = set() if weights is None else set(weights) - set(LEDGER_LOSS_NAMES)
        if unknown:
            raise ValueError("unknown ledger loss weights: %s" % sorted(unknown))
        self.weights = {
            name: 1.0 if weights is None else float(weights.get(name, 1.0))
            for name in LEDGER_LOSS_NAMES
        }
        self.cf_margin = cf_margin
        self.repair_action_cost_weight = float(repair_action_cost_weight)
        self.repair_task_risk_weight = float(repair_task_risk_weight)

    @staticmethod
    def _mask_for(
        task: str,
        targets: Mapping[str, Tensor],
        masks: Optional[Mapping[str, Tensor]],
    ) -> Optional[Tensor]:
        if masks is not None:
            value = _first_present(
                masks, (task, task + "_mask", "ledger_" + task + "_mask")
            )
            if value is not None:
                return value
        names: Tuple[str, ...] = (task + "_mask", "ledger_" + task + "_mask")
        if task == "repair":
            names = names + (
                "repair_action_mask",
                "ledger_repair_action_mask",
            )
        return _first_present(targets, names)

    @staticmethod
    def _anchor(outputs: Mapping[str, Tensor]) -> Tensor:
        for value in outputs.values():
            if torch.is_tensor(value) and value.is_floating_point():
                return value.sum() * 0.0
        raise ValueError("outputs must contain at least one floating-point tensor")

    def forward(
        self,
        outputs: Mapping[str, Tensor],
        targets: Optional[Mapping[str, Tensor]] = None,
        masks: Optional[Mapping[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        targets = {} if targets is None else targets
        zero = self._anchor(outputs)
        losses = {name: zero for name in LEDGER_LOSS_NAMES}

        target = _first_present(targets, self._TARGET_ALIASES["claim"])
        if target is not None:
            losses["claim"] = _probability_loss(
                outputs["claim_logits"],
                target,
                self._mask_for("claim", targets, masks),
                with_logits=True,
            )

        target = _first_present(targets, self._TARGET_ALIASES["presence"])
        if target is not None:
            losses["presence"] = _probability_loss(
                outputs["presence_logits"],
                target,
                self._mask_for("presence", targets, masks),
                with_logits=True,
            )

        for task in (
            "claim_type",
            "subject",
            "object",
            "precondition",
            "effect",
        ):
            target = _first_present(targets, self._TARGET_ALIASES[task])
            if target is not None:
                losses[task] = _classification_loss(
                    outputs[task + "_logits"],
                    target,
                    self._mask_for(task, targets, masks),
                )

        target = _first_present(targets, self._TARGET_ALIASES["evidence"])
        if target is not None:
            losses["evidence"] = _probability_loss(
                outputs["evidence_logits"],
                target,
                self._mask_for("evidence", targets, masks),
                with_logits=True,
            )

        target = _first_present(targets, self._TARGET_ALIASES["uncertainty"])
        if target is not None:
            losses["uncertainty"] = _probability_loss(
                outputs["uncertainty_logits"],
                target,
                self._mask_for("uncertainty", targets, masks),
                with_logits=True,
            )

        target = _first_present(targets, self._TARGET_ALIASES["dependency"])
        if target is not None:
            losses["dependency"] = _probability_loss(
                outputs["dependency"],
                target,
                self._mask_for("dependency", targets, masks),
                with_logits=False,
            )

        target = _first_present(targets, self._TARGET_ALIASES["dependency_matrix"])
        if target is not None:
            losses["dependency_matrix"] = _probability_loss(
                outputs["dependency_matrix_logits"],
                target,
                self._mask_for("dependency_matrix", targets, masks),
                with_logits=True,
            )

        target = _first_present(targets, self._TARGET_ALIASES["debt"])
        if target is not None:
            losses["debt"] = _probability_loss(
                outputs["debt"],
                target,
                self._mask_for("debt", targets, masks),
                with_logits=False,
                regression=True,
            )

        target = _first_present(targets, self._TARGET_ALIASES["repair_cost"])
        if target is not None:
            losses["repair_cost"] = _regression_loss(
                outputs["repair_cost"],
                target,
                self._mask_for("repair_cost", targets, masks),
                non_negative=True,
            )

        target = _first_present(targets, self._TARGET_ALIASES["observability"])
        if target is not None:
            losses["observability"] = _probability_loss(
                outputs["observability_logits"],
                target,
                self._mask_for("observability", targets, masks),
                with_logits=True,
            )

        target = _first_present(targets, self._TARGET_ALIASES["importance"])
        if target is not None:
            losses["importance"] = _probability_loss(
                outputs["importance"],
                target,
                self._mask_for("importance", targets, masks),
                with_logits=False,
                regression=True,
            )

        target = _first_present(targets, self._TARGET_ALIASES["relation"])
        if target is not None:
            losses["relation"] = _classification_loss(
                outputs["relation_logits"],
                target,
                self._mask_for("relation", targets, masks),
            )

        target = _first_present(targets, self._TARGET_ALIASES["rollback"])
        if target is not None:
            losses["rollback"] = _classification_loss(
                outputs["rollback_logits"],
                target,
                self._mask_for("rollback", targets, masks),
            )

        target = _first_present(targets, self._TARGET_ALIASES["cf"])
        factual_target = _first_present(targets, ("factual_delta", "factual_target"))
        counterfactual_target = _first_present(
            targets, ("counterfactual_delta", "counterfactual_target")
        )
        cf_mask = self._mask_for("cf", targets, masks)
        delta_difference = outputs["factual_delta"] - outputs["counterfactual_delta"]
        cf_loss_terms: List[Tensor] = []
        if target is not None:
            if target.shape == delta_difference.shape:
                raw = F.smooth_l1_loss(
                    delta_difference, target.to(delta_difference), reduction="none"
                )
            else:
                # Normalized L1 keeps the margin independent of delta width;
                # a raw sum over the default 512 dimensions would already
                # exceed margin=1 at initialization and yield zero gradients.
                distance = delta_difference.abs().mean(dim=-1)
                target = target.to(distance)
                if target.ndim == distance.ndim + 1:
                    target = target.squeeze(-1)
                if target.shape != distance.shape:
                    raise ValueError("cf target must match delta or claim-slot shape")
                # target=1 means the actions should have different effects;
                # target=0 means their effects should agree.
                raw = (
                    target * F.relu(self.cf_margin - distance)
                    + (1.0 - target) * distance
                )
            cf_loss_terms.append(_masked_mean(raw, cf_mask))
        elif factual_target is not None or counterfactual_target is not None:
            terms: List[Tensor] = []
            if factual_target is not None:
                terms.append(
                    F.smooth_l1_loss(
                        outputs["factual_delta"],
                        factual_target.to(outputs["factual_delta"]),
                        reduction="none",
                    )
                )
            if counterfactual_target is not None:
                terms.append(
                    F.smooth_l1_loss(
                        outputs["counterfactual_delta"],
                        counterfactual_target.to(outputs["counterfactual_delta"]),
                        reduction="none",
                    )
                )
            cf_loss_terms.extend(_masked_mean(term, cf_mask) for term in terms)

        global_cf_target = _first_present(
            targets, ("cf_global", "cf_global_target", "global_cf_target")
        )
        if global_cf_target is not None:
            global_slot_mask = None
            if masks is not None:
                global_slot_mask = _first_present(
                    masks,
                    (
                        "cf_global_slot",
                        "cf_global_slot_mask",
                        "ledger_cf_global_slot_mask",
                    ),
                )
            if global_slot_mask is None:
                global_slot_mask = _first_present(
                    targets,
                    ("cf_global_slot_mask", "ledger_cf_global_slot_mask"),
                )
            slot_distance = delta_difference.abs().mean(dim=-1)
            if global_slot_mask is not None:
                if global_slot_mask.shape != slot_distance.shape:
                    raise ValueError("cf_global_slot_mask must have shape [B, S]")
                slot_weight = global_slot_mask.to(slot_distance)
                global_distance = (slot_distance * slot_weight).sum(dim=-1) / (
                    slot_weight.sum(dim=-1).clamp_min(1.0)
                )
            else:
                global_distance = slot_distance.mean(dim=-1)
            global_cf_target = global_cf_target.to(global_distance)
            if global_cf_target.ndim == 2 and global_cf_target.shape[-1] == 1:
                global_cf_target = global_cf_target.squeeze(-1)
            if global_cf_target.shape != global_distance.shape:
                raise ValueError("cf_global target must have shape [B]")
            global_raw = (
                global_cf_target * F.relu(self.cf_margin - global_distance)
                + (1.0 - global_cf_target) * global_distance
            )
            cf_loss_terms.append(
                _masked_mean(
                    global_raw,
                    self._mask_for("cf_global", targets, masks),
                )
            )
        if cf_loss_terms:
            losses["cf"] = torch.stack(cf_loss_terms).mean()

        repair_target = _first_present(targets, self._TARGET_ALIASES["repair"])
        if repair_target is not None:
            repair_mask = self._mask_for("repair", targets, masks)
            policy_loss = _classification_loss(
                outputs["repair_logits"], repair_target, repair_mask
            )
            # repair_scores is now the calibrated signed debt reduction only;
            # cost, task risk, and policy prior remain explicit runtime terms.
            # Ranking it here lets repair-only annotations train the same
            # primary utility signal used online.
            utility_loss = _classification_loss(
                outputs["repair_scores"], repair_target, repair_mask
            )
            losses["repair"] = 0.5 * (policy_loss + utility_loss)

        target = _first_present(targets, self._TARGET_ALIASES["action_cost"])
        if target is not None:
            losses["action_cost"] = _probability_loss(
                outputs["repair_action_cost"],
                target,
                self._mask_for("action_cost", targets, masks),
                with_logits=False,
                regression=True,
            )

        target = _first_present(targets, self._TARGET_ALIASES["repair_world"])
        if target is not None:
            repair_world_mask = self._mask_for("repair_world", targets, masks)
            selected_repair_valid = None
            # Post-debt supervision is the canonical sidecar meaning.  Check it
            # before latent world-delta shape so D == num_claim_slots cannot
            # silently route debt labels into the wrong head.
            if target.shape == outputs["repair_post_debt_per_claim"].shape:
                prediction = outputs["repair_post_debt_per_claim"]
            elif target.shape == outputs["repair_post_debt"].shape:
                prediction = outputs["repair_post_debt"]
            elif target.shape == outputs["repair_world_delta"].shape:
                prediction = outputs["repair_world_delta"]
            elif (
                repair_target is not None
                and target.ndim == outputs["repair_world_delta"].ndim - 1
                and target.shape[0] == outputs["repair_world_delta"].shape[0]
                and target.shape[-1] == outputs["repair_world_delta"].shape[-1]
            ):
                raw_indices = repair_target.to(
                    device=outputs["repair_world_delta"].device
                )
                if raw_indices.ndim != 1:
                    raise ValueError("selected repair labels must have shape [B]")
                selected_repair_valid = (
                    torch.isfinite(raw_indices.float())
                    & raw_indices.ge(0)
                    & raw_indices.lt(outputs["repair_world_delta"].shape[1])
                )
                indices = torch.where(
                    selected_repair_valid,
                    raw_indices,
                    torch.zeros_like(raw_indices),
                ).long()
                prediction = outputs["repair_world_delta"][
                    torch.arange(indices.shape[0], device=indices.device), indices
                ]
            else:
                raise ValueError("repair_world target has an unsupported shape")
            raw = F.smooth_l1_loss(prediction, target.to(prediction), reduction="none")
            if selected_repair_valid is not None:
                repair_world_mask = _combine_loss_masks(
                    repair_world_mask, selected_repair_valid, raw
                )
            losses["repair_world"] = _masked_mean(raw, repair_world_mask)

        target = _first_present(targets, self._TARGET_ALIASES["debt_reward"])
        if target is not None:
            prediction = outputs["repair_debt_reduction"]
            if self.repair_action_cost_weight:
                prediction = prediction - (
                    self.repair_action_cost_weight * outputs["repair_action_cost"]
                )
            if self.repair_task_risk_weight:
                task_risk = _first_present(
                    targets,
                    (
                        "repair_task_risk",
                        "repair_task_risk_labels",
                        "task_risk",
                        "repair_risk",
                    ),
                )
                if task_risk is not None:
                    task_risk = task_risk.to(prediction)
                    if task_risk.shape != prediction.shape:
                        raise ValueError("repair_task_risk must have shape [B,R]")
                    prediction = prediction - self.repair_task_risk_weight * task_risk
            if target.shape != prediction.shape:
                if repair_target is None or target.shape != prediction.shape[:1]:
                    raise ValueError(
                        "debt_reward must be [B,R], or [B] with repair labels"
                    )
                raw_indices = repair_target.to(device=prediction.device)
                if raw_indices.ndim != 1:
                    raise ValueError("selected repair labels must have shape [B]")
                selected_repair_valid = (
                    torch.isfinite(raw_indices.float())
                    & raw_indices.ge(0)
                    & raw_indices.lt(prediction.shape[1])
                )
                indices = torch.where(
                    selected_repair_valid,
                    raw_indices,
                    torch.zeros_like(raw_indices),
                ).long()
                prediction = prediction[
                    torch.arange(indices.shape[0], device=indices.device), indices
                ]
            raw = F.smooth_l1_loss(prediction, target.to(prediction), reduction="none")
            debt_reward_mask = self._mask_for("debt_reward", targets, masks)
            if target.shape != outputs["repair_debt_reduction"].shape:
                assert selected_repair_valid is not None
                debt_reward_mask = _combine_loss_masks(
                    debt_reward_mask, selected_repair_valid, raw
                )
            losses["debt_reward"] = _masked_mean(raw, debt_reward_mask)

        recurrent_loss = outputs.get("recurrent_consistency_loss")
        if recurrent_loss is not None:
            if not torch.is_tensor(recurrent_loss):
                raise ValueError("recurrent_consistency_loss must be a tensor")
            losses["recurrent"] = recurrent_loss.mean()

        total = zero
        for name in LEDGER_LOSS_NAMES:
            total = total + self.weights[name] * losses[name]
        losses["total"] = total
        return losses


def build_repair_debt_reward_targets(
    current_debt: Tensor,
    current_debt_mask: Tensor,
    post_repair_debt: Tensor,
    post_repair_mask: Tensor,
    repair_labels: Tensor,
    repair_label_mask: Tensor,
    num_repair_actions: int,
    claim_mask: Optional[Tensor] = None,
    importance: Optional[Tensor] = None,
    importance_mask: Optional[Tensor] = None,
    action_cost: Optional[Tensor] = None,
    task_risk: Optional[Tensor] = None,
    cost_weight: float = 0.0,
    risk_weight: float = 0.0,
    ignore_index: float = -100.0,
) -> Tuple[Tensor, Tensor]:
    """Build signed global-risk-reduction targets for repair actions.

    Untargeted claims retain their current debt.  Targeted claims use the
    annotated post-repair debt, and both states are aggregated with the same
    importance-normalized definition used by the neural repair head.  Optional
    action-cost and task-risk terms make the target equal the paper's repair
    reward ``D(L_t) - D(L_{t+1}) - beta C(a) - gamma R(a)``.
    """

    reference_shape = current_debt.shape
    tensors = (
        current_debt_mask,
        post_repair_debt,
        post_repair_mask,
        repair_labels,
        repair_label_mask,
    )
    if current_debt.ndim != 2 or any(
        value.shape != reference_shape for value in tensors
    ):
        raise ValueError("repair target inputs must all have shape [B, S]")
    if num_repair_actions <= 0:
        raise ValueError("num_repair_actions must be positive")
    if cost_weight < 0.0 or risk_weight < 0.0:
        raise ValueError("cost_weight and risk_weight must be non-negative")
    if claim_mask is None:
        claim_mask = torch.ones_like(current_debt_mask, dtype=torch.bool)
    elif claim_mask.shape != reference_shape:
        raise ValueError("claim_mask must have shape [B, S]")
    if (importance is None) != (importance_mask is None):
        raise ValueError("importance and importance_mask must be supplied together")
    if importance is not None:
        if importance_mask is None:
            raise ValueError("importance_mask is required with importance")
        if (
            importance.shape != reference_shape
            or importance_mask.shape != reference_shape
        ):
            raise ValueError("importance inputs must have shape [B, S]")
    repair_shape = (current_debt.shape[0], num_repair_actions)
    if action_cost is not None and action_cost.shape != repair_shape:
        raise ValueError("action_cost must have shape [B, R]")
    if task_risk is not None and task_risk.shape != repair_shape:
        raise ValueError("task_risk must have shape [B, R]")

    active = current_debt_mask.bool() & claim_mask.bool()
    if importance is None:
        weights = torch.ones_like(current_debt)
    else:
        resolved_importance_mask = cast(Tensor, importance_mask)
        weights = torch.where(
            resolved_importance_mask.bool(),
            importance.to(current_debt).clamp_min(0.0),
            torch.ones_like(current_debt),
        )
    weights = weights * active.to(current_debt)
    safe_current = torch.where(active, current_debt, torch.zeros_like(current_debt))
    affected = (
        post_repair_mask.bool()
        & repair_label_mask.bool()
        & active
        & repair_labels.ge(0)
        & repair_labels.lt(num_repair_actions)
    )

    batch_size = current_debt.shape[0]
    targets = torch.full(
        (batch_size, num_repair_actions),
        float(ignore_index),
        dtype=current_debt.dtype,
        device=current_debt.device,
    )
    target_mask = torch.zeros_like(targets, dtype=torch.bool)
    for batch_index in range(batch_size):
        denominator = weights[batch_index].sum()
        if not bool(denominator.gt(0)):
            continue
        current_risk = (
            safe_current[batch_index] * weights[batch_index]
        ).sum() / denominator
        for repair_index in range(num_repair_actions):
            selected = affected[batch_index] & repair_labels[batch_index].eq(
                repair_index
            )
            if not bool(selected.any()):
                continue
            repaired = safe_current[batch_index].clone()
            repaired[selected] = post_repair_debt[batch_index, selected]
            post_risk = (repaired * weights[batch_index]).sum() / denominator
            targets[batch_index, repair_index] = current_risk - post_risk
            target_mask[batch_index, repair_index] = True
    if action_cost is not None and cost_weight:
        targets = torch.where(
            target_mask,
            targets - float(cost_weight) * action_cost.to(targets),
            targets,
        )
    if task_risk is not None and risk_weight:
        targets = torch.where(
            target_mask,
            targets - float(risk_weight) * task_risk.to(targets),
            targets,
        )
    return targets, target_mask


def compute_ledger_losses(
    outputs: Mapping[str, Tensor],
    targets: Optional[Mapping[str, Tensor]] = None,
    masks: Optional[Mapping[str, Tensor]] = None,
    weights: Optional[Mapping[str, float]] = None,
    cf_margin: float = 1.0,
    repair_action_cost_weight: float = 0.0,
    repair_task_risk_weight: float = 0.0,
) -> Dict[str, Tensor]:
    """Functional convenience wrapper around :class:`NeuralLedgerLoss`."""

    return cast(
        Dict[str, Tensor],
        NeuralLedgerLoss(
            weights=weights,
            cf_margin=cf_margin,
            repair_action_cost_weight=repair_action_cost_weight,
            repair_task_risk_weight=repair_task_risk_weight,
        )(
            outputs, targets=targets, masks=masks
        ),
    )


# Small aliases keep downstream integration readable while retaining one
# implementation and state-dict layout.
NeuralCausalLedger = NeuralLedgerHead
LedgerWAMHead = NeuralLedgerHead
LedgerMultiTaskLoss = NeuralLedgerLoss
LedgerLoss = NeuralLedgerLoss


__all__ = [
    "LEDGER_LOSS_NAMES",
    "NeuralLedgerHead",
    "NeuralCausalLedger",
    "LedgerWAMHead",
    "NeuralLedgerLoss",
    "LedgerMultiTaskLoss",
    "LedgerLoss",
    "build_repair_debt_reward_targets",
    "compute_ledger_losses",
]
