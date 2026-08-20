from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .object_slots import DynamicObjectSlotEncoder


DEFAULT_CLAIM_NAMES = (
    "contact",
    "grasped",
    "supported",
    "contained",
    "visible",
    "persistent",
    "precondition_met",
    "effect_achieved",
)

DEFAULT_RELATION_NAMES = (
    "contact",
    "support",
    "containment",
    "occlusion",
    "relative_pose",
    "co_motion",
    "precondition",
    "effect",
)

DEFAULT_ENTITY_NAMES = (
    "robot",
    "end_effector",
    "target_object",
    "container",
    "tool",
    "support_surface",
    "occluder",
    "unknown",
)


@dataclass
class CausalLedgerState:
    """Recurrent, tensor-only state carried between closed-loop replans."""

    hidden: torch.Tensor
    confidence: torch.Tensor
    debt: torch.Tensor
    object_slots: Optional[torch.Tensor] = None
    step: int = 0

    def detached(self) -> "CausalLedgerState":
        return CausalLedgerState(
            hidden=self.hidden.detach(),
            confidence=self.confidence.detach(),
            debt=self.debt.detach(),
            object_slots=None if self.object_slots is None else self.object_slots.detach(),
            step=int(self.step),
        )

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "CausalLedgerState":
        return CausalLedgerState(
            hidden=self.hidden.to(device=device, dtype=dtype),
            confidence=self.confidence.to(device=device, dtype=dtype),
            debt=self.debt.to(device=device, dtype=dtype),
            object_slots=(
                None
                if self.object_slots is None
                else self.object_slots.to(device=device, dtype=dtype)
            ),
            step=int(self.step),
        )


@dataclass
class CausalLedgerOutput:
    hidden: torch.Tensor
    confidence_logits: torch.Tensor
    confidence: torch.Tensor
    evidence: torch.Tensor
    uncertainty: torch.Tensor
    dependency: torch.Tensor
    dependency_by_step: torch.Tensor
    repair_cost: torch.Tensor
    observability: torch.Tensor
    importance: torch.Tensor
    debt: torch.Tensor
    global_debt: torch.Tensor
    global_risk: torch.Tensor
    rollback_logits: torch.Tensor
    relation_logits: torch.Tensor
    entity_logits: torch.Tensor
    object_entity_logits: torch.Tensor
    precondition_logits: torch.Tensor
    effect_logits: torch.Tensor
    precondition_embedding: torch.Tensor
    effect_embedding: torch.Tensor
    repair_action: torch.Tensor
    expected_debt_reduction: torch.Tensor
    repair_risk: torch.Tensor
    evidence_gate: torch.Tensor
    object_slots: torch.Tensor
    object_presence: torch.Tensor
    object_assignment: torch.Tensor
    next_state: CausalLedgerState

    def summary(
        self,
        claim_names: Sequence[str],
        relation_names: Sequence[str],
        entity_names: Sequence[str],
        batch_index: int = 0,
    ) -> dict[str, Any]:
        """Convert one batch element to a JSON-safe, inspectable ledger."""

        relation_idx = self.relation_logits[batch_index].argmax(dim=-1).detach().cpu().tolist()
        rollback_idx = self.rollback_logits[batch_index].argmax(dim=-1).detach().cpu().tolist()
        precondition_idx = (
            self.precondition_logits[batch_index].argmax(dim=-1).detach().cpu().tolist()
        )
        effect_idx = self.effect_logits[batch_index].argmax(dim=-1).detach().cpu().tolist()
        entity_prob = torch.sigmoid(self.entity_logits[batch_index]).detach().float().cpu()
        object_entity_prob = torch.sigmoid(
            self.object_entity_logits[batch_index]
        ).detach().float().cpu()
        values = {
            "confidence": self.confidence[batch_index],
            "evidence": self.evidence[batch_index],
            "uncertainty": self.uncertainty[batch_index],
            "dependency": self.dependency[batch_index],
            "repair_cost": self.repair_cost[batch_index],
            "observability": self.observability[batch_index],
            "importance": self.importance[batch_index],
            "debt": self.debt[batch_index],
        }
        cpu_values = {
            key: tensor.detach().float().cpu().tolist() for key, tensor in values.items()
        }
        claims = []
        for index, claim_name in enumerate(claim_names):
            relation = relation_names[int(relation_idx[index])]
            selected_entities = [
                entity_names[entity_idx]
                for entity_idx, probability in enumerate(entity_prob[index].tolist())
                if probability >= 0.5
            ]
            if not selected_entities:
                selected_entities = [entity_names[int(entity_prob[index].argmax().item())]]
            claims.append(
                {
                    "id": int(index),
                    "name": str(claim_name),
                    "entities": selected_entities,
                    "object_slot_ids": [
                        int(slot_index)
                        for slot_index, probability in enumerate(
                            object_entity_prob[index].tolist()
                        )
                        if probability >= 0.5
                    ],
                    "relation": str(relation),
                    "precondition": str(claim_names[int(precondition_idx[index])])
                    if int(precondition_idx[index]) < len(claim_names)
                    else "none",
                    "effect": str(claim_names[int(effect_idx[index])])
                    if int(effect_idx[index]) < len(claim_names)
                    else "none",
                    "confidence": float(cpu_values["confidence"][index]),
                    "evidence": float(cpu_values["evidence"][index]),
                    "uncertainty": float(cpu_values["uncertainty"][index]),
                    "dependency": float(cpu_values["dependency"][index]),
                    "repair_cost": float(cpu_values["repair_cost"][index]),
                    "observability": float(cpu_values["observability"][index]),
                    "importance": float(cpu_values["importance"][index]),
                    "debt": float(cpu_values["debt"][index]),
                    "rollback_step": int(rollback_idx[index]),
                }
            )
        return {
            "step": int(self.next_state.step),
            "global_debt": float(self.global_debt[batch_index].detach().float().cpu()),
            "global_risk": float(self.global_risk[batch_index].detach().float().cpu()),
            "object_slots": [
                {
                    "id": int(index),
                    "presence": float(value),
                }
                for index, value in enumerate(
                    self.object_presence[batch_index].detach().float().cpu().tolist()
                )
            ],
            "claims": claims,
        }


class CausalBeliefLedger(nn.Module):
    """Turns Fast-WAM action features into explicit action-conditioned claims.

    The module intentionally operates on the already-computed action expert tokens. This
    preserves Fast-WAM's inference advantage: no future video rollout is introduced.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int = 512,
        claim_names: Sequence[str] = DEFAULT_CLAIM_NAMES,
        relation_names: Sequence[str] = DEFAULT_RELATION_NAMES,
        entity_names: Sequence[str] = DEFAULT_ENTITY_NAMES,
        max_rollback_steps: int = 32,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        effect_dim: int = 128,
        num_repair_actions: int = 4,
        visual_input_dim: Optional[int] = None,
        num_object_slots: int = 12,
        object_slot_iterations: int = 3,
        evidence_dim: int = 16,
        dependency_horizon: int = 8,
        use_object_slots: bool = True,
        use_sensor_evidence: bool = True,
        debt_initial_weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        if not claim_names:
            raise ValueError("`claim_names` must contain at least one claim.")
        if not relation_names:
            raise ValueError("`relation_names` must contain at least one relation.")
        if hidden_dim % num_attention_heads != 0:
            raise ValueError("`hidden_dim` must be divisible by `num_attention_heads`.")
        if len(debt_initial_weights) != 5:
            raise ValueError("`debt_initial_weights` must contain five values.")

        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.hidden_dim = int(hidden_dim)
        self.claim_names = tuple(str(name) for name in claim_names)
        self.relation_names = tuple(str(name) for name in relation_names)
        self.entity_names = tuple(str(name) for name in entity_names)
        self.num_claims = len(self.claim_names)
        self.num_relations = len(self.relation_names)
        self.num_entities = len(self.entity_names)
        self.max_rollback_steps = int(max_rollback_steps)
        self.num_repair_actions = int(num_repair_actions)
        self.visual_input_dim = int(visual_input_dim or input_dim)
        self.num_object_slots = int(num_object_slots)
        self.evidence_dim = int(evidence_dim)
        self.dependency_horizon = int(dependency_horizon)
        self.use_object_slots = bool(use_object_slots)
        self.use_sensor_evidence = bool(use_sensor_evidence)

        self.input_projection = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.action_projection = nn.Sequential(
            nn.Linear(self.action_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.object_slots = DynamicObjectSlotEncoder(
            input_dim=self.visual_input_dim,
            hidden_dim=self.hidden_dim,
            num_slots=self.num_object_slots,
            iterations=int(object_slot_iterations),
        )
        self.missing_visual_token = nn.Parameter(
            torch.zeros(1, 1, self.visual_input_dim)
        )
        self.evidence_projection = nn.Sequential(
            nn.LayerNorm(self.evidence_dim),
            nn.Linear(self.evidence_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.claim_queries = nn.Parameter(
            torch.randn(1, self.num_claims, self.hidden_dim) / self.hidden_dim**0.5
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=int(num_attention_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.claim_norm = nn.LayerNorm(self.hidden_dim)
        self.memory_update = nn.GRUCell(self.hidden_dim, self.hidden_dim)

        self.confidence_head = nn.Linear(self.hidden_dim, 1)
        self.evidence_head = nn.Linear(self.hidden_dim, 1)
        self.uncertainty_head = nn.Linear(self.hidden_dim, 1)
        self.dependency_claim_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dependency_action_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.repair_cost_head = nn.Linear(self.hidden_dim, 1)
        self.observability_head = nn.Linear(self.hidden_dim, 1)
        self.importance_head = nn.Linear(self.hidden_dim, 1)
        self.evidence_gate_head = nn.Linear(self.hidden_dim, 1)
        self.rollback_head = nn.Linear(self.hidden_dim, self.max_rollback_steps)
        self.relation_head = nn.Linear(self.hidden_dim, self.num_relations)
        self.entity_head = nn.Linear(self.hidden_dim, self.num_entities)
        self.object_entity_claim_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.object_entity_slot_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.precondition_classifier = nn.Linear(self.hidden_dim, self.num_claims + 1)
        self.effect_classifier = nn.Linear(self.hidden_dim, self.num_claims + 1)
        self.precondition_head = nn.Linear(self.hidden_dim, int(effect_dim))
        self.effect_head = nn.Linear(self.hidden_dim, int(effect_dim))

        planner_input_dim = self.hidden_dim * 2
        self.repair_action_head = nn.Sequential(
            nn.LayerNorm(planner_input_dim),
            nn.Linear(planner_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(
                self.hidden_dim,
                self.num_repair_actions * self.action_horizon * self.action_dim,
            ),
            nn.Tanh(),
        )
        self.repair_value_head = nn.Sequential(
            nn.LayerNorm(planner_input_dim),
            nn.Linear(planner_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.num_repair_actions * 2),
        )

        initial = torch.as_tensor(debt_initial_weights, dtype=torch.float32).clamp(min=1e-4)
        # softplus(raw) starts at the configured positive coefficients.
        self.debt_weight_raw = nn.Parameter(torch.log(torch.expm1(initial)))
        # Neutral, untrained heads start at debt 0.5 instead of forcing repairs.
        self.debt_bias = nn.Parameter(-0.5 * initial.sum())

    def _validate_previous_state(
        self,
        previous_state: Optional[CausalLedgerState],
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[CausalLedgerState]:
        if previous_state is None:
            return None
        expected_hidden = (batch_size, self.num_claims, self.hidden_dim)
        expected_scalar = (batch_size, self.num_claims)
        if tuple(previous_state.hidden.shape) != expected_hidden:
            raise ValueError(
                f"Previous ledger hidden shape must be {expected_hidden}, "
                f"got {tuple(previous_state.hidden.shape)}."
            )
        if tuple(previous_state.confidence.shape) != expected_scalar:
            raise ValueError("Previous ledger confidence shape mismatch.")
        if tuple(previous_state.debt.shape) != expected_scalar:
            raise ValueError("Previous ledger debt shape mismatch.")
        if previous_state.object_slots is not None and tuple(previous_state.object_slots.shape) != (
            batch_size,
            self.num_object_slots,
            self.hidden_dim,
        ):
            raise ValueError("Previous object-slot shape mismatch.")
        return previous_state.to(device=device, dtype=dtype)

    def forward(
        self,
        state_tokens: torch.Tensor,
        action: torch.Tensor,
        previous_state: Optional[CausalLedgerState] = None,
        visual_tokens: Optional[torch.Tensor] = None,
        sensor_evidence: Optional[torch.Tensor] = None,
    ) -> CausalLedgerOutput:
        if state_tokens.ndim != 3:
            raise ValueError("`state_tokens` must have shape [B,T,D].")
        if action.ndim != 3:
            raise ValueError("`action` must have shape [B,T,A].")
        if state_tokens.shape[0] != action.shape[0]:
            raise ValueError("State-token and action batch sizes must match.")
        if state_tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected state token dim {self.input_dim}, got {state_tokens.shape[-1]}."
            )
        if action.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action dim {self.action_dim}, got {action.shape[-1]}.")

        batch_size = state_tokens.shape[0]
        projected_tokens = self.input_projection(state_tokens)
        previous_state = self._validate_previous_state(
            previous_state,
            batch_size=batch_size,
            dtype=projected_tokens.dtype,
            device=projected_tokens.device,
        )
        if not self.use_object_slots:
            visual_tokens = self.missing_visual_token.expand(batch_size, -1, -1)
        elif visual_tokens is None:
            if self.visual_input_dim == self.input_dim:
                visual_tokens = state_tokens
            else:
                visual_tokens = self.missing_visual_token.expand(batch_size, -1, -1)
        visual_tokens = visual_tokens.to(device=state_tokens.device, dtype=state_tokens.dtype)
        slot_output = self.object_slots(
            visual_tokens,
            previous_slots=None if previous_state is None else previous_state.object_slots,
        )

        action_tokens = self.action_projection(action)
        action_context = action_tokens.mean(dim=1)
        if sensor_evidence is None or not self.use_sensor_evidence:
            sensor_evidence = torch.zeros(
                batch_size,
                self.evidence_dim,
                device=state_tokens.device,
                dtype=state_tokens.dtype,
            )
        if sensor_evidence.ndim != 2 or tuple(sensor_evidence.shape) != (
            batch_size,
            self.evidence_dim,
        ):
            raise ValueError(
                "`sensor_evidence` must have shape "
                f"[{batch_size},{self.evidence_dim}], got {tuple(sensor_evidence.shape)}."
            )
        evidence_context = self.evidence_projection(sensor_evidence.to(state_tokens))
        queries = (
            self.claim_queries.expand(batch_size, -1, -1)
            + action_context.unsqueeze(1)
            + evidence_context.unsqueeze(1)
        )
        attended, _ = self.cross_attention(
            queries,
            torch.cat((projected_tokens, slot_output.slots), dim=1),
            torch.cat((projected_tokens, slot_output.slots), dim=1),
        )
        candidate_hidden = self.claim_norm(queries + attended)

        if previous_state is None:
            hidden = candidate_hidden
            previous_confidence = None
            next_step = 0
        else:
            hidden = self.memory_update(
                candidate_hidden.reshape(-1, self.hidden_dim),
                previous_state.hidden.reshape(-1, self.hidden_dim),
            ).view(batch_size, self.num_claims, self.hidden_dim)
            previous_confidence = previous_state.confidence
            next_step = int(previous_state.step) + 1

        raw_confidence_logits = self.confidence_head(hidden).squeeze(-1)
        evidence_gate = torch.sigmoid(self.evidence_gate_head(hidden).squeeze(-1))
        if previous_confidence is None:
            confidence_logits = raw_confidence_logits
        else:
            eps = torch.finfo(raw_confidence_logits.dtype).eps
            previous_logits = torch.logit(previous_confidence.clamp(min=eps, max=1.0 - eps))
            confidence_logits = (
                evidence_gate * raw_confidence_logits + (1.0 - evidence_gate) * previous_logits
            )

        confidence = torch.sigmoid(confidence_logits)
        evidence = torch.sigmoid(
            self.evidence_head(hidden).squeeze(-1)
            + torch.einsum(
                "bnh,bh->bn",
                F.normalize(hidden, dim=-1),
                F.normalize(evidence_context, dim=-1),
            )
        )
        uncertainty = torch.sigmoid(self.uncertainty_head(hidden).squeeze(-1))
        future_actions = F.interpolate(
            action_tokens.transpose(1, 2),
            size=self.dependency_horizon,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        dependency_by_step = torch.sigmoid(
            torch.einsum(
                "bnh,bkh->bnk",
                self.dependency_claim_projection(hidden),
                self.dependency_action_projection(future_actions),
            )
            / self.hidden_dim**0.5
        )
        dependency = dependency_by_step.mean(dim=-1)
        repair_cost = torch.sigmoid(self.repair_cost_head(hidden).squeeze(-1))
        observability = torch.sigmoid(self.observability_head(hidden).squeeze(-1))
        importance = torch.sigmoid(self.importance_head(hidden).squeeze(-1))

        debt_features = torch.stack(
            (1.0 - confidence, uncertainty, dependency, repair_cost, 1.0 - observability),
            dim=-1,
        )
        debt_weights = F.softplus(self.debt_weight_raw).to(
            device=hidden.device, dtype=hidden.dtype
        )
        debt = torch.sigmoid((debt_features * debt_weights).sum(dim=-1) + self.debt_bias)
        global_debt = (importance * debt).sum(dim=-1)
        global_risk = global_debt / importance.sum(dim=-1).clamp(min=1e-6)

        pooled_claims = (importance.unsqueeze(-1) * hidden).sum(dim=1)
        pooled_claims = pooled_claims / importance.sum(dim=1, keepdim=True).clamp(min=1e-6)
        planner_features = torch.cat((pooled_claims, action_context), dim=-1)
        repair_action = self.repair_action_head(planner_features).view(
            batch_size,
            self.num_repair_actions,
            self.action_horizon,
            self.action_dim,
        )
        repair_values = self.repair_value_head(planner_features).view(
            batch_size, self.num_repair_actions, 2
        )
        expected_debt_reduction = torch.sigmoid(repair_values[..., 0])
        repair_risk = torch.sigmoid(repair_values[..., 1])

        next_state = CausalLedgerState(
            hidden=hidden,
            confidence=confidence,
            debt=debt,
            object_slots=slot_output.slots,
            step=next_step,
        )
        object_entity_logits = torch.einsum(
            "bnh,bsh->bns",
            self.object_entity_claim_projection(hidden),
            self.object_entity_slot_projection(slot_output.slots),
        ) / self.hidden_dim**0.5
        return CausalLedgerOutput(
            hidden=hidden,
            confidence_logits=confidence_logits,
            confidence=confidence,
            evidence=evidence,
            uncertainty=uncertainty,
            dependency=dependency,
            dependency_by_step=dependency_by_step,
            repair_cost=repair_cost,
            observability=observability,
            importance=importance,
            debt=debt,
            global_debt=global_debt,
            global_risk=global_risk,
            rollback_logits=self.rollback_head(hidden),
            relation_logits=self.relation_head(hidden),
            entity_logits=self.entity_head(hidden),
            object_entity_logits=object_entity_logits,
            precondition_logits=self.precondition_classifier(hidden),
            effect_logits=self.effect_classifier(hidden),
            precondition_embedding=F.normalize(self.precondition_head(hidden), dim=-1),
            effect_embedding=F.normalize(self.effect_head(hidden), dim=-1),
            repair_action=repair_action,
            expected_debt_reduction=expected_debt_reduction,
            repair_risk=repair_risk,
            evidence_gate=evidence_gate,
            object_slots=slot_output.slots,
            object_presence=slot_output.presence,
            object_assignment=slot_output.assignment,
            next_state=next_state,
        )

    def forward_sequence(
        self,
        state_tokens: torch.Tensor,
        action: torch.Tensor,
        *,
        visual_tokens: Optional[torch.Tensor] = None,
        sensor_evidence: Optional[torch.Tensor] = None,
        previous_state: Optional[CausalLedgerState] = None,
        num_steps: int = 4,
    ) -> list[CausalLedgerOutput]:
        """Unrolls the recurrent ledger over a contiguous training window."""

        if num_steps <= 0:
            raise ValueError("`num_steps` must be positive.")
        token_chunks = torch.tensor_split(state_tokens, num_steps, dim=1)
        action_chunks = torch.tensor_split(action, num_steps, dim=1)
        visual_chunks = (
            [None] * num_steps
            if visual_tokens is None
            else list(torch.tensor_split(visual_tokens, num_steps, dim=1))
        )
        if sensor_evidence is None:
            evidence_steps: list[Optional[torch.Tensor]] = [None] * num_steps
        elif sensor_evidence.ndim == 2:
            evidence_steps = [sensor_evidence] * num_steps
        elif sensor_evidence.ndim == 3 and sensor_evidence.shape[1] == num_steps:
            evidence_steps = [sensor_evidence[:, index] for index in range(num_steps)]
        else:
            raise ValueError(
                "Sequence evidence must be [B,E] or [B,num_steps,E], got "
                f"{tuple(sensor_evidence.shape)}."
            )

        outputs: list[CausalLedgerOutput] = []
        state = previous_state
        for token_chunk, action_chunk, visual_chunk, evidence_step in zip(
            token_chunks, action_chunks, visual_chunks, evidence_steps
        ):
            if token_chunk.shape[1] == 0 or action_chunk.shape[1] == 0:
                continue
            output = self(
                token_chunk,
                action_chunk,
                previous_state=state,
                visual_tokens=visual_chunk,
                sensor_evidence=evidence_step,
            )
            outputs.append(output)
            state = output.next_state
        if not outputs:
            raise ValueError("Temporal ledger unroll produced no non-empty steps.")
        return outputs
