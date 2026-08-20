from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ledger import CausalLedgerOutput
from .world_predictor import CandidateWorldPrediction


@dataclass(frozen=True)
class LedgerLossConfig:
    lambda_claim: float = 1.0
    lambda_debt: float = 1.0
    lambda_dependency: float = 0.25
    lambda_evidence: float = 0.25
    lambda_slot_identity: float = 0.05
    lambda_rollback: float = 0.25
    lambda_relation: float = 0.25
    lambda_structure: float = 0.25
    lambda_counterfactual: float = 0.1
    lambda_repair: float = 0.5
    lambda_repair_value: float = 0.25
    lambda_world: float = 0.5
    counterfactual_margin: float = 0.2


class LedgerLoss(nn.Module):
    """Supervised and counterfactual objectives described by Ledger-WAM."""

    def __init__(self, config: LedgerLossConfig) -> None:
        super().__init__()
        self.config = config

    @staticmethod
    def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=value.device, dtype=value.dtype)
        return (value * mask).sum() / mask.sum().clamp(min=1.0)

    @staticmethod
    def _target(
        sample: Mapping[str, torch.Tensor],
        key: str,
        reference: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[torch.Tensor]:
        value = sample.get(key)
        if value is None:
            return None
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        return value.to(
            device=reference.device,
            dtype=reference.dtype if dtype is None else dtype,
        )

    def forward(
        self,
        output: CausalLedgerOutput,
        sample: Mapping[str, torch.Tensor],
        counterfactual_output: Optional[CausalLedgerOutput] = None,
        world_prediction: Optional[CandidateWorldPrediction] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        zero = output.hidden.sum() * 0.0
        total = zero
        metrics: dict[str, float] = {}

        target_claim = self._target(sample, "ledger_claim_labels", output.confidence)
        claim_mask = self._target(sample, "ledger_claim_mask", output.confidence)
        if target_claim is not None:
            if target_claim.shape != output.confidence.shape:
                raise ValueError("`ledger_claim_labels` shape must match [B,num_claims].")
            if claim_mask is None:
                claim_mask = torch.ones_like(target_claim)
            claim_loss_raw = F.binary_cross_entropy_with_logits(
                output.confidence_logits.float(), target_claim.float(), reduction="none"
            )
            claim_loss = self._masked_mean(claim_loss_raw, claim_mask)
            total = total + self.config.lambda_claim * claim_loss
            metrics["loss_claim"] = self.config.lambda_claim * float(claim_loss.detach())

        target_debt = self._target(sample, "ledger_debt_targets", output.debt)
        debt_mask = self._target(sample, "ledger_debt_mask", output.debt)
        if target_debt is not None:
            if target_debt.shape != output.debt.shape:
                raise ValueError("`ledger_debt_targets` shape must match [B,num_claims].")
            if debt_mask is None:
                debt_mask = claim_mask if claim_mask is not None else torch.ones_like(target_debt)
            debt_loss = self._masked_mean(F.smooth_l1_loss(
                output.debt.float(), target_debt.float(), reduction="none"
            ), debt_mask)
            total = total + self.config.lambda_debt * debt_loss
            metrics["loss_debt"] = self.config.lambda_debt * float(debt_loss.detach())

        dependency_target = self._target(
            sample, "ledger_dependency_targets", output.dependency
        )
        if dependency_target is not None:
            prediction = output.dependency
            if dependency_target.shape == output.dependency_by_step.shape:
                prediction = output.dependency_by_step
            elif dependency_target.shape != output.dependency.shape:
                raise ValueError(
                    "`ledger_dependency_targets` must be [B,N] or [B,N,K]."
                )
            dependency_mask = self._target(
                sample, "ledger_dependency_mask", dependency_target
            )
            if dependency_mask is None:
                dependency_mask = torch.ones_like(dependency_target)
            dependency_loss = self._masked_mean(
                F.binary_cross_entropy(
                    prediction.float(), dependency_target.float(), reduction="none"
                ),
                dependency_mask,
            )
            total = total + self.config.lambda_dependency * dependency_loss
            metrics["loss_dependency"] = self.config.lambda_dependency * float(
                dependency_loss.detach()
            )

        evidence_target = self._target(
            sample, "ledger_evidence_strength_targets", output.evidence
        )
        if evidence_target is not None:
            if evidence_target.shape != output.evidence.shape:
                raise ValueError("`ledger_evidence_strength_targets` must be [B,num_claims].")
            evidence_mask = self._target(
                sample, "ledger_evidence_strength_mask", output.evidence
            )
            if evidence_mask is None:
                evidence_mask = torch.ones_like(evidence_target)
            evidence_loss = self._masked_mean(
                F.binary_cross_entropy(
                    output.evidence.float(), evidence_target.float(), reduction="none"
                ),
                evidence_mask,
            )
            total = total + self.config.lambda_evidence * evidence_loss
            metrics["loss_evidence"] = self.config.lambda_evidence * float(
                evidence_loss.detach()
            )

        if self.config.lambda_slot_identity > 0 and output.object_slots.shape[1] > 1:
            assignment = output.object_assignment.float().clamp(min=1e-8)
            assignment_entropy = -(assignment * assignment.log()).sum(dim=-1).mean()
            assignment_entropy = assignment_entropy / torch.log(
                torch.tensor(float(assignment.shape[-1]), device=assignment.device)
            )
            normalized_slots = F.normalize(output.object_slots.float(), dim=-1)
            similarity = torch.einsum("bih,bjh->bij", normalized_slots, normalized_slots)
            eye = torch.eye(
                similarity.shape[-1], device=similarity.device, dtype=torch.bool
            ).unsqueeze(0)
            diversity = F.relu(similarity.masked_fill(eye, 0.0) - 0.2)
            diversity_loss = diversity.sum() / (~eye).expand_as(diversity).sum().clamp(min=1)
            slot_loss = assignment_entropy + diversity_loss
            total = total + self.config.lambda_slot_identity * slot_loss
            metrics["loss_slot_identity"] = self.config.lambda_slot_identity * float(
                slot_loss.detach()
            )

        rollback_target = self._target(
            sample, "ledger_rollback_targets", output.debt, dtype=torch.long
        )
        if rollback_target is not None:
            if rollback_target.shape != output.debt.shape:
                raise ValueError("`ledger_rollback_targets` shape must match [B,num_claims].")
            rollback_mask = self._target(sample, "ledger_rollback_mask", output.debt)
            if rollback_mask is None:
                rollback_mask = claim_mask if claim_mask is not None else torch.ones_like(output.debt)
            rollback_raw = F.cross_entropy(
                output.rollback_logits.float().reshape(-1, output.rollback_logits.shape[-1]),
                rollback_target.reshape(-1),
                reduction="none",
            ).view_as(output.debt)
            rollback_loss = self._masked_mean(rollback_raw, rollback_mask)
            total = total + self.config.lambda_rollback * rollback_loss
            metrics["loss_rollback"] = self.config.lambda_rollback * float(rollback_loss.detach())

        relation_target = self._target(
            sample, "ledger_relation_targets", output.debt, dtype=torch.long
        )
        if relation_target is not None:
            if relation_target.shape != output.debt.shape:
                raise ValueError("`ledger_relation_targets` shape must match [B,num_claims].")
            relation_mask = self._target(sample, "ledger_relation_mask", output.debt)
            if relation_mask is None:
                relation_mask = claim_mask if claim_mask is not None else torch.ones_like(output.debt)
            relation_raw = F.cross_entropy(
                output.relation_logits.float().reshape(-1, output.relation_logits.shape[-1]),
                relation_target.reshape(-1),
                reduction="none",
            ).view_as(output.debt)
            relation_loss = self._masked_mean(relation_raw, relation_mask)
            total = total + self.config.lambda_relation * relation_loss
            metrics["loss_relation"] = self.config.lambda_relation * float(relation_loss.detach())

        entity_target = self._target(sample, "ledger_entity_targets", output.entity_logits)
        if entity_target is not None:
            if entity_target.shape != output.entity_logits.shape:
                raise ValueError(
                    "`ledger_entity_targets` shape must match [B,num_claims,num_entities]."
                )
            entity_mask = self._target(sample, "ledger_entity_mask", output.debt)
            if entity_mask is None:
                entity_mask = claim_mask if claim_mask is not None else torch.ones_like(output.debt)
            entity_raw = F.binary_cross_entropy_with_logits(
                output.entity_logits.float(), entity_target.float(), reduction="none"
            ).mean(dim=-1)
            entity_loss = self._masked_mean(entity_raw, entity_mask)
            total = total + self.config.lambda_structure * entity_loss
            metrics["loss_entity"] = self.config.lambda_structure * float(entity_loss.detach())

        object_entity_target = self._target(
            sample, "ledger_object_entity_targets", output.object_entity_logits
        )
        if object_entity_target is not None:
            if object_entity_target.shape != output.object_entity_logits.shape:
                raise ValueError(
                    "`ledger_object_entity_targets` must be [B,num_claims,num_slots]."
                )
            object_entity_mask = self._target(
                sample, "ledger_object_entity_mask", output.debt
            )
            if object_entity_mask is None:
                object_entity_mask = torch.ones_like(output.debt)
            object_entity_raw = F.binary_cross_entropy_with_logits(
                output.object_entity_logits.float(),
                object_entity_target.float(),
                reduction="none",
            ).mean(dim=-1)
            object_entity_loss = self._masked_mean(
                object_entity_raw, object_entity_mask
            )
            total = total + self.config.lambda_structure * object_entity_loss
            metrics["loss_object_entity"] = self.config.lambda_structure * float(
                object_entity_loss.detach()
            )

        structure_loss = zero
        structure_count = 0
        for key, logits in (
            ("ledger_precondition_targets", output.precondition_logits),
            ("ledger_effect_targets", output.effect_logits),
        ):
            target = self._target(sample, key, output.debt, dtype=torch.long)
            if target is None:
                continue
            if target.shape != output.debt.shape:
                raise ValueError(f"`{key}` shape must match [B,num_claims].")
            raw = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                target.reshape(-1),
                reduction="none",
            ).view_as(output.debt)
            mask = claim_mask if claim_mask is not None else torch.ones_like(output.debt)
            structure_loss = structure_loss + self._masked_mean(raw, mask)
            structure_count += 1
        if structure_count:
            structure_loss = structure_loss / structure_count
            total = total + self.config.lambda_structure * structure_loss
            metrics["loss_structure"] = self.config.lambda_structure * float(
                structure_loss.detach()
            )

        if counterfactual_output is not None:
            effect_distance = (
                output.effect_embedding.float() - counterfactual_output.effect_embedding.float()
            ).abs().mean(dim=-1)
            cf_loss = F.relu(self.config.counterfactual_margin - effect_distance).mean()
            total = total + self.config.lambda_counterfactual * cf_loss
            metrics["loss_counterfactual"] = self.config.lambda_counterfactual * float(cf_loss.detach())

        repair_target = self._target(sample, "ledger_repair_action", output.repair_action)
        repair_type = self._target(
            sample, "ledger_repair_type", output.global_risk, dtype=torch.long
        )
        if repair_target is not None:
            if repair_type is None:
                raise ValueError("`ledger_repair_type` is required with `ledger_repair_action`.")
            batch_indices = torch.arange(output.repair_action.shape[0], device=output.hidden.device)
            selected_repair = output.repair_action[batch_indices, repair_type]
            if selected_repair.shape != repair_target.shape:
                raise ValueError("`ledger_repair_action` shape must match [B,T,action_dim].")
            repair_raw = F.smooth_l1_loss(
                selected_repair.float(), repair_target.float(), reduction="none"
            ).mean(dim=(1, 2))
            repair_mask = self._target(sample, "ledger_repair_mask", output.global_risk)
            if repair_mask is None:
                repair_mask = torch.ones_like(output.global_risk)
            repair_loss = self._masked_mean(repair_raw, repair_mask)
            total = total + self.config.lambda_repair * repair_loss
            metrics["loss_repair"] = self.config.lambda_repair * float(repair_loss.detach())

        debt_reduction_target = self._target(
            sample, "ledger_repair_debt_reduction", output.expected_debt_reduction
        )
        repair_risk_target = self._target(sample, "ledger_repair_risk", output.repair_risk)
        if debt_reduction_target is not None or repair_risk_target is not None:
            repair_value_loss = zero
            repair_value_mask = self._target(
                sample, "ledger_repair_value_mask", output.global_risk
            )
            if repair_value_mask is None:
                repair_value_mask = torch.ones_like(output.global_risk)
            if debt_reduction_target is not None:
                debt_reduction_raw = F.smooth_l1_loss(
                    output.expected_debt_reduction.float(),
                    debt_reduction_target.float(),
                    reduction="none",
                ).mean(dim=-1)
                repair_value_loss = repair_value_loss + self._masked_mean(
                    debt_reduction_raw, repair_value_mask
                )
            if repair_risk_target is not None:
                repair_risk_raw = F.binary_cross_entropy(
                    output.repair_risk.float(), repair_risk_target.float(), reduction="none"
                ).mean(dim=-1)
                repair_value_loss = repair_value_loss + self._masked_mean(
                    repair_risk_raw, repair_value_mask
                )
            total = total + self.config.lambda_repair_value * repair_value_loss
            metrics["loss_repair_value"] = self.config.lambda_repair_value * float(
                repair_value_loss.detach()
            )

        if world_prediction is not None:
            world_loss = zero
            world_terms = 0
            world_value_mask = self._target(
                sample, "ledger_repair_value_mask", output.global_risk
            )
            if world_value_mask is None:
                world_value_mask = torch.ones_like(output.global_risk)
            if debt_reduction_target is not None:
                reduction_raw = F.smooth_l1_loss(
                    world_prediction.expected_debt_reduction.float(),
                    debt_reduction_target.float(),
                    reduction="none",
                ).mean(dim=-1)
                world_loss = world_loss + self._masked_mean(
                    reduction_raw, world_value_mask
                )
                world_terms += 1
            if repair_risk_target is not None:
                risk_raw = F.binary_cross_entropy(
                    world_prediction.predicted_failure_risk.float(),
                    repair_risk_target.float(),
                    reduction="none",
                ).mean(dim=-1)
                world_loss = world_loss + self._masked_mean(risk_raw, world_value_mask)
                world_terms += 1

            repair_type_for_world = repair_type
            if repair_type_for_world is None:
                repair_type_for_world = torch.zeros(
                    output.hidden.shape[0], device=output.hidden.device, dtype=torch.long
                )
            batch_indices = torch.arange(output.hidden.shape[0], device=output.hidden.device)
            next_claim_target = self._target(
                sample, "ledger_next_claim_labels", output.confidence
            )
            if next_claim_target is not None:
                selected_logits = world_prediction.confidence_logits[
                    batch_indices, repair_type_for_world
                ]
                if selected_logits.shape != next_claim_target.shape:
                    raise ValueError("`ledger_next_claim_labels` must be [B,num_claims].")
                world_loss = world_loss + F.binary_cross_entropy_with_logits(
                    selected_logits.float(), next_claim_target.float()
                )
                world_terms += 1
            next_debt_target = self._target(sample, "ledger_next_debt_targets", output.debt)
            if next_debt_target is not None:
                selected_debt = world_prediction.debt[batch_indices, repair_type_for_world]
                if selected_debt.shape != next_debt_target.shape:
                    raise ValueError("`ledger_next_debt_targets` must be [B,num_claims].")
                world_loss = world_loss + F.smooth_l1_loss(
                    selected_debt.float(), next_debt_target.float()
                )
                world_terms += 1
            observation_target = self._target(
                sample,
                "ledger_next_observation_embedding",
                world_prediction.predicted_observation,
            )
            if observation_target is not None:
                selected_observation = world_prediction.predicted_observation[
                    batch_indices, repair_type_for_world
                ]
                if selected_observation.shape != observation_target.shape:
                    raise ValueError(
                        "`ledger_next_observation_embedding` must be [B,hidden_dim]."
                    )
                world_loss = world_loss + (1.0 - F.cosine_similarity(
                    selected_observation.float(), observation_target.float(), dim=-1
                )).mean()
                world_terms += 1
            if world_terms:
                world_loss = world_loss / world_terms
                total = total + self.config.lambda_world * world_loss
                metrics["loss_world"] = self.config.lambda_world * float(world_loss.detach())

        metrics["ledger_global_risk"] = float(output.global_risk.detach().mean())
        metrics["ledger_mean_debt"] = float(output.debt.detach().mean())
        return total, metrics
