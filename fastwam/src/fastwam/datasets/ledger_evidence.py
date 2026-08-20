from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import torch.nn.functional as F


EVIDENCE_FEATURE_NAMES = (
    "visual_change_mean",
    "visual_change_std",
    "brightness_mean",
    "brightness_std",
    "spatial_edge_energy",
    "proprio_motion_mean",
    "proprio_motion_max",
    "action_magnitude_mean",
    "action_magnitude_max",
    "translation_magnitude",
    "rotation_magnitude",
    "gripper_command",
    "gripper_change",
    "action_observation_consistency",
    "valid_fraction",
    "bias",
)


def _resample_time(value: torch.Tensor, steps: int) -> torch.Tensor:
    if value.shape[0] == steps:
        return value
    value = value.transpose(0, 1).unsqueeze(0)
    value = F.interpolate(value.float(), size=steps, mode="linear", align_corners=False)
    return value.squeeze(0).transpose(0, 1).to(dtype=value.dtype)


def build_ledger_evidence(
    video: torch.Tensor,
    proprio: torch.Tensor,
    action: torch.Tensor,
    *,
    steps: int = 4,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Derives explicit multimodal causal evidence for each temporal ledger step."""

    if video.ndim != 4:
        raise ValueError("`video` must be [C,T,H,W].")
    if proprio.ndim != 2 or action.ndim != 2:
        raise ValueError("`proprio` and `action` must be [T,D].")
    frame = video.float().mean(dim=0)
    frame_stats = torch.stack(
        (
            frame.mean(dim=(-2, -1)),
            frame.std(dim=(-2, -1)),
            (frame[:, 1:] - frame[:, :-1]).abs().mean(dim=(-2, -1)),
            (frame[:, :, 1:] - frame[:, :, :-1]).abs().mean(dim=(-2, -1)),
        ),
        dim=-1,
    )
    frame_stats = _resample_time(frame_stats, steps)
    visual_delta = frame_stats[:, :2] - torch.roll(frame_stats[:, :2], 1, 0)
    visual_delta[0].zero_()

    action_steps = _resample_time(action.float(), steps)
    proprio_steps = _resample_time(proprio.float(), steps)
    proprio_delta = proprio_steps - torch.roll(proprio_steps, 1, 0)
    proprio_delta[0].zero_()
    action_delta = action_steps - torch.roll(action_steps, 1, 0)
    action_delta[0].zero_()

    translation = action_steps[:, : min(3, action_steps.shape[-1])].norm(dim=-1)
    rotation_start = min(3, action_steps.shape[-1])
    rotation_end = min(6, action_steps.shape[-1])
    rotation = (
        action_steps[:, rotation_start:rotation_end].norm(dim=-1)
        if rotation_end > rotation_start
        else torch.zeros_like(translation)
    )
    gripper = action_steps[:, -1] if action_steps.shape[-1] else torch.zeros_like(translation)
    gripper_delta = action_delta[:, -1] if action_delta.shape[-1] else torch.zeros_like(translation)
    action_magnitude = action_steps.norm(dim=-1)
    proprio_magnitude = proprio_delta.norm(dim=-1)
    consistency = torch.exp(-(action_magnitude - proprio_magnitude).abs())
    if valid_mask is None:
        valid_fraction = torch.ones_like(translation)
    else:
        valid_fraction = _resample_time(valid_mask.float().view(-1, 1), steps).squeeze(-1)

    return torch.stack(
        (
            visual_delta[:, 0].abs(),
            visual_delta[:, 1].abs(),
            frame_stats[:, 0],
            frame_stats[:, 1],
            0.5 * (frame_stats[:, 2] + frame_stats[:, 3]),
            proprio_magnitude,
            proprio_delta.abs().amax(dim=-1),
            action_magnitude,
            action_steps.abs().amax(dim=-1),
            translation,
            rotation,
            gripper,
            gripper_delta,
            consistency,
            valid_fraction,
            torch.ones_like(translation),
        ),
        dim=-1,
    )


@dataclass
class WeakCausalAnnotator:
    """Creates dense weak supervision when simulator event labels are unavailable."""

    num_claims: int = 8
    num_entities: int = 8
    num_repair_actions: int = 4
    steps: int = 4

    def annotate(
        self,
        video: torch.Tensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
        *,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Mapping[str, torch.Tensor]:
        evidence = build_ledger_evidence(
            video, proprio, action, steps=self.steps, valid_mask=valid_mask
        )
        visual_change = evidence[:, 0]
        proprio_motion = evidence[:, 5]
        translation = evidence[:, 9]
        gripper = evidence[:, 11]
        gripper_change = evidence[:, 12]
        consistency = evidence[:, 13]

        labels = torch.stack(
            (
                torch.sigmoid(3.0 * (translation + proprio_motion) - 1.0),
                torch.sigmoid(-2.0 * gripper + 2.0 * consistency - 0.5),
                torch.sigmoid(1.5 - 3.0 * proprio_motion),
                torch.sigmoid(gripper_change + consistency - 0.5),
                torch.sigmoid(2.0 * evidence[:, 3] - visual_change),
                torch.sigmoid(2.0 * consistency - visual_change),
                torch.sigmoid(1.5 * consistency - translation),
                torch.sigmoid(1.5 * consistency + visual_change - 0.5),
            ),
            dim=-1,
        )[:, : self.num_claims]
        uncertainty = 1.0 - (labels - 0.5).abs() * 2.0
        evidence_strength = 1.0 - uncertainty
        dependency = torch.linspace(0.4, 1.0, self.num_claims, device=labels.device)
        debt = torch.sigmoid(
            (1.0 - labels) + uncertainty + dependency.unsqueeze(0) + (1.0 - consistency[:, None])
            - 2.0
        )
        mask = torch.full_like(labels, 0.35)
        relation = torch.arange(self.num_claims, device=labels.device).clamp(max=7)
        entity_targets = torch.zeros(
            self.steps, self.num_claims, self.num_entities, device=labels.device
        )
        if self.num_entities >= 3:
            entity_targets[:, :, 2] = 1.0
            entity_targets[:, :2, 1] = 1.0
        preconditions = torch.full(
            (self.steps, self.num_claims), self.num_claims, dtype=torch.long
        )
        effects = preconditions.clone()
        if self.num_claims >= 8:
            preconditions[:, 1] = 0
            preconditions[:, 6] = 2
            effects[:, 0] = 1
            effects[:, 1] = 5
            effects[:, 6] = 7
        rollback = torch.zeros(self.steps, self.num_claims, dtype=torch.long)
        low_confidence = labels < 0.35
        rollback[low_confidence] = torch.arange(self.steps)[:, None].expand_as(rollback)[
            low_confidence
        ]
        next_labels = torch.cat((labels[1:], labels[-1:]), dim=0)
        next_debt = torch.cat((debt[1:], debt[-1:]), dim=0)

        return {
            "ledger_evidence": evidence,
            "ledger_claim_labels_sequence": labels,
            "ledger_claim_mask_sequence": mask,
            "ledger_debt_targets_sequence": debt,
            "ledger_debt_mask_sequence": mask,
            "ledger_dependency_targets_sequence": dependency.unsqueeze(0).expand(
                self.steps, -1
            ),
            "ledger_dependency_mask_sequence": mask,
            "ledger_evidence_strength_targets_sequence": evidence_strength,
            "ledger_evidence_strength_mask_sequence": mask,
            "ledger_rollback_targets_sequence": rollback,
            "ledger_relation_targets_sequence": relation.unsqueeze(0).expand(self.steps, -1),
            "ledger_entity_targets_sequence": entity_targets,
            "ledger_precondition_targets_sequence": preconditions,
            "ledger_effect_targets_sequence": effects,
            "ledger_claim_labels": labels[-1],
            "ledger_claim_mask": mask[-1],
            "ledger_debt_targets": debt[-1],
            "ledger_debt_mask": mask[-1],
            "ledger_dependency_targets": dependency,
            "ledger_dependency_mask": mask[-1],
            "ledger_evidence_strength_targets": evidence_strength[-1],
            "ledger_evidence_strength_mask": mask[-1],
            "ledger_rollback_targets": rollback[-1],
            "ledger_relation_targets": relation,
            "ledger_entity_targets": entity_targets[-1],
            "ledger_entity_mask": mask[-1],
            "ledger_precondition_targets": preconditions[-1],
            "ledger_effect_targets": effects[-1],
            "ledger_next_claim_labels": next_labels[-1],
            "ledger_next_debt_targets": next_debt[-1],
            "ledger_repair_debt_reduction": torch.zeros(self.num_repair_actions),
            "ledger_repair_risk": torch.zeros(self.num_repair_actions),
            "ledger_repair_value_mask": torch.tensor(0.0),
        }
