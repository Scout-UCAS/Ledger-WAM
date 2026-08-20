from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch


@dataclass(frozen=True)
class TaskRollback:
    checkpoint_step: int
    subplan_start: int
    invalidated_claims: tuple[int, ...]
    reason_claim: int

    def summary(self) -> dict[str, Any]:
        return {
            "checkpoint_step": int(self.checkpoint_step),
            "subplan_start": int(self.subplan_start),
            "invalidated_claims": list(self.invalidated_claims),
            "reason_claim": int(self.reason_claim),
        }


class CausalTaskGraph:
    """Runtime DAG for dependency-aware descendant invalidation and replanning."""

    def __init__(
        self,
        claim_names: Sequence[str],
        edges: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> None:
        self.claim_names = tuple(str(name) for name in claim_names)
        self.name_to_index = {name: index for index, name in enumerate(self.claim_names)}
        if edges is None:
            edges = {
                "contact": ("grasped",),
                "grasped": ("supported", "contained", "persistent"),
                "visible": ("persistent", "precondition_met"),
                "supported": ("precondition_met",),
                "contained": ("precondition_met",),
                "precondition_met": ("effect_achieved",),
                "persistent": ("effect_achieved",),
            }
        self.children: dict[int, set[int]] = {index: set() for index in range(len(self.claim_names))}
        for parent_name, child_names in edges.items():
            if parent_name not in self.name_to_index:
                continue
            parent = self.name_to_index[parent_name]
            for child_name in child_names:
                if child_name in self.name_to_index:
                    self.children[parent].add(self.name_to_index[child_name])
        self.reset()

    def reset(self) -> None:
        self.verified = torch.zeros(len(self.claim_names), dtype=torch.bool)
        self.claim_checkpoint = torch.full((len(self.claim_names),), -1, dtype=torch.long)
        self.current_subplan_start = 0

    def descendants(self, claim: int) -> tuple[int, ...]:
        seen: set[int] = set()
        frontier = list(self.children.get(int(claim), ()))
        while frontier:
            node = frontier.pop()
            if node in seen:
                continue
            seen.add(node)
            frontier.extend(self.children.get(node, ()))
        return tuple(sorted(seen))

    def update(self, confidence: torch.Tensor, step: int, threshold: float = 0.7) -> None:
        if confidence.ndim != 1 or confidence.shape[0] != len(self.claim_names):
            raise ValueError("Task-graph confidence must be [num_claims].")
        confident = confidence.detach().float().cpu() >= float(threshold)
        newly_verified = confident & ~self.verified
        self.verified |= confident
        self.claim_checkpoint[newly_verified] = int(step)

    def update_structure(
        self,
        precondition_logits: torch.Tensor,
        effect_logits: torch.Tensor,
        confidence_threshold: float = 0.6,
    ) -> None:
        """Adds high-level dependencies predicted by the current causal ledger."""

        if precondition_logits.ndim != 2 or effect_logits.ndim != 2:
            raise ValueError("Task-graph structure logits must be [claims,claims+1].")
        if precondition_logits.shape[0] != len(self.claim_names):
            raise ValueError("Task-graph structure claim count mismatch.")
        precondition_probability = precondition_logits.detach().float().cpu().softmax(dim=-1)
        effect_probability = effect_logits.detach().float().cpu().softmax(dim=-1)
        precondition_confidence, preconditions = precondition_probability.max(dim=-1)
        effect_confidence, effects = effect_probability.max(dim=-1)
        for claim, precondition in enumerate(preconditions.tolist()):
            if 0 <= precondition < len(self.claim_names) and precondition != claim:
                if float(precondition_confidence[claim]) >= confidence_threshold:
                    self.children[precondition].add(claim)
        for claim, effect in enumerate(effects.tolist()):
            if 0 <= effect < len(self.claim_names) and effect != claim:
                if float(effect_confidence[claim]) >= confidence_threshold:
                    self.children[claim].add(effect)

    def rollback(
        self,
        reason_claim: int,
        requested_lookback: int,
        current_step: int,
    ) -> TaskRollback:
        reason_claim = int(reason_claim)
        invalidated = (reason_claim,) + self.descendants(reason_claim)
        checkpoint_candidates = self.claim_checkpoint[list(invalidated)]
        valid = checkpoint_candidates[checkpoint_candidates >= 0]
        learned_checkpoint = int(valid.min().item()) if valid.numel() else int(current_step)
        requested_checkpoint = max(0, int(current_step) - max(0, int(requested_lookback)))
        checkpoint = min(learned_checkpoint, requested_checkpoint)
        self.verified[list(invalidated)] = False
        self.claim_checkpoint[list(invalidated)] = -1
        self.current_subplan_start = checkpoint
        return TaskRollback(
            checkpoint_step=checkpoint,
            subplan_start=checkpoint,
            invalidated_claims=tuple(int(index) for index in invalidated),
            reason_claim=reason_claim,
        )
