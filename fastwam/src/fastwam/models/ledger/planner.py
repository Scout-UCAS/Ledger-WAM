from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch

from .ledger import CausalLedgerOutput
from .world_predictor import CandidateWorldPrediction


@dataclass(frozen=True)
class PlannerDecision:
    action: torch.Tensor
    mode: str
    repair_name: str | None
    execution_horizon: int
    target_claim: int | None
    rollback_step: int | None
    global_risk: float
    score: float | None

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "repair_name": self.repair_name,
            "execution_horizon": int(self.execution_horizon),
            "target_claim": self.target_claim,
            "rollback_step": self.rollback_step,
            "global_risk": float(self.global_risk),
            "score": self.score,
        }


class SelfHealingPlanner:
    """Debt-aware task/repair gate with targeted local rollback."""

    def __init__(
        self,
        repair_names: Sequence[str] = ("verify", "hold", "retract", "regrasp"),
        global_risk_threshold: float = 0.6,
        claim_debt_threshold: float = 0.65,
        importance_threshold: float = 0.35,
        conflict_confidence_threshold: float = 0.3,
        verification_horizon: int = 1,
        task_execution_horizon: int = 8,
        action_costs: Sequence[float] = (0.05, 0.1, 0.25, 0.4),
        cost_weight: float = 0.25,
        risk_weight: float = 0.5,
        use_learned_repair_actions: bool = False,
        repair_horizons: Optional[Sequence[int]] = None,
    ) -> None:
        self.repair_names = tuple(str(name) for name in repair_names)
        self.global_risk_threshold = float(global_risk_threshold)
        self.claim_debt_threshold = float(claim_debt_threshold)
        self.importance_threshold = float(importance_threshold)
        self.conflict_confidence_threshold = float(conflict_confidence_threshold)
        self.verification_horizon = int(verification_horizon)
        self.task_execution_horizon = int(task_execution_horizon)
        self.action_costs = tuple(float(cost) for cost in action_costs)
        self.cost_weight = float(cost_weight)
        self.risk_weight = float(risk_weight)
        self.use_learned_repair_actions = bool(use_learned_repair_actions)
        self.repair_horizons = tuple(
            int(value)
            for value in (
                repair_horizons
                if repair_horizons is not None
                else [verification_horizon] * len(self.repair_names)
            )
        )
        if len(self.repair_names) != len(self.action_costs):
            raise ValueError("`repair_names` and `action_costs` must have equal lengths.")
        if len(self.repair_names) != len(self.repair_horizons):
            raise ValueError("`repair_names` and `repair_horizons` must have equal lengths.")
        if self.verification_horizon <= 0 or self.task_execution_horizon <= 0:
            raise ValueError("Planner execution horizons must be positive.")

    def build_repair_candidates(self, task_action: torch.Tensor) -> torch.Tensor:
        """Builds verify/hold/retract/regrasp candidates in normalized action space."""

        if task_action.ndim == 2:
            task_action = task_action.unsqueeze(0)
        if task_action.ndim != 3:
            raise ValueError("`task_action` must be [T,A] or [B,T,A].")
        batch_size, horizon, action_dim = task_action.shape
        candidates = []
        for repair_name in self.repair_names:
            if repair_name == "verify":
                candidate = task_action.clone()
            elif repair_name == "hold":
                candidate = torch.zeros_like(task_action)
                if action_dim:
                    candidate[..., -1] = task_action[:, :1, -1]
            elif repair_name == "retract":
                candidate = torch.zeros_like(task_action)
                motion_dims = min(3, action_dim)
                candidate[..., :motion_dims] = -0.5 * task_action[:, :1, :motion_dims]
                if action_dim:
                    candidate[..., -1] = task_action[:, :1, -1]
            elif repair_name == "regrasp":
                candidate = torch.zeros_like(task_action)
                if action_dim:
                    midpoint = max(1, horizon // 2)
                    candidate[:, :midpoint, -1] = -1.0
                    candidate[:, midpoint:, -1] = 1.0
            else:
                candidate = task_action.clone()
            candidates.append(candidate)
        return torch.stack(candidates, dim=1).view(
            batch_size, len(self.repair_names), horizon, action_dim
        )

    def decide(
        self,
        task_action: torch.Tensor,
        ledger: CausalLedgerOutput,
        batch_index: int = 0,
        world_prediction: Optional[CandidateWorldPrediction] = None,
        repair_candidates: Optional[torch.Tensor] = None,
    ) -> PlannerDecision:
        if task_action.ndim != 2:
            raise ValueError("`task_action` must have shape [T,A].")
        risk = float(ledger.global_risk[batch_index].detach().float().cpu())
        debt = ledger.debt[batch_index]
        importance = ledger.importance[batch_index]
        critical_mask = (debt >= self.claim_debt_threshold) & (
            importance >= self.importance_threshold
        )

        if risk < self.global_risk_threshold or not bool(critical_mask.any().item()):
            return PlannerDecision(
                action=task_action,
                mode="task",
                repair_name=None,
                execution_horizon=min(self.task_execution_horizon, task_action.shape[0]),
                target_claim=None,
                rollback_step=None,
                global_risk=risk,
                score=None,
            )

        critical_score = debt * importance * critical_mask.to(debt.dtype)
        target_claim = int(critical_score.argmax().item())
        rollback_step = None
        mode = "repair"
        if float(ledger.confidence[batch_index, target_claim].detach().float().cpu()) < (
            self.conflict_confidence_threshold
        ):
            rollback_step = int(
                ledger.rollback_logits[batch_index, target_claim].argmax(dim=-1).item()
            )
            mode = "rollback"

        if world_prediction is None:
            values = ledger.expected_debt_reduction[batch_index]
            repair_risk = ledger.repair_risk[batch_index]
        else:
            values = world_prediction.expected_debt_reduction[batch_index]
            repair_risk = world_prediction.predicted_failure_risk[batch_index]
        if values.numel() != len(self.repair_names):
            raise ValueError("Ledger repair head and planner repair library sizes do not match.")
        action_cost = torch.as_tensor(
            self.action_costs, device=values.device, dtype=values.dtype
        )
        scores = values - self.cost_weight * action_cost - self.risk_weight * repair_risk
        repair_index = int(scores.argmax().item())
        repair_name = self.repair_names[repair_index]

        if self.use_learned_repair_actions:
            repair_action = ledger.repair_action[batch_index, repair_index]
            repair_action = repair_action[: task_action.shape[0]].to(
                device=task_action.device, dtype=task_action.dtype
            )
        elif repair_candidates is not None:
            if repair_candidates.ndim == 4:
                repair_action = repair_candidates[batch_index, repair_index]
            elif repair_candidates.ndim == 3:
                repair_action = repair_candidates[repair_index]
            else:
                raise ValueError("Repair candidates must have shape [B,R,T,A] or [R,T,A].")
            repair_action = repair_action.to(device=task_action.device, dtype=task_action.dtype)
        else:
            repair_action = self.build_repair_candidates(task_action)[0, repair_index]

        return PlannerDecision(
            action=repair_action,
            mode=mode,
            repair_name=repair_name,
            execution_horizon=min(self.repair_horizons[repair_index], repair_action.shape[0]),
            target_claim=target_claim,
            rollback_step=rollback_step,
            global_risk=risk,
            score=float(scores[repair_index].detach().float().cpu()),
        )
