from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Optional

import torch

from fastwam.utils.logging_config import get_logger

from ..ledger import (
    CausalBeliefLedger,
    CausalLedgerOutput,
    CausalLedgerState,
    CausalTaskGraph,
    CausalWorldPredictor,
    LedgerLoss,
    LedgerLossConfig,
    SelfHealingPlanner,
)
from .fastwam import FastWAM

logger = get_logger(__name__)


class LedgerWAM(FastWAM):
    """Fast-WAM with a recurrent causal ledger and debt-aware action gate."""

    def __init__(
        self,
        *args,
        ledger_config: Optional[Mapping[str, Any]] = None,
        ledger_loss_config: Optional[Mapping[str, Any]] = None,
        planner_config: Optional[Mapping[str, Any]] = None,
        world_predictor_config: Optional[Mapping[str, Any]] = None,
        task_graph_config: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        ledger_config = dict(ledger_config or {})
        ledger_loss_config = dict(ledger_loss_config or {})
        planner_config = dict(planner_config or {})
        world_predictor_config = dict(world_predictor_config or {})
        task_graph_config = dict(task_graph_config or {})

        action_horizon = int(ledger_config.pop("action_horizon"))
        self.temporal_unroll_steps = int(ledger_config.pop("temporal_unroll_steps", 4))
        ledger_config.setdefault("input_dim", int(self.action_expert.hidden_dim))
        ledger_config.setdefault(
            "visual_input_dim",
            int(getattr(self.video_expert, "hidden_dim", self.action_expert.hidden_dim)),
        )
        ledger_config.setdefault("action_dim", int(self.action_expert.action_dim))
        ledger_config.setdefault("action_horizon", action_horizon)
        self.ledger = CausalBeliefLedger(**ledger_config).to(
            device=self.device, dtype=self.torch_dtype
        )
        self.ledger_loss = LedgerLoss(LedgerLossConfig(**ledger_loss_config))
        self.world_predictor_enabled = bool(world_predictor_config.pop("enabled", True))
        self.world_predictor = CausalWorldPredictor(
            hidden_dim=self.ledger.hidden_dim,
            action_dim=self.ledger.action_dim,
            num_claims=self.ledger.num_claims,
            num_object_slots=self.ledger.num_object_slots,
            **world_predictor_config,
        ).to(device=self.device, dtype=self.torch_dtype)

        planner_config.setdefault("task_execution_horizon", action_horizon)
        self.self_healing_planner = SelfHealingPlanner(**planner_config)
        if len(self.self_healing_planner.repair_names) != self.ledger.num_repair_actions:
            raise ValueError(
                "Planner `repair_names` count must match ledger `num_repair_actions`."
            )
        self.task_graph = CausalTaskGraph(
            claim_names=self.ledger.claim_names,
            **task_graph_config,
        )

        self._runtime_state: Optional[CausalLedgerState] = None
        self._runtime_history: deque[CausalLedgerState] = deque(
            maxlen=max(1, self.ledger.max_rollback_steps)
        )
        self._episode_events: list[dict[str, Any]] = []

    def auxiliary_training_loss(
        self,
        sample,
        state_tokens: torch.Tensor,
        video_state_tokens: Optional[torch.Tensor],
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        sensor_evidence = sample.get("ledger_evidence")
        if sensor_evidence is not None:
            sensor_evidence = sensor_evidence.to(device=state_tokens.device, dtype=state_tokens.dtype)
        outputs = self.ledger.forward_sequence(
            state_tokens=state_tokens,
            action=action,
            visual_tokens=video_state_tokens,
            sensor_evidence=sensor_evidence,
            num_steps=self.temporal_unroll_steps,
        )
        output = outputs[-1]

        counterfactual_output = None
        if self.ledger_loss.config.lambda_counterfactual > 0:
            if action.shape[1] > 1:
                counterfactual_action = action.roll(shifts=1, dims=1)
            else:
                counterfactual_action = -action
            counterfactual_output = self.ledger.forward_sequence(
                state_tokens=state_tokens,
                action=counterfactual_action,
                visual_tokens=video_state_tokens,
                sensor_evidence=sensor_evidence,
                num_steps=self.temporal_unroll_steps,
            )[-1]

        repair_candidates = self.self_healing_planner.build_repair_candidates(action)
        world_prediction = (
            self.world_predictor(output, repair_candidates)
            if self.world_predictor_enabled
            else None
        )
        loss, metrics = self.ledger_loss(
            output=output,
            sample=sample,
            counterfactual_output=counterfactual_output,
            world_prediction=world_prediction,
        )
        if world_prediction is not None and self.ledger_loss.config.lambda_world > 0:
            target_observation = (
                output.object_presence.unsqueeze(-1) * output.object_slots
            ).sum(dim=1) / output.object_presence.sum(dim=1, keepdim=True).clamp(min=1e-6)
            observation_loss = (
                1.0
                - torch.nn.functional.cosine_similarity(
                    world_prediction.predicted_observation[:, 0].float(),
                    target_observation.detach().float(),
                    dim=-1,
                )
            ).mean()
            weighted_observation_loss = (
                self.ledger_loss.config.lambda_world * observation_loss
            )
            loss = loss + weighted_observation_loss
            metrics["loss_world_observation"] = float(
                weighted_observation_loss.detach()
            )
        # Dense sequence sidecars supervise every recurrent update.  Final-only
        # sidecars still backpropagate through the complete unroll above.
        if "ledger_claim_labels_sequence" in sample and len(outputs) > 1:
            sequence_loss = loss * 0.0
            sequence_metrics: dict[str, float] = {}
            intermediate_outputs = outputs[:-1]
            for step_index, step_output in enumerate(intermediate_outputs):
                step_sample = self._sequence_sample(sample, step_index)
                step_loss, step_metrics = self.ledger_loss(step_output, step_sample)
                sequence_loss = sequence_loss + step_loss / len(intermediate_outputs)
                for key, value in step_metrics.items():
                    sequence_metrics[key] = (
                        sequence_metrics.get(key, 0.0) + value / len(intermediate_outputs)
                    )
            loss = loss + sequence_loss
            metrics.update({f"temporal_{key}": value for key, value in sequence_metrics.items()})
        return loss, metrics

    @staticmethod
    def _sequence_sample(sample: Mapping[str, Any], step_index: int) -> dict[str, Any]:
        output = dict(sample)
        for key, value in sample.items():
            if not key.endswith("_sequence"):
                continue
            base_key = key[: -len("_sequence")]
            if isinstance(value, torch.Tensor):
                index = min(step_index, value.shape[1] - 1)
                output[base_key] = value[:, index]
        return output

    def _rollback_runtime(self, lookback: int) -> None:
        if not self._runtime_history:
            self._runtime_state = None
            return
        depth = min(max(int(lookback), 0), len(self._runtime_history) - 1)
        target_index = len(self._runtime_history) - 1 - depth
        restored = list(self._runtime_history)[target_index].detached()
        while len(self._runtime_history) > target_index + 1:
            self._runtime_history.pop()
        self._runtime_state = restored

    def reset_ledger(self) -> None:
        self._runtime_state = None
        self._runtime_history.clear()
        self._episode_events.clear()
        self.task_graph.reset()

    def get_ledger_episode_metrics(self, reset: bool = False) -> dict[str, Any]:
        events = list(self._episode_events)
        risks = [float(event["planner"]["global_risk"]) for event in events]
        repair_count = sum(event["planner"]["mode"] == "repair" for event in events)
        rollback_count = sum(event["planner"]["mode"] == "rollback" for event in events)
        repair_indices = [
            index
            for index, event in enumerate(events)
            if event["planner"]["mode"] in {"repair", "rollback"}
        ]
        recovered = []
        recovery_steps = []
        unnecessary = []
        world_errors = []
        for index in repair_indices:
            planner = events[index]["planner"]
            future = [
                step
                for step in range(index + 1, len(events))
                if events[step]["planner"]["global_risk"]
                < self.self_healing_planner.global_risk_threshold
            ]
            recovered.append(bool(future))
            recovery_steps.append((future[0] - index) if future else float("inf"))
            prediction = planner.get("world_prediction", {})
            repair_name = planner.get("repair_name")
            if repair_name in self.self_healing_planner.repair_names:
                candidate = self.self_healing_planner.repair_names.index(repair_name)
                reductions = prediction.get("expected_debt_reduction", [])
                unnecessary.append(
                    bool(candidate < len(reductions) and reductions[candidate] <= 0.0)
                )
                risks_after = prediction.get("candidate_global_risk", [])
                if index + 1 < len(events) and candidate < len(risks_after):
                    world_errors.append(
                        abs(
                            float(risks_after[candidate])
                            - float(events[index + 1]["planner"]["global_risk"])
                        )
                    )
        finite_recovery_steps = [value for value in recovery_steps if value != float("inf")]
        result = {
            "num_replans": len(events),
            "num_repairs": int(repair_count),
            "num_rollbacks": int(rollback_count),
            "repair_rate": float(repair_count / max(len(events), 1)),
            "rollback_rate": float(rollback_count / max(len(events), 1)),
            "mean_global_risk": float(sum(risks) / max(len(risks), 1)),
            "max_global_risk": float(max(risks, default=0.0)),
            "recovery_success_rate": float(sum(recovered) / max(len(recovered), 1)),
            "mean_recovery_replans": float(
                sum(finite_recovery_steps) / max(len(finite_recovery_steps), 1)
            ),
            "unnecessary_repair_rate": float(
                sum(unnecessary) / max(len(unnecessary), 1)
            ),
            "world_risk_prediction_mae": float(
                sum(world_errors) / max(len(world_errors), 1)
            ),
            "events": events,
        }
        if reset:
            self.reset_ledger()
        return result

    @torch.no_grad()
    def infer_action(self, *args, **kwargs) -> dict[str, Any]:
        if kwargs.pop("return_features", False):
            logger.warning("LedgerWAM owns `return_features`; returning ledger metadata instead.")
        sensor_evidence = kwargs.pop("ledger_evidence", None)
        base_output = super().infer_action(*args, return_features=True, **kwargs)
        action_features = base_output.pop("action_features")
        video_features = base_output.pop("video_features", None)
        task_action = base_output["action"]
        action_for_ledger = task_action.unsqueeze(0).to(
            device=action_features.device, dtype=action_features.dtype
        )
        previous_state = self._runtime_state
        if sensor_evidence is not None:
            sensor_evidence = torch.as_tensor(
                sensor_evidence, device=action_features.device, dtype=action_features.dtype
            )
            if sensor_evidence.ndim == 1:
                sensor_evidence = sensor_evidence.unsqueeze(0)
        ledger_output = self.ledger(
            state_tokens=action_features,
            action=action_for_ledger,
            previous_state=previous_state,
            visual_tokens=video_features,
            sensor_evidence=sensor_evidence,
        )
        repair_candidates = self.self_healing_planner.build_repair_candidates(
            action_for_ledger
        )
        world_prediction = (
            self.world_predictor(ledger_output, repair_candidates)
            if self.world_predictor_enabled
            else None
        )
        decision = self.self_healing_planner.decide(
            task_action=task_action,
            ledger=ledger_output,
            world_prediction=world_prediction,
            repair_candidates=repair_candidates,
        )

        ledger_summary = ledger_output.summary(
            claim_names=self.ledger.claim_names,
            relation_names=self.ledger.relation_names,
            entity_names=self.ledger.entity_names,
        )
        planner_summary = decision.summary()
        if world_prediction is not None:
            planner_summary["world_prediction"] = world_prediction.summary()
        self.task_graph.update_structure(
            ledger_output.precondition_logits[0],
            ledger_output.effect_logits[0],
        )
        self.task_graph.update(
            ledger_output.confidence[0],
            step=ledger_output.next_state.step,
        )
        if decision.mode == "rollback" and decision.rollback_step is not None:
            rollback = self.task_graph.rollback(
                reason_claim=int(decision.target_claim or 0),
                requested_lookback=decision.rollback_step,
                current_step=ledger_output.next_state.step,
            )
            planner_summary["task_rollback"] = rollback.summary()
        event = {"ledger": ledger_summary, "planner": planner_summary}
        self._episode_events.append(event)

        if decision.mode == "rollback" and decision.rollback_step is not None:
            self._rollback_runtime(decision.rollback_step)
        else:
            self._runtime_state = ledger_output.next_state.detached()
            self._runtime_history.append(self._runtime_state)

        base_output["action"] = decision.action.detach().to(device="cpu", dtype=torch.float32)
        base_output["execution_horizon"] = int(decision.execution_horizon)
        base_output["ledger"] = ledger_summary
        base_output["planner"] = planner_summary
        return base_output

    def auxiliary_checkpoint_state(self) -> dict[str, Any]:
        return {
            "ledger": self.ledger.state_dict(),
            "world_predictor": self.world_predictor.state_dict(),
        }

    def load_auxiliary_checkpoint_state(self, state: Optional[dict[str, Any]]) -> None:
        if state is None or "ledger" not in state:
            logger.warning(
                "Checkpoint has no Ledger-WAM auxiliary weights; ledger heads remain initialized."
            )
            return
        incompatible = self.ledger.load_state_dict(state["ledger"], strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            logger.warning(
                "Ledger checkpoint schema changed; initialized missing=%s ignored=%s",
                incompatible.missing_keys,
                incompatible.unexpected_keys,
            )
        if "world_predictor" in state:
            self.world_predictor.load_state_dict(state["world_predictor"], strict=True)
        else:
            logger.warning("Checkpoint has no causal world-predictor weights.")

    def set_train_mode_for_training(self) -> None:
        super().set_train_mode_for_training()
        self.ledger.train()
        self.ledger.requires_grad_(True)
        self.world_predictor.train()
        self.world_predictor.requires_grad_(True)
