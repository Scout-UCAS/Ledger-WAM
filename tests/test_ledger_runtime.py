"""Unit tests for the dependency-free Ledger-WAM runtime."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType
from typing import Dict


# The upstream ``wan_va`` package eagerly imports optional training dependencies.
# Isolate this standard-library-only runtime so these tests remain runnable in a
# lightweight Python 3.9 environment.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if "wan_va" not in sys.modules:
    wan_va_package = ModuleType("wan_va")
    wan_va_package.__path__ = [str(REPOSITORY_ROOT / "wan_va")]
    sys.modules["wan_va"] = wan_va_package

from wan_va.ledger import (  # noqa: E402
    CausalBeliefLedger,
    CausalClaim,
    ClaimStatus,
    DebtWeights,
    Evidence,
    EvidencePolarity,
    LedgerRuntimeState,
    LogicalPlanState,
    PlannerDecisionType,
    PlanningCheckpoint,
    RepairCandidate,
    SelfHealingPlanner,
    SelfHealingPlannerConfig,
)


def make_claim(claim_id: str, **overrides: object) -> CausalClaim:
    values: Dict[str, object] = {
        "claim_id": claim_id,
        "entities": ("cup", "gripper"),
        "relation": "grasped_by",
        "confidence": 1.0,
        "uncertainty": 0.0,
        "dependency": 0.0,
        "repair_cost": 0.0,
        "observability": 1.0,
        "importance": 1.0,
    }
    values.update(overrides)
    return CausalClaim(**values)


class StructuredLedgerTests(unittest.TestCase):
    def test_evidence_updates_status_and_preserves_structure(self) -> None:
        claim = make_claim(
            "grasp",
            confidence=0.5,
            preconditions=("gripper_aligned",),
            effects=("cup_moves_with_gripper",),
            metadata={"skill": "pick", "object_slot": 2},
        )
        ledger = CausalBeliefLedger([claim])
        evidence = Evidence(
            source="wrist_camera",
            polarity=EvidencePolarity.SUPPORTS,
            strength=0.8,
            timestamp=4,
            observation_id="frame-4",
            metadata={"detector": {"version": 1}},
        )

        invalidated = ledger.record_evidence("grasp", evidence)

        self.assertEqual(invalidated, ())
        self.assertEqual(claim.status, ClaimStatus.VERIFIED)
        self.assertAlmostEqual(claim.confidence, 0.9)
        self.assertEqual(claim.preconditions, ("gripper_aligned",))
        self.assertEqual(claim.effects, ("cup_moves_with_gripper",))
        self.assertEqual(claim.evidence[0].observation_id, "frame-4")

    def test_contradiction_refutes_claim_and_invalidates_descendants(self) -> None:
        grasp = make_claim("grasp", confidence=0.5)
        transport = make_claim("transport")
        place = make_claim("place")
        ledger = CausalBeliefLedger([grasp, transport, place])
        ledger.add_dependency("grasp", "transport")
        ledger.add_dependency("transport", "place")

        invalidated = ledger.record_evidence(
            "grasp",
            Evidence(
                source="object_tracker",
                polarity=EvidencePolarity.CONTRADICTS,
                strength=0.8,
                timestamp=5,
            ),
        )

        self.assertEqual(grasp.status, ClaimStatus.REFUTED)
        self.assertEqual(set(invalidated), {"transport", "place"})
        self.assertEqual(transport.status, ClaimStatus.INVALIDATED)
        self.assertEqual(place.status, ClaimStatus.INVALIDATED)


class DebtAndRiskTests(unittest.TestCase):
    def test_corrected_debt_is_monotonic_in_every_risk_factor(self) -> None:
        weights = DebtWeights()
        base = weights.calculate(make_claim("base"))
        riskier_claims = (
            make_claim("low-confidence", confidence=0.5),
            make_claim("uncertain", uncertainty=0.5),
            make_claim("depended-on", dependency=0.5),
            make_claim("expensive", repair_cost=0.5),
            make_claim("occluded", observability=0.5),
        )

        for claim in riskier_claims:
            with self.subTest(claim=claim.claim_id):
                self.assertGreater(weights.calculate(claim), base)

        with self.assertRaises(ValueError):
            DebtWeights(dependency=-0.1)

    def test_global_risk_uses_normalized_importance(self) -> None:
        first = make_claim("first", debt=0.2, importance=1.0)
        second = make_claim("second", debt=0.8, importance=3.0)
        ledger = CausalBeliefLedger([first, second])

        self.assertAlmostEqual(ledger.global_risk(), 0.65)
        self.assertAlmostEqual(sum(ledger.normalized_importance().values()), 1.0)

        # Adding a claim at the current weighted mean cannot change global risk.
        ledger.add_claim(make_claim("duplicate-mean", debt=0.65, importance=4.0))
        self.assertAlmostEqual(ledger.global_risk(), 0.65)


class DependencyAndRollbackTests(unittest.TestCase):
    def test_predicted_dependencies_can_be_replaced(self) -> None:
        claims = [make_claim(name) for name in ("a", "b", "c")]
        ledger = CausalBeliefLedger(claims)
        ledger.add_dependency("a", "b")
        ledger.add_dependency("b", "c")

        ledger.remove_dependency("a", "b")
        self.assertEqual(ledger.dependencies, (("b", "c"),))
        ledger.clear_dependencies(("b", "c"))
        self.assertEqual(ledger.dependencies, ())

    def test_descendant_invalidation_is_transitive_and_local(self) -> None:
        claims = [make_claim(name) for name in ("search", "grasp", "move", "open")]
        ledger = CausalBeliefLedger(claims)
        ledger.add_dependency("search", "grasp")
        ledger.add_dependency("grasp", "move")

        invalidated = ledger.refute_claim("grasp")

        self.assertEqual(invalidated, ("move",))
        self.assertEqual(ledger.get_claim("grasp").status, ClaimStatus.REFUTED)
        self.assertEqual(ledger.get_claim("move").status, ClaimStatus.INVALIDATED)
        self.assertEqual(ledger.get_claim("search").status, ClaimStatus.HYPOTHESIZED)
        self.assertEqual(ledger.get_claim("open").status, ClaimStatus.HYPOTHESIZED)
        with self.assertRaises(ValueError):
            ledger.add_dependency("move", "search")

    def test_refutation_rolls_back_only_logical_plan_state(self) -> None:
        grasp = make_claim(
            "grasp",
            confidence=0.0,
            status=ClaimStatus.REFUTED,
            rollback_checkpoint="before-grasp",
        )
        move = make_claim("move")
        ledger = CausalBeliefLedger([grasp, move], [("grasp", "move")])
        plan = LogicalPlanState(cursor=8, active_subgoal="place")
        plan.add_checkpoint(
            PlanningCheckpoint("before-grasp", cursor=3, subgoal="grasp")
        )
        physical_state = {"pose": [1.0, 2.0, 3.0], "gripper": "closed"}
        original_physical_state = copy.deepcopy(physical_state)

        decision = SelfHealingPlanner().decide(
            ledger=ledger,
            plan_state=plan,
            task_action_id="continue-place",
        )

        self.assertEqual(decision.decision_type, PlannerDecisionType.ROLLBACK)
        self.assertEqual(plan.cursor, 3)
        self.assertEqual(plan.active_subgoal, "grasp")
        self.assertEqual(move.status, ClaimStatus.INVALIDATED)
        self.assertFalse(decision.physical_state_rolled_back)
        self.assertIsNotNone(decision.rollback_event)
        self.assertFalse(decision.rollback_event.physical_state_rolled_back)
        self.assertEqual(physical_state, original_physical_state)


class PlannerDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SelfHealingPlannerConfig(
            global_risk_threshold=0.5,
            claim_debt_threshold=0.5,
            cost_weight=0.1,
            risk_weight=0.2,
            minimum_repair_score=0.0,
        )
        self.planner = SelfHealingPlanner(self.config)

    def test_safe_ledger_selects_task_action(self) -> None:
        ledger = CausalBeliefLedger([make_claim("verified")])
        decision = self.planner.decide(
            ledger,
            LogicalPlanState(cursor=1),
            task_action_id="move-to-target",
        )

        self.assertEqual(decision.decision_type, PlannerDecisionType.TASK)
        self.assertEqual(decision.action_id, "move-to-target")

    def test_calibrated_neural_debt_can_skip_runtime_recompute(self) -> None:
        claim = make_claim("calibrated", debt=0.1, uncertainty=1.0)
        planner = SelfHealingPlanner(
            SelfHealingPlannerConfig(
                global_risk_threshold=0.5,
                claim_debt_threshold=0.5,
                recompute_debt=False,
            )
        )
        decision = planner.decide(
            CausalBeliefLedger([claim]),
            LogicalPlanState(),
            task_action_id="continue",
        )
        self.assertEqual(decision.decision_type, PlannerDecisionType.TASK)

    def test_unsafe_ledger_selects_highest_utility_repair(self) -> None:
        risky = make_claim(
            "grasp",
            confidence=0.0,
            uncertainty=1.0,
            dependency=1.0,
            repair_cost=1.0,
            observability=0.0,
        )
        ledger = CausalBeliefLedger([risky])
        candidates = (
            RepairCandidate(
                "lift-and-observe",
                ("grasp",),
                expected_global_risk=0.2,
                action_cost=0.1,
                task_risk=0.1,
            ),
            RepairCandidate(
                "full-regrasp",
                ("grasp",),
                expected_global_risk=0.05,
                action_cost=10.0,
                task_risk=0.2,
            ),
        )

        decision = self.planner.decide(
            ledger,
            LogicalPlanState(cursor=5),
            task_action_id="transport",
            repair_candidates=candidates,
        )

        self.assertEqual(decision.decision_type, PlannerDecisionType.REPAIR)
        self.assertEqual(decision.action_id, "lift-and-observe")
        self.assertGreater(decision.repair_score, 0.0)

    def test_unsafe_ledger_without_useful_repair_requests_local_rollback(self) -> None:
        risky = make_claim(
            "grasp",
            confidence=0.0,
            uncertainty=1.0,
            dependency=1.0,
            repair_cost=1.0,
            observability=0.0,
            rollback_checkpoint="grasp-stage",
        )
        ledger = CausalBeliefLedger([risky])
        plan = LogicalPlanState(cursor=6)
        plan.add_checkpoint(PlanningCheckpoint("grasp-stage", cursor=2))
        bad_candidate = RepairCandidate(
            "dangerous-repair",
            ("grasp",),
            expected_global_risk=0.9,
            action_cost=10.0,
            task_risk=1.0,
        )

        decision = self.planner.decide(
            ledger,
            plan,
            task_action_id="transport",
            repair_candidates=(bad_candidate,),
        )

        self.assertEqual(decision.decision_type, PlannerDecisionType.ROLLBACK)
        self.assertEqual(plan.cursor, 2)
        self.assertIsNone(decision.action_id)
        self.assertFalse(decision.physical_state_rolled_back)

    def test_policy_prior_is_separate_from_physical_action_cost(self) -> None:
        risky = make_claim(
            "grasp",
            confidence=0.0,
            uncertainty=1.0,
            dependency=1.0,
            observability=0.0,
        )
        planner = SelfHealingPlanner(
            SelfHealingPlannerConfig(
                global_risk_threshold=0.5,
                claim_debt_threshold=0.5,
                cost_weight=0.1,
                risk_weight=0.2,
                policy_weight=0.1,
            )
        )
        candidates = (
            RepairCandidate(
                "unlikely",
                ("grasp",),
                expected_global_risk=0.2,
                action_cost=0.1,
                policy_log_probability=-3.0,
            ),
            RepairCandidate(
                "likely",
                ("grasp",),
                expected_global_risk=0.2,
                action_cost=0.1,
                policy_log_probability=-0.1,
            ),
        )
        decision = planner.decide(
            CausalBeliefLedger([risky]),
            LogicalPlanState(),
            task_action_id="continue",
            repair_candidates=candidates,
        )
        self.assertEqual(decision.action_id, "likely")
        self.assertEqual(candidates[1].action_cost, 0.1)


class SerializationTests(unittest.TestCase):
    def test_runtime_json_round_trip_preserves_structured_state(self) -> None:
        grasp = make_claim(
            "grasp",
            confidence=0.7,
            status=ClaimStatus.HYPOTHESIZED,
            rollback_checkpoint="before-grasp",
            metadata={"labels": ["contact", "support"]},
        )
        grasp.evidence.append(
            Evidence(
                source="tactile",
                polarity=EvidencePolarity.SUPPORTS,
                strength=0.4,
                timestamp=3,
            )
        )
        move = make_claim("move", importance=2.0)
        ledger = CausalBeliefLedger([grasp, move], [("grasp", "move")])
        ledger.recompute_debts(DebtWeights())
        plan = LogicalPlanState(cursor=7, active_subgoal="place")
        plan.add_checkpoint(
            PlanningCheckpoint(
                "before-grasp", cursor=2, subgoal="grasp", metadata={"phase": 1}
            )
        )
        plan.rollback_to(
            "before-grasp",
            trigger_claim_ids=("grasp",),
            invalidated_claim_ids=("move",),
            reason="test logical rollback",
        )
        state = LedgerRuntimeState(ledger=ledger, plan_state=plan)

        payload = state.to_json(indent=2)
        restored = LedgerRuntimeState.from_json(payload)

        self.assertEqual(restored.to_dict(), state.to_dict())
        self.assertFalse(json.loads(payload)["contains_physical_state"])

        unsafe_payload = json.loads(payload)
        unsafe_payload["contains_physical_state"] = True
        with self.assertRaises(ValueError):
            LedgerRuntimeState.from_json(json.dumps(unsafe_payload))


if __name__ == "__main__":
    unittest.main()
