import unittest

import numpy as np
import torch

from experiments.common import SimulatorEventParser, apply_cartesian_repair, apply_joint_repair
from fastwam.datasets.ledger_evidence import WeakCausalAnnotator, build_ledger_evidence
from fastwam.evaluation import object_identity_consistency, world_prediction_mae
from fastwam.models.ledger import (
    CausalBeliefLedger,
    CausalTaskGraph,
    CausalWorldPredictor,
    SelfHealingPlanner,
)
from fastwam.models.wan22.ledger_wam import LedgerWAM
from scripts.compile_simulator_events import compile_record


class FullLedgerMethodTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.ledger = CausalBeliefLedger(
            input_dim=16,
            visual_input_dim=12,
            action_dim=7,
            action_horizon=8,
            hidden_dim=16,
            num_attention_heads=4,
            num_object_slots=5,
            dependency_horizon=6,
            max_rollback_steps=4,
            dropout=0.0,
        )
        self.tokens = torch.randn(2, 8, 16)
        self.visual = torch.randn(2, 12, 12)
        self.action = torch.randn(2, 8, 7)
        self.evidence = torch.randn(2, 4, 16)

    def test_temporal_slots_evidence_and_k_step_dependency(self):
        outputs = self.ledger.forward_sequence(
            self.tokens,
            self.action,
            visual_tokens=self.visual,
            sensor_evidence=self.evidence,
            num_steps=4,
        )
        self.assertEqual(len(outputs), 4)
        self.assertEqual(outputs[-1].next_state.step, 3)
        self.assertEqual(tuple(outputs[-1].object_slots.shape), (2, 5, 16))
        self.assertEqual(tuple(outputs[-1].dependency_by_step.shape), (2, 8, 6))
        self.assertGreater(object_identity_consistency(outputs[-1].object_assignment), 0.0)

    def test_candidate_world_model_drives_planner(self):
        output = self.ledger(
            self.tokens,
            self.action,
            visual_tokens=self.visual,
            sensor_evidence=self.evidence[:, -1],
        )
        planner = SelfHealingPlanner(
            global_risk_threshold=0.1,
            claim_debt_threshold=0.0,
            importance_threshold=0.0,
            repair_horizons=(1, 1, 4, 6),
        )
        candidates = planner.build_repair_candidates(self.action)
        predictor = CausalWorldPredictor(16, 7, 8, 5)
        world = predictor(output, candidates)
        self.assertEqual(tuple(world.debt.shape), (2, 4, 8))
        self.assertEqual(tuple(world.predicted_observation.shape), (2, 4, 16))
        with torch.no_grad():
            output.global_risk[0] = 0.9
            output.debt[0].fill_(0.9)
            output.importance[0].fill_(0.9)
            world.expected_debt_reduction[0] = torch.tensor([0.1, 0.8, 0.2, 0.1])
            world.predicted_failure_risk[0].zero_()
        decision = planner.decide(
            self.action[0], output, world_prediction=world, repair_candidates=candidates
        )
        self.assertEqual(decision.repair_name, "hold")

    def test_task_graph_invalidates_descendants(self):
        graph = CausalTaskGraph(self.ledger.claim_names)
        graph.update(torch.ones(8), step=5)
        rollback = graph.rollback(0, requested_lookback=2, current_step=6)
        self.assertIn(1, rollback.invalidated_claims)
        self.assertIn(7, rollback.invalidated_claims)
        self.assertFalse(bool(graph.verified[0]))

    def test_weak_evidence_and_simulator_compiler(self):
        video = torch.randn(3, 9, 16, 16)
        proprio = torch.randn(8, 7)
        action = torch.randn(8, 7)
        evidence = build_ledger_evidence(video, proprio, action, steps=4)
        self.assertEqual(tuple(evidence.shape), (4, 16))
        annotation = WeakCausalAnnotator(steps=4).annotate(video, proprio, action)
        self.assertEqual(tuple(annotation["ledger_claim_labels_sequence"].shape), (4, 8))

        parser = SimulatorEventParser()
        timeline = []
        for step in range(4):
            _, event = parser.observe(
                {
                    "robot0_eef_pos": [0.0, 0.0, 0.1],
                    "target_object_pos": [0.01 * step, 0.0, 0.1],
                    "container_pos": [0.2, 0.0, 0.1],
                },
                proposed_action=np.zeros(7),
            )
            timeline.append(event)
        compiled = compile_record({"idx": 3, "timeline": timeline}, steps=4)
        self.assertEqual(len(compiled["claim_labels_sequence"]), 4)
        self.assertEqual(len(compiled["evidence"][0]), 16)

    def test_environment_repairs_and_world_metric(self):
        cartesian = np.ones((6, 7), dtype=np.float32) * 0.2
        self.assertTrue(np.allclose(apply_cartesian_repair(cartesian, "hold")[:, :6], 0.0))
        joints = np.ones((6, 14), dtype=np.float32)
        current = np.zeros(14, dtype=np.float32)
        self.assertTrue(np.allclose(apply_joint_repair(joints, "hold", current_qpos=current), 0.0))
        self.assertAlmostEqual(world_prediction_mae([0.1, 0.8], [0.0, 1.0]), 0.15, places=6)

    def test_integrated_temporal_world_loss_backward(self):
        class ActionExpert(torch.nn.Module):
            hidden_dim = 16
            action_dim = 7

        class VideoExpert(torch.nn.Module):
            hidden_dim = 12
            video_attention_mask_mode = "first_frame_causal"

        model = LedgerWAM(
            video_expert=VideoExpert(),
            action_expert=ActionExpert(),
            mot=torch.nn.Linear(2, 2),
            vae=torch.nn.Identity(),
            text_dim=8,
            ledger_config={
                "action_horizon": 8,
                "hidden_dim": 16,
                "num_attention_heads": 4,
                "num_object_slots": 5,
                "dependency_horizon": 6,
                "max_rollback_steps": 4,
                "temporal_unroll_steps": 4,
                "dropout": 0.0,
            },
        )
        sample = {
            "ledger_evidence": self.evidence,
            "ledger_claim_labels": torch.rand(2, 8),
            "ledger_claim_mask": torch.ones(2, 8),
            "ledger_debt_targets": torch.rand(2, 8),
            "ledger_repair_debt_reduction": torch.rand(2, 4),
            "ledger_repair_risk": torch.rand(2, 4),
            "ledger_repair_value_mask": torch.ones(2),
            "ledger_claim_labels_sequence": torch.rand(2, 4, 8),
            "ledger_claim_mask_sequence": torch.ones(2, 4, 8),
        }
        loss, metrics = model.auxiliary_training_loss(
            sample,
            state_tokens=self.tokens,
            video_state_tokens=self.visual,
            action=self.action,
        )
        loss.backward()
        self.assertIn("loss_world", metrics)
        self.assertIsNotNone(model.world_predictor.component_head.weight.grad)


if __name__ == "__main__":
    unittest.main()
