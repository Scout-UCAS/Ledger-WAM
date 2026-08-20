import unittest
import tempfile
from pathlib import Path
from unittest import mock

import torch

from fastwam.models.ledger import (
    CausalBeliefLedger,
    LedgerLoss,
    LedgerLossConfig,
    SelfHealingPlanner,
)
from fastwam.models.wan22.ledger_wam import LedgerWAM
from fastwam.models.wan22.fastwam import FastWAM


class _FakeActionExpert(torch.nn.Module):
    hidden_dim = 16
    action_dim = 3


class _FakeVideoExpert(torch.nn.Module):
    video_attention_mask_mode = "first_frame_causal"


class _FakeVAE(torch.nn.Module):
    pass


class LedgerModuleTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.module = CausalBeliefLedger(
            input_dim=16,
            action_dim=3,
            action_horizon=4,
            hidden_dim=16,
            num_attention_heads=4,
            max_rollback_steps=4,
            effect_dim=8,
            dropout=0.0,
        )
        self.tokens = torch.randn(2, 4, 16, requires_grad=True)
        self.action = torch.randn(2, 4, 3)

    def test_forward_recurrence_and_summary(self):
        first = self.module(self.tokens, self.action)
        second = self.module(self.tokens, self.action, first.next_state.detached())
        self.assertEqual(tuple(first.debt.shape), (2, 8))
        self.assertEqual(tuple(first.rollback_logits.shape), (2, 8, 4))
        self.assertEqual(tuple(first.entity_logits.shape), (2, 8, 8))
        self.assertEqual(tuple(first.repair_action.shape), (2, 4, 4, 3))
        self.assertEqual(second.next_state.step, 1)
        self.assertTrue(torch.all(first.debt >= 0.0))
        self.assertTrue(torch.all(first.debt <= 1.0))
        summary = first.summary(
            self.module.claim_names,
            self.module.relation_names,
            self.module.entity_names,
        )
        self.assertEqual(len(summary["claims"]), 8)
        self.assertIn("entities", summary["claims"][0])

    def test_all_losses_backpropagate(self):
        output = self.module(self.tokens, self.action)
        counterfactual = self.module(self.tokens, -self.action)
        sample = {
            "ledger_claim_labels": torch.randint(0, 2, (2, 8)).float(),
            "ledger_claim_mask": torch.ones(2, 8),
            "ledger_debt_targets": torch.rand(2, 8),
            "ledger_rollback_targets": torch.randint(0, 4, (2, 8)),
            "ledger_relation_targets": torch.randint(0, 8, (2, 8)),
            "ledger_entity_targets": torch.randint(0, 2, (2, 8, 8)).float(),
            "ledger_precondition_targets": torch.randint(0, 9, (2, 8)),
            "ledger_effect_targets": torch.randint(0, 9, (2, 8)),
            "ledger_repair_action": torch.randn(2, 4, 3),
            "ledger_repair_type": torch.tensor([0, 2]),
            "ledger_repair_debt_reduction": torch.rand(2, 4),
            "ledger_repair_risk": torch.rand(2, 4),
        }
        loss, metrics = LedgerLoss(LedgerLossConfig())(output, sample, counterfactual)
        loss.backward()
        self.assertGreater(float(loss.detach()), 0.0)
        self.assertIn("loss_claim", metrics)
        self.assertIsNotNone(self.module.claim_queries.grad)

    def test_planner_gates_and_localizes(self):
        output = self.module(self.tokens, self.action)
        planner = SelfHealingPlanner(
            global_risk_threshold=0.6,
            claim_debt_threshold=0.65,
            importance_threshold=0.35,
            task_execution_horizon=3,
            verification_horizon=1,
        )
        with torch.no_grad():
            output.global_risk[0] = 0.2
        task = planner.decide(self.action[0], output)
        self.assertEqual(task.mode, "task")
        self.assertEqual(task.execution_horizon, 3)

        with torch.no_grad():
            output.global_risk[0] = 0.9
            output.debt[0].fill_(0.1)
            output.importance[0].fill_(0.1)
            output.debt[0, 2] = 0.95
            output.importance[0, 2] = 0.9
            output.confidence[0, 2] = 0.1
            output.rollback_logits[0, 2].fill_(-1.0)
            output.rollback_logits[0, 2, 3] = 5.0
            output.expected_debt_reduction[0] = torch.tensor([0.9, 0.1, 0.1, 0.1])
            output.repair_risk[0].zero_()
        repair = planner.decide(self.action[0], output)
        self.assertEqual(repair.mode, "rollback")
        self.assertEqual(repair.target_claim, 2)
        self.assertEqual(repair.rollback_step, 3)
        self.assertEqual(repair.execution_horizon, 1)

    def test_ledger_wam_checkpoint_and_trainable_heads(self):
        model = LedgerWAM(
            video_expert=_FakeVideoExpert(),
            action_expert=_FakeActionExpert(),
            mot=torch.nn.Linear(2, 2),
            vae=_FakeVAE(),
            text_dim=8,
            proprio_dim=None,
            ledger_config={
                "action_horizon": 4,
                "hidden_dim": 16,
                "num_attention_heads": 4,
                "max_rollback_steps": 4,
                "effect_dim": 8,
            },
        )
        model.set_train_mode_for_training()
        self.assertTrue(all(parameter.requires_grad for parameter in model.ledger.parameters()))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            model.save_checkpoint(path)
            before = model.ledger.claim_queries.detach().clone()
            with torch.no_grad():
                model.ledger.claim_queries.add_(1.0)
            model.load_checkpoint(path)
            self.assertTrue(torch.equal(before, model.ledger.claim_queries.detach()))
        base_prediction = {
            "action": torch.randn(4, 3),
            "action_features": torch.randn(1, 4, 16),
        }
        with mock.patch.object(FastWAM, "infer_action", return_value=base_prediction):
            prediction = model.infer_action()
        self.assertIn("ledger", prediction)
        self.assertIn("planner", prediction)
        self.assertIn("execution_horizon", prediction)
        self.assertNotIn("action_features", prediction)


if __name__ == "__main__":
    unittest.main()
