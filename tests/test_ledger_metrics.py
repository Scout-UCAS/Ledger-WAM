"""Unit tests for Ledger-WAM evaluation metrics."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

import torch


# Load by path so this CPU-only test does not import optional Wan dependencies.
MODULE_PATH = Path(__file__).parents[1] / "wan_va" / "ledger" / "metrics.py"
SPEC = importlib.util.spec_from_file_location("ledger_metrics", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ledger_metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ledger_metrics)


class ClaimRootCauseMetricTests(unittest.TestCase):
    def test_precision_recall_f1_and_mask(self) -> None:
        result = ledger_metrics.claim_root_cause_metrics(
            predictions=torch.tensor([0.9, 0.8, 0.7, float("nan")]),
            targets=torch.tensor([1, 0, 1, -100]),
            mask=torch.tensor([1, 1, 1, 0], dtype=torch.bool),
        )

        self.assertAlmostEqual(result["claim_root_cause_precision"], 2.0 / 3.0)
        self.assertEqual(result["claim_root_cause_recall"], 1.0)
        self.assertAlmostEqual(result["claim_root_cause_f1"], 0.8)
        self.assertEqual(result["claim_root_cause_true_positives"], 2)
        self.assertEqual(result["claim_root_cause_count"], 3)
        json.dumps(result, allow_nan=False)

    def test_empty_and_all_negative_inputs_are_defined(self) -> None:
        empty = ledger_metrics.claim_root_cause_metrics([], [])
        self.assertEqual(empty["claim_root_cause_count"], 0)
        self.assertEqual(empty["claim_root_cause_f1"], 0.0)

        all_negative = ledger_metrics.claim_root_cause_metrics([0.1, 0.2], [0, 0])
        self.assertEqual(all_negative["claim_root_cause_precision"], 0.0)
        self.assertEqual(all_negative["claim_root_cause_recall"], 0.0)


class TopKMetricTests(unittest.TestCase):
    def test_multihot_targets_and_candidate_mask(self) -> None:
        scores = torch.tensor([[0.1, 0.9, 0.8], [0.9, 0.2, 0.1], [100.0, 0.8, 0.7]])
        targets = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=torch.long)
        candidate_mask = torch.tensor(
            [[1, 1, 1], [1, 1, 1], [0, 1, 1]], dtype=torch.bool
        )

        result = ledger_metrics.top_k_accuracy(
            scores, targets, k=(1, 2), mask=candidate_mask
        )

        self.assertAlmostEqual(result["top_1_accuracy"], 2.0 / 3.0)
        self.assertEqual(result["top_2_accuracy"], 1.0)
        self.assertEqual(result["top_k_count"], 3)
        json.dumps(result, allow_nan=False)

    def test_index_targets_sample_mask_and_empty_batch(self) -> None:
        result = ledger_metrics.topk_accuracy(
            [[0.9, 0.1], [0.2, 0.8], [float("nan"), float("nan")]],
            [0, 0, -100],
            k=1,
            mask=[1, 1, 0],
        )
        self.assertEqual(result["top_1_accuracy"], 0.5)
        self.assertEqual(result["top_k_count"], 2)

        empty = ledger_metrics.top_k_accuracy(
            torch.empty(0, 3), torch.empty(0, dtype=torch.long), k=(1, 5)
        )
        self.assertEqual(empty["top_1_accuracy"], 0.0)
        self.assertEqual(empty["top_5_accuracy"], 0.0)
        self.assertEqual(empty["top_k_count"], 0)


class DebtCalibrationMetricTests(unittest.TestCase):
    def test_brier_and_ece(self) -> None:
        result = ledger_metrics.debt_calibration_metrics(
            [0.1, 0.8, float("nan")],
            [0, 1, -100],
            mask=[1, 1, 0],
            num_bins=2,
        )

        self.assertAlmostEqual(result["debt_brier"], 0.025)
        self.assertAlmostEqual(result["debt_ece"], 0.15)
        self.assertEqual(result["debt_count"], 2)
        json.dumps(result, allow_nan=False)

    def test_empty_calibration(self) -> None:
        result = ledger_metrics.debt_calibration_metrics([], [], num_bins=5)
        self.assertEqual(result, {"debt_brier": 0.0, "debt_ece": 0.0, "debt_count": 0})


class RollbackAndRepairMetricTests(unittest.TestCase):
    def test_rollback_logits_accuracy_and_distance(self) -> None:
        logits = torch.tensor(
            [
                [0.0, 0.0, 2.0],
                [2.0, 0.0, 0.0],
                [float("nan"), float("nan"), float("nan")],
            ]
        )
        result = ledger_metrics.rollback_metrics(logits, [2, 1, -100], mask=[1, 1, 0])

        self.assertEqual(result["rollback_accuracy"], 0.5)
        self.assertEqual(result["rollback_mean_distance"], 0.5)
        self.assertEqual(result["rollback_count"], 2)

    def test_repair_metrics_use_signed_debt_drop(self) -> None:
        result = ledger_metrics.repair_metrics(
            successes=[1, 0, -100],
            action_counts=[2, 4, float("nan")],
            debt_before=[0.8, 0.6, float("nan")],
            debt_after=[0.2, 0.5, float("nan")],
            mask=[1, 1, 0],
        )

        self.assertEqual(result["repair_success_rate"], 0.5)
        self.assertEqual(result["repair_mean_actions"], 3.0)
        self.assertAlmostEqual(result["repair_mean_debt_drop"], 0.35)
        self.assertEqual(result["repair_count"], 2)
        json.dumps(result, allow_nan=False)

    def test_empty_rollback_and_repair_metrics(self) -> None:
        rollback = ledger_metrics.rollback_metrics([], [])
        repair = ledger_metrics.repair_metrics([], [], [], [])
        self.assertEqual(rollback["rollback_count"], 0)
        self.assertEqual(repair["repair_count"], 0)
        self.assertEqual(repair["repair_mean_debt_drop"], 0.0)


class LocalRollbackAndAggregateTests(unittest.TestCase):
    def test_local_rollback_ratio_uses_only_rollback_events(self) -> None:
        result = ledger_metrics.local_rollback_ratio(
            is_local=[1, 0, 1, 1],
            is_rollback=[1, 1, 0, 1],
            mask=[1, 1, 1, 0],
        )

        self.assertEqual(result["local_rollback_ratio"], 0.5)
        self.assertEqual(result["local_rollback_count"], 1)
        self.assertEqual(result["total_rollback_count"], 2)

        empty = ledger_metrics.local_rollback_metrics([], [])
        self.assertEqual(empty["local_rollback_ratio"], 0.0)

    def test_aggregate_skips_missing_groups_and_includes_empty_groups(self) -> None:
        self.assertEqual(ledger_metrics.compute_ledger_metrics(), {})

        result = ledger_metrics.compute_ledger_metrics(
            claim_predictions=[],
            claim_targets=[],
            debt_predictions=[0.0, 1.0],
            debt_targets=[0, 1],
            rollback_predictions=[],
            rollback_targets=[],
            local_rollback_flags=[],
        )
        self.assertEqual(result["claim_root_cause_count"], 0)
        self.assertEqual(result["debt_brier"], 0.0)
        self.assertEqual(result["rollback_count"], 0)
        self.assertEqual(result["total_rollback_count"], 0)
        json.dumps(result, allow_nan=False)

    def test_aggregate_rejects_partial_groups(self) -> None:
        with self.assertRaisesRegex(ValueError, "both predictions and targets"):
            ledger_metrics.compute_ledger_metrics(claim_predictions=[0.5])
        with self.assertRaisesRegex(ValueError, "repair metrics require"):
            ledger_metrics.compute_ledger_metrics(repair_successes=[1])


if __name__ == "__main__":
    unittest.main()
