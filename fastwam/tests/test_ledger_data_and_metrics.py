import json
import tempfile
import unittest
from pathlib import Path

from fastwam.datasets.ledger_annotations import LedgerAnnotationStore
from fastwam.evaluation import (
    debt_calibration_error,
    failure_localization_accuracy,
    repair_efficiency,
    rollback_accuracy,
)
from scripts.build_ledger_annotations import compile_record


class LedgerDataAndMetricsTest(unittest.TestCase):
    def test_sparse_event_compilation_and_loading(self):
        record = compile_record(
            {
                "idx": 12,
                "claims": [
                    {
                        "name": "grasped",
                        "truth": True,
                        "confidence": 0.8,
                        "dependency": 0.9,
                        "entities": ["end_effector", "target_object"],
                        "relation": "contact",
                        "precondition": "contact",
                        "effect": "supported",
                        "rollback_step": 2,
                    }
                ],
            },
            [1.0] * 5,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "annotations.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            store = LedgerAnnotationStore(str(path), num_claims=8, num_entities=8)
            tensors = store.get(12)
        self.assertEqual(tuple(tensors["ledger_claim_labels"].shape), (8,))
        self.assertEqual(tuple(tensors["ledger_entity_targets"].shape), (8, 8))
        self.assertEqual(int(tensors["ledger_rollback_targets"][1]), 2)

    def test_metrics(self):
        self.assertEqual(
            failure_localization_accuracy([[0.1, 0.9], [0.8, 0.2]], [1, 0]),
            1.0,
        )
        self.assertEqual(rollback_accuracy([[0.1, 0.9], [0.8, 0.2]], [1, 0]), 1.0)
        self.assertLess(debt_calibration_error([0.1, 0.9], [0.0, 1.0], bins=2), 0.11)
        self.assertAlmostEqual(repair_efficiency([0.8], [0.4], [0.2]), 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
