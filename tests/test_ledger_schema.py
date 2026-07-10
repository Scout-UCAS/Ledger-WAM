"""Unit tests for the dependency-free Ledger-WAM sidecar schema."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, relative_path):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ledger_schema = _load_module(
    "ledger_schema_under_test", "wan_va/dataset/ledger_schema.py"
)
ledger_config = _load_module(
    "ledger_config_under_test", "wan_va/configs/ledger_config.py"
)


class LedgerSchemaTest(unittest.TestCase):
    def test_jsonl_record_is_padded_and_masked(self):
        record = {
            "key": "3:10:20",
            "claims": [
                {
                    "claim": 1,
                    "claim_type": 4,
                    "subject_id": 7,
                    "object_id": 9,
                    "relation": 2,
                    "dependency": 0.8,
                    "debt": 0.7,
                    "rollback": 1,
                    "repair_action": 6,
                    "post_repair_debt": 0.2,
                },
                {"truth": 0, "relation_id": 5, "debt_target": 0.1},
            ],
            "dependency_matrix": [[0.0, 1.0], [0.25, 0.0]],
            "counterfactual_actions": [
                {"action": [0.1, 0.2, 0.3, 0.4], "delta": [-0.5, 0.3]}
            ],
        }
        spec = ledger_schema.LedgerTensorSpec(
            max_claims=3,
            max_counterfactuals=2,
            action_dim=4,
            strict=True,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ledger.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            store = ledger_schema.LedgerAnnotationStore(path, spec)
            tensors = store.tensors_for(3, 10, 20)

        self.assertTrue(tensors["ledger_available"].item())
        self.assertEqual(tensors["ledger_claim_mask"].tolist(), [True, True, False])
        self.assertEqual(tensors["ledger_claim_labels"][:2].tolist(), [1.0, 0.0])
        self.assertEqual(tensors["ledger_relation_labels"][:2].tolist(), [2, 5])
        self.assertEqual(tensors["ledger_rollback_labels"][0].item(), 1)
        self.assertEqual(tensors["ledger_repair_action_labels"][0].item(), 6)
        self.assertAlmostEqual(
            tensors["ledger_post_repair_debt_labels"][0].item(), 0.2, places=6
        )
        self.assertEqual(tensors["ledger_debt_mask"].tolist(), [True, True, False])
        self.assertEqual(tensors["ledger_rollback_mask"].tolist(), [True, False, False])
        self.assertEqual(
            tensors["ledger_dependency_matrix_mask"][:2, :2].tolist(),
            [[True, True], [True, True]],
        )
        self.assertEqual(tensors["ledger_counterfactual_mask"].tolist(), [True, False])
        self.assertTrue(tensors["ledger_counterfactual_action_mask"][0].all().item())
        self.assertEqual(
            tensors["ledger_counterfactual_delta_mask"][0].tolist(),
            [True, True, False],
        )
        self.assertEqual(tensors["ledger_claim_labels"][2].item(), -100.0)
        self.assertEqual(tensors["ledger_relation_labels"][2].item(), -100)

    def test_json_mapping_uses_segment_keys(self):
        payload = {
            "0:0:5": {"claims": [{"claim": 1, "relation": 3}]},
            "1:5:9": {"claims": [{"claim": 0, "debt": 0.9}]},
        }
        spec = ledger_schema.LedgerTensorSpec(max_claims=2, action_dim=2)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ledger.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            store = ledger_schema.LedgerAnnotationStore(path, spec)
            first = store.tensors_for(0, 0, 5)
            second = store.tensors_for(1, 5, 9)

        self.assertEqual(first["ledger_relation_labels"][0].item(), 3)
        self.assertAlmostEqual(second["ledger_debt_labels"][0].item(), 0.9)

    def test_non_strict_mode_truncates_and_returns_empty_missing_record(self):
        payload = [
            {
                "episode_index": 2,
                "start_frame": 0,
                "end_frame": 8,
                "claims": [
                    {"claim": 1, "relation": 1},
                    {"claim": 2.0, "relation": "not-an-id"},
                    {"claim": 0, "relation": 3},
                ],
                "counterfactual_actions": [[1.0, 2.0, 3.0]],
                "counterfactual_deltas": [[-0.2, 0.1, 0.9]],
            }
        ]
        spec = ledger_schema.LedgerTensorSpec(
            max_claims=2,
            max_counterfactuals=1,
            action_dim=2,
            strict=False,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ledger.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            store = ledger_schema.LedgerAnnotationStore(path, spec)
            present = store.tensors_for(2, 0, 8)
            missing = store.tensors_for(99, 0, 1)

        # Invalid probability-like labels are clamped; invalid categories remain ignored.
        self.assertEqual(present["ledger_claim_labels"].tolist(), [1.0, 1.0])
        self.assertFalse(present["ledger_relation_mask"][1].item())
        self.assertEqual(
            present["ledger_counterfactual_actions"][0].tolist(), [1.0, 2.0]
        )
        self.assertFalse(missing["ledger_available"].item())
        self.assertFalse(missing["ledger_claim_mask"].any().item())
        self.assertTrue((missing["ledger_debt_labels"] == -100.0).all().item())

    def test_strict_mode_rejects_missing_and_oversized_records(self):
        spec = ledger_schema.LedgerTensorSpec(
            max_claims=1,
            max_counterfactuals=1,
            action_dim=2,
            strict=True,
        )
        payload = {
            "key": "4:0:2",
            "claims": [{"claim": 1}, {"claim": 0}],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ledger.jsonl"
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            store = ledger_schema.LedgerAnnotationStore(path, spec)
            with self.assertRaises(ledger_schema.LedgerSchemaError):
                store.tensors_for(4, 0, 2)
            with self.assertRaises(KeyError):
                store.tensors_for(4, 3, 5)

        with self.assertRaises(FileNotFoundError):
            ledger_schema.LedgerAnnotationStore("does-not-exist.jsonl", spec)

    def test_sparse_dependency_edges_and_delta_only_counterfactual(self):
        spec = ledger_schema.LedgerTensorSpec(
            max_claims=3, max_counterfactuals=2, action_dim=2, strict=True
        )
        tensors = ledger_schema.tensorize_ledger_record(
            {
                "claims": [{"claim": 1}, {"claim": 1}],
                "dependency_edges": [{"source": 0, "target": 1, "weight": 0.75}],
                "counterfactual_delta": [[-0.1, 0.2]],
            },
            spec,
        )
        self.assertAlmostEqual(tensors["ledger_dependency_matrix"][0, 1].item(), 0.75)
        self.assertEqual(
            tensors["ledger_dependency_matrix"][:2, :2].tolist(),
            [[0.0, 0.75], [0.0, 0.0]],
        )
        self.assertEqual(
            tensors["ledger_dependency_matrix_mask"].tolist(),
            [
                [True, True, False],
                [True, True, False],
                [False, False, False],
            ],
        )
        self.assertTrue(tensors["ledger_counterfactual_mask"][0].item())
        self.assertFalse(tensors["ledger_counterfactual_action_mask"][0].any().item())
        self.assertEqual(
            tensors["ledger_counterfactual_delta_mask"][0].tolist(),
            [True, True, False],
        )

    def test_repository_example_sidecar_is_strictly_tensorizable(self):
        path = REPO_ROOT / "example" / "ledger_annotations.example.jsonl"
        record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(record["key"], "0:0:8")
        self.assertEqual(
            (
                record["episode_index"],
                record["start_frame"],
                record["end_frame"],
            ),
            (0, 0, 8),
        )

        spec = ledger_schema.LedgerTensorSpec(
            max_claims=16,
            max_counterfactuals=4,
            action_dim=30,
            strict=True,
        )
        store = ledger_schema.LedgerAnnotationStore(path, spec, strict=True)
        tensors = store.tensors_for(0, 0, 8)

        self.assertTrue(tensors["ledger_available"].item())
        self.assertEqual(tensors["ledger_claim_mask"].sum().item(), 2)
        self.assertEqual(
            tensors["ledger_counterfactual_action_mask"][0].sum().item(), 2
        )


class LedgerConfigTest(unittest.TestCase):
    def test_defaults_are_isolated_and_can_be_applied(self):
        first = ledger_config.get_ledger_config(enabled=True)
        second = ledger_config.get_ledger_config()
        first["ledger_loss_weights"]["claim"] = 99.0

        self.assertTrue(first["ledger_enabled"])
        self.assertFalse(second["ledger_enabled"])
        self.assertEqual(second["ledger_loss_weights"]["claim"], 1.0)
        self.assertIn(
            "local_regrasp", {item["name"] for item in second["ledger_repair_catalog"]}
        )

        target = {}
        returned = ledger_config.apply_ledger_defaults(
            target, enabled=True, ledger_max_claims=8
        )
        self.assertIs(returned, target)
        self.assertEqual(target["ledger_max_claims"], 8)
        self.assertTrue(target["ledger_enabled"])

        with self.assertRaises(KeyError):
            ledger_config.get_ledger_config(unknown_option=True)


if __name__ == "__main__":
    unittest.main()
