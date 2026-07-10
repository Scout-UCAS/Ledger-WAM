"""Pure-logic tests for repair action handoff and acknowledgement."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import ModuleType

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if "wan_va" not in sys.modules:
    wan_va_package = ModuleType("wan_va")
    wan_va_package.__path__ = [str(REPOSITORY_ROOT / "wan_va")]
    sys.modules["wan_va"] = wan_va_package

from wan_va.ledger import (  # noqa: E402
    RepairExecutionTracker,
    repair_execution_ack_required,
    validate_repair_action_chunk,
    validate_repair_catalog,
)


class RepairExecutionTrackerTests(unittest.TestCase):
    def test_issue_does_not_claim_execution_before_ack(self) -> None:
        tracker = RepairExecutionTracker()
        issued = tracker.issue(
            "local_regrasp",
            ("grasp", "transport"),
            source="supplied_chunk",
            issued_at=8,
        )

        self.assertEqual(issued.action_id, "local_regrasp")
        self.assertIs(tracker.outstanding, issued)
        self.assertIsNone(tracker.acknowledge())
        self.assertIs(tracker.outstanding, issued)

    def test_success_ack_releases_claims_for_post_repair_verification(self) -> None:
        tracker = RepairExecutionTracker()
        issued = tracker.issue(
            "lift_test",
            ("grasp",),
            source="prompt_recovery",
            issued_at=3,
        )

        acknowledgement = tracker.acknowledge(
            {
                "action_id": "lift_test",
                "execution_id": issued.execution_id,
                "success": True,
            }
        )

        self.assertTrue(acknowledgement.success)
        self.assertEqual(acknowledgement.target_claim_ids, ("grasp",))
        self.assertFalse(acknowledgement.implicit)
        self.assertIsNone(tracker.outstanding)

    def test_failed_ack_does_not_report_success(self) -> None:
        tracker = RepairExecutionTracker()
        issued = tracker.issue(
            "realign", ("alignment",), source="supplied_chunk", issued_at=5
        )

        acknowledgement = tracker.acknowledge(
            {
                "action_id": "realign",
                "execution_id": issued.execution_id,
                "success": False,
            }
        )

        self.assertFalse(acknowledgement.success)
        self.assertIsNone(tracker.outstanding)

    def test_legacy_state_upload_can_acknowledge_implicitly(self) -> None:
        tracker = RepairExecutionTracker()
        tracker.issue(
            "short_retreat", ("collision_free",), source="supplied_chunk", issued_at=2
        )

        acknowledgement = tracker.acknowledge(implicit_success=True)

        self.assertTrue(acknowledgement.success)
        self.assertTrue(acknowledgement.implicit)
        self.assertEqual(acknowledgement.action_id, "short_retreat")

    def test_mismatched_or_malformed_ack_is_rejected_without_consuming(self) -> None:
        tracker = RepairExecutionTracker()
        issued = tracker.issue(
            "tactile_check", ("contact",), source="supplied_chunk", issued_at=1
        )

        with self.assertRaisesRegex(ValueError, "does not match"):
            tracker.acknowledge(
                {
                    "action_id": "lift_test",
                    "execution_id": issued.execution_id,
                    "success": True,
                }
            )
        self.assertIsNotNone(tracker.outstanding)
        with self.assertRaisesRegex(TypeError, "boolean"):
            tracker.acknowledge(
                {
                    "action_id": "tactile_check",
                    "execution_id": issued.execution_id,
                    "success": 1,
                }
            )
        self.assertIsNotNone(tracker.outstanding)

    def test_replayed_ack_cannot_confirm_a_later_same_named_action(self) -> None:
        tracker = RepairExecutionTracker()
        first = tracker.issue(
            "lift_test", ("grasp",), source="supplied_chunk", issued_at=1
        )
        first_ack = {
            "action_id": first.action_id,
            "execution_id": first.execution_id,
            "success": True,
        }
        tracker.acknowledge(first_ack)
        second = tracker.issue(
            "lift_test", ("grasp",), source="supplied_chunk", issued_at=2
        )
        self.assertNotEqual(first.execution_id, second.execution_id)
        with self.assertRaisesRegex(ValueError, "execution_id"):
            tracker.acknowledge(first_ack)
        self.assertIs(tracker.outstanding, second)

    def test_second_repair_cannot_be_issued_before_ack(self) -> None:
        tracker = RepairExecutionTracker()
        tracker.issue("first", (), source="supplied_chunk", issued_at=0)

        with self.assertRaisesRegex(RuntimeError, "must be acknowledged"):
            tracker.issue("second", (), source="supplied_chunk", issued_at=1)


class RepairActionValidationTests(unittest.TestCase):
    def test_ack_required_accepts_both_response_spellings(self) -> None:
        self.assertTrue(
            repair_execution_ack_required(
                {"repair_execution_ack_required": True}
            )
        )
        self.assertTrue(
            repair_execution_ack_required(
                {"requires_repair_execution_ack": True}
            )
        )
        self.assertFalse(repair_execution_ack_required({}))

    def test_repair_catalog_ids_follow_stable_list_indices(self) -> None:
        validate_repair_catalog(
            (
                {"id": 0, "name": "observe"},
                {"id": 1, "name": "regrasp"},
            )
        )
        with self.assertRaisesRegex(ValueError, "list positions"):
            validate_repair_catalog(
                (
                    {"id": 1, "name": "observe"},
                    {"id": 0, "name": "regrasp"},
                )
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_repair_catalog(
                (
                    {"id": 0, "name": "same"},
                    {"id": 1, "name": "same"},
                )
            )
        with self.assertRaisesRegex(ValueError, "at least one"):
            validate_repair_catalog(())

    def test_valid_chunk_is_returned_as_float32(self) -> None:
        chunk = np.zeros((7, 4, 4), dtype=np.float64)

        validated = validate_repair_action_chunk(
            chunk,
            expected_channels=7,
            expected_frames=4,
            actions_per_frame=4,
        )

        self.assertEqual(validated.shape, (7, 4, 4))
        self.assertEqual(validated.dtype, np.float32)

    def test_wrong_temporal_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must have shape"):
            validate_repair_action_chunk(
                np.zeros((7, 3, 4)),
                expected_channels=7,
                expected_frames=4,
                actions_per_frame=4,
            )

    def test_non_finite_or_non_numeric_values_are_rejected(self) -> None:
        non_finite = np.zeros((7, 4, 4))
        non_finite[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_repair_action_chunk(
                non_finite,
                expected_channels=7,
                expected_frames=4,
                actions_per_frame=4,
            )
        with self.assertRaisesRegex(TypeError, "numeric"):
            validate_repair_action_chunk(
                np.full((7, 4, 4), "move"),
                expected_channels=7,
                expected_frames=4,
                actions_per_frame=4,
            )


if __name__ == "__main__":
    unittest.main()
