#!/usr/bin/env python3
"""Compute Ledger-WAM failure localization, calibration, rollback and repair metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastwam.evaluation import (
    debt_calibration_error,
    failure_localization_accuracy,
    repair_efficiency,
    rollback_accuracy,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    with args.predictions.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    result = {
        "failure_localization_accuracy": failure_localization_accuracy(
            payload["debt"], payload["failed_claim"], payload.get("failure_mask")
        ),
        "debt_calibration_error": debt_calibration_error(
            payload["predicted_debt"], payload["observed_failure"]
        ),
        "rollback_accuracy": rollback_accuracy(
            payload["rollback_logits"], payload["rollback_target"], payload.get("rollback_mask")
        ),
        "repair_efficiency": repair_efficiency(
            payload["debt_before"], payload["debt_after"], payload["repair_cost"]
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
