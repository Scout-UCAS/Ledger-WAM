#!/usr/bin/env python3
"""Compile dense LIBERO/RoboTwin/VLABench event traces into temporal sidecars.

Each input JSONL record contains ``idx`` and a ``timeline`` emitted by
``experiments.common.SimulatorEventParser``.  Optional ``repair_outcomes`` provide
counterfactual candidate rollouts with debt-before/debt-after and failure status.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _resample(timeline: list[dict[str, Any]], steps: int) -> list[dict[str, Any]]:
    if not timeline:
        raise ValueError("Every record must contain at least one timeline event.")
    indices = np.linspace(0, len(timeline) - 1, steps).round().astype(int)
    return [timeline[int(index)] for index in indices]


def compile_record(record: dict[str, Any], steps: int = 4) -> dict[str, Any]:
    events = _resample(list(record["timeline"]), steps)
    claims = np.asarray([event["claims"] for event in events], dtype=np.float32)
    if claims.shape != (steps, 8):
        raise ValueError(f"Timeline claims must resolve to [{steps},8], got {claims.shape}.")
    evidence = np.asarray(
        [event.get("evidence", np.zeros(16)) for event in events], dtype=np.float32
    )
    if evidence.shape != (steps, 16):
        raise ValueError(f"Timeline evidence must resolve to [{steps},16], got {evidence.shape}.")
    uncertainty = 1.0 - np.abs(claims - 0.5) * 2.0
    dependency = np.linspace(0.4, 1.0, 8, dtype=np.float32)[None]
    observability = np.clip(evidence[:, 14:15], 0.0, 1.0)
    debt = 1.0 / (
        1.0
        + np.exp(
            -(
                (1.0 - claims)
                + uncertainty
                + dependency
                + (1.0 - observability)
                - 1.5
            )
        )
    )
    rollback = np.zeros((steps, 8), dtype=np.int64)
    for step_index in range(steps):
        for claim_index in range(8):
            if claims[step_index, claim_index] >= 0.5:
                continue
            prior = np.flatnonzero(claims[:step_index, claim_index] >= 0.5)
            rollback[step_index, claim_index] = (
                step_index - int(prior[-1]) if prior.size else step_index
            )
    relation = np.arange(8, dtype=np.int64)[None].repeat(steps, axis=0)
    entities = np.zeros((steps, 8, 8), dtype=np.float32)
    entities[:, :, 2] = 1.0
    entities[:, :2, 1] = 1.0
    preconditions = np.full((steps, 8), 8, dtype=np.int64)
    effects = preconditions.copy()
    preconditions[:, 1] = 0
    preconditions[:, 6] = 2
    effects[:, 0] = 1
    effects[:, 1] = 5
    effects[:, 6] = 7

    reduction = np.zeros(4, dtype=np.float32)
    risk = np.zeros(4, dtype=np.float32)
    repair_mask = 0.0
    repair_type = 0
    repair_action = None
    repair_names = ("verify", "hold", "retract", "regrasp")
    for outcome in record.get("repair_outcomes", []):
        index = repair_names.index(str(outcome["type"]))
        reduction[index] = float(outcome["debt_before"]) - float(outcome["debt_after"])
        risk[index] = float(outcome.get("failure", False))
        if repair_action is None or reduction[index] > reduction[repair_type]:
            repair_type = index
            repair_action = outcome.get("action")
            repair_mask = float(repair_action is not None)

    output = {
        "idx": int(record["idx"]),
        "evidence": evidence.tolist(),
        "claim_labels_sequence": claims.tolist(),
        "claim_mask_sequence": np.ones_like(claims).tolist(),
        "debt_targets_sequence": debt.tolist(),
        "debt_mask_sequence": np.ones_like(debt).tolist(),
        "dependency_targets_sequence": dependency.repeat(steps, axis=0).tolist(),
        "dependency_mask_sequence": np.ones_like(debt).tolist(),
        "evidence_strength_targets_sequence": (1.0 - uncertainty).tolist(),
        "evidence_strength_mask_sequence": np.ones_like(debt).tolist(),
        "rollback_targets_sequence": rollback.tolist(),
        "relation_targets_sequence": relation.tolist(),
        "entity_targets_sequence": entities.tolist(),
        "precondition_targets_sequence": preconditions.tolist(),
        "effect_targets_sequence": effects.tolist(),
        "claim_labels": claims[-1].tolist(),
        "claim_mask": np.ones(8).tolist(),
        "debt_targets": debt[-1].tolist(),
        "debt_mask": np.ones(8).tolist(),
        "dependency_targets": dependency[0].tolist(),
        "dependency_mask": np.ones(8).tolist(),
        "evidence_strength_targets": (1.0 - uncertainty[-1]).tolist(),
        "evidence_strength_mask": np.ones(8).tolist(),
        "rollback_targets": rollback[-1].tolist(),
        "rollback_mask": np.ones(8).tolist(),
        "relation_targets": relation[-1].tolist(),
        "relation_mask": np.ones(8).tolist(),
        "entity_targets": entities[-1].tolist(),
        "entity_mask": np.ones(8).tolist(),
        "precondition_targets": preconditions[-1].tolist(),
        "effect_targets": effects[-1].tolist(),
        "next_claim_labels": claims[-1].tolist(),
        "next_debt_targets": debt[-1].tolist(),
        "repair_type": repair_type,
        "repair_mask": repair_mask,
        "repair_debt_reduction": reduction.tolist(),
        "repair_risk": risk.tolist(),
        "repair_value_mask": float(bool(record.get("repair_outcomes"))),
    }
    if repair_action is not None:
        output["repair_action"] = repair_action
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    with args.input.open("r", encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]
    compiled = [compile_record(record, steps=args.steps) for record in records]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as target:
        for record in compiled:
            target.write(json.dumps(record) + "\n")
    print(f"Wrote {len(compiled)} temporal simulator sidecars to {args.output}")


if __name__ == "__main__":
    main()
