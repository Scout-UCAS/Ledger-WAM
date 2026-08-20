#!/usr/bin/env python3
"""Compile sparse simulator/event annotations into Ledger-WAM training sidecars.

Input is JSON or JSONL. Each record has an ``idx`` and a sparse ``claims`` list.
See ``docs/ledger_annotations.md`` for the complete schema.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from fastwam.models.ledger import (
    DEFAULT_CLAIM_NAMES,
    DEFAULT_ENTITY_NAMES,
    DEFAULT_RELATION_NAMES,
)


REPAIR_NAMES = ("verify", "hold", "retract", "regrasp")


def _load(path: Path) -> list[Mapping[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "records" in payload:
        return payload["records"]
    if isinstance(payload, dict):
        return [dict(value, idx=key) for key, value in payload.items()]
    if isinstance(payload, list):
        return payload
    raise ValueError("Input must be JSONL, a JSON list, or a JSON mapping.")


def _index(name: str, names: tuple[str, ...], field: str) -> int:
    try:
        return names.index(str(name))
    except ValueError as exc:
        raise ValueError(f"Unknown {field} {name!r}; expected one of {names}.") from exc


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(value)))


def _debt_target(claim: Mapping[str, Any], weights: list[float]) -> float:
    if "debt" in claim:
        return float(claim["debt"])
    confidence = float(claim.get("confidence", claim.get("truth", 0.5)))
    features = (
        1.0 - confidence,
        float(claim.get("uncertainty", 1.0 - confidence)),
        float(claim.get("dependency", 0.5)),
        float(claim.get("repair_cost", 0.5)),
        1.0 - float(claim.get("observability", 0.5)),
    )
    return _sigmoid(sum(weight * feature for weight, feature in zip(weights, features)))


def compile_record(record: Mapping[str, Any], weights: list[float]) -> dict[str, Any]:
    num_claims = len(DEFAULT_CLAIM_NAMES)
    num_entities = len(DEFAULT_ENTITY_NAMES)
    output: dict[str, Any] = {
        "idx": int(record["idx"]),
        "claim_labels": [0.0] * num_claims,
        "claim_mask": [0.0] * num_claims,
        "debt_targets": [0.0] * num_claims,
        "debt_mask": [0.0] * num_claims,
        "dependency_targets": [0.0] * num_claims,
        "dependency_mask": [0.0] * num_claims,
        "evidence_strength_targets": [0.0] * num_claims,
        "evidence_strength_mask": [0.0] * num_claims,
        "rollback_targets": [0] * num_claims,
        "rollback_mask": [0.0] * num_claims,
        "relation_targets": [0] * num_claims,
        "relation_mask": [0.0] * num_claims,
        "entity_targets": [[0.0] * num_entities for _ in range(num_claims)],
        "entity_mask": [0.0] * num_claims,
        "precondition_targets": [num_claims] * num_claims,
        "effect_targets": [num_claims] * num_claims,
    }
    seen: set[int] = set()
    for claim in record.get("claims", []):
        claim_idx = _index(claim["name"], DEFAULT_CLAIM_NAMES, "claim")
        if claim_idx in seen:
            raise ValueError(f"Duplicate claim {claim['name']!r} for idx={record['idx']}.")
        seen.add(claim_idx)
        output["claim_labels"][claim_idx] = float(bool(claim["truth"]))
        output["claim_mask"][claim_idx] = 1.0
        output["debt_targets"][claim_idx] = _debt_target(claim, weights)
        output["debt_mask"][claim_idx] = 1.0
        if "dependency" in claim:
            output["dependency_targets"][claim_idx] = float(claim["dependency"])
            output["dependency_mask"][claim_idx] = 1.0
        if "evidence_strength" in claim or "uncertainty" in claim:
            output["evidence_strength_targets"][claim_idx] = float(
                claim.get("evidence_strength", 1.0 - float(claim["uncertainty"]))
            )
            output["evidence_strength_mask"][claim_idx] = 1.0

        if "rollback_step" in claim:
            output["rollback_targets"][claim_idx] = int(claim["rollback_step"])
            output["rollback_mask"][claim_idx] = 1.0
        if "relation" in claim:
            output["relation_targets"][claim_idx] = _index(
                claim["relation"], DEFAULT_RELATION_NAMES, "relation"
            )
            output["relation_mask"][claim_idx] = 1.0
        if "entities" in claim:
            for entity in claim["entities"]:
                entity_idx = _index(entity, DEFAULT_ENTITY_NAMES, "entity")
                output["entity_targets"][claim_idx][entity_idx] = 1.0
            output["entity_mask"][claim_idx] = 1.0
        if "precondition" in claim and claim["precondition"] is not None:
            output["precondition_targets"][claim_idx] = _index(
                claim["precondition"], DEFAULT_CLAIM_NAMES, "precondition"
            )
        if "effect" in claim and claim["effect"] is not None:
            output["effect_targets"][claim_idx] = _index(
                claim["effect"], DEFAULT_CLAIM_NAMES, "effect"
            )

    repair = record.get("repair")
    if repair is not None:
        output["repair_type"] = _index(repair["type"], REPAIR_NAMES, "repair type")
        if "action" in repair:
            output["repair_action"] = repair["action"]
            output["repair_mask"] = 1.0
        if "debt_reduction" in repair:
            output["repair_debt_reduction"] = repair["debt_reduction"]
        if "risk" in repair:
            output["repair_risk"] = repair["risk"]
        if "debt_reduction" in repair or "risk" in repair:
            output["repair_value_mask"] = 1.0
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--debt-weights",
        nargs=5,
        type=float,
        default=[1.0, 1.0, 1.0, 1.0, 1.0],
        metavar=("CONF", "UNC", "DEP", "COST", "OCC"),
    )
    args = parser.parse_args()

    records = _load(args.input)
    compiled = [compile_record(record, list(args.debt_weights)) for record in records]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in compiled:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(compiled)} Ledger-WAM annotations to {args.output}")


if __name__ == "__main__":
    main()
