from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import torch


_FIELD_SPECS = {
    "claim_labels": ("ledger_claim_labels", torch.float32),
    "claim_mask": ("ledger_claim_mask", torch.float32),
    "debt_targets": ("ledger_debt_targets", torch.float32),
    "debt_mask": ("ledger_debt_mask", torch.float32),
    "dependency_targets": ("ledger_dependency_targets", torch.float32),
    "dependency_mask": ("ledger_dependency_mask", torch.float32),
    "evidence_strength_targets": ("ledger_evidence_strength_targets", torch.float32),
    "evidence_strength_mask": ("ledger_evidence_strength_mask", torch.float32),
    "rollback_targets": ("ledger_rollback_targets", torch.long),
    "rollback_mask": ("ledger_rollback_mask", torch.float32),
    "relation_targets": ("ledger_relation_targets", torch.long),
    "relation_mask": ("ledger_relation_mask", torch.float32),
    "entity_targets": ("ledger_entity_targets", torch.float32),
    "entity_mask": ("ledger_entity_mask", torch.float32),
    "object_entity_targets": ("ledger_object_entity_targets", torch.float32),
    "object_entity_mask": ("ledger_object_entity_mask", torch.float32),
    "precondition_targets": ("ledger_precondition_targets", torch.long),
    "effect_targets": ("ledger_effect_targets", torch.long),
    "repair_action": ("ledger_repair_action", torch.float32),
    "repair_type": ("ledger_repair_type", torch.long),
    "repair_mask": ("ledger_repair_mask", torch.float32),
    "repair_debt_reduction": ("ledger_repair_debt_reduction", torch.float32),
    "repair_risk": ("ledger_repair_risk", torch.float32),
    "repair_value_mask": ("ledger_repair_value_mask", torch.float32),
    "evidence": ("ledger_evidence", torch.float32),
    "next_claim_labels": ("ledger_next_claim_labels", torch.float32),
    "next_debt_targets": ("ledger_next_debt_targets", torch.float32),
    "next_observation_embedding": ("ledger_next_observation_embedding", torch.float32),
    "claim_labels_sequence": ("ledger_claim_labels_sequence", torch.float32),
    "claim_mask_sequence": ("ledger_claim_mask_sequence", torch.float32),
    "debt_targets_sequence": ("ledger_debt_targets_sequence", torch.float32),
    "debt_mask_sequence": ("ledger_debt_mask_sequence", torch.float32),
    "dependency_targets_sequence": ("ledger_dependency_targets_sequence", torch.float32),
    "dependency_mask_sequence": ("ledger_dependency_mask_sequence", torch.float32),
    "evidence_strength_targets_sequence": (
        "ledger_evidence_strength_targets_sequence",
        torch.float32,
    ),
    "evidence_strength_mask_sequence": (
        "ledger_evidence_strength_mask_sequence",
        torch.float32,
    ),
    "rollback_targets_sequence": ("ledger_rollback_targets_sequence", torch.long),
    "relation_targets_sequence": ("ledger_relation_targets_sequence", torch.long),
    "entity_targets_sequence": ("ledger_entity_targets_sequence", torch.float32),
    "precondition_targets_sequence": ("ledger_precondition_targets_sequence", torch.long),
    "effect_targets_sequence": ("ledger_effect_targets_sequence", torch.long),
}


class LedgerAnnotationStore:
    """Index-addressable Ledger-WAM labels loaded from JSON or JSONL."""

    def __init__(
        self,
        path: str,
        *,
        num_claims: int,
        num_entities: int = 8,
        num_repair_actions: int = 4,
        strict_coverage: bool = True,
    ) -> None:
        self.path = Path(path).expanduser()
        self.num_claims = int(num_claims)
        self.num_entities = int(num_entities)
        self.num_repair_actions = int(num_repair_actions)
        self.strict_coverage = bool(strict_coverage)
        if not self.path.exists():
            raise FileNotFoundError(f"Ledger annotation file not found: {self.path}")
        records = self._load_records(self.path)
        self._by_index: dict[int, Mapping[str, Any]] = {}
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("Every ledger annotation record must be an object.")
            index = record.get("idx", record.get("sample_idx"))
            if index is None:
                raise ValueError("Ledger annotation record is missing `idx`.")
            index = int(index)
            if index in self._by_index:
                raise ValueError(f"Duplicate ledger annotation index: {index}")
            self._by_index[index] = record

    @staticmethod
    def _load_records(path: Path) -> list[Mapping[str, Any]]:
        if path.suffix.lower() == ".jsonl":
            records = []
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSONL at {path}:{line_number}: {exc}"
                        ) from exc
            return records
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            if "records" in payload:
                return payload["records"]
            return [dict(record, idx=index) for index, record in payload.items()]
        raise ValueError("Ledger annotation JSON must be a list, mapping, or {records: [...]}.")

    def __len__(self) -> int:
        return len(self._by_index)

    def _validate_claim_vector(self, key: str, value: torch.Tensor) -> None:
        if value.ndim != 1 or value.shape[0] != self.num_claims:
            raise ValueError(
                f"`{key}` must have shape [{self.num_claims}], got {tuple(value.shape)}."
            )

    def get(
        self,
        index: int,
        *,
        action_horizon: Optional[int] = None,
        action_dim: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        record = self._by_index.get(int(index))
        if record is None:
            if self.strict_coverage:
                raise KeyError(f"Missing ledger annotation for dataset index {index}.")
            return {}

        output: dict[str, torch.Tensor] = {}
        for source_key, (target_key, dtype) in _FIELD_SPECS.items():
            if source_key not in record:
                continue
            value = torch.as_tensor(record[source_key], dtype=dtype)
            output[target_key] = value

        for key in (
            "ledger_claim_labels",
            "ledger_claim_mask",
            "ledger_debt_targets",
            "ledger_debt_mask",
            "ledger_rollback_targets",
            "ledger_rollback_mask",
            "ledger_relation_targets",
            "ledger_relation_mask",
            "ledger_entity_mask",
            "ledger_precondition_targets",
            "ledger_effect_targets",
            "ledger_next_claim_labels",
            "ledger_next_debt_targets",
            "ledger_evidence_strength_targets",
            "ledger_evidence_strength_mask",
        ):
            if key in output:
                self._validate_claim_vector(key, output[key])

        entity_targets = output.get("ledger_entity_targets")
        if entity_targets is not None and tuple(entity_targets.shape) != (
            self.num_claims,
            self.num_entities,
        ):
            raise ValueError(
                "`entity_targets` must have shape "
                f"[{self.num_claims},{self.num_entities}], got {tuple(entity_targets.shape)}."
            )

        for key in (
            "ledger_claim_labels_sequence",
            "ledger_claim_mask_sequence",
            "ledger_debt_targets_sequence",
            "ledger_debt_mask_sequence",
            "ledger_rollback_targets_sequence",
            "ledger_relation_targets_sequence",
            "ledger_precondition_targets_sequence",
            "ledger_effect_targets_sequence",
            "ledger_evidence_strength_targets_sequence",
            "ledger_evidence_strength_mask_sequence",
        ):
            value = output.get(key)
            if value is not None and (value.ndim != 2 or value.shape[1] != self.num_claims):
                raise ValueError(
                    f"`{key}` must have shape [steps,{self.num_claims}], got {tuple(value.shape)}."
                )
        entity_sequence = output.get("ledger_entity_targets_sequence")
        if entity_sequence is not None and (
            entity_sequence.ndim != 3
            or tuple(entity_sequence.shape[1:]) != (self.num_claims, self.num_entities)
        ):
            raise ValueError(
                "`ledger_entity_targets_sequence` must have shape "
                f"[steps,{self.num_claims},{self.num_entities}]."
            )

        dependency_targets = output.get("ledger_dependency_targets")
        if dependency_targets is not None and not (
            (dependency_targets.ndim == 1 and dependency_targets.shape[0] == self.num_claims)
            or (dependency_targets.ndim == 2 and dependency_targets.shape[0] == self.num_claims)
        ):
            raise ValueError("`ledger_dependency_targets` must be [claims] or [claims,K].")
        dependency_mask = output.get("ledger_dependency_mask")
        if dependency_mask is not None and dependency_targets is not None and (
            dependency_mask.shape != dependency_targets.shape
        ):
            raise ValueError("`ledger_dependency_mask` must match dependency targets.")
        dependency_sequence = output.get("ledger_dependency_targets_sequence")
        if dependency_sequence is not None and not (
            dependency_sequence.ndim in (2, 3)
            and dependency_sequence.shape[1] == self.num_claims
        ):
            raise ValueError(
                "`ledger_dependency_targets_sequence` must be [steps,claims] or [steps,claims,K]."
            )
        dependency_sequence_mask = output.get("ledger_dependency_mask_sequence")
        if (
            dependency_sequence_mask is not None
            and dependency_sequence is not None
            and dependency_sequence_mask.shape != dependency_sequence.shape
        ):
            raise ValueError(
                "`ledger_dependency_mask_sequence` must match sequence dependency targets."
            )

        repair_action = output.get("ledger_repair_action")
        if action_horizon is not None and action_dim is not None:
            expected_action_shape = (int(action_horizon), int(action_dim))
            if repair_action is not None and tuple(repair_action.shape) != expected_action_shape:
                raise ValueError(
                    "`repair_action` must match the processed action shape "
                    f"[{action_horizon},{action_dim}], got {tuple(repair_action.shape)}."
                )
            if repair_action is None:
                output["ledger_repair_action"] = torch.zeros(
                    expected_action_shape, dtype=torch.float32
                )
                output["ledger_repair_type"] = torch.zeros((), dtype=torch.long)
                output["ledger_repair_mask"] = torch.zeros((), dtype=torch.float32)
            else:
                output.setdefault("ledger_repair_type", torch.zeros((), dtype=torch.long))
                output.setdefault("ledger_repair_mask", torch.ones((), dtype=torch.float32))

        for key in ("ledger_repair_debt_reduction", "ledger_repair_risk"):
            value = output.get(key)
            if value is not None and tuple(value.shape) != (self.num_repair_actions,):
                raise ValueError(
                    f"`{key}` must have shape [{self.num_repair_actions}], got {tuple(value.shape)}."
                )
            if value is None:
                output[key] = torch.zeros(self.num_repair_actions, dtype=torch.float32)
        has_repair_values = (
            "repair_debt_reduction" in record or "repair_risk" in record
        )
        output.setdefault(
            "ledger_repair_value_mask",
            torch.tensor(float(has_repair_values), dtype=torch.float32),
        )
        return output
