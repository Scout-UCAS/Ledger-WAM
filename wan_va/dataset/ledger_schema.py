"""Schema and sidecar loader for Ledger-WAM supervision.

The annotation sidecar is deliberately independent from LeRobot so it can be
generated from simulator events, human labels, or an offline event parser.  A
record is addressed by ``"episode_index:start_frame:end_frame"`` and may be
stored either as JSON or JSONL.  The canonical JSONL form is::

    {
      "key": "12:30:90",
      "claims": [
        {
          "claim": 1,
          "claim_type": 3,
          "relation": 5,
          "dependency": 0.8,
          "debt": 0.6,
          "rollback": 2,
          "repair_action": 4,
          "post_repair_debt": 0.1
        }
      ],
      "dependency_matrix": [[0.0]],
      "counterfactual_actions": [
        {"action": [0.0, 0.1], "delta": [-0.5]}
      ]
    }

All variable-length values are converted to fixed-size tensors.  Padding and
missing labels use ``ignore_index`` and are always accompanied by boolean
masks.  This keeps default PyTorch collation safe and prevents an absent label
from being mistaken for a negative example.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


DEFAULT_IGNORE_INDEX = -100


class LedgerSchemaError(ValueError):
    """Raised when a strict Ledger-WAM annotation cannot be validated."""


def make_segment_key(episode_index: int, start_frame: int, end_frame: int) -> str:
    """Return the canonical key used by JSON/JSONL ledger sidecars."""

    episode_index = int(episode_index)
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    if episode_index < 0 or start_frame < 0 or end_frame < start_frame:
        raise LedgerSchemaError(
            "Invalid ledger segment: episode={}, start={}, end={}".format(
                episode_index, start_frame, end_frame
            )
        )
    return "{}:{}:{}".format(episode_index, start_frame, end_frame)


def canonical_segment_key(value: Any) -> str:
    """Validate and canonicalize a ``episode:start:end`` key."""

    if not isinstance(value, str):
        raise LedgerSchemaError("Ledger record key must be a string")
    parts = value.split(":")
    if len(parts) != 3:
        raise LedgerSchemaError(
            "Ledger record key must have form episode:start:end, got {!r}".format(value)
        )
    try:
        return make_segment_key(*(int(part) for part in parts))
    except (TypeError, ValueError) as exc:
        raise LedgerSchemaError("Invalid ledger record key {!r}".format(value)) from exc


@dataclass(frozen=True)
class LedgerTensorSpec:
    """Fixed dimensions and validation policy for ledger annotations."""

    max_claims: int = 16
    max_counterfactuals: int = 4
    action_dim: int = 30
    ignore_index: int = DEFAULT_IGNORE_INDEX
    strict: bool = False

    def __post_init__(self) -> None:
        if self.max_claims <= 0:
            raise ValueError("max_claims must be positive")
        if self.max_counterfactuals < 0:
            raise ValueError("max_counterfactuals must be non-negative")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")


# Numeric claim fields.  Each tuple is
# (output stem, aliases, dtype, probability_like).
_CLAIM_FIELDS: Tuple[Tuple[str, Tuple[str, ...], torch.dtype, bool], ...] = (
    ("claim", ("claim_label", "claim", "truth", "label"), torch.float32, True),
    ("claim_type", ("claim_type", "type", "claim_type_id"), torch.long, False),
    ("subject", ("subject_id", "subject"), torch.long, False),
    ("object", ("object_id", "object"), torch.long, False),
    ("relation", ("relation", "relation_id", "relation_label"), torch.long, False),
    (
        "precondition",
        ("precondition", "precondition_id", "precondition_label"),
        torch.long,
        False,
    ),
    ("effect", ("effect", "effect_id", "effect_label"), torch.long, False),
    ("evidence", ("evidence", "evidence_score"), torch.float32, True),
    (
        "uncertainty",
        ("uncertainty", "uncertainty_score"),
        torch.float32,
        True,
    ),
    (
        "dependency",
        ("dependency", "dependency_score", "downstream_dependency"),
        torch.float32,
        True,
    ),
    ("debt", ("debt", "causal_debt", "debt_target"), torch.float32, True),
    (
        "observability",
        ("observability", "observable", "observability_score"),
        torch.float32,
        True,
    ),
    ("repair_cost", ("repair_cost", "recovery_cost"), torch.float32, False),
    ("importance", ("importance", "task_importance"), torch.float32, True),
    (
        "rollback",
        ("rollback", "rollback_stage", "rollback_index"),
        torch.long,
        False,
    ),
    (
        "repair_action",
        ("repair_action", "repair_action_id", "repair_label"),
        torch.long,
        False,
    ),
    (
        "post_repair_debt",
        ("post_repair_debt", "repaired_debt", "next_debt"),
        torch.float32,
        True,
    ),
)


def _ignore_tensor(
    shape: Sequence[int], dtype: torch.dtype, ignore_index: int
) -> torch.Tensor:
    value = float(ignore_index) if dtype.is_floating_point else int(ignore_index)
    return torch.full(tuple(shape), value, dtype=dtype)


def empty_ledger_tensors(
    spec: LedgerTensorSpec, available: bool = False
) -> Dict[str, torch.Tensor]:
    """Create a completely padded ledger sample with stable tensor shapes."""

    max_claims = spec.max_claims
    max_counterfactuals = spec.max_counterfactuals
    output: Dict[str, torch.Tensor] = {
        "ledger_available": torch.tensor(bool(available), dtype=torch.bool),
        "ledger_claim_mask": torch.zeros(max_claims, dtype=torch.bool),
    }
    for stem, _aliases, dtype, _probability_like in _CLAIM_FIELDS:
        output["ledger_{}_labels".format(stem)] = _ignore_tensor(
            (max_claims,), dtype, spec.ignore_index
        )
        output["ledger_{}_mask".format(stem)] = torch.zeros(
            max_claims, dtype=torch.bool
        )

    output["ledger_dependency_matrix"] = _ignore_tensor(
        (max_claims, max_claims), torch.float32, spec.ignore_index
    )
    output["ledger_dependency_matrix_mask"] = torch.zeros(
        max_claims, max_claims, dtype=torch.bool
    )
    output["ledger_counterfactual_mask"] = torch.zeros(
        max_counterfactuals, dtype=torch.bool
    )
    output["ledger_counterfactual_actions"] = _ignore_tensor(
        (max_counterfactuals, spec.action_dim), torch.float32, spec.ignore_index
    )
    output["ledger_counterfactual_action_mask"] = torch.zeros(
        max_counterfactuals, spec.action_dim, dtype=torch.bool
    )
    output["ledger_counterfactual_deltas"] = _ignore_tensor(
        (max_counterfactuals, max_claims), torch.float32, spec.ignore_index
    )
    output["ledger_counterfactual_delta_mask"] = torch.zeros(
        max_counterfactuals, max_claims, dtype=torch.bool
    )
    return output


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _first_present(mapping: Mapping[str, Any], aliases: Iterable[str]) -> Any:
    for key in aliases:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _coerce_number(
    value: Any,
    dtype: torch.dtype,
    field_name: str,
    strict: bool,
    probability_like: bool = False,
) -> Optional[Any]:
    try:
        if isinstance(value, bool) and dtype == torch.long:
            result: Any = int(value)
        elif dtype == torch.long:
            number = float(value)
            if not math.isfinite(number) or not number.is_integer():
                raise ValueError
            result = int(number)
        else:
            result = float(value)
            if not math.isfinite(result):
                raise ValueError
    except (TypeError, ValueError, OverflowError):
        if strict:
            raise LedgerSchemaError(
                "Field {!r} must be a finite numeric value, got {!r}".format(
                    field_name, value
                )
            )
        return None

    if probability_like and not 0.0 <= float(result) <= 1.0:
        if strict:
            raise LedgerSchemaError(
                "Field {!r} must be in [0, 1], got {!r}".format(field_name, value)
            )
        result = min(1.0, max(0.0, float(result)))
    if field_name == "repair_cost" and float(result) < 0.0:
        if strict:
            raise LedgerSchemaError("repair_cost must be non-negative")
        result = 0.0
    return result


def _normalise_claims(
    record: Mapping[str, Any], strict: bool
) -> List[Mapping[str, Any]]:
    claims = record.get("claims", [])
    if claims is None:
        return []
    if not _is_sequence(claims):
        if strict:
            raise LedgerSchemaError("claims must be a list")
        return []

    output: List[Mapping[str, Any]] = []
    for claim in claims:
        if isinstance(claim, Mapping):
            output.append(claim)
        elif strict:
            raise LedgerSchemaError("Every claim must be an object")
        else:
            # A numeric shorthand means a claim-validity target.
            value = _coerce_number(claim, torch.float32, "claim", False, True)
            if value is not None:
                output.append({"claim": value})
    return output


def _fill_claim_fields(
    output: Dict[str, torch.Tensor],
    claims: Sequence[Mapping[str, Any]],
    spec: LedgerTensorSpec,
) -> None:
    if len(claims) > spec.max_claims and spec.strict:
        raise LedgerSchemaError(
            "Record has {} claims, maximum is {}".format(len(claims), spec.max_claims)
        )
    selected_claims = claims[: spec.max_claims]
    output["ledger_claim_mask"][: len(selected_claims)] = True

    for claim_index, claim in enumerate(selected_claims):
        for stem, aliases, dtype, probability_like in _CLAIM_FIELDS:
            value = _first_present(claim, aliases)
            if value is None:
                continue
            value = _coerce_number(
                value,
                dtype,
                stem,
                spec.strict,
                probability_like=probability_like,
            )
            if value is None:
                continue
            output["ledger_{}_labels".format(stem)][claim_index] = value
            output["ledger_{}_mask".format(stem)][claim_index] = True


def _fill_dependency_matrix(
    output: Dict[str, torch.Tensor],
    record: Mapping[str, Any],
    spec: LedgerTensorSpec,
) -> None:
    matrix = record.get("dependency_matrix")
    if matrix is not None:
        if not _is_sequence(matrix):
            if spec.strict:
                raise LedgerSchemaError("dependency_matrix must be a nested list")
            return
        if len(matrix) > spec.max_claims and spec.strict:
            raise LedgerSchemaError("dependency_matrix has too many rows")
        for row_index, row in enumerate(matrix[: spec.max_claims]):
            if not _is_sequence(row):
                if spec.strict:
                    raise LedgerSchemaError("dependency_matrix rows must be lists")
                continue
            if len(row) > spec.max_claims and spec.strict:
                raise LedgerSchemaError("dependency_matrix has too many columns")
            for column_index, value in enumerate(row[: spec.max_claims]):
                value = _coerce_number(
                    value,
                    torch.float32,
                    "dependency_matrix",
                    spec.strict,
                    probability_like=True,
                )
                if value is None:
                    continue
                output["ledger_dependency_matrix"][row_index, column_index] = value
                output["ledger_dependency_matrix_mask"][row_index, column_index] = True
        return

    # Sparse edge form: [{"source": 0, "target": 1, "weight": 1.0}, ...].
    edges = record.get("dependency_edges", record.get("dependencies"))
    if edges is None:
        return
    if not _is_sequence(edges):
        if spec.strict:
            raise LedgerSchemaError("dependency_edges must be a list")
        return

    # Sparse dependencies use adjacency-list semantics: among the claims that
    # are present in this record, an omitted directed pair is a supervised
    # negative.  Without these negatives the pairwise BCE objective could be
    # minimized by predicting every possible dependency edge.  Padded claim
    # slots remain masked and therefore never become artificial negatives.
    active_claims = output["ledger_claim_mask"]
    active_pairs = active_claims[:, None] & active_claims[None, :]
    output["ledger_dependency_matrix"][active_pairs] = 0.0
    output["ledger_dependency_matrix_mask"][active_pairs] = True
    for edge in edges:
        if not isinstance(edge, Mapping):
            if spec.strict:
                raise LedgerSchemaError("Every dependency edge must be an object")
            continue
        source = _coerce_number(
            edge.get("source"), torch.long, "dependency source", spec.strict
        )
        target = _coerce_number(
            edge.get("target"), torch.long, "dependency target", spec.strict
        )
        weight = _coerce_number(
            edge.get("weight", 1.0),
            torch.float32,
            "dependency weight",
            spec.strict,
            probability_like=True,
        )
        if source is None or target is None or weight is None:
            continue
        if not 0 <= source < spec.max_claims or not 0 <= target < spec.max_claims:
            if spec.strict:
                raise LedgerSchemaError("Dependency edge index is out of range")
            continue
        output["ledger_dependency_matrix"][source, target] = weight
        output["ledger_dependency_matrix_mask"][source, target] = True


def _as_vector_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if not _is_sequence(value):
        return [value]
    if len(value) == 0:
        return []
    if all(not isinstance(item, Mapping) and not _is_sequence(item) for item in value):
        return [value]
    return list(value)


def _normalise_counterfactuals(
    record: Mapping[str, Any],
) -> Tuple[List[Any], List[Any]]:
    raw_actions = _as_vector_list(record.get("counterfactual_actions"))
    raw_deltas = _as_vector_list(
        record.get("counterfactual_deltas", record.get("counterfactual_delta"))
    )

    actions: List[Any] = []
    deltas: List[Any] = []
    if raw_actions and all(isinstance(item, Mapping) for item in raw_actions):
        for item in raw_actions:
            actions.append(item.get("action"))
            deltas.append(item.get("delta", item.get("ledger_delta")))
        # Explicit top-level deltas override missing per-item deltas.
        for index, delta in enumerate(raw_deltas):
            if index >= len(deltas):
                deltas.append(delta)
            elif deltas[index] is None:
                deltas[index] = delta
    else:
        actions = raw_actions
        deltas = raw_deltas
    return actions, deltas


def _fill_vector(
    destination: torch.Tensor,
    destination_mask: torch.Tensor,
    raw_vector: Any,
    field_name: str,
    strict: bool,
) -> bool:
    if raw_vector is None:
        return False
    if not _is_sequence(raw_vector):
        if strict:
            raise LedgerSchemaError("{} must be a numeric list".format(field_name))
        return False
    if len(raw_vector) > destination.numel() and strict:
        raise LedgerSchemaError(
            "{} has length {}, maximum is {}".format(
                field_name, len(raw_vector), destination.numel()
            )
        )
    wrote_value = False
    for index, value in enumerate(raw_vector[: destination.numel()]):
        value = _coerce_number(
            value, torch.float32, field_name, strict, probability_like=False
        )
        if value is None:
            continue
        destination[index] = value
        destination_mask[index] = True
        wrote_value = True
    return wrote_value


def _fill_counterfactuals(
    output: Dict[str, torch.Tensor],
    record: Mapping[str, Any],
    spec: LedgerTensorSpec,
) -> None:
    actions, deltas = _normalise_counterfactuals(record)
    count = max(len(actions), len(deltas))
    if count > spec.max_counterfactuals and spec.strict:
        raise LedgerSchemaError(
            "Record has {} counterfactuals, maximum is {}".format(
                count, spec.max_counterfactuals
            )
        )
    for index in range(min(count, spec.max_counterfactuals)):
        wrote_action = False
        wrote_delta = False
        if index < len(actions):
            wrote_action = _fill_vector(
                output["ledger_counterfactual_actions"][index],
                output["ledger_counterfactual_action_mask"][index],
                actions[index],
                "counterfactual action",
                spec.strict,
            )
        if index < len(deltas):
            wrote_delta = _fill_vector(
                output["ledger_counterfactual_deltas"][index],
                output["ledger_counterfactual_delta_mask"][index],
                deltas[index],
                "counterfactual delta",
                spec.strict,
            )
        output["ledger_counterfactual_mask"][index] = wrote_action or wrote_delta


def tensorize_ledger_record(
    record: Optional[Mapping[str, Any]], spec: LedgerTensorSpec
) -> Dict[str, torch.Tensor]:
    """Convert one decoded sidecar record into padded supervision tensors."""

    if record is None:
        return empty_ledger_tensors(spec, available=False)
    if not isinstance(record, Mapping):
        if spec.strict:
            raise LedgerSchemaError("Ledger record must be a JSON object")
        return empty_ledger_tensors(spec, available=False)

    output = empty_ledger_tensors(spec, available=True)
    claims = _normalise_claims(record, spec.strict)
    _fill_claim_fields(output, claims, spec)
    _fill_dependency_matrix(output, record, spec)
    _fill_counterfactuals(output, record, spec)
    return output


class LedgerAnnotationStore:
    """In-memory index for JSON/JSONL ledger annotations."""

    _DIRECTORY_CANDIDATES = (
        "ledger_annotations.jsonl",
        "ledger.jsonl",
        "ledger_annotations.json",
        "ledger.json",
    )

    def __init__(
        self,
        path: Optional[Any],
        spec: LedgerTensorSpec,
        strict: Optional[bool] = None,
    ) -> None:
        self.spec = spec
        self.strict = spec.strict if strict is None else bool(strict)
        self.path = self._resolve_path(path)
        self.records: Dict[str, Mapping[str, Any]] = {}
        if self.path is not None:
            self.records = self._load(self.path)

    def _resolve_path(self, path: Optional[Any]) -> Optional[Path]:
        if path is None:
            if self.strict:
                raise FileNotFoundError(
                    "Ledger annotation path is required in strict mode"
                )
            return None
        resolved = Path(path)
        if resolved.is_dir():
            for filename in self._DIRECTORY_CANDIDATES:
                candidate = resolved / filename
                if candidate.is_file():
                    return candidate
            if self.strict:
                raise FileNotFoundError(
                    "No ledger JSON/JSONL sidecar found in {}".format(resolved)
                )
            return None
        if not resolved.is_file():
            if self.strict:
                raise FileNotFoundError(
                    "Ledger annotation sidecar not found: {}".format(resolved)
                )
            return None
        return resolved

    def _load(self, path: Path) -> Dict[str, Mapping[str, Any]]:
        if path.suffix.lower() == ".jsonl":
            decoded: Any = []
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        decoded.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        if self.strict:
                            raise LedgerSchemaError(
                                "Invalid JSON on line {} of {}".format(
                                    line_number, path
                                )
                            ) from exc
        else:
            try:
                with path.open("r", encoding="utf-8") as handle:
                    decoded = json.load(handle)
            except json.JSONDecodeError as exc:
                if self.strict:
                    raise LedgerSchemaError(
                        "Invalid ledger JSON file: {}".format(path)
                    ) from exc
                return {}

        records: Dict[str, Mapping[str, Any]] = {}
        for raw_key, record in self._iter_records(decoded):
            try:
                key = self._record_key(record, raw_key)
            except LedgerSchemaError:
                if self.strict:
                    raise
                continue
            if key in records and self.strict:
                raise LedgerSchemaError("Duplicate ledger record key: {}".format(key))
            records[key] = record
        return records

    def _iter_records(
        self, decoded: Any
    ) -> Iterable[Tuple[Optional[str], Mapping[str, Any]]]:
        if isinstance(decoded, Mapping):
            if "records" in decoded:
                decoded = decoded["records"]
            elif "key" in decoded or "episode_index" in decoded:
                decoded = [decoded]
            else:
                for key, value in decoded.items():
                    if isinstance(value, Mapping):
                        yield str(key), value
                    elif self.strict:
                        raise LedgerSchemaError("Ledger mapping values must be objects")
                return

        if not _is_sequence(decoded):
            if self.strict:
                raise LedgerSchemaError("Ledger file must contain records")
            return
        for value in decoded:
            if isinstance(value, Mapping):
                yield None, value
            elif self.strict:
                raise LedgerSchemaError("Every ledger record must be an object")

    def _record_key(
        self, record: Mapping[str, Any], fallback_key: Optional[str]
    ) -> str:
        if "key" in record:
            return canonical_segment_key(record["key"])
        if fallback_key is not None:
            return canonical_segment_key(fallback_key)

        episode = record.get("episode_index", record.get("episode"))
        start = record.get("start_frame", record.get("start"))
        end = record.get("end_frame", record.get("end"))
        if episode is None or start is None or end is None:
            raise LedgerSchemaError(
                "Ledger record needs key or episode_index/start_frame/end_frame"
            )
        try:
            return make_segment_key(int(episode), int(start), int(end))
        except (TypeError, ValueError) as exc:
            raise LedgerSchemaError("Invalid ledger segment fields") from exc

    def get_record(
        self, episode_index: int, start_frame: int, end_frame: int
    ) -> Optional[Mapping[str, Any]]:
        key = make_segment_key(episode_index, start_frame, end_frame)
        record = self.records.get(key)
        if record is None and self.strict:
            raise KeyError("Missing ledger annotation for {}".format(key))
        return record

    def tensors_for(
        self, episode_index: int, start_frame: int, end_frame: int
    ) -> Dict[str, torch.Tensor]:
        """Look up a segment and return its fixed-shape tensor dictionary."""

        return tensorize_ledger_record(
            self.get_record(episode_index, start_frame, end_frame), self.spec
        )


__all__ = [
    "DEFAULT_IGNORE_INDEX",
    "LedgerAnnotationStore",
    "LedgerSchemaError",
    "LedgerTensorSpec",
    "canonical_segment_key",
    "empty_ledger_tensors",
    "make_segment_key",
    "tensorize_ledger_record",
]
