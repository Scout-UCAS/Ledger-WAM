"""Pure-Python runtime primitives for Ledger-WAM.

The classes in this module intentionally do not depend on a model framework.  They
hold structured causal claims, compute causal debt, maintain claim dependencies,
and arbitrate between task, repair, and local-planning rollback decisions.

Rollback is *logical only*: it moves the plan cursor and invalidates dependent
beliefs.  It never rewinds observations, robot proprioception, simulator state, or
any other physical-world state.  A controller must execute recovery actions from
the current physical state after a rollback decision.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)


Metadata = Dict[str, object]


def _require_non_empty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a non-empty string" % name)


def _require_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError("%s must be finite" % name)


def _require_unit_interval(value: float, name: str) -> None:
    _require_finite(value, name)
    if value < 0.0 or value > 1.0:
        raise ValueError("%s must be in [0, 1]" % name)


def _require_non_negative(value: float, name: str) -> None:
    _require_finite(value, name)
    if value < 0.0:
        raise ValueError("%s must be non-negative" % name)


def _json_safe_copy(value: Mapping[str, object], name: str) -> Metadata:
    """Return a detached JSON-safe metadata dictionary."""

    try:
        encoded = json.dumps(value, allow_nan=False, sort_keys=True)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "%s must contain only JSON-serializable values" % name
        ) from exc
    if not isinstance(decoded, dict):
        raise ValueError("%s must be a JSON object" % name)
    return cast(Metadata, decoded)


def _as_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError("%s must be a JSON object" % name)
    return cast(Mapping[str, object], value)


def _as_sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ValueError("%s must be a JSON array" % name)
    return cast(Sequence[object], value)


def _as_string_tuple(value: object, name: str) -> Tuple[str, ...]:
    values = _as_sequence(value, name)
    result: List[str] = []
    for item in values:
        if not isinstance(item, str):
            raise ValueError("every item in %s must be a string" % name)
        result.append(item)
    return tuple(result)


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("%s must be an integer" % name)
    return value


def _as_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a number" % name)
    result = float(value)
    _require_finite(result, name)
    return result


def _as_optional_string(value: object, name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("%s must be a string or null" % name)
    return value


def _stable_sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class EvidencePolarity(str, Enum):
    """How a piece of evidence bears on a claim."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


class ClaimStatus(str, Enum):
    """Lifecycle state of a causal claim."""

    HYPOTHESIZED = "hypothesized"
    VERIFIED = "verified"
    REFUTED = "refuted"
    INVALIDATED = "invalidated"


class PlannerDecisionType(str, Enum):
    """The three runtime branches exposed by the self-healing planner."""

    TASK = "task"
    REPAIR = "repair"
    ROLLBACK = "rollback"


@dataclass
class Evidence:
    """A timestamped, attributable observation about a causal claim."""

    source: str
    polarity: EvidencePolarity
    strength: float = 1.0
    timestamp: int = 0
    observation_id: Optional[str] = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.source, "source")
        if isinstance(self.polarity, str):
            self.polarity = EvidencePolarity(self.polarity)
        _require_unit_interval(self.strength, "strength")
        if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, int):
            raise ValueError("timestamp must be an integer")
        if self.observation_id is not None:
            _require_non_empty(self.observation_id, "observation_id")
        self.metadata = _json_safe_copy(self.metadata, "metadata")

    def to_dict(self) -> Dict[str, object]:
        return {
            "source": self.source,
            "polarity": self.polarity.value,
            "strength": self.strength,
            "timestamp": self.timestamp,
            "observation_id": self.observation_id,
            "metadata": _json_safe_copy(self.metadata, "metadata"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Evidence:
        source = data.get("source")
        polarity = data.get("polarity")
        if not isinstance(source, str) or not isinstance(polarity, str):
            raise ValueError("evidence source and polarity must be strings")
        return cls(
            source=source,
            polarity=EvidencePolarity(polarity),
            strength=_as_float(data.get("strength", 1.0), "strength"),
            timestamp=_as_int(data.get("timestamp", 0), "timestamp"),
            observation_id=_as_optional_string(
                data.get("observation_id"), "observation_id"
            ),
            metadata=_json_safe_copy(
                _as_mapping(data.get("metadata", {}), "metadata"), "metadata"
            ),
        )


@dataclass
class CausalClaim:
    """One structured action-conditioned assertion in the belief ledger."""

    claim_id: str
    entities: Tuple[str, ...]
    relation: str
    preconditions: Tuple[str, ...] = ()
    effects: Tuple[str, ...] = ()
    evidence: List[Evidence] = field(default_factory=list)
    confidence: float = 0.5
    uncertainty: float = 1.0
    dependency: float = 0.0
    repair_cost: float = 0.0
    observability: float = 1.0
    importance: float = 1.0
    debt: float = 0.0
    rollback_checkpoint: Optional[str] = None
    status: ClaimStatus = ClaimStatus.HYPOTHESIZED
    created_at: int = 0
    updated_at: int = 0
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.claim_id, "claim_id")
        self.entities = tuple(self.entities)
        if not self.entities:
            raise ValueError("entities must not be empty")
        for entity in self.entities:
            _require_non_empty(entity, "entity")
        _require_non_empty(self.relation, "relation")
        self.preconditions = tuple(self.preconditions)
        self.effects = tuple(self.effects)
        for precondition in self.preconditions:
            _require_non_empty(precondition, "precondition")
        for effect in self.effects:
            _require_non_empty(effect, "effect")
        self.evidence = list(self.evidence)
        if not all(isinstance(item, Evidence) for item in self.evidence):
            raise ValueError("evidence must contain Evidence objects")
        _require_unit_interval(self.confidence, "confidence")
        _require_unit_interval(self.uncertainty, "uncertainty")
        _require_unit_interval(self.dependency, "dependency")
        _require_non_negative(self.repair_cost, "repair_cost")
        _require_unit_interval(self.observability, "observability")
        _require_non_negative(self.importance, "importance")
        _require_unit_interval(self.debt, "debt")
        if self.rollback_checkpoint is not None:
            _require_non_empty(self.rollback_checkpoint, "rollback_checkpoint")
        if isinstance(self.status, str):
            self.status = ClaimStatus(self.status)
        if isinstance(self.created_at, bool) or not isinstance(self.created_at, int):
            raise ValueError("created_at must be an integer")
        if isinstance(self.updated_at, bool) or not isinstance(self.updated_at, int):
            raise ValueError("updated_at must be an integer")
        self.metadata = _json_safe_copy(self.metadata, "metadata")

    def apply_evidence(
        self,
        evidence: Evidence,
        verification_threshold: float = 0.8,
        refutation_threshold: float = 0.2,
    ) -> ClaimStatus:
        """Fuse evidence into confidence and return the resulting claim status.

        This is a small deterministic runtime fusion rule, not a learned confidence
        estimator.  Learned systems may set ``confidence`` directly and still use
        the rest of the ledger runtime.
        """

        _require_unit_interval(verification_threshold, "verification_threshold")
        _require_unit_interval(refutation_threshold, "refutation_threshold")
        if refutation_threshold >= verification_threshold:
            raise ValueError(
                "refutation_threshold must be smaller than verification_threshold"
            )
        self.evidence.append(evidence)
        if evidence.polarity is EvidencePolarity.SUPPORTS:
            self.confidence += evidence.strength * (1.0 - self.confidence)
        elif evidence.polarity is EvidencePolarity.CONTRADICTS:
            self.confidence *= 1.0 - evidence.strength
        self.confidence = min(1.0, max(0.0, self.confidence))
        self.updated_at = max(self.updated_at, evidence.timestamp)

        if self.confidence <= refutation_threshold:
            self.status = ClaimStatus.REFUTED
        elif self.confidence >= verification_threshold:
            self.status = ClaimStatus.VERIFIED
        elif evidence.polarity is not EvidencePolarity.NEUTRAL:
            self.status = ClaimStatus.HYPOTHESIZED
        return self.status

    def to_dict(self) -> Dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "entities": list(self.entities),
            "relation": self.relation,
            "preconditions": list(self.preconditions),
            "effects": list(self.effects),
            "evidence": [item.to_dict() for item in self.evidence],
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "dependency": self.dependency,
            "repair_cost": self.repair_cost,
            "observability": self.observability,
            "importance": self.importance,
            "debt": self.debt,
            "rollback_checkpoint": self.rollback_checkpoint,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": _json_safe_copy(self.metadata, "metadata"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> CausalClaim:
        claim_id = data.get("claim_id")
        relation = data.get("relation")
        status = data.get("status", ClaimStatus.HYPOTHESIZED.value)
        if not isinstance(claim_id, str) or not isinstance(relation, str):
            raise ValueError("claim_id and relation must be strings")
        if not isinstance(status, str):
            raise ValueError("status must be a string")
        raw_evidence = _as_sequence(data.get("evidence", []), "evidence")
        evidence = [
            Evidence.from_dict(_as_mapping(item, "evidence item"))
            for item in raw_evidence
        ]
        return cls(
            claim_id=claim_id,
            entities=_as_string_tuple(data.get("entities", []), "entities"),
            relation=relation,
            preconditions=_as_string_tuple(
                data.get("preconditions", []), "preconditions"
            ),
            effects=_as_string_tuple(data.get("effects", []), "effects"),
            evidence=evidence,
            confidence=_as_float(data.get("confidence", 0.5), "confidence"),
            uncertainty=_as_float(data.get("uncertainty", 1.0), "uncertainty"),
            dependency=_as_float(data.get("dependency", 0.0), "dependency"),
            repair_cost=_as_float(data.get("repair_cost", 0.0), "repair_cost"),
            observability=_as_float(data.get("observability", 1.0), "observability"),
            importance=_as_float(data.get("importance", 1.0), "importance"),
            debt=_as_float(data.get("debt", 0.0), "debt"),
            rollback_checkpoint=_as_optional_string(
                data.get("rollback_checkpoint"), "rollback_checkpoint"
            ),
            status=ClaimStatus(status),
            created_at=_as_int(data.get("created_at", 0), "created_at"),
            updated_at=_as_int(data.get("updated_at", 0), "updated_at"),
            metadata=_json_safe_copy(
                _as_mapping(data.get("metadata", {}), "metadata"), "metadata"
            ),
        )


@dataclass
class DebtWeights:
    """Non-negative coefficients for the corrected causal-debt formula.

    ``debt = sigmoid(bias + w_s*(1-confidence) + w_u*uncertainty
    + w_d*dependency + w_r*repair_cost + w_o*(1-observability))``.

    Non-negative coefficients guarantee that every stated risk factor changes debt
    monotonically in the intended direction.
    """

    confidence_gap: float = 1.0
    uncertainty: float = 1.0
    dependency: float = 1.0
    repair_cost: float = 1.0
    unobservability: float = 1.0
    bias: float = -2.0

    def __post_init__(self) -> None:
        for name in (
            "confidence_gap",
            "uncertainty",
            "dependency",
            "repair_cost",
            "unobservability",
        ):
            _require_non_negative(float(getattr(self, name)), name)
        _require_finite(self.bias, "bias")

    def calculate(self, claim: CausalClaim) -> float:
        logit = (
            self.bias
            + self.confidence_gap * (1.0 - claim.confidence)
            + self.uncertainty * claim.uncertainty
            + self.dependency * claim.dependency
            + self.repair_cost * claim.repair_cost
            + self.unobservability * (1.0 - claim.observability)
        )
        return _stable_sigmoid(logit)

    def to_dict(self) -> Dict[str, object]:
        return {
            "confidence_gap": self.confidence_gap,
            "uncertainty": self.uncertainty,
            "dependency": self.dependency,
            "repair_cost": self.repair_cost,
            "unobservability": self.unobservability,
            "bias": self.bias,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> DebtWeights:
        return cls(
            confidence_gap=_as_float(data.get("confidence_gap", 1.0), "confidence_gap"),
            uncertainty=_as_float(data.get("uncertainty", 1.0), "uncertainty"),
            dependency=_as_float(data.get("dependency", 1.0), "dependency"),
            repair_cost=_as_float(data.get("repair_cost", 1.0), "repair_cost"),
            unobservability=_as_float(
                data.get("unobservability", 1.0), "unobservability"
            ),
            bias=_as_float(data.get("bias", -2.0), "bias"),
        )


class CausalBeliefLedger:
    """Mutable collection of causal claims and their dependency DAG."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        claims: Optional[Iterable[CausalClaim]] = None,
        dependencies: Optional[Iterable[Tuple[str, str]]] = None,
    ) -> None:
        self._claims: Dict[str, CausalClaim] = {}
        self._children: Dict[str, Set[str]] = {}
        self._parents: Dict[str, Set[str]] = {}
        if claims is not None:
            for claim in claims:
                self.add_claim(claim)
        if dependencies is not None:
            for prerequisite_id, dependent_id in dependencies:
                self.add_dependency(prerequisite_id, dependent_id)

    def __len__(self) -> int:
        return len(self._claims)

    @property
    def claims(self) -> Mapping[str, CausalClaim]:
        return MappingProxyType(self._claims)

    @property
    def dependencies(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(
            sorted(
                (parent, child)
                for parent, children in self._children.items()
                for child in children
            )
        )

    def add_claim(self, claim: CausalClaim, replace: bool = False) -> None:
        if claim.claim_id in self._claims and not replace:
            raise ValueError("claim %r already exists" % claim.claim_id)
        self._claims[claim.claim_id] = claim
        self._children.setdefault(claim.claim_id, set())
        self._parents.setdefault(claim.claim_id, set())

    def get_claim(self, claim_id: str) -> CausalClaim:
        try:
            return self._claims[claim_id]
        except KeyError as exc:
            raise KeyError("unknown claim %r" % claim_id) from exc

    def add_dependency(self, prerequisite_id: str, dependent_id: str) -> None:
        """Add ``prerequisite -> dependent`` while preserving a DAG."""

        self.get_claim(prerequisite_id)
        self.get_claim(dependent_id)
        if prerequisite_id == dependent_id:
            raise ValueError("a claim cannot depend on itself")
        if prerequisite_id in self.descendants_of(dependent_id):
            raise ValueError("dependency would create a cycle")
        self._children[prerequisite_id].add(dependent_id)
        self._parents[dependent_id].add(prerequisite_id)

    def remove_dependency(self, prerequisite_id: str, dependent_id: str) -> None:
        """Remove one edge if present, preserving both adjacency indexes."""
        self.get_claim(prerequisite_id)
        self.get_claim(dependent_id)
        self._children[prerequisite_id].discard(dependent_id)
        self._parents[dependent_id].discard(prerequisite_id)

    def clear_dependencies(self, claim_ids: Optional[Iterable[str]] = None) -> None:
        """Clear all edges, or only edges whose endpoints are in ``claim_ids``."""
        if claim_ids is None:
            selected = set(self._claims)
        else:
            selected = set(claim_ids)
            unknown = selected - set(self._claims)
            if unknown:
                raise KeyError("unknown claims: %s" % sorted(unknown))
        for prerequisite_id, dependent_id in tuple(self.dependencies):
            if prerequisite_id in selected and dependent_id in selected:
                self.remove_dependency(prerequisite_id, dependent_id)

    def children_of(self, claim_id: str) -> Tuple[str, ...]:
        self.get_claim(claim_id)
        return tuple(sorted(self._children[claim_id]))

    def parents_of(self, claim_id: str) -> Tuple[str, ...]:
        self.get_claim(claim_id)
        return tuple(sorted(self._parents[claim_id]))

    def descendants_of(self, claim_id: str) -> Tuple[str, ...]:
        self.get_claim(claim_id)
        descendants: Set[str] = set()
        queue = deque(sorted(self._children[claim_id]))
        while queue:
            current = queue.popleft()
            if current in descendants:
                continue
            descendants.add(current)
            queue.extend(sorted(self._children[current]))
        return tuple(sorted(descendants))

    def invalidate_descendants(
        self, claim_id: str, include_claim: bool = False
    ) -> Tuple[str, ...]:
        """Invalidate all transitive consumers of a failed assumption."""

        target_ids = set(self.descendants_of(claim_id))
        if include_claim:
            target_ids.add(claim_id)
        for target_id in target_ids:
            self._claims[target_id].status = ClaimStatus.INVALIDATED
        return tuple(sorted(target_ids))

    def refute_claim(self, claim_id: str) -> Tuple[str, ...]:
        claim = self.get_claim(claim_id)
        claim.status = ClaimStatus.REFUTED
        return self.invalidate_descendants(claim_id)

    def record_evidence(
        self,
        claim_id: str,
        evidence: Evidence,
        verification_threshold: float = 0.8,
        refutation_threshold: float = 0.2,
    ) -> Tuple[str, ...]:
        claim = self.get_claim(claim_id)
        status = claim.apply_evidence(
            evidence,
            verification_threshold=verification_threshold,
            refutation_threshold=refutation_threshold,
        )
        if status is ClaimStatus.REFUTED:
            return self.invalidate_descendants(claim_id)
        return ()

    def recompute_debts(self, weights: DebtWeights) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for claim_id, claim in self._claims.items():
            claim.debt = weights.calculate(claim)
            result[claim_id] = claim.debt
        return result

    def normalized_importance(self) -> Dict[str, float]:
        active = [
            claim
            for claim in self._claims.values()
            if claim.status is not ClaimStatus.INVALIDATED
        ]
        total = sum(claim.importance for claim in active)
        if total <= 0.0:
            return {claim.claim_id: 0.0 for claim in active}
        return {claim.claim_id: claim.importance / total for claim in active}

    def global_risk(self, weights: Optional[DebtWeights] = None) -> float:
        """Return the importance-normalized risk, independent of ledger length."""

        if weights is not None:
            self.recompute_debts(weights)
        normalized = self.normalized_importance()
        return sum(
            normalized[claim_id] * self._claims[claim_id].debt
            for claim_id in normalized
        )

    def high_risk_claims(
        self,
        debt_threshold: float,
        min_normalized_importance: float = 0.0,
    ) -> Tuple[CausalClaim, ...]:
        _require_unit_interval(debt_threshold, "debt_threshold")
        _require_unit_interval(min_normalized_importance, "min_normalized_importance")
        normalized = self.normalized_importance()
        selected = [
            claim
            for claim in self._claims.values()
            if claim.status is not ClaimStatus.INVALIDATED
            and claim.debt >= debt_threshold
            and normalized.get(claim.claim_id, 0.0) > min_normalized_importance
        ]
        selected.sort(
            key=lambda claim: (
                -claim.debt,
                -normalized.get(claim.claim_id, 0.0),
                claim.claim_id,
            )
        )
        return tuple(selected)

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "claims": [
                self._claims[claim_id].to_dict() for claim_id in sorted(self._claims)
            ],
            "dependencies": [list(edge) for edge in self.dependencies],
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(
            self.to_dict(), allow_nan=False, indent=indent, sort_keys=True
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> CausalBeliefLedger:
        version = _as_int(data.get("schema_version", 0), "schema_version")
        if version != cls.SCHEMA_VERSION:
            raise ValueError("unsupported ledger schema version %d" % version)
        raw_claims = _as_sequence(data.get("claims", []), "claims")
        claims = [
            CausalClaim.from_dict(_as_mapping(item, "claim")) for item in raw_claims
        ]
        ledger = cls(claims=claims)
        raw_dependencies = _as_sequence(data.get("dependencies", []), "dependencies")
        for raw_edge in raw_dependencies:
            edge = _as_sequence(raw_edge, "dependency")
            if len(edge) != 2 or not all(isinstance(item, str) for item in edge):
                raise ValueError("each dependency must contain two claim IDs")
            ledger.add_dependency(cast(str, edge[0]), cast(str, edge[1]))
        return ledger

    @classmethod
    def from_json(cls, payload: str) -> CausalBeliefLedger:
        try:
            decoded = json.loads(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid ledger JSON") from exc
        return cls.from_dict(_as_mapping(decoded, "ledger"))


@dataclass
class PlanningCheckpoint:
    """A checkpoint in the *logical* task plan."""

    checkpoint_id: str
    cursor: int
    subgoal: Optional[str] = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.checkpoint_id, "checkpoint_id")
        if isinstance(self.cursor, bool) or not isinstance(self.cursor, int):
            raise ValueError("cursor must be an integer")
        if self.cursor < 0:
            raise ValueError("cursor must be non-negative")
        if self.subgoal is not None:
            _require_non_empty(self.subgoal, "subgoal")
        self.metadata = _json_safe_copy(self.metadata, "metadata")

    def to_dict(self) -> Dict[str, object]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "cursor": self.cursor,
            "subgoal": self.subgoal,
            "metadata": _json_safe_copy(self.metadata, "metadata"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> PlanningCheckpoint:
        checkpoint_id = data.get("checkpoint_id")
        if not isinstance(checkpoint_id, str):
            raise ValueError("checkpoint_id must be a string")
        return cls(
            checkpoint_id=checkpoint_id,
            cursor=_as_int(data.get("cursor", 0), "cursor"),
            subgoal=_as_optional_string(data.get("subgoal"), "subgoal"),
            metadata=_json_safe_copy(
                _as_mapping(data.get("metadata", {}), "metadata"), "metadata"
            ),
        )


@dataclass
class RollbackEvent:
    """Audit record for a local logical rollback.

    ``physical_state_rolled_back`` is deliberately immutable-by-construction and
    always false.  Runtime callers must recover from the current physical state.
    """

    trigger_claim_ids: Tuple[str, ...]
    invalidated_claim_ids: Tuple[str, ...]
    from_cursor: int
    to_cursor: int
    checkpoint_id: Optional[str]
    reason: str
    physical_state_rolled_back: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.trigger_claim_ids = tuple(self.trigger_claim_ids)
        self.invalidated_claim_ids = tuple(self.invalidated_claim_ids)
        for claim_id in self.trigger_claim_ids + self.invalidated_claim_ids:
            _require_non_empty(claim_id, "claim_id")
        if self.from_cursor < 0 or self.to_cursor < 0:
            raise ValueError("rollback cursors must be non-negative")
        if self.to_cursor > self.from_cursor:
            raise ValueError("logical rollback cannot move the plan cursor forward")
        if self.checkpoint_id is not None:
            _require_non_empty(self.checkpoint_id, "checkpoint_id")
        _require_non_empty(self.reason, "reason")

    def to_dict(self) -> Dict[str, object]:
        return {
            "trigger_claim_ids": list(self.trigger_claim_ids),
            "invalidated_claim_ids": list(self.invalidated_claim_ids),
            "from_cursor": self.from_cursor,
            "to_cursor": self.to_cursor,
            "checkpoint_id": self.checkpoint_id,
            "reason": self.reason,
            "physical_state_rolled_back": False,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RollbackEvent:
        if data.get("physical_state_rolled_back", False) is not False:
            raise ValueError("physical state rollback is not supported")
        reason = data.get("reason")
        if not isinstance(reason, str):
            raise ValueError("reason must be a string")
        return cls(
            trigger_claim_ids=_as_string_tuple(
                data.get("trigger_claim_ids", []), "trigger_claim_ids"
            ),
            invalidated_claim_ids=_as_string_tuple(
                data.get("invalidated_claim_ids", []), "invalidated_claim_ids"
            ),
            from_cursor=_as_int(data.get("from_cursor", 0), "from_cursor"),
            to_cursor=_as_int(data.get("to_cursor", 0), "to_cursor"),
            checkpoint_id=_as_optional_string(
                data.get("checkpoint_id"), "checkpoint_id"
            ),
            reason=reason,
        )


@dataclass
class LogicalPlanState:
    """Serializable logical-plan state; it contains no physical-world snapshot."""

    cursor: int = 0
    active_subgoal: Optional[str] = None
    checkpoints: Dict[str, PlanningCheckpoint] = field(default_factory=dict)
    rollback_history: List[RollbackEvent] = field(default_factory=list)
    handled_refutations: Set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if isinstance(self.cursor, bool) or not isinstance(self.cursor, int):
            raise ValueError("cursor must be an integer")
        if self.cursor < 0:
            raise ValueError("cursor must be non-negative")
        if self.active_subgoal is not None:
            _require_non_empty(self.active_subgoal, "active_subgoal")
        self.checkpoints = dict(self.checkpoints)
        self.rollback_history = list(self.rollback_history)
        self.handled_refutations = set(self.handled_refutations)

    def add_checkpoint(self, checkpoint: PlanningCheckpoint) -> None:
        if checkpoint.checkpoint_id in self.checkpoints:
            raise ValueError("checkpoint %r already exists" % checkpoint.checkpoint_id)
        self.checkpoints[checkpoint.checkpoint_id] = checkpoint

    def rollback_to(
        self,
        checkpoint_id: Optional[str],
        trigger_claim_ids: Sequence[str],
        invalidated_claim_ids: Sequence[str],
        reason: str,
    ) -> RollbackEvent:
        """Move only the logical cursor; physical state is intentionally untouched."""

        from_cursor = self.cursor
        if checkpoint_id is None:
            to_cursor = from_cursor
        else:
            try:
                checkpoint = self.checkpoints[checkpoint_id]
            except KeyError as exc:
                raise KeyError("unknown checkpoint %r" % checkpoint_id) from exc
            if checkpoint.cursor > from_cursor:
                raise ValueError("cannot roll forward to a future checkpoint")
            to_cursor = checkpoint.cursor
            self.cursor = checkpoint.cursor
            self.active_subgoal = checkpoint.subgoal

        event = RollbackEvent(
            trigger_claim_ids=tuple(sorted(set(trigger_claim_ids))),
            invalidated_claim_ids=tuple(sorted(set(invalidated_claim_ids))),
            from_cursor=from_cursor,
            to_cursor=to_cursor,
            checkpoint_id=checkpoint_id,
            reason=reason,
        )
        self.rollback_history.append(event)
        return event

    def to_dict(self) -> Dict[str, object]:
        return {
            "cursor": self.cursor,
            "active_subgoal": self.active_subgoal,
            "checkpoints": [
                self.checkpoints[checkpoint_id].to_dict()
                for checkpoint_id in sorted(self.checkpoints)
            ],
            "rollback_history": [event.to_dict() for event in self.rollback_history],
            "handled_refutations": sorted(self.handled_refutations),
            "contains_physical_state": False,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(
            self.to_dict(), allow_nan=False, indent=indent, sort_keys=True
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> LogicalPlanState:
        if data.get("contains_physical_state", False) is not False:
            raise ValueError("logical plan state cannot contain physical state")
        raw_checkpoints = _as_sequence(data.get("checkpoints", []), "checkpoints")
        checkpoints_list = [
            PlanningCheckpoint.from_dict(_as_mapping(item, "checkpoint"))
            for item in raw_checkpoints
        ]
        checkpoints = {
            checkpoint.checkpoint_id: checkpoint for checkpoint in checkpoints_list
        }
        if len(checkpoints) != len(checkpoints_list):
            raise ValueError("checkpoint IDs must be unique")
        raw_history = _as_sequence(data.get("rollback_history", []), "rollback_history")
        history = [
            RollbackEvent.from_dict(_as_mapping(item, "rollback event"))
            for item in raw_history
        ]
        return cls(
            cursor=_as_int(data.get("cursor", 0), "cursor"),
            active_subgoal=_as_optional_string(
                data.get("active_subgoal"), "active_subgoal"
            ),
            checkpoints=checkpoints,
            rollback_history=history,
            handled_refutations=set(
                _as_string_tuple(
                    data.get("handled_refutations", []), "handled_refutations"
                )
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> LogicalPlanState:
        try:
            decoded = json.loads(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid logical-plan JSON") from exc
        return cls.from_dict(_as_mapping(decoded, "logical plan"))


@dataclass
class RepairCandidate:
    """Predicted result and execution costs for one repair action."""

    action_id: str
    target_claim_ids: Tuple[str, ...]
    expected_global_risk: float
    action_cost: float = 0.0
    task_risk: float = 0.0
    policy_log_probability: float = 0.0
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.action_id, "action_id")
        self.target_claim_ids = tuple(self.target_claim_ids)
        for claim_id in self.target_claim_ids:
            _require_non_empty(claim_id, "target claim ID")
        _require_unit_interval(self.expected_global_risk, "expected_global_risk")
        _require_non_negative(self.action_cost, "action_cost")
        _require_unit_interval(self.task_risk, "task_risk")
        _require_finite(self.policy_log_probability, "policy_log_probability")
        if self.policy_log_probability > 0.0:
            raise ValueError("policy_log_probability must be non-positive")
        self.metadata = _json_safe_copy(self.metadata, "metadata")

    def score(
        self,
        current_global_risk: float,
        cost_weight: float,
        risk_weight: float,
        policy_weight: float = 0.0,
    ) -> float:
        _require_unit_interval(current_global_risk, "current_global_risk")
        _require_non_negative(cost_weight, "cost_weight")
        _require_non_negative(risk_weight, "risk_weight")
        _require_non_negative(policy_weight, "policy_weight")
        return (
            current_global_risk
            - self.expected_global_risk
            - cost_weight * self.action_cost
            - risk_weight * self.task_risk
            + policy_weight * self.policy_log_probability
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "action_id": self.action_id,
            "target_claim_ids": list(self.target_claim_ids),
            "expected_global_risk": self.expected_global_risk,
            "action_cost": self.action_cost,
            "task_risk": self.task_risk,
            "policy_log_probability": self.policy_log_probability,
            "metadata": _json_safe_copy(self.metadata, "metadata"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RepairCandidate:
        action_id = data.get("action_id")
        if not isinstance(action_id, str):
            raise ValueError("action_id must be a string")
        return cls(
            action_id=action_id,
            target_claim_ids=_as_string_tuple(
                data.get("target_claim_ids", []), "target_claim_ids"
            ),
            expected_global_risk=_as_float(
                data.get("expected_global_risk", 0.0), "expected_global_risk"
            ),
            action_cost=_as_float(data.get("action_cost", 0.0), "action_cost"),
            task_risk=_as_float(data.get("task_risk", 0.0), "task_risk"),
            policy_log_probability=_as_float(
                data.get("policy_log_probability", 0.0),
                "policy_log_probability",
            ),
            metadata=_json_safe_copy(
                _as_mapping(data.get("metadata", {}), "metadata"), "metadata"
            ),
        )


@dataclass
class PlannerDecision:
    """One task, repair, or logical rollback decision."""

    decision_type: PlannerDecisionType
    action_id: Optional[str]
    reason: str
    global_risk: float
    target_claim_ids: Tuple[str, ...] = ()
    repair_score: Optional[float] = None
    rollback_event: Optional[RollbackEvent] = None

    def __post_init__(self) -> None:
        if isinstance(self.decision_type, str):
            self.decision_type = PlannerDecisionType(self.decision_type)
        if self.action_id is not None:
            _require_non_empty(self.action_id, "action_id")
        _require_non_empty(self.reason, "reason")
        _require_unit_interval(self.global_risk, "global_risk")
        self.target_claim_ids = tuple(self.target_claim_ids)
        if self.repair_score is not None:
            _require_finite(self.repair_score, "repair_score")
        if self.decision_type is PlannerDecisionType.ROLLBACK:
            if self.rollback_event is None:
                raise ValueError("rollback decisions require a rollback event")
            if self.action_id is not None:
                raise ValueError("logical rollback decisions do not execute an action")

    @property
    def physical_state_rolled_back(self) -> bool:
        return False

    def to_dict(self) -> Dict[str, object]:
        return {
            "decision_type": self.decision_type.value,
            "action_id": self.action_id,
            "reason": self.reason,
            "global_risk": self.global_risk,
            "target_claim_ids": list(self.target_claim_ids),
            "repair_score": self.repair_score,
            "rollback_event": (
                None if self.rollback_event is None else self.rollback_event.to_dict()
            ),
            "physical_state_rolled_back": False,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(
            self.to_dict(), allow_nan=False, indent=indent, sort_keys=True
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> PlannerDecision:
        if data.get("physical_state_rolled_back", False) is not False:
            raise ValueError("physical state rollback is not supported")
        decision_type = data.get("decision_type")
        reason = data.get("reason")
        if not isinstance(decision_type, str) or not isinstance(reason, str):
            raise ValueError("decision_type and reason must be strings")
        raw_event = data.get("rollback_event")
        rollback_event = (
            None
            if raw_event is None
            else RollbackEvent.from_dict(_as_mapping(raw_event, "rollback_event"))
        )
        raw_score = data.get("repair_score")
        return cls(
            decision_type=PlannerDecisionType(decision_type),
            action_id=_as_optional_string(data.get("action_id"), "action_id"),
            reason=reason,
            global_risk=_as_float(data.get("global_risk", 0.0), "global_risk"),
            target_claim_ids=_as_string_tuple(
                data.get("target_claim_ids", []), "target_claim_ids"
            ),
            repair_score=(
                None if raw_score is None else _as_float(raw_score, "repair_score")
            ),
            rollback_event=rollback_event,
        )

    @classmethod
    def from_json(cls, payload: str) -> PlannerDecision:
        try:
            decoded = json.loads(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid planner-decision JSON") from exc
        return cls.from_dict(_as_mapping(decoded, "planner decision"))


@dataclass
class SelfHealingPlannerConfig:
    """Runtime thresholds and repair-score coefficients."""

    global_risk_threshold: float = 0.5
    claim_debt_threshold: float = 0.5
    min_normalized_importance: float = 0.0
    cost_weight: float = 0.1
    risk_weight: float = 0.2
    policy_weight: float = 0.0
    minimum_repair_score: float = 0.0
    debt_weights: DebtWeights = field(default_factory=DebtWeights)
    recompute_debt: bool = True

    def __post_init__(self) -> None:
        _require_unit_interval(self.global_risk_threshold, "global_risk_threshold")
        _require_unit_interval(self.claim_debt_threshold, "claim_debt_threshold")
        _require_unit_interval(
            self.min_normalized_importance, "min_normalized_importance"
        )
        _require_non_negative(self.cost_weight, "cost_weight")
        _require_non_negative(self.risk_weight, "risk_weight")
        _require_non_negative(self.policy_weight, "policy_weight")
        _require_finite(self.minimum_repair_score, "minimum_repair_score")


class SelfHealingPlanner:
    """Debt-aware runtime arbiter for task, repair, and local rollback."""

    def __init__(self, config: Optional[SelfHealingPlannerConfig] = None) -> None:
        self.config = config or SelfHealingPlannerConfig()

    def decide(
        self,
        ledger: CausalBeliefLedger,
        plan_state: LogicalPlanState,
        task_action_id: Optional[str],
        repair_candidates: Sequence[RepairCandidate] = (),
    ) -> PlannerDecision:
        if self.config.recompute_debt:
            ledger.recompute_debts(self.config.debt_weights)
        global_risk = ledger.global_risk()

        unresolved_refuted = [
            claim
            for claim in ledger.claims.values()
            if claim.status is ClaimStatus.REFUTED
            and claim.claim_id not in plan_state.handled_refutations
        ]
        if unresolved_refuted:
            unresolved_refuted.sort(
                key=lambda claim: (-claim.importance * claim.debt, claim.claim_id)
            )
            trigger_ids = tuple(claim.claim_id for claim in unresolved_refuted)
            invalidated: Set[str] = set()
            for claim in unresolved_refuted:
                invalidated.update(ledger.invalidate_descendants(claim.claim_id))
            checkpoint_id = self._select_checkpoint(unresolved_refuted, plan_state)
            event = plan_state.rollback_to(
                checkpoint_id=checkpoint_id,
                trigger_claim_ids=trigger_ids,
                invalidated_claim_ids=tuple(invalidated),
                reason="refuted causal claim requires local recovery and replanning",
            )
            plan_state.handled_refutations.update(trigger_ids)
            return PlannerDecision(
                decision_type=PlannerDecisionType.ROLLBACK,
                action_id=None,
                reason=event.reason,
                global_risk=global_risk,
                target_claim_ids=trigger_ids,
                rollback_event=event,
            )

        high_risk = ledger.high_risk_claims(
            debt_threshold=self.config.claim_debt_threshold,
            min_normalized_importance=self.config.min_normalized_importance,
        )
        unsafe = bool(high_risk) or global_risk >= self.config.global_risk_threshold
        if not unsafe:
            if task_action_id is None:
                raise ValueError("a safe task decision requires task_action_id")
            return PlannerDecision(
                decision_type=PlannerDecisionType.TASK,
                action_id=task_action_id,
                reason="ledger risk is below task-execution thresholds",
                global_risk=global_risk,
            )

        high_risk_ids = {claim.claim_id for claim in high_risk}
        scored: List[Tuple[float, RepairCandidate]] = []
        for candidate in repair_candidates:
            unknown_targets = set(candidate.target_claim_ids) - set(ledger.claims)
            if unknown_targets:
                raise ValueError(
                    "repair candidate %r targets unknown claims: %s"
                    % (candidate.action_id, sorted(unknown_targets))
                )
            if (
                high_risk_ids
                and candidate.target_claim_ids
                and not high_risk_ids.intersection(candidate.target_claim_ids)
            ):
                continue
            score = candidate.score(
                current_global_risk=global_risk,
                cost_weight=self.config.cost_weight,
                risk_weight=self.config.risk_weight,
                policy_weight=self.config.policy_weight,
            )
            scored.append((score, candidate))

        if scored:
            scored.sort(
                key=lambda item: (
                    -item[0],
                    item[1].expected_global_risk,
                    item[1].action_id,
                )
            )
            best_score, best_candidate = scored[0]
            if best_score >= self.config.minimum_repair_score:
                return PlannerDecision(
                    decision_type=PlannerDecisionType.REPAIR,
                    action_id=best_candidate.action_id,
                    reason="repair has the best positive debt-reduction utility",
                    global_risk=global_risk,
                    target_claim_ids=best_candidate.target_claim_ids,
                    repair_score=best_score,
                )

        rollback_claims = list(high_risk)
        if not rollback_claims:
            active_claims = [
                claim
                for claim in ledger.claims.values()
                if claim.status is not ClaimStatus.INVALIDATED
            ]
            active_claims.sort(
                key=lambda claim: (-claim.importance * claim.debt, claim.claim_id)
            )
            rollback_claims = active_claims[:1]
        trigger_ids = tuple(claim.claim_id for claim in rollback_claims)
        invalidated = set()
        for claim in rollback_claims:
            invalidated.update(ledger.invalidate_descendants(claim.claim_id))
        checkpoint_id = self._select_checkpoint(rollback_claims, plan_state)
        event = plan_state.rollback_to(
            checkpoint_id=checkpoint_id,
            trigger_claim_ids=trigger_ids,
            invalidated_claim_ids=tuple(invalidated),
            reason=(
                "ledger is unsafe and no cost-effective repair is available; "
                "replan locally from the current physical state"
            ),
        )
        return PlannerDecision(
            decision_type=PlannerDecisionType.ROLLBACK,
            action_id=None,
            reason=event.reason,
            global_risk=global_risk,
            target_claim_ids=trigger_ids,
            rollback_event=event,
        )

    @staticmethod
    def _select_checkpoint(
        claims: Sequence[CausalClaim], plan_state: LogicalPlanState
    ) -> Optional[str]:
        candidates: List[Tuple[int, str]] = []
        for claim in claims:
            checkpoint_id = claim.rollback_checkpoint
            if checkpoint_id is None:
                continue
            checkpoint = plan_state.checkpoints.get(checkpoint_id)
            if checkpoint is not None and checkpoint.cursor <= plan_state.cursor:
                candidates.append((checkpoint.cursor, checkpoint_id))
        if not candidates:
            return None
        # Multiple simultaneous failures require the earliest relevant local point.
        return min(candidates)[1]


@dataclass
class LedgerRuntimeState:
    """Serializable snapshot of belief and logical-plan state only."""

    ledger: CausalBeliefLedger
    plan_state: LogicalPlanState

    SCHEMA_VERSION = 1

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "ledger": self.ledger.to_dict(),
            "plan_state": self.plan_state.to_dict(),
            "contains_physical_state": False,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(
            self.to_dict(), allow_nan=False, indent=indent, sort_keys=True
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> LedgerRuntimeState:
        version = _as_int(data.get("schema_version", 0), "schema_version")
        if version != cls.SCHEMA_VERSION:
            raise ValueError("unsupported runtime schema version %d" % version)
        if data.get("contains_physical_state", False) is not False:
            raise ValueError("runtime snapshots cannot contain physical state")
        return cls(
            ledger=CausalBeliefLedger.from_dict(
                _as_mapping(data.get("ledger"), "ledger")
            ),
            plan_state=LogicalPlanState.from_dict(
                _as_mapping(data.get("plan_state"), "plan_state")
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> LedgerRuntimeState:
        try:
            decoded = json.loads(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid runtime JSON") from exc
        return cls.from_dict(_as_mapping(decoded, "runtime state"))


__all__ = [
    "CausalBeliefLedger",
    "CausalClaim",
    "ClaimStatus",
    "DebtWeights",
    "Evidence",
    "EvidencePolarity",
    "LedgerRuntimeState",
    "LogicalPlanState",
    "PlannerDecision",
    "PlannerDecisionType",
    "PlanningCheckpoint",
    "RepairCandidate",
    "RollbackEvent",
    "SelfHealingPlanner",
    "SelfHealingPlannerConfig",
]
