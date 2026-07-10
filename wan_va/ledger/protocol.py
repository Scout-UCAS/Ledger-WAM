"""Repair-action handoff protocol shared by Ledger-WAM serving code.

The planner may decide that a repair is needed, but that decision is not proof
that a physical action was available or executed.  This module keeps those
states separate and is intentionally independent from the model runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class IssuedRepairAction:
    """A concrete repair chunk returned to a controller but not yet acknowledged."""

    action_id: str
    target_claim_ids: Tuple[str, ...]
    source: str
    issued_at: int
    execution_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "target_claim_ids": list(self.target_claim_ids),
            "source": self.source,
            "issued_at": self.issued_at,
            "execution_id": self.execution_id,
        }


@dataclass(frozen=True)
class RepairExecutionAcknowledgement:
    """Validated acknowledgement for a previously issued repair action."""

    action_id: str
    success: bool
    target_claim_ids: Tuple[str, ...]
    execution_id: str
    implicit: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "success": self.success,
            "target_claim_ids": list(self.target_claim_ids),
            "execution_id": self.execution_id,
            "implicit": self.implicit,
        }


class RepairExecutionTracker:
    """Track one in-flight physical repair without assuming it was executed."""

    def __init__(self) -> None:
        self._outstanding: Optional[IssuedRepairAction] = None

    @property
    def outstanding(self) -> Optional[IssuedRepairAction]:
        return self._outstanding

    def reset(self) -> None:
        self._outstanding = None

    def issue(
        self,
        action_id: str,
        target_claim_ids: Sequence[str],
        *,
        source: str,
        issued_at: int,
    ) -> IssuedRepairAction:
        if self._outstanding is not None:
            raise RuntimeError(
                "the previous repair action must be acknowledged before issuing "
                "another one"
            )
        action_id = str(action_id).strip()
        if not action_id:
            raise ValueError("repair action_id must be non-empty")
        source = str(source).strip()
        if not source:
            raise ValueError("repair source must be non-empty")
        record = IssuedRepairAction(
            action_id=action_id,
            target_claim_ids=tuple(str(item) for item in target_claim_ids),
            source=source,
            issued_at=int(issued_at),
            execution_id=uuid4().hex,
        )
        self._outstanding = record
        return record

    def acknowledge(
        self,
        payload: Optional[Mapping[str, Any]] = None,
        *,
        implicit_success: bool = False,
    ) -> Optional[RepairExecutionAcknowledgement]:
        """Consume an explicit acknowledgement or a legacy implicit success.

        An implicit success is used only when an older controller uploads the
        executed action/state in its cache-update request.  Supplying an
        explicit acknowledgement always takes precedence.
        """

        if payload is None and not implicit_success:
            return None
        if payload is not None and not isinstance(payload, Mapping):
            raise TypeError("repair_execution_ack must be a mapping")
        if self._outstanding is None:
            if payload is None:
                return None
            raise ValueError("repair_execution_ack has no outstanding repair action")

        outstanding = self._outstanding
        if payload is None:
            implicit = True
            action_id = outstanding.action_id
            success = True
        else:
            implicit = False
            if (
                "action_id" not in payload
                or "execution_id" not in payload
                or "success" not in payload
            ):
                raise ValueError(
                    "repair_execution_ack requires action_id, execution_id, "
                    "and success"
                )
            action_id = str(payload["action_id"])
            if action_id != outstanding.action_id:
                raise ValueError(
                    "repair_execution_ack action_id {!r} does not match "
                    "outstanding action {!r}".format(action_id, outstanding.action_id)
                )
            execution_id = str(payload["execution_id"])
            if execution_id != outstanding.execution_id:
                raise ValueError(
                    "repair_execution_ack execution_id does not match the "
                    "outstanding repair action"
                )
            success_value = payload["success"]
            if type(success_value) is not bool:
                raise TypeError("repair_execution_ack success must be a boolean")
            success = success_value

        self._outstanding = None
        return RepairExecutionAcknowledgement(
            action_id=outstanding.action_id,
            success=success,
            target_claim_ids=outstanding.target_claim_ids,
            execution_id=outstanding.execution_id,
            implicit=implicit,
        )


def validate_repair_action_chunk(
    value: Any,
    *,
    expected_channels: int,
    expected_frames: int,
    actions_per_frame: int,
) -> NDArray[np.float32]:
    """Validate a controller/model repair chunk before it can be issued."""

    action = np.asarray(value)
    expected_shape = (
        int(expected_channels),
        int(expected_frames),
        int(actions_per_frame),
    )
    if action.shape != expected_shape:
        raise ValueError(
            "repair action chunk must have shape {}, got {}".format(
                expected_shape, action.shape
            )
        )
    if not np.issubdtype(action.dtype, np.number):
        raise TypeError("repair action chunk must contain numeric values")
    if not bool(np.isfinite(action).all()):
        raise ValueError("repair action chunk must contain only finite values")
    action = action.astype(np.float32, copy=False)
    if not bool(np.isfinite(action).all()):
        raise ValueError("repair action chunk values must fit in float32")
    return action


def repair_execution_ack_required(payload: Mapping[str, Any]) -> bool:
    """Return True for either supported repair-ack response spelling."""

    return bool(
        payload.get("repair_execution_ack_required")
        or payload.get("requires_repair_execution_ack")
    )


def validate_repair_catalog(
    catalog: Sequence[Mapping[str, Any]],
) -> None:
    """Require stable list-index semantics for repair-action labels."""

    if not catalog:
        raise ValueError("repair catalog must contain at least one action")
    seen_names = set()
    for index, entry in enumerate(catalog):
        if not isinstance(entry, Mapping):
            raise TypeError("every repair catalog entry must be a mapping")
        action_id = entry.get("id")
        if isinstance(action_id, bool) or not isinstance(action_id, int):
            raise TypeError("repair catalog ids must be integers")
        if action_id != index:
            raise ValueError(
                "repair catalog ids must equal their list positions; "
                "expected {}, got {}".format(index, action_id)
            )
        name = str(entry.get("name", "")).strip()
        if not name:
            raise ValueError("repair catalog names must be non-empty")
        if name in seen_names:
            raise ValueError("repair catalog names must be unique: {!r}".format(name))
        seen_names.add(name)
