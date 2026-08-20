from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np


def _flatten(payload: Any, prefix: str = "") -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten(value, child))
    else:
        try:
            value = np.asarray(payload)
        except Exception:
            return output
        if value.dtype.kind in "biufc" and value.size:
            output[prefix.lower()] = value.astype(np.float32, copy=False)
    return output


@dataclass
class SimulatorEventParser:
    """Extracts privileged simulator events and online policy evidence.

    The parser is key-name tolerant so it works with LIBERO, RoboTwin, and VLABench
    observations without importing those heavyweight simulators.
    """

    contact_distance: float = 0.07
    containment_distance: float = 0.12
    previous_flat: dict[str, np.ndarray] = field(default_factory=dict)
    last_action: Optional[np.ndarray] = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.previous_flat.clear()
        self.last_action = None
        self.events.clear()

    @staticmethod
    def _position_items(flat: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        items = {}
        for key, value in flat.items():
            if value.size < 3:
                continue
            if any(token in key for token in ("image", "rgb", "depth", "matrix", "quat")):
                continue
            if key.endswith(("pos", "position", "xyz")) or "_pos" in key:
                items[key] = value.reshape(-1)[:3]
        return items

    @staticmethod
    def _eef_key(positions: Mapping[str, np.ndarray]) -> Optional[str]:
        for token in ("eef", "end_effector", "ee_pos", "gripper_pos", "tcp"):
            for key in positions:
                if token in key:
                    return key
        return None

    def observe(
        self,
        observation: Mapping[str, Any],
        *,
        image: Optional[np.ndarray] = None,
        proposed_action: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        flat = _flatten(observation)
        positions = self._position_items(flat)
        eef_key = self._eef_key(positions)
        eef = positions.get(eef_key) if eef_key is not None else None
        objects = {
            key: value
            for key, value in positions.items()
            if key != eef_key and not any(
                token in key for token in ("robot", "joint", "camera", "base", "mocap")
            )
        }
        distances = (
            [float(np.linalg.norm(value - eef)) for value in objects.values()]
            if eef is not None
            else []
        )
        min_distance = min(distances, default=1.0)
        contact = min_distance <= self.contact_distance

        object_motion = []
        for key, value in objects.items():
            previous = self.previous_flat.get(key)
            if previous is not None and previous.size >= 3:
                object_motion.append(float(np.linalg.norm(value - previous.reshape(-1)[:3])))
        eef_motion = 0.0
        if eef_key is not None and eef_key in self.previous_flat:
            eef_motion = float(
                np.linalg.norm(eef - self.previous_flat[eef_key].reshape(-1)[:3])
            )
        co_motion = max(object_motion, default=0.0)
        co_motion_score = float(np.exp(-abs(co_motion - eef_motion) * 30.0))
        gripper_values = [
            value.reshape(-1)[-1]
            for key, value in flat.items()
            if "gripper" in key and value.size <= 32
        ]
        gripper = float(gripper_values[0]) if gripper_values else 0.0
        grasped = bool(contact and (co_motion_score > 0.55 or abs(gripper) > 0.3))

        container_positions = [
            value
            for key, value in objects.items()
            if any(token in key for token in ("container", "bowl", "box", "drawer", "basket"))
        ]
        target_positions = [
            value
            for key, value in objects.items()
            if value is not None and not any(
                token in key for token in ("container", "bowl", "box", "drawer", "basket")
            )
        ]
        contained = any(
            np.linalg.norm(target - container) <= self.containment_distance
            for target in target_positions
            for container in container_positions
        )
        visible = bool(objects) or image is not None
        persistent = bool(objects) and any(key in self.previous_flat for key in objects)
        supported = bool(objects) and max(object_motion, default=0.0) < 0.015 and not grasped
        precondition_met = visible and (contact or supported or contained)
        effect_achieved = bool(contained or (grasped and co_motion_score > 0.6))
        claims = np.asarray(
            [contact, grasped, supported, contained, visible, persistent, precondition_met, effect_achieved],
            dtype=np.float32,
        )

        action = np.asarray(
            proposed_action if proposed_action is not None else (
                self.last_action if self.last_action is not None else np.zeros(7)
            ),
            dtype=np.float32,
        ).reshape(-1)
        image_array = None if image is None else np.asarray(image, dtype=np.float32)
        brightness = float(image_array.mean() / 255.0) if image_array is not None else 0.0
        contrast = float(image_array.std() / 255.0) if image_array is not None else 0.0
        action_magnitude = float(np.linalg.norm(action))
        translation = float(np.linalg.norm(action[:3])) if action.size >= 3 else action_magnitude
        rotation = float(np.linalg.norm(action[3:6])) if action.size >= 6 else 0.0
        action_change = (
            float(np.linalg.norm(action - self.last_action[: action.size]))
            if self.last_action is not None and self.last_action.size >= action.size
            else 0.0
        )
        evidence = np.asarray(
            [
                max(object_motion, default=0.0),
                co_motion,
                brightness,
                contrast,
                contrast,
                eef_motion,
                co_motion,
                action_magnitude,
                float(np.abs(action).max(initial=0.0)),
                translation,
                rotation,
                float(action[-1]) if action.size else gripper,
                action_change,
                co_motion_score,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )
        event = {
            "claims": claims.tolist(),
            "contact": contact,
            "grasped": grasped,
            "contained": bool(contained),
            "visible": visible,
            "persistent": persistent,
            "min_object_eef_distance": min_distance,
            "object_motion": max(object_motion, default=0.0),
            "eef_motion": eef_motion,
            "co_motion_score": co_motion_score,
            "source": "simulator_state" if positions else "observation_heuristic",
            "evidence": evidence.tolist(),
        }
        self.previous_flat = flat
        if proposed_action is not None:
            self.last_action = action
        self.events.append(event)
        return evidence, event
