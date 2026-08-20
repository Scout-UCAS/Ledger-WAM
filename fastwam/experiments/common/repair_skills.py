from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np


def apply_cartesian_repair(
    action_chunk: np.ndarray,
    repair_name: Optional[str],
    *,
    current_state: Optional[np.ndarray] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    """Environment-safe delta-pose repair library for LIBERO/VLABench."""

    action = np.asarray(action_chunk, dtype=np.float32).copy()
    if not repair_name or repair_name == "verify":
        return action
    cfg = dict(config or {})
    horizon = action.shape[0]
    if repair_name == "hold":
        gripper = action[0, -1] if action.shape[1] else 0.0
        action.fill(0.0)
        if action.shape[1]:
            action[:, -1] = gripper
    elif repair_name == "retract":
        gripper = action[0, -1] if action.shape[1] else 0.0
        action.fill(0.0)
        if action.shape[1] >= 3:
            action[:, 2] = float(cfg.get("retract_z", 0.04))
            action[:, :2] = -float(cfg.get("retract_xy_scale", 0.25)) * action_chunk[:1, :2]
        if action.shape[1]:
            action[:, -1] = gripper
    elif repair_name == "regrasp":
        action.fill(0.0)
        midpoint = max(1, horizon // 2)
        if action.shape[1]:
            action[:midpoint, -1] = float(cfg.get("open_value", 1.0))
            action[midpoint:, -1] = float(cfg.get("close_value", -1.0))
        if action.shape[1] >= 3:
            action[:midpoint, 2] = float(cfg.get("lift_z", 0.02))
            action[midpoint:, 2] = -float(cfg.get("approach_z", 0.015))
    return np.clip(action, float(cfg.get("min_action", -1.0)), float(cfg.get("max_action", 1.0)))


def apply_joint_repair(
    action_chunk: np.ndarray,
    repair_name: Optional[str],
    *,
    current_qpos: np.ndarray,
    config: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    """Joint-space repair library for RoboTwin dual-arm qpos control."""

    action = np.asarray(action_chunk, dtype=np.float32).copy()
    current = np.asarray(current_qpos, dtype=np.float32).reshape(-1)
    if not repair_name or repair_name == "verify":
        return action
    cfg = dict(config or {})
    if action.shape[1] != current.size:
        return action
    if repair_name == "hold":
        action[:] = current
    elif repair_name == "retract":
        scale = float(cfg.get("retract_scale", 0.35))
        action[:] = current[None] + scale * (current[None] - action)
    elif repair_name == "regrasp":
        action[:] = current
        indices = tuple(int(value) for value in cfg.get("gripper_indices", (6, 13)))
        midpoint = max(1, action.shape[0] // 2)
        for index in indices:
            if 0 <= index < action.shape[1]:
                action[:midpoint, index] = float(cfg.get("open_value", 1.0))
                action[midpoint:, index] = float(cfg.get("close_value", 0.0))
    return action
