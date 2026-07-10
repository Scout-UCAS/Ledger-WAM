# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Dataset APIs with optional LeRobot imports kept lazy."""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = ["MultiLatentLeRobotDataset"]


def __getattr__(name: str) -> Any:
    if name == "MultiLatentLeRobotDataset":
        value = import_module(
            ".lerobot_latent_dataset", __name__
        ).MultiLatentLeRobotDataset
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
