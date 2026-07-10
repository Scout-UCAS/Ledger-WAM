# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""LingBot-VA package.

Subpackages are imported lazily so lightweight utilities (for example the
Ledger-WAM runtime) do not require the full CUDA/diffusers training stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["configs", "distributed", "ledger", "modules"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
