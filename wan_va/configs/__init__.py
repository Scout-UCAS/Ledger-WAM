# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from .va_franka_cfg import va_franka_cfg
from .va_robotwin_cfg import va_robotwin_cfg
from .va_franka_i2va import va_franka_i2va_cfg
from .va_robotwin_i2va import va_robotwin_i2va_cfg
from .va_robotwin_train_cfg import va_robotwin_train_cfg
from .va_demo_train_cfg import va_demo_train_cfg
from .va_demo_cfg import va_demo_cfg
from .va_demo_i2va import va_demo_i2va_cfg
from .va_libero_cfg import va_libero_cfg
from .va_libero_train_cfg import va_libero_train_cfg
from .va_libero_i2va import va_libero_i2va_cfg
from copy import deepcopy

from .ledger_config import apply_ledger_defaults


_BASE_CONFIGS = (
    va_franka_cfg,
    va_robotwin_cfg,
    va_franka_i2va_cfg,
    va_robotwin_i2va_cfg,
    va_robotwin_train_cfg,
    va_demo_cfg,
    va_demo_train_cfg,
    va_demo_i2va_cfg,
    va_libero_cfg,
    va_libero_train_cfg,
    va_libero_i2va_cfg,
)
for _config in _BASE_CONFIGS:
    apply_ledger_defaults(_config, enabled=False)


def _ledger_variant(config, name):
    variant = deepcopy(config)
    variant.__name__ = name
    apply_ledger_defaults(variant, enabled=True)
    return variant


ledger_robotwin_cfg = _ledger_variant(va_robotwin_cfg, "Config: Ledger-WAM RoboTwin")
ledger_franka_cfg = _ledger_variant(va_franka_cfg, "Config: Ledger-WAM Franka")
ledger_demo_cfg = _ledger_variant(va_demo_cfg, "Config: Ledger-WAM demo")
ledger_libero_cfg = _ledger_variant(va_libero_cfg, "Config: Ledger-WAM LIBERO")
ledger_robotwin_train_cfg = _ledger_variant(
    va_robotwin_train_cfg, "Config: Ledger-WAM RoboTwin train"
)
ledger_demo_train_cfg = _ledger_variant(
    va_demo_train_cfg, "Config: Ledger-WAM demo train"
)
ledger_libero_train_cfg = _ledger_variant(
    va_libero_train_cfg, "Config: Ledger-WAM LIBERO train"
)

for _train_config in (
    ledger_robotwin_train_cfg,
    ledger_demo_train_cfg,
    ledger_libero_train_cfg,
):
    # Opt in explicitly after credentials are configured; placeholder
    # credentials should never make a local Ledger-WAM run fail at startup.
    _train_config.enable_wandb = False
    # A Ledger training run without sidecar coverage would silently optimize
    # only the legacy losses and leave the planner random.
    _train_config.ledger_strict = True

for _server_config in (
    ledger_robotwin_cfg,
    ledger_franka_cfg,
    ledger_demo_cfg,
    ledger_libero_cfg,
):
    _server_config.infer_mode = "server"

VA_CONFIGS = {
    "robotwin": va_robotwin_cfg,
    "franka": va_franka_cfg,
    "robotwin_i2av": va_robotwin_i2va_cfg,
    "franka_i2av": va_franka_i2va_cfg,
    "robotwin_train": va_robotwin_train_cfg,
    "demo": va_demo_cfg,
    "demo_train": va_demo_train_cfg,
    "demo_i2av": va_demo_i2va_cfg,
    "libero": va_libero_cfg,
    "libero_train": va_libero_train_cfg,
    "libero_i2av": va_libero_i2va_cfg,
    "ledger_robotwin": ledger_robotwin_cfg,
    "ledger_franka": ledger_franka_cfg,
    "ledger_demo": ledger_demo_cfg,
    "ledger_libero": ledger_libero_cfg,
    "ledger_robotwin_train": ledger_robotwin_train_cfg,
    "ledger_demo_train": ledger_demo_train_cfg,
    "ledger_libero_train": ledger_libero_train_cfg,
}
