"""Default Ledger-WAM configuration.

The existing project uses mutable ``EasyDict`` instances.  This module keeps
the defaults as plain Python data so importing it does not require EasyDict and
so callers can safely copy the values into any mapping- or attribute-based
configuration object.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional


LEDGER_REPAIR_CATALOG = (
    {
        "id": 0,
        "name": "viewpoint_adjust",
        "description": "Move the camera or wrist to reveal an occluded relation.",
        "cost": 0.10,
        "risk": 0.05,
    },
    {
        "id": 1,
        "name": "lift_test",
        "description": "Lift slightly to verify grasp contact and object following.",
        "cost": 0.15,
        "risk": 0.10,
    },
    {
        "id": 2,
        "name": "short_retreat",
        "description": "Retreat a short distance before retrying a local action.",
        "cost": 0.15,
        "risk": 0.08,
    },
    {
        "id": 3,
        "name": "reclose_gripper",
        "description": "Close the gripper again to restore grasp force.",
        "cost": 0.12,
        "risk": 0.12,
    },
    {
        "id": 4,
        "name": "tactile_check",
        "description": "Use available force or tactile sensing to verify contact.",
        "cost": 0.08,
        "risk": 0.03,
    },
    {
        "id": 5,
        "name": "realign",
        "description": "Locally realign the end effector and target geometry.",
        "cost": 0.20,
        "risk": 0.12,
    },
    {
        "id": 6,
        "name": "local_regrasp",
        "description": "Roll back to the grasp stage and execute a local regrasp.",
        "cost": 0.35,
        "risk": 0.20,
    },
)


LEDGER_LOSS_WEIGHTS = {
    # Existing LingBot-VA objectives.
    "video": 1.0,
    "action": 1.0,
    # Ledger-WAM objectives from the paper draft.
    "claim": 1.0,
    "presence": 0.5,
    "claim_type": 0.25,
    "subject": 0.25,
    "object": 0.25,
    "relation": 0.5,
    "precondition": 0.25,
    "effect": 0.25,
    "evidence": 0.5,
    "uncertainty": 0.25,
    "dependency": 0.5,
    "dependency_matrix": 0.5,
    "debt": 1.0,
    "repair_cost": 0.25,
    "observability": 0.25,
    "importance": 0.25,
    "rollback": 0.5,
    "repair_action": 1.0,
    "action_cost": 0.25,
    "post_repair_debt": 0.5,
    "counterfactual": 0.25,
    "debt_reward": 0.25,
    "recurrent": 0.25,
}


# Positive coefficients implement the monotonic debt definition described in
# prose: uncertainty, downstream dependency, repair cost, and lack of
# observability should all increase causal debt.
LEDGER_DEBT_WEIGHTS = {
    "lack_of_confidence": 1.0,
    "uncertainty": 1.0,
    "dependency": 1.0,
    "repair_cost": 0.5,
    "lack_of_observability": 0.75,
}


LEDGER_CONFIG_DEFAULTS: Dict[str, Any] = {
    # The opt-in switch guarantees the legacy dataset output is unchanged.
    "ledger_enabled": False,
    # None resolves to <LeRobot dataset root>/meta/ledger_annotations.jsonl.
    "ledger_annotation_path": None,
    "ledger_strict": False,
    "ledger_ignore_index": -100,
    "ledger_max_claims": 16,
    "ledger_max_counterfactuals": 4,
    "ledger_action_dim": 30,
    "dataset_init_workers": 8,
    # Temporal/object representation defaults for the model implementation.
    # The current post-training dataset contains a future segment rather than
    # an explicit past-history tensor.  Restrict the side head to the first
    # observed frame to avoid leaking future actions through global pooling.
    "ledger_history_horizon": 1,
    "ledger_hidden_dim": 512,
    "ledger_num_heads": 8,
    "ledger_dropout": 0.1,
    "ledger_num_relations": 12,
    "ledger_num_claim_types": 8,
    "ledger_num_subjects": 64,
    "ledger_num_objects": 64,
    "ledger_num_preconditions": 16,
    "ledger_num_effects": 16,
    "ledger_delta_dim": 512,
    "ledger_max_rollback_stages": 16,
    "ledger_rollback_stage_ontology": (
        "task_start",
        "search",
        "approach",
        "pre_grasp",
        "grasp",
        "post_grasp",
        "transport",
        "pre_open",
        "open",
        "post_open",
        "pre_place",
        "place",
        "post_place",
        "verify",
        "recover",
        "current_subgoal",
    ),
    "ledger_relation_ontology": (
        "contact",
        "support",
        "contain",
        "occlude",
        "left_of",
        "right_of",
        "above",
        "below",
        "near",
        "grasped_by",
        "moves_with",
        "open",
    ),
    # Claim- and ledger-level decision thresholds from Sec. 3.3.
    "ledger_debt_threshold": 0.60,  # delta
    "ledger_importance_threshold": 0.05,  # normalized epsilon
    "ledger_global_risk_threshold": 0.50,  # tau
    "ledger_confidence_threshold": 0.30,  # kappa
    # Counterfactual separation and repair scoring S(a).
    # Margin on normalized L1 delta distance (mean over delta dimensions and
    # active annotated claim slots), independent of S and delta width.
    "ledger_counterfactual_margin": 1.0,
    "ledger_repair_cost_weight": 0.10,  # beta
    "ledger_repair_risk_weight": 0.20,  # gamma
    "ledger_repair_policy_prior_weight": 0.05,
    "ledger_repair_cost_scale": 1.0,
    "ledger_minimum_repair_score": 0.0,
    "ledger_verification_threshold": 0.8,
    "ledger_dependency_edge_threshold": 0.5,
    "ledger_external_evidence_weight": 0.7,
    "ledger_allow_random_head": False,
    # Production environments must map discrete repair skills to robot-specific
    # continuous chunks.  Prompt-only fallback is explicitly opt-in.
    "ledger_allow_prompt_repair_fallback": False,
    "ledger_counterfactual_sample_probability": 0.5,
    # Training schedule knobs.  A caller may freeze the large WAM backbone
    # while warming up newly initialized ledger heads.
    "ledger_head_warmup_steps": 1_000,
    "ledger_freeze_backbone_during_warmup": True,
    "ledger_loss_weights": LEDGER_LOSS_WEIGHTS,
    "ledger_debt_weights": LEDGER_DEBT_WEIGHTS,
    "ledger_repair_catalog": LEDGER_REPAIR_CATALOG,
}


def get_ledger_config(
    enabled: Optional[bool] = None, **overrides: Any
) -> Dict[str, Any]:
    """Return an isolated dictionary of Ledger-WAM defaults."""

    values = deepcopy(LEDGER_CONFIG_DEFAULTS)
    if enabled is not None:
        values["ledger_enabled"] = bool(enabled)
    unknown = sorted(set(overrides) - set(values))
    if unknown:
        raise KeyError("Unknown Ledger-WAM config keys: {}".format(unknown))
    values.update(overrides)
    return values


def apply_ledger_defaults(
    config: Any, enabled: Optional[bool] = None, **overrides: Any
) -> Any:
    """Copy Ledger-WAM defaults into an EasyDict, dict, or config object.

    The same object is returned to support the style used by existing config
    modules.  Nested values are deep-copied to avoid cross-config mutation.
    """

    values = get_ledger_config(enabled=enabled, **overrides)
    if isinstance(config, dict):
        config.update(values)
    else:
        for key, value in values.items():
            setattr(config, key, value)
    return config


__all__ = [
    "LEDGER_CONFIG_DEFAULTS",
    "LEDGER_DEBT_WEIGHTS",
    "LEDGER_LOSS_WEIGHTS",
    "LEDGER_REPAIR_CATALOG",
    "apply_ledger_defaults",
    "get_ledger_config",
]
