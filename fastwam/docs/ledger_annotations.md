# Ledger-WAM annotation schema

Ledger-WAM always computes a 16-D explicit evidence vector and dense weak labels
when `ledger_auto_annotate=true`. A JSON/JSONL sidecar can replace those targets
with privileged simulator or human annotations.

## Sparse records

Compile sparse annotations with:

```bash
PYTHONPATH=src:. python scripts/build_ledger_annotations.py \
  --input data/ledger_events.jsonl \
  --output data/ledger_annotations.jsonl
```

Each record has a post-split LeRobot `idx`, sparse claims, and optional candidate
repair outcomes:

```json
{
  "idx": 42,
  "claims": [
    {
      "name": "grasped",
      "truth": true,
      "confidence": 0.8,
      "uncertainty": 0.2,
      "dependency": 0.95,
      "repair_cost": 0.4,
      "observability": 0.3,
      "entities": ["end_effector", "target_object"],
      "relation": "contact",
      "precondition": "contact",
      "effect": "persistent",
      "rollback_step": 3
    }
  ],
  "repair": {
    "type": "verify",
    "debt_reduction": [0.7, 0.4, 0.3, 0.6],
    "risk": [0.05, 0.1, 0.25, 0.35]
  }
}
```

Missing claims are masked, not treated as false. `repair.action`, when present,
must use the processed normalized action representation and full model horizon.

## Dense simulator traces

`SimulatorEventParser` produces timeline entries containing `claims`, `evidence`,
event source, object/end-effector motion, contact distance, and co-motion. Store
one JSONL record per dataset window:

```json
{
  "idx": 42,
  "timeline": [
    {"claims": [1, 0, 1, 0, 1, 1, 1, 0], "evidence": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]}
  ],
  "repair_outcomes": [
    {"type": "regrasp", "debt_before": 0.8, "debt_after": 0.2, "failure": false}
  ]
}
```

Compile it with `scripts/compile_simulator_events.py`. The output includes four
temporal steps of claim, debt, evidence-strength, K-step dependency, rollback,
relation, entity, precondition, and effect supervision, plus next-ledger and
candidate-repair targets.

Supported claims are `contact`, `grasped`, `supported`, `contained`, `visible`,
`persistent`, `precondition_met`, and `effect_achieved`. Relations, semantic
entities, repair names, temporal steps, dynamic slot count, and dependency horizon
are configured in `configs/model/ledger_wam.yaml`.
