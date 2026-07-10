# Ledger-WAM implementation guide

This repository extends LingBot-VA with the trainable and runtime components
described in the Ledger-WAM draft. The implementation is opt-in: legacy
configurations keep `ledger_enabled=False`, while configuration names prefixed
with `ledger_` enable the new path.

## What is implemented

- A recurrent neural causal-ledger head over observed video latents, past
  actions, and language embeddings.
- Claim slots with learned presence, identity-preserving matching to the
  previous ledger, entity/relation/precondition/effect heads, evidence,
  confidence, uncertainty, dependency, observability, repair cost, importance,
  debt, and rollback predictions.
- A directed claim dependency graph and pairwise dependency supervision.
- A monotone causal-debt estimator. Low confidence, uncertainty, downstream
  dependency, recovery cost, and unobservability increase debt by construction.
- Importance-normalized global risk, independent of the number of active claims.
- Repair transition and policy heads that estimate post-repair debt, action
  cost, signed repair utility (including harmful repairs), and counterfactual
  ledger changes.
- A shared Siamese action-to-ledger transition head. Factual and
  counterfactual actions cannot satisfy the margin through separate-head bias.
- Ledger-conditioned action tokens. The adapter is zero-initialized, does not
  alter the existing causal token layout or KV-cache capacity, and is gated off
  on samples without annotations.
- A serializable runtime ledger, evidence fusion, dependency invalidation,
  targeted repair selection, and local logical rollback.
- Metrics for root-cause localization, debt calibration, rollback, repair
  efficiency, and local-replanning ratio.

Physical state is never rolled back. A rollback only invalidates dependent
beliefs and moves the logical plan cursor; the robot must execute a recovery
action from its current real state.

## Causal debt

The draft's prose and displayed debt equation use inconsistent signs. The code
uses the monotone form:

```text
d_i = sigmoid(
    b
    + w_s * (1 - confidence_i)
    + w_u * uncertainty_i
    + w_d * dependency_i
    + w_r * repair_cost_i
    + w_o * (1 - observability_i)
)
```

All five weights are constrained positive. Global risk is:

```text
D(L) = sum_i presence_i * importance_i * debt_i
       / sum_i presence_i * importance_i
```

The neural debt is supervised directly and retained by the online planner.
External evidence that changes confidence triggers debt recomputation with the
same learned positive weights and bias, so current and post-repair risk remain
on one calibrated scale.

## Annotation sidecar

Ledger post-training configurations are strict by default. Each LeRobot dataset
must contain `meta/ledger_annotations.jsonl`, or
`ledger_annotation_path` must point to a JSON/JSONL file.

Records are keyed by `episode_index:start_frame:end_frame`:

See also the ready-to-copy
[`example/ledger_annotations.example.jsonl`](example/ledger_annotations.example.jsonl).

```json
{
  "key": "12:30:90",
  "claims": [
    {
      "claim": 1.0,
      "claim_type": 1,
      "subject": 3,
      "object": 7,
      "relation": 9,
      "precondition": 4,
      "effect": 2,
      "evidence": 0.8,
      "uncertainty": 0.2,
      "dependency": 0.9,
      "observability": 0.4,
      "repair_cost": 0.35,
      "importance": 1.0,
      "debt": 0.75,
      "rollback": 4,
      "repair_action": 1,
      "post_repair_debt": 0.15
    }
  ],
  "dependency_edges": [
    {"source": 0, "target": 1, "weight": 1.0}
  ],
  "counterfactual_actions": [
    {"action": [0.0, 0.1], "delta": [1.0]}
  ]
}
```

`dependency_edges` is an adjacency-list annotation: for claim slots present in
the record, omitted directed pairs are supervised as non-edges; padded slots
remain ignored. Use a dense `dependency_matrix` when individual pairs should be
left unknown through omission.

All fields are optional and masked independently. Missing labels use `-100` and
never become negative examples. `counterfactual_actions[].action` contains up
to 30 normalized, model-aligned current-action values; omitted components retain
their observed value. It changes only the current action position, not the full
trajectory, and invalid embodiment channels remain masked. Raw repair costs are divided by
`ledger_repair_cost_scale` and clipped to `[0, 1]` before neural supervision.
Repair reward supervision follows the paper formula
`debt_reduction - beta * action_cost - gamma * task_risk`, using
`ledger_repair_cost_weight`, `ledger_repair_risk_weight`, and the configured
repair catalog costs/risks.
An explicit alternative action without per-claim deltas still receives a
whole-ledger contrastive target over annotated claim slots. When deltas are
present, their masks provide affected/unaffected claim supervision. The
counterfactual margin is measured with normalized L1 distance, making it
independent of claim-slot count and delta width.

Repair catalog entries use fixed list-index labels. Their `id` values must be
exactly `0, 1, ..., R-1` in list order, and names must be unique; startup fails
early if this invariant is violated.

The rollback label indexes the fixed `ledger_rollback_stage_ontology`. It does
not index a dynamically growing list of execution chunks. Environment adapters
should register checkpoints using those stable stage IDs.

## Training

Set dataset/model paths in the existing configuration files, then run:

```bash
CONFIG_NAME=ledger_robotwin_train \
  bash script/run_va_posttrain.sh \
  --ledger-annotation-path /path/to/ledger_annotations.jsonl \
  --ledger-strict
```

Use `ledger_libero_train` or `ledger_demo_train` for the other environments.

The trainer:

1. Loads a released LingBot-VA checkpoint with low-memory loading disabled so
   newly initialized Ledger parameters can be materialized.
2. Trains the Ledger head while the WAM backbone learning rate is held at zero
   for `ledger_head_warmup_steps` when configured.
3. Optimizes video/action flow matching together with presence, structure,
   confidence/evidence, dependency, debt, rollback, counterfactual, repair,
   transition, cost, and debt-reduction objectives.
4. Saves the transformer, Ledger parameters, optimizer, scheduler, and ontology
   configuration in each checkpoint.

Legacy segment tensors contain future targets, not explicit history. To prevent
causal leakage, the Ledger head only sees their first observation/action frame.
The second segment frame, when available, is used only as a stopped-gradient
self-distillation target for the recurrent slot updater and is never injected
into current action tokens. Multiple frames enter the main Ledger context only
when a dataset adapter supplies explicit `history_latents` and
`history_actions` tensors.

## Online server contract

Use a trained Ledger checkpoint with a `ledger_*` server configuration. The
server rejects a legacy checkpoint whose `config.json` has no trained Ledger
head unless `ledger_allow_random_head=True` is explicitly set for debugging.

Reset may include stable logical checkpoints or a serialized runtime state:

```python
client.infer({
    "reset": True,
    "prompt": "put the cup in the box",
    "planning_checkpoints": [
        {"checkpoint_id": "task_start", "cursor": 0},
        {"checkpoint_id": "grasp", "cursor": 3, "subgoal": "grasp"}
    ]
})
```

Every observation request may include simulator/event-parser updates:

```python
request = {
    "obs": observation,
    "ledger_claims": [serialized_claim],
    "ledger_dependencies": [["grasp", "transport"]],
    "ledger_evidence": [
        {
            "claim_id": "grasp",
            "source": "tactile_sensor",
            "polarity": "contradicts",
            "strength": 0.9,
            "timestamp": 12
        }
    ]
}
response = client.infer(request)
```

The response contains `planner`, `ledger`, and `logical_plan`. For a repair or
rollback decision, production environments must map the selected discrete skill
to a physical action chunk:

```python
request["repair_action_chunks"] = {
    "lift_test": lift_test_action_chunk,
    "local_rollback": local_recovery_action_chunk
}
```

Action chunks use the normal server output layout:
`[used_action_channels, frame_chunk_size, actions_per_frame]` in physical action
units. Shape, numeric dtype, and finite values are validated before dispatch.
If the required mapping is absent, the server returns:

```python
{
    "action": None,
    "requires_repair_action": True,
    "repair_action_id": "lift_test",
    "repair_instruction": "..."
}
```

This fail-closed behavior avoids executing an unrelated task action and does
not commit a proposed logical rollback. Once a concrete repair chunk is
returned, the server records it as issued but not yet executed. The controller
must acknowledge execution on the next cache-update observation request so the
post-repair state reaches both the Ledger and Transformer KV cache:

```python
request["compute_kv_cache"] = True
request["repair_execution_ack"] = {
    "action_id": response["repair_action_id"],
    "execution_id": response["repair_execution"]["execution_id"],
    "success": True
}
```

Until acknowledgement, the server returns
`requires_repair_execution_ack=True` and emits no second action. Only a
successful acknowledgement marks the targeted claims for post-repair
verification. A failed acknowledgement clears the in-flight action without
pretending the repair succeeded. The unique `execution_id` prevents a delayed
acknowledgement for an older same-named skill from confirming a newer action. A
prompt-conditioned recovery fallback exists only when
`ledger_allow_prompt_repair_fallback=True`; it is a debugging path, not a
substitute for environment-specific repair-policy training.

## Evaluation

`wan_va.ledger.metrics` provides:

- `claim_root_cause_metrics`
- `top_k_accuracy`
- `debt_calibration_metrics`
- `rollback_metrics`
- `repair_metrics`
- `local_rollback_metrics`
- `compute_ledger_metrics`

The implementation supplies the model, objectives, runtime, data contract, and
metrics. Reproducing paper-level task-success numbers still requires benchmark
datasets with simulator-derived causal labels, injected failure/recovery
trajectories, trained checkpoints, and robot-specific repair executors.
