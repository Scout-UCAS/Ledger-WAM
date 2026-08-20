# Ledger-WAM: Causal Ledger World-Action Model

Ledger-WAM is a Ledger-augmented world-action model for embodied robot control.
This repository contains two backend implementations of structured causal-belief
tracking, causal debt estimation, counterfactual action modeling, self-healing
repair selection, and local logical rollback.

## Fast-WAM implementation

The paper-aligned Fast-WAM implementation is available in [`fastwam/`](fastwam/).
It includes the complete Fast-WAM baseline, recurrent causal ledger, dynamic
object slots, K-step dependency prediction, candidate-action world predictor,
task-graph rollback, physical repair skills, strong/weak annotation pipelines,
LIBERO/RoboTwin/RMBench/VLABench adapters, metrics, ablations, and tests.

Start with [`fastwam/LEDGER_WAM.md`](fastwam/LEDGER_WAM.md) and the
[paper-to-code coverage table](fastwam/docs/method_coverage.md). The existing
`wan_va/` tree is retained as the LingBot-VA backend for compatibility.

The LingBot-VA implementation is opt-in. Legacy LingBot-VA configurations keep
their original behavior, while Ledger-WAM paths are enabled through
`ledger_enabled` or the `ledger_*` training/server configurations.

## Highlights

- Structured causal ledger over observed video latents, action history, and
  language embeddings.
- Claim slots with truth, presence, relation, entity, precondition, effect,
  evidence, uncertainty, dependency, repair cost, observability, importance,
  debt, rollback, and repair-action predictions.
- Monotone causal debt: lower confidence, higher uncertainty, stronger
  dependency, higher repair cost, and lower observability increase risk by
  construction.
- Directed dependency graph and local logical rollback. Physical robot state is
  never rolled back; only the logical plan and invalidated beliefs are updated.
- Counterfactual action supervision with a shared Siamese transition head.
- Self-healing planner that decides between normal task execution, repair
  actions, and local rollback.
- Online repair execution handshake. A repair is not treated as executed until
  the controller returns `repair_execution_ack`.
- RoboTwin and LIBERO clients, Ledger sidecar schema, unit tests, metrics, and
  detailed Chinese documentation.

## Current Status

Implemented:

- Ledger neural head and training losses.
- Ledger sidecar schema for JSON/JSONL causal annotations.
- Runtime causal ledger, evidence fusion, dependency invalidation, repair
  selection, and logical rollback.
- Repair-action issue/ack protocol.
- Ledger-conditioned action-token adapter.
- RoboTwin and LIBERO evaluation clients.
- Metrics for root-cause localization, debt calibration, rollback, repair
  efficiency, and local rollback ratio.
- Example ledger annotations and unit tests.

Known scope limits:

- VLABench is not yet a one-command benchmark in this repository. It requires a
  dedicated `evaluation/vlabench` adapter that maps VLABench observations and
  actions to the Ledger-WAM server protocol.
- Paper-level reproduction requires real causal labels, trained Ledger
  checkpoints, benchmark simulators, and robot-specific repair executors.
- Lightweight tests pass locally, but full CUDA/GPU training and simulator
  evaluation must be run in the target environment.

## Repository Layout

```text
.
├── README.md                         # Project entry point
├── README_CN.md                      # Detailed Chinese documentation
├── LEDGER_WAM.md                     # Ledger-WAM schema and runtime protocol
├── INSTALL.md                        # Installation notes
├── requirements.txt
├── pyproject.toml
├── example/
│   ├── ledger_annotations.example.jsonl
│   ├── robotwin/
│   ├── libero/
│   ├── franka/
│   └── demo/
├── evaluation/
│   ├── libero/
│   └── robotwin/
├── script/
│   ├── run_va_posttrain.sh
│   └── run_launch_va_server_sync.sh
├── tests/
└── wan_va/
    ├── configs/
    ├── dataset/
    ├── ledger/
    ├── modules/
    ├── train.py
    └── wan_va_server.py
```

## Requirements

Recommended environment:

```text
Python >= 3.10, < 4.0
CUDA 12.6
PyTorch 2.9.0
torchvision 0.24.0
torchaudio 2.9.0
diffusers 0.36.0
transformers 4.55.2
lerobot 0.3.3
```

Install:

```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

Package install:

```bash
pip install .
pip install .[train]
```

## Model and Data

This project reuses LingBot-VA checkpoints and LeRobot-format post-training
datasets. Useful upstream resources include:

```text
robbyant/lingbot-va-base
robbyant/lingbot-va-posttrain-robotwin
robbyant/lingbot-va-posttrain-libero-long
robbyant/robotwin-clean-and-aug-lerobot
robbyant/libero-long-lerobot
```

Set local data paths through environment variables:

```bash
export LEDGER_WAM_DATASET_PATH=/path/to/your/dataset
export ROBOWIN_ROOT=/path/to/RoboTwin
```

`ROBOWIN_ROOT` may also be passed directly to the RoboTwin client:

```bash
python -m evaluation.robotwin.eval_polict_client_openpi \
  --robowin_root /path/to/RoboTwin \
  --config policy/ACT/deploy_policy.yml \
  --overrides ...
```

## Important Configurations

Enable Ledger-WAM:

```python
ledger_enabled = True
```

Common Ledger fields:

```text
ledger_annotation_path
ledger_strict
ledger_max_claims
ledger_max_counterfactuals
ledger_action_dim
ledger_debt_threshold
ledger_global_risk_threshold
ledger_confidence_threshold
ledger_repair_cost_weight
ledger_repair_risk_weight
ledger_repair_catalog
ledger_allow_random_head
ledger_allow_prompt_repair_fallback
```

Action pose conversion is configured per environment:

```python
relative_action_pose_slices = ((0, 7), (8, 15))  # RoboTwin
relative_action_pose_slices = ((0, 7), (7, 14))  # Franka
```

Each slice must be `[x, y, z, qx, qy, qz, qw]`. LIBERO and demo configurations
do not enable relative pose conversion by default.

## Ledger Annotation Sidecar

Ledger post-training uses a JSON/JSONL sidecar. Default location:

```text
<LeRobot dataset root>/meta/ledger_annotations.jsonl
```

You can override it with `ledger_annotation_path`.

Example:

```json
{
  "key": "12:30:90",
  "episode_index": 12,
  "start_frame": 30,
  "end_frame": 90,
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

Notes:

- `key` is `episode_index:start_frame:end_frame`.
- Missing labels are masked independently and are never treated as negative
  labels.
- `dependency_edges` represents directed causal dependencies. Dense
  `dependency_matrix` is also supported.
- `repair_action` indexes `ledger_repair_catalog`; each catalog `id` must equal
  its list position.
- `rollback` indexes the fixed `ledger_rollback_stage_ontology`.

See [LEDGER_WAM.md](LEDGER_WAM.md) for the full schema.

## Training

Standard LingBot-VA post-training:

```bash
CONFIG_NAME=robotwin_train bash script/run_va_posttrain.sh
```

Ledger-WAM post-training:

```bash
CONFIG_NAME=ledger_robotwin_train \
  bash script/run_va_posttrain.sh \
  --ledger-annotation-path /path/to/ledger_annotations.jsonl \
  --ledger-strict
```

Other Ledger training configs:

```text
ledger_libero_train
ledger_demo_train
```

The trainer optimizes LingBot-VA video/action objectives together with Ledger
objectives: claim structure, evidence, uncertainty, dependency, debt, rollback,
repair action, repair cost, post-repair debt, counterfactual transitions, and
recurrent slot consistency.

Repair reward supervision follows:

```text
debt_reduction - beta * action_cost - gamma * task_risk
```

## Inference and Evaluation

RoboTwin server:

```bash
bash evaluation/robotwin/launch_server.sh
```

RoboTwin client:

```bash
export ROBOWIN_ROOT=/path/to/RoboTwin

task_name="adjust_bottle"
save_root="results"
bash evaluation/robotwin/launch_client.sh ${save_root} ${task_name}
```

LIBERO:

```bash
bash evaluation/libero/launch_server.sh
bash evaluation/libero/launch_client.sh
```

Image-to-video-action generation:

```bash
NGPU=1 CONFIG_NAME='robotwin_i2av' bash script/run_launch_va_server_sync.sh
```

## Online Repair Protocol

Reset:

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

Observation with external causal evidence:

```python
client.infer({
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
})
```

Repair action chunks must be supplied by production controllers:

```python
request["repair_action_chunks"] = {
    "lift_test": lift_test_action_chunk,
    "local_rollback": local_recovery_action_chunk
}
```

Chunk shape:

```text
[used_action_channels, frame_chunk_size, actions_per_frame]
```

If no executable repair chunk is available, the server fails closed:

```python
{
    "action": None,
    "requires_repair_action": True,
    "repair_action_id": "lift_test",
    "repair_instruction": "..."
}
```

After executing a repair chunk, the controller must acknowledge execution:

```python
client.infer({
    "compute_kv_cache": True,
    "obs": key_frame_list,
    "state": executed_action,
    "repair_execution_ack": {
        "action_id": response["repair_action_id"],
        "execution_id": response["repair_execution"]["execution_id"],
        "success": True
    }
})
```

The server will not issue a second repair action while one is awaiting
acknowledgement.

## Metrics

`wan_va/ledger/metrics.py` provides:

```text
claim_root_cause_metrics
top_k_accuracy
debt_calibration_metrics
rollback_metrics
repair_metrics
local_rollback_metrics
compute_ledger_metrics
```

## Verification

Lightweight checks:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m compileall -q \
  wan_va tests evaluation script

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

Latest local verification:

```text
63 tests OK
compileall OK for key files
```

## FAQ

### RoboTwin cannot be found

Set:

```bash
export ROBOWIN_ROOT=/path/to/RoboTwin
```

or pass:

```bash
--robowin_root /path/to/RoboTwin
```

### Unsupported RoboTwin action channels

The current RoboTwin client supports:

```text
14 channels: dual-arm xyz+rpy+gripper
16 channels: dual-arm xyz+quat+gripper
```

Other layouts require an environment-specific action converter.

### Ledger checkpoint loading fails

When `ledger_enabled=True`, the server expects a checkpoint with a trained
Ledger head. For connectivity debugging only:

```python
ledger_allow_random_head = True
```

### `attn_mode` mismatch

Use:

```text
training:  "flex"
inference: "torch" or "flashattn"
```

This value is usually stored in `<checkpoint>/transformer/config.json`.

## Acknowledgements

This repository builds on LingBot-VA and keeps the original LingBot-VA model,
data, and evaluation infrastructure where appropriate. Ledger-WAM adds the
causal-ledger, self-healing, counterfactual, and repair-protocol layers needed
for causal world-action modeling.

## License

This project follows the Apache-2.0 license in [LICENSE.txt](LICENSE.txt).
