# Ledger-WAM on Fast-WAM

This checkout implements the method in *Ledger-WAM: Causal Debt-Aware World
Action Models for Self-Healing Long-Horizon Planning* on the official Fast-WAM
baseline.

## Method implementation

The implementation is end-to-end rather than a standalone classifier:

- `CausalBeliefLedger` recurrently unrolls four updates per training window and
  persists state across closed-loop replans.
- `DynamicObjectSlotEncoder` extracts 12 non-semantic object slots from video
  tokens, differentiably matches their identity to the previous replan, predicts
  presence, and associates each claim with both semantic entity classes and
  concrete object-slot IDs.
- Every claim predicts relation, precondition, effect, evidence, confidence,
  uncertainty, observability, importance, repair cost, rollback point, and an
  explicit dependency value for each of the next `K=8` action steps.
- Debt uses positive learned coefficients over confidence deficit, uncertainty,
  K-step dependency, repair cost, and observability deficit. Global risk is the
  importance-weighted claim debt.
- `CausalWorldPredictor` rolls every repair candidate forward to a predicted
  object/claim state, observation embedding, next ledger, failure probability,
  and next global debt. `SelfHealingPlanner` optimizes expected debt reduction
  minus action cost and predicted risk.
- `CausalTaskGraph` maintains checkpoints, invalidates descendants of a
  contradicted claim, selects a subplan restart point, and restores the matching
  recurrent ledger state.
- The environment layer implements `verify`, `hold`, `retract`, and `regrasp` in
  Cartesian delta-pose or dual-arm joint space. Repairs are executed physically;
  every repair is followed by a new observation and replan.

The loss includes action/video flow matching, claim/debt/dependency/evidence
supervision, relation/entity/precondition/effect structure, object-slot identity,
rollback classification, action counterfactual separation, repair imitation,
candidate value calibration, next-ledger prediction, and future-observation
consistency. Auxiliary checkpoint state contains both the ledger and causal world
predictor.

## Supervision

Two label paths are supported and can be combined:

1. `WeakCausalAnnotator` automatically derives dense four-step evidence and weak
   labels from video, proprioception, actions, padding, co-motion, and gripper
   changes. It is enabled in all Ledger-WAM data configs, so training never falls
   back to a counterfactual-only objective.
2. `SimulatorEventParser` extracts contact, grasp, support, containment,
   visibility, persistence, co-motion, and effect events from privileged LIBERO,
   RoboTwin, or VLABench state. Strong JSONL traces are compiled with:

```bash
PYTHONPATH=src:. python scripts/compile_simulator_events.py \
  --input data/simulator_events.jsonl \
  --output data/ledger_annotations.jsonl
```

Strong sidecar fields override weak labels for the same sample. Sparse human or
simulator annotations remain supported through `scripts/build_ledger_annotations.py`.
See [the annotation schema](docs/ledger_annotations.md).

## Train

Install and prepare the original Fast-WAM backbone, datasets, normalization
statistics, and text caches first. Then run:

```bash
# LIBERO
bash scripts/train_zero1.sh 8 task=libero_ledger_2cam224_1e-4

# RoboTwin / RMBench-compatible trajectories
bash scripts/train_zero1.sh 8 task=robotwin_ledger_3cam_384_1e-4

# Official RMBench memory tasks (14-D joints, three views)
bash scripts/train_zero1.sh 8 task=rmbench_ledger_3cam384_1e-4

# VLABench primitive + composite LeRobot datasets
bash scripts/train_zero1.sh 8 task=vlabench_ledger_3cam224_1e-4
```

Add `data.train.ledger_annotations_path=/path/to/sidecar.jsonl` for strong labels.
Without it, the automatic weak annotator still supplies dense temporal training
targets.

## Evaluate

LIBERO and RoboTwin evaluation use the existing managers. Both pass online causal
evidence into the model, respect the dynamic execution horizon, execute physical
repair skills, and record ledger/world-model/recovery metrics.

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_ledger_2cam224_1e-4 ckpt=/path/to/ledger_wam.pt \
  EVALUATION.dataset_stats_path=/path/to/dataset_stats.json \
  MULTIRUN.num_gpus=1

python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_ledger_3cam_384_1e-4 ckpt=/path/to/ledger_wam.pt \
  EVALUATION.dataset_stats_path=/path/to/dataset_stats.json \
  MULTIRUN.num_gpus=1

# RMBench uses the official seamless RoboTwin policy interface.
python experiments/robotwin/run_robotwin_manager.py \
  --config-name sim_rmbench \
  ckpt=/path/to/ledger_wam.pt \
  EVALUATION.dataset_stats_path=/path/to/rmbench_dataset_stats.json
```

The VLABench adapter implements the current official `Policy.predict(obs)` API
for three RGB cameras, a 7/8-value end-effector state, and 7-D delta actions. It
can be passed directly to the official `Evaluator`:

```bash
export VLABENCH_ROOT=/path/to/OpenMOSS/VLABench
PYTHONPATH=src:. python experiments/vlabench/evaluate.py \
  --checkpoint /path/to/ledger_wam.pt \
  --dataset-stats /path/to/dataset_stats.json \
  --track track_1_in_distribution --episodes 10
```

The evaluator reports the official success, intention, and progress scores plus
Ledger-WAM repair rate, rollback rate, recovery success/time, unnecessary repair
rate, debt calibration, and causal world-risk prediction error where labels are
available.

## Ablations

Append one Hydra override to a Ledger-WAM command:

```bash
+ablation=ledger_no_debt
+ablation=ledger_no_counterfactual
+ablation=ledger_no_rollback
+ablation=ledger_no_self_healing
+ablation=ledger_no_world_prediction
+ablation=ledger_no_temporal_memory
+ablation=ledger_no_object_slots
+ablation=ledger_no_sensor_evidence
```

Use the official Fast-WAM task config as the baseline and initialize every run
from the same backbone checkpoint.

## Verification

```bash
PYTHONPATH=src:. python -m pytest tests -q
PYTHONPATH=src:. python -m compileall -q src scripts experiments tests
```

The repository tests cover recurrence, dynamic object identity, K-step
dependency, multimodal evidence, candidate world prediction, temporal/world loss
backpropagation, task-graph rollback, simulator trace compilation, environment
repair skills, checkpointing, and evaluation metrics. Full benchmark scores still
require the external pretrained weights, datasets, normalization statistics,
simulators, and VLABench assets; those artifacts are not vendored in Fast-WAM.
