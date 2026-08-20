# Paper-to-code coverage

| Paper component | Implementation |
| --- | --- |
| Causal belief ledger tuple | `models/ledger/ledger.py` claim, entity, relation, precondition, effect, evidence, confidence, debt, rollback outputs |
| Dynamic persistent entities | `object_slots.py` slot attention, recurrent identity assignment, presence, claim-slot association |
| K-step downstream dependency | `dependency_by_step` action-conditioned claim × future-step matrix |
| Causal debt | Positive learned five-term debt equation and importance-weighted global risk |
| Temporal belief updates | `forward_sequence` during training and recurrent runtime state across replans |
| Counterfactual objective | Action-order intervention and effect-embedding separation |
| Candidate world action model | `world_predictor.py` predicts next observation, claim state, ledger components, debt, and failure risk per repair |
| Self-healing planning | `planner.py` scores debt reduction minus action cost and failure risk |
| Local rollback | `task_graph.py` checkpoint selection, descendant invalidation, subplan restart, ledger-state restoration |
| Physical repairs | `experiments/common/repair_skills.py` Cartesian and joint-space verify/hold/retract/regrasp |
| Multimodal evidence | `ledger_evidence.py` video/proprio/action features and `causal_events.py` simulator event parser |
| Strong/weak supervision | Automatic weak annotator, sparse compiler, and dense temporal simulator compiler |
| LIBERO | Online evidence, physical repairs, dynamic horizons, event/ledger metrics |
| RoboTwin | Three-camera policy, online evidence, joint-space repairs, event/ledger metrics |
| RMBench | Dedicated memory-dataset/task/simulator configs over the official seamless RoboTwin policy interface |
| VLABench | Official Policy/Evaluator adapter, primitive/composite training config, official SR/IS/PS metrics |
| Ablations | Debt, counterfactual, rollback, self-healing, world prediction, temporal memory, object slots, sensor evidence |

Reproducing paper tables is an experiment artifact rather than a missing code
path. It requires downloading the original Fast-WAM/Wan/ActionDiT weights,
benchmark datasets, dataset statistics, text caches, simulators, and VLABench
assets, then running the documented training and evaluation matrix.
