from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from experiments.vlabench.fastwam_policy import LedgerFastWAMVLABenchPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Ledger-WAM on official VLABench.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--track", default="track_1_in_distribution")
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--save-dir", default="evaluate_results/vlabench")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    try:
        from VLABench.evaluation.evaluator import Evaluator
    except ImportError as exc:
        raise RuntimeError(
            "Install the official OpenMOSS/VLABench package and assets before evaluation."
        ) from exc

    project_root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(version_base="1.3", config_dir=str(project_root / "configs")):
        cfg = compose(config_name="sim_vlabench")
    policy = LedgerFastWAMVLABenchPolicy(
        cfg.model,
        cfg.data.train.processor,
        args.checkpoint,
        args.dataset_stats,
        device=args.device,
        model_dtype=torch.bfloat16,
        action_horizon=int(cfg.EVALUATION.action_horizon),
        replan_steps=int(cfg.EVALUATION.replan_steps),
        num_inference_steps=int(cfg.EVALUATION.num_inference_steps),
        repair_skill_config=OmegaConf.to_container(cfg.EVALUATION.repair_skills, resolve=True),
    )
    vlabench_root = Path(os.environ["VLABENCH_ROOT"])
    episode_config = json.loads(
        (vlabench_root / "configs" / "evaluation" / "tracks" / f"{args.track}.json").read_text()
    )
    tasks = args.tasks or list(episode_config)
    evaluator = Evaluator(
        tasks=tasks,
        n_episodes=args.episodes,
        episode_config=episode_config,
        max_substeps=1,
        save_dir=args.save_dir,
        visulization=False,
        metrics=["success_rate", "intention_score", "progress_score"],
    )
    result = evaluator.evaluate(policy)
    output = Path(args.save_dir) / "ledger_fastwam" / args.track / "evaluation_result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
