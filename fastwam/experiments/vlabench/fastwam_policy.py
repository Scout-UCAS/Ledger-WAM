from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from experiments.common import SimulatorEventParser, apply_cartesian_repair
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

try:
    from VLABench.evaluation.model.policy.base import Policy
    from VLABench.utils.utils import quaternion_to_euler as _official_quaternion_to_euler
except ImportError:  # Allows adapter tests without installing the MuJoCo benchmark.
    _official_quaternion_to_euler = None
    class Policy:  # type: ignore[no-redef]
        def __init__(self, model):
            self.model = model


def _quaternion_to_euler(quaternion: np.ndarray) -> np.ndarray:
    if _official_quaternion_to_euler is not None:
        return np.asarray(_official_quaternion_to_euler(quaternion), dtype=np.float32)
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.asarray([roll, pitch, yaw], dtype=np.float32)


class LedgerFastWAMVLABenchPolicy(Policy):
    """Official VLABench `Policy` adapter for Ledger-WAM/Fast-WAM."""

    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: str,
        *,
        device: str = "cuda",
        model_dtype: torch.dtype = torch.bfloat16,
        action_horizon: int = 32,
        replan_steps: int = 5,
        num_inference_steps: int = 10,
        repair_skill_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        cfg = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        cfg.load_text_encoder = True
        model = instantiate(cfg, model_dtype=model_dtype, device=device)
        model.load_checkpoint(Path(checkpoint_path))
        model = model.to(device).eval()
        super().__init__(model)

        self.processor: FastWAMProcessor = instantiate(processor_cfg).eval()
        self.processor.set_normalizer_from_stats(
            load_dataset_stats_from_json(str(dataset_stats_path))
        )
        self.action_horizon = int(action_horizon)
        self.replan_steps = int(replan_steps)
        self.num_inference_steps = int(num_inference_steps)
        self.repair_skill_config = dict(repair_skill_config or {})
        self.pending_actions: deque[np.ndarray] = deque()
        self.events = SimulatorEventParser()

    @property
    def name(self) -> str:
        return "LedgerFastWAM"

    @property
    def control_mode(self) -> str:
        return "ee"

    def reset(self) -> None:
        self.pending_actions.clear()
        self.events.reset()
        if hasattr(self.model, "reset_ledger"):
            self.model.reset_ledger()

    def _state(self, obs: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw = np.asarray(obs["ee_state"], dtype=np.float32).reshape(-1)
        if raw.size == 8:
            position = raw[:3]
            euler = _quaternion_to_euler(raw[3:7])
            gripper = raw[7:8]
        elif raw.size == 7:
            position, euler, gripper = raw[:3], raw[3:6], raw[6:7]
        else:
            raise ValueError(f"VLABench ee_state must contain 7 or 8 values, got {raw.size}.")
        return position, euler, np.concatenate((position, euler, gripper))

    def _image(self, obs: Mapping[str, Any]) -> torch.Tensor:
        cameras = obs.get("rgb")
        if not isinstance(cameras, (list, tuple)) or len(cameras) < 3:
            raise ValueError("VLABench observation must expose three images in `obs['rgb']`.")
        resized = [
            np.asarray(Image.fromarray(np.asarray(camera, dtype=np.uint8)).resize((224, 224)))
            for camera in cameras[:3]
        ]
        image = np.concatenate(resized, axis=1)
        return (
            torch.from_numpy(image.copy())
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=self.model.device, dtype=self.model.torch_dtype)
            * (2.0 / 255.0)
            - 1.0
        )

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_key = self.processor.shape_meta["state"][0]["key"]
        batch = {"state": {state_key: torch.from_numpy(state).float().unsqueeze(0)}}
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        return batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        action_key = self.processor.shape_meta["action"][0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        return normalizer.backward(action.float().cpu()).numpy()[0]

    def _replan(self, obs: Mapping[str, Any], state: np.ndarray) -> None:
        image = self._image(obs)
        evidence, _ = self.events.observe(obs, image=np.asarray(obs["rgb"][0]))
        prediction = self.model.infer_action(
            prompt=DEFAULT_PROMPT.format(task=str(obs["instruction"])),
            input_image=image,
            action_horizon=self.action_horizon,
            proprio=self._normalize_state(state),
            num_inference_steps=self.num_inference_steps,
            ledger_evidence=evidence,
        )
        horizon = max(
            1,
            min(
                int(prediction.get("execution_horizon", self.replan_steps)),
                self.replan_steps,
                prediction["action"].shape[0],
            ),
        )
        action = self._denormalize_action(prediction["action"][:horizon])
        action = apply_cartesian_repair(
            action,
            prediction.get("planner", {}).get("repair_name"),
            current_state=state,
            config=self.repair_skill_config,
        )
        self.pending_actions.extend(action)

    def predict(self, obs: Mapping[str, Any], **_: Any):
        position, euler, state = self._state(obs)
        if not self.pending_actions:
            self._replan(obs, state)
        delta = np.asarray(self.pending_actions.popleft(), dtype=np.float32)
        target_position = position + delta[:3]
        target_euler = euler + delta[3:6]
        gripper = np.ones(2, dtype=np.float32) * (0.04 if delta[-1] >= 0.1 else 0.0)
        self.events.last_action = delta
        return target_position, target_euler, gripper
