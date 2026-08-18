"""
The behaviour-cloned Stretch policy, as a MolmoSpaces `InferencePolicy`.

This is the "learned policy" half of the benchmark story: `training/collect.py`
records the scripted expert, `training/train_bc.py` fits `StretchBCNet` to it,
and this class loads the resulting checkpoint and runs it inside the same
evaluation harness the expert was scored in.

The checkpoint is self-describing -- it carries the observation normalisation
statistics, the camera names and the chunk size the network was trained with --
so a policy config only has to name a file. That matters because a chunk-size or
normalisation mismatch does not raise; it just produces a policy that moves
smoothly and accomplishes nothing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from examples.machine_learning.molmospaces.policies.networks import (
    ACTION_DIM,
    IMAGE_SIZE,
    STATE_DIM,
    StretchBCNet,
    decode_action,
    encode_state,
)
from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.policy.base_policy import InferencePolicy, PolicyFactory
from molmo_spaces.utils.function_utils import make_lenient

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
    from molmo_spaces.tasks.task import BaseMujocoTask

log = logging.getLogger(__name__)


class StretchBCPolicyConfig(BasePolicyConfig):
    """Configuration for `StretchBCPolicy`."""

    policy_type: str = "learned"
    policy_cls: type | None = None
    policy_factory: PolicyFactory | None = None

    checkpoint_path: str | None = None
    """
    Path to a checkpoint written by `training/train_bc.py`. Usually supplied on
    the command line instead; `run_evaluation(checkpoint_path=...)` overwrites
    whatever is set here.
    """

    camera_names: list[str] = []
    """
    Cameras to feed the network, in order. Left empty by default so the
    checkpoint's own list wins; set it only to deliberately override.
    """

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    execute_chunk_steps: int | None = None
    """
    How many steps of each predicted chunk to execute before re-querying the
    network. Defaults to the whole chunk. Executing fewer trades inference cost
    for responsiveness.
    """

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.policy_cls is None:
            self.policy_cls = StretchBCPolicy
            self.policy_factory = make_lenient(StretchBCPolicy)


class StretchBCPolicy(InferencePolicy):
    """Runs a trained `StretchBCNet`, one action chunk at a time."""

    def __init__(self, config: "MlSpacesExpConfig", task: "BaseMujocoTask" = None) -> None:
        super().__init__(config, task)
        self._model: StretchBCNet | None = None
        self._pending_chunk: list[np.ndarray] = []
        self._camera_names: list[str] = list(config.policy_config.camera_names)
        self._state_mean = np.zeros(STATE_DIM, dtype=np.float32)
        self._state_std = np.ones(STATE_DIM, dtype=np.float32)
        self._action_mean = np.zeros(ACTION_DIM, dtype=np.float32)
        self._action_std = np.ones(ACTION_DIM, dtype=np.float32)
        self._execute_steps = 1
        self.prepare_model(config.policy_config.checkpoint_path)

    # =========================================================================
    # Model lifecycle
    # =========================================================================

    def prepare_model(self, model_name: str | None = None) -> None:
        checkpoint_path = model_name or self.config.policy_config.checkpoint_path
        if not checkpoint_path:
            raise ValueError(
                "StretchBCPolicy needs a checkpoint. Pass --checkpoint_path, or set "
                "checkpoint_path on the policy config. Train one with "
                "`python -m examples.machine_learning.molmospaces.training.train_bc`."
            )
        path = Path(checkpoint_path)
        if path.is_dir():
            path = path / "checkpoint.pt"
        if not path.exists():
            raise FileNotFoundError(f"No BC checkpoint at {path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        # The checkpoint's camera list is authoritative unless the config named
        # one explicitly: the network's per-camera trunks are positional, so
        # feeding them in a different order silently degrades the policy.
        self._camera_names = self._camera_names or list(checkpoint["camera_names"])
        if len(self._camera_names) != len(checkpoint["camera_names"]):
            raise ValueError(
                f"Checkpoint was trained on {len(checkpoint['camera_names'])} cameras "
                f"{checkpoint['camera_names']} but the config asks for {self._camera_names}."
            )

        self._model = StretchBCNet(
            num_cameras=len(self._camera_names),
            chunk_size=int(checkpoint["chunk_size"]),
        )
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(self.config.policy_config.device).eval()

        normalisation = checkpoint["normalisation"]
        self._state_mean = np.asarray(normalisation["state_mean"], dtype=np.float32)
        self._state_std = np.asarray(normalisation["state_std"], dtype=np.float32)
        self._action_mean = np.asarray(normalisation["action_mean"], dtype=np.float32)
        self._action_std = np.asarray(normalisation["action_std"], dtype=np.float32)

        configured = self.config.policy_config.execute_chunk_steps
        self._execute_steps = int(configured or self._model.chunk_size)
        log.info(
            f"[stretch-bc] loaded {path} | cameras={self._camera_names} "
            f"| chunk={self._model.chunk_size} executing {self._execute_steps}"
        )

    def reset(self) -> None:
        self._pending_chunk = []

    # =========================================================================
    # InferencePolicy
    # =========================================================================

    def obs_to_model_input(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        observation = obs[0] if isinstance(obs, list) else obs
        images = np.stack([_preprocess_image(observation[name]) for name in self._camera_names])
        state = encode_state(self.task.env.current_robot.robot_view.get_qpos_dict())
        normalised_state = (state - self._state_mean) / self._state_std

        device = self.config.policy_config.device
        return (
            torch.from_numpy(images).unsqueeze(0).to(device),
            torch.from_numpy(normalised_state).unsqueeze(0).to(device),
        )

    @torch.no_grad()
    def inference_model(self, model_input) -> np.ndarray:
        images, state = model_input
        chunk = self._model(images, state)[0].cpu().numpy()
        return chunk * self._action_std + self._action_mean

    def model_output_to_action(self, model_output: np.ndarray) -> dict[str, Any]:
        self._pending_chunk = list(model_output[: self._execute_steps])
        return self._decode(self._pending_chunk.pop(0))

    def get_action(self, observation) -> dict[str, Any]:
        """Serve from the pending chunk, re-querying the network when it runs out."""
        if self._pending_chunk:
            return self._decode(self._pending_chunk.pop(0))
        return super().get_action(observation)

    def _decode(self, action: np.ndarray) -> dict[str, Any]:
        base_pose = self.task.env.current_robot.robot_view.base.pose
        base_xytheta = np.array(
            [base_pose[0, 3], base_pose[1, 3], np.arctan2(base_pose[1, 0], base_pose[0, 0])]
        )
        return decode_action(action, base_xytheta)

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy"] = "stretch_bc"
        info["checkpoint_path"] = self.config.policy_config.checkpoint_path
        return info


def _preprocess_image(image: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> (3, IMAGE_SIZE, IMAGE_SIZE) float32 in [0, 1]."""
    import cv2

    resized = cv2.resize(np.asarray(image), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))
