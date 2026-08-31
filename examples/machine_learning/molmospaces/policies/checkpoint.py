"""
Loading and running a trained checkpoint, independent of any robot stack.

Two very different things run the same trained network:

    policies/bc_policy.py   inside the MolmoSpaces benchmark evaluator, on the
                            holonomic-base robot from `stretch/robot.py`
    live_policy.py          inside `Stretch4MujocoSimulator`, on the real
                            omniwheel robot, in an interactive viewer

Both must load, normalise, chunk and decode identically -- a mismatch in any of
those does not raise, it just produces a policy that moves smoothly and
accomplishes nothing. So all of it lives here, behind an interface that takes
plain arrays: `{camera_name: HxWx3 uint8}` and a 7-vector of proprioception in,
per-move-group joint targets out.

That the two stacks can share this at all is worth stating, because it is not
obvious. `StatusStretchJoints` from the simulator and `get_qpos_dict()` from the
MolmoSpaces robot view report the same seven numbers in the same units: the
simulator's `arm.pos` is the tendon length, which is the total telescoping
extension `StretchTelescopingArmGroup` also reports, and its
`gripper_left/right_finger.pos` are the raw URDF finger angles the move group
uses. The base differs -- holonomic joints there, omniwheels here -- which is
exactly why `networks.py` encodes the base action as a *relative* step in the
base's own frame rather than as a world pose.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from examples.machine_learning.molmospaces.policies.networks import (
    IMAGE_SIZE,
    StretchBCNet,
    decode_action,
)

log = logging.getLogger(__name__)


def resolve_checkpoint_path(checkpoint_path: str | Path) -> Path:
    """Accept either a checkpoint file or a directory containing `checkpoint.pt`."""
    path = Path(checkpoint_path)
    if path.is_dir():
        path = path / "checkpoint.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"No BC checkpoint at {path}. Train one with "
            "`python -m examples.machine_learning.molmospaces.training.train_bc`."
        )
    return path


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> (3, IMAGE_SIZE, IMAGE_SIZE) float32 in [0, 1]."""
    import cv2

    resized = cv2.resize(np.asarray(image), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))


@dataclass
class TrainedPolicy:
    """A loaded checkpoint: the network plus everything needed to use it correctly.

    Construct with `TrainedPolicy.load()`. Call `act()` once per control step;
    it serves from the current action chunk and only re-queries the network when
    that chunk runs out.
    """

    model: StretchBCNet
    camera_names: list[str]
    chunk_size: int
    execute_chunk_steps: int
    device: str
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    checkpoint_path: Path
    metadata: dict = field(default_factory=dict)

    _pending_chunk: list[np.ndarray] = field(default_factory=list, repr=False)

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        device: str | None = None,
        camera_names: list[str] | None = None,
        execute_chunk_steps: int | None = None,
    ) -> "TrainedPolicy":
        """Read a checkpoint written by `training/train_bc.py`.

        Args:
            checkpoint_path: the `.pt` file, or a directory holding `checkpoint.pt`.
            device: torch device. Defaults to CUDA when available.
            camera_names: override the checkpoint's camera list. Only do this
                deliberately -- the network's per-camera trunks are positional,
                so a different order silently degrades the policy.
            execute_chunk_steps: how many steps of each predicted chunk to run
                before re-querying. Defaults to the whole chunk.
        """
        path = resolve_checkpoint_path(checkpoint_path)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        trained_cameras = list(checkpoint["camera_names"])
        cameras = list(camera_names) if camera_names else trained_cameras
        if len(cameras) != len(trained_cameras):
            raise ValueError(
                f"Checkpoint was trained on {len(trained_cameras)} cameras {trained_cameras} "
                f"but {len(cameras)} were requested: {cameras}."
            )

        chunk_size = int(checkpoint["chunk_size"])
        model = StretchBCNet(num_cameras=len(cameras), chunk_size=chunk_size)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()

        normalisation = checkpoint["normalisation"]
        policy = cls(
            model=model,
            camera_names=cameras,
            chunk_size=chunk_size,
            execute_chunk_steps=int(execute_chunk_steps or chunk_size),
            device=device,
            state_mean=np.asarray(normalisation["state_mean"], dtype=np.float32),
            state_std=np.asarray(normalisation["state_std"], dtype=np.float32),
            action_mean=np.asarray(normalisation["action_mean"], dtype=np.float32),
            action_std=np.asarray(normalisation["action_std"], dtype=np.float32),
            checkpoint_path=path,
            metadata={
                "epoch": checkpoint.get("epoch"),
                "validation_loss": checkpoint.get("validation_loss"),
            },
        )
        log.info(
            f"[trained-policy] {path.name} | cameras={cameras} | chunk={chunk_size} "
            f"executing {policy.execute_chunk_steps} | device={device} "
            f"| val_loss={policy.metadata.get('validation_loss')}"
        )
        return policy

    def reset(self) -> None:
        """Drop any chunk left over from the previous episode."""
        self._pending_chunk = []

    @property
    def steps_left_in_chunk(self) -> int:
        return len(self._pending_chunk)

    def act(
        self,
        images: dict[str, np.ndarray],
        state: np.ndarray,
        base_xytheta: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """One control step's worth of joint targets.

        Args:
            images: `{camera_name: HxWx3 uint8}`. Must contain every name in
                `camera_names`; extra entries are ignored.
            state: the 7-vector from `networks.encode_state`.
            base_xytheta: the base's current world (x, y, yaw), used to turn the
                network's relative base step back into an absolute target.

        Returns:
            `{move_group: targets}` -- "base" as an absolute world (x, y, yaw),
            the rest as absolute joint positions.
        """
        if not self._pending_chunk:
            self._pending_chunk = list(self.predict_chunk(images, state))
        return decode_action(self._pending_chunk.pop(0), base_xytheta)

    @torch.no_grad()
    def predict_chunk(self, images: dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
        """`(execute_chunk_steps, ACTION_DIM)` of denormalised actions."""
        missing = [name for name in self.camera_names if name not in images]
        if missing:
            raise KeyError(
                f"Missing camera(s) {missing} for this checkpoint; got {sorted(images)}."
            )

        image_batch = np.stack([preprocess_image(images[name]) for name in self.camera_names])
        normalised_state = (np.asarray(state, dtype=np.float32) - self.state_mean) / self.state_std

        chunk = (
            self.model(
                torch.from_numpy(image_batch).unsqueeze(0).to(self.device),
                torch.from_numpy(normalised_state).unsqueeze(0).to(self.device),
            )[0]
            .cpu()
            .numpy()
        )
        return (chunk * self.action_std + self.action_mean)[: self.execute_chunk_steps]
