"""
The behaviour-cloning network, and the action/state encoding it is trained on.

The encoding is defined here rather than in the trainer or the policy because
all three have to agree on it exactly: `training/collect.py` writes it,
`training/train_bc.py` fits it, and `policies/bc_policy.py` reads it back at
evaluation time. Getting it wrong is silent -- the policy runs and does nothing
useful -- so there is one definition and everyone imports it.

State (7 numbers), all robot-relative so it means the same thing in every house:

    lift        1   mast height, metres
    arm         1   total telescoping extension, metres
    wrist       3   yaw, pitch, roll
    gripper     2   the two finger joints

Action (10 numbers):

    base        3   *relative* (forward, left, yaw) step in the base's own frame
    lift        1   absolute target
    arm         1   absolute target
    wrist       3   absolute targets
    gripper     2   absolute targets

The base is relative and everything else absolute for the same reason: absolute
base coordinates are world coordinates, which carry no information a policy can
transfer between houses, whereas absolute arm targets are exactly the
proprioceptive frame the policy already observes.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

STATE_GROUPS: tuple[tuple[str, int], ...] = (
    ("lift", 1),
    ("arm", 1),
    ("wrist", 3),
    ("gripper", 2),
)
STATE_DIM = sum(width for _, width in STATE_GROUPS)

ACTION_GROUPS: tuple[tuple[str, int], ...] = (
    ("base", 3),
    ("lift", 1),
    ("arm", 1),
    ("wrist", 3),
    ("gripper", 2),
)
ACTION_DIM = sum(width for _, width in ACTION_GROUPS)

IMAGE_SIZE = 112
"""Side length images are resized to before they reach the network."""


def encode_state(qpos: dict[str, np.ndarray]) -> np.ndarray:
    """Pack a `robot_view.get_qpos_dict()` into the fixed-width state vector."""
    return np.concatenate(
        [
            np.asarray(qpos[group], dtype=np.float32).reshape(-1)[:width]
            for group, width in STATE_GROUPS
        ]
    ).astype(np.float32)


def encode_action(commanded: dict[str, np.ndarray], base_pose_xytheta: np.ndarray) -> np.ndarray:
    """Pack a commanded joint-target dict into the fixed-width action vector.

    Args:
        commanded: absolute per-move-group targets, as the policies emit them.
        base_pose_xytheta: the base's *current* (x, y, theta), used to turn the
            commanded absolute base pose into a step in the base's own frame.
    """
    parts = []
    for group, width in ACTION_GROUPS:
        value = np.asarray(commanded[group], dtype=np.float32).reshape(-1)[:width]
        if group == "base":
            value = world_base_step_to_local(value, base_pose_xytheta)
        parts.append(value)
    return np.concatenate(parts).astype(np.float32)


def world_base_step_to_local(
    commanded_base: np.ndarray, base_pose_xytheta: np.ndarray
) -> np.ndarray:
    """Commanded absolute base pose -> (forward, left, yaw) step in the base frame."""
    delta_world = (
        np.asarray(commanded_base, dtype=np.float32)[:2]
        - np.asarray(base_pose_xytheta, dtype=np.float32)[:2]
    )
    yaw = float(base_pose_xytheta[2])
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    forward = cos_yaw * delta_world[0] + sin_yaw * delta_world[1]
    left = -sin_yaw * delta_world[0] + cos_yaw * delta_world[1]
    delta_yaw = wrap_angle(float(commanded_base[2]) - yaw)
    return np.array([forward, left, delta_yaw], dtype=np.float32)


def local_base_step_to_world(local_step: np.ndarray, base_pose_xytheta: np.ndarray) -> np.ndarray:
    """Inverse of `world_base_step_to_local`: back to an absolute base target."""
    yaw = float(base_pose_xytheta[2])
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    x = base_pose_xytheta[0] + cos_yaw * local_step[0] - sin_yaw * local_step[1]
    y = base_pose_xytheta[1] + sin_yaw * local_step[0] + cos_yaw * local_step[1]
    return np.array([x, y, yaw + local_step[2]], dtype=np.float32)


def wrap_angle(angle: float) -> float:
    """Wrap to (-pi, pi]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def decode_action(action: np.ndarray, base_pose_xytheta: np.ndarray) -> dict[str, np.ndarray]:
    """Unpack a network output back into per-move-group absolute targets."""
    targets: dict[str, np.ndarray] = {}
    offset = 0
    for group, width in ACTION_GROUPS:
        value = np.asarray(action[offset : offset + width], dtype=np.float32)
        if group == "base":
            value = local_base_step_to_world(value, base_pose_xytheta)
        targets[group] = value
        offset += width
    return targets


class _VisionTrunk(nn.Module):
    """A small strided CNN, one per camera.

    Deliberately trained from scratch and kept tiny: the benchmark data this
    repo can realistically collect is thousands of episodes, not millions, and a
    pretrained ViT would spend that budget on fine-tuning rather than on learning
    the task. Swap in a frozen encoder here if you have the data for it.
    """

    OUTPUT_DIM = 256

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


class StretchBCNet(nn.Module):
    """Images + proprioception -> a chunk of future actions.

    Predicting a chunk rather than a single action is what makes an open-loop
    behaviour cloner usable at 15Hz: a single-step policy has to be re-queried
    every 66ms and drifts whenever two nearby observations imply opposite motions
    (the classic pause-at-the-grasp failure), whereas a chunk commits to a short
    trajectory the way the scripted teacher does.
    """

    def __init__(
        self,
        num_cameras: int = 2,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        chunk_size: int = 8,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.num_cameras = num_cameras
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        self.vision = nn.ModuleList(_VisionTrunk() for _ in range(num_cameras))
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 128)
        )
        fused_dim = num_cameras * _VisionTrunk.OUTPUT_DIM + 128
        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, chunk_size * action_dim),
        )

    def forward(self, images: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (batch, num_cameras, 3, H, W), float in [0, 1].
            state: (batch, state_dim), normalised.

        Returns:
            (batch, chunk_size, action_dim) of normalised actions.
        """
        features = [trunk(images[:, i]) for i, trunk in enumerate(self.vision)]
        features.append(self.state_encoder(state))
        fused = torch.cat(features, dim=-1)
        return self.head(fused).view(-1, self.chunk_size, self.action_dim)
