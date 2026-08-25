"""
The remapping itself: Franka joint space in, Stretch move groups out.

This is the piece that lets a policy trained to drive a Franka drive Stretch.
It is stateful -- it keeps a *shadow* Franka whose joints are the ones the VLA
believes it is commanding -- and it works in both directions:

    observation()   Stretch's tool pose  ->  7 Franka joints + gripper
    action()        7 Franka joints + gripper  ->  Stretch's five move groups

Both directions go through the same intermediate quantity, the tool pose, for
the reason spelled out in `franka_arm.py`: a joint vector is a
robot-specific *parameterisation* of a tool pose, and the tool pose is the part
that means the same thing on both robots.

The chain, outbound:

    VLA action (7 joints)
      -> FrankaArm.forward()           tool pose in the authoring arm's frame
      -> FrankaEpisodeFrame            the same pose in the *world*
      -> StretchPoseSolver.solve()     lift / arm / wrist / base targets

and inbound, the same path run backwards through `FrankaArm.inverse()`.

The world is the hinge. `episode_frame.py` reconstructs where the authoring
Franka's shoulder stood in the episode, so a VLA action lands at an absolute
world pose -- and the object the episode is about is at the same absolute world
pose for either robot. Nothing here needs a hand-fitted workspace calibration,
because the episode already contains one.

Three things this deliberately does *not* try to fix, all recorded in
`RemapTelemetry` instead:

- Stretch's tool cannot come closer than ~0.39m to its own base axis, and a
  Franka's home posture is tucked in at ~0.23m. The retarget of a pose in that
  hole is the nearest reachable one.
- Stretch has no independent lateral wrist DOF, so matching a pose exactly means
  turning the base a little (~0.13 rad in practice; see `pose_solver.py`).
- The cameras are Stretch's, not the Franka's. A model that has never seen a
  Stretch head camera is being asked to act on one. That is what
  `finetuning/` is for, and it is why `--remap` and fine-tuning are two paths to
  the same place rather than alternatives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

from examples.machine_learning.molmospaces.franka_remapping import episode_frame
from examples.machine_learning.molmospaces.franka_remapping.episode_frame import FrankaEpisodeFrame
from examples.machine_learning.molmospaces.franka_remapping.franka_arm import FrankaArm
from examples.machine_learning.molmospaces.franka_remapping.pose_solver import (
    EXACT_ORIENTATION_DOFS,
    StretchPoseSolver,
)
from examples.machine_learning.molmospaces.stretch.robot_view import StretchGripperGroup

if TYPE_CHECKING:
    from molmo_spaces.configs.robot_configs import BaseRobotConfig

log = logging.getLogger(__name__)

ACTION_SPACES = ("joint_position", "joint_velocity")
"""
How to read the seven arm numbers in a VLA action.

`joint_position` is absolute targets, which is what MolmoSpaces' own Franka
configs run (`FrankaRobotConfig.command_mode["arm"] == "joint_position"`) and
what its reference clients assume. `joint_velocity` integrates the action onto
the shadow arm instead, for checkpoints trained on DROID's velocity action
space.
"""

FRAME_SOURCES = ("episode", "mast")
"""
Which frame a VLA's joint numbers are taken to be expressed in.

`episode` uses the authoring arm's own recorded pose (`episode_frame.current()`),
falling back to the mast mount where no episode recorded one. That is the
faithful choice for a *pretrained* Franka checkpoint: the episode's author put
that shoulder exactly where its workspace covered the target, so an absolute
joint vector lands where the model meant it to.

`mast` always uses `episode_frame.default_frame_for()` -- a virtual Franka
bolted to Stretch's own base. Use it for a checkpoint fine-tuned on a dataset
from `finetuning/lerobot_export.py --action-space franka`, because that is the
frame the export encoded in. Getting this wrong is quiet and expensive: the two
frames differ by the standoff between the authoring Franka's shoulder and
Stretch's mast, so every action a fine-tuned model emits would be offset by it
and the arm would reach consistently short or long.
"""

GRIPPER_MAPPINGS = ("normalized", "aperture")
"""
How to turn the VLA's [0, 1] gripper channel into Stretch's finger joints.

`normalized` maps the channel across Stretch's whole travel, so "open" opens all
the way. `aperture` matches the Robotiq's *absolute* fingertip separation
instead, which is more faithful to the trained behaviour but leaves Stretch's
fingers only 46% open when the model asks for fully open -- the Robotiq spans
87mm and Stretch spans 188mm. `normalized` is the default because Stretch's
fingers are the longer pair: an approach with them half-closed sweeps the object
off its support before the grasp.
"""


@dataclass
class RemapTelemetry:
    """What the last remap cost, for logging and for the report."""

    position_error: float = 0.0
    """Metres between the pose the VLA asked for and the pose Stretch can hold."""

    orientation_error: float = 0.0
    """Radians of the same."""

    base_rotation: float = 0.0
    """Radians the base has turned from where the episode placed it."""

    base_translation: float = 0.0
    """Metres the base has moved from where the episode placed it."""

    shadow_position_error: float = 0.0
    """Metres the shadow arm's IK missed by when building the VLA's proprioception."""

    clipped_joints: tuple[str, ...] = field(default_factory=tuple)
    """Stretch joints sitting on a limit in the last solve."""

    unreachable_steps: int = 0
    """How many steps so far asked for a pose Stretch could not reach."""

    total_steps: int = 0
    """How many steps have been remapped this episode."""

    def as_dict(self) -> dict:
        return {
            "remap_position_error_m": round(self.position_error, 4),
            "remap_orientation_error_rad": round(self.orientation_error, 4),
            "remap_base_rotation_rad": round(self.base_rotation, 4),
            "remap_base_translation_m": round(self.base_translation, 4),
            "remap_shadow_position_error_m": round(self.shadow_position_error, 4),
            "remap_clipped_joints": list(self.clipped_joints),
            "remap_unreachable_steps": self.unreachable_steps,
            "remap_total_steps": self.total_steps,
            "remap_unreachable_fraction": round(
                self.unreachable_steps / max(self.total_steps, 1), 3
            ),
        }


class FrankaActionRemapper:
    """Two-way translation between a Franka-space policy and Stretch's move groups.

    One instance per episode's worth of work; call `reset()` between episodes.
    Construction compiles a scratch Stretch and a scratch Franka, so build it
    once per policy and reset it per episode rather than the other way round.
    """

    def __init__(
        self,
        robot_config: "BaseRobotConfig",
        action_space: str = "joint_position",
        gripper_mapping: str = "normalized",
        frame_source: str = "episode",
        dofs: tuple[str, ...] = EXACT_ORIENTATION_DOFS,
        max_base_rotation: float = np.pi,
        max_base_translation: float = 0.0,
        unreachable_threshold_m: float = 0.02,
        velocity_dt: float = 1.0 / 15.0,
    ) -> None:
        if action_space not in ACTION_SPACES:
            raise ValueError(f"action_space must be one of {ACTION_SPACES}, got {action_space!r}")
        if gripper_mapping not in GRIPPER_MAPPINGS:
            raise ValueError(
                f"gripper_mapping must be one of {GRIPPER_MAPPINGS}, got {gripper_mapping!r}"
            )
        if frame_source not in FRAME_SOURCES:
            raise ValueError(f"frame_source must be one of {FRAME_SOURCES}, got {frame_source!r}")

        self.franka = FrankaArm()
        self.solver = StretchPoseSolver(robot_config)
        self.action_space = action_space
        self.gripper_mapping = gripper_mapping
        self.frame_source = frame_source
        self.dofs = tuple(dofs)
        self.max_base_rotation = max_base_rotation
        self.max_base_translation = max_base_translation
        self.unreachable_threshold_m = unreachable_threshold_m
        self.velocity_dt = velocity_dt

        self.frame: FrankaEpisodeFrame | None = None
        self.telemetry = RemapTelemetry()
        self._shadow_qpos = self.franka.clip(episode_frame.default_frame_for(np.eye(4)).init_qpos)
        self._episode_base_pose = np.eye(4)
        self._last_solution: dict[str, np.ndarray] | None = None

    # =========================================================================
    # Episode lifecycle
    # =========================================================================

    def reset(self, base_pose: np.ndarray) -> None:
        """Start a new episode from the frame `episode_frame.current()` recorded.

        Args:
            base_pose: 4x4 world pose Stretch's base was placed at. Kept so the
                base-motion budgets are measured against where the episode put
                the robot rather than against wherever it has crept to, which is
                the mistake `policies/scripted.py` documents in
                `_remaining_base_budget`.
        """
        recorded = episode_frame.current() if self.frame_source == "episode" else None
        self.frame = recorded or episode_frame.default_frame_for(base_pose)
        self._episode_base_pose = np.asarray(base_pose, dtype=float).copy()
        self._shadow_qpos = self.franka.clip(self.frame.init_qpos)
        self._last_solution = None
        self.telemetry = RemapTelemetry()
        log.info(
            f"[remap] episode frame from {self.frame.metadata.get('source', 'episode')}; "
            f"authoring arm at {np.round(self.frame.base_pose[:3, 3], 3).tolist()}"
        )

    @property
    def handover_tool_pose(self) -> np.ndarray:
        """World tool pose the authoring arm starts the episode in.

        Driving Stretch here before the first VLA query is the closest thing to
        letting the model start where it expects to: it is exactly the pose
        `episode_spec.robot.init_qpos` would have put the Franka's gripper in.
        Stretch spawns stowed instead -- see `stretch_home_init_qpos()` for why
        it has to -- so somebody has to close that gap, and doing it before the
        model is consulted keeps it out of the model's action history.
        """
        return self.frame.tool_pose_to_world(self.franka.forward(self.frame.init_qpos))

    # =========================================================================
    # Inbound: Stretch -> the VLA's observation
    # =========================================================================

    def observation(self, tool_pose_world: np.ndarray, gripper_closedness: float) -> dict:
        """The Franka-space proprioception a VLA expects, from Stretch's real state.

        Args:
            tool_pose_world: 4x4 world pose of Stretch's tool centre, read from
                the live robot view.
            gripper_closedness: Stretch's gripper on a [0, 1] closed scale.

        Returns:
            `{"joint_position": (7,), "gripper_position": float}`, in the units
            and sign conventions of `observation/joint_position` and
            `observation/gripper_position` in MolmoSpaces' reference clients.

        The seven joints are the shadow arm's, solved to put *its* tool where
        Stretch's actually is. Feeding the model back a pose it can verify
        against the images is the point: an open-loop shadow that just echoed
        the last action would hide every metre of tracking error the retarget
        introduced.
        """
        target = self.frame.tool_pose_from_world(tool_pose_world)
        solution = self.franka.inverse(target, seed=self._shadow_qpos)
        self._shadow_qpos = solution.qpos
        self.telemetry.shadow_position_error = solution.position_error
        return {
            "joint_position": solution.qpos.copy(),
            "gripper_position": float(np.clip(gripper_closedness, 0.0, 1.0)),
        }

    # =========================================================================
    # Outbound: the VLA's action -> Stretch
    # =========================================================================

    def action(
        self,
        vla_action: np.ndarray,
        base_pose: np.ndarray,
        current_qpos: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """A VLA's 8-vector -> absolute per-move-group targets for Stretch.

        Args:
            vla_action: `[q1..q7, gripper]`. The seven arm numbers are read
                according to `action_space`; the gripper channel is 0 open,
                1 closed, as the reference clients emit it.
            base_pose: 4x4 world pose of Stretch's base right now.
            current_qpos: the live `{group: joint_pos}` mapping, used to seed the
                solve so successive steps stay continuous.

        Returns:
            `{"base", "lift", "arm", "wrist", "gripper"}`, absolute targets.
        """
        vla_action = np.asarray(vla_action, dtype=float).reshape(-1)
        if vla_action.size < 8:
            raise ValueError(
                f"A Franka-space VLA action is 7 joints plus a gripper channel; got "
                f"{vla_action.size} numbers."
            )

        self._shadow_qpos = self._integrate_arm_action(vla_action[:7])
        tool_pose_world = self.frame.tool_pose_to_world(self.franka.forward(self._shadow_qpos))
        action = self.tool_pose_action(tool_pose_world, base_pose, current_qpos)
        action["gripper"] = self.gripper_action(float(vla_action[7]))
        return action

    def tool_pose_action(
        self,
        tool_pose_world: np.ndarray,
        base_pose: np.ndarray,
        current_qpos: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Solve Stretch's joints for a world tool pose and record what it cost.

        Separate from `action()` so the handover phase -- which has a tool pose
        but no VLA action behind it -- goes through exactly the same solve, and
        so `finetuning/lerobot_export.py` can reuse it.
        """
        solution = self.solver.solve(
            np.asarray(base_pose, dtype=float),
            tool_pose_world,
            seed=self._seed_from(current_qpos),
            dofs=self.dofs,
            max_base_rotation=self._remaining_rotation(base_pose),
            max_base_translation=self._remaining_translation(base_pose),
        )
        self._last_solution = solution.configuration

        self.telemetry.total_steps += 1
        self.telemetry.position_error = solution.position_error
        self.telemetry.orientation_error = solution.orientation_error
        self.telemetry.clipped_joints = solution.clipped_joints
        self.telemetry.base_rotation = _wrap(
            float(solution.configuration["base"][2]) - _yaw_of(self._episode_base_pose)
        )
        self.telemetry.base_translation = float(
            np.linalg.norm(solution.configuration["base"][:2] - self._episode_base_pose[:2, 3])
        )
        if solution.position_error > self.unreachable_threshold_m:
            self.telemetry.unreachable_steps += 1
            log.debug(
                f"[remap] requested tool pose is {solution.position_error:.3f}m out of reach; "
                f"commanding the nearest pose instead (joints at limits: "
                f"{solution.clipped_joints or 'none'})"
            )

        return {
            "base": solution.configuration["base"].copy(),
            "lift": solution.configuration["lift"].copy(),
            "arm": solution.configuration["arm"].copy(),
            "wrist": solution.configuration["wrist"].copy(),
        }

    def gripper_action(self, closedness: float) -> np.ndarray:
        """The VLA's [0, 1] gripper channel -> a Stretch gripper command.

        Stretch has one commanded gripper degree of freedom (`stretch_gripper`),
        so the two numbers returned are the same target twice: the MJCF models
        the fingers as a mirrored pair and `StretchGripperGroup` owns both, but
        there is only ever one thing to say to them.
        """
        closedness = float(np.clip(closedness, 0.0, 1.0))
        if self.gripper_mapping == "normalized":
            target = (1.0 - closedness) * StretchGripperGroup.OPEN_JOINT_POS
        else:
            aperture = self.franka.gripper_aperture_m(closedness)
            low, high = StretchGripperGroup.INTER_FINGER_DIST_RANGE
            fraction = (aperture - low) / (high - low)
            target = float(np.clip(fraction, 0.0, 1.0)) * StretchGripperGroup.OPEN_JOINT_POS
        return np.array([target, target])

    @staticmethod
    def stretch_gripper_closedness(gripper_qpos: np.ndarray) -> float:
        """Stretch's gripper opening -> the [0, 1] closedness a VLA is fed.

        Averages the mirrored finger pair, which is the same number twice for a
        commanded pose and differs by hundredths of a radian when the fingers are
        loaded unevenly.

        The inverse of `gripper_action` under the `normalized` mapping. Used for
        the observation direction, where the point is a number the model can
        read rather than a faithful aperture -- so it is the normalised one
        regardless of which mapping the action direction uses, and the two agree
        by construction in the default configuration.
        """
        opening = float(np.mean(np.asarray(gripper_qpos, dtype=float).reshape(-1)[:2]))
        return float(np.clip(1.0 - opening / StretchGripperGroup.OPEN_JOINT_POS, 0.0, 1.0))

    # =========================================================================
    # Internals
    # =========================================================================

    def _integrate_arm_action(self, arm_action: np.ndarray) -> np.ndarray:
        """Apply the seven arm numbers to the shadow arm, per `action_space`."""
        if self.action_space == "joint_position":
            return self.franka.clip(arm_action)
        return self.franka.clip(self._shadow_qpos + arm_action * self.velocity_dt)

    def _seed_from(self, current_qpos: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Seed the pose solve from the live robot, or from the last solve.

        Preferring the last *solution* over the live joints when one exists
        keeps the solve continuous even while the position controllers are still
        catching up to it, which they always are: at 15Hz the arm is a step or
        two behind every commanded target.
        """
        if self._last_solution is not None:
            return self._last_solution
        return {
            "lift": np.asarray(current_qpos["lift"], dtype=float).copy(),
            "arm": np.asarray(current_qpos["arm"], dtype=float).copy(),
            "wrist": np.asarray(current_qpos["wrist"], dtype=float).copy(),
        }

    def _remaining_rotation(self, base_pose: np.ndarray) -> float:
        """Rotation budget left, measured from where the episode placed the base.

        The solver caps rotation relative to the pose it is handed, and it is
        handed the *live* pose every step, so without this the base could turn
        the full allowance once per step and spin on the spot.
        """
        turned = abs(_wrap(_yaw_of(base_pose) - _yaw_of(self._episode_base_pose)))
        return max(0.0, self.max_base_rotation - turned)

    def _remaining_translation(self, base_pose: np.ndarray) -> float:
        """Translation budget left, measured from where the episode placed the base."""
        if self.max_base_translation <= 0.0:
            return 0.0
        moved = float(np.linalg.norm(base_pose[:2, 3] - self._episode_base_pose[:2, 3]))
        return max(0.0, self.max_base_translation - moved)


def _yaw_of(pose: np.ndarray) -> float:
    return float(np.arctan2(pose[1, 0], pose[0, 0]))


def _wrap(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def pose_from_position_quaternion(pose7) -> np.ndarray:
    """`[x, y, z, qw, qx, qy, qz]` -> a 4x4 pose. Re-exported for convenience."""
    pose7 = np.asarray(pose7, dtype=float).reshape(-1)
    pose = np.eye(4)
    pose[:3, 3] = pose7[:3]
    pose[:3, :3] = R.from_quat(pose7[3:7], scalar_first=True).as_matrix()
    return pose
