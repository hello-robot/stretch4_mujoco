"""
Full 6-DOF tool-pose solving for Stretch 4.

`policies/kinematics.py` solves for a tool *position* with the approach chosen
by the caller, which is all a scripted grasp needs. A retargeted VLA action is a
whole pose -- the model chose an orientation, and that choice is most of what
distinguishes a grasp that closes on the object from one that closes beside it --
so this module solves position and orientation together.

Stretch's wrist makes that unusually clean. Measured on the compiled MJCF (and
asserted in `tests/test_franka_remapping.py`), the tool's orientation in the
base frame is

    R_tool = Rz(wrist_yaw) @ Ry(wrist_pitch) @ Rx(wrist_roll)

to machine precision: the three wrist joints are a textbook ZYX Euler triple.
Add the base's own yaw and the tool's orientation in the *world* is

    R_tool_world = Rz(base_yaw + wrist_yaw) @ Ry(wrist_pitch) @ Rx(wrist_roll)

So a requested world orientation can be read straight off as Euler angles
(`fit_wrist()`): pitch and roll are forced, and only the *sum* of
base yaw and wrist yaw is constrained.

That leaves the interesting question, which is what to do with the one wrist
joint that is also Stretch's only source of lateral reach. Two answers, both
implemented, selected by which DOF the caller puts in `dofs`:

`yaw_split` -- **exact orientation.** Base yaw and wrist yaw move in opposite
    directions (`base_yaw += s, wrist_yaw -= s`), which swings the arm around
    the base axis while the tool keeps pointing exactly where it was asked to.
    Position then solves over `(lift, arm, s)`: square, and -- unlike the
    wrist-yaw solve -- not singular in a top-down grasp, because its lever arm
    is base-axis-to-wrist-axis rather than wrist-axis-to-tool. The cost is that
    the base turns, which swings Stretch's head camera.

`wrist_yaw` -- **exact position, free azimuth.** The wrist yaw is spent on
    reaching instead, and the approach *azimuth* comes out as whatever that
    leaves. Cheap, and the error is usually small because
    `episode_overrides.retarget_base_pose()` has already aimed the base at the
    target, but it goes singular reaching straight down for the reason
    `policies/kinematics.py` documents.

Neither is universally better, so `solve()` takes the choice and
`StretchPoseSolution` reports both residuals. `action_remap.py` picks.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from mujoco import MjSpec
from scipy.spatial.transform import Rotation as R

from examples.machine_learning.molmospaces.stretch.robot_view import Stretch4RobotView

if TYPE_CHECKING:
    from molmo_spaces.configs.robot_configs import BaseRobotConfig

log = logging.getLogger(__name__)

EXACT_ORIENTATION_DOFS = ("lift", "arm", "yaw_split")
"""Match the requested orientation exactly and turn the base to reach. See module docstring."""

FREE_AZIMUTH_DOFS = ("lift", "arm", "wrist_yaw")
"""Match position with the wrist and let the approach azimuth fall where it may."""

TRANSLATING_DOFS = ("lift", "arm", "yaw_split", "base_x", "base_y")
"""`EXACT_ORIENTATION_DOFS` plus permission to slide the base."""

_DOF_WEIGHT = {
    "lift": 1.0,
    "arm": 1.0,
    "yaw_split": 1.0,
    "wrist_yaw": 1.0,
    # As in `policies/kinematics.py`: sliding the base is a last resort, worth
    # two orders of magnitude less than moving the arm.
    "base_x": 0.01,
    "base_y": 0.01,
}

NATURAL_PITCH_RANGE = np.array([-1.135, 1.7])
"""
The band of wrist pitch a retarget prefers to stay inside, in radians.

The MJCF lets pitch run to 4.276 rad, because every joint of Stretch's DW4 wrist
is the same servo with the same ~5.4 rad of travel. That matters here because a
ZYX triple is not unique: any orientation also has a mirror representation
`(A + pi, pi - p, r + pi)`, and the wide pitch limit makes *both* of them legal.
They describe the same tool orientation but very different arm postures -- the
mirror folds the gripper back over the telescoping arm -- and the folded one
costs roughly 0.3m of forward reach, which is enough to turn a reachable grasp
into a saturated arm.

So the mirror branch is scored down outside this band rather than forbidden:
0 rad points the tool along the base's +x, `pi/2` points it straight down, and
the band covers a little past both, which is every approach a benchmark grasp
actually asks for.
"""

GIMBAL_LOCK_TOLERANCE = 0.05
"""
How close to vertical the approach has to be, in radians of `cos(pitch)`, before
the azimuth/roll split is treated as free.

At `pitch = +-pi/2` the tool's approach axis is the base's z axis and the Euler
decomposition degenerates: `Rz(A) Ry(pi/2) Rx(r)` depends only on `A - r` (and
on `A + r` at `-pi/2`). That is exactly the top-down grasp, i.e. the most common
orientation in the whole benchmark, so the degeneracy is the normal case rather
than an edge case -- and it is a gift: it means a top-down approach can put its
azimuth wherever the wrist can reach and pay for it in roll, which for a
symmetric two-finger gripper reaching straight down is very nearly free.
"""


@dataclass
class StretchPoseSolution:
    """Joint targets for a requested tool pose, and how well they match it."""

    configuration: dict[str, np.ndarray]
    """`{"base": (3,), "lift": (1,), "arm": (1,), "wrist": (3,)}`, as the move groups take them."""

    position_error: float
    """Metres between the solved tool position and the requested one."""

    orientation_error: float
    """Radians between the solved tool orientation and the requested one."""

    base_rotation: float
    """Radians the solve turned the base by, signed. Watch this one: it moves the cameras."""

    base_translation: float
    """Metres the solve slid the base by. Zero unless `base_x`/`base_y` were in `dofs`."""

    converged: bool
    """Whether the position residual met the requested tolerance."""

    clipped_joints: tuple[str, ...] = field(default_factory=tuple)
    """Which joints the requested pose pushed into a limit, for logging."""


def wrist_candidates(
    rotation: np.ndarray, reference_azimuth: float
) -> list[tuple[float, float, float]]:
    """Every `(azimuth, pitch, roll)` triple that reproduces `rotation` exactly.

    `Rz(azimuth) @ Ry(pitch) @ Rx(roll)` equals `rotation` for all of them --
    they are alternative *representations*, not approximations -- and which one
    Stretch should use is a question about joint limits and arm posture rather
    than about orientation. Three sources of multiplicity:

    - the canonical ZYX decomposition;
    - its mirror branch `(A + pi, pi - p, r + pi)`, the same orientation with
      the wrist folded over (see `NATURAL_PITCH_RANGE` for why that is not free);
    - near-vertical approaches, where the decomposition degenerates and azimuth
      and roll trade off one-for-one (see `GIMBAL_LOCK_TOLERANCE`). Those get a
      candidate with the azimuth moved to `reference_azimuth` and the difference
      paid for in roll, which is the whole reason a top-down grasp can be
      retargeted onto Stretch without giving anything up.

    Args:
        rotation: 3x3 orientation to represent.
        reference_azimuth: the azimuth the caller would prefer, used only to
            place the degenerate candidate.
    """
    with warnings.catch_warnings():
        # SciPy warns on every gimbal-locked decomposition, and a top-down grasp
        # *is* gimbal-locked -- so this would fire on most steps of most
        # episodes. The degeneracy is not a problem here, it is the thing the
        # third candidate below exploits.
        warnings.filterwarnings("ignore", message="Gimbal lock detected")
        azimuth, pitch, roll = (float(value) for value in R.from_matrix(rotation).as_euler("ZYX"))
    candidates = [
        (azimuth, pitch, roll),
        (azimuth + np.pi, np.pi - pitch, roll + np.pi),
    ]
    for candidate_azimuth, candidate_pitch, candidate_roll in list(candidates):
        if abs(np.cos(candidate_pitch)) < GIMBAL_LOCK_TOLERANCE:
            shift = _wrap(reference_azimuth - candidate_azimuth)
            sign = 1.0 if np.sin(candidate_pitch) > 0 else -1.0
            candidates.append(
                (candidate_azimuth + shift, candidate_pitch, candidate_roll + sign * shift)
            )
    return candidates


def fit_wrist(
    rotation: np.ndarray,
    base_yaw: float,
    wrist_limits: np.ndarray,
    wrist_yaw: float | None = None,
) -> np.ndarray:
    """The wrist joint triple that best holds `rotation`, given the base's yaw.

    This is the one place the two solving modes differ, and they differ only in
    what the wrist yaw is for:

    - `wrist_yaw=None` (the exact-orientation mode): the yaw is chosen to hit
      the requested azimuth, and the azimuth is therefore matched exactly.
      Reaching is left to the yaw *split*, which does not disturb it.
    - `wrist_yaw=<value>` (the free-azimuth mode): the yaw has already been
      spent on reaching, so the azimuth is whatever it leaves. Pitch and roll
      are then fitted to the requested orientation *at that yaw*, which -- for
      a near-vertical approach -- absorbs the whole discrepancy into roll and
      costs nothing. That fit is why a reaching wrist does not have to mean a
      mis-oriented gripper.

    Args:
        rotation: 3x3 world orientation the tool should hold.
        base_yaw: the base's current yaw.
        wrist_limits: `(3, 2)` joint limits for yaw, pitch and roll.
        wrist_yaw: the yaw to keep, or None to solve for it.

    Returns:
        `(yaw, pitch, roll)`, clipped into `wrist_limits`.
    """
    solving_for_yaw = wrist_yaw is None
    reference_azimuth = base_yaw if solving_for_yaw else base_yaw + wrist_yaw

    # An azimuth the wrist cannot deliver is an orientation error, so it is worth
    # much more when the yaw is already committed than when it is still free.
    azimuth_weight = 0.2 if solving_for_yaw else 5.0

    best = None
    best_score = np.inf
    for azimuth, pitch, roll in wrist_candidates(rotation, reference_azimuth):
        pitch = _in_limits_revolution(pitch, wrist_limits[1])
        roll = _in_limits_revolution(roll, wrist_limits[2])
        yaw = (
            _in_limits_revolution(azimuth - base_yaw, wrist_limits[0])
            if solving_for_yaw
            else wrist_yaw
        )
        score = (
            # Outside a joint limit is disqualifying, whichever joint it is.
            10.0
            * (
                _limit_excess(pitch, wrist_limits[1])
                + _limit_excess(roll, wrist_limits[2])
                + _limit_excess(yaw, wrist_limits[0])
            )
            # Then: keep the wrist in a posture the arm can reach out of.
            + 2.0 * _limit_excess(pitch, NATURAL_PITCH_RANGE)
            + azimuth_weight * abs(_wrap(azimuth - (base_yaw + yaw)))
            + 0.5 * abs(roll)
        )
        if score < best_score:
            best_score = score
            best = np.array(
                [
                    np.clip(yaw, *wrist_limits[0]),
                    np.clip(pitch, *wrist_limits[1]),
                    np.clip(roll, *wrist_limits[2]),
                ]
            )
    return best


class StretchPoseSolver:
    """Solves Stretch's joints for a world tool pose, on a scratch copy of the robot.

    A private, mesh-free copy of the robot attached to an empty scene, so a
    candidate pose can be evaluated without disturbing the live simulation.

    Note this is *not* how `policies/kinematics.StretchReachSolver` works any
    more: that one solves through `stretch4_kinematics` (Pinocchio) and keeps a
    MuJoCo copy only long enough to measure the fixed offset between the two
    models' base frames. This solver is still the MuJoCo-native one because it
    exists to answer a different question -- what the *retarget* should do with a
    Franka-authored pose -- and is not on the per-step policy path.
    """

    def __init__(self, robot_config: "BaseRobotConfig") -> None:
        spec = MjSpec()
        robot_config.robot_cls.add_robot_to_scene(
            robot_config,
            spec,
            prefix=robot_config.robot_namespace,
            pos=[0.0, 0.0, 0.0],
            quat=[1.0, 0.0, 0.0, 0.0],
            strip_meshes=True,
        )
        self._model = spec.compile()
        self._data = mujoco.MjData(self._model)
        self._view = Stretch4RobotView(self._data, robot_config.robot_namespace)
        self._limits = {
            group: self._view.get_move_group(group).joint_pos_limits
            for group in ("lift", "arm", "wrist")
        }

    @property
    def joint_limits(self) -> dict[str, np.ndarray]:
        """Per-group `(n, 2)` joint limit arrays for lift, arm and wrist."""
        return self._limits

    def forward(self, configuration: dict[str, np.ndarray]) -> np.ndarray:
        """World 4x4 tool pose for a `{group: joint_pos}` mapping."""
        self._apply(configuration)
        return self._view.get_move_group("gripper").leaf_frame_to_world

    def solve(
        self,
        base_pose: np.ndarray,
        target_pose: np.ndarray,
        seed: dict[str, np.ndarray] | None = None,
        dofs: tuple[str, ...] = EXACT_ORIENTATION_DOFS,
        max_base_rotation: float = np.pi,
        max_base_translation: float = 0.0,
        tolerance: float = 5e-3,
        max_iterations: int = 80,
        step_damping: float = 1e-5,
    ) -> StretchPoseSolution:
        """Joint targets that put the tool at `target_pose`.

        Args:
            base_pose: 4x4 world pose the base starts at.
            target_pose: 4x4 world pose the tool should reach.
            seed: starting configuration keyed by move group. Seeding from the
                robot's current pose keeps successive solves continuous, which
                matters when this runs every policy step.
            dofs: which degrees of freedom to solve position over. See
                `EXACT_ORIENTATION_DOFS` and `FREE_AZIMUTH_DOFS`; including
                `wrist_yaw` selects the free-azimuth behaviour.
            max_base_rotation: radians the yaw split may turn the base from
                `base_pose`. A guard, not a tuning knob -- raise it rather than
                lower it. Measured over the pick benchmark's grasp
                trajectories, the split a good solve actually wants is 0.13 rad
                (0.16 rad worst case), but a *binding* cap distorts the descent
                direction rather than merely truncating it: capping at 0.5 rad
                takes the median grasp-pose error from 5mm to 257mm, because the
                step gets clipped to the boundary every iteration and the wrist
                is then re-fitted to a base yaw the solve did not choose.
            max_base_translation: metres the base may slide. Only has an effect
                if `dofs` includes `base_x`/`base_y`.
            tolerance: position error, in metres, that counts as solved.
            max_iterations: give up after this many damped least-squares steps.
            step_damping: Levenberg-Marquardt damping.

        Returns:
            A `StretchPoseSolution`, always. The configuration is the best
            iterate found and `converged` says whether it met `tolerance`. A
            pose solve feeds a controller that has to be given *something* every
            step, so a near-miss is more useful than a None -- and the residuals
            are what tell the caller the VLA asked for a pose Stretch cannot
            hold.
        """
        target_pose = np.asarray(target_pose, dtype=float)
        base_pose = np.asarray(base_pose, dtype=float)
        free_azimuth = "wrist_yaw" in dofs
        target_rotation = target_pose[:3, :3]

        start_yaw = float(np.arctan2(base_pose[1, 0], base_pose[0, 0]))
        start_xy = base_pose[:2, 3].copy()
        configuration = self._seed(base_pose, seed, target_rotation, free_azimuth)
        weights = np.array([_DOF_WEIGHT[dof] for dof in dofs])

        best: StretchPoseSolution | None = None
        for _ in range(max_iterations):
            self._apply(configuration)
            achieved = self._view.get_move_group("gripper").leaf_frame_to_world
            error = target_pose[:3, 3] - achieved[:3, 3]
            residual = float(np.linalg.norm(error))

            if best is None or residual < best.position_error:
                best = StretchPoseSolution(
                    configuration={key: value.copy() for key, value in configuration.items()},
                    position_error=residual,
                    orientation_error=_orientation_error(achieved[:3, :3], target_rotation),
                    base_rotation=_wrap(float(configuration["base"][2]) - start_yaw),
                    base_translation=float(np.linalg.norm(configuration["base"][:2] - start_xy)),
                    converged=residual < tolerance,
                    clipped_joints=self._joints_at_limits(configuration),
                )
            if best.converged:
                return best

            jacobian = self._position_jacobian(dofs)
            weighted = jacobian * weights
            step = weights * (
                jacobian.T
                @ np.linalg.solve(weighted @ jacobian.T + step_damping * np.eye(3), error)
            )
            configuration = self._integrate(
                configuration,
                dofs,
                step,
                target_rotation,
                start_yaw,
                start_xy,
                max_base_rotation,
                max_base_translation,
                free_azimuth,
            )

        return best

    def _position_jacobian(self, dofs: tuple[str, ...]) -> np.ndarray:
        """3 x len(dofs) Jacobian of the tool position over the requested DOFs.

        `yaw_split` is not a joint, so its column is assembled here: turning the
        base by +s and the wrist yaw by -s is the difference of those two
        columns. Doing it as a linear combination rather than as a new MuJoCo
        DOF is what keeps the orientation exactly fixed along the step.
        """
        full = self._view.get_jacobian("gripper", ["base", "lift", "arm", "wrist"])[:3]
        columns = {
            "base_x": full[:, 0],
            "base_y": full[:, 1],
            "lift": full[:, 3],
            "arm": full[:, 4],
            "wrist_yaw": full[:, 5],
            "yaw_split": full[:, 2] - full[:, 5],
        }
        return np.stack([columns[dof] for dof in dofs], axis=1)

    def _seed(
        self,
        base_pose: np.ndarray,
        seed: dict[str, np.ndarray] | None,
        target_rotation: np.ndarray,
        free_azimuth: bool,
    ) -> dict[str, np.ndarray]:
        yaw = float(np.arctan2(base_pose[1, 0], base_pose[0, 0]))
        if seed is None:
            lift, arm, wrist_yaw = 0.6, 0.1, 0.0
        else:
            lift = float(np.asarray(seed["lift"]).reshape(-1)[0])
            arm = float(np.asarray(seed["arm"]).reshape(-1)[0])
            wrist_yaw = float(np.asarray(seed["wrist"]).reshape(-1)[0]) if "wrist" in seed else 0.0
        configuration = {
            "base": np.array([base_pose[0, 3], base_pose[1, 3], yaw]),
            "lift": np.array([np.clip(lift, *self._limits["lift"][0])]),
            "arm": np.array([np.clip(arm, *self._limits["arm"][0])]),
            "wrist": np.array([wrist_yaw, 0.0, 0.0]),
        }
        return self._refit_wrist(configuration, target_rotation, free_azimuth)

    def _refit_wrist(
        self,
        configuration: dict[str, np.ndarray],
        target_rotation: np.ndarray,
        free_azimuth: bool,
    ) -> dict[str, np.ndarray]:
        """Re-derive the wrist for the base yaw (and, if free, the wrist yaw) in hand.

        Called after every integration step, not once up front, because both
        modes make the wrist a function of something the step changes: the yaw
        split moves the base yaw, and the free-azimuth mode moves the wrist yaw
        itself. Re-fitting is what lets a near-vertical approach keep its exact
        orientation while the yaw wanders -- see `fit_wrist`.
        """
        configuration["wrist"] = fit_wrist(
            target_rotation,
            float(configuration["base"][2]),
            self._limits["wrist"],
            wrist_yaw=float(configuration["wrist"][0]) if free_azimuth else None,
        )
        return configuration

    def _joints_at_limits(self, configuration: dict[str, np.ndarray]) -> tuple[str, ...]:
        """Which of the solved joints are sitting on a limit, for logging."""
        names = []
        for group, joint_names in (
            ("lift", ("lift",)),
            ("arm", ("arm",)),
            ("wrist", ("wrist_yaw", "wrist_pitch", "wrist_roll")),
        ):
            values = np.asarray(configuration[group]).reshape(-1)
            for index, name in enumerate(joint_names):
                low, high = self._limits[group][index]
                if min(abs(values[index] - low), abs(values[index] - high)) < 1e-6:
                    names.append(name)
        return tuple(names)

    def _integrate(
        self,
        configuration: dict[str, np.ndarray],
        dofs: tuple[str, ...],
        step: np.ndarray,
        target_rotation: np.ndarray,
        start_yaw: float,
        start_xy: np.ndarray,
        max_base_rotation: float,
        max_base_translation: float,
        free_azimuth: bool,
    ) -> dict[str, np.ndarray]:
        updated = {key: value.copy() for key, value in configuration.items()}
        for dof, delta in zip(dofs, step):
            if dof == "lift":
                updated["lift"][0] = np.clip(updated["lift"][0] + delta, *self._limits["lift"][0])
            elif dof == "arm":
                updated["arm"][0] = np.clip(updated["arm"][0] + delta, *self._limits["arm"][0])
            elif dof == "wrist_yaw":
                updated["wrist"][0] = np.clip(
                    updated["wrist"][0] + delta, *self._limits["wrist"][0]
                )
            elif dof == "yaw_split":
                turned = _wrap(updated["base"][2] + delta - start_yaw)
                turned = float(np.clip(turned, -max_base_rotation, max_base_rotation))
                updated["base"][2] = start_yaw + turned
            elif dof == "base_x":
                updated["base"][0] += delta
            elif dof == "base_y":
                updated["base"][1] += delta

        travel = updated["base"][:2] - start_xy
        distance = float(np.linalg.norm(travel))
        if max_base_translation <= 0.0:
            updated["base"][:2] = start_xy
        elif distance > max_base_translation:
            updated["base"][:2] = start_xy + travel * (max_base_translation / distance)

        return self._refit_wrist(updated, target_rotation, free_azimuth)

    def _apply(self, configuration: dict[str, np.ndarray]) -> None:
        base = configuration["base"]
        pose = np.eye(4)
        pose[:2, 3] = base[:2]
        pose[:3, :3] = R.from_euler("z", base[2]).as_matrix()
        self._view.base.pose = pose
        self._view.get_move_group("lift").joint_pos = configuration["lift"]
        self._view.get_move_group("arm").joint_pos = configuration["arm"]
        self._view.get_move_group("wrist").joint_pos = configuration["wrist"]
        mujoco.mj_kinematics(self._model, self._data)
        mujoco.mj_comPos(self._model, self._data)


def _orientation_error(achieved: np.ndarray, requested: np.ndarray) -> float:
    """Angle, in radians, between two rotations."""
    relative = requested @ achieved.T
    return float(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))


def _wrap(angle: float) -> float:
    """An angle folded into (-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _limit_excess(value: float, limits: np.ndarray) -> float:
    """How far `value` falls outside `limits`. Zero if it is inside them."""
    return float(max(0.0, limits[0] - value, value - limits[1]))


def _in_limits_revolution(angle: float, limits: np.ndarray) -> float:
    """The revolution of `angle` that sits inside `limits`, if one does.

    Stretch's wrist yaw and pitch each run from -1.135 to 4.276 rad -- over five
    radians of travel, well past pi. So an angle naively folded into (-pi, pi]
    can land outside the limit while the same direction expressed a turn further
    round sits comfortably inside it, and clipping the folded value would swing
    the joint to a stop for a pose it can hold exactly.
    """
    candidates = [_wrap(angle) + turns * 2.0 * np.pi for turns in (-1, 0, 1)]
    inside = [value for value in candidates if limits[0] <= value <= limits[1]]
    if inside:
        return min(inside, key=abs)
    return min(candidates, key=lambda value: _limit_excess(value, limits))
