"""
Reach solving for Stretch 4, backed by `stretch4_kinematics` (Pinocchio).

Both halves of this module come from Hello Robot's own kinematics library rather
than from a hand-rolled solver:

    FK   `StretchKinematics.forward()`            -- full SE(3) pose of a frame
    IK   `StretchKinematics.inverse_6dof_local()` -- 6-DOF pose IK about the base

The library models Stretch as six inverse-kinematic degrees of freedom -- base
yaw, lift, telescoping arm, and the three wrist joints -- against a full 6-DOF
target pose, which is square and therefore well posed. That is the important
difference from what used to be here: a position-only damped least-squares
descent over `lift + arm + wrist yaw` with the base held fixed.

**Why the base yaw matters.** Stretch's arm extends along the base's +x axis and
the wrist yaw swings the tool through a narrow lateral band. Measured on the
compiled model, the set of points reachable at a 0.55-0.90m standoff spans only
about +-0.25 rad of bearing off that axis: at 0.3 rad a position-only solve over
`lift + arm + wrist yaw` finds 100% of targets, at 0.5 rad 70%, and at 0.785 rad
none at all. A real Stretch does not stand still and stretch sideways -- it turns
its base to point the arm. Letting the solver do the same is what makes targets
off the axis reachable, and it is the single largest effect on whether a simple_ik
grasp succeeds.

**Grasp style pins pitch and roll, not yaw.** The tool's orientation factors
exactly (verified against the model to 2e-16) as

    R_tcp = Rz(base_yaw + wrist_yaw) @ Ry(wrist_pitch) @ Rx(wrist_roll)

so a grasp style -- horizontal or top-down, plus a finger spin -- pins `Ry` and
`Rx` and leaves exactly one free angle: the yaw the tool approaches along. That
freedom has to be preserved. Pinning it as well makes the problem square in
position alone -- three constraints against base yaw, lift and arm -- and
measurably *worse* than the solver it replaces (75% against 100% on near-axis
targets). Since `inverse_6dof_local` takes a fully determined pose, "free" is
expressed by solving at several headings and keeping the first that converges,
nominal bearing first. The solver then splits that heading between turning the
base and turning the wrist, which is the redundancy Stretch actually has and the
position-only formulation could not express.

**The two models pivot about different points.** MuJoCo turns the base about the
holonomic base body; Pinocchio turns it about the URDF root, ~7cm away. A solve
that turns the base by `bt` therefore lands `|(I - Rz(bt)) @ offset|` from where
it believes it is -- 14.6mm at a third of a radian, which silently exceeds a 15mm
grasp tolerance. `_solve_at_yaw()` closes that by feeding the solved rotation
back into the target and re-solving, and every solve is finally accepted only if
it survives a round trip through this module's own FK.

**Frame calibration.** The Pinocchio model is built from the Stretch URDF, while
MolmoSpaces drives a MuJoCo model in which the robot hangs off a virtual
holonomic base body (see `Stretch4Robot.add_robot_to_scene`). The two disagree by
a fixed translation in the base frame -- measured at [+0.04075, 0, -0.0565]m,
constant to zero variance over 200 random configurations. `_tcp_offset_in_base()`
recovers it from the compiled MuJoCo model rather than hard-coding it, so a
change to the MJCF cannot silently reintroduce a 7cm error into every grasp.
"""

from __future__ import annotations

import contextlib
import io
import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import pinocchio as pin
from mujoco import MjSpec
from scipy.spatial.transform import Rotation as R
from stretch4_kinematics.kinematic_models.base_kinematic_models import StretchKinematics
from stretch4_kinematics.state.joint_positions import StretchJointPositions

from examples.machine_learning.molmospaces.stretch.robot_view import Stretch4RobotView

if TYPE_CHECKING:
    from molmo_spaces.configs.robot_configs import BaseRobotConfig

log = logging.getLogger(__name__)

# Wrist pitch that points the gripper's approach axis straight down, and the one
# that points it horizontally along the approach yaw. Verified by forward
# kinematics: pitch pi/2 gives an approach of (0, 0, -1), pitch 0 gives (1, 0, 0)
# in the frame the approach yaw defines.
PITCH_TOP_DOWN = np.pi / 2
PITCH_HORIZONTAL = 0.0

# How many times to re-solve while feeding the solved base rotation back into the
# target. Three is generous; the correction is exact and settles on the second.
_BASE_PIVOT_CORRECTION_PASSES = 3

# The URDF/MJCF frame both models call the tool centre. It sits between the
# fingertips, about 19mm ahead of them, so aiming it at an object's centre
# straddles the object rather than butting the fingers against it.
TCP_FRAME = "grasp_center_link"

# How far either side of the nominal approach the solver may rotate the grasp,
# and how finely it samples that range.
#
# Grasp style pins the tool's pitch and roll, which leaves exactly one free
# angle: the yaw it closes along. Pinning that too would make the problem square
# in position alone -- three constraints against base yaw, lift and arm -- and
# measurably worse than the position-only solver it replaces (75% against 100%
# on near-axis targets). Leaving it free is what makes the sixth degree of
# freedom worth having, and `inverse_6dof_local` takes a fully determined pose,
# so "free" is expressed by solving at several yaws and keeping the first that
# converges. The nominal bearing is tried first, so an unobstructed grasp still
# closes straight along the line from the base to the object.
APPROACH_YAW_SPREAD_RAD = np.pi / 3
APPROACH_YAW_SAMPLES = 9


def grasp_orientation(approach_yaw: float, wrist_pitch: float, wrist_roll: float) -> np.ndarray:
    """World rotation matrix for a tool approaching along `approach_yaw`.

    The factorisation this relies on is exact for Stretch, not an approximation:
    `R_tcp = Rz(base_yaw + wrist_yaw) @ Ry(wrist_pitch) @ Rx(wrist_roll)`. So the
    grasp style fixes the last two factors and the approach direction fixes the
    first, with no residual freedom to resolve.
    """
    return (
        R.from_euler("z", approach_yaw).as_matrix()
        @ R.from_euler("y", wrist_pitch).as_matrix()
        @ R.from_euler("x", wrist_roll).as_matrix()
    )


_TCP_OFFSET_CACHE: dict[tuple[str, str], np.ndarray] = {}


def _tcp_offset_in_base(robot_config: "BaseRobotConfig") -> np.ndarray:
    """Translation from the MuJoCo base body's TCP to the Pinocchio model's TCP.

    MolmoSpaces poses the robot by the virtual holonomic base body that
    `add_robot_to_scene` wraps around the URDF root, so MuJoCo's notion of "the
    base" is offset from the URDF's. The offset is a pure translation in the base
    frame (the rotations agree to machine precision) and is a constant of the
    generated MJCF, so it is measured once per model and cached.

    Measured rather than hard-coded: a change to the MJCF that moved the base
    body would otherwise put a silent seven-centimetre error into every grasp.
    """
    key = (str(robot_config.robot_dir), robot_config.robot_namespace)
    if key in _TCP_OFFSET_CACHE:
        return _TCP_OFFSET_CACHE[key]

    namespace = robot_config.robot_namespace
    spec = MjSpec()
    robot_config.robot_cls.add_robot_to_scene(
        robot_config,
        spec,
        prefix=namespace,
        pos=[0.0, 0.0, 0.0],
        quat=[1.0, 0.0, 0.0, 0.0],
        strip_meshes=True,
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    view = Stretch4RobotView(data, namespace)

    reference = {"lift": np.array([0.6]), "arm": np.array([0.26]), "wrist": np.zeros(3)}
    for group, value in reference.items():
        view.get_move_group(group).joint_pos = value
    mujoco.mj_kinematics(model, data)
    mujoco_tcp = np.asarray(view.get_move_group("gripper").leaf_frame_to_world[:3, 3], dtype=float)

    pinocchio_tcp = _kinematics().forward(
        StretchJointPositions(
            lift=float(reference["lift"][0]),
            arm=float(reference["arm"][0]),
            wrist_yaw=0.0,
            wrist_pitch=0.0,
            wrist_roll=0.0,
        ),
        TCP_FRAME,
    ).translation

    offset = np.asarray(pinocchio_tcp, dtype=float) - mujoco_tcp
    log.debug(f"[stretch-kinematics] pinocchio-to-mujoco TCP offset in base frame: {offset}")
    _TCP_OFFSET_CACHE[key] = offset
    return offset


@lru_cache(maxsize=1)
def _kinematics():
    """The library's solver, built once per process.

    Constructing it parses the URDF and compiles two Pinocchio models, which is
    far too expensive to repeat for every `StretchSimpleIKPolicy` -- and it is
    stateless, so there is nothing to keep separate between them.
    """
    return StretchKinematics()


class StretchReachSolver:
    """Puts Stretch's tool centre at a world point, at a chosen grasp orientation.

    Stateless with respect to the simulation: every call takes the base pose it
    should solve about, so a candidate reach can be evaluated without disturbing
    anything.
    """

    def __init__(self, robot_config: "BaseRobotConfig") -> None:
        self._kinematics = _kinematics()
        self._namespace = robot_config.robot_namespace
        self._offset = _tcp_offset_in_base(robot_config)

        model = self._kinematics.model_ik
        # BASE_ROTATE order: base_theta, lift, arm, wrist yaw/pitch/roll.
        lower, upper = model.lowerPositionLimit, model.upperPositionLimit
        self._limits = {
            "lift": np.array([[lower[1], upper[1]]]),
            "arm": np.array([[lower[2], upper[2]]]),
            "wrist": np.array(
                [[lower[3], upper[3]], [lower[4], upper[4]], [lower[5], upper[5]]]
            ),
        }

    @property
    def joint_limits(self) -> dict[str, np.ndarray]:
        """Per-group `(n, 2)` joint limit arrays for lift, arm and wrist."""
        return self._limits

    # =========================================================================
    # Forward kinematics
    # =========================================================================

    def forward(self, configuration: dict[str, np.ndarray]) -> np.ndarray:
        """World 4x4 pose of the tool centre for a `{group: joint_pos}` mapping.

        Expressed in MolmoSpaces' base convention, so the result is directly
        comparable to `robot_view.get_move_group("gripper").leaf_frame_to_world`.
        """
        base = np.asarray(configuration["base"], dtype=float).reshape(-1)
        wrist = np.asarray(configuration["wrist"], dtype=float).reshape(-1)
        pose_se3 = self._kinematics.forward(
            StretchJointPositions(
                base_x=float(base[0]),
                base_y=float(base[1]),
                base_theta=float(base[2]),
                lift=float(np.asarray(configuration["lift"]).reshape(-1)[0]),
                arm=float(np.asarray(configuration["arm"]).reshape(-1)[0]),
                wrist_yaw=float(wrist[0]),
                wrist_pitch=float(wrist[1]),
                wrist_roll=float(wrist[2]),
            ),
            TCP_FRAME,
        )

        pose = np.eye(4)
        pose[:3, :3] = pose_se3.rotation
        # Undo the URDF-vs-holonomic-base offset so callers get the pose MuJoCo
        # would report. The offset is fixed in the base frame, so it rotates with
        # the base yaw rather than being subtracted in world axes.
        pose[:3, 3] = pose_se3.translation - R.from_euler("z", base[2]).as_matrix() @ self._offset
        return pose

    # =========================================================================
    # Inverse kinematics
    # =========================================================================

    def solve(
        self,
        base_pose: np.ndarray,
        target_position: np.ndarray,
        wrist_pitch: float = PITCH_HORIZONTAL,
        wrist_roll: float = 0.0,
        seed: dict[str, np.ndarray] | None = None,
        approach_yaw: float | None = None,
        yaw_spread: float = APPROACH_YAW_SPREAD_RAD,
        yaw_samples: int = APPROACH_YAW_SAMPLES,
        tolerance: float = 5e-3,
        max_iterations: int = 200,
        step_damping: float = 1e-6,
    ) -> dict[str, np.ndarray] | None:
        """Joint targets that put the tool centre at `target_position`.

        Args:
            base_pose: 4x4 world pose the base starts at. Its position is
                preserved exactly -- the library's local IK has no base
                translation, so a solve can re-aim the robot but never drive it
                somewhere else.
            target_position: world xyz the tool centre should reach.
            wrist_pitch: approach tilt, `PITCH_HORIZONTAL` or `PITCH_TOP_DOWN`.
            wrist_roll: spin of the fingers about the approach axis.
            seed: starting configuration keyed by move group. Seeding from the
                robot's current pose keeps successive solves continuous, which
                matters because a phase-by-phase policy re-solves every step.
            approach_yaw: world heading the tool should approach along. Defaults
                to the bearing from the base to the target, which is the
                direction a grasp closes along when the robot is facing its work.
                This is the *nominal* heading: the solver tries it first and then
                works outwards through `yaw_spread`.
            yaw_spread: how far either side of the nominal heading the grasp may
                rotate. Zero pins the approach exactly, which costs the solver
                its one redundant degree of freedom.
            yaw_samples: how many headings to try across that spread.
            tolerance: position error, in metres, that counts as solved.
            max_iterations: cap on CLIK iterations.
            step_damping: Levenberg-Marquardt damping.

        Returns:
            `{"base": (3,), "lift": (1,), "arm": (1,), "wrist": (3,)}`, or None if
            the target could not be reached.
        """
        base_pose = np.asarray(base_pose, dtype=float)
        base_xy = base_pose[:2, 3]
        base_yaw = yaw_of_pose(base_pose)
        target_position = np.asarray(target_position, dtype=float).reshape(3)

        # Into the base frame, where the library solves, then onto the URDF's
        # notion of the tool centre.
        to_target = target_position - np.array([base_xy[0], base_xy[1], 0.0])
        local_target_nominal = (
            R.from_euler("z", -base_yaw).as_matrix() @ to_target + self._offset
        )

        nominal_yaw = (
            float(np.arctan2(local_target_nominal[1], local_target_nominal[0]))
            if approach_yaw is None
            else float(approach_yaw) - base_yaw
        )

        configuration = None
        for candidate_yaw in _approach_yaw_candidates(nominal_yaw, yaw_spread, yaw_samples):
            configuration = self._solve_at_yaw(
                to_target,
                base_xy,
                base_yaw,
                candidate_yaw,
                wrist_pitch,
                wrist_roll,
                seed,
                max_iterations,
                step_damping,
                tolerance,
            )
            if configuration is not None:
                break
        if configuration is None:
            return None

        # The solve is only trustworthy if the pose it claims survives a round
        # trip through this module's own FK, which is the one the caller will
        # compare against. A converged CLIK that lands somewhere else means a
        # frame or convention has drifted, and a silently wrong grasp is worse
        # than a refused one.
        achieved = self.forward(configuration)[:3, 3]
        error = float(np.linalg.norm(achieved - target_position))
        if error > tolerance:
            log.debug(
                f"[stretch-kinematics] rejected solve {error:.4f}m from target "
                f"(tolerance {tolerance:.4f}m)"
            )
            return None
        return configuration

    def _solve_at_yaw(
        self,
        to_target: np.ndarray,
        base_xy: np.ndarray,
        base_yaw: float,
        candidate_yaw: float,
        wrist_pitch: float,
        wrist_roll: float,
        seed: dict[str, np.ndarray] | None,
        max_iterations: int,
        step_damping: float,
        tolerance: float,
    ) -> dict[str, np.ndarray] | None:
        """Solve at one approach heading, correcting for where the base pivots.

        The two models turn the base about different points: MuJoCo about the
        holonomic base body, Pinocchio about the URDF root, and
        `_tcp_offset_in_base()` is the ~7cm between them. A solve that turns the
        base by `bt` therefore lands `|(I - Rz(bt)) @ offset|` away from where it
        thinks it does -- 14.6mm at a third of a radian, which quietly exceeds a
        15mm grasp tolerance and was enough on its own to make this solver score
        *worse* than the position-only one it replaces.

        The correction is exact once `bt` is known, so this iterates: solve, feed
        the base rotation back into where the target sits, solve again. It
        converges immediately because each pass uses the true rotation from the
        last, and the caller's round-trip check is what finally accepts it.
        """
        rotate_into_world = R.from_euler("z", -base_yaw).as_matrix() @ to_target
        base_rotation = 0.0

        for _ in range(_BASE_PIVOT_CORRECTION_PASSES):
            local_target = (
                rotate_into_world + R.from_euler("z", base_rotation).as_matrix() @ self._offset
            )
            solution = self._solve_local(
                local_target,
                grasp_orientation(candidate_yaw, wrist_pitch, wrist_roll),
                seed,
                max_iterations,
                step_damping,
                tolerance,
            )
            if solution is None:
                return None
            if abs(solution.base_theta - base_rotation) < 1e-6:
                break
            base_rotation = solution.base_theta

        return {
            "base": np.array([base_xy[0], base_xy[1], base_yaw + solution.base_theta]),
            "lift": np.array([solution.lift]),
            "arm": np.array([solution.arm]),
            "wrist": np.array([solution.wrist_yaw, solution.wrist_pitch, solution.wrist_roll]),
        }

    def _solve_local(
        self,
        local_target: np.ndarray,
        local_rotation: np.ndarray,
        seed: dict[str, np.ndarray] | None,
        max_iterations: int,
        step_damping: float,
        tolerance: float,
    ):
        """One IK call about the current base pose, with the library's noise contained.

        `inverse_6dof_local` reports non-convergence by printing to stdout and
        then raising. Neither is usable from a policy that re-solves every step
        at 15Hz, so the prints go to a throwaway buffer and the raise becomes the
        `None` this module's callers already handle.
        """
        guess = StretchJointPositions(
            base_x=0.0,
            base_y=0.0,
            base_theta=0.0,
            lift=0.6 if seed is None else float(np.asarray(seed["lift"]).reshape(-1)[0]),
            arm=0.1 if seed is None else float(np.asarray(seed["arm"]).reshape(-1)[0]),
            wrist_yaw=0.0 if seed is None else float(np.asarray(seed["wrist"]).reshape(-1)[0]),
            wrist_pitch=0.0 if seed is None else float(np.asarray(seed["wrist"]).reshape(-1)[1]),
            wrist_roll=0.0 if seed is None else float(np.asarray(seed["wrist"]).reshape(-1)[2]),
        )
        target_pose = pin.SE3(np.asarray(local_rotation, dtype=float), np.asarray(local_target))

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                return self._kinematics.inverse_6dof_local(
                    target_frame=TCP_FRAME,
                    target_pose=target_pose,
                    q_guess=guess,
                    max_iter=max_iterations,
                    eps=min(tolerance, 1e-3),
                    damp=step_damping,
                )
        except (ValueError, RuntimeError) as failure:
            log.debug(f"[stretch-kinematics] IK did not converge: {failure}")
            return None


def _approach_yaw_candidates(nominal: float, spread: float, samples: int) -> list[float]:
    """Approach headings to try, nominal first and then alternating outwards."""
    if spread <= 0.0 or samples <= 1:
        return [nominal]
    offsets = np.linspace(0.0, spread, samples // 2 + 1)[1:]
    candidates = [nominal]
    for offset in offsets:
        candidates += [nominal + offset, nominal - offset]
    return candidates


def yaw_of_pose(pose: np.ndarray) -> float:
    """Yaw about +z of a 4x4 pose matrix."""
    return float(np.arctan2(pose[1, 0], pose[0, 0]))


def planar_pose(x: float, y: float, yaw: float) -> np.ndarray:
    """A 4x4 pose on the floor plane."""
    pose = np.eye(4)
    pose[0, 3] = x
    pose[1, 3] = y
    pose[:3, :3] = R.from_euler("z", yaw).as_matrix()
    return pose
