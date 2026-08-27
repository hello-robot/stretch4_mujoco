"""
A simple IK-driven Stretch 4 expert for the manipulation and articulation benchmarks.

"Simple IK" is the whole design: every task compiles to a list of tool-centre
waypoints, and each waypoint is pursued by asking `policies/kinematics.py` for
the joint configuration that puts the tool there. There is no trajectory
optimisation and no search -- just a reach solve per step against a simple_ik
curve.

This policy serves two purposes:

1. **A baseline.** It is the non-learned number to beat on each of the eight
   benchmark evaluations, and the thing to look at when a learned policy scores
   zero -- if the expert also scores zero on a benchmark, the problem is the
   retargeting or the robot, not the policy.
2. **A teacher.** `finetuning/generate_dataset.py` runs it over procedurally
   sampled scenes, and both learners clone the successful trajectories:
   `training/` into a small from-scratch net, `finetuning/` into a pretrained
   VLA. This is why it uses privileged state (object poses read straight out of
   `MjData`) rather than the camera images: the student learns the mapping from
   images to these actions.

It is deliberately *not* a planner. There is no collision-aware motion planning
-- MolmoSpaces' own solvers do that, but they need CuRobo. What is here instead
is a waypoint machine: every task family compiles down to a list of `Waypoint`s
for the tool centre, and one executor drives them.

Grasps, though, are not invented here. MolmoSpaces ships authored grasp poses per
asset and the task samplers already require them (`filter_for_grasps`), so this
policy reads the object's grasp library and picks a pose out of it rather than
guessing at a tilt. Those libraries were generated for the DROID Franka gripper,
which is *not* a problem: a grasp pose records an approach axis and a closing
axis, not a robot configuration, and Stretch's fingers open to 0.19m against the
0.08m the grasps were authored for -- so anything a Franka can straddle, Stretch
can. `policies/kinematics.py:tcp_orientation_from_grasp` is the whole of the
translation. The hand-written horizontal and top-down styles survive as the
fallback for the ~5% of objects where no authored grasp is reachable.

That uniformity is possible because of Stretch's kinematics. A drawer pull, a
door swing and a place motion are all "hold the gripper closed and move the tool
along this curve"; the only thing that differs is the curve, and
`StretchReachSolver` will recruit the holonomic base when the arm alone cannot
follow it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from examples.machine_learning.molmospaces.policies.kinematics import (
    APPROACH_YAW_SPREAD_RAD,
    PITCH_HORIZONTAL,
    PITCH_TOP_DOWN,
    StretchReachSolver,
    grasp_orientation,
    tcp_orientation_from_grasp,
)
from examples.machine_learning.molmospaces.stretch.robot_view import StretchGripperGroup
from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.policy.base_policy import BasePolicy, PolicyFactory
from molmo_spaces.utils.function_utils import make_lenient

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
    from molmo_spaces.tasks.task import BaseMujocoTask

log = logging.getLogger(__name__)

# Joint arrival threshold for a joint-space waypoint, in radians (and metres for
# the prismatic groups, which are the same order of magnitude here).
_JOINT_ARRIVAL_TOLERANCE_RAD = 0.05

PICK_TASK_CLASSES = frozenset(
    {
        "molmo_spaces.tasks.pick_task.PickTask",
        "molmo_spaces.tasks.pick_and_place_task.PickAndPlaceTask",
        "molmo_spaces.tasks.pick_and_place_next_to_task.PickAndPlaceNextToTask",
        "molmo_spaces.tasks.pick_and_place_color_task.PickAndPlaceColorTask",
    }
)
PLACE_TASK_CLASSES = PICK_TASK_CLASSES - {"molmo_spaces.tasks.pick_task.PickTask"}
OPENING_TASK_CLASS = "molmo_spaces.tasks.opening_tasks.OpeningTask"
DOOR_OPENING_TASK_CLASS = "molmo_spaces.tasks.opening_tasks.DoorOpeningTask"


def _finger_joint_for_width(gripper: StretchGripperGroup, width_m: float) -> float:
    """Finger joint position that leaves the fingertips `width_m` apart.

    Interpolated between the gripper group's own declared endpoints rather than
    against a fitted constant. The mapping is linear to within 1.2mm across the
    whole range, checked against the compiled model, and deriving it from
    `inter_finger_dist_range` means a change to the fingers cannot leave a stale
    coefficient behind here.
    """
    closed_width, open_width = gripper.inter_finger_dist_range
    span = open_width - closed_width
    if span <= 0.0:
        return gripper.CLOSED_JOINT_POS
    if width_m >= open_width:
        # Wider than the fingers open. Clamping the *fraction* here would ask for
        # a fully open gripper, which is the opposite of a grasp -- an object too
        # big to straddle is one to squeeze, so fall back to closing.
        log.debug(
            f"[stretch-simple-ik] requested grip of {width_m:.3f}m exceeds the gripper's "
            f"{open_width:.3f}m opening; closing fully instead"
        )
        return gripper.CLOSED_JOINT_POS
    fraction = float(np.clip((width_m - closed_width) / span, 0.0, 1.0))
    return gripper.CLOSED_JOINT_POS + fraction * (gripper.OPEN_JOINT_POS - gripper.CLOSED_JOINT_POS)


@dataclass
class ToolGrasp:
    """A tool-centre pose to grasp an object at, and where it came from.

    The orientation is carried as the three angles the solver takes rather than
    as a matrix, because that is the form Stretch's kinematics factor into
    exactly -- see `policies/kinematics.py`. `rotation`, `approach` and
    `closing_axis` derive the geometry back out of them, so there is one source
    of truth for which way this grasp points.

    Attributes:
        position: world xyz to put the tool centre at.
        approach_yaw: world heading the tool closes along.
        wrist_pitch: approach tilt.
        wrist_roll: spin of the fingers about the approach axis.
        authored: this pose came out of the asset's grasp library, so all three
            angles are meant and the solver must honour them. A grasp built from
            a hand-written style instead only means its pitch and roll, and its
            yaw is a nominal heading the solver is free to improve on.
    """

    position: np.ndarray
    approach_yaw: float
    wrist_pitch: float
    wrist_roll: float
    authored: bool

    @property
    def solver_yaw(self) -> float | None:
        """The heading to pin the solve to, or None to leave it free.

        An authored grasp determines all three angles, so there is nothing left
        to search over. A styled grasp keeps the free yaw that is Stretch's one
        piece of redundancy, and giving it up measurably costs reach.
        """
        return self.approach_yaw if self.authored else None

    @property
    def rotation(self) -> np.ndarray:
        """World 3x3 orientation of the tool at this grasp."""
        return grasp_orientation(self.approach_yaw, self.wrist_pitch, self.wrist_roll)

    @property
    def approach(self) -> np.ndarray:
        """Unit vector the tool travels along as it closes on the object."""
        return self.rotation[:, 0]

    @property
    def closing_axis(self) -> np.ndarray:
        """Unit vector the fingers separate along."""
        return self.rotation[:, 1]


@dataclass
class Waypoint:
    """One tool-centre target in a simple_ik plan.

    Attributes:
        position: world xyz for the tool centre.
        wrist_pitch: approach tilt to hold while getting there.
        wrist_roll: finger spin about the approach axis to hold while getting there.
        approach_yaw: heading to pin the tool to, or None to let the solver
            choose one. See `ToolGrasp.solver_yaw`.
        gripper_open: whether the fingers should be open on arrival.
        label: phase name, surfaced through `PolicyPhaseSensor` and the logs.
        settle_steps: extra policy steps to hold once the position is reached,
            for motions whose effect is not visible in the tool pose -- closing
            the fingers being the obvious one.
        establishes_grasp: this waypoint is the one that closes on the object, so
            the tool-to-object offset it ends with is what a held grasp looks
            like.
        verify_grasp: check, on leaving this waypoint, that the object has come
            with the tool. See `_grasp_still_held()`.
        grip_width_m: how wide the object is where it will be gripped, when that
            could be measured. Recorded for the trajectory and used to reject
            grasp candidates the fingers cannot span; the closing itself goes by
            contact, not by this number. See `_finger_target()`.
        joint_targets: when set, drive these move groups straight to these joint
            values and ignore `position` entirely. For motions whose point is the
            configuration rather than where the tool ends up -- see the unstow
            step in `_plan_pick_and_place()`.
        turn_base: whether the solver may re-aim the base to reach this point.
            Stretch's arm has almost no lateral authority of its own, so turning
            the base is how it reaches anything off its +x axis; the solver
            cannot translate the base, only rotate it.
        tolerance: arrival threshold in metres.
    """

    position: np.ndarray
    wrist_pitch: float
    gripper_open: bool
    label: str
    wrist_roll: float = 0.0
    approach_yaw: float | None = None
    settle_steps: int = 0
    establishes_grasp: bool = False
    verify_grasp: bool = False
    grip_width_m: float | None = None
    joint_targets: dict[str, np.ndarray] | None = None
    turn_base: bool = True
    tolerance: float = 0.03


class StretchSimpleIKPolicyConfig(BasePolicyConfig):
    """Configuration for `StretchSimpleIKPolicy`."""

    policy_type: str = "planner"
    policy_cls: type | None = None
    policy_factory: PolicyFactory | None = None

    use_authored_grasps: bool = True
    """
    Whether to grasp at a pose from the asset's own grasp library.

    On by default, because a hand-written tilt is wrong for most objects. A
    horizontal grasp closes across whatever happens to lie on the line from the
    base to the object's origin, which for a mug is its body rather than its
    handle, for a pan is the pan rather than the grip, and for a bowl is the full
    0.27m rim-to-rim -- wider than the gripper opens. MolmoSpaces authored a
    grasp library per asset for exactly this reason and the task samplers already
    require one (`filter_for_grasps`), so the graspable poses are known.

    Measured over 60 sampled `droid` assets placed at a typical counter standoff,
    95% have at least one authored grasp Stretch can reach with the orientation
    it was authored at. `grasp_style` is what the other 5% fall back to, so
    turning this off is what makes an ablation of that fallback.
    """

    grasp_style: str = "horizontal"
    """
    'horizontal' or 'top_down' -- the tilt to use when no authored grasp works.

    Horizontal is the default because it is what Stretch's kinematics support:
    the tool sits ahead of the wrist yaw axis, so yaw gives lateral authority and
    the arm can reach without driving the base. Reaching straight down loses that
    authority entirely -- see `policies/kinematics.py`.
    """

    pregrasp_standoff_m: float = 0.24
    """How far back along the approach axis to pause before closing in."""

    grasp_depth_m: float = 0.01
    """
    How far *past* the object centre to drive the tool. A small positive value
    seats the object between the fingers rather than at their tips.
    """

    lift_height_m: float = 0.18
    """How far to raise the object after grasping, in metres."""

    place_hover_m: float = 0.12
    """Height above the place target to release from."""

    articulation_travel_m: float = 0.45
    """
    Arc length to drag an articulated handle through. Longer than the ~20-30%
    of joint range the open/close tasks score at, so a partially-slipped grasp
    can still succeed.
    """

    articulation_waypoints: int = 6
    """How finely to sample the articulation arc. More waypoints hold the grasp
    better but spend more of the episode's step budget."""

    grasp_close_step_rad: float = 0.03
    """
    How much further to ask the fingers to close on each policy step.

    The fingers are position-controlled at kp=500 with a 30Nm clamp, so a target
    far from where they actually are is a large force. Commanding *fully closed*
    on a rigid 5cm object is a 0.13rad error, saturating the actuator and
    squirting the object out rather than holding it -- an episode was observed
    running its entire plan inside tolerance and still finishing with the object
    12cm below where it started.

    Closing a step at a time keeps the position error, and so the force, bounded
    by `kp * grasp_close_step_rad` -- about 15Nm here -- whatever the object turns
    out to be. At 0.03rad a step the fingers travel their full range in about
    seventeen steps, well inside a waypoint's budget.
    """

    grasp_hold_preload_rad: float = 0.02
    """
    How far past first contact to hold the fingers, once they have found the object.

    This is the grip itself: `kp * grasp_hold_preload_rad`, about 10Nm. Measuring
    the object and aiming for its width was tried first and is not reliable --
    estimated from collision geometry, a bowl came back wider than the gripper
    opens and a potato came back as nothing at all. Contact is the ground truth a
    real gripper uses, and it needs no estimate.
    """

    unstow_clearance_m: float = 2.25
    """
    How far above the grasp to swing the arm out of its stowed pose.

    Stretch spawns stowed, wrist yawed right round to 3.14, and that is
    deliberate: an unstowed Stretch has its tool 0.57m in front of the base,
    which is inside the standoff it gets placed at, so it would spawn embedded in
    whatever the object is sitting on (see `stretch_home_init_qpos`).

    The consequence is that every episode begins by swinging the gripper through
    a 3.14rad arc, and at the height it spawns at that arc goes through the
    furniture. Observed directly: the wrist unfolds from 3.14 to 1.27, is pushed
    back to 1.52 and jams there while the solver keeps asking for -0.16, the arm
    stays at 0.08 against a commanded 0.33, and the base drifts because the whole
    robot is being shoved. The tool never gets within half a metre of the object.

    Raising the lift first puts that arc in free air over the surface instead.
    """

    grasp_library_candidates: int = 80
    """
    How many authored grasps to try before giving up and using `grasp_style`.

    A library runs to ~2000 poses once flipped, and each candidate costs one ~4ms
    solve, so the list has to be thinned. Eighty is where the return stops:
    measured over 60 sampled assets, 40 candidates cover 90% of them, 80 cover
    95%, and 160 still cover 95%.

    Sweeping all eighty is the cost of a *failure*, about 0.3s once per episode.
    A success is far cheaper -- ranked as below, the median object is grasped at
    the first candidate tried.
    """

    # How to rank authored grasps, since the first reachable one is the one used.
    #
    # The library is not ordered by anything useful. Taking an even stride through
    # it unranked lands on grasps 0.10m out from the object's origin at a mean
    # |approach . z| of 0.41 -- half-diagonal lunges at the edge of the object,
    # found at a median of three candidates tried. Ranking first picks grasps
    # 0.065m out at 0.06, near-level and nearer the middle, at the same 95%
    # coverage and a median of one candidate tried.
    #
    # The first three terms are MolmoSpaces' own, from
    # `utils/grasp_sample.select_grasp_pose`, at its weights where they carry over.

    grasp_origin_cost_weight: float = 8.0
    """Prefer grasps near the object's origin. Dominant in MolmoSpaces' cost too,
    and for the same reason: a grasp near the centre of mass is one the object
    cannot twist its own weight out of."""

    grasp_reach_cost_weight: float = 1.0
    """Prefer grasps near where the tool already is -- a proxy for reachability
    that costs no solve to evaluate."""

    grasp_horizontal_cost_weight: float = 2.0
    """Prefer a level approach, scored as |approach . world z|. This term is
    Stretch's rather than MolmoSpaces': their Franka prefers to come down from
    above, whereas Stretch's lateral authority comes from the wrist yaw and is
    lost looking straight down (`policies/kinematics.py`)."""

    grasp_alignment_cost_weight: float = 1.0
    """
    Prefer grasps that close roughly along the line from the base to the object,
    scored as the angle between the two.

    Also Stretch's own term. Pinning an authored heading means the base has to
    turn to face it, and every degree of that is the base pushing itself around a
    scene it shares with furniture -- the failure documented at length in
    `unstow_clearance_m`. Measured over 60 assets it takes the mean base turn from
    17 to 14 degrees and the 90th percentile from 29 to 25, for the same 95%
    coverage, and it drops the median search from three candidates to one because
    the grasp Stretch is already pointing at is usually the one it can reach.
    """

    grip_measurement_reach_m: float = 0.045
    """
    How far from the grasp point object geometry still counts towards its width.

    Roughly the depth of the fingers, because that is the material they close
    around. Measuring the whole object instead reports the widest part of it,
    which for anything with a base or a handle is not the part being gripped --
    a boiler came back as 0.31m, well over the gripper's 0.19m opening.
    """

    grasp_slip_tolerance_m: float = 0.05
    """
    How far the object may drift out of the gripper before the pick is called failed.

    Measured as the change in the tool-to-object offset between closing the
    fingers and finishing the lift. An object that is really held keeps that
    offset fixed -- it rides with the tool -- so drift is slip, and 5cm of it
    means the fingers came away with nothing.

    This exists because reaching the lift waypoint proves nothing about the
    grasp. An episode was observed executing every waypoint inside tolerance and
    still ending with the object 12cm *below* where it started: the arm did
    exactly what it was told and the object was never attached to it. Without
    this check that episode spends the rest of the horizon miming a carry.
    """

    reach_tolerance_m: float = 0.025
    gripper_settle_steps: int = 8
    max_steps_per_waypoint: int = 120
    """
    Give up on a waypoint after this many policy steps and move on, rather than
    burning the whole horizon on one unreachable pose.

    120 steps is 8s at the 66ms policy tick. The previous 45 was 3s, which is not
    long enough for the arm to physically cross the workspace: episodes were
    observed timing out 13cm short of a reachable grasp and then closing the
    fingers on empty air. There is room for this -- a four-waypoint pick plan now
    spends at most 480 of the task's 500 steps, and a plan that is going to
    converge does so in a few dozen.
    """

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.policy_cls is None:
            self.policy_cls = StretchSimpleIKPolicy
            self.policy_factory = make_lenient(StretchSimpleIKPolicy)
        assert self.grasp_style in (
            "horizontal",
            "top_down",
        ), f"grasp_style must be 'horizontal' or 'top_down', got {self.grasp_style!r}"


TCP_TO_FINGERTIP_EDGE_M: float = 0.035
PITCH_RAMP_STEP_RAD: float = 0.03
ARM_RAMP_STEP_M: float = 0.005


class StretchSimpleIKPolicy(BasePolicy):
    """Executes a per-episode waypoint plan built from privileged scene state."""

    def __init__(self, config: "MlSpacesExpConfig", task: "BaseMujocoTask" = None) -> None:
        super().__init__(config, task)
        self._solver = StretchReachSolver(config.robot_config)
        self._plan: list[Waypoint] | None = None
        self._waypoint_index = 0
        self._steps_in_waypoint = 0
        self._settled_steps = 0
        self._grasp_offset: np.ndarray | None = None
        self._grasp_lost = False
        self._grip_hold: float | None = None
        self._grasp: ToolGrasp | None = None

    # =========================================================================
    # BasePolicy
    # =========================================================================

    def reset(self) -> None:
        self._plan = None
        self._waypoint_index = 0
        self._steps_in_waypoint = 0
        self._settled_steps = 0
        self._grasp_offset = None
        self._grasp_lost = False
        self._grip_hold = None
        self._grasp = None

    def get_info(self) -> dict:
        return {
            "policy": "stretch_simple_ik",
            "grasp_style": self.config.policy_config.grasp_style,
            # Which of the two the episode actually grasped with, so a benchmark
            # run can be split by it. The fallback rate is the number to watch:
            # a run where most episodes come back "styled" is one where the grasp
            # libraries are not being found, not one where Stretch cannot reach.
            "grasp_source": (
                None if self._grasp is None else ("authored" if self._grasp.authored else "styled")
            ),
            "waypoints_total": 0 if self._plan is None else len(self._plan),
            "waypoints_reached": self._waypoint_index,
            "grasp_lost": self._grasp_lost,
        }

    def get_action(self, observation) -> dict[str, Any]:
        del observation  # this policy reads privileged state, not sensors

        if self._plan is None:
            self._plan = self._build_plan()
            log.info(
                f"[stretch-simple-ik] planned {len(self._plan)} waypoints: "
                f"{[waypoint.label for waypoint in self._plan]}"
            )

        robot_view = self.task.env.current_robot.robot_view
        if self._waypoint_index >= len(self._plan):
            # Hold still, but keep the grip the plan finished with. The generic
            # noop re-derives open-versus-closed from how far apart the fingers
            # currently are (`GripperGroup.noop_ctrl` -> `is_open`), and anything
            # held wider than the midpoint of their range reads as "open" -- so a
            # bowl or a book gets released the moment the plan runs out. Success
            # is judged at the last step of the episode, hundreds of noop steps
            # later, which is exactly when that matters.
            action = robot_view.get_noop_ctrl_dict()
            if self._plan:
                finger_target = self._finger_target(self._plan[-1])
                action["gripper"] = np.array([finger_target, finger_target])
            return action

        waypoint = self._plan[self._waypoint_index]
        current = robot_view.get_qpos_dict()
        action = self._command_for(waypoint, current)
        self._advance(waypoint, robot_view)

        if self._grasp_lost:
            # The lift finished without the object. Nothing later in the plan can
            # recover that -- a place motion would mime carrying something that
            # is back on the table -- so end the episode instead of spending the
            # rest of the horizon on it. `BaseMujocoTask._apply_action` turns this
            # into a terminal step, and the task's own success criterion then
            # scores the episode a failure on its merits.
            return {**robot_view.get_noop_ctrl_dict(), "done": True}

        return action

    # =========================================================================
    # Execution
    # =========================================================================

    def _command_for(
        self, waypoint: Waypoint, current: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Absolute joint targets that pursue `waypoint` from `current`."""
        policy_config = self.config.policy_config
        finger_target = self._finger_target(waypoint)
        action: dict[str, np.ndarray] = {
            "gripper": np.array([finger_target, finger_target]),
        }

        if waypoint.joint_targets is not None:
            for group, value in waypoint.joint_targets.items():
                action[group] = np.asarray(value, dtype=float).copy()
            for group in ("base", "lift", "arm", "wrist"):
                action.setdefault(group, current[group].copy())
            return action

        base_pose = self.task.env.current_robot.robot_view.base.pose
        log.info(f"Executing {waypoint.label=}, {base_pose=}, {waypoint=}")
        solution = self._solver.solve(
            base_pose,
            waypoint.position,
            wrist_pitch=waypoint.wrist_pitch,
            wrist_roll=waypoint.wrist_roll,
            approach_yaw=waypoint.approach_yaw,
            # A waypoint that names a heading means it: an authored grasp's yaw is
            # which way the fingers close, and letting the solver rotate it to
            # something easier to reach would be closing across a different part
            # of the object than the one the grasp was authored for.
            yaw_spread=0.0 if waypoint.approach_yaw is not None else APPROACH_YAW_SPREAD_RAD,
            seed=current,
            tolerance=policy_config.reach_tolerance_m * 0.5,
        )
        if solution is None:
            # Out of reach. If executing the lift waypoint, raise the lift joint directly.
            if waypoint.label == "lift":
                lift_max = float(self._solver.joint_limits["lift"][0, 1])
                wrist_pitch_max = float(self._solver.joint_limits["wrist"][1, 1])
                arm_min = float(self._solver.joint_limits["arm"][0, 0])

                lift_target = min(float(current["lift"][0]) + policy_config.lift_height_m, lift_max)
                action["lift"] = np.array([lift_target])

                # When the object is tall and mast is near ceiling, lift as much as possible,
                # pitch wrist up to lift the object, and retract arm slightly to clear obstacles.
                lift_headroom = lift_max - float(current["lift"][0])
                wrist_action = current["wrist"].copy()
                if lift_headroom < policy_config.lift_height_m * 0.9:
                    target_pitch = min(float(current["wrist"][1]) + 0.5, wrist_pitch_max)
                    target_arm = max(float(current["arm"][0]) - 0.08, arm_min)
                    wrist_action[1] = min(float(current["wrist"][1]) + PITCH_RAMP_STEP_RAD, target_pitch)
                    action["arm"] = np.array([max(float(current["arm"][0]) - ARM_RAMP_STEP_M, target_arm)])
                else:
                    action["arm"] = current["arm"].copy()

                action["wrist"] = wrist_action
                action["base"] = current["base"].copy()
                return action

            # Hold the arm where it is rather than commanding a
            # half-converged pose, and let the step budget move the plan on.
            for group in ("base", "lift", "arm", "wrist"):
                action[group] = current[group].copy()
            return action

        action["lift"] = solution["lift"]
        action["arm"] = solution["arm"]
        action["wrist"] = solution["wrist"]
        action["base"] = solution["base"] if waypoint.turn_base else current["base"].copy()

        # If lift headroom is constrained for tall objects, pitch wrist up and retract arm smoothly
        if waypoint.label == "lift":
            lift_max = float(self._solver.joint_limits["lift"][0, 1])
            wrist_pitch_max = float(self._solver.joint_limits["wrist"][1, 1])
            arm_min = float(self._solver.joint_limits["arm"][0, 0])
            lift_headroom = lift_max - float(current["lift"][0])
            if lift_headroom < policy_config.lift_height_m * 0.9:
                target_pitch = min(float(action["wrist"][1]) + 0.5, wrist_pitch_max)
                target_arm = max(float(action["arm"][0]) - 0.08, arm_min)
                action["wrist"][1] = min(float(current["wrist"][1]) + PITCH_RAMP_STEP_RAD, target_pitch)
                action["arm"] = np.array([max(float(current["arm"][0]) - ARM_RAMP_STEP_M, target_arm)])
        return action

    def _finger_target(self, waypoint: Waypoint) -> float:
        """Finger joint position for this waypoint's grip.

        Opening is just the end of the range. Closing is a controller: the
        fingers are asked to move one `grasp_close_step_rad` further shut each
        step until they touch the object, and then held a fixed
        `grasp_hold_preload_rad` past where contact was made. Because the
        commanded target is never far from where the fingers actually are, the
        position error -- and so the force, at kp=500 -- stays bounded the whole
        way in, instead of saturating the actuator the instant something rigid
        gets in the way.
        """
        policy_config = self.config.policy_config
        gripper = self._gripper_group()
        if waypoint.gripper_open:
            self._grip_hold = None
            return gripper.OPEN_JOINT_POS

        closed = gripper.CLOSED_JOINT_POS
        opened = gripper.OPEN_JOINT_POS
        current = float(np.mean(np.asarray(gripper.joint_pos, dtype=float)))
        inward = -1.0 if closed < opened else 1.0

        left_touch, right_touch = self._fingers_touching_pickup()
        both_touch = left_touch and right_touch
        either_touch = left_touch or right_touch

        if self._grip_hold is None and (both_touch or (either_touch and abs(current - closed) < 0.15)):
            self._grip_hold = current + inward * policy_config.grasp_hold_preload_rad
        if self._grip_hold is not None:
            return float(np.clip(self._grip_hold, min(closed, opened), max(closed, opened)))

        stepped = current + inward * policy_config.grasp_close_step_rad
        return float(np.clip(stepped, min(closed, opened), max(closed, opened)))

    def _fingers_touching_pickup(self) -> tuple[bool, bool]:
        """Whether (left_touching, right_touching) the object being picked up."""
        object_name = self.config.task_config.pickup_obj_name
        if not object_name:
            return False, False
        environment = self.task.env
        object_manager = environment.object_managers[environment.current_batch_index]
        scene_object = object_manager.get_object_by_name(object_name)
        if scene_object is None:
            return False, False

        model, data = environment.current_model, environment.current_data
        gripper_root = self._gripper_group().root_body_id
        left_touch = False
        right_touch = False

        for index in range(data.ncon):
            contact = data.contact[index]
            if contact.dist > 0:
                continue
            roots = (
                model.body_rootid[model.geom_bodyid[contact.geom1]],
                model.body_rootid[model.geom_bodyid[contact.geom2]],
            )
            if scene_object.body_id not in roots:
                continue
            other = roots[0] if roots[1] == scene_object.body_id else roots[1]
            if not model.body(other).name.startswith(self.config.robot_config.robot_namespace):
                continue
            robot_geom = contact.geom1 if roots[0] != scene_object.body_id else contact.geom2
            body = model.geom_bodyid[robot_geom]
            while body != 0:
                body_name = model.body(body).name
                if "left" in body_name:
                    left_touch = True
                elif "right" in body_name:
                    right_touch = True
                if body == gripper_root:
                    break
                body = model.body_parentid[body]
        return left_touch, right_touch

    def _advance(self, waypoint: Waypoint, robot_view) -> None:
        """Move to the next waypoint once this one is reached, settled or timed out."""
        self._steps_in_waypoint += 1
        tool_position = robot_view.get_move_group("gripper").leaf_frame_to_world[:3, 3]
        if waypoint.joint_targets is None:
            reached = float(np.linalg.norm(tool_position - waypoint.position)) <= waypoint.tolerance
        else:
            # A joint-space waypoint is about the configuration, so arrival is
            # measured there; the tool is wherever that configuration puts it.
            reached = all(
                float(
                    np.max(
                        np.abs(
                            np.ravel(robot_view.get_move_group(group).joint_pos)
                            - np.ravel(target)
                        )
                    )
                )
                <= _JOINT_ARRIVAL_TOLERANCE_RAD
                for group, target in waypoint.joint_targets.items()
            )

        if waypoint.establishes_grasp:
            # A grasp-closing waypoint is only reached once both fingers have made
            # contact with the object or reached their closed limit, and preload is held.
            left_touch, right_touch = self._fingers_touching_pickup()
            gripper = self._gripper_group()
            closed_pos = gripper.CLOSED_JOINT_POS
            current_grip = float(np.mean(np.asarray(gripper.joint_pos, dtype=float)))
            fully_closed = abs(current_grip - closed_pos) < 0.02
            grip_established = (left_touch and right_touch) or fully_closed or (
                self._grip_hold is not None and abs(current_grip - self._grip_hold) < 0.03
            )
            reached = reached and grip_established

        if reached:
            self._settled_steps += 1
        timed_out = self._steps_in_waypoint >= self.config.policy_config.max_steps_per_waypoint

        if (reached and self._settled_steps > waypoint.settle_steps) or timed_out:
            if timed_out and not reached:
                log.info(
                    f"[stretch-simple-ik] waypoint '{waypoint.label}' timed out "
                    f"{float(np.linalg.norm(tool_position - waypoint.position)):.3f}m short"
                )
            if waypoint.establishes_grasp:
                self._grasp_offset = self._tool_to_object_offset(tool_position)
            elif waypoint.verify_grasp and not self._grasp_still_held(tool_position):
                self._grasp_lost = True

            self._waypoint_index += 1
            self._steps_in_waypoint = 0
            self._settled_steps = 0

    def _tool_to_object_offset(self, tool_position: np.ndarray) -> np.ndarray | None:
        """Where the pickup object sits relative to the tool, right now."""
        object_position = self._object_grasp_point(self.config.task_config.pickup_obj_name)
        if object_position is None:
            return None
        return np.asarray(object_position, dtype=float) - np.asarray(tool_position, dtype=float)

    def _grasp_still_held(self, tool_position: np.ndarray) -> bool:
        """Whether the object has travelled with the tool since the fingers closed.

        Reaching the lift waypoint says only that the *arm* got there. The object
        is held if and only if it kept its place relative to the tool, so that
        offset is what gets compared -- not the object's height, which also rises
        when the fingers merely shove it up a slope, and not contact, which a
        brush registers as firmly as a grip.

        Unknowable cases count as held: a task with no pickup object, or one
        where the grasp offset was never captured, has nothing to fail here and
        should be left to the task's own success criterion.
        """
        if self._grasp_offset is None:
            return True
        offset = self._tool_to_object_offset(tool_position)
        if offset is None:
            return True

        drift = float(np.linalg.norm(offset - self._grasp_offset))
        if drift > self.config.policy_config.grasp_slip_tolerance_m:
            log.info(
                f"[stretch-simple-ik] grasp lost: object drifted {drift:.3f}m out of the "
                f"gripper during the lift (tolerance "
                f"{self.config.policy_config.grasp_slip_tolerance_m:.3f}m)"
            )
            return False
        return True

    # =========================================================================
    # Planning
    # =========================================================================

    def _build_plan(self) -> list[Waypoint]:
        task_cls = self.config.task_config.task_cls
        task_cls_name = (
            task_cls if isinstance(task_cls, str) else f"{task_cls.__module__}.{task_cls.__name__}"
        )

        if task_cls_name in PICK_TASK_CLASSES:
            return self._plan_pick_and_place(with_place=task_cls_name in PLACE_TASK_CLASSES)
        if task_cls_name in (OPENING_TASK_CLASS, DOOR_OPENING_TASK_CLASS):
            return self._plan_articulation(task_cls_name)

        log.warning(
            f"[stretch-simple-ik] no plan builder for task class {task_cls_name!r}; holding still. "
            "Navigation benchmarks should use the A* planner policy instead, see configs.py."
        )
        return []

    def _plan_pick_and_place(self, with_place: bool) -> list[Waypoint]:
        grasp = self._pickup_grasp()
        if grasp is None:
            return []
        self._grasp = grasp

        policy_config = self.config.policy_config
        grasp_point = grasp.position
        approach = grasp.approach

        # Every waypoint in a pick holds the grasp orientation: the phases before
        # the close are lining up for it and the phases after are carrying
        # something in it, so re-tilting anywhere in between would either miss
        # the grasp or wring the object out of the fingers.
        orientation = {
            "wrist_pitch": grasp.wrist_pitch,
            "wrist_roll": grasp.wrist_roll,
            "approach_yaw": grasp.solver_yaw,
        }

        grip_width = self._object_grasp_width(
            self.config.task_config.pickup_obj_name, grasp.closing_axis, grasp_point
        )
        log.info(
            f"[stretch-simple-ik] grasp width along the closing axis: "
            f"{'unknown' if grip_width is None else f'{grip_width:.3f}m'}"
        )

        waypoints = self._unstow_waypoints(grasp)
        pregrasp = self._reachable_standoff(grasp)
        if pregrasp is not None:
            waypoints.append(
                Waypoint(
                    position=pregrasp,
                    gripper_open=True,
                    label="pregrasp",
                    tolerance=policy_config.reach_tolerance_m,
                    **orientation,
                )
            )

        reach_position = grasp_point + approach * policy_config.grasp_depth_m
        lift_height = self._reachable_lift_height(grasp, grasp_point)

        waypoints += [
            Waypoint(
                position=reach_position,
                gripper_open=True,
                label="reach",
                tolerance=policy_config.reach_tolerance_m,
                **orientation,
            ),
            Waypoint(
                position=reach_position,
                gripper_open=False,
                label="close",
                settle_steps=max(policy_config.gripper_settle_steps, 15),
                establishes_grasp=True,
                grip_width_m=grip_width,
                tolerance=policy_config.reach_tolerance_m * 2.0,
                **orientation,
            ),
            Waypoint(
                position=grasp_point + np.array([0.0, 0.0, lift_height]),
                gripper_open=False,
                label="lift",
                settle_steps=policy_config.gripper_settle_steps,
                verify_grasp=True,
                grip_width_m=grip_width,
                tolerance=policy_config.reach_tolerance_m,
                **orientation,
            ),
        ]
        if not with_place:
            return waypoints

        place_point = self._object_grasp_point(self.config.task_config.place_receptacle_name)
        if place_point is None:
            return waypoints

        hover = place_point + np.array([0.0, 0.0, policy_config.place_hover_m])
        waypoints += [
            Waypoint(
                position=hover,
                gripper_open=False,
                label="transfer",
                grip_width_m=grip_width,
                # Carrying an object across the workspace is the one manipulation
                # motion whose span routinely exceeds the arm's, so allow the
                # base to help here even though the grasp phases may not.
                tolerance=policy_config.reach_tolerance_m * 1.5,
                **orientation,
            ),
            Waypoint(
                position=hover,
                gripper_open=True,
                label="release",
                settle_steps=policy_config.gripper_settle_steps,
                tolerance=policy_config.reach_tolerance_m * 1.5,
                **orientation,
            ),
            Waypoint(
                position=hover + np.array([0.0, 0.0, policy_config.place_hover_m]),
                gripper_open=True,
                label="retreat",
                tolerance=policy_config.reach_tolerance_m * 2.0,
                **orientation,
            ),
        ]
        return waypoints

    def _plan_articulation(self, task_cls_name: str) -> list[Waypoint]:
        """Grasp a handle, then drag it along the joint's own motion.

        The arc is generated from the articulation's MuJoCo joint rather than
        from any assumption about drawers versus doors: a slide joint gives a
        straight line along the joint axis, a hinge gives a circular arc about
        the joint anchor. Both then reduce to the same list of tool waypoints.
        """
        handle_arc = self._articulation_arc(task_cls_name)
        if handle_arc is None:
            return []

        policy_config = self.config.policy_config
        # Handles are grasped from the front, never from above -- a top-down
        # approach on a drawer pull collides with the cabinet face. This is the
        # one place a hand-written style is still the right answer rather than a
        # fallback: MolmoSpaces does author per-joint grasps for its articulated
        # assets (`utils/grasps.get_joint_grasps`), but a handle has a single
        # obvious approach and the joint's own axis already says where to drag it.
        pitch = PITCH_HORIZONTAL
        start = handle_arc[0]
        approach = self._styled_grasp(start, pitch).approach

        plan = [
            Waypoint(
                position=start - approach * policy_config.pregrasp_standoff_m,
                wrist_pitch=pitch,
                gripper_open=True,
                label="pre_handle",
                tolerance=policy_config.reach_tolerance_m,
            ),
            Waypoint(
                position=start,
                wrist_pitch=pitch,
                gripper_open=True,
                label="at_handle",
                tolerance=policy_config.reach_tolerance_m,
            ),
            Waypoint(
                position=start,
                wrist_pitch=pitch,
                gripper_open=False,
                label="grip_handle",
                settle_steps=policy_config.gripper_settle_steps,
                tolerance=policy_config.reach_tolerance_m * 2.0,
            ),
        ]
        for index, point in enumerate(handle_arc[1:], start=1):
            plan.append(
                Waypoint(
                    position=point,
                    wrist_pitch=pitch,
                    gripper_open=False,
                    label=f"actuate_{index}",
                    # A drawer or door is dragged mostly by driving the base;
                    # the arm's 0.52m of travel cannot cover a full swing.
                    tolerance=policy_config.reach_tolerance_m * 2.0,
                )
            )
        return plan

    # =========================================================================
    # Scene queries
    # =========================================================================

    def _gripper_group(self) -> StretchGripperGroup:
        return self.task.env.current_robot.robot_view.get_gripper("gripper")

    def _styled_grasp(self, position: np.ndarray, pitch: float) -> ToolGrasp:
        """A grasp at `position` built from a hand-written tilt.

        The nominal heading is the bearing from the base to the target, which is
        the direction a grasp closes along when the robot is facing its work, but
        it is only nominal -- `authored` is False, so the solver keeps its freedom
        to rotate the approach and the heading here is used for the geometry
        (which way to back off, which way the fingers span) rather than as a
        constraint.

        At `PITCH_TOP_DOWN` that geometry comes out as an approach straight down
        and a closing axis across the bearing, which is what a top-down grasp is.
        """
        position = np.asarray(position, dtype=float)
        base_xy = self.task.env.current_robot.robot_view.base.pose[:2, 3]
        direction = position[:2] - base_xy
        bearing = (
            0.0
            if float(np.linalg.norm(direction)) < 1e-6
            else float(np.arctan2(direction[1], direction[0]))
        )
        return ToolGrasp(
            position=position,
            approach_yaw=bearing,
            wrist_pitch=pitch,
            wrist_roll=0.0,
            authored=False,
        )

    def _reachable_standoff(self, grasp: ToolGrasp) -> np.ndarray | None:
        """The furthest-back pregrasp pose the arm can actually hold, or None.

        Backing off along a *horizontal* approach means retracting the
        telescoping arm, and the arm does not retract past 0.467m from the base
        -- so on a target that is already near Stretch's minimum reach, the
        nominal pregrasp standoff lands inside the robot. Left unchecked that
        costs a whole waypoint's step budget failing to solve, which is what an
        early run of this policy spent its first 45 steps doing. Shrinking the
        standoff until it solves keeps the approach when there is room for one
        and skips it when there is not.
        """
        standoff = self.config.policy_config.pregrasp_standoff_m
        base_pose = self.task.env.current_robot.robot_view.base.pose
        approach = grasp.approach
        for fraction in (1.0, 0.66, 0.33):
            candidate = grasp.position - approach * (standoff * fraction)
            if self._solve_at(base_pose, candidate, grasp) is not None:
                return candidate
        log.info("[stretch-simple-ik] no reachable pregrasp standoff; approaching directly")
        return None

    def _reachable_lift_height(self, grasp: ToolGrasp, grasp_point: np.ndarray) -> float:
        """The highest vertical lift from grasp_point that Stretch can physically solve."""
        nominal = self.config.policy_config.lift_height_m
        base_pose = self.task.env.current_robot.robot_view.base.pose
        for fraction in (1.0, 0.8, 0.6, 0.4, 0.2, 0.1):
            h = nominal * fraction
            candidate = grasp_point + np.array([0.0, 0.0, h])
            if self._solve_at(base_pose, candidate, grasp) is not None:
                return h
        return nominal * 0.1

    def _solve_at(
        self, base_pose: np.ndarray, position: np.ndarray, grasp: ToolGrasp
    ) -> dict[str, np.ndarray] | None:
        """Whether the tool can be put at `position` in `grasp`'s orientation.

        Planning-time counterpart to `_command_for`, and it has to ask for the
        same thing: an authored grasp's heading is pinned in both, so a candidate
        that passes here is one the executor can also hold.
        """
        return self._solver.solve(
            base_pose,
            position,
            wrist_pitch=grasp.wrist_pitch,
            wrist_roll=grasp.wrist_roll,
            approach_yaw=grasp.solver_yaw,
            yaw_spread=0.0 if grasp.authored else APPROACH_YAW_SPREAD_RAD,
        )

    def _object_height(self, scene_object) -> float:
        """Vertical bounding height of the object in world coordinates."""
        environment = self.task.env
        model, data = environment.current_model, environment.current_data
        body_id = scene_object.body_id
        z_min = float("inf")
        z_max = float("-inf")
        found = False
        for gid in range(model.ngeom):
            bid = model.geom_bodyid[gid]
            if bid == body_id or model.body_rootid[bid] == body_id:
                found = True
                pos_z = float(data.geom_xpos[gid][2])
                size = model.geom(gid).size
                extent_z = float(size[2] if len(size) > 2 else size[0])
                z_min = min(z_min, pos_z - extent_z)
                z_max = max(z_max, pos_z + extent_z)
        if found and z_max > z_min:
            return z_max - z_min

        try:
            aabb = scene_object.aabb_size
            if aabb is not None and len(aabb) >= 3:
                return float(aabb[2])
        except Exception:
            pass
        return 0.05

    def _object_grasp_point(self, object_name: str | None) -> np.ndarray | None:
        """A world point on `object_name` worth aiming the tool at.

        Uses the object's current body origin rather than its axis-aligned
        bounding-box centre: `MlSpacesObject.aabb_center` comes from the compiled
        model and so describes the object's *initial* pose, which for a
        benchmark episode is not where the object was moved to.

        If the object is smaller in height than the distance between the TCP and
        the bottom edge of the fingertips, offsets upward so the object is grasped
        by the tips of the fingertips instead of colliding with the supporting surface.
        """
        if not object_name:
            return None
        environment = self.task.env
        object_manager = environment.object_managers[environment.current_batch_index]
        scene_object = object_manager.get_object_by_name(object_name)
        if scene_object is None:
            log.warning(f"[stretch-simple-ik] object {object_name!r} not found in scene")
            return None
        pos = np.asarray(scene_object.position, dtype=float).copy()

        if object_name == self.config.task_config.pickup_obj_name:
            height = self._object_height(scene_object)
            if height < TCP_TO_FINGERTIP_EDGE_M:
                offset_z = TCP_TO_FINGERTIP_EDGE_M - (height * 0.5)
                pos[2] += offset_z
                log.info(
                    f"[stretch-simple-ik] object {object_name!r} height ({height:.3f}m) < "
                    f"TCP-to-fingertip-edge distance ({TCP_TO_FINGERTIP_EDGE_M:.3f}m); "
                    f"offsetting grasp point by +{offset_z:.3f}m to grasp with fingertip tips"
                )
        return pos

    def _unstow_waypoints(self, grasp: ToolGrasp) -> list[Waypoint]:
        """Get the arm out of its stowed pose, clear of whatever it is reaching over.

        Joint-space on purpose. The point of this move is the configuration --
        arm retracted so the swing radius is as small as the robot gets, wrist
        turned to face forward, lift high enough that the arc passes over the
        surface rather than through it. Asking for a tool position instead would
        let the solver extend the arm during the swing, which is the opposite of
        what is wanted.

        Two waypoints rather than one, because the height and the swing have to
        happen *in that order*. Commanded together they race, and the wrist wins:
        the lift is hauling the whole arm through half a metre at kp=2500 while
        the wrist is a couple of kilos at kp=20, so measured in free space the
        wrist is 80% through its 3.14rad arc while the lift is barely half way --
        tool at z=0.86 and swung out to 0.47m, which is exactly counter height at
        exactly counter distance. The arc that was supposed to pass over the
        surface goes through it, and the gripper jams there: observed on the
        debug config, the fingertips stay in contact with the countertop for all
        500 steps of the episode while the base is shoved 0.26m out of reach, and
        every reach solve from then on fails. Raising first and swinging second
        costs about ten policy steps and puts the arc in free air.
        """
        policy_config = self.config.policy_config
        solver = self._solver
        pitch, roll = grasp.wrist_pitch, grasp.wrist_roll
        retracted = {
            "base": np.zeros(3),
            "lift": np.array([0.0]),
            "arm": np.array([0.0]),
            "wrist": np.array([0.0, pitch, roll]),
        }
        # Tool height rises one-for-one with the lift, so the offset between them
        # is whatever the tool sits at with the lift at zero.
        tool_height_at_zero_lift = float(solver.forward(retracted)[2, 3])
        base_z = float(self.task.env.current_robot.robot_view.base.pose[2, 3])
        wanted = float(grasp.position[2]) + policy_config.unstow_clearance_m
        lift_limits = solver.joint_limits["lift"][0]
        lift = float(
            np.clip(wanted - base_z - tool_height_at_zero_lift, lift_limits[0], lift_limits[1])
        )

        position = np.asarray(grasp.position, dtype=float)
        return [
            Waypoint(
                position=position,
                wrist_pitch=pitch,
                wrist_roll=roll,
                gripper_open=True,
                label="raise",
                # The wrist is deliberately absent: `_command_for` holds every
                # group a joint-space waypoint does not name at its current
                # value, which keeps the arm stowed on the way up without also
                # making a stowed wrist something this waypoint waits to arrive
                # at. It is already there.
                joint_targets={
                    "lift": np.array([lift]),
                    "arm": np.array([0.0]),
                },
            ),
            Waypoint(
                position=position,
                wrist_pitch=pitch,
                wrist_roll=roll,
                gripper_open=True,
                label="unstow",
                # Pitch and roll come out of the grasp, so the wrist arrives at
                # the tilt the object is going to be taken at and the reach that
                # follows only has to aim. Yaw is zero rather than the grasp's
                # heading because this waypoint's whole purpose is to unwind the
                # stowed 3.14rad yaw in free air; where it then points is the
                # solver's business.
                joint_targets={
                    "lift": np.array([lift]),
                    "arm": np.array([0.0]),
                    "wrist": np.array([0.0, pitch, roll]),
                },
            ),
        ]

    def _pickup_grasp(self) -> ToolGrasp | None:
        """How to grasp the task's pickup object.

        Prefers a pose out of the asset's authored grasp library, and falls back
        to `grasp_style` at the object's origin when there is no library or
        nothing in it is reachable. Returns None only when there is no pickup
        object to grasp at all.
        """
        object_name = self.config.task_config.pickup_obj_name
        origin = self._object_grasp_point(object_name)
        if origin is None:
            return None
        if self.config.policy_config.use_authored_grasps:
            authored = self._authored_grasp(object_name, origin)
            if authored is not None:
                return authored
        return self._styled_grasp(origin, self._fallback_pitch())

    def _fallback_pitch(self) -> float:
        return (
            PITCH_TOP_DOWN
            if self.config.policy_config.grasp_style == "top_down"
            else PITCH_HORIZONTAL
        )

    def _authored_grasp(self, object_name: str | None, origin: np.ndarray) -> ToolGrasp | None:
        """The best-ranked authored grasp Stretch can actually take, or None.

        A candidate has to survive three tests:
        1. Orientation clearance: grasps approaching from below the horizontal
           plane (wrist_pitch < -0.05) are filtered out to prevent table collisions.
           Candidates are sorted by descending pitch angle (top-down / overhead
           grasps prioritized first).
        2. Finger span: the object width at the candidate closing axis must fit
           within the gripper's open span.
        3. Reachability: the candidate pose must be solvable by StretchReachSolver.
        """
        poses = self._library_grasp_poses(object_name, origin)
        if poses is None:
            return None

        _, open_width = self._gripper_group().inter_finger_dist_range
        base_pose = self.task.env.current_robot.robot_view.base.pose

        candidates: list[tuple[float, ToolGrasp]] = []
        for pose in poses:
            approach_yaw, wrist_pitch, wrist_roll = tcp_orientation_from_grasp(pose)
            if wrist_pitch < -0.05:
                # Approaching from below the horizontal plane; skip to avoid table collisions.
                continue
            candidates.append(
                (
                    wrist_pitch,
                    ToolGrasp(
                        position=np.asarray(pose[:3, 3], dtype=float),
                        approach_yaw=approach_yaw,
                        wrist_pitch=wrist_pitch,
                        wrist_roll=wrist_roll,
                        authored=True,
                    ),
                )
            )

        # Sort by descending pitch: top-down (pitch ~ pi/2) first, horizontal last.
        candidates.sort(key=lambda item: item[0], reverse=True)

        unreachable = 0
        too_wide = 0
        for pitch, candidate in candidates:
            width = self._object_grasp_width(
                object_name, candidate.closing_axis, candidate.position
            )
            if width is not None and width >= open_width:
                too_wide += 1
                continue
            if self._solve_at(base_pose, candidate.position, candidate) is None:
                unreachable += 1
                continue
            log.info(
                f"[stretch-simple-ik] grasping at an authored grasp "
                f"{float(np.linalg.norm(candidate.position - origin)):.3f}m from the object "
                f"origin, pitch {candidate.wrist_pitch:+.2f} roll {candidate.wrist_roll:+.2f} "
                f"(rejected {too_wide} too wide, {unreachable} unreachable)"
            )
            return candidate

        log.info(
            f"[stretch-simple-ik] none of {len(poses)} authored grasps was reachable "
            f"and table-clearing; falling back to a {self.config.policy_config.grasp_style} grasp"
        )
        return None

    def _library_grasp_poses(
        self, object_name: str | None, origin: np.ndarray
    ) -> np.ndarray | None:
        """Authored world grasp poses for `object_name`, ranked and thinned.

        Returns None whenever the library cannot be consulted -- no metadata, no
        grasps for this asset -- which is a fallback, not an error, and the caller
        treats it as one.
        """
        if not object_name:
            return None
        environment = self.task.env
        object_manager = environment.object_managers[environment.current_batch_index]
        scene_object = object_manager.get_object_by_name(object_name)
        if scene_object is None:
            return None

        try:
            from molmo_spaces.utils.grasps import get_pickup_grasps

            task_sampler_config = getattr(self.config, "task_sampler_config", None)
            grasp_libraries = getattr(task_sampler_config, "grasp_libraries", None)
            poses = get_pickup_grasps(
                environment,
                scene_object,
                include_flipped=True,
                grasp_libraries=grasp_libraries,
            )
        except Exception as failure:  # noqa: BLE001 - the library is optional here
            log.info(f"[stretch-simple-ik] no grasp library for {object_name!r}: {failure}")
            return None

        if poses is None or len(poses) == 0:
            return None

        poses = np.asarray(poses, dtype=float).copy()
        height = self._object_height(scene_object)
        if height < TCP_TO_FINGERTIP_EDGE_M:
            offset_z = TCP_TO_FINGERTIP_EDGE_M - (height * 0.5)
            poses[:, 2, 3] += offset_z

        policy_config = self.config.policy_config
        tool_position = self.task.env.current_robot.robot_view.get_move_group(
            "gripper"
        ).leaf_frame_to_world[:3, 3]

        # The library's own order means nothing here, so rank before thinning.
        # Column 2 of each pose is the grasp frame's z axis, which is its approach
        # axis (see `kinematics.GRASP_LIBRARY_TO_TCP`), so the last two terms read
        # straight off the matrix without building a rotation per candidate.
        base_xy = self.task.env.current_robot.robot_view.base.pose[:2, 3]
        approach = poses[:, :3, 2]
        approach_heading = np.arctan2(approach[:, 1], approach[:, 0])
        bearing = np.arctan2(poses[:, 1, 3] - base_xy[1], poses[:, 0, 3] - base_xy[0])
        # Wrapped into (-pi, pi] before taking the magnitude, so a heading either
        # side of the +-pi seam is a small misalignment rather than a full turn.
        offset = approach_heading - bearing
        misalignment = np.abs(np.arctan2(np.sin(offset), np.cos(offset)))

        cost = (
            policy_config.grasp_origin_cost_weight
            * np.linalg.norm(poses[:, :3, 3] - np.asarray(origin, dtype=float), axis=1)
            + policy_config.grasp_reach_cost_weight
            * np.linalg.norm(poses[:, :3, 3] - tool_position, axis=1)
            + policy_config.grasp_horizontal_cost_weight * np.abs(approach[:, 2])
            + policy_config.grasp_alignment_cost_weight * misalignment
        )
        poses = poses[np.argsort(cost, kind="stable")]

        limit = policy_config.grasp_library_candidates
        if len(poses) > limit:
            # Even strides through the ranked order rather than its head. The two
            # measure the same on an unobstructed counter -- 95% of assets covered
            # either way, both at a median of one candidate tried -- so this is a
            # hedge rather than a win: a library's best-ranked few dozen poses are
            # minor variants of one grasp, and the ranking is only proxies (no
            # collision checking, and MolmoSpaces' own planner does check). On the
            # cluttered scenes the measurement does not cover, one grasp's worth of
            # candidates is a thin thing to have staked the episode on.
            poses = poses[:: max(1, len(poses) // limit)][:limit]
        return poses

    def _object_grasp_width(
        self, object_name: str | None, closing_axis: np.ndarray, grasp_point: np.ndarray
    ) -> float | None:
        """How wide `object_name` is along the line the fingers close on.

        Measured from the object's collision geometry rather than a bounding
        sphere: the fingers close on one axis, and a mug's width across its
        handle is not its width across its body. Every geom's oriented box is
        projected onto that axis and the span taken, so a multi-geom object is
        covered as a whole.

        Returns None when the object has no geometry to measure, which leaves the
        caller commanding a plain close.
        """
        if not object_name:
            return None
        environment = self.task.env
        object_manager = environment.object_managers[environment.current_batch_index]
        scene_object = object_manager.get_object_by_name(object_name)
        if scene_object is None:
            return None

        from molmo_spaces.utils.mj_model_and_data_utils import descendant_geoms

        model = environment.current_model
        data = environment.current_data
        geoms = descendant_geoms(model, scene_object.body_id)
        if geoms is None or len(geoms) == 0:
            return None

        axis = np.asarray(closing_axis, dtype=float)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            return None
        axis = axis / norm
        grasp_point = np.asarray(grasp_point, dtype=float)

        # Corners of each geom's local axis-aligned box, pushed out to world.
        signs = np.array(
            [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float
        )
        reach = self.config.policy_config.grip_measurement_reach_m
        low = np.inf
        high = -np.inf
        for geom_id in geoms:
            centre, half_extent = model.geom_aabb[geom_id][:3], model.geom_aabb[geom_id][3:]
            corners = centre + signs * half_extent
            world = data.geom_xpos[geom_id] + corners @ np.asarray(
                data.geom_xmat[geom_id]
            ).reshape(3, 3).T

            # Only material the fingers will actually close around counts. The
            # whole-object span is the wrong number: a boiler measured 0.31m
            # across, three times what the gripper opens to, because its base is
            # wide -- while the part being grasped was never that thick. Distance
            # is taken perpendicular to the closing axis, since along that axis
            # is exactly the direction being measured.
            offset = world - grasp_point
            perpendicular = offset - np.outer(offset @ axis, axis)
            near = np.linalg.norm(perpendicular, axis=1) <= reach
            if not near.any():
                continue

            projected = world[near] @ axis
            low = min(low, float(projected.min()))
            high = max(high, float(projected.max()))

        if not np.isfinite(low) or not np.isfinite(high):
            return None
        return high - low

    def _articulation_arc(self, task_cls_name: str) -> list[np.ndarray] | None:
        """Sample the path the handle sweeps as its joint opens (or closes)."""
        articulation, joint_index, handle_position = self._articulation_and_handle(task_cls_name)
        if articulation is None:
            return None

        policy_config = self.config.policy_config
        closing = self.config.task_type == "close"

        import mujoco

        joint_type = articulation.get_joint_type(joint_index)
        axis = np.asarray(articulation.get_joint_axis(joint_index), dtype=float)
        joint_range = articulation.get_joint_range(joint_index)
        current = articulation.get_joint_position(joint_index)

        # Open moves towards whichever end of the range is further from zero;
        # close moves back to zero. The tasks define "closed" as joint value 0
        # for both [0, r] and [-r, 0] ranges (see `OpeningTask.get_reward`).
        open_limit = joint_range[1] if abs(joint_range[1]) > abs(joint_range[0]) else joint_range[0]
        target_value = 0.0 if closing else open_limit
        travel = target_value - current
        if abs(travel) < 1e-3:
            return None

        samples = max(2, policy_config.articulation_waypoints)
        fractions = np.linspace(0.0, 1.0, samples)

        if joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
            # A drawer: the handle translates along the joint axis.
            direction = axis / max(float(np.linalg.norm(axis)), 1e-9)
            distance = float(np.clip(abs(travel), 0.0, policy_config.articulation_travel_m))
            distance *= np.sign(travel)
            return [handle_position + direction * (distance * f) for f in fractions]

        # A cabinet door or room door: the handle sweeps an arc about the anchor.
        anchor = np.asarray(articulation.get_joint_anchor_position(joint_index), dtype=float)
        radius_vector = handle_position - anchor
        unit_axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
        arc = []
        for fraction in fractions:
            angle = travel * fraction
            rotated = (
                radius_vector * np.cos(angle)
                + np.cross(unit_axis, radius_vector) * np.sin(angle)
                + unit_axis * float(np.dot(unit_axis, radius_vector)) * (1.0 - np.cos(angle))
            )
            arc.append(anchor + rotated)
        return arc

    def _articulation_and_handle(
        self, task_cls_name: str
    ) -> tuple[Any | None, int, np.ndarray | None]:
        """The articulation object, the joint to drive, and where its handle is."""
        environment = self.task.env
        batch_index = environment.current_batch_index

        if task_cls_name == DOOR_OPENING_TASK_CLASS:
            door = getattr(self.task, "door_object", None)
            if door is None:
                return None, 0, None
            return (
                door,
                door.get_hinge_joint_index(),
                np.asarray(self.task.get_door_handle_position(), dtype=float),
            )

        articulation_objects = getattr(self.task, "articulation_objects", None)
        if not articulation_objects or not articulation_objects[batch_index]:
            return None, 0, None
        articulation = articulation_objects[batch_index][0]
        joint_index = int(getattr(self.config.task_config, "joint_index", 0) or 0)

        # `OpeningTask`'s articulations are furniture, not doors, so they have no
        # handle geometry helper. The moving leaf body's origin is the closest
        # thing to a handle the model exposes, and for a drawer front or a
        # cabinet door it sits on the face that has to be pulled.
        handle_position = np.asarray(
            articulation.get_joint_leaf_body_position(joint_index), dtype=float
        )
        return articulation, joint_index, handle_position
