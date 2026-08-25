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
and no grasp-pose search -- MolmoSpaces' own solvers do both, but they need
CuRobo and a per-robot grasp library that does not exist for Stretch. What is
here instead is a waypoint machine: every task family compiles down to a list of
`Waypoint`s for the tool centre, and one executor drives them.

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
    PITCH_HORIZONTAL,
    PITCH_TOP_DOWN,
    StretchReachSolver,
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
class Waypoint:
    """One tool-centre target in a simple_ik plan.

    Attributes:
        position: world xyz for the tool centre.
        wrist_pitch: approach tilt to hold while getting there.
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

    grasp_style: str = "horizontal"
    """
    'horizontal' or 'top_down'. Horizontal is the default because it is what
    Stretch's kinematics support: the tool sits ahead of the wrist yaw axis, so
    yaw gives lateral authority and the arm can reach without driving the base.
    Reaching straight down loses that authority entirely -- see
    `policies/kinematics.py`.
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

    unstow_clearance_m: float = 0.25
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

    grasp_library_candidates: int = 40
    """
    How many authored grasp points to consider before falling back to the object's origin.

    The object's body origin is a poor thing to aim at for anything that is not
    roughly a blob. A bowl's origin is its centre, where the object measures its
    full 0.27m rim-to-rim -- wider than the gripper opens -- while the rim itself
    is a few millimetres thick and perfectly graspable. MolmoSpaces ships authored
    grasp poses per asset and the task sampler already requires them
    (`filter_for_grasps`), so the graspable places are known; this policy just has
    to pick one it can also reach.

    Forty is a compromise: enough to find a reachable one on a cluttered shelf,
    few enough that the search costs a fraction of a second at ~2.5ms per solve.
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

    reach_tolerance_m: float = 0.03
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

    def get_info(self) -> dict:
        return {
            "policy": "stretch_simple_ik",
            "grasp_style": self.config.policy_config.grasp_style,
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
        solution = self._solver.solve(
            base_pose,
            waypoint.position,
            wrist_pitch=waypoint.wrist_pitch,
            wrist_roll=0.0,
            seed=current,
            tolerance=policy_config.reach_tolerance_m * 0.5,
        )
        if solution is None:
            # Out of reach. Hold the arm where it is rather than commanding a
            # half-converged pose, and let the step budget move the plan on.
            for group in ("base", "lift", "arm", "wrist"):
                action[group] = current[group].copy()
            return action

        action["lift"] = solution["lift"]
        action["arm"] = solution["arm"]
        action["wrist"] = solution["wrist"]
        action["base"] = solution["base"] if waypoint.turn_base else current["base"].copy()
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

        if self._grip_hold is None and self._fingers_touching_pickup():
            self._grip_hold = current + inward * policy_config.grasp_hold_preload_rad
        if self._grip_hold is not None:
            return float(np.clip(self._grip_hold, min(closed, opened), max(closed, opened)))

        stepped = current + inward * policy_config.grasp_close_step_rad
        return float(np.clip(stepped, min(closed, opened), max(closed, opened)))

    def _fingers_touching_pickup(self) -> bool:
        """Whether either finger is in contact with the object being picked up."""
        object_name = self.config.task_config.pickup_obj_name
        if not object_name:
            return False
        environment = self.task.env
        object_manager = environment.object_managers[environment.current_batch_index]
        scene_object = object_manager.get_object_by_name(object_name)
        if scene_object is None:
            return False

        model, data = environment.current_model, environment.current_data
        gripper_root = self._gripper_group().root_body_id
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
            # Any robot contact will do: the fingers are what is closing, and the
            # arm is stationary at this point in the plan.
            robot_geom = contact.geom1 if roots[0] != scene_object.body_id else contact.geom2
            body = model.geom_bodyid[robot_geom]
            while body != 0:
                if body == gripper_root:
                    return True
                body = model.body_parentid[body]
        return False

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

        if reached:
            self._settled_steps += 1
        timed_out = self._steps_in_waypoint >= self.config.policy_config.max_steps_per_waypoint

        if (reached and self._settled_steps > waypoint.settle_steps) or timed_out:
            if timed_out and not reached:
                log.debug(
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
        pitch = self._grasp_pitch()
        is_top_down = self.config.policy_config.grasp_style == "top_down"
        grasp_point = self._pickup_grasp_point(pitch)
        if grasp_point is None:
            return []

        approach = self._approach_direction(pitch, grasp_point)
        policy_config = self.config.policy_config

        # The fingers separate along the tool's y axis, which for an unrolled
        # grasp is perpendicular to both the approach and the vertical.
        closing_axis = np.cross(np.array([0.0, 0.0, 1.0]), approach)
        grip_width = self._object_grasp_width(
            self.config.task_config.pickup_obj_name, closing_axis, grasp_point
        )
        log.info(
            f"[stretch-simple-ik] grasp width along the closing axis: "
            f"{'unknown' if grip_width is None else f'{grip_width:.3f}m'}"
        )

        plan = self._unstow_waypoints(grasp_point, pitch)
        pregrasp = self._reachable_standoff(grasp_point, approach, pitch)
        if pregrasp is not None:
            plan.append(
                Waypoint(
                    position=pregrasp,
                    wrist_pitch=pitch,
                    gripper_open=True,
                    label="pregrasp",
                    tolerance=policy_config.reach_tolerance_m,
                )
            )
        plan += [
            Waypoint(
                position=grasp_point + approach * policy_config.grasp_depth_m,
                wrist_pitch=pitch,
                gripper_open=True,
                label="reach",
                tolerance=policy_config.reach_tolerance_m,
            ),
            Waypoint(
                position=grasp_point + approach * policy_config.grasp_depth_m,
                wrist_pitch=pitch,
                gripper_open=False,
                label="close",
                settle_steps=policy_config.gripper_settle_steps,
                establishes_grasp=True,
                grip_width_m=grip_width,
                tolerance=policy_config.reach_tolerance_m * 2.0,
            ),
            Waypoint(
                position=grasp_point + np.array([0.0, 0.0, policy_config.lift_height_m]),
                wrist_pitch=pitch,
                gripper_open=False,
                label="lift",
                settle_steps=policy_config.gripper_settle_steps,
                verify_grasp=True,
                grip_width_m=grip_width,
                tolerance=policy_config.reach_tolerance_m,
            ),
        ]
        if not with_place:
            return plan

        place_point = self._object_grasp_point(self.config.task_config.place_receptacle_name)
        if place_point is None:
            return plan

        hover = place_point + np.array([0.0, 0.0, policy_config.place_hover_m])
        plan += [
            Waypoint(
                position=hover,
                wrist_pitch=pitch,
                gripper_open=False,
                label="transfer",
                grip_width_m=grip_width,
                # Carrying an object across the workspace is the one manipulation
                # motion whose span routinely exceeds the arm's, so allow the
                # base to help here even though the grasp phases may not.
                tolerance=policy_config.reach_tolerance_m * 1.5,
            ),
            Waypoint(
                position=hover,
                wrist_pitch=pitch,
                gripper_open=True,
                label="release",
                settle_steps=policy_config.gripper_settle_steps,
                tolerance=policy_config.reach_tolerance_m * 1.5,
            ),
            Waypoint(
                position=hover + np.array([0.0, 0.0, policy_config.place_hover_m]),
                wrist_pitch=pitch,
                gripper_open=True,
                label="retreat",
                tolerance=policy_config.reach_tolerance_m * 2.0,
            ),
        ]
        return plan

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
        # approach on a drawer pull collides with the cabinet face.
        pitch = PITCH_HORIZONTAL
        start = handle_arc[0]
        approach = self._approach_direction(pitch, start)

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

    def _grasp_pitch(self) -> float:
        return (
            PITCH_TOP_DOWN
            if self.config.policy_config.grasp_style == "top_down"
            else PITCH_HORIZONTAL
        )

    def _approach_direction(self, pitch: float, target: np.ndarray) -> np.ndarray:
        """Unit vector the tool travels along as it closes on `target`.

        Straight down for a top-down grasp; for a horizontal grasp, from the
        robot's base towards the target in the floor plane.
        """
        if pitch == PITCH_TOP_DOWN:
            return np.array([0.0, 0.0, -1.0])
        base_xy = self.task.env.current_robot.robot_view.base.pose[:2, 3]
        direction = np.asarray(target[:2], dtype=float) - base_xy
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return np.array([1.0, 0.0, 0.0])
        return np.array([direction[0] / norm, direction[1] / norm, 0.0])

    def _reachable_standoff(
        self, grasp_point: np.ndarray, approach: np.ndarray, pitch: float
    ) -> np.ndarray | None:
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
        for fraction in (1.0, 0.66, 0.33):
            candidate = grasp_point - approach * (standoff * fraction)
            if self._solver.solve(base_pose, candidate, wrist_pitch=pitch) is not None:
                return candidate
        log.info("[stretch-simple-ik] no reachable pregrasp standoff; approaching directly")
        return None

    def _object_grasp_point(self, object_name: str | None) -> np.ndarray | None:
        """A world point on `object_name` worth aiming the tool at.

        Uses the object's current body origin rather than its axis-aligned
        bounding-box centre: `MlSpacesObject.aabb_center` comes from the compiled
        model and so describes the object's *initial* pose, which for a
        benchmark episode is not where the object was moved to.
        """
        if not object_name:
            return None
        environment = self.task.env
        object_manager = environment.object_managers[environment.current_batch_index]
        scene_object = object_manager.get_object_by_name(object_name)
        if scene_object is None:
            log.warning(f"[stretch-simple-ik] object {object_name!r} not found in scene")
            return None
        return np.asarray(scene_object.position, dtype=float)

    def _unstow_waypoints(self, grasp_point: np.ndarray, pitch: float) -> list[Waypoint]:
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
        retracted = {
            "base": np.zeros(3),
            "lift": np.array([0.0]),
            "arm": np.array([0.0]),
            "wrist": np.array([0.0, pitch, 0.0]),
        }
        # Tool height rises one-for-one with the lift, so the offset between them
        # is whatever the tool sits at with the lift at zero.
        tool_height_at_zero_lift = float(solver.forward(retracted)[2, 3])
        base_z = float(self.task.env.current_robot.robot_view.base.pose[2, 3])
        wanted = float(grasp_point[2]) + policy_config.unstow_clearance_m
        lift_limits = solver.joint_limits["lift"][0]
        lift = float(
            np.clip(wanted - base_z - tool_height_at_zero_lift, lift_limits[0], lift_limits[1])
        )

        position = np.asarray(grasp_point, dtype=float)
        return [
            Waypoint(
                position=position,
                wrist_pitch=pitch,
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
                gripper_open=True,
                label="unstow",
                joint_targets={
                    "lift": np.array([lift]),
                    "arm": np.array([0.0]),
                    "wrist": np.array([0.0, pitch, 0.0]),
                },
            ),
        ]

    def _pickup_grasp_point(self, pitch: float) -> np.ndarray | None:
        """Where to aim the tool to pick up the task's object.

        Prefers a point from the asset's authored grasp library over the body
        origin, because the origin is only a sensible grasp for objects shaped
        roughly like blobs. Candidates are kept only if Stretch can both *reach*
        them and *close on* them -- a grasp authored for a wider gripper is no use
        here -- and the object's origin remains the fallback when the library is
        missing or nothing in it works.
        """
        object_name = self.config.task_config.pickup_obj_name
        origin = self._object_grasp_point(object_name)
        candidates = self._library_grasp_points(object_name)
        if candidates is None or origin is None:
            return origin

        gripper = self._gripper_group()
        _, open_width = gripper.inter_finger_dist_range
        base_pose = self.task.env.current_robot.robot_view.base.pose

        for candidate in candidates:
            approach = self._approach_direction(pitch, candidate)
            closing_axis = np.cross(np.array([0.0, 0.0, 1.0]), approach)
            width = self._object_grasp_width(object_name, closing_axis, candidate)
            if width is not None and width >= open_width:
                continue
            if self._solver.solve(base_pose, candidate, wrist_pitch=pitch) is None:
                continue
            log.info(
                f"[stretch-simple-ik] grasping at an authored grasp point "
                f"{float(np.linalg.norm(candidate - origin)):.3f}m from the object origin"
            )
            return np.asarray(candidate, dtype=float)

        log.info(
            "[stretch-simple-ik] no authored grasp point was both reachable and narrow "
            "enough; aiming at the object origin"
        )
        return origin

    def _library_grasp_points(self, object_name: str | None) -> np.ndarray | None:
        """Authored grasp positions for `object_name`, thinned to a workable number.

        Returns None whenever the library cannot be consulted -- no metadata, no
        grasps for this asset -- which is a fallback, not an error: the origin is
        still a usable aim point and the caller treats it as one.
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

            poses = get_pickup_grasps(
                environment,
                scene_object,
                include_flipped=True,
                grasp_libraries=self.config.task_sampler_config.grasp_libraries,
            )
        except Exception as failure:  # noqa: BLE001 - the library is optional here
            log.debug(f"[stretch-simple-ik] no grasp library for {object_name!r}: {failure}")
            return None

        if poses is None or len(poses) == 0:
            return None

        positions = np.asarray(poses, dtype=float)[:, :3, 3]
        limit = self.config.policy_config.grasp_library_candidates
        if len(positions) > limit:
            # Even strides rather than the first N: the library is ordered, and
            # the first N are all much the same grasp.
            positions = positions[:: max(1, len(positions) // limit)][:limit]
        return positions

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
