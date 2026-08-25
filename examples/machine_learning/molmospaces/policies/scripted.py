"""
A scripted Stretch 4 expert for the manipulation and articulation benchmarks.

This policy serves two purposes:

1. **A baseline.** It is the non-learned number to beat on each of the eight
   benchmark evaluations, and the thing to look at when a learned policy scores
   zero -- if the expert also scores zero on a benchmark, the problem is the
   retargeting or the robot, not the policy.
2. **A teacher.** `training/collect.py` runs it over benchmark scenes and keeps
   the successful trajectories as behaviour-cloning data. This is why it uses
   privileged state (object poses read straight out of `MjData`) rather than the
   camera images: the student learns the mapping from images to these actions.

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


@dataclass
class Waypoint:
    """One tool-centre target in a scripted plan.

    Attributes:
        position: world xyz for the tool centre.
        wrist_pitch: approach tilt to hold while getting there.
        gripper_open: whether the fingers should be open on arrival.
        label: phase name, surfaced through `PolicyPhaseSensor` and the logs.
        settle_steps: extra policy steps to hold once the position is reached,
            for motions whose effect is not visible in the tool pose -- closing
            the fingers being the obvious one.
        max_base_travel: how far this waypoint may drive the base. Reaching and
            grasping should not move the robot; dragging a drawer open has to.
        tolerance: arrival threshold in metres.
    """

    position: np.ndarray
    wrist_pitch: float
    gripper_open: bool
    label: str
    settle_steps: int = 0
    max_base_travel: float = 0.0
    tolerance: float = 0.03


class StretchScriptedPolicyConfig(BasePolicyConfig):
    """Configuration for `StretchScriptedPolicy`."""

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

    pregrasp_standoff_m: float = 0.14
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

    grasp_base_assist_m: float = 0.12
    """
    How far the base may inch forward to close the last of the distance to a
    grasp. `franka_remapping/episode_overrides.py` aims for the middle of Stretch's reach
    band, but a pick-and-place base pose is a compromise between two targets and
    can leave either of them marginal; a hand's breadth of base motion recovers
    those without meaningfully changing where the robot stands.
    """

    reach_tolerance_m: float = 0.03
    gripper_settle_steps: int = 8
    max_steps_per_waypoint: int = 45
    """Give up on a waypoint after this many policy steps and move on, rather
    than burning the whole horizon on one unreachable pose."""

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.policy_cls is None:
            self.policy_cls = StretchScriptedPolicy
            self.policy_factory = make_lenient(StretchScriptedPolicy)
        assert self.grasp_style in (
            "horizontal",
            "top_down",
        ), f"grasp_style must be 'horizontal' or 'top_down', got {self.grasp_style!r}"


class StretchScriptedPolicy(BasePolicy):
    """Executes a per-episode waypoint plan built from privileged scene state."""

    def __init__(self, config: "MlSpacesExpConfig", task: "BaseMujocoTask" = None) -> None:
        super().__init__(config, task)
        self._solver = StretchReachSolver(config.robot_config)
        self._plan: list[Waypoint] | None = None
        self._plan_origin_xy: np.ndarray | None = None
        self._waypoint_index = 0
        self._steps_in_waypoint = 0
        self._settled_steps = 0

    # =========================================================================
    # BasePolicy
    # =========================================================================

    def reset(self) -> None:
        self._plan = None
        self._plan_origin_xy = None
        self._waypoint_index = 0
        self._steps_in_waypoint = 0
        self._settled_steps = 0

    def get_info(self) -> dict:
        return {
            "policy": "stretch_scripted",
            "grasp_style": self.config.policy_config.grasp_style,
            "waypoints_total": 0 if self._plan is None else len(self._plan),
            "waypoints_reached": self._waypoint_index,
        }

    def get_action(self, observation) -> dict[str, Any]:
        del observation  # this policy reads privileged state, not sensors

        if self._plan is None:
            self._plan_origin_xy = np.asarray(
                self.task.env.current_robot.robot_view.base.pose[:2, 3], dtype=float
            )
            self._plan = self._build_plan()
            log.info(
                f"[stretch-scripted] planned {len(self._plan)} waypoints: "
                f"{[waypoint.label for waypoint in self._plan]}"
            )

        robot_view = self.task.env.current_robot.robot_view
        if self._waypoint_index >= len(self._plan):
            return robot_view.get_noop_ctrl_dict()

        waypoint = self._plan[self._waypoint_index]
        current = robot_view.get_qpos_dict()
        action = self._command_for(waypoint, current)
        self._advance(waypoint, robot_view)
        return action

    # =========================================================================
    # Execution
    # =========================================================================

    def _command_for(
        self, waypoint: Waypoint, current: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Absolute joint targets that pursue `waypoint` from `current`."""
        policy_config = self.config.policy_config
        gripper = self._gripper_group()
        finger_target = (
            gripper.OPEN_JOINT_POS if waypoint.gripper_open else gripper.CLOSED_JOINT_POS
        )
        action: dict[str, np.ndarray] = {
            "gripper": np.array([finger_target, finger_target]),
        }

        base_pose = self.task.env.current_robot.robot_view.base.pose
        solution = self._solver.solve(
            base_pose,
            waypoint.position,
            wrist_pitch=waypoint.wrist_pitch,
            wrist_roll=0.0,
            seed=current,
            max_base_travel=self._remaining_base_budget(waypoint, base_pose),
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
        action["base"] = (
            solution["base"] if waypoint.max_base_travel > 0.0 else current["base"].copy()
        )
        return action

    def _remaining_base_budget(self, waypoint: Waypoint, base_pose: np.ndarray) -> float:
        """How much further this waypoint may drive the base.

        The solver caps travel relative to the base pose it is handed, and it is
        handed the *live* pose every step -- so an uncapped-per-step budget would
        let the base creep a full allowance per policy step and wander off across
        the house. Budgeting against the pose the plan started from keeps the
        total displacement bounded by `Waypoint.max_base_travel`.
        """
        if waypoint.max_base_travel <= 0.0 or self._plan_origin_xy is None:
            return 0.0
        travelled = float(np.linalg.norm(np.asarray(base_pose[:2, 3]) - self._plan_origin_xy))
        return max(0.0, waypoint.max_base_travel - travelled)

    def _advance(self, waypoint: Waypoint, robot_view) -> None:
        """Move to the next waypoint once this one is reached, settled or timed out."""
        self._steps_in_waypoint += 1
        tool_position = robot_view.get_move_group("gripper").leaf_frame_to_world[:3, 3]
        reached = float(np.linalg.norm(tool_position - waypoint.position)) <= waypoint.tolerance

        if reached:
            self._settled_steps += 1
        timed_out = self._steps_in_waypoint >= self.config.policy_config.max_steps_per_waypoint

        if (reached and self._settled_steps > waypoint.settle_steps) or timed_out:
            if timed_out and not reached:
                log.debug(
                    f"[stretch-scripted] waypoint '{waypoint.label}' timed out "
                    f"{float(np.linalg.norm(tool_position - waypoint.position)):.3f}m short"
                )
            self._waypoint_index += 1
            self._steps_in_waypoint = 0
            self._settled_steps = 0

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
            f"[stretch-scripted] no plan builder for task class {task_cls_name!r}; holding still. "
            "Navigation benchmarks should use the A* planner policy instead, see configs.py."
        )
        return []

    def _plan_pick_and_place(self, with_place: bool) -> list[Waypoint]:
        pitch = self._grasp_pitch()
        grasp_point = self._object_grasp_point(self.config.task_config.pickup_obj_name)
        if grasp_point is None:
            return []

        approach = self._approach_direction(pitch, grasp_point)
        policy_config = self.config.policy_config

        plan = []
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
                max_base_travel=policy_config.grasp_base_assist_m,
                tolerance=policy_config.reach_tolerance_m,
            ),
            Waypoint(
                position=grasp_point + approach * policy_config.grasp_depth_m,
                wrist_pitch=pitch,
                gripper_open=False,
                label="close",
                settle_steps=policy_config.gripper_settle_steps,
                tolerance=policy_config.reach_tolerance_m * 2.0,
            ),
            Waypoint(
                position=grasp_point + np.array([0.0, 0.0, policy_config.lift_height_m]),
                wrist_pitch=pitch,
                gripper_open=False,
                label="lift",
                settle_steps=policy_config.gripper_settle_steps,
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
                # Carrying an object across the workspace is the one manipulation
                # motion whose span routinely exceeds the arm's, so allow the
                # base to help here even though the grasp phases may not.
                max_base_travel=0.3,
                tolerance=policy_config.reach_tolerance_m * 1.5,
            ),
            Waypoint(
                position=hover,
                wrist_pitch=pitch,
                gripper_open=True,
                label="release",
                settle_steps=policy_config.gripper_settle_steps,
                max_base_travel=0.3,
                tolerance=policy_config.reach_tolerance_m * 1.5,
            ),
            Waypoint(
                position=hover + np.array([0.0, 0.0, policy_config.place_hover_m]),
                wrist_pitch=pitch,
                gripper_open=True,
                label="retreat",
                max_base_travel=0.3,
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
                    max_base_travel=policy_config.articulation_travel_m + 0.2,
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
        log.info("[stretch-scripted] no reachable pregrasp standoff; approaching directly")
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
            log.warning(f"[stretch-scripted] object {object_name!r} not found in scene")
            return None
        return np.asarray(scene_object.position, dtype=float)

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
