"""
A MolmoSpaces `Robot` for Stretch 4.

This is the counterpart to `examples/molmo_environment.py`. That module attaches
the Stretch MJCF into a MolmoSpaces scene so `Stretch4MujocoSimulator` can drive
it; this one attaches it the way *MolmoSpaces* expects, so their task samplers,
sensor suites, controllers and benchmark evaluator can drive it instead.

The two differ in one substantial way: the base.

`molmo_environment.py` keeps Stretch's real, free-jointed, three-omniwheel base
and retargets the wheel contact pairs onto the scene's floors, because the point
there is to drive the actual robot. MolmoSpaces benchmarks instead freeze the
robot at a per-episode `task.robot_base_pose` and expect
`robot_view.base.pose = ...` to place it exactly, and their position controllers
expect the base to be commandable in (x, y, theta). Stretch's velocity-controlled
omniwheels satisfy neither. So, exactly as MolmoSpaces does for its own mobile
robots (`robots/mobile_franka.py`, and RBY1 under `use_holo_base`), the robot is
hung off three *virtual* holonomic joints -- two slides and a hinge -- driven by
position actuators.

The real wheels stay in the model, visually and inertially, but never touch
anything: Stretch's wheel geoms use `contype=6 conaffinity=4`, which does not
match the `contype=1 conaffinity=1` of a MolmoSpaces floor, so they only ever
collided through the explicit `<pair>` elements in `models/stretch_4/contact.xml`
that `_prepare_robot_spec()` removes. With the pairs gone the wheels are inert
and the base slides on its holonomic joints; the lowest remaining collision geom
(`base_link_collision`) clears the floor by ~8cm.
"""

import logging
from typing import TYPE_CHECKING, cast

import mujoco
import numpy as np
from mujoco import MjData, MjSpec
from scipy.spatial.transform import Rotation as R

from examples.machine_learning.molmospaces.stretch.robot_view import Stretch4RobotView
from molmo_spaces.controllers.abstract import Controller
from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.controllers.joint_rel_pos import JointRelPosController
from molmo_spaces.env.sensors import TCPPoseSensor
from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.kinematics.parallel.dummy_parallel_kinematics import DummyParallelKinematics
from molmo_spaces.robots.abstract import Robot

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
    from molmo_spaces.configs.robot_configs import BaseRobotConfig

    from examples.machine_learning.molmospaces.stretch.config import Stretch4RobotConfig

log = logging.getLogger(__name__)

# The body `mjcf_generator.generate_mjcf()` gives the robot root. It carries a
# freejoint, which `_prepare_robot_spec()` strips before the body is grafted onto
# the holonomic base.
STRETCH_ROOT_BODY = "stretch4"

# Move groups whose command mode is configurable via `robot_config.command_mode`.
# The gripper is always absolute position: relative finger commands make grasp
# force impossible to hold.
_RELATIVE_CAPABLE_GROUPS = ("lift", "arm", "wrist")


class Stretch4Robot(Robot):
    """Stretch 4 as a MolmoSpaces robot: holonomic base, lift, arm, wrist, gripper."""

    def __init__(self, mj_data: MjData, exp_config: "MlSpacesExpConfig") -> None:
        super().__init__(mj_data, exp_config)
        robot_config = exp_config.robot_config

        self._robot_view = robot_config.robot_view_factory(mj_data, robot_config.robot_namespace)
        self._kinematics = MlSpacesKinematics(robot_config)
        # Stretch has 8 manipulator DOFs, so the batched IK that the CuRobo/warp
        # planners need is not worth a warp kernel; the sequential fallback that
        # MolmoSpaces ships for its other small robots is enough.
        self._parallel_kinematics = DummyParallelKinematics(robot_config, self._kinematics)

        base_controller_cls: type[Controller] = {
            "holo_joint_planar_position": JointPosController,
            "holo_joint_rel_planar_position": JointRelPosController,
        }[robot_config.command_mode["base"]]

        self._controllers: dict[str, Controller] = {
            "base": base_controller_cls(self._robot_view.get_move_group("base")),
            "gripper": JointPosController(self._robot_view.get_move_group("gripper")),
        }
        for group in _RELATIVE_CAPABLE_GROUPS:
            controller_cls = (
                JointRelPosController
                if robot_config.command_mode.get(group) == "joint_rel_position"
                else JointPosController
            )
            self._controllers[group] = controller_cls(self._robot_view.get_move_group(group))

    @property
    def namespace(self) -> str:
        return self.exp_config.robot_config.robot_namespace

    @property
    def robot_view(self) -> Stretch4RobotView:
        return self._robot_view

    @property
    def kinematics(self) -> MlSpacesKinematics:
        return self._kinematics

    @property
    def parallel_kinematics(self) -> DummyParallelKinematics:
        return self._parallel_kinematics

    @property
    def controllers(self) -> dict[str, Controller]:
        return self._controllers

    def create_robot_sensors(self):
        return super().create_robot_sensors() + [TCPPoseSensor(uuid="tcp_pose")]

    def get_arm_move_group_ids(self) -> list[str]:
        """Groups that `Robot.apply_action_noise()` perturbs in TCP space.

        Only the wrist qualifies. That noise model maps a TCP delta back through
        `J^+`, which needs the group's Jacobian to be well conditioned for
        6-DOF motion; the lift and arm each contribute a single translational
        column, so a least-squares solve against them is dominated by the
        component they cannot produce.
        """
        return ["wrist"]

    def reset(self) -> None:
        """Restore the arm to its configured start pose. Deliberately not the base.

        `init_qpos["base"]` is the origin, and it is meant for standalone use --
        in an evaluation the base pose belongs to the episode
        (`task.robot_base_pose`), not to the robot config. Applying it here would
        either teleport a correctly-placed robot to the world origin or, worse,
        leave the base *controllers* targeting the origin while the base itself
        stands elsewhere, so the first control interval of every episode would
        command a full-speed drive across the house.
        """
        for mg_id, default_pos in self.exp_config.robot_config.init_qpos.items():
            if mg_id != "base" and mg_id in self._robot_view.move_group_ids():
                self._robot_view.get_move_group(mg_id).joint_pos = np.asarray(default_pos)
        for controller in self._controllers.values():
            controller.reset()

    @staticmethod
    def robot_model_root_name() -> str:
        return STRETCH_ROOT_BODY

    # =========================================================================
    # Scene assembly
    # =========================================================================

    @classmethod
    def _prepare_robot_spec(cls, robot_spec: MjSpec) -> None:
        """Make the standalone Stretch MJCF safe to graft into a foreign scene.

        Three things in the generated model only make sense for the standalone
        `scene_stretch4.xml`:

        1. The freejoint on the root body. The root is about to become a child of
           the holonomic base body, and a free-jointed body cannot hang off one
           whose pose is already determined by other joints.
        2. The omniwheel `<pair>` elements. They name a geom called "floor" that
           a MolmoSpaces house need not contain, which is a hard compile error,
           and with the holonomic base there is nothing for them to do -- see the
           module docstring.
        3. The keyframes. Their `ctrl` vectors are sized for the standalone
           model's 10 actuators; this scene will have more (three base actuators,
           plus whatever articulated furniture the house brings). MuJoCo warns
           about exactly this on attach ("child model has pending keyframes").
        """
        root = robot_spec.body(STRETCH_ROOT_BODY)
        if root is None:
            raise ValueError(
                f"Body '{STRETCH_ROOT_BODY}' not found in the Stretch MJCF. "
                "models/stretch_4/mjcf_generator.py is expected to emit it as the root."
            )
        for joint in list(root.joints):
            if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                robot_spec.delete(joint)

        for pair in list(robot_spec.pairs):
            robot_spec.delete(pair)

        # Keyframes stay "pending" -- invisible to `spec.keys` and undeletable --
        # until a compile resolves them, so the throwaway compile below is what
        # makes the deletion possible. It has to come after the pairs are gone,
        # since those name a "floor" geom the robot model does not define.
        robot_spec.compile()
        for key in list(robot_spec.keys):
            robot_spec.delete(key)

        # Both the robot spec and any MolmoSpaces house call their root default
        # class "main". Compiling a merged spec with two of them is fine, but
        # serialising it back out produces a second unnamed <default> block that
        # MuJoCo then refuses to read. Renaming is cosmetic -- elements hold a
        # pointer to their default, not its name.
        robot_spec.default.name = "stretch_main"

    @classmethod
    def add_robot_to_scene(
        cls,
        robot_config: "BaseRobotConfig",
        spec: MjSpec,
        prefix: str,
        pos: list[float],
        quat: list[float],
        randomize_textures: bool = False,
        strip_meshes: bool = False,
    ) -> None:
        """Attach Stretch onto three virtual holonomic joints in `spec`.

        The resulting structure, for `prefix="robot_0/"`:

            worldbody
              site "robot_0/world"                 -- reference frame for the
                                                      base slide actuators
              body "robot_0/base"                  -- carries the holonomic joints
                site "robot_0/base_site"
                joint "robot_0/base_x"     (slide)
                joint "robot_0/base_y"     (slide)
                joint "robot_0/base_theta" (hinge)
                body "robot_0/stretch4"            -- the whole real robot

        `StretchBaseGroup` looks all of those up by exactly these names.

        Args:
            robot_config: a `Stretch4RobotConfig`.
            spec: the scene to attach into, modified in place.
            prefix: robot namespace, e.g. "robot_0/".
            pos: spawn position, [x, y] or [x, y, z]. MolmoSpaces' own task
                sampler always passes the origin and repositions the robot later
                via `robot_view.base.pose`, but a non-zero spawn is supported so
                the same code can be used standalone.
            quat: spawn orientation as [w, x, y, z]. Must be a yaw-only rotation:
                the holonomic base has no roll or pitch degree of freedom, so a
                tilted spawn could not be represented afterwards.
            randomize_textures: unused; Stretch ships a single appearance.
            strip_meshes: drop mesh geoms, for the kinematics-only model that
                `MlSpacesKinematics` builds.
        """
        robot_config = cast("Stretch4RobotConfig", robot_config)
        pos = list(pos) + [0.0] if len(pos) == 2 else list(pos)

        spawn_rotation = R.from_quat(quat, scalar_first=True)
        roll, pitch, yaw = spawn_rotation.as_euler("xyz")
        if not np.allclose([roll, pitch], 0.0, atol=1e-6):
            raise ValueError(
                f"Stretch's holonomic base is planar, so the spawn orientation must be yaw-only; "
                f"got roll={roll:.4f}, pitch={pitch:.4f}."
            )

        robot_spec = cls._load_robot_spec(robot_config, strip_meshes=strip_meshes)
        cls._prepare_robot_spec(robot_spec)

        spec.worldbody.add_site(name=f"{prefix}world", pos=[0, 0, 0], quat=[1, 0, 0, 0])
        base_body = spec.worldbody.add_body(name=f"{prefix}base", pos=pos, quat=quat)
        base_body.add_site(name=f"{prefix}base_site", pos=[0, 0, 0], quat=[1, 0, 0, 0])

        for axis_index, axis in enumerate(("x", "y")):
            params = robot_config.base_control_params[f"base_{axis}_act"]
            # The joint lives inside the (already yawed) base body, so its axis
            # has to be expressed in that body's frame for the joint to slide
            # along world x / world y.
            joint_axis = np.zeros(3)
            joint_axis[axis_index] = 1.0
            base_body.add_joint(
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                name=f"{prefix}base_{axis}",
                axis=spawn_rotation.inv().apply(joint_axis),
                range=[-params["ctrlrange"], params["ctrlrange"]],
                ref=pos[axis_index],
            )
            actuator = spec.add_actuator(
                name=f"{prefix}base_{axis}_act",
                target=f"{prefix}base_site",
                refsite=f"{prefix}world",
                trntype=mujoco.mjtTrn.mjTRN_SITE,
                biastype=mujoco.mjtBias.mjBIAS_AFFINE,
            )
            actuator.ctrlrange = np.array([-params["ctrlrange"], params["ctrlrange"]])
            actuator.gainprm[0] = params["kp"]
            actuator.biasprm[1] = -params["kp"]
            actuator.biasprm[2] = -params["kd"]
            gear = [0.0] * 6
            gear[axis_index] = 1.0
            actuator.gear = gear

        theta_params = robot_config.base_control_params["base_theta_act"]
        base_body.add_joint(
            type=mujoco.mjtJoint.mjJNT_HINGE,
            name=f"{prefix}base_theta",
            axis=[0, 0, 1],
            ref=yaw,
        )
        theta_actuator = spec.add_actuator(
            name=f"{prefix}base_theta_act",
            target=f"{prefix}base_theta",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        )
        # Without an explicit range this defaults to [0, 0], and JointPosController
        # clips its target to the actuator's control range -- which would silently
        # pin the base yaw at zero.
        theta_actuator.ctrlrange = np.array([-theta_params["ctrlrange"], theta_params["ctrlrange"]])
        theta_actuator.gainprm[0] = theta_params["kp"]
        theta_actuator.biasprm[1] = -theta_params["kp"]
        theta_actuator.biasprm[2] = -theta_params["kd"]

        attach_frame = base_body.add_frame(pos=[0, 0, 0], quat=[1, 0, 0, 0])
        attach_frame.attach_body(robot_spec.body(STRETCH_ROOT_BODY), prefix, "")
