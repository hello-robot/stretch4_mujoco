"""
MolmoBot-DROID driving a real Stretch 4, through Franka retargeting. Works like
`run_stretch4_sim.py`: type instructions at the prompt and watch in Rerun (the policy's views,
the raw cameras, the robot and the ghost Franka it is retargeted from).

On the robot, run the image + joint-state publisher from stretch4_compliant_gripper first:

    python send_gripper_and_head_images_with_joint_states.py -r --head_camera left --wrist_camera left --wrist_resolution 400

with its head and wrist camera sides matching --exo_camera and --gripper_camera. Commands go
through stretch4_body's `RobotClient`; with --wait-for-arrival (the default) each action waits
for `wait_command()`. Ctrl+C stops the rollout and the robot.

There is no scene to read the target's height from, so give it with --object-height: the
virtual Franka stands where molmospaces would put it for an object at that height.

Usage:
    python -m examples.vla.molmobot_droid.run_stretch4_real --robot_ip 10.0.0.12 --object-height 0.80 \\
        --exo_camera left --gripper_camera left --slow
"""

from __future__ import annotations

import math
import time

import click
import cv2
import mujoco
import numpy as np

from examples.vla.molmobot_droid import rerun_scene
from examples.vla.molmobot_droid.checkpoint import (
    FRANKA_HOME_QPOS,
    GRIPPER_CLOSED,
    POLICY_DT,
    build_instruction,
    load_policy,
    unload_policy,
)
from examples.vla.molmobot_droid.droid import (
    FrankaSpawn,
    Observation,
    RobotPose,
    add_franka_ghost,
    franka_link0_height_for_object,
    hold_ghost_pose,
    mat_to_quat,
)
from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import (
    MIN_BASE_ROTATION,
    SLOW_FACTOR,
    STEP_ARRIVAL_TIMEOUT,
    FrankaStretchRetargeter,
    RetargetParams,
    StretchJoints,
    StretchTargets,
    params_from_kwargs,
    planar_transform,
    prepare_exo,
    retarget_options,
    wrist_view,
)
from examples.vla.molmobot_droid.rollout import StepInfo, interactive_session

IMAGE_PORT = 4409
"""`gripper_networking.gripper_and_joints_port` in stretch4_compliant_gripper."""

GRIPPER_PCT_OPEN = 300.0
"""stretch4_body StretchGripper `pct_max_open` for SE4 (range_deg [-100, 300]); 0 is fingertips touching."""

PARALLEL_GRIPPER_OPEN_MM = 80.0
"""The parallel gripper's widest opening: two fingers of 40 mm travel (the PG4 URDF); 0 is closed."""

# stretch4_body SE4 default motion profiles; --slow runs at SLOW_FACTOR of them.
DEFAULT_SPEEDS = {"lift": 0.3, "arm": 0.4, "wrist": 7.0, "base_rotate": 2.0}


def rotate_head_image_to_upright(image: np.ndarray, head_camera_side: str) -> np.ndarray:
    """
    Stretch 4's head fisheyes are mounted with landscape sensors: left turns 90° CCW, right 90° CW
    (as in recv_gripper_and_head_images_with_joint_states.py).
    """
    if head_camera_side == "left":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if head_camera_side == "right":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def _decode(message: dict, *keys: str) -> np.ndarray | None:
    """The first image present under `keys` (JPEG bytes or raw BGR), as RGB."""
    for key in keys:
        if key not in message:
            continue
        value = message[key]
        image = cv2.imdecode(np.frombuffer(value, np.uint8), cv2.IMREAD_COLOR) if key.endswith("compressed") else value
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return None


class ImageAndJointReceiver:
    """
    Subscribes to the robot's synced wrist + head frames and joint states, keeping only the
    newest message, as recv_gripper_and_head_images_with_joint_states.py does.
    """

    def __init__(self, robot_ip: str, port: int = IMAGE_PORT):
        import zmq

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect(f"tcp://{robot_ip}:{port}")

    def receive(self, timeout_s: float = 5.0) -> dict:
        if not self.socket.poll(int(timeout_s * 1000)):
            raise TimeoutError("No images from the robot; is the publisher running with -r?")
        message = self.socket.recv_pyobj()
        message["wrist_rgb"] = _decode(
            message, "wrist_color_image_compressed", "wrist_color_image", "color_image_compressed", "color_image"
        )
        head = _decode(message, "head_color_image_compressed", "head_color_image")
        message["head_rgb"] = (
            rotate_head_image_to_upright(head, message.get("head_camera_side", "")) if head is not None else None
        )
        return message

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


class RobotModel:
    """
    Stretch 4 (the stretch4_mujoco model) and the ghost Franka on a floor, posed from the real
    robot's joint states, for Rerun. Also where the virtual Franka stands for retargeting.
    """

    def __init__(self, object_height: float, tool_name: str):
        from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator

        spec = mujoco.MjSpec.from_file(Stretch4MujocoSimulator.get_robot_xml_path(tool_name))
        # The wheels' contact pairs name a geom "floor".
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1])
        self.franka = add_franka_ghost(spec, RobotPose(0.0, 0.0, 0.0), franka_link0_height_for_object(object_height))
        self.model = spec.compile()
        hold_ghost_pose(self.model, self.franka)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def pose(self, world_from_footprint: np.ndarray, joints: StretchJoints, franka_q7) -> None:
        data = self.data
        data.joint("floating_base").qpos = np.concatenate(
            [world_from_footprint[:3, 3], mat_to_quat(world_from_footprint[:3, :3])]
        )
        data.joint("lift_joint").qpos = joints.lift
        for i in range(1, 5):
            data.joint(f"arm_l{i}_joint").qpos = joints.arm / 4
        for joint in ("wrist_yaw", "wrist_pitch", "wrist_roll"):
            data.joint(f"{joint}_joint").qpos = getattr(joints, joint)
        for i, value in enumerate(franka_q7):
            data.joint(f"{self.franka.prefix}fr3_joint{i + 1}").qpos = value
        mujoco.mj_kinematics(self.model, data)


class RealStretch4Env:
    """A real Stretch 4 driven by Franka actions; the real-robot counterpart of `Stretch4SimEnv`."""

    def __init__(self, robot, receiver: ImageAndJointReceiver, params: RetargetParams, franka: FrankaSpawn):
        self.robot = robot
        self.receiver = receiver
        self.params = params
        self.retargeter = FrankaStretchRetargeter(franka, params)
        self._gripper_closed: bool | None = None
        self._odometry_origin: np.ndarray | None = None
        self.last_message: dict = {}
        self.check_camera_sides()

    def check_camera_sides(self) -> None:
        message = self.receiver.receive()
        sides = {"head": message.get("head_camera_side"), "wrist": message.get("wrist_camera_side")}
        if self.params.exo_camera != sides["head"] or self.params.gripper_camera != sides["wrist"]:
            raise click.ClickException(
                f"The robot publishes head={sides['head']}, wrist={sides['wrist']}; run with "
                f"--exo_camera {sides['head']} --gripper_camera {sides['wrist']} or restart its publisher"
            )
        if message["head_rgb"] is None:
            raise click.ClickException("The robot is not publishing a head image")

    # -- state ------------------------------------------------------------

    def _state(self, message: dict) -> tuple[np.ndarray, StretchJoints]:
        state = message["closest_joint_state"]
        odometry = state["base_odometry"]
        pose = np.array([odometry["x"], odometry["y"], odometry["theta"]])
        if self._odometry_origin is None:
            self._odometry_origin = pose
        # The world frame is the footprint where the run started, which is where the Franka stands.
        x0, y0, theta0 = self._odometry_origin
        dx, dy = pose[0] - x0, pose[1] - y0
        c, s = math.cos(-theta0), math.sin(-theta0)
        world_from_footprint = planar_transform(c * dx - s * dy, s * dx + c * dy, pose[2] - theta0)
        gripper = state["gripper"]
        if "pos_mm" in gripper:
            open_fraction = gripper["pos_mm"] / PARALLEL_GRIPPER_OPEN_MM
        else:
            open_fraction = gripper.get("pos_pct", 0.0) / GRIPPER_PCT_OPEN
        joints = StretchJoints(
            lift=state["lift"]["height"],
            arm=state["arm"]["extension"],
            wrist_yaw=state["wrist_yaw"]["angle"],
            wrist_pitch=state["wrist_pitch"]["angle"],
            wrist_roll=state["wrist_roll"]["angle"],
            gripper_open_fraction=float(np.clip(open_fraction, 0, 1)),
        )
        return world_from_footprint, joints

    def observe(self) -> Observation:
        message = self.receiver.receive()
        self.last_message = message
        footprint, joints = self._state(message)
        state8, _ = self.retargeter.stretch_to_franka(footprint, joints)
        return Observation(
            exo_rgb=prepare_exo(message["head_rgb"], self.params),
            wrist_rgb=wrist_view(message["wrist_rgb"], self.retargeter.tcp_flipped),
            state8=state8,
            extra_cameras={"head_raw": message["head_rgb"], "gripper_raw": message["wrist_rgb"]},
        )

    # -- acting -------------------------------------------------------------

    def step(self, action8: np.ndarray) -> StretchTargets | None:
        footprint, joints = self._state(self.receiver.receive())
        targets = self.retargeter.franka_to_stretch(action8, footprint, joints)
        start = time.monotonic()
        if targets is not None:
            self.send(targets)
        else:
            self.send_gripper(bool(action8[7] >= GRIPPER_CLOSED / 2))
            self.robot.push_command()
        if self.params.wait_for_arrival:
            self.robot.wait_command(timeout=STEP_ARRIVAL_TIMEOUT)
        remaining = POLICY_DT - (time.monotonic() - start)
        if remaining > 0:
            time.sleep(remaining)
        return targets

    def _speed(self, joint: str) -> float | None:
        return SLOW_FACTOR * DEFAULT_SPEEDS[joint] if self.params.slow else None

    def send(self, targets: StretchTargets) -> None:
        robot = self.robot
        print(
            f"  base {math.degrees(targets.base_rotate_by):+.1f}°  lift {targets.lift:.3f}  arm {targets.arm:.3f}  "
            f"yaw {targets.wrist_yaw:+.2f}  pitch {targets.wrist_pitch:+.2f}  roll {targets.wrist_roll:+.2f}  "
            f"gripper {'closed' if targets.gripper_closed else 'open'}" + ("  (clamped)" if targets.clamped else "")
        )
        if abs(targets.base_rotate_by) > MIN_BASE_ROTATION:
            robot.base.rotate_by(targets.base_rotate_by, v_r=self._speed("base_rotate"))
        robot.lift.move_to(targets.lift, v_m=self._speed("lift"))
        robot.arm.move_to(targets.arm, v_m=self._speed("arm"))
        for joint in ("wrist_yaw", "wrist_pitch", "wrist_roll"):
            robot.end_of_arm.move_to(joint, getattr(targets, joint), v_r=self._speed("wrist"))
        self.send_gripper(targets.gripper_closed)
        robot.push_command()

    def send_gripper(self, closed: bool) -> None:
        if closed != self._gripper_closed:
            self._gripper_closed = closed
            joint = "parallel_gripper" if self.params.use_parallel_gripper else "stretch_gripper"
            self.robot.end_of_arm.pose(joint, "close" if closed else "open")

    def move_to_franka_pose(self, franka_q7=FRANKA_HOME_QPOS) -> None:
        self.retargeter.reset()
        self.retargeter.franka_seed = np.asarray(franka_q7, dtype=float)
        footprint, joints = self._state(self.receiver.receive())
        targets = self.retargeter.franka_to_stretch(np.concatenate([franka_q7, [0.0]]), footprint, joints)
        if targets is None:
            raise click.ClickException("Stretch 4 cannot reach the Franka's home pose; check --object-height")
        self._gripper_closed = None
        self.send(targets)
        self.robot.wait_command(timeout=30.0)

    def stop_motion(self) -> None:
        """Hold every joint where it is."""
        footprint, joints = self._state(self.receiver.receive())
        self.robot.lift.move_to(joints.lift)
        self.robot.arm.move_to(joints.arm)
        self.robot.base.set_velocity(0.0, 0.0, 0.0)
        self.robot.push_command()


@click.command()
@click.option("--robot_ip", "--robot-ip", "robot_ip", required=True, help="Stretch 4's IP address.")
@click.option("--object-height", type=float, default=0.80, show_default=True,
              help="Height of the target object above the floor (m); sets the virtual Franka's height.")
@click.option("--object-name", default="object", show_default=True, help="For the default instruction.")
@click.option("--max-steps", type=int, default=300, show_default=True)
@click.option("--checkpoint", default=None)
@click.option("--rerun/--no-rerun", default=True, show_default=True)
@retarget_options
def main(robot_ip, object_height, object_name, max_steps, checkpoint, rerun, **kwargs):
    params = params_from_kwargs(kwargs)
    if params.exo_camera == "droid":
        raise click.BadParameter("the real robot has no DROID exo camera; use left, right or center", param_hint="--exo_camera")

    from stretch4_body.robot.robot_client import RobotClient

    robot = RobotClient(ip_address=robot_ip)
    if not robot.startup():
        raise click.ClickException(f"Could not connect to Stretch 4 at {robot_ip}")
    if not robot.is_homed():
        robot.stop()
        raise click.ClickException("Home the robot first")

    gripper_joint = "parallel_gripper" if params.use_parallel_gripper else "stretch_gripper"
    if gripper_joint not in robot.end_of_arm.joints:
        robot.stop()
        raise click.ClickException(
            f"This robot's end of arm has {robot.end_of_arm.joints}, not {gripper_joint}; "
            + ("drop" if params.use_parallel_gripper else "add") + " --use_parallel_gripper"
        )
    robot_model = RobotModel(object_height, params.tool_name)
    receiver = ImageAndJointReceiver(robot_ip)
    env = RealStretch4Env(robot, receiver, params, robot_model.franka)

    logger = None
    if rerun:
        rerun_scene.init_rerun("MolmoBot-DROID Stretch 4 (real)", ["exo", "wrist", "head_raw", "gripper_raw"])
        logger = rerun_scene.RerunScene(
            robot_model.model, robot_model.data, ["stretch4", robot_model.franka.base_name]
        )
    step_count = 0

    def on_step(info: StepInfo):
        nonlocal step_count
        step_count += 1
        if logger is None:
            return
        rerun_scene.set_step(step_count)
        observation = info.observation
        rerun_scene.log_cameras(
            {"exo": observation.exo_rgb, "wrist": observation.wrist_rgb, **observation.extra_cameras}
        )
        footprint, joints = env._state(env.last_message)
        robot_model.pose(footprint, joints, observation.state8[:7])
        logger.log(robot_model.data)
        rerun_scene.log_tool_poses(
            env.retargeter.last_target_tool_world, env.retargeter.stretch_tool_world(footprint, joints)
        )

    click.secho("Moving Stretch 4 to the Franka's home pose...", fg="yellow")
    env.move_to_franka_pose()
    policy = load_policy(checkpoint)
    try:
        interactive_session(
            env,
            policy,
            build_instruction("pick", object_name),
            params.execute_horizon,
            params.execute_first_n,
            max_steps,
            reset=env.move_to_franka_pose,
            on_step=on_step,
        )
    finally:
        env.stop_motion()
        unload_policy()
        receiver.close()
        robot.stop()


if __name__ == "__main__":
    main()
