import threading
import time

import click
import cv2
from examples.camera_feeds import show_camera_feeds_sync
from gamepad_controller import GamePadController, ButtonPressCounter, JointEffortTracker

from examples.rerun_utils import RerunLogger
from examples.laser_scan import show_laser_scan
from stretch4_mujoco import StretchMujocoSimulator
from stretch4_mujoco.enums.stretch_cameras import StretchCameras
from stretch4_mujoco.enums.stretch_sensors import StretchSensors
from stretch4_mujoco.enums.actuators import Actuators
from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator
from enum import Enum
import logging

sim: StretchMujocoSimulator
gamepad: GamePadController


@click.command()
@click.option("--scene-xml-path", type=str, default=None, help="Path to the scene xml file")
@click.option("--select_env", is_flag=True, help="Interactively select an environment")
@click.option("--headless", is_flag=True, help="Run in headless mode")
@click.option("--imagery", is_flag=True, help="Show all the cameras' imagery")
@click.option("--lidar2d", is_flag=True, help="Show the lidar scan in Matplotlib")
@click.option("--lidar3d", is_flag=True, help="Show the point cloud in Rerun")
@click.option("--print-ratio", is_flag=True, help="Print the sim-to-real time ratio to the cli.")
@click.option("--use_stretch_3", type=bool, is_flag=True, help="Use Stretch 3")
def main(
    scene_xml_path: str | None,
    select_env: bool,
    headless: bool,
    imagery: bool,
    lidar2d: bool,
    lidar3d: bool,
    print_ratio: bool,
    use_stretch_3: bool,
):
    global sim, gamepad

    rerun_logger = RerunLogger()

    simulator_class = StretchMujocoSimulator if use_stretch_3 else Stretch4MujocoSimulator

    use_head_joints = simulator_class is not Stretch4MujocoSimulator

    cameras_to_use = simulator_class.get_rgb_cameras() if imagery else  [StretchCameras.cam_gripper_rgb]

    if lidar3d and not simulator_class is Stretch4MujocoSimulator:
        raise NotImplementedError("3D Lidar is only supported in Stretch4MujocoSimulator.")

    if lidar3d:
        cameras_to_use += StretchCameras.hemispherical_lidars()
        rerun_logger.init_pointcloud_viz()

    use_imagery = len(cameras_to_use) > 0

    model = None

    if select_env:
        from stretch4_mujoco.robocasa_gen import model_generation_wizard

        model, xml, objects_info = model_generation_wizard(
            stretch_xml_absolute=simulator_class.get_robot_xml_path(),
            objects_list=["apple", "cup", "can", "milk"],
        )

    sim = simulator_class(
        model=model,
        scene_xml_path=scene_xml_path,
        cameras_to_use=cameras_to_use,
        camera_hz=10.00 if lidar3d else 30.0,
    )
    gamepad = GamePadController()
    try:
        sim.start(headless=headless)
        gamepad.start()
        threading.Thread(target=gamepad_loop, daemon=True, args=[use_head_joints]).start()
        while sim.is_running():
            if not lidar2d and not use_imagery:
                time.sleep(0.05)

            if print_ratio:
                print(f"{sim.pull_status().sim_to_real_time_ratio_msg}")

            if use_imagery:
                show_camera_feeds_sync(sim, False)

            if lidar3d:
                rerun_logger.update_pointcloud_viz(
                    sim.pull_hemi_lidar_points(in_world_frame=True), "world/lidar_points"
                )

            if lidar2d:
                sensor_data = sim.pull_sensor_data()

                try:
                    show_laser_scan(
                        scan_data=sensor_data.get_data(StretchSensors.base_lidar),
                        is_se4=simulator_class is Stretch4MujocoSimulator,
                    )
                except:
                    ...
    except KeyboardInterrupt:
        pass
    finally:
        rerun_logger.stop()
        sim.stop()
        cv2.destroyAllWindows()


def gamepad_loop(use_head_joints):
    global sim, gamepad
    dex_switch = False

    mapping = ControlMapping.MANIPULATION
    motion_profile = MotionProfile.DEFAULT
    adapter = GamepadTeleopAdapter(sim)
    robot = MockRobot()

    select_button_counter = ButtonPressCounter("select_button_pressed")
    start_button_counter = ButtonPressCounter("start_button_pressed")
    top_button_counter = ButtonPressCounter("top_button_pressed")
    left_button_counter = ButtonPressCounter("left_button_pressed")

    def change_mapping():
        nonlocal mapping
        mapping = mapping.cycle(is_forward=True)
        print(f"Switched mapping to {mapping.name}")
        gamepad.vibrate(duration_ms=150, strong_magnitude=1.0, weak_magnitude=1.0)

    def change_handedness():
        adapter.gripper_handedness = (
            GripperHandedness.LEFT
            if adapter.gripper_handedness == GripperHandedness.RIGHT
            else GripperHandedness.RIGHT
        )
        adapter.gripper_handedness.move_to(sim)
        duration = 150 * (adapter.gripper_handedness.value + 1)
        gamepad.vibrate(duration_ms=duration, strong_magnitude=1.0, weak_magnitude=1.0)

    def change_motion_profile():
        nonlocal motion_profile
        motion_profile = motion_profile.cycle(is_forward=True)
        print(f"Switched motion profile to {motion_profile.name}")
        duration = 150 * motion_profile.value
        gamepad.vibrate(duration_ms=duration, strong_magnitude=1.0, weak_magnitude=1.0)

    def toggle_dex_switch():
        nonlocal dex_switch
        dex_switch = not dex_switch
        print(f"Setting dex_switch to {dex_switch}")

    effort_trackers = {
        "joint_lift": JointEffortTracker("lift", pos_thresholds=[100.0, 200.0]),
        "joint_arm": JointEffortTracker("arm", pos_thresholds=[50.0, 100.0]),
        "joint_wrist_yaw": JointEffortTracker(
            "eoa", pos_thresholds=[2.0, 5.0], joint_name="wrist_yaw"
        ),
        "joint_wrist_pitch": JointEffortTracker(
            "eoa", pos_thresholds=[2.0, 5.0], joint_name="wrist_pitch"
        ),
        "joint_wrist_roll": JointEffortTracker(
            "eoa", pos_thresholds=[2.0, 5.0], joint_name="wrist_roll"
        ),
        "gripper": JointEffortTracker("eoa", pos_thresholds=[5.0, 15.0], joint_name="gripper"),
    }

    while sim.is_running():
        time.sleep(1 / 15)
        gamepad_state = gamepad.get_state()

        robot.precision_multiplier = 1.0 - 0.75 * gamepad_state.get("left_trigger_pulled", 0.0)
        adapter.controller_state = gamepad_state.copy()
        adapter._i += 1

        # Use back_button_pressed if it's pressed as fallback for select
        if gamepad_state.get("back_button_pressed", False):
            gamepad_state["select_button_pressed"] = True

        select_button_counter.step(gamepad_state)
        start_button_counter.step(gamepad_state)
        top_button_counter.step(gamepad_state)
        left_button_counter.step(gamepad_state)

        select_button_counter.trigger_on_tap(change_mapping)
        start_button_counter.trigger_on_tap(change_handedness)
        top_button_counter.trigger_on_tap(change_motion_profile)

        if use_head_joints:
            left_button_counter.trigger_on_tap(toggle_dex_switch)

            if dex_switch:
                if gamepad_state.get("bottom_pad_pressed"):
                    sim.move_by(Actuators["head_tilt"], 1 * 0.2 * robot.precision_multiplier)
                elif gamepad_state.get("top_pad_pressed"):
                    sim.move_by(Actuators["head_tilt"], -1 * 0.2 * robot.precision_multiplier)

                if gamepad_state.get("left_pad_pressed"):
                    sim.move_by(Actuators["head_pan"], 1 * 0.2 * robot.precision_multiplier)
                elif gamepad_state.get("right_pad_pressed"):
                    sim.move_by(Actuators["head_pan"], -1 * 0.2 * robot.precision_multiplier)

                adapter.controller_state["bottom_pad_pressed"] = False
                adapter.controller_state["top_pad_pressed"] = False
                adapter.controller_state["left_pad_pressed"] = False
                adapter.controller_state["right_pad_pressed"] = False

        robot.profile_multiplier = motion_profile.multiplier

        actuated_joints = mapping.do_motion(robot, adapter)

        status = sim.pull_status()
        if status.is_self_colliding:
            gamepad.vibrate_sequence(
                sequence_ms=[150, 100, 200],
                strong_magnitude=0.5,
                weak_magnitude=1.0,
                tag="collision",
                cooldown=2.0,
            )

        if actuated_joints and adapter.precision_mode:
            for joint_id, tracker in effort_trackers.items():
                direction = actuated_joints.get(joint_id, 0)
                is_actuated = direction != 0

                tracker.step(status, is_actuated, direction)

                def trigger_vibrate(effort, j_id=joint_id, t=tracker):
                    try:
                        # if j_id in ['joint_wrist_yaw', 'joint_wrist_pitch', 'joint_wrist_roll', 'gripper']:
                        #     return
                        abs_effort = abs(effort)
                        max_e = (
                            t.pos_thresholds[1] if t.last_direction >= 0 else t.neg_thresholds[1]
                        )
                        min_e = (
                            t.pos_thresholds[0] if t.last_direction >= 0 else t.neg_thresholds[0]
                        )
                        if max_e > min_e:
                            fraction = min(1.0, max(0.0, (abs_effort - min_e) / (max_e - min_e)))
                        else:
                            fraction = 1.0 if abs_effort >= min_e else 0.0

                        gamepad.vibrate(
                            duration_ms=100,
                            strong_magnitude=fraction,
                            weak_magnitude=fraction,
                            tag=f"effort_{j_id}",
                            cooldown=0.25,
                        )
                    except Exception as e:
                        logging.error(f"Got error {e}", exc_info=True)

                tracker.trigger_on_hold(0.1, trigger_vibrate)


class GripperHandedness(Enum):
    LEFT = 0
    RIGHT = 1

    def move_to(self, sim: StretchMujocoSimulator):
        """Moves the gripper to achieve this handedness"""
        print(f"Moving wrist to {self.name}")
        import math

        if self is GripperHandedness.RIGHT:
            yaw_to = 0.0
            pitch_to = 0.0
            roll_to = 0.0
            sim.move_to(Actuators["wrist_roll"], roll_to)
            sim.wait_until_at_setpoint(Actuators["wrist_roll"])
            sim.move_to(Actuators["wrist_pitch"], pitch_to)
            sim.wait_until_at_setpoint(Actuators["wrist_pitch"])
            sim.move_to(Actuators["wrist_yaw"], yaw_to)
            sim.wait_until_at_setpoint(Actuators["wrist_yaw"])
        elif self is GripperHandedness.LEFT:
            yaw_to = -math.pi
            pitch_to = -math.pi
            roll_to = math.pi
            sim.move_to(Actuators["wrist_yaw"], yaw_to)
            sim.wait_until_at_setpoint(Actuators["wrist_yaw"])
            sim.move_to(Actuators["wrist_pitch"], pitch_to)
            sim.wait_until_at_setpoint(Actuators["wrist_pitch"])
            sim.move_to(Actuators["wrist_roll"], roll_to)
            sim.wait_until_at_setpoint(Actuators["wrist_roll"])
        else:
            raise NotImplementedError(f"No move_to defined for {self}")


class MotionProfile(Enum):
    SLOW = 1
    DEFAULT = 2
    FAST = 3
    MAX = 4

    def cycle(self, is_forward: bool):
        index_offset = 1 if is_forward else -1
        members = list(type(self))
        index = members.index(self)
        return members[(index + index_offset) % len(members)]

    @property
    def multiplier(self):
        if self == MotionProfile.SLOW:
            return 0.5
        elif self == MotionProfile.DEFAULT:
            return 1.5
        elif self == MotionProfile.FAST:
            return 2.5
        elif self == MotionProfile.MAX:
            return 3.0


class ControlMapping(Enum):
    OMNIBASE = 1
    MANIPULATION = 2

    def cycle(self, is_forward: bool):
        index_offset = 1 if is_forward else -1
        members = list(type(self))
        index = members.index(self)
        return members[(index + index_offset) % len(members)]

    def do_motion(self, robot, gamepad_teleop):
        if self == ControlMapping.OMNIBASE:
            return self._map_omnibase(robot, gamepad_teleop)
        elif self == ControlMapping.MANIPULATION:
            return self._map_manipulation(robot, gamepad_teleop)
        else:
            raise NotImplementedError(f"No controls callback for {self}")

    def _map_omnibase(self, robot, gamepad_teleop):
        dxl_zero_vel_set_division_factor = 3
        actuated_joints = {}
        if gamepad_teleop.use_devices.get("eoa"):
            if gamepad_teleop.controller_state.get("right_shoulder_button_pressed"):
                gamepad_teleop.wrist_yaw_command.command_button_to_motion(-1, robot)
                actuated_joints["joint_wrist_yaw"] = 1
            elif gamepad_teleop.controller_state.get("left_shoulder_button_pressed"):
                gamepad_teleop.wrist_yaw_command.command_button_to_motion(1, robot)
                actuated_joints["joint_wrist_yaw"] = -1
            else:
                if gamepad_teleop._i % dxl_zero_vel_set_division_factor == 0:
                    gamepad_teleop.wrist_yaw_command.stop_motion(robot)
            if gamepad_teleop.controller_state.get("top_pad_pressed"):
                cmd = 1 if gamepad_teleop.gripper_handedness is GripperHandedness.RIGHT else -1
                gamepad_teleop.wrist_pitch_command.command_button_to_motion(cmd, robot)
                actuated_joints["joint_wrist_pitch"] = cmd
            elif gamepad_teleop.controller_state.get("bottom_pad_pressed"):
                cmd = -1 if gamepad_teleop.gripper_handedness is GripperHandedness.RIGHT else 1
                gamepad_teleop.wrist_pitch_command.command_button_to_motion(cmd, robot)
                actuated_joints["joint_wrist_pitch"] = cmd
            else:
                if gamepad_teleop._i % dxl_zero_vel_set_division_factor == 0:
                    gamepad_teleop.wrist_pitch_command.stop_motion(robot)
            if gamepad_teleop.controller_state.get("left_pad_pressed"):
                gamepad_teleop.wrist_roll_command.command_button_to_motion(1, robot)
                actuated_joints["joint_wrist_roll"] = -1
            elif gamepad_teleop.controller_state.get("right_pad_pressed"):
                gamepad_teleop.wrist_roll_command.command_button_to_motion(-1, robot)
                actuated_joints["joint_wrist_roll"] = 1
            else:
                if gamepad_teleop._i % dxl_zero_vel_set_division_factor == 0:
                    gamepad_teleop.wrist_roll_command.stop_motion(robot)

        if gamepad_teleop.use_devices.get("arm"):
            cmd = (
                gamepad_teleop.controller_state.get("right_stick_x", 0)
                if gamepad_teleop.use_arm_lift_mode
                else 0
            )
            gamepad_teleop.arm_command.command_stick_to_motion(cmd, robot)
            if abs(cmd) > 0.1:
                actuated_joints["arm"] = cmd
        if gamepad_teleop.use_devices.get("lift"):
            cmd = (
                gamepad_teleop.controller_state.get("right_stick_y", 0)
                if gamepad_teleop.use_arm_lift_mode
                else 0
            )
            gamepad_teleop.lift_command.command_stick_to_motion(cmd, robot)
            if abs(cmd) > 0.1:
                actuated_joints["lift"] = cmd
        if gamepad_teleop.use_devices.get("base"):
            cmd_y = gamepad_teleop.controller_state.get("left_stick_y", 0)
            cmd_x = -gamepad_teleop.controller_state.get("left_stick_x", 0)
            cmd_t = (
                -gamepad_teleop.controller_state.get("right_stick_x", 0)
                if not gamepad_teleop.use_arm_lift_mode
                else 0
            )
            gamepad_teleop.base_command.command_stick_to_motion(cmd_y, cmd_x, cmd_t, robot)
            if abs(cmd_y) > 0.1 or abs(cmd_x) > 0.1 or abs(cmd_t) > 0.1:
                actuated_joints["base"] = cmd_x + cmd_y + cmd_t

        if gamepad_teleop.use_devices.get("gripper"):
            if gamepad_teleop.controller_state.get("right_button_pressed"):
                gamepad_teleop.gripper.open_gripper(robot)
                actuated_joints[gamepad_teleop.gripper.name] = 1
            elif gamepad_teleop.controller_state.get("bottom_button_pressed"):
                gamepad_teleop.gripper.close_gripper(robot)
                actuated_joints[gamepad_teleop.gripper.name] = -1
            else:
                gamepad_teleop.gripper.stop_gripper(robot)

        return actuated_joints

    def _map_manipulation(self, robot, gamepad_teleop):
        gamepad_teleop.precision_mode = (
            gamepad_teleop.controller_state.get("left_trigger_pulled", 0) > 0.9
        )
        gamepad_teleop.use_arm_lift_mode = (
            gamepad_teleop.controller_state.get("right_trigger_pulled", 0) > 0.9
        )

        dxl_zero_vel_set_division_factor = 3

        right_stick_x = gamepad_teleop.controller_state.get("right_stick_x", 0)
        right_stick_y = gamepad_teleop.controller_state.get("right_stick_y", 0)

        actuated_joints = {}
        if gamepad_teleop.use_devices.get("lift"):
            if gamepad_teleop.controller_state.get("top_pad_pressed"):
                gamepad_teleop.lift_command.command_button_to_motion(0.4, robot)
                actuated_joints["lift"] = 0.4
            elif gamepad_teleop.controller_state.get("bottom_pad_pressed"):
                gamepad_teleop.lift_command.command_button_to_motion(-0.4, robot)
                actuated_joints["lift"] = -0.4
            else:
                if gamepad_teleop._i % dxl_zero_vel_set_division_factor == 0:
                    gamepad_teleop.lift_command.stop_motion(robot)

        if gamepad_teleop.use_devices.get("eoa") and gamepad_teleop.use_arm_lift_mode:
            gamepad_teleop.base_command.stop_motion(robot)

            if abs(right_stick_x) > 0.1:
                gamepad_teleop.wrist_yaw_command.command_stick_to_motion(right_stick_x, robot)
                actuated_joints["joint_wrist_yaw"] = right_stick_x

            if abs(right_stick_y) > 0.1:
                handedness_inversion = (
                    -1 if gamepad_teleop.gripper_handedness is GripperHandedness.LEFT else 1
                )
                cmd = handedness_inversion * right_stick_y
                gamepad_teleop.wrist_pitch_command.command_stick_to_motion(cmd, robot)
                actuated_joints["joint_wrist_pitch"] = right_stick_y

            if gamepad_teleop.controller_state.get("left_pad_pressed"):
                gamepad_teleop.wrist_roll_command.command_button_to_motion(1, robot)
                actuated_joints["joint_wrist_roll"] = -1
            elif gamepad_teleop.controller_state.get("right_pad_pressed"):
                gamepad_teleop.wrist_roll_command.command_button_to_motion(-1, robot)
                actuated_joints["joint_wrist_roll"] = 1
            else:
                if gamepad_teleop._i % dxl_zero_vel_set_division_factor == 0:
                    gamepad_teleop.wrist_roll_command.stop_motion(robot)

            if gamepad_teleop.use_devices.get("arm"):
                cmd = (
                    gamepad_teleop.controller_state.get("left_stick_y", 0)
                    if gamepad_teleop.use_arm_lift_mode
                    else 0
                )
                gamepad_teleop.arm_command.command_stick_to_motion(cmd, robot)
                if abs(cmd) > 0.1:
                    actuated_joints["arm"] = cmd

        else:
            if gamepad_teleop.use_devices.get("arm"):
                gamepad_teleop.arm_command.stop_motion(robot)
            if gamepad_teleop.use_devices.get("eoa"):
                gamepad_teleop.wrist_yaw_command.stop_motion(robot)
                gamepad_teleop.wrist_pitch_command.stop_motion(robot)
                gamepad_teleop.wrist_roll_command.stop_motion(robot)

            if gamepad_teleop.use_devices.get("base"):
                cmd_y = (
                    gamepad_teleop.controller_state.get("left_stick_y", 0)
                    if not gamepad_teleop.use_arm_lift_mode
                    else 0
                )
                cmd_x = (
                    -gamepad_teleop.controller_state.get("left_stick_x", 0)
                    if not gamepad_teleop.use_arm_lift_mode
                    else 0
                )
                cmd_t = (
                    -gamepad_teleop.controller_state.get("right_stick_x", 0)
                    if not gamepad_teleop.use_arm_lift_mode
                    else 0
                )
                gamepad_teleop.base_command.command_stick_to_motion(cmd_y, cmd_x, cmd_t, robot)
                if abs(cmd_y) > 0.1 or abs(cmd_x) > 0.1 or abs(cmd_t) > 0.1:
                    actuated_joints["base"] = cmd_x + cmd_y + cmd_t

        if gamepad_teleop.use_devices.get("gripper"):
            if gamepad_teleop.controller_state.get("right_button_pressed"):
                gamepad_teleop.gripper.open_gripper(robot)
                actuated_joints[gamepad_teleop.gripper.name] = 1
            elif gamepad_teleop.controller_state.get("bottom_button_pressed"):
                gamepad_teleop.gripper.close_gripper(robot)
                actuated_joints[gamepad_teleop.gripper.name] = -1
            else:
                gamepad_teleop.gripper.stop_gripper(robot)

        return actuated_joints


class MockCommand:
    def __init__(self, sim, actuator_name, scale=1.0):
        self.sim = sim
        self.actuator_name = actuator_name
        self.scale = scale
        self.name = actuator_name

    def command_button_to_motion(self, val, robot):
        self.sim.move_by(
            Actuators[self.actuator_name],
            val * self.scale * robot.precision_multiplier * robot.profile_multiplier,
        )

    def command_stick_to_motion(self, val, robot):
        self.sim.move_by(
            Actuators[self.actuator_name],
            val * self.scale * robot.precision_multiplier * robot.profile_multiplier,
        )

    def stop_motion(self, robot):
        pass


class MockGripperCommand:
    def __init__(self, sim):
        self.sim = sim
        self.name = "gripper"

    def open_gripper(self, robot):
        val = 1
        if isinstance(self.sim, Stretch4MujocoSimulator):
            val = 5.0
        else:
            val = 0.07
        self.sim.move_by(Actuators["gripper"], val * robot.precision_multiplier)

    def close_gripper(self, robot):
        val = -1
        if isinstance(self.sim, Stretch4MujocoSimulator):
            val = -5.0
        else:
            val = -0.07
        self.sim.move_by(Actuators["gripper"], val * robot.precision_multiplier)

    def stop_gripper(self, robot):
        pass


class MockBaseCommand:
    def __init__(self, sim):
        self.sim = sim
        self.name = "base"

    def command_stick_to_motion(self, cmd_y, cmd_x, cmd_t, robot):
        if abs(cmd_x) < 0.001:
            cmd_x = 0
        if abs(cmd_y) < 0.001:
            cmd_y = 0
        if abs(cmd_t) < 0.001:
            cmd_t = 0

        velocity = 1.0  # m/s
        angular_velocity = 3.0  # rad/s

        v_x_linear = cmd_y * velocity * robot.precision_multiplier * robot.profile_multiplier
        v_y_linear = cmd_x * velocity * robot.precision_multiplier * robot.profile_multiplier
        omega = cmd_t * angular_velocity * robot.precision_multiplier * robot.profile_multiplier

        if isinstance(self.sim, Stretch4MujocoSimulator):
            self.sim.set_base_velocity(v_x=v_x_linear, v_y=v_y_linear, omega=omega)
        else:
            self.sim.set_base_velocity(v_y_linear, omega)

    def stop_motion(self, robot):
        if isinstance(self.sim, Stretch4MujocoSimulator):
            self.sim.set_base_velocity(0.0, 0.0, 0.0)
        else:
            self.sim.set_base_velocity(0.0, 0.0)


class GamepadTeleopAdapter:
    def __init__(self, sim):
        self._i = 0
        self.use_devices = {"eoa": True, "arm": True, "lift": True, "base": True, "gripper": True}
        self.gripper_handedness = GripperHandedness.RIGHT
        self.use_arm_lift_mode = False
        self.precision_mode = False
        self.controller_state = {}

        self.wrist_yaw_command = MockCommand(sim, "wrist_yaw", 0.2)
        self.wrist_pitch_command = MockCommand(sim, "wrist_pitch", 0.2)
        self.wrist_roll_command = MockCommand(sim, "wrist_roll", 0.2)
        self.arm_command = MockCommand(sim, "arm", 0.05)
        self.lift_command = MockCommand(sim, "lift", 0.1)
        self.base_command = MockBaseCommand(sim)
        self.gripper = MockGripperCommand(sim)


class MockRobot:
    def __init__(self):
        self.precision_multiplier = 1.0
        self.profile_multiplier = 1.0


if __name__ == "__main__":
    main()
