import atexit
from multiprocessing import Lock, Manager, Process

import multiprocessing
import platform
import signal
import sys
import threading
import time

import click
import numpy as np
from mujoco._structs import MjModel

from stretch4_mujoco.datamodels.status_stretch_camera import StatusStretchCameras
from stretch4_mujoco.datamodels.status_stretch_joints import StatusStretchJoints
from stretch4_mujoco.datamodels.status_stretch_sensors import StatusStretchSensors
from stretch4_mujoco.enums.actuators import Actuators
from stretch4_mujoco.enums.stretch_cameras import StretchCameras
from stretch4_mujoco.mujoco_server import MujocoServer, MujocoServerProxies
from stretch4_mujoco.mujoco_server_managed import MujocoServerManaged
from stretch4_mujoco.mujoco_server_passive import MujocoServerPassive
from stretch4_mujoco.datamodels.status_command import (
    CommandBaseVelocity,
    CommandCoordinateFrameArrowsViz,
    CommandKeyframe,
    CommandMove,
    StatusCommand,
)
import stretch4_mujoco.utils as utils
from stretch4_mujoco.utils import require_connection, block_until_check_succeeds


class StretchMujocoSimulator:
    """
    Stretch Mujoco Simulator class for interfacing with the Mujoco Server.

    Calling `start()` will spawn a new process that runs `MujocoServer` and the simulator.

    You can specify `start(headless=True)` to run the simulation without a GUI.

    Data from the MujocoServer is sent to StretchMujocoSimulator using proxies.

    Use `pull_status()` and `pull_camera_data()` to access simulation data.
    """

    def __init__(
        self,
        scene_xml_path: str | None = None,
        model: MjModel | None = None,
        camera_hz: float = 30,
        cameras_to_use: list[StretchCameras] = [],
        start_translation: list | None = None,
        start_rotation_quat: list | None = None,
    ) -> None:
        self.scene_xml_path = scene_xml_path
        self.model = model
        self.camera_hz = camera_hz
        self.urdf_model = utils.URDFmodel(self.get_urdf_path())
        self._server_process = None
        self._cameras_to_use = cameras_to_use
        self._start_translation = start_translation
        self._start_rotation_quat = start_rotation_quat

        self.is_stop_called = False

        self._manager = Manager()
        self._stop_mujoco_process_event = self._manager.Event()

        self.data_proxies = MujocoServerProxies.default(self._manager)

        self._command_lock = Lock()

        self.base = BaseSubsystem(self)
        self.omnibase = self.base
        self.arm = JointSubsystem(self, 'arm', Actuators.arm)
        self.lift = JointSubsystem(self, 'lift', Actuators.lift)
        self.end_of_arm = EndOfArmSubsystem(self)
        
        # Stretch 4 does not have head
        if self.__class__.__name__ != "Stretch4MujocoSimulator":
            self.head = HeadSubsystem(self)

    @staticmethod
    def get_scene_xml_path() -> str:
        """
        Returns the default scene XML path for the Stretch Mujoco Simulator.
        """
        return str(utils.models_path / "scene.xml")

    @staticmethod
    def get_robot_xml_path() -> str:
        """
        Returns the default robot XML path for the Stretch Mujoco Simulator.
        """
        return utils.get_absolute_path_stretch_xml(
            str(utils.models_path / "stretch_3" / "stretch.xml")
        )

    @staticmethod
    def get_urdf_path() -> str:
        pkg_path = utils.get_urdf_package_path("stretch_urdf")
        model_name = "SE3"  # RE1V0, RE2V0, SE3
        tool_name = "eoa_wrist_dw3_tool_sg3"  # eoa_wrist_dw3_tool_sg3, tool_stretch_gripper, etc

        return str(pkg_path / model_name / utils.get_urdf_file_name(model_name, tool_name))

    @staticmethod
    def get_all_cameras() -> list[StretchCameras]:
        return StretchCameras.all_stretch3()

    @staticmethod
    def get_rgb_cameras() -> list[StretchCameras]:
        return StretchCameras.rgb_stretch3()

    def start(
        self,
        show_viewer_ui: bool = False,
        headless: bool = False,
        use_passive_viewer: bool = True,
        viewer_track_body: str | None = None,
        viewer_look_at_body: str | None = None,
    ) -> None:
        """
        Start the simulator

        Args:
            show_viewer_ui: bool, whether to show the Mujoco viewer UI
            headless: bool, whether to run the simulation in headless mode
            use_passive_viewer: bool, to use the passive or managed mujoco UI viewer.
            viewer_track_body: name of a body for the viewer camera to follow,
                e.g. "stretch4". Worth setting in a large scene, where Mujoco's
                default framing of the whole model leaves the robot a few pixels
                across. Only the passive viewer honours it; the managed viewer
                gives no handle to configure. You can still orbit and zoom
                normally afterwards.
            viewer_look_at_body: name of a body to aim the viewer's *free* camera
                at once, at startup. Same fix as viewer_track_body for the same
                problem, but the camera then stays where the mouse leaves it
                instead of following the body -- panning included, which a
                tracking camera overrides. Ignored if viewer_track_body is set.
        """
        self.is_stop_called = False

        mujoco_server = MujocoServer  # Headless

        if not headless:
            mujoco_server = MujocoServerPassive if use_passive_viewer else MujocoServerManaged

        if platform.system() == "Darwin" and mujoco_server is MujocoServerPassive:
            # On a mac, the process for MujocoServerPassive needs to be started with mjpython
            mjpython_path = sys.executable.replace("bin/python3", "bin/mjpython").replace(
                "bin/python", "bin/mjpython"
            )
            print(f"{mjpython_path=}")
            multiprocessing.set_executable(mjpython_path)

        multiprocessing.set_start_method("spawn", force=True)

        self._server_process = Process(
            target=mujoco_server.launch_server,
            name="MujocoProcess",
            args=(
                self.scene_xml_path,
                self.model,
                self.camera_hz,
                show_viewer_ui,
                self._stop_mujoco_process_event,
                self.data_proxies,
                self._cameras_to_use,
                self._start_translation,
                self._start_rotation_quat,
                viewer_track_body,
                viewer_look_at_body,
            ),
            daemon=False,  # We're gonna handle terminating this in stop_mujoco_process()
        )
        self._server_process.start()

        # Handle stopping, in all its various ways:
        signal.signal(signal.SIGTERM, lambda num, sig: self.stop())
        signal.signal(signal.SIGINT, lambda num, sig: self.stop())
        atexit.register(self.stop)

        click.secho("Starting Stretch Mujoco Simulator...", fg="green")
        while self.pull_status().time == 0 or self.pull_camera_data().time == 0:
            time.sleep(1)
            click.secho("Still waiting to connect to the Mujoco Simulator.", fg="yellow")

            if not self.is_running():
                click.secho("The simulator is not running anymore, quitting..", fg="yellow")
                return

        click.secho("The Mujoco Simulator is connected.", fg="green")

        self.home()

    def stop(self) -> None:
        """
        This is called at exit to gracefully terminate the simulation and the Mujoco Process, and their many threads.

        Fingers-crossed we get a SIGTERM, and not a SIGKILL..
        """
        if self.is_stop_called:
            return

        self.is_stop_called = True

        try:
            simulation_time_message = self.data_proxies.get_status().time
            simulation_time_message = f" simulated runtime= {simulation_time_message:.1f}s"
        except:
            simulation_time_message = ""

        click.secho(
            f"Stopping Stretch Mujoco Simulator...{simulation_time_message}",
            fg="red",
        )

        self.stop_mujoco_process()

        # We're going to try to wait for threads to end. They might not gracefully stop before hitting an exception. Race conditions are rampant.
        # For example, the main thread or a thread may not be checking `sim.is_running()` and is oblivious that it should stop. Nothing we can do to stop it except sigkill.
        active_threads = threading.enumerate()
        for index, thread in enumerate(active_threads):
            if (
                thread != threading.current_thread()
                and thread != threading.main_thread()
                and not isinstance(thread, threading._DummyThread)
            ):
                click.secho(
                    f"Stopping thread {index}/{len(active_threads)-1}.",
                    fg="yellow",
                )
                thread.join(timeout=10.0)
                if thread.is_alive():
                    click.secho(
                        f"{thread.name} is not terminating. Make sure to check 'sim.is_running()' in threading loops.",
                        fg="red",
                    )

        click.secho(
            f"The Stretch Mujoco Simulator has ended. Good-bye!",
            fg="red",
        )

    def stop_mujoco_process(self):

        if self._server_process and not self._server_process.is_alive():
            click.secho(
                f"The Mujoco process has already terminated.",
                fg="red",
            )
            return

        click.secho(
            f"Sending signal to stop the Mujoco process...",
            fg="red",
        )

        # Wait until the main control loop ends before sending this stop event.
        self._stop_mujoco_process_event.set()
        if self._server_process:
            # self._server_process.terminate() # ask it nicely.
            self._server_process.join()

        click.secho(
            f"The Mujoco process has ended.",
            fg="red",
        )

    @require_connection
    def home(self) -> None:
        """
        Move the robot to home position
        """
        with self._command_lock:
            self.data_proxies.set_command(
                StatusCommand(keyframe=CommandKeyframe(name="home", trigger=True))
            )
        self.wait_while_is_moving(Actuators.lift)

    @require_connection
    def stow(self) -> None:
        """
        Move the robot to stow position
        """
        with self._command_lock:
            self.data_proxies.set_command(
                StatusCommand(keyframe=CommandKeyframe(name="stow", trigger=True))
            )

        self.wait_while_is_moving(Actuators.wrist_pitch)

    def is_reached_set_position(self, actuator: str | Actuators, position_tolerance: float = 0.05):
        """
        Checks if the joint has reached a previously commanded location.

        Only listens to the `move_to` command.
        """
        if isinstance(actuator, str):
            actuator = Actuators[actuator]

        if actuator in [
            Actuators.base_rotate,
            Actuators.base_translate,
            Actuators.left_wheel_vel,
            Actuators.right_wheel_vel,
        ]:
            raise NotImplementedError(f"Check joint reached is not supported for {actuator}.")

        move_command = self.data_proxies.get_command().move_to.get(actuator.name)

        if not move_command:
            click.secho(
                "Warning: Position check requested, but the joint was not commanded to move.",
                fg="yellow",
            )
            return True

        set_position = move_command.pos

        current_position = actuator.get_position(self.pull_status())

        return bool(np.isclose(current_position, set_position, atol=position_tolerance))

    def wait_until_at_setpoint(
        self, actuator: str | Actuators, timeout: float = 5.0, position_tolerance: float = 0.05
    ):
        """Blocks until the actuator reaches its previously set point."""
        if isinstance(actuator, str):
            actuator = Actuators[actuator]

        move_command = self.data_proxies.get_command().move_to.get(actuator.name)

        if not move_command:
            return True

        if not block_until_check_succeeds(
            wait_timeout=timeout,
            check=lambda: self.is_reached_set_position(
                actuator=actuator, position_tolerance=position_tolerance
            )
            == True,
            is_alive=self.is_running,
            time_fn=lambda: self.pull_status().time,
        ):
            pos = move_command.pos
            actual = actuator.get_position(self.pull_status())
            error = pos - actual
            click.secho(
                f"Timeout: Joint {actuator.name} did not reach {pos}. Actual: {actual:.4f} Diff: {error*100:.4f}cm",
                fg="red",
            )
            return False
        return True

    def wait_command(self, timeout: float = 15.0, check_interval: float = 0.1, position_tolerance: float = 0.001) -> bool:
        """
        Pause program execution until all motion is complete.
        This loops and checks the positions of lift, arm, base, and end_of_arm.
        When all positions remain stable (change < position_tolerance) over check_interval,
        the motion is considered complete.
        """
        # Determine the actuators to watch
        actuators = [
            Actuators.lift,
            Actuators.arm,
            Actuators.wrist_yaw,
            Actuators.gripper,
            Actuators.base_translate,
            Actuators.base_translate_y,
            Actuators.base_rotate,
        ]
        if hasattr(self, "end_of_arm"):
            if hasattr(self.end_of_arm, "wrist_pitch"):
                actuators.append(Actuators.wrist_pitch)
            if hasattr(self.end_of_arm, "wrist_roll"):
                actuators.append(Actuators.wrist_roll)
        if hasattr(self, "head"):
            actuators.append(Actuators.head_pan)
            actuators.append(Actuators.head_tilt)

        # Helper to get all current positions
        def get_all_positions():
            positions = {}
            status = self.pull_status()
            for act in actuators:
                if act in [Actuators.base_translate, Actuators.base_translate_y, Actuators.base_rotate]:
                    # Relative or absolute base position
                    rel_pos = act.get_position_relative(status)
                    if act == Actuators.base_translate:
                        positions[act] = rel_pos[0]
                    elif act == Actuators.base_translate_y:
                        positions[act] = rel_pos[1]
                    elif act == Actuators.base_rotate:
                        positions[act] = rel_pos[2]
                else:
                    positions[act] = act.get_position(status)
            return positions

        try:
            start_time = self.pull_status().time
            last_positions = get_all_positions()
        except Exception:
            # If server isn't ready or status pull fails initially, wait and retry
            time.sleep(check_interval)
            start_time = self.pull_status().time
            last_positions = get_all_positions()
        
        while self.pull_status().time - start_time < timeout:
            if not self.is_running():
                break
            time.sleep(check_interval)
            try:
                current_positions = get_all_positions()
            except Exception:
                continue
            
            # Check if any position changed significantly
            any_moving = False
            for act in actuators:
                last_p = last_positions[act]
                curr_p = current_positions[act]
                if not np.isclose(curr_p, last_p, atol=position_tolerance):
                    any_moving = True
                    break
            
            # Also check server-side active flags for base translation/rotation
            status = self.pull_status()
            if status.base.active_translate_x or status.base.active_translate_y or status.base.active_rotate:
                any_moving = True

            # ...and for joints whose motion profile has not finished. A
            # rate-limited joint leaves the position checks above unmoved for the
            # first tick or two of a move, while it accelerates from rest.
            if status.actuators_in_motion:
                any_moving = True
                    
            if not any_moving:
                # All joints are stable!
                return True
                
            last_positions = current_positions
            
        return False

    _last_movement_positions: dict[Actuators, float | tuple[float, float, float]] = {}

    def _has_move_in_flight(self, actuator: Actuators, status: StatusStretchJoints) -> bool:
        """Whether the server still has a rate-limited move in flight for `actuator`.

        Position stability on its own no longer means "stopped": a rate-limited
        joint accelerates from rest, so for the first tick or two of a move it has
        not measurably left where it started. `status.actuators_in_motion` is the
        server saying otherwise.
        """
        in_motion = set(status.actuators_in_motion)
        if not in_motion:
            return False
        if actuator == Actuators.gripper:
            # Commanded as an aperture, driven as two fingers on Stretch 4.
            return bool(
                in_motion
                & {
                    Actuators.gripper.name,
                    Actuators.gripper_left_finger.name,
                    Actuators.gripper_right_finger.name,
                }
            )
        return actuator.name in in_motion

    def wait_while_is_moving(
        self,
        actuator: str | Actuators,
        timeout: float | None = 5.0,
        check_interval: float = 0.1,
        position_tolerance: float = 0.0005,
    ):
        """
        Checks position after a delay, and blocks if position has changed.
        If `timeout` is None, will block indefinitely.
        """
        if isinstance(actuator, str):
            actuator = Actuators[actuator]

        def check_if_moved():
            """Checks movement, returns True if movement is detected."""
            time.sleep(check_interval)

            if self._has_move_in_flight(actuator, self.pull_status()):
                return True

            if actuator in [
                Actuators.left_wheel_vel,
                Actuators.right_wheel_vel,
                Actuators.base_rotate,
                Actuators.base_translate,
                Actuators.base_translate_y,
            ]:
                current_position = actuator.get_position_relative(self.pull_status())
                if actuator == Actuators.left_wheel_vel or actuator == Actuators.base_translate:
                    current_position = current_position[0]
                elif actuator == Actuators.base_translate_y:
                    current_position = current_position[1]
                elif actuator == Actuators.right_wheel_vel:
                    current_position = current_position[1]
                elif actuator == Actuators.base_rotate:
                    current_position = current_position[2]
            else:
                current_position = actuator.get_position(self.pull_status())

            if not actuator in self._last_movement_positions:
                self._last_movement_positions[actuator] = current_position
                return True

            last_position = self._last_movement_positions[actuator]

            is_moved = not np.isclose(current_position, last_position, atol=position_tolerance)

            self._last_movement_positions[actuator] = current_position

            return is_moved

        if not block_until_check_succeeds(
            wait_timeout=timeout,
            check=lambda: check_if_moved() == False,
            is_alive=self.is_running,
            time_fn=lambda: self.pull_status().time,
        ):
            if timeout is not None:
                click.secho(
                    f"Timeout: Joint {actuator.name} is still moving after {timeout}.",
                    fg="red",
                )
            return False
        return True

    @require_connection
    def _move_to(self, actuator: str | Actuators, pos: float) -> None:
        """
        Move the actuator to an absolute position.
        Args:
            actuator: string name of the actuator or Actuator enum instance
            pos: float, absolute position goal

        Use `wait_until_at_setpoint()` or `wait_while_is_moving()` to block until the joint reaches its location.
        """
        if isinstance(actuator, str):
            actuator = Actuators[actuator]

        if actuator in [
            Actuators.left_wheel_vel,
            Actuators.right_wheel_vel,
            Actuators.base_rotate,
            Actuators.base_translate,
        ]:
            raise Exception(
                f"Cannot set an absolute position for a continuous joint {actuator.name}"
            )

        with self._command_lock:
            command = self.data_proxies.get_command()
            command.set_move_to(CommandMove(actuator_name=actuator.name, pos=pos, trigger=True))

            self.data_proxies.set_command(command)

    @require_connection
    def _move_by(self, actuator: str | Actuators, pos: float):
        """
        Move the actuator by a relative amount.
        Args:
            actuator: string name of the actuator or Actuator enum instance
            pos: float, position to increment by

        Use `wait_until_at_setpoint()` or `wait_while_is_moving()` to block until the joint reaches its location.
        """
        if isinstance(actuator, str):
            actuator = Actuators[actuator]

        if actuator in [Actuators.left_wheel_vel, Actuators.right_wheel_vel]:
            click.secho(
                f"Cannot set a position for a velocity joint {actuator.name}",
                fg="red",
            )
            raise Exception(
                f"Cannot set an absolute position for a continuous joint {actuator.name}"
            )

        with self._command_lock:
            command = self.data_proxies.get_command()

            command.set_move_by(
                # We set the pos here, and not new_position, because this relative motion math is handled by mujoco_server:
                CommandMove(actuator_name=actuator.name, pos=pos, trigger=True)
            )

            self.data_proxies.set_command(command)

    @require_connection
    def _set_joint_velocity(self, actuator: str | Actuators, v_m: float):
        """
        Set continuous velocity for a joint.
        """
        if isinstance(actuator, str):
            actuator = Actuators[actuator]

        with self._command_lock:
            command = self.data_proxies.get_command()
            command.set_joint_velocity(actuator.name, v_m)
            self.data_proxies.set_command(command)

    @require_connection
    def _set_base_velocity(self, v_linear: float, omega: float) -> None:
        """
        Set the base velocity of the robot
        Args:
            v_linear: float, linear velocity
            omega: float, angular velocity
        """

        with self._command_lock:
            command = self.data_proxies.get_command()
            command.set_base_velocity(
                CommandBaseVelocity(v_x=v_linear, v_y=0, omega=omega, trigger=True)
            )

            self.data_proxies.set_command(command)

    @require_connection
    def add_world_frame(
        self,
        position: tuple[float, float, float],
        rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """
        Add a world frame to the simulator for visualization.
        Args:
            position: tuple of (x, y, z) coordinates in the world frame
            rotation: tuple of (x, y, z) angle in radians for the rotation around each axis
        """
        with self._command_lock:
            command = self.data_proxies.get_command()
            command.coordinate_frame_arrows_viz.append(
                CommandCoordinateFrameArrowsViz(position=position, rotation=rotation, trigger=True)
            )
            self.data_proxies.set_command(command)

    @require_connection
    def get_base_pose(self):
        """Get the se(2) base pose: x, y, and theta"""
        status = self.pull_status()
        return (status.base.x, status.base.y, status.base.theta)

    @require_connection
    def get_ee_pose(self) -> np.ndarray:
        return self.get_link_pose("link_grasp_center")

    @require_connection
    def get_link_pose(self, link_name: str) -> np.ndarray:
        """Pose of link in world frame"""
        status = self.pull_status()
        cfg = {
            "wrist_yaw": status.wrist_yaw.pos,
            "wrist_pitch": status.wrist_pitch.pos,
            "wrist_roll": status.wrist_roll.pos,
            "lift": status.lift.pos,
            "arm": status.arm.pos,
            "head_pan": status.head_pan.pos,
            "head_tilt": status.head_tilt.pos,
        }
        transform = self.urdf_model.get_transform(cfg, link_name)
        base_xyt = self.get_base_pose()
        base_4x4 = np.eye(4)
        base_4x4[:3, :3] = utils.Rz(base_xyt[2])
        base_4x4[:2, 3] = base_xyt[:2]
        world_coord = np.matmul(base_4x4, transform)
        return world_coord

    @require_connection
    def pull_camera_data(self) -> StatusStretchCameras:
        """
        Pull camera data from the simulator and return as a StatusStretchCameras
        """
        return self.data_proxies.get_cameras()

    @require_connection
    def pull_sensor_data(self) -> StatusStretchSensors:
        """
        Pull sensor data from the simulator and return as a StatusStretchSensors
        """
        return self.data_proxies.get_sensors()

    @require_connection
    def pull_status(self) -> StatusStretchJoints:
        """
        Pull robot joint states from the simulator and return as a StatusStretchJoints
        """
        return self.data_proxies.get_status()

    @require_connection
    def pull_joint_limits(self) -> dict[Actuators, tuple[float, float]]:
        """
        Pull robot joint limuts from the simulator and return as a dict
        """
        return self.data_proxies.get_joint_limits()

    def is_mujoco_process_dead_or_stopevent_triggered(self):
        return (
            self._server_process is None
            or not self._server_process.is_alive()
            or self._stop_mujoco_process_event.is_set()
        )

    def is_running(self) -> bool:
        """
        Check if the simulator and mujoco are running, or if the stopevent signal has been triggered.

        Side-effect here is that if the mujoco process is terminated or the stopevent is triggered, `self.stop()` is called.
        """
        if self.is_mujoco_process_dead_or_stopevent_triggered():
            # Send the signal to stop the program:
            self.stop()
            return False

        return not self.is_stop_called


class BaseSubsystem:
    def __init__(self, sim: "StretchMujocoSimulator"):
        self._sim = sim

    @property
    def status(self):
        return self._sim.pull_status().base

    def translate_by(self, x_m, y_m=0.0, v_m=None, a_m=None):
        if x_m != 0.0:
            self._sim._move_by(Actuators.base_translate, x_m)
        if y_m != 0.0:
            self._sim._move_by(Actuators.base_translate_y, y_m)

    def rotate_by(self, w_r, v_r=None, a_r=None):
        self._sim._move_by(Actuators.base_rotate, w_r)

    def set_velocity(self, vx_m, vy_m, w_r, a_m=None, a_r=None):
        if self._sim.__class__.__name__ == "Stretch4MujocoSimulator":
            self._sim._set_base_velocity(vx_m, vy_m, w_r)
        else:
            self._sim._set_base_velocity(vx_m, w_r)



class JointSubsystem:
    def __init__(self, sim: "StretchMujocoSimulator", name: str, actuator: Actuators):
        self._sim = sim
        self._name = name
        self._actuator = actuator

    @property
    def status(self):
        return getattr(self._sim.pull_status(), self._name)

    def move_to(self, x_m, v_m=None, a_m=None, stiffness=None, req_calibration=True, contact_sensitivity_pos=None, contact_sensitivity_neg=None):
        self._sim._move_to(self._actuator, x_m)

    def move_by(self, x_m, v_m=None, a_m=None, stiffness=None, req_calibration=True, contact_sensitivity_pos=None, contact_sensitivity_neg=None):
        self._sim._move_by(self._actuator, x_m)

    def set_velocity(self, v_m, a_m=None, stiffness=None, req_calibration=True, contact_sensitivity_pos=None, contact_sensitivity_neg=None):
        self._sim._set_joint_velocity(self._actuator, v_m)


class EndOfArmSubsystem:
    def __init__(self, sim: "StretchMujocoSimulator"):
        self._sim = sim
        self.wrist_yaw = JointSubsystem(sim, 'wrist_yaw', Actuators.wrist_yaw)
        self.wrist_pitch = JointSubsystem(sim, 'wrist_pitch', Actuators.wrist_pitch)
        self.wrist_roll = JointSubsystem(sim, 'wrist_roll', Actuators.wrist_roll)
        self.stretch_gripper = JointSubsystem(sim, 'gripper', Actuators.gripper)
        self.parallel_gripper = JointSubsystem(sim, 'gripper', Actuators.gripper)


class HeadSubsystem:
    def __init__(self, sim: "StretchMujocoSimulator"):
        self._sim = sim
        self.head_pan = JointSubsystem(sim, 'head_pan', Actuators.head_pan)
        self.head_tilt = JointSubsystem(sim, 'head_tilt', Actuators.head_tilt)
