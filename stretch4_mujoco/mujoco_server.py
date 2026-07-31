import contextlib
from dataclasses import dataclass
from multiprocessing.managers import DictProxy, SyncManager
import os
import signal
import threading
import time
from typing import Callable

import click
import mujoco
import mujoco._functions
import mujoco._enums
import numpy as np
from mujoco._structs import MjData, MjModel
import mujoco._enums

from stretch4_mujoco.datamodels.status_stretch_camera import StatusStretchCameras
from stretch4_mujoco.datamodels.status_stretch_joints import StatusStretchJoints
from stretch4_mujoco.datamodels.status_stretch_sensors import StatusStretchSensors
from stretch4_mujoco.enums.actuators import Actuators
from stretch4_mujoco.enums.stretch_cameras import StretchCameras
import stretch4_mujoco.config as config
from stretch4_mujoco.enums.stretch_sensors import StretchSensors
from stretch4_mujoco.mujoco_server_camera_manager import (
    MujocoServerCameraManagerThreaded,
    MujocoServerCameraManagerSync,
)
from stretch4_mujoco.datamodels.status_command import CommandBaseVelocity, CommandMove, StatusCommand
from stretch4_mujoco.mujoco_server_sensor_manager import MujocoServerSensorManagerThreaded
import stretch4_mujoco.utils as utils
from stretch4_mujoco.utils import FpsCounter, change_start_pose, H0_from_driving_dir, inverse_3x3_matrix, rotation_3x3_matrix
from stretch4_mujoco.trapezoidal_profile import TrapezoidalProfile

@dataclass
class MujocoServerProxies:
    _command: "DictProxy[str, StatusCommand]"
    _status: "DictProxy[str, StatusStretchJoints]"
    _cameras: "DictProxy[str, StatusStretchCameras]"
    _sensors: "DictProxy[str, StatusStretchSensors]"
    _joint_limits: "DictProxy[str, dict[Actuators, tuple[float, float]]]"

    def __setattr__(self, name: str, value) -> None:
        try:
            super().__setattr__(name, value)
        except BrokenPipeError:
            ...

    def get_status(self) -> StatusStretchJoints:
        return self._status["val"]

    def set_status(self, value: StatusStretchJoints):
        self._status["val"] = value

    def get_command(self) -> StatusCommand:
        return self._command["val"]

    def set_command(self, value: StatusCommand):
        self._command["val"] = value

    def get_cameras(self) -> StatusStretchCameras:
        return self._cameras["val"]

    def set_cameras(self, value: StatusStretchCameras):
        self._cameras["val"] = value

    def get_sensors(self) -> StatusStretchSensors:
        return self._sensors["val"]

    def set_sensors(self, value: StatusStretchSensors):
        self._sensors["val"] = value

    def get_joint_limits(self) -> dict[Actuators, tuple[float, float]]:
        return self._joint_limits["val"]

    def set_joint_limit(self, actuator: Actuators, min_max: tuple[float, float]):
        limits = self._joint_limits["val"]
        limits[actuator] = min_max

        self._joint_limits["val"] = limits

    @staticmethod
    def default(manager: SyncManager) -> "MujocoServerProxies":
        return MujocoServerProxies(
            _command=manager.dict({"val": StatusCommand.default()}),
            _status=manager.dict({"val": StatusStretchJoints.default()}),
            _cameras=manager.dict({"val": StatusStretchCameras.default()}),
            _sensors=manager.dict({"val": StatusStretchSensors.default()}),
            _joint_limits=manager.dict({"val": {}}),
        )


class BaseController:

    def __init__(self, mujoco_server: "MujocoServer") -> None:
        self.mujoco_server = mujoco_server
        self.active_translate_x: CommandMove | None = None
        self.active_translate_y: CommandMove | None = None
        self.active_rotate: CommandMove | None = None
        self.active_velocity: CommandBaseVelocity | None = None
        
        self.start_pose_x = 0.0
        self.start_pose_y = 0.0
        self.start_pose_theta = 0.0

        # Trapezoidal profiles for omni wheels
        # Values can be tuned. max_vel and max_accel should match or slightly exceed robot capabilities.
        # Stepper motors have a max velocity around 50.0 rad/s
        max_vel_multiplier = 1.0
        self.left_wheel_profile = TrapezoidalProfile(max_vel=50.0*max_vel_multiplier, max_accel=15.0*max_vel_multiplier)
        self.right_wheel_profile = TrapezoidalProfile(max_vel=50.0*max_vel_multiplier, max_accel=15.0*max_vel_multiplier)
        self.back_wheel_profile = TrapezoidalProfile(max_vel=50.0*max_vel_multiplier, max_accel=15.0*max_vel_multiplier)
        self.profiles_initialized = False

        # omnibase params from config
        self.params = config.robot_settings_se4['omnibase']
        motion_default = self.params['motion']['default']
        self.curr_max_accel_xy_m = motion_default.get('accel_xy_m', 0.25) * max_vel_multiplier
        self.curr_max_accel_w_r = motion_default.get('accel_w_r', 2.0) * max_vel_multiplier
        self.curr_max_vel_xy_m = motion_default.get('vel_xy_m', 0.3) * max_vel_multiplier
        self.curr_max_vel_w_r = motion_default.get('vel_w_r', 2.0) * max_vel_multiplier

        # Omnibase kinematics
        self.H0 = H0_from_driving_dir(self.params['wheel_diameter_m'], self.params['base_radius_m'], self.params['forward_dir'])
        self.H0_inv = inverse_3x3_matrix(self.H0)

        # Status
        self.status = {'x': 0, 'y': 0, 'theta': 0, 'x_vel': 0, 'y_vel': 0, 'theta_vel': 0,
                       'pose_time_s': 0, 'wheel0_vel': 0, 'wheel1_vel': 0, 'wheel2_vel': 0}

    def push_command(self, command: CommandMove | CommandBaseVelocity):
        """Push a command to the base. Call `update()` to set the next trajectory."""
        if isinstance(command, CommandBaseVelocity):
            if command.v_x == 0.0 and command.v_y == 0.0 and command.omega == 0.0:
                self._clear_command(is_stop_motion=True)
            else:
                self.active_velocity = command
                self.active_translate_x = None
                self.active_translate_y = None
                self.active_rotate = None
        elif isinstance(command, CommandMove):
            self.active_velocity = None
            if command.actuator_name == Actuators.base_translate.name:
                self.active_rotate = None
                if self.active_translate_y is None:
                    curr_pose = self.get_base_pose()
                    self.start_pose_x = curr_pose[0]
                    self.start_pose_y = curr_pose[1]
                    self.start_pose_theta = curr_pose[2]
                self.active_translate_x = command
            elif command.actuator_name == Actuators.base_translate_y.name:
                self.active_rotate = None
                if self.active_translate_x is None:
                    curr_pose = self.get_base_pose()
                    self.start_pose_x = curr_pose[0]
                    self.start_pose_y = curr_pose[1]
                    self.start_pose_theta = curr_pose[2]
                self.active_translate_y = command
            elif command.actuator_name == Actuators.base_rotate.name:
                self.active_translate_x = None
                self.active_translate_y = None
                self.active_rotate = command
                self.start_pose_theta = self.get_base_pose()[2]

    def _clear_command(self, is_stop_motion: bool):
        self.active_translate_x = None
        self.active_translate_y = None
        self.active_rotate = None
        self.active_velocity = None

        if is_stop_motion:
            self._set_base_velocity(0.0, 0.0, 0.0, a_m=self.curr_max_accel_xy_m, a_r=self.curr_max_accel_w_r)

    def _set_wheel_vel(self, left_wheel_vel, back_wheel_vel, right_wheel_vel):
        # Wheel number is based on http://3.12.229.27/index.php/Base_Frame_Convention_%26_Wheel_Odometry
        # Wheel polarity was trial and error
        self.status['wheel0_vel'] = self.params['wheel0_polarity'] * left_wheel_vel * self.params['gr']
        self.status['wheel1_vel'] = self.params['wheel1_polarity'] * back_wheel_vel * self.params['gr']
        self.status['wheel2_vel'] = self.params['wheel2_polarity'] * right_wheel_vel * self.params['gr']

    def _control_wheel_vel(self, wheel0_vel, wheel0_accel, wheel1_vel, wheel1_accel, wheel2_vel, wheel2_accel):
        # Wheel number is based on http://3.12.229.27/index.php/Base_Frame_Convention_%26_Wheel_Odometry
        # Wheel polarity was trial and error
        # default_accel = 150.0
        default_accel = 15.0
        self.left_wheel_profile.max_accel = abs(wheel0_accel) if abs(wheel0_accel) > 1e-3 else default_accel
        self.left_wheel_profile.set_target_velocity(self.params['wheel0_polarity'] * wheel0_vel)

        self.back_wheel_profile.max_accel = abs(wheel1_accel) if abs(wheel1_accel) > 1e-3 else default_accel
        self.back_wheel_profile.set_target_velocity(self.params['wheel1_polarity'] * wheel1_vel)

        self.right_wheel_profile.max_accel = abs(wheel2_accel) if abs(wheel2_accel) > 1e-3 else default_accel
        self.right_wheel_profile.set_target_velocity(self.params['wheel2_polarity'] * wheel2_vel)

    def _update_odom(self, dt):
        """
        Calculate SE2 position of the base in odom frame from wheel odometry
        Important:
         - Assumes pull_status() was just called
         - Assumes the user calls this method at a regular frequency (>30hz)

        dt: 1/frequency at which this method is called
        """
        wheel_speeds = np.array([self.status['wheel0_vel'],
                                 self.status['wheel1_vel'],
                                 self.status['wheel2_vel']])
        Vb = self.H0_inv @ (wheel_speeds/self.params['gr'])
        # TODO x_vel and y_vel are in the base frame. Does it need to be rotated by status['theta']?
        base_speeds=self.motor_vel_to_base_vel(wheel_speeds)
        self.status['x_vel'] = float(base_speeds[0])
        self.status['y_vel'] = float(base_speeds[1])
        self.status['theta_vel'] = float(base_speeds[2])

        if dt is None:
            if self.status['pose_time_s'] is None:
                #print("pose_time_s is NONE")
                self.status['pose_time_s'] = time.time()

            dt = time.time() - self.status['pose_time_s']

        Sb = Vb*dt
        Sb = rotation_3x3_matrix(self.status['theta']) @ Sb
        self.status['x'] += float(Sb[0])
        self.status['y'] += float(Sb[1])
        self.status['theta'] += float(Sb[2])
        # if True:
        #     print(f"DEBUG_ODOM: wheel_speeds={wheel_speeds}, Vb={Vb}, dt={dt:.4f}, Sb={Sb}, status_y={self.status['y']:.4f}, status_theta={self.status['theta']:.4f}")
        self.status['pose_time_s'] = time.time()

    def update(self):
        """
        The update method to set mujoco ctrl's for the base while in motion.
        """
        if not self.profiles_initialized:
            # Need to initialize profiles with current positions to avoid sudden jumps
            self.left_wheel_profile.set_position(self.mujoco_server.mjdata.actuator(Actuators.left_wheel_vel.name).length[0] * self.params['gr'])
            self.right_wheel_profile.set_position(self.mujoco_server.mjdata.actuator(Actuators.right_wheel_vel.name).length[0] * self.params['gr'])
            self.back_wheel_profile.set_position(self.mujoco_server.mjdata.actuator(Actuators.back_wheel_vel.name).length[0] * self.params['gr'])
            self.profiles_initialized = True

        # Always step the profiles if we are in omni mode
        if not self.mujoco_server.use_diff_drive:
            # Step profiles
            # dt = self.mujoco_server.mjmodel.opt.timestep
            dt = 1.0 / self.mujoco_server.control_rate_hz
            l_pos = self.left_wheel_profile.update(dt)
            r_pos = self.right_wheel_profile.update(dt)
            b_pos = self.back_wheel_profile.update(dt)

            # if abs(self.left_wheel_profile.current_vel) > 1e-3 or abs(self.right_wheel_profile.current_vel) > 1e-3 or abs(self.back_wheel_profile.current_vel) > 1e-3:
            #     print(f"DEBUG_WHEELS: L_vel={self.left_wheel_profile.current_vel:.4f}/{self.left_wheel_profile.target_vel:.4f}, R_vel={self.right_wheel_profile.current_vel:.4f}/{self.right_wheel_profile.target_vel:.4f}, B_vel={self.back_wheel_profile.current_vel:.4f}/{self.back_wheel_profile.target_vel:.4f}, L_pos={l_pos:.4f}, R_pos={r_pos:.4f}, B_pos={b_pos:.4f}")

            # Apply to actuators (which are velocity actuators with gear=6)
            self.mujoco_server.mjdata.actuator(Actuators.left_wheel_vel.name).ctrl = self.left_wheel_profile.current_vel
            self.mujoco_server.mjdata.actuator(Actuators.right_wheel_vel.name).ctrl = self.right_wheel_profile.current_vel
            self.mujoco_server.mjdata.actuator(Actuators.back_wheel_vel.name).ctrl = self.back_wheel_profile.current_vel

        if self.active_velocity is not None:
            return self._set_base_velocity(
                translational_velocity_x=self.active_velocity.v_x,
                translational_velocity_y=self.active_velocity.v_y,
                angular_velocity_z=self.active_velocity.omega
            )

        if self.active_rotate is not None:
            return self._base_rotate_by(self.active_rotate.pos)

        if self.active_translate_x is not None or self.active_translate_y is not None:
            # Re-read active translate properties, because one might be finished
            x_inc = self.active_translate_x.pos if self.active_translate_x is not None else 0.0
            y_inc = self.active_translate_y.pos if self.active_translate_y is not None else 0.0
            if x_inc == 0.0 and y_inc == 0.0:
                 return self._clear_command(is_stop_motion=True)
            return self._base_translate_by_combined(x_inc, y_inc)

    def get_base_pose(self) -> np.ndarray:
        """Get the se(2) base pose: x, y, and theta"""
        xyz = self.mujoco_server.mjdata.body("base_link").xpos
        rotation = self.mujoco_server.mjdata.body("base_link").xmat.reshape(3, 3)
        theta = np.arctan2(rotation[1, 0], rotation[0, 0])
        return np.array([xyz[0], xyz[1], theta])

    def _base_translate_by_combined(self, x_inc: float, y_inc: float) -> None:
        """
        Translate the base by certain increments in X and/or Y using current body frame error.
        """
        curr_pose = self.get_base_pose()
        curr_x, curr_y, curr_theta = curr_pose[0], curr_pose[1], curr_pose[2]
        
        # Global target position calculated from starting pose
        start_th = self.start_pose_theta
        target_x = self.start_pose_x + x_inc * np.cos(start_th) - y_inc * np.sin(start_th)
        target_y = self.start_pose_y + x_inc * np.sin(start_th) + y_inc * np.cos(start_th)
        
        # Position error in global frame
        dx_global = target_x - curr_x
        dy_global = target_y - curr_y
        
        # Transform error into CURRENT body frame
        err_x = dx_global * np.cos(curr_theta) + dy_global * np.sin(curr_theta)
        err_y = -dx_global * np.sin(curr_theta) + dy_global * np.cos(curr_theta)

        Kp = 8.0
        a_xy = self.curr_max_accel_xy_m

        # Calculate velocity commands for X and Y in CURRENT body frame
        if self.active_translate_x is not None:
            if abs(err_x) < 0.003:
                self.active_translate_x = None
                x_v = 0.0
            else:
                v_decel_x = np.sqrt(2.0 * a_xy * abs(err_x))
                x_v = np.clip(Kp * err_x, -self.curr_max_vel_xy_m, self.curr_max_vel_xy_m)
                x_v = np.copysign(min(abs(x_v), v_decel_x), err_x) if err_x != 0 else 0.0
                if abs(x_v) > 0 and abs(x_v) < 0.035:
                    x_v = np.copysign(0.035, x_v)
        else:
            x_v = 0.0

        if self.active_translate_y is not None:
            if abs(err_y) < 0.003:
                self.active_translate_y = None
                y_v = 0.0
            else:
                v_decel_y = np.sqrt(2.0 * a_xy * abs(err_y))
                y_v = np.clip(Kp * err_y, -self.curr_max_vel_xy_m, self.curr_max_vel_xy_m)
                y_v = np.copysign(min(abs(y_v), v_decel_y), err_y) if err_y != 0 else 0.0
                if abs(y_v) > 0 and abs(y_v) < 0.035:
                    y_v = np.copysign(0.035, y_v)
        else:
            y_v = 0.0

        if self.active_translate_x is None and self.active_translate_y is None:
            return self._clear_command(is_stop_motion=True)

        # Active smooth heading & cross-track steering stabilization
        theta_err = (self.start_pose_theta - curr_theta + np.pi) % (2 * np.pi) - np.pi
        
        # Steering correction: align heading while gently steering towards the target line
        if self.active_translate_x is not None and self.active_translate_y is None:
            steering_corr = np.clip(-3.0 * err_y, -0.2, 0.2)
        elif self.active_translate_y is not None and self.active_translate_x is None:
            steering_corr = np.clip(3.0 * err_x, -0.2, 0.2)
        else:
            steering_corr = 0.0

        w_v = np.clip(10.0 * theta_err + steering_corr, -0.5, 0.5)
        
        self._set_base_velocity(x_v, y_v, w_v)

    def _base_rotate_by(self, theta_inc: float) -> None:
        """
        Rotate the base by a certain w.r.t base global pose
        """
        target_theta = self.start_pose_theta + theta_inc
        curr_theta = self.get_base_pose()[2]
        
        # Normalize angle difference to [-pi, pi]
        theta_err = (target_theta - curr_theta + np.pi) % (2 * np.pi) - np.pi

        if abs(theta_err) < 0.012: 
            return self._clear_command(is_stop_motion=True)

        a_r = self.curr_max_accel_w_r
        v_decel = np.sqrt(2.0 * a_r * abs(theta_err))

        Kp = 5.0
        w_v = np.clip(Kp * theta_err, -self.curr_max_vel_w_r, self.curr_max_vel_w_r)
        w_v = np.copysign(min(abs(w_v), v_decel), theta_err)
        if abs(w_v) > 0 and abs(w_v) < 0.10:
            w_v = np.copysign(0.10, w_v)

        self._set_base_velocity(0.0, 0.0, w_v)


    def _set_base_velocity_diff_drive(self, v_linear: float, omega: float) -> None:
        """
        Set the base velocity of the robot
        Args:
            v_linear: float, linear velocity
            omega: float, angular velocity
        """
        w_left, w_right = utils.diff_drive_inv_kinematics(v_linear, omega)
        self.mujoco_server.mjdata.actuator(Actuators.left_wheel_vel.name).ctrl = w_left
        self.mujoco_server.mjdata.actuator(Actuators.right_wheel_vel.name).ctrl = w_right

    def _set_base_velocity_omni_drive(self,
    translational_velocity_x: float, translational_velocity_y:float, angular_velocity_z: float, a_m=None, a_r=None) -> None:
        """
        Set the base velocity of the robot
        Args:
            v_linear: float, linear velocity
            omega: float, angular velocity
        """
        # motion limits
        translational_velocity_x = np.clip(translational_velocity_x, -self.curr_max_vel_xy_m,
                                           self.curr_max_vel_xy_m)
        translational_velocity_y = np.clip(translational_velocity_y, -self.curr_max_vel_xy_m,
                                           self.curr_max_vel_xy_m)
        angular_velocity_z = np.clip(angular_velocity_z, -self.curr_max_vel_w_r,
                                     self.curr_max_vel_w_r)

        if a_m is not None:
            a_m = abs(a_m)
        else:
            a_m = self.curr_max_accel_xy_m

        if a_r is not None:
            a_r = abs(a_r)
        else:
            a_r = self.curr_max_accel_w_r

        a_m_wheel = (2 / self.params['wheel_diameter_m']) * (
                    a_m + self.params['base_radius_m'] * a_r)  # max wheel velocity

        # calculate the motor vels
        u = self.base_vel_to_motor_vel([translational_velocity_x, translational_velocity_y, angular_velocity_z])
        aa = self.compute_motor_acceleration(u, a_m_wheel)

        # Set targets and accels for the profiles (targets need to be motor frame)
        self._control_wheel_vel(u[0], aa[0], u[1], aa[1], u[2], aa[2])

    def base_vel_to_motor_vel(self,v):
        #Convert base velocities to motor velocities
        Vb = np.array(v)
        u_w = self.H0 @ Vb #  wheel target velocites
        u = u_w * self.params['gr'] # motor target velocities
        return u

    def motor_vel_to_base_vel(self,u):
        u_w = u / self.params['gr']
        Vb = self.H0_inv @ u_w
        return Vb

    def compute_motor_acceleration(self, u_target, a_m_wheel):
        """
        Take target velocities (motor frame) (rad/s) and a desired acceleration
        Return the accelerations (motor frame) that achieve the target
        at the same time in the future.
        """
        u_target_w=u_target/self.params['gr']
        return self.compute_wheel_acceleration(u_target_w,a_m_wheel)

    def compute_wheel_acceleration(self, u_target_w, a_m_wheel):
        """
        Take target velocities (wheel frame) (rad/s) and a desired acceleration
        Return the accelerations (motor frame) that achieve the target
        at the same time in the future.
        """
        if hasattr(self, 'left_wheel_profile'):
            u_current_w = np.array([
                self.params['wheel0_polarity'] * self.left_wheel_profile.current_vel,
                self.params['wheel1_polarity'] * self.back_wheel_profile.current_vel,
                self.params['wheel2_polarity'] * self.right_wheel_profile.current_vel
            ]) / self.params['gr']
        else:
            wheel_speeds = np.array([self.status['wheel0_vel'],
                                     self.status['wheel1_vel'],
                                     self.status['wheel2_vel']])  # current motor velocities
            u_current_w = wheel_speeds / self.params['gr']  # current wheel velocities

        delta_u = u_target_w - u_current_w
        max_delta = np.max(np.abs(delta_u))

        if max_delta == 0:
            accel = np.zeros_like(delta_u)
        else:
            T = max_delta / a_m_wheel #Worst case time to accel/deccel to target
            accel = delta_u / T

        return accel * self.params["gr"]  # Convert to motor velocities

    def _set_base_velocity(self,
    translational_velocity_x: float, translational_velocity_y:float, angular_velocity_z: float, a_m=None, a_r=None) -> None:
        if self.mujoco_server.use_diff_drive:
            return self._set_base_velocity_diff_drive(translational_velocity_x, angular_velocity_z)

        return self._set_base_velocity_omni_drive(translational_velocity_x, translational_velocity_y, angular_velocity_z, a_m=a_m, a_r=a_r)


class MujocoServer:
    """
    Use `MucocoServer.launch_server()` to start the headless simulator.

    This uses the mujoco simulator in headless mode.
    """

    @classmethod
    def launch_server(
        cls,
        scene_xml_path: str | None,
        model: MjModel | None,
        camera_hz: float,
        show_viewer_ui: bool,
        stop_mujoco_process_event: threading.Event,
        data_proxies: MujocoServerProxies,
        cameras_to_use: list[StretchCameras],
        start_translation: list|None,
        start_rotation_quat: list|None
    ):
        server = cls(scene_xml_path, model, stop_mujoco_process_event, data_proxies, start_translation, start_rotation_quat)
        server.run(
            show_viewer_ui=show_viewer_ui,
            camera_hz=camera_hz,
            cameras_to_use=cameras_to_use,
        )

    def __init__(
        self,
        scene_xml_path: str | None,
        model: MjModel | None,
        stop_mujoco_process_event: threading.Event,
        data_proxies: MujocoServerProxies,
        start_translation: list|None,
        start_rotation_quat: list|None
    ):
        """
        Initialize the Simulator handle with a scene
        Args:
            scene_xml_path: str, path to the scene xml file
            model: MjModel, Mujoco model object
        """
        if model is not None and scene_xml_path is not None:
            raise ValueError("You should not provide both a model and a scene_xml_path. Please provide only one.")

        if scene_xml_path is None:
            # Import here to avoid circular dependency
            from stretch4_mujoco import StretchMujocoSimulator
            scene_xml_path = StretchMujocoSimulator.get_scene_xml_path()

        if model is None:
            model = MjModel.from_xml_path(scene_xml_path)

        change_start_pose(model, name="stretch4", translation=start_translation, rotation_quat=start_rotation_quat)

        self.mjmodel = model

        self.mjdata = MjData(self.mjmodel)
        mujoco.mj_forward(self.mjmodel, self.mjdata)
        # Initialize position actuators to current positions to prevent initial slumping
        for i in range(self.mjmodel.na):
            if self.mjmodel.actuator_biastype[i] == 1: # position/affine actuators
                self.mjdata.ctrl[i] = self.mjdata.actuator(i).length[0]

        self.use_diff_drive = True
        for b_name in Actuators.back_wheel_vel.get_joint_names_in_mjcf():
            if mujoco.mj_name2id(self.mjmodel, mujoco.mjtObj.mjOBJ_JOINT, b_name) != -1:
                self.use_diff_drive = False
                break

        self._base_in_pos_motion = False

        self._stop_mujoco_process_event = stop_mujoco_process_event

        self.data_proxies = data_proxies

        self.base_controller = BaseController(self)

        self.physics_fps_counter = FpsCounter()

        self._last_wall_time = time.perf_counter()
        self._last_sim_time = 0.0

        self.sensor_manager = MujocoServerSensorManagerThreaded(
            sensor_hz=15,
            sensors_to_use=StretchSensors.from_mjmodel(self.mjmodel),
            mujoco_server=self,
        )

        self.update_joint_limits()
        
        self.control_rate_hz = 100.0
        self.physics_dt = self.mjmodel.opt.timestep
        self.physics_steps_per_control_step = int(1.0 / (self.control_rate_hz * self.physics_dt)) 
        print(f"Physics dt: {self.physics_dt}, Control rate: {self.control_rate_hz}Hz, "
              f"Physics steps per control step: {self.physics_steps_per_control_step}")

        signal.signal(signal.SIGTERM, lambda num, h: self.request_to_stop())
        signal.signal(signal.SIGINT, lambda num, h: self.request_to_stop())

    @property
    def robot_settings(self):
        """Get the robot settings from config.py"""
        # This assumes self.use_diff_drive==True means we're using Stretch 3. This could be ambiguous. TODO: Add an explicit flag for robot version.
        if self.use_diff_drive:
            return config.robot_settings
        return config.robot_settings_se4

    def update_joint_limits(self):
        limits = {}
        for i in range(self.mjmodel.njnt):
            name = mujoco._functions.mj_id2name(self.mjmodel, mujoco._enums.mjtObj.mjOBJ_JOINT, i)
            joint_range = self.mjmodel.jnt_range[i]  # This gives [lower_limit, upper_limit]
            try:
                actuator = Actuators.get_actuator_by_joint_names_in_mjcf(name)
                if actuator not in limits:
                    limits[actuator] = []
                limits[actuator].append((joint_range[0], joint_range[1]))
            except:
                ...

        for actuator, ranges in limits.items():
            if actuator == Actuators.arm:
                min_limit = 0.0
                max_limit = sum(r[1] for r in ranges)
            elif actuator == Actuators.gripper and not self.use_diff_drive:
                    def get_angle_from_chord_length_and_radius(radius_m, chord_m):
                        return 2 * np.arcsin(chord_m / (2 * radius_m))   # radians
        
                    max_limit = get_angle_from_chord_length_and_radius(
                                            self.robot_settings['gripper_conversion']['finger_length_m'],
                                            self.robot_settings['gripper_conversion']['aperture_open_m'],
                                            )
                    min_limit = get_angle_from_chord_length_and_radius(
                                            self.robot_settings['gripper_conversion']['finger_length_m'],
                                            self.robot_settings['gripper_conversion']['aperture_closed_m'],
                                            )

            else:
                min_limit, max_limit = ranges[-1]

            self.data_proxies.set_joint_limit(
                actuator=actuator, min_max=(min_limit, max_limit)
            )

    def set_camera_manager(
        self,
        camera_hz: float,
        cameras_to_use: list[StretchCameras],
        *,
        use_camera_thread: bool,
        use_threadpool_executor: bool,
    ):
        """
        This should be called before trying to render offscreen cameras.

        If `use_camera_thread` is false, `self.camera_manager.pull_camera_data_at_camera_rate()` should be called on a UI thread.
        This is the recommended usage.

        If `use_camera_thread` is true, a thread will be spawned to call Renderer.render().
        This may not work on all platforms since rendering should happen on the main thread.
        This mode is mainly used with the Mujoco Managed Viewer, to avoid rendering on the physics thread.
        """
        if use_camera_thread or use_threadpool_executor:
            self.camera_manager = MujocoServerCameraManagerThreaded(
                use_camera_thread=use_camera_thread,
                use_threadpool_executor=use_threadpool_executor,
                camera_hz=camera_hz,
                cameras_to_use=cameras_to_use,
                mujoco_server=self,
            )
        else:
            self.camera_manager = MujocoServerCameraManagerSync(
                camera_hz=camera_hz,
                cameras_to_use=cameras_to_use,
                mujoco_server=self,
            )

    def run(
        self,
        show_viewer_ui: bool,
        camera_hz: float,
        cameras_to_use: list[StretchCameras],
    ):
        # self.__run_headless_simulation(camera_hz=camera_hz, cameras_to_use=cameras_to_use)
        self.__run_headless_simulation_with_physics_thread(
            camera_hz=camera_hz,
            cameras_to_use=cameras_to_use,
        )

    def _is_requested_to_stop(self):
        try:
            return self._stop_mujoco_process_event.is_set()
        except (EOFError, BrokenPipeError):
            # We likely lost connection to the main process if we've hit this.
            return True

    def request_to_stop(self):
        try:
            self._stop_mujoco_process_event.set()
        except (EOFError, BrokenPipeError):
            # We likely lost connection to the main process if we've hit this.
            ...

    def close(self):
        """
        Clean up C++ resources
        """
        self.request_to_stop()

        if isinstance(self.camera_manager, MujocoServerCameraManagerThreaded):
            self.camera_manager.cameras_thread.join()

        if isinstance(self.sensor_manager, MujocoServerSensorManagerThreaded):
            self.sensor_manager.sensors_thread.join()

        self.camera_manager.close()

    def _run_ui_simulation(self, show_viewer_ui: bool) -> None:
        """
        Run the simulation with the viewer
        """
        raise NotImplementedError(
            "This is headless mode. Use MujocoServerPassive or MujocoServerManaged to run the UI simulator."
        )

    def _physics_step(self, lock: contextlib.AbstractContextManager):
        """
        Calls mj_step multiple times and _ctrl_callback once.
        """
        with lock:
            # Run physics steps
            for _ in range(self.physics_steps_per_control_step):
                mujoco._functions.mj_step(self.mjmodel, self.mjdata)

            self._ctrl_callback(self.mjmodel, self.mjdata)

    def _physics_loop(
        self, lock: contextlib.AbstractContextManager, termination_check: Callable[[], bool]
    ):
        """
        A loop to use when starting physics in a thread, maintaining precise real-time synchronization.
        """
        target_period = 1.0 / self.control_rate_hz
        next_target_time = time.perf_counter()

        while termination_check():
            next_target_time += target_period
            self._physics_step(lock=lock)

            time_until_next = next_target_time - time.perf_counter()
            if time_until_next > 0.001:
                time.sleep(time_until_next - 0.0005)

            while time.perf_counter() < next_target_time:
                pass

            if next_target_time < time.perf_counter() - (target_period * 10):
                next_target_time = time.perf_counter()

        click.secho("Physics Loop has terminated.", fg="red")

    def __run_headless_simulation(
        self, camera_hz: float, cameras_to_use: list[StretchCameras]
    ) -> None:
        """
        Run the simulation without the viewer headless.

        Headless mode manages its own `set_camera_manager()` call.
        """
        print("Running headless simulation...")

        self.set_camera_manager(
            use_camera_thread=False,
            use_threadpool_executor=False,
            camera_hz=camera_hz,
            cameras_to_use=cameras_to_use,
        )

        target_period = 1.0 / self.control_rate_hz
        next_target_time = time.perf_counter()

        while not self._is_requested_to_stop():
            next_target_time += target_period
            self._physics_step(contextlib.nullcontext())
            self.camera_manager.pull_camera_data_at_camera_rate(is_sleep_until_ready=False)

            time_until_next = next_target_time - time.perf_counter()
            if time_until_next > 0.001:
                time.sleep(time_until_next - 0.0005)

            while time.perf_counter() < next_target_time:
                pass

            if next_target_time < time.perf_counter() - (target_period * 10):
                next_target_time = time.perf_counter()

        self.close()

    def __run_headless_simulation_with_physics_thread(
        self,
        camera_hz: float,
        cameras_to_use: list[StretchCameras],
    ) -> None:
        """
        Run the simulation without the viewer headless.

        Headless mode manages its own `set_camera_manager()` call.
        """
        print("Running headless simulation...")

        self.set_camera_manager(
            use_camera_thread=False,
            use_threadpool_executor=False,
            camera_hz=camera_hz,
            cameras_to_use=cameras_to_use,
        )

        physics_thread = threading.Thread(
            target=self._physics_loop,
            args=(self.camera_manager.camera_lock, lambda: not self._is_requested_to_stop()),
            daemon=True,
        )
        physics_thread.start()

        while not self._is_requested_to_stop():
            self.camera_manager.pull_camera_data_at_camera_rate(is_sleep_until_ready=True)

        physics_thread.join()
        self.close()

    def _ctrl_callback(self, model: MjModel, data: MjData) -> None:
        """
        Callback function that gets executed with mj_step
        """
        self.mjdata = data
        self.mjmodel = model

        if not self.mjdata or not self.mjdata.time:
            print("WARNING: no mujoco data to report")
            return

        self.physics_fps_counter.tick(sim_time=data.time)
        self.pull_status()
        self.push_command(self.data_proxies.get_command())

    def pull_status(self):
        """
        Pull joints status of the robot from the simulator
        """

        new_status = StatusStretchJoints.default()
        new_status.fps = self.physics_fps_counter.fps

        new_status.time = self.mjdata.time
        new_status.sim_to_real_time_ratio_msg = self.physics_fps_counter.sim_to_real_time_ratio_msg
        new_status.lift.pos = self.mjdata.actuator(Actuators.lift.name).length[0]
        new_status.lift.vel = self.mjdata.actuator(Actuators.lift.name).velocity[0]
        new_status.lift.effort = self.mjdata.actuator(Actuators.lift.name).force[0]

        new_status.arm.pos = self.mjdata.actuator(Actuators.arm.name).length[0]
        new_status.arm.vel = self.mjdata.actuator(Actuators.arm.name).velocity[0]
        new_status.arm.effort = self.mjdata.actuator(Actuators.arm.name).force[0]

        if self.use_diff_drive:
            # Head joints are only available in Stretch 3. Using diff drive is assuming we're using Stretch 3. TODO: add an explicit flag for head joint to avoid confusion
            new_status.head_pan.pos = self.mjdata.actuator(Actuators.head_pan.name).length[0]
            new_status.head_pan.vel = self.mjdata.actuator(Actuators.head_pan.name).velocity[0]
            new_status.head_pan.effort = self.mjdata.actuator(Actuators.head_pan.name).force[0]

            new_status.head_tilt.pos = self.mjdata.actuator(Actuators.head_tilt.name).length[0]
            new_status.head_tilt.vel = self.mjdata.actuator(Actuators.head_tilt.name).velocity[0]
            new_status.head_tilt.effort = self.mjdata.actuator(Actuators.head_tilt.name).force[0]

        new_status.wrist_yaw.pos = self.mjdata.actuator(Actuators.wrist_yaw.name).length[0]
        new_status.wrist_yaw.vel = self.mjdata.actuator(Actuators.wrist_yaw.name).velocity[0]
        new_status.wrist_yaw.effort = self.mjdata.actuator(Actuators.wrist_yaw.name).force[0]

        new_status.wrist_pitch.pos = self.mjdata.actuator(Actuators.wrist_pitch.name).length[0]
        new_status.wrist_pitch.vel = self.mjdata.actuator(Actuators.wrist_pitch.name).velocity[0]
        new_status.wrist_pitch.effort = self.mjdata.actuator(Actuators.wrist_pitch.name).force[0]

        new_status.wrist_roll.pos = self.mjdata.actuator(Actuators.wrist_roll.name).length[0]
        new_status.wrist_roll.vel = self.mjdata.actuator(Actuators.wrist_roll.name).velocity[0]
        new_status.wrist_roll.effort = self.mjdata.actuator(Actuators.wrist_roll.name).force[0]


        if self.use_diff_drive:
            new_status.gripper.pos = self._to_real_gripper_range(
                self.mjdata.actuator("gripper").length[0]
            )
            new_status.gripper.vel = self.mjdata.actuator("gripper").velocity[
                0
            ]  # This is still in sim gripper range
            new_status.gripper.effort = self.mjdata.actuator("gripper").force[0]
        else:
            new_status.gripper_left_finger.pos = self.mjdata.actuator(Actuators.gripper_left_finger.name).length[0]
            new_status.gripper_left_finger.vel = self.mjdata.actuator(Actuators.gripper_left_finger.name).velocity[0]
            new_status.gripper_left_finger.effort = self.mjdata.actuator(Actuators.gripper_left_finger.name).force[0]
            new_status.gripper_right_finger.pos = self.mjdata.actuator(Actuators.gripper_right_finger.name).length[0]
            new_status.gripper_right_finger.vel = self.mjdata.actuator(Actuators.gripper_right_finger.name).velocity[0]
            new_status.gripper_right_finger.effort = self.mjdata.actuator(Actuators.gripper_right_finger.name).force[0]

            # Populate gripper status (aperture in radians) by converting from finger URDF joint angle
            avg_finger_pos = (new_status.gripper_left_finger.pos + new_status.gripper_right_finger.pos) / 2.0
            new_status.gripper.pos = self.urdf_angle_radians_to_aperture_angle_radians(avg_finger_pos)
            new_status.gripper.vel = (new_status.gripper_left_finger.vel + new_status.gripper_right_finger.vel) / 2.0
            new_status.gripper.effort = (new_status.gripper_left_finger.effort + new_status.gripper_right_finger.effort) / 2.0

        left_wheel_vel = self.mjdata.actuator(Actuators.left_wheel_vel.name).velocity[0]
        right_wheel_vel = self.mjdata.actuator(Actuators.right_wheel_vel.name).velocity[0]

        if not self.use_diff_drive:
            # We use ground truth wheel velocity to calculate wheel odometry
            # We won't see motor or encoder noise represented, but we will see wheel slip and numerical integration.
            left_wheel_vel = self.mjdata.joint(Actuators.left_wheel_vel.get_joint_names_in_mjcf()[1]).qvel[0] / self.base_controller.params['gr']
            back_wheel_vel = self.mjdata.joint(Actuators.back_wheel_vel.get_joint_names_in_mjcf()[1]).qvel[0] / self.base_controller.params['gr']
            right_wheel_vel = self.mjdata.joint(Actuators.right_wheel_vel.get_joint_names_in_mjcf()[1]).qvel[0] / self.base_controller.params['gr']
            self.base_controller._set_wheel_vel(left_wheel_vel=left_wheel_vel, back_wheel_vel=back_wheel_vel, right_wheel_vel=right_wheel_vel)

            # Compute wheel odometry and assign it
            self.base_controller._update_odom(1.0 / self.control_rate_hz)
            new_status.base.x_vel = self.base_controller.status['x_vel']
            new_status.base.y_vel = self.base_controller.status['y_vel']
            new_status.base.theta_vel = self.base_controller.status['theta_vel']
            new_status.base.x, new_status.base.y, new_status.base.theta = self.base_controller.get_base_pose()
            new_status.base.active_translate_x = (self.base_controller.active_translate_x is not None)
            new_status.base.active_translate_y = (self.base_controller.active_translate_y is not None)
            new_status.base.active_rotate = (self.base_controller.active_rotate is not None)
        else:
            (
                new_status.base.x_vel,
                new_status.base.theta_vel,
            ) = utils.diff_drive_fwd_kinematics(left_wheel_vel, right_wheel_vel)

            (
                new_status.base.x,
                new_status.base.y,
                new_status.base.theta,
            ) = self.base_controller.get_base_pose()



        new_status.is_self_colliding = False
        for i in range(self.mjdata.ncon):
            contact = self.mjdata.contact[i]
            geom1_name = mujoco.mj_id2name(self.mjmodel, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mujoco.mj_id2name(self.mjmodel, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            
            def is_robot_geom(g_name):
                if not g_name: return False
                return "link_" in g_name or "_link" in g_name or "stretch" in g_name or "base_" in g_name
                
            if is_robot_geom(geom1_name) and is_robot_geom(geom2_name):
                # A lot of geoms might technically intersect by design depending on limits, but ncon tracks active contacts.
                new_status.is_self_colliding = True
                break

        self.data_proxies.set_status(new_status)


        self._last_wall_time = time.perf_counter()
        self._last_sim_time = self.mjdata.time

    def _to_real_gripper_range(self, pos: float) -> float:
        """
        Map the gripper position to real gripper range
        """
        return utils.map_between_ranges(
            pos,
            self.robot_settings["sim_gripper_min_max"],
            self.robot_settings["gripper_min_max"],
        )

    def push_command(self, command_status:StatusCommand):
        """
        Handles setting mujoco ctrl properties to move joints.
        """
        modified = False
        # move_by
        for _, command in command_status.move_by.items():
            if command.trigger:
                command.trigger = False
                modified = True
                actuator_name = command.actuator_name
                pos = command.pos
                if actuator_name in (Actuators.base_translate.name, Actuators.base_translate_y.name, Actuators.base_rotate.name):
                    self.base_controller.push_command(command)
                elif actuator_name == Actuators.gripper.name:
                    if self.use_diff_drive:
                        current_value = self._to_real_gripper_range(
                            self.mjdata.actuator(actuator_name).length[0]
                        )
                        self.mjdata.actuator(actuator_name).ctrl = self._to_sim_gripper_range(
                            current_value + pos
                        )
                    else:
                        current_ctrl_left = self.mjdata.actuator(Actuators.gripper_left_finger.name).ctrl[0]
                        current_aperture = self.urdf_angle_radians_to_aperture_angle_radians(current_ctrl_left)
                        target_aperture = current_aperture + pos
                        finger_pos = self.aperture_angle_radians_to_urdf_angle_radians(target_aperture)
                        self.mjdata.actuator(Actuators.gripper_left_finger.name).ctrl = finger_pos
                        self.mjdata.actuator(Actuators.gripper_right_finger.name).ctrl = finger_pos
                else:
                    current_value = self.mjdata.actuator(actuator_name).length[0]
                    self.mjdata.actuator(actuator_name).ctrl = current_value + pos

        # move_to
        for _, command in command_status.move_to.items():
            if command.trigger:
                command.trigger = False
                modified = True
                actuator_name = command.actuator_name

                pos = command.pos
                if actuator_name in (Actuators.base_translate.name, Actuators.base_translate_y.name, Actuators.base_rotate.name):
                    raise NotImplementedError(
                        f"Cannot set move_to for {actuator_name}, which is a relative joint."
                    )
                elif actuator_name == Actuators.gripper.name:
                    if self.use_diff_drive:
                        self.mjdata.actuator(actuator_name).ctrl = self._to_sim_gripper_range(pos)
                    else:
                        finger_pos = self.aperture_angle_radians_to_urdf_angle_radians(pos)
                        self.mjdata.actuator(Actuators.gripper_left_finger.name).ctrl = finger_pos
                        self.mjdata.actuator(Actuators.gripper_right_finger.name).ctrl = finger_pos
                else:
                    self.mjdata.actuator(actuator_name).ctrl = pos

        # joint_velocities (continuous integration control)
        dt = 1.0 / self.control_rate_hz
        for actuator_name, target_vel in list(command_status.joint_velocities.items()):
            if target_vel != 0.0:
                if actuator_name == Actuators.gripper.name:
                    if self.use_diff_drive:
                        current_ctrl = self._to_real_gripper_range(
                            self.mjdata.actuator(actuator_name).ctrl[0]
                        )
                        self.mjdata.actuator(actuator_name).ctrl = self._to_sim_gripper_range(
                            current_ctrl + target_vel * dt
                        )
                    else:
                        current_ctrl_left = self.mjdata.actuator(Actuators.gripper_left_finger.name).ctrl[0]
                        current_aperture = self.urdf_angle_radians_to_aperture_angle_radians(current_ctrl_left)
                        target_aperture = current_aperture + target_vel * dt
                        finger_pos = self.aperture_angle_radians_to_urdf_angle_radians(target_aperture)
                        self.mjdata.actuator(Actuators.gripper_left_finger.name).ctrl = finger_pos
                        self.mjdata.actuator(Actuators.gripper_right_finger.name).ctrl = finger_pos
                else:
                    current_ctrl = self.mjdata.actuator(actuator_name).ctrl[0]
                    self.mjdata.actuator(actuator_name).ctrl = current_ctrl + target_vel * dt

        # set_base_velocity
        if command_status.base_velocity is not None and command_status.base_velocity.trigger:
            command_status.base_velocity.trigger = False
            modified = True
            self.base_controller.push_command(command_status.base_velocity)

        # keyframe
        if command_status.keyframe is not None and command_status.keyframe.trigger:
            command_status.keyframe.trigger = False
            modified = True
            self.mjdata.ctrl = self.mjmodel.keyframe(command_status.keyframe.name).ctrl

        self.base_controller.update()

        if modified:
            self.data_proxies.set_command(command_status)

    def _to_sim_gripper_range(self, pos: float) -> float:
        """
        Map the gripper position to sim gripper range
        """
        return utils.map_between_ranges(
            pos,
            self.robot_settings["gripper_min_max"],
            self.robot_settings["sim_gripper_min_max"],
        )

    # NOTE: This is copied from gripper_conversions.py to avoid a stretch4_body dependency, except the urdf offset hsa been removed
    def aperture_angle_radians_to_urdf_angle_radians(self, aperture_angle_radians):

        def get_angle_from_chord_length_and_radius(radius_m, chord_m):
            return 2 * np.arcsin(chord_m / (2 * radius_m))   # radians

        aperture_open_rad = get_angle_from_chord_length_and_radius(
                                self.robot_settings['gripper_conversion']['finger_length_m'],
                                self.robot_settings['gripper_conversion']['aperture_open_m'],
                                )

        aperture_close_rad = get_angle_from_chord_length_and_radius(
                                self.robot_settings['gripper_conversion']['finger_length_m'],
                                self.robot_settings['gripper_conversion']['aperture_closed_m'],
                                )

        return utils.map_between_ranges(
            aperture_angle_radians,
            (aperture_close_rad, aperture_open_rad),
            (self.robot_settings['gripper_conversion']['urdf_closed_rad'], self.robot_settings['gripper_conversion']['urdf_open_rad'])
            )

    def urdf_angle_radians_to_aperture_angle_radians(self, urdf_angle_radians):
        def get_angle_from_chord_length_and_radius(radius_m, chord_m):
            return 2 * np.arcsin(chord_m / (2 * radius_m))   # radians

        aperture_open_rad = get_angle_from_chord_length_and_radius(
                                self.robot_settings['gripper_conversion']['finger_length_m'],
                                self.robot_settings['gripper_conversion']['aperture_open_m'],
                                )

        aperture_close_rad = get_angle_from_chord_length_and_radius(
                                self.robot_settings['gripper_conversion']['finger_length_m'],
                                self.robot_settings['gripper_conversion']['aperture_closed_m'],
                                )

        return utils.map_between_ranges(
            urdf_angle_radians,
            (self.robot_settings['gripper_conversion']['urdf_closed_rad'], self.robot_settings['gripper_conversion']['urdf_open_rad']),
            (aperture_close_rad, aperture_open_rad)
            )
