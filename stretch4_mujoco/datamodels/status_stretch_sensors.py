import copy
from dataclasses import asdict, dataclass

import numpy as np

from stretch4_mujoco.enums.stretch_sensors import StretchSensors
from stretch4_mujoco.utils import dataclass_from_dict, gravity_tilt_from_quaternion


@dataclass
class StatusStretchSensors:
    """
    A dataclass and helper methods to pack camera data.
    """

    time: float
    fps: float

    base_gyro: np.ndarray | None = None
    base_imu: np.ndarray | None = None
    base_quat: np.ndarray | None = None
    lidar: np.ndarray | None = None

    @property
    def gravity_tilt(self) -> float:
        """How far the base leans off vertical, in radians.

        The same quantity the robot publishes as
        `power_periph.imu.status['gravity_tilt']`, so a tilt threshold written
        against the robot reads the same here. Upright is 0. Reads 0 when the
        `base_quat` sensor is absent from the model.
        """
        if self.base_quat is None:
            return 0.0
        return gravity_tilt_from_quaternion(self.base_quat)

    def get_data(self, sensor: StretchSensors) -> np.ndarray:
        """
        Use this to match a StretchSensors enum instance with its property in StatusStretchSensors dataclass, to get the camera data property.

        A note to nip confusion - this get the values inside this dataclass, it does not poll from the simulator.
        """

        data: np.ndarray | None = None
        if sensor == StretchSensors.base_gyro:
            data = self.base_gyro
        elif sensor == StretchSensors.base_accel:
            data = self.base_imu
        elif sensor == StretchSensors.base_quat:
            data = self.base_quat
        elif sensor == StretchSensors.base_lidar:
            data = self.lidar

        if data is None:
            raise ValueError(f"Tried to get {sensor} data, but it is empty.")

        return data

    def set_data(self, sensor: StretchSensors, value: np.ndarray):
        """
        Use this to match a StretchSensors enum instance with its property in StatusStretchSensors dataclass, to set the camera data property.

        A note to nip confusion - this get the values inside this dataclass, it does not send data to the simulator.
        """

        if sensor == StretchSensors.base_gyro:
            self.base_gyro = value
            return
        if sensor == StretchSensors.base_accel:
            self.base_imu = value
            return
        if sensor == StretchSensors.base_quat:
            self.base_quat = value
            return
        if sensor == StretchSensors.base_lidar:
            self.lidar = value
            return

        raise NotImplementedError(f"Sensor {sensor} is not implemented")

    @staticmethod
    def default():
        """
        Returns an empty instance with None or zeros for properties.
        """
        return StatusStretchSensors(time=0, fps=0)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(dict_data: dict) -> "StatusStretchSensors":
        return dataclass_from_dict(StatusStretchSensors, dict_data)  # type: ignore

    def copy(self):
        return StatusStretchSensors.from_dict(copy.copy(self.to_dict()))
