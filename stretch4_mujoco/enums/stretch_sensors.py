from enum import Enum
from functools import cache

import mujoco
import mujoco._structs


class StretchSensors(Enum):
    """
    An enum of the sensors available to the simulation.
    """

    base_gyro = 0
    base_accel = 1
    base_lidar = 2

    @staticmethod
    def all() -> list["StretchSensors"]:
        """
        Returns all the available sensors
        """
        return [sensor for sensor in StretchSensors]

    @staticmethod
    def none() -> list["StretchSensors"]:
        """
        Short-hand for not using any sensor.
        """
        return []

    @staticmethod
    def from_mjmodel(mjmodel: mujoco._structs.MjModel) -> "list[StretchSensors]":
        """Get all the sensors in an mjmodel. We don't have the spec, only the compiled model. We're gonna try to find all the sensors."""
        sensors: set[StretchSensors] = set()
        remaining_sensors = [s for s in StretchSensors]
        try:
            index = 0
            while True:
                # We have no way of pulling the number of sensors via API.
                # When we exceed the sensors in mjmodel.sensor, an IndexError will be thrown.
                name = mjmodel.sensor(index).name
                index += 1
                for sensor in remaining_sensors:
                    # base_lidar is replicated, so it's called base_lidar000 -> base_lidar359 in this list, this is why we're using `sensor.name in name` below:
                    if sensor.name in name:
                        sensors.add(sensor)
                        remaining_sensors.remove(sensor)

                if len(remaining_sensors) == 0:
                    break

        except IndexError: ...

        # If base_lidar wasn't found in XML sensors, check if lidar sites exist in model
        if StretchSensors.base_lidar in remaining_sensors:
            for index in range(mjmodel.nsite):
                site_name = mujoco.mj_id2name(mjmodel, mujoco.mjtObj.mjOBJ_SITE, index)
                if site_name and "lidar" in site_name:
                    sensors.add(StretchSensors.base_lidar)
                    break

        return list(sensors)


    @staticmethod
    def get_sensor_names_from_mjmodel(mjmodel: mujoco._structs.MjModel, sensor:"StretchSensors") -> list[str]:
        """Get all the sensors in an mjmodel matching the sensor type `sensor`. We don't have the spec, only the compiled model, we have to iteratively find all the sensors.

        This is useful for sensors that are replicated, like base_lidar. For example, if the resolution is 360, then base_lidar will have 360 sensors named base_lidar000 -> base_lidar359. Since they are all lidars, we return these names.
        """
        sensor_names = []
        try:
            index = 0
            while True:
                # We have no way of pulling the number of sensors via API.
                # When we exceed the sensors in mjmodel.sensor, an IndexError will be thrown.
                name = mjmodel.sensor(index).name
                index += 1
                if sensor.name in name:
                    sensor_names.append(name)

        except IndexError: ...

        return sensor_names
