import csv
import os
import threading
import time
from typing import TYPE_CHECKING
import numpy as np
import mujoco

from stretch4_mujoco.enums.stretch_sensors import StretchSensors
from stretch4_mujoco.datamodels.status_stretch_sensors import StatusStretchSensors
from stretch4_mujoco.utils import FpsCounter

if TYPE_CHECKING:
    from stretch4_mujoco.mujoco_server import MujocoServer


class NativeMjLidar:
    """
    Native MuJoCo ray tracer using official C++ bindings (`mujoco.mj_multiRay`).
    Eliminates external third-party dependencies while providing 100% identical performance and accuracy.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        site_name: str,
        cutoff_dist: float = 30.0,
        geomgroup: np.ndarray | None = None,
        bodyexclude: int = -1,
    ):
        self.mj_model = mj_model
        self.site_name = site_name
        self.site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        self.cutoff_dist = cutoff_dist
        self.geomgroup = geomgroup if geomgroup is not None else np.ones(6, dtype=np.uint8)
        self.bodyexclude = bodyexclude
        self._hit_points: np.ndarray | None = None
        self._dist: np.ndarray | None = None

    def trace_rays(self, mj_data: mujoco.MjData, ray_theta: np.ndarray, ray_phi: np.ndarray, site_name: str | None = None) -> None:
        target_site_id = self.site_id
        if site_name is not None and site_name != self.site_name:
            target_site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)

        if target_site_id == -1:
            return

        n_rays = len(ray_phi)
        site_pos = mj_data.site_xpos[target_site_id]
        site_mat = mj_data.site_xmat[target_site_id].reshape(3, 3)

        x = np.cos(ray_phi) * np.cos(ray_theta)
        y = np.cos(ray_phi) * np.sin(ray_theta)
        z = np.sin(ray_phi)
        local_vecs = np.stack((x, y, z), axis=-1)

        world_vecs = local_vecs @ site_mat.T
        world_vecs /= np.linalg.norm(world_vecs, axis=1, keepdims=True)
        world_vecs_flat = world_vecs.flatten().astype(np.float64)

        pnt = site_pos.astype(np.float64).reshape(3, 1)
        self._dist = np.full(n_rays, self.cutoff_dist, dtype=np.float64)
        _geomid = np.zeros(n_rays, dtype=np.int32)

        mujoco.mj_multiRay(
            m=self.mj_model,
            d=mj_data,
            pnt=pnt,
            vec=world_vecs_flat,
            geomgroup=self.geomgroup,
            flg_static=1,
            bodyexclude=self.bodyexclude,
            geomid=_geomid,
            dist=self._dist,
            nray=n_rays,
            cutoff=self.cutoff_dist,
        )

        self._dist[_geomid == -1] = -1
        self._hit_points = local_vecs * np.maximum(self._dist, 0)[:, np.newaxis]

    def get_hit_points(self) -> np.ndarray:
        return self._hit_points if self._hit_points is not None else np.empty((0, 3))

    def get_distances(self) -> np.ndarray:
        return self._dist if self._dist is not None else np.empty(0)


class MujocoServerSensorManagerSync:
    """
    Handles rendering scene sensors to a buffer.

    Call `pull_sensor_data_at_sensor_rate()` from the UI thread and the sensors will be rendered at the specified `sensor_hz`.
    """

    def __init__(
        self, sensor_hz: float, sensors_to_use: list[StretchSensors], mujoco_server: "MujocoServer"
    ) -> None:

        self.mujoco_server = mujoco_server

        self.sensor_rate = 1 / sensor_hz  # Hz to seconds

        self.sensors_to_use = sensors_to_use

        self.sensor_fps_counter = FpsCounter()

        self.time_start = time.perf_counter()

        self.sensor_lock = threading.Lock()

        self.lidar_sensor_names = StretchSensors.get_sensor_names_from_mjmodel(
            self.mujoco_server.mjmodel, StretchSensors.base_lidar
        )

        # Initialize native MuJoCo LiDAR wrappers for Hesai J128 3D lidars using official Hesai JT128 calibration table
        self.hesai_wrappers: dict[str, NativeMjLidar] = {}
        csv_path = os.path.join(
            os.path.dirname(__file__), "models", "stretch_4", "hesai_jt128_calibration.csv"
        )
        elevations_deg, azimuth_offsets_deg = [], []
        if os.path.exists(csv_path):
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    elevations_deg.append(float(row["Elevation"]))
                    azimuth_offsets_deg.append(float(row["Azimuth"]))

        if elevations_deg:
            elevations_rad = np.deg2rad(np.array(elevations_deg))
            azimuth_offsets_rad = np.deg2rad(np.array(azimuth_offsets_deg))
            num_azimuth_steps = 250
            spin_angles = np.linspace(-np.pi, np.pi, num_azimuth_steps)
            theta_grid = spin_angles[:, None] + azimuth_offsets_rad[None, :]
            phi_grid = np.tile(elevations_rad[None, :], (num_azimuth_steps, 1))
            theta_grid = (theta_grid + np.pi) % (2 * np.pi) - np.pi
            self.hesai_theta = theta_grid.flatten()
            self.hesai_phi = phi_grid.flatten()
        else:
            num_ray_cols = 240
            num_ray_rows = 128
            theta_grid, phi_grid = np.meshgrid(
                np.linspace(-np.pi, np.pi, num_ray_cols),
                np.linspace(-np.deg2rad(93.5), np.deg2rad(93.5), num_ray_rows),
            )
            self.hesai_theta = theta_grid.flatten()
            self.hesai_phi = phi_grid.flatten()

        # Ray trace against Group 0 geoms (environment/room/floor) and Group 3 (robot body geoms), excluding head_link
        head_body_id = mujoco.mj_name2id(self.mujoco_server.mjmodel, mujoco.mjtObj.mjOBJ_BODY, "head_link")
        scene_and_robot_geomgroup = np.array([1, 0, 0, 1, 0, 0], dtype=np.uint8)
        scene_geomgroup = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)

        for site_name in ["lidar_left", "lidar_right"]:
            site_id = mujoco.mj_name2id(self.mujoco_server.mjmodel, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id != -1:
                try:
                    self.hesai_wrappers[site_name] = NativeMjLidar(
                        self.mujoco_server.mjmodel,
                        site_name=site_name,
                        cutoff_dist=30.0,
                        geomgroup=scene_and_robot_geomgroup,
                        bodyexclude=head_body_id,
                    )
                except Exception as e:
                    print(f"Failed to initialize NativeMjLidar for site '{site_name}': {e}")

        # Initialize native MuJoCo LiDAR wrapper for 2D base lidar if site 'lidar' exists
        site_id_base = mujoco.mj_name2id(self.mujoco_server.mjmodel, mujoco.mjtObj.mjOBJ_SITE, "lidar")
        if site_id_base != -1:
            try:
                self.base_lidar_wrapper = NativeMjLidar(
                    self.mujoco_server.mjmodel,
                    site_name="lidar",
                    cutoff_dist=10.0,
                    geomgroup=scene_geomgroup,
                )
                self.base_lidar_theta = np.linspace(-np.pi, np.pi, 360)
                self.base_lidar_phi = np.zeros(360)
            except Exception:
                self.base_lidar_wrapper = None
        else:
            self.base_lidar_wrapper = None

    def is_ready_to_pull_sensor_data(self, is_sleep_until_ready: bool = False):
        """
        Checks to see if a duration of time has passed since the last call
        to this function to render sensor at the specified `self.sensor_rate`.
        """
        elapsed = time.perf_counter() - self.time_start
        if elapsed < self.sensor_rate:
            # If we're not ready to render sensor, don't render:
            if not is_sleep_until_ready:
                return False
            # sleep until ready:
            time.sleep(self.sensor_rate - elapsed)

        self.time_start = time.perf_counter()
        return True

    def pull_sensor_data_at_sensor_rate(self, is_sleep_until_ready: bool):
        """
        Call this on the UI thread to render sensor data.
        """

        if not self.is_ready_to_pull_sensor_data(is_sleep_until_ready):
            return

        self._pull_sensor_data()

        self.sensor_fps_counter.tick()

    def pull_hesai_lidar_points(self, in_world_frame: bool = True) -> list[tuple[str, np.ndarray]]:
        """
        Traces rays for Hesai J128 Lidars using MuJoCo-LiDAR and returns hit points.
        """
        results = []
        model = self.mujoco_server.mjmodel
        data = self.mujoco_server.mjdata

        for site_name, wrapper in self.hesai_wrappers.items():
            wrapper.trace_rays(data, self.hesai_theta, self.hesai_phi, site_name=site_name)
            local_pts = wrapper.get_hit_points()
            dists = wrapper.get_distances()
            valid_mask = (dists > 0) & (dists < wrapper.cutoff_dist)
            pts = local_pts[valid_mask]

            if in_world_frame and len(pts) > 0:
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                if site_id != -1:
                    site_pos = data.site_xpos[site_id]
                    site_mat = data.site_xmat[site_id].reshape(3, 3)
                    pts = pts @ site_mat.T + site_pos

            results.append((site_name, pts))

        return results

    def _compute_lidar_rays(self) -> np.ndarray:
        if self.base_lidar_wrapper is not None:
            self.base_lidar_wrapper.trace_rays(
                self.mujoco_server.mjdata, self.base_lidar_theta, self.base_lidar_phi
            )
            dists = self.base_lidar_wrapper.get_distances().copy()
            dists[dists < 0] = 10.0
            return dists

        model = self.mujoco_server.mjmodel
        data = self.mujoco_server.mjdata
        geom_id = np.zeros(1, dtype=np.int32)
        ranges = []
        for i in range(360):
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"lidar{i:03d}")
            if site_id == -1:
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"lidar_{i}")
            if site_id != -1:
                pnt = data.site_xpos[site_id]
                vec = data.site_xmat[site_id].reshape(3, 3)[:, 2]
                dist = mujoco.mj_ray(model, data, pnt, vec, None, 1, -1, geom_id)
                if dist < 0 or dist > 10.0:
                    dist = 10.0
                ranges.append(dist)
            else:
                ranges.append(10.0)
        return np.array(ranges)

    def _pull_sensor_data(self):
        """
        Pull data from the simulator.
        """
        lock_mgr = getattr(self.mujoco_server, "camera_manager", None)
        camera_lock = lock_mgr.camera_lock if lock_mgr and hasattr(lock_mgr, "camera_lock") else self.sensor_lock

        with camera_lock:
            sensor_status = StatusStretchSensors.default()
            sensor_status.time = self.mujoco_server.mjdata.time
            sensor_status.fps = self.sensor_fps_counter.fps

            for sensor in self.sensors_to_use:
                data: np.ndarray
                if sensor == StretchSensors.base_lidar:
                    if self.lidar_sensor_names:
                        data = np.array(
                            [
                                d
                                for lidar_name in self.lidar_sensor_names
                                for d in self.mujoco_server.mjdata.sensor(lidar_name).data
                            ]
                        )
                    else:
                        data = self._compute_lidar_rays()
                else:
                    data = self.mujoco_server.mjdata.sensor(sensor.name).data

                sensor_status.set_data(sensor, data)

            self.mujoco_server.data_proxies.set_sensors(sensor_status)

            if self.hesai_wrappers:
                hesai_pts = self.pull_hesai_lidar_points(in_world_frame=True)
                self.mujoco_server.data_proxies.set_hesai_lidar_points(hesai_pts)


class MujocoServerSensorManagerThreaded(MujocoServerSensorManagerSync):
    """
    Starts a sensor loop on init to pull sensor data using threading.
    """

    def __init__(
        self,
        sensor_hz: float,
        sensors_to_use: list[StretchSensors],
        mujoco_server: "MujocoServer",
    ):
        """
        `use_threadpool_executor` will use a ThreadPoolExecutor to render all sensors. Setting to false will render each one synchronously.

        `use_sensor_thread` can be set to false to use the ThreadPoolExecutor without the sensor thread. `pull_sensor_data_at_sensor_rate()` must be called on the UI thread if this mode is used.
        """

        super().__init__(sensor_hz, sensors_to_use, mujoco_server)

        self.sensors_thread = threading.Thread(target=self._sensor_loop, daemon=True)
        self.sensors_thread.start()

    def _sensor_loop(self):
        """
        This is the thread loop that handles sensor rendering.
        """

        while (
            self.mujoco_server.data_proxies.get_status().time == 0
        ) and not self.mujoco_server._is_requested_to_stop():
            # wait for sim to start
            time.sleep(0.1)

        while not self.mujoco_server._is_requested_to_stop():

            if not self.is_ready_to_pull_sensor_data(is_sleep_until_ready=True):
                continue

            self._pull_sensor_data()

            self.sensor_fps_counter.tick()
