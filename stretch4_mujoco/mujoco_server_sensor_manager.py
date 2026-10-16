import csv
import os
import threading
import time
from typing import TYPE_CHECKING
import numpy as np
import mujoco

from stretch4_mujoco.enums.line_sensor import (
    LINE_SENSOR_MAX_RANGE_M,
    LINE_SENSOR_FOV_DEG,
    LINE_SENSOR_NUM_BINS,
    NUM_LINE_SENSORS,
    LineSensorCode,
    line_sensor_site_name,
)
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
            normal=None,
            nray=n_rays,
            cutoff=self.cutoff_dist,
        )

        self._dist[_geomid == -1] = -1
        self._hit_points = local_vecs * np.maximum(self._dist, 0)[:, np.newaxis]

    def get_hit_points(self) -> np.ndarray:
        return self._hit_points if self._hit_points is not None else np.empty((0, 3))

    def get_distances(self) -> np.ndarray:
        return self._dist if self._dist is not None else np.empty(0)


class NativeMjLineSensors:
    """
    Ray casts the six base line sensors with `mujoco.mj_multiRay`.

    Each sensor fans `LINE_SENSOR_NUM_BINS` rays across `LINE_SENSOR_FOV_DEG`
    in the plane of its site's x (boresight) and y (across the fan) axes. Like
    the real sensor, a bin reports depth along the boresight, not ray length,
    so a flat floor reads the same range in every bin.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        geomgroup: np.ndarray,
        bodyexclude: int,
        max_range: float = LINE_SENSOR_MAX_RANGE_M,
        cutoff_dist: float = 5.0,
    ):
        self.mj_model = mj_model
        self.geomgroup = geomgroup
        self.bodyexclude = bodyexclude
        self.max_range = max_range
        self.cutoff_dist = cutoff_dist

        self.site_ids = [
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, line_sensor_site_name(i))
            for i in range(NUM_LINE_SENSORS)
        ]
        if -1 in self.site_ids:
            raise ValueError("The model is missing a line sensor site.")

        # Bin 0 is the left edge of the fan (+y), as the real robot orders them.
        half_fov = np.deg2rad(LINE_SENSOR_FOV_DEG) / 2
        fan_angles = np.linspace(half_fov, -half_fov, LINE_SENSOR_NUM_BINS)
        self._cos_fan = np.cos(fan_angles)
        self._local_vecs = np.stack(
            (self._cos_fan, np.sin(fan_angles), np.zeros(LINE_SENSOR_NUM_BINS)), axis=-1
        )

        self._dist = np.zeros(LINE_SENSOR_NUM_BINS, dtype=np.float64)
        self._geomid = np.zeros(LINE_SENSOR_NUM_BINS, dtype=np.int32)

    def trace(
        self, mj_data: mujoco.MjData, root_body_id: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns `(ranges, codes, points)`, see the `line_sensor_*` fields of
        `StatusStretchSensors`. Points are expressed in `root_body_id`'s frame.
        """
        shape = (NUM_LINE_SENSORS, LINE_SENSOR_NUM_BINS)
        ranges = np.full(shape, np.nan)
        codes = np.full(shape, LineSensorCode.valid, dtype=np.uint8)
        points = np.full((*shape, 3), np.nan)

        root_pos = mj_data.xpos[root_body_id]
        root_mat = mj_data.xmat[root_body_id].reshape(3, 3)

        for i, site_id in enumerate(self.site_ids):
            site_pos = mj_data.site_xpos[site_id]
            world_vecs = self._local_vecs @ mj_data.site_xmat[site_id].reshape(3, 3).T

            mujoco.mj_multiRay(
                m=self.mj_model,
                d=mj_data,
                pnt=site_pos.astype(np.float64).reshape(3, 1),
                vec=world_vecs.flatten(),
                geomgroup=self.geomgroup,
                flg_static=1,
                bodyexclude=self.bodyexclude,
                geomid=self._geomid,
                dist=self._dist,
                normal=None,
                nray=LINE_SENSOR_NUM_BINS,
                cutoff=self.cutoff_dist,
            )

            is_hit = self._geomid != -1
            depth = self._dist * self._cos_fan
            is_valid = is_hit & (depth <= self.max_range)

            codes[i, ~is_hit] = LineSensorCode.no_return
            codes[i, is_hit & ~is_valid] = LineSensorCode.beyond_limit
            ranges[i, is_valid] = depth[is_valid]

            world_pts = site_pos + world_vecs[is_valid] * self._dist[is_valid, np.newaxis]
            points[i, is_valid] = (world_pts - root_pos) @ root_mat

        return ranges, codes, points


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

        # Scratch MjData that the lidar and line sensor ray casting runs against. Tracing 64k
        # rays takes ~50ms, and in the passive viewer the lock guarding mjdata
        # is the viewer lock -- the same one the 100Hz physics thread needs
        # every step. Holding it that long starves physics badly enough to drop
        # the sim to a fraction of realtime, so instead mjdata is snapshotted
        # into here under the lock (~0.01ms) and the rays are traced outside it.
        self._lidar_scratch_data = mujoco.MjData(self.mujoco_server.mjmodel)

        self.lidar_sensor_names = StretchSensors.get_sensor_names_from_mjmodel(
            self.mujoco_server.mjmodel, StretchSensors.base_lidar
        )

        # Initialize a calibration table from a real lidar
        self.lidar_wrappers: dict[str, NativeMjLidar] = {}
        csv_path = os.path.join(
            os.path.dirname(__file__), "models", "stretch_4", "lidar_calibration.csv"
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
            self.lidar_theta = theta_grid.flatten()
            self.lidar_phi = phi_grid.flatten()
        else:
            num_ray_cols = 240
            num_ray_rows = 128
            theta_grid, phi_grid = np.meshgrid(
                np.linspace(-np.pi, np.pi, num_ray_cols),
                np.linspace(-np.deg2rad(93.5), np.deg2rad(93.5), num_ray_rows),
            )
            self.lidar_theta = theta_grid.flatten()
            self.lidar_phi = phi_grid.flatten()

        # Ray trace against Group 0 geoms (environment/room/floor) and Group 3 (robot body geoms), excluding head_link
        head_body_id = mujoco.mj_name2id(self.mujoco_server.mjmodel, mujoco.mjtObj.mjOBJ_BODY, "head_link")
        scene_and_robot_geomgroup = np.array([1, 0, 0, 1, 0, 0], dtype=np.uint8)
        scene_geomgroup = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)

        # "stretch4" is the floating root body (base_footprint equivalent). urdf2mjcf
        # raises it above the URDF's reference height so the wheels don't start
        # clipped into the floor, so world-frame points must be re-expressed
        # relative to this body's *live* pose (not the compiled reference pose,
        # which the physics won't exactly hold once the robot settles/moves).
        self.root_body_id = mujoco.mj_name2id(self.mujoco_server.mjmodel, mujoco.mjtObj.mjOBJ_BODY, "stretch4")

        for site_name in ["lidar_left", "lidar_right"]:
            site_id = mujoco.mj_name2id(self.mujoco_server.mjmodel, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id != -1:
                try:
                    self.lidar_wrappers[site_name] = NativeMjLidar(
                        self.mujoco_server.mjmodel,
                        site_name=site_name,
                        cutoff_dist=30.0,
                        geomgroup=scene_and_robot_geomgroup,
                        bodyexclude=head_body_id,
                    )
                except Exception as e:
                    print(f"Failed to initialize NativeMjLidar for site '{site_name}': {e}")

        # The line sensors point down at the floor from the rim of the base, so
        # they see the scene and the robot (a lowered gripper, say) but not the
        # base shell their emitters sit on.
        self.line_sensors: NativeMjLineSensors | None = None
        if StretchSensors.line_sensors in self.sensors_to_use:
            self.line_sensors = NativeMjLineSensors(
                self.mujoco_server.mjmodel,
                geomgroup=scene_and_robot_geomgroup,
                bodyexclude=mujoco.mj_name2id(
                    self.mujoco_server.mjmodel, mujoco.mjtObj.mjOBJ_BODY, "base_link"
                ),
            )

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

    def pull_lidar_points(
        self, in_world_frame: bool = True, data: "mujoco.MjData | None" = None
    ) -> dict[str, np.ndarray]:
        """
        Traces rays for lidars and returns hit points.

        Pass `data` to trace against a snapshot of mjdata rather than the live
        one, so the caller can release the physics lock first -- see
        `_lidar_scratch_data`.
        """
        results = {}
        model = self.mujoco_server.mjmodel
        if data is None:
            data = self.mujoco_server.mjdata

        for site_name, wrapper in self.lidar_wrappers.items():
            wrapper.trace_rays(data, self.lidar_theta, self.lidar_phi, site_name=site_name)
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

                    if self.root_body_id != -1:
                        # Re-express relative to the root body's current pose so
                        # points line up with base_footprint (see __init__ note).
                        root_pos = data.xpos[self.root_body_id]
                        root_mat = data.xmat[self.root_body_id].reshape(3, 3)
                        pts = (pts - root_pos) @ root_mat

            key = "left" if "left" in site_name else "right"
            results[key] = pts

        if "left" not in results:
            results["left"] = np.empty((0, 3), dtype=np.float64)
        if "right" not in results:
            results["right"] = np.empty((0, 3), dtype=np.float64)

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
                elif sensor == StretchSensors.line_sensors:
                    # Ray cast below, outside the lock.
                    continue
                else:
                    data = self.mujoco_server.mjdata.sensor(sensor.name).data

                sensor_status.set_data(sensor, data)

            if self.lidar_wrappers or self.line_sensors is not None:
                # Cheap memcpy so the expensive ray casting below can run with
                # the physics lock released, see `_lidar_scratch_data`.
                mujoco.mj_copyData(
                    self._lidar_scratch_data,
                    self.mujoco_server.mjmodel,
                    self.mujoco_server.mjdata,
                )

        if self.line_sensors is not None:
            (
                sensor_status.line_sensor_ranges,
                sensor_status.line_sensor_codes,
                sensor_status.line_sensor_points,
            ) = self.line_sensors.trace(self._lidar_scratch_data, self.root_body_id)

        self.mujoco_server.data_proxies.set_sensors(sensor_status)

        if self.lidar_wrappers:
            lidar_pts = self.pull_lidar_points(
                in_world_frame=True, data=self._lidar_scratch_data
            )
            self.mujoco_server.data_proxies.set_lidar_points(lidar_pts)


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
