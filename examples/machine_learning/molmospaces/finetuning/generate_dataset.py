"""
Generate Stretch demonstrations, then export them for fine-tuning.

One command for the whole data half of the pipeline:

    # smallest thing that proves the setup works: 2 episodes, 1 house
    python -m examples.machine_learning.molmospaces.finetuning.generate_dataset \\
        --task debug

    # the same, watched live in MuJoCo's viewer rather than read off a log
    python -m examples.machine_learning.molmospaces.finetuning.generate_dataset \
        --task debug --output-dir data/stretch_debug --no-export --visualize

    # slow down playback in the viewer (e.g. 2x slower than real-time)
    python -m examples.machine_learning.molmospaces.finetuning.generate_dataset \
        --task debug --output-dir data/stretch_debug --no-export --visualize --slow_rate 2.0

    # a real run: 2000 pick episodes across procthor-objaverse, 8 workers
    python -m examples.machine_learning.molmospaces.finetuning.generate_dataset \
        --task pick --episodes 2000 --num-workers 8 --output-dir data/stretch_pick

    # several families pooled into one training set
    python -m examples.machine_learning.molmospaces.finetuning.generate_dataset \
        --task pick --task pnp --task open --episodes 1000 \
        --output-dir data/stretch_manipulation

    # rollouts already on disk; just re-export them
    python -m examples.machine_learning.molmospaces.finetuning.generate_dataset \
        --rollouts data/stretch_pick/rollouts --output-dir data/stretch_pick

Two stages, either of which can be run alone (`--no-export`, `--rollouts`):

1. **Generate.** MolmoSpaces' `ParallelRolloutRunner` over one of the Stretch
   datagen configs in `datagen_configs.py`, which writes HDF5 trajectories plus
   side-car MP4s under `<output-dir>/rollouts/<task>/`.
2. **Export.** `lerobot_export.py` turns those into a LeRobot dataset under
   `<output-dir>/lerobot/`, which is what a fine-tuning run consumes.

The generation stage is the expensive one -- it is a full physics rollout with
rendering per episode -- so it keeps its raw output rather than streaming
straight into the export. Re-exporting into a different action space is then
seconds rather than hours, which matters because `--action-space` is the choice
you are most likely to want to change your mind about.
"""

from __future__ import annotations

import ctypes
import gc
import importlib
import logging
import os
import pprint
import random
import time
import traceback
from pathlib import Path
from typing import Any

import click
import mujoco
import numpy as np

from examples.machine_learning.molmospaces.finetuning.datagen_configs import (
    DATAGEN_CONFIGS,
    qualified_config_name,
)
from examples.machine_learning.molmospaces.finetuning.lerobot_export import (
    ACTION_SPACES,
    export_lerobot_dataset,
)
from molmo_spaces.data_generation.config_registry import get_config_class
from molmo_spaces.data_generation.pipeline import (
    ParallelRolloutRunner,
    cleanup_context,
    cleanup_episode_resources,
    get_worker_logger,
    log_memory_usage,
    mp_context,
    setup_house_dirs,
    setup_policy,
    setup_viewer,
    worker_stdout_context,
)
from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask
from molmo_spaces.utils.profiler_utils import DatagenProfiler
from molmo_spaces.utils.save_utils import prepare_episode_for_saving, save_trajectories

log = logging.getLogger(__name__)


def trim_memory() -> None:
    """Forces Python GC and glibc heap trimmer to release unused memory back to the OS.

    Only ever a second-order effect: the camera frames that dominate a worker's
    footprint are multi-hundred-kilobyte numpy arrays, which glibc mostly serves
    with `mmap` and returns at `free()` without help. This is here for the churn
    of small objects underneath them, and is only worth calling at a point where
    the big allocations have *already* gone out of scope -- see
    `flush_episode_to_disk`, which is what actually bounds the footprint.
    """
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def flush_episode_to_disk(
    worker_logger: Any,
    history: dict,
    sensor_suite: Any,
    save_dir: Path,
    exp_config: Any,
    batch_suffix: str,
    episode_idx: int,
    datagen_profiler: Any = None,
) -> dict | None:
    """Encode one finished episode's videos and drop its camera frames immediately.

    This is the whole memory story of a datagen run. An episode's observation
    history is ~4.3 MiB per step -- five 640x368 RGB streams plus a float32 depth
    stream -- so a 300-step episode is ~1.3 GiB and a 500-step one ~2.1 GiB, and
    *none* of it lands in the HDF5: the frames go out as side-car MP4s.

    The pipeline's own `save_house_trajectories` does this encoding once per
    house, at the end, which means all `samples_per_house` episodes are held as
    raw frames simultaneously -- 4 x 2.1 GiB per worker at the default, times
    `--num-workers`. Calling it per episode instead trades nothing (the encoding
    work is identical and the MP4 filenames come out the same) for a peak of one
    episode's frames rather than a houseful.

    What comes back is the camera-stripped batched tensor dict -- the ~10 MiB of
    poses, joint states and per-camera intrinsics that actually go into the HDF5
    -- so accumulating those across a house costs nothing worth counting.

    Returns:
        The prepared episode, or None if there was nothing to save.
    """
    os.makedirs(save_dir, exist_ok=True)

    if datagen_profiler is not None:
        datagen_profiler.start("save_batch_prep")
    try:
        prepared = prepare_episode_for_saving(
            history,
            sensor_suite,
            fps=exp_config.fps,
            save_dir=save_dir,
            episode_idx=episode_idx,
            save_file_suffix=batch_suffix,
        )
    except Exception as e:
        # A failed encode costs one episode, not the house, and must not be
        # mistaken for a rollout failure by the caller's retry counters.
        worker_logger.error(f"Failed to prepare episode {episode_idx} for saving: {e}")
        traceback.print_exc()
        prepared = None
    finally:
        if datagen_profiler is not None:
            datagen_profiler.end("save_batch_prep")

    return prepared


def save_prepared_trajectories(
    worker_logger: Any,
    prepared_episodes: list[dict],
    save_dir: Path,
    exp_config: Any,
    batch_suffix: str,
    datagen_profiler: Any = None,
    batch_num: int | None = None,
    total_batches: int | None = None,
) -> None:
    """Write already-prepared (camera-stripped) episodes into the house's HDF5.

    The back half of `save_house_trajectories`; the front half -- video encoding
    and frame release -- has already happened per episode in
    `flush_episode_to_disk`.
    """
    if not prepared_episodes:
        worker_logger.warning(f"No trajectory data to save for {save_dir.name}")
        return

    batch_info = f" batch {batch_num}/{total_batches}" if batch_num is not None else ""
    worker_logger.info(
        f"Saving trajectory data for {save_dir.name}{batch_info}: "
        f"{len(prepared_episodes)} episodes"
    )

    try:
        t_start = time.perf_counter()
        if datagen_profiler is not None:
            datagen_profiler.start("save_trajectories")
        save_trajectories(
            prepared_episodes,
            save_dir=save_dir,
            fps=exp_config.fps,
            save_file_suffix=batch_suffix,
            save_mp4s=True,
            logger=worker_logger,
        )
        if datagen_profiler is not None:
            datagen_profiler.end("save_trajectories")
        worker_logger.info(
            f"Successfully saved trajectory data for {save_dir.name} "
            f"in {time.perf_counter() - t_start:.2f}s"
        )
    except Exception as e:
        worker_logger.error(f"Failed to save trajectory data for {save_dir.name}: {e}")
        traceback.print_exc()


class StretchRerunVisualizer:
    """Streams 3D robot meshes, object meshes, coordinate frames, target grasp, and waypoints to Rerun."""

    def __init__(self, spawn: bool = True, port: int = 9876):
        self._spawn = spawn
        self._port = port
        self._initialized = False
        self._logged_meshes: set[str] = set()
        self._last_logged_waypoint_idx = -1
        self._logged_grasp_lost = False

    def start_episode(self, episode_seed: int, task: Any, policy: Any = None) -> None:
        """Starts a new Rerun recording for each episode."""
        try:
            import uuid
            import rerun as rr
            import rerun.blueprint as rrb

            rec_id = f"episode_{episode_seed}_{uuid.uuid4().hex[:8]}"
            app_id = "Stretch4 Datagen"

            blueprint = rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Tabs(
                        rrb.Spatial3DView(
                            origin="world",
                            contents=["+ $origin/**", "- $origin/scene_objects/**"],
                            name="3D Scene",
                        ),
                        rrb.Spatial3DView(
                            origin="world",
                            contents=["+ $origin/**"],
                            name="3D Scene (Complete)",
                        ),
                        active_tab=0,
                    ),
                    rrb.Vertical(
                        rrb.TextDocumentView(origin="planner/waypoint", name="Current Waypoint"),
                        rrb.TextLogView(origin="logs/waypoints", name="Waypoint Log"),
                    ),
                ),
                collapse_panels=True,
            )

            if not self._initialized:
                rr.init(app_id, recording_id=rec_id, spawn=False, default_blueprint=blueprint)
                if self._spawn:
                    try:
                        rr.spawn(port=self._port, memory_limit="4GB")
                    except Exception as e:
                        log.debug(f"rr.spawn note: {e}")
                self._initialized = True
            else:
                rr.init(app_id, recording_id=rec_id, spawn=False, default_blueprint=blueprint)

            if self._spawn:
                try:
                    rr.connect_grpc(f"rerun+http://127.0.0.1:{self._port}/proxy")
                except Exception as e:
                    log.debug(f"rr.connect_grpc note: {e}")

            try:
                rr.send_recording_name(f"Episode {episode_seed}")
            except Exception:
                pass

            try:
                rr.send_blueprint(blueprint)
            except Exception as e:
                log.debug(f"Could not send Rerun blueprint: {e}")

            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
            self._logged_meshes.clear()
            self._last_logged_waypoint_idx = -1
            self._logged_grasp_lost = False

            pickup_obj_name = self._get_pickup_object_name(task)
            if hasattr(task, "env") and hasattr(task.env, "current_model"):
                self.setup_meshes(task.env.current_model, pickup_obj_name)
        except Exception as e:
            log.warning(f"Failed to start Rerun episode recording: {e}")

    @staticmethod
    def _unwrap_policy(policy: Any) -> Any:
        """Unwraps wrappers around policy to reach the underlying waypoint/grasp planner."""
        curr = policy
        for _ in range(5):
            if curr is None:
                break
            if hasattr(curr, "_plan") or hasattr(curr, "_grasp"):
                return curr
            if hasattr(curr, "policy"):
                curr = curr.policy
            elif hasattr(curr, "_policy"):
                curr = curr._policy
            elif hasattr(curr, "inner_policy"):
                curr = curr.inner_policy
            elif hasattr(curr, "wrapped_policy"):
                curr = curr.wrapped_policy
            else:
                break
        return curr

    @staticmethod
    def _get_pickup_object_name(task: Any) -> str | None:
        if hasattr(task, "get_task_objects"):
            try:
                objs = task.get_task_objects()
                if isinstance(objs, dict):
                    for k in ["pickup_obj", "target_obj", "pickup_object", "target_object", "manipulated_object"]:
                        if k in objs and objs[k]:
                            return str(objs[k])
            except Exception:
                pass
        if hasattr(task, "config") and hasattr(task.config, "task_config"):
            tc = task.config.task_config
            for k in ["pickup_obj_name", "target_obj_name", "object_name"]:
                val = getattr(tc, k, None)
                if val:
                    return str(val)
        for attr in ["target_obj_name", "pickup_obj_name", "target_object"]:
            val = getattr(task, attr, None)
            if val and isinstance(val, str):
                return val
        return None

    @staticmethod
    def _get_object_body_ids(model: Any, pickup_obj_name: str | None) -> set[int]:
        """Finds all body IDs belonging to the manipulated object and its children."""
        if not pickup_obj_name:
            return set()
        root_ids = set()
        for b_id in range(1, model.nbody):  # Skip body 0 (world)
            b_name = model.body(b_id).name
            if "robot_0" in b_name:
                continue
            if (
                pickup_obj_name == b_name
                or b_name.startswith(f"{pickup_obj_name}_")
                or b_name.startswith(f"{pickup_obj_name}|")
                or (
                    pickup_obj_name in b_name
                    and not any(
                        k in b_name.lower()
                        for k in [
                            "wall",
                            "floor",
                            "room",
                            "house",
                            "ceiling",
                            "counter",
                            "table",
                            "chair",
                            "sofa",
                            "bed",
                            "shelf",
                        ]
                    )
                )
            ):
                root_ids.add(b_id)

        descendants = set(root_ids)
        parent_ids = model.body_parentid
        queue = list(root_ids)
        while queue:
            curr = queue.pop(0)
            for i, pid in enumerate(parent_ids):
                if i > 0 and pid == curr and i not in descendants:
                    b_name = model.body(i).name
                    if "robot_0" not in b_name:
                        descendants.add(i)
                        queue.append(i)
        return descendants

    def setup_meshes(self, model: Any, pickup_obj_name: str | None) -> None:
        if not self._initialized:
            return
        import rerun as rr
        from scipy.spatial.transform import Rotation as R

        def _build_mesh3d(g: int, verts_local: np.ndarray, faces: np.ndarray, is_robot: bool) -> Any:
            g_mesh = model.geom_dataid[g] if hasattr(model, "geom_dataid") else -1
            uvs = None
            tex_img = None

            if g_mesh >= 0 and hasattr(model, "mesh_texcoordadr") and hasattr(model, "mesh_texcoord"):
                texadr = int(model.mesh_texcoordadr[g_mesh])
                texnum = int(model.mesh_texcoordnum[g_mesh])
                if texnum > 0:
                    uvs = model.mesh_texcoord[texadr : texadr + texnum]

            mat_id = int(model.geom_matid[g]) if hasattr(model, "geom_matid") else -1
            if mat_id >= 0 and hasattr(model, "mat_texid") and hasattr(model, "tex_data"):
                tex_ids = model.mat_texid[mat_id]
                tex_id = -1
                for tid in tex_ids:
                    if int(tid) >= 0:
                        tex_id = int(tid)
                        break
                if 0 <= tex_id < model.ntex and uvs is not None and len(uvs) == len(verts_local):
                    w = int(model.tex_width[tex_id])
                    h = int(model.tex_height[tex_id])
                    adr = int(model.tex_adr[tex_id])
                    tex_img = model.tex_data[adr : adr + w * h * 3].reshape(h, w, 3)

            if tex_img is not None and uvs is not None:
                return rr.Mesh3D(
                    vertex_positions=verts_local,
                    triangle_indices=faces,
                    vertex_texcoords=uvs,
                    albedo_texture=tex_img,
                )

            rgba = None
            if mat_id >= 0 and hasattr(model, "mat_rgba"):
                mat_rgba = model.mat_rgba[mat_id]
                if not np.allclose(mat_rgba[:3], 1.0) and not np.allclose(mat_rgba[:3], 0.5):
                    rgba = (mat_rgba * 255).astype(np.uint8).tolist()

            if rgba is None and hasattr(model, "geom_rgba"):
                geom_rgba = model.geom_rgba[g]
                if is_robot or not np.allclose(geom_rgba[:3], 0.5):
                    rgba = (geom_rgba * 255).astype(np.uint8).tolist()

            if rgba is None:
                rgba = [220, 225, 235, 255] if is_robot else [225, 120, 50, 255]

            return rr.Mesh3D(
                vertex_positions=verts_local,
                triangle_indices=faces,
                albedo_factor=rgba,
            )

        # 1. Stretch 4 robot meshes (from stretch4_urdf)
        robot_visual_links = set()
        for g in range(model.ngeom):
            b_id = model.geom_bodyid[g]
            b_name = model.body(b_id).name
            g_mesh = model.geom_dataid[g]
            if "robot_0" in b_name and g_mesh >= 0:
                mesh_name = model.mesh(g_mesh).name
                if "collision" not in mesh_name:
                    robot_visual_links.add(b_name)

        for g in range(model.ngeom):
            b_id = model.geom_bodyid[g]
            b_name = model.body(b_id).name
            g_mesh = model.geom_dataid[g]

            geom_key = f"robot/{b_name}/{g}"
            if "robot_0" in b_name and g_mesh >= 0:
                mesh_name = model.mesh(g_mesh).name
                if "collision" in mesh_name and b_name in robot_visual_links:
                    continue

                if geom_key not in self._logged_meshes:
                    vertadr = int(model.mesh_vertadr[g_mesh])
                    vertnum = int(model.mesh_vertnum[g_mesh])
                    faceadr = int(model.mesh_faceadr[g_mesh])
                    facenum = int(model.mesh_facenum[g_mesh])
                    verts = model.mesh_vert[vertadr : vertadr + vertnum].astype(np.float32)
                    faces = model.mesh_face[faceadr : faceadr + facenum].astype(np.uint32)

                    rot = R.from_quat(model.geom_quat[g], scalar_first=True)
                    verts_local = rot.apply(verts) + model.geom_pos[g]

                    mesh_3d = _build_mesh3d(g, verts_local, faces, is_robot=True)
                    b_key = b_name.replace("/", "_")
                    rr.log(
                        f"world/robot/{b_key}/geom_{g}",
                        mesh_3d,
                        static=True,
                    )
                    self._logged_meshes.add(geom_key)

        # 2. Manipulated object meshes & primitives from MolmoSpaces scene assets
        obj_body_ids = self._get_object_body_ids(model, pickup_obj_name)
        obj_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in obj_body_ids]
        has_visual_mesh = any(
            model.geom_dataid[g] >= 0
            and "collision" not in model.mesh(model.geom_dataid[g]).name
            for g in obj_geoms
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        )

        for g in obj_geoms:
            b_id = model.geom_bodyid[g]
            b_name = model.body(b_id).name
            g_type = model.geom_type[g]
            g_mesh = model.geom_dataid[g]
            mesh_name = model.mesh(g_mesh).name if g_mesh >= 0 else ""

            if has_visual_mesh and ("collision" in mesh_name or g_mesh < 0):
                continue

            obj_geom_key = f"object/{b_name}/{g}"
            if obj_geom_key not in self._logged_meshes:
                b_key = b_name.replace("/", "_")
                if g_type == mujoco.mjtGeom.mjGEOM_MESH and g_mesh >= 0:
                    vertadr = int(model.mesh_vertadr[g_mesh])
                    vertnum = int(model.mesh_vertnum[g_mesh])
                    faceadr = int(model.mesh_faceadr[g_mesh])
                    facenum = int(model.mesh_facenum[g_mesh])
                    verts = model.mesh_vert[vertadr : vertadr + vertnum].astype(np.float32)
                    faces = model.mesh_face[faceadr : faceadr + facenum].astype(np.uint32)

                    rot = R.from_quat(model.geom_quat[g], scalar_first=True)
                    verts_local = rot.apply(verts) + model.geom_pos[g]

                    mesh_3d = _build_mesh3d(g, verts_local, faces, is_robot=False)
                    rr.log(
                        f"world/object/{b_key}/geom_{g}",
                        mesh_3d,
                        static=True,
                    )
                elif g_type == mujoco.mjtGeom.mjGEOM_BOX:
                    color = (model.geom_rgba[g] * 255).astype(np.uint8).tolist() if hasattr(model, "geom_rgba") else [225, 120, 50, 255]
                    rr.log(
                        f"world/object/{b_key}/geom_{g}",
                        rr.Boxes3D(
                            half_sizes=[model.geom_size[g]],
                            centers=[model.geom_pos[g]],
                            colors=[color],
                        ),
                        static=True,
                    )
                elif g_type in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                    s = float(model.geom_size[g][0])
                    color = (model.geom_rgba[g] * 255).astype(np.uint8).tolist() if hasattr(model, "geom_rgba") else [225, 120, 50, 255]
                    rr.log(
                        f"world/object/{b_key}/geom_{g}",
                        rr.Ellipsoids3D(
                            half_sizes=[[s, s, s]],
                            centers=[model.geom_pos[g]],
                            colors=[color],
                        ),
                        static=True,
                    )
                elif g_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    r = float(model.geom_size[g][0])
                    h = float(model.geom_size[g][1]) * 2
                    color = (model.geom_rgba[g] * 255).astype(np.uint8).tolist() if hasattr(model, "geom_rgba") else [225, 120, 50, 255]
                    rr.log(
                        f"world/object/{b_key}/geom_{g}",
                        rr.Cylinders3D(
                            radii=[r],
                            lengths=[h],
                            centers=[model.geom_pos[g]],
                            colors=[color],
                        ),
                        static=True,
                    )
                elif g_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    r = float(model.geom_size[g][0])
                    h = float(model.geom_size[g][1]) * 2
                    color = (model.geom_rgba[g] * 255).astype(np.uint8).tolist() if hasattr(model, "geom_rgba") else [225, 120, 50, 255]
                    rr.log(
                        f"world/object/{b_key}/geom_{g}",
                        rr.Capsules3D(
                            radii=[r],
                            lengths=[h],
                            centers=[model.geom_pos[g]],
                            colors=[color],
                        ),
                        static=True,
                    )
                self._logged_meshes.add(obj_geom_key)

        # 3. Other scene objects / environment (furniture, fixtures, walls, floor, tables)
        scene_geoms = [
            g
            for g in range(model.ngeom)
            if model.geom_bodyid[g] not in obj_body_ids
            and "robot_0" not in model.body(model.geom_bodyid[g]).name
        ]
        scene_visual_bodies = set()
        for g in scene_geoms:
            b_id = model.geom_bodyid[g]
            g_mesh = model.geom_dataid[g] if hasattr(model, "geom_dataid") else -1
            if g_mesh >= 0:
                mesh_name = model.mesh(g_mesh).name
                if "collision" not in mesh_name:
                    scene_visual_bodies.add(b_id)

        for g in scene_geoms:
            b_id = model.geom_bodyid[g]
            b_name = model.body(b_id).name
            g_type = model.geom_type[g]
            g_mesh = model.geom_dataid[g] if hasattr(model, "geom_dataid") else -1
            mesh_name = model.mesh(g_mesh).name if g_mesh >= 0 else ""

            if b_id in scene_visual_bodies and ("collision" in mesh_name or g_mesh < 0):
                continue

            scene_geom_key = f"scene_objects/{b_name}/{g}"
            if scene_geom_key not in self._logged_meshes:
                b_key = b_name.replace("/", "_") if b_name else f"body_{b_id}"
                if g_type == mujoco.mjtGeom.mjGEOM_MESH and g_mesh >= 0:
                    vertadr = int(model.mesh_vertadr[g_mesh])
                    vertnum = int(model.mesh_vertnum[g_mesh])
                    faceadr = int(model.mesh_faceadr[g_mesh])
                    facenum = int(model.mesh_facenum[g_mesh])
                    verts = model.mesh_vert[vertadr : vertadr + vertnum].astype(np.float32)
                    faces = model.mesh_face[faceadr : faceadr + facenum].astype(np.uint32)

                    rot = R.from_quat(model.geom_quat[g], scalar_first=True)
                    verts_local = rot.apply(verts) + model.geom_pos[g]

                    mesh_3d = _build_mesh3d(g, verts_local, faces, is_robot=False)
                    rr.log(
                        f"world/scene_objects/{b_key}/geom_{g}",
                        mesh_3d,
                        static=True,
                    )
                elif g_type == mujoco.mjtGeom.mjGEOM_BOX:
                    color = (model.geom_rgba[g] * 255).astype(np.uint8).tolist() if hasattr(model, "geom_rgba") else [200, 200, 200, 255]
                    rr.log(
                        f"world/scene_objects/{b_key}/geom_{g}",
                        rr.Boxes3D(
                            half_sizes=[model.geom_size[g]],
                            centers=[model.geom_pos[g]],
                            colors=[color],
                        ),
                        static=True,
                    )
                elif g_type == mujoco.mjtGeom.mjGEOM_PLANE:
                    sx = float(model.geom_size[g][0]) if model.geom_size[g][0] > 0 else 10.0
                    sy = float(model.geom_size[g][1]) if model.geom_size[g][1] > 0 else 10.0
                    color = (model.geom_rgba[g] * 255).astype(np.uint8).tolist() if hasattr(model, "geom_rgba") else [180, 180, 180, 255]
                    rr.log(
                        f"world/scene_objects/{b_key}/geom_{g}",
                        rr.Boxes3D(
                            half_sizes=[[sx, sy, 0.002]],
                            centers=[model.geom_pos[g]],
                            colors=[color],
                        ),
                        static=True,
                    )
                elif g_type in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                    s = float(model.geom_size[g][0])
                    color = (model.geom_rgba[g] * 255).astype(np.uint8).tolist() if hasattr(model, "geom_rgba") else [200, 200, 200, 255]
                    rr.log(
                        f"world/scene_objects/{b_key}/geom_{g}",
                        rr.Ellipsoids3D(
                            half_sizes=[[s, s, s]],
                            centers=[model.geom_pos[g]],
                            colors=[color],
                        ),
                        static=True,
                    )
                elif g_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    r = float(model.geom_size[g][0])
                    h = float(model.geom_size[g][1]) * 2
                    color = (model.geom_rgba[g] * 255).astype(np.uint8).tolist() if hasattr(model, "geom_rgba") else [200, 200, 200, 255]
                    rr.log(
                        f"world/scene_objects/{b_key}/geom_{g}",
                        rr.Cylinders3D(
                            radii=[r],
                            lengths=[h],
                            centers=[model.geom_pos[g]],
                            colors=[color],
                        ),
                        static=True,
                    )
                elif g_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    r = float(model.geom_size[g][0])
                    h = float(model.geom_size[g][1]) * 2
                    color = (model.geom_rgba[g] * 255).astype(np.uint8).tolist() if hasattr(model, "geom_rgba") else [200, 200, 200, 255]
                    rr.log(
                        f"world/scene_objects/{b_key}/geom_{g}",
                        rr.Capsules3D(
                            radii=[r],
                            lengths=[h],
                            centers=[model.geom_pos[g]],
                            colors=[color],
                        ),
                        static=True,
                    )
                self._logged_meshes.add(scene_geom_key)

    def log_step(
        self,
        step_idx: int,
        task: Any,
        observation: Any = None,
        policy: Any = None,
    ) -> None:
        if not self._initialized:
            return
        try:
            import rerun as rr

            model = task.env.current_model
            data = task.env.mj_datas[task.env.current_batch_index]
            pickup_obj_name = self._get_pickup_object_name(task)
            obj_body_ids = self._get_object_body_ids(model, pickup_obj_name)

            rr.set_time("step", sequence=int(step_idx))
            if hasattr(data, "time") and data.time is not None:
                rr.set_time("sim_time", duration=float(data.time))

            # Update body transforms for Stretch robot, manipulated object, and scene objects
            for b_id in range(model.nbody):
                b_name = model.body(b_id).name
                b_key = b_name.replace("/", "_") if b_name else f"body_{b_id}"
                pos = data.xpos[b_id]
                mat = data.xmat[b_id].reshape(3, 3)
                if "robot_0" in b_name:
                    rr.log(f"world/robot/{b_key}", rr.Transform3D(translation=pos, mat3x3=mat))
                elif b_id in obj_body_ids:
                    rr.log(f"world/object/{b_key}", rr.Transform3D(translation=pos, mat3x3=mat))
                else:
                    rr.log(f"world/scene_objects/{b_key}", rr.Transform3D(translation=pos, mat3x3=mat))

            # Coordinate frame axes config
            axis_len = 0.08
            axes_vectors = [[axis_len, 0.0, 0.0], [0.0, axis_len, 0.0], [0.0, 0.0, axis_len]]
            axes_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
            axes_labels = ["X", "Y", "Z"]

            # 1. Wrist Center frame
            wrist_bodies = [
                "robot_0/wrist_roll_link",
                "robot_0/wrist_pitch_link",
                "robot_0/wrist_yaw_link",
                "robot_0/wrist_link",
                "robot_0/tool_attachment_site_link",
            ]
            for wb in wrist_bodies:
                try:
                    w_id = model.body(wb).id
                    pos = data.xpos[w_id]
                    mat = data.xmat[w_id].reshape(3, 3)
                    rr.log("world/frames/wrist_center", rr.Transform3D(translation=pos, mat3x3=mat))
                    rr.log("world/frames/wrist_center/axes", rr.Arrows3D(vectors=axes_vectors, colors=axes_colors, labels=axes_labels, show_labels=True))
                    rr.log("world/frames/wrist_center/label", rr.Points3D(positions=[[0.0, 0.0, 0.0]], radii=[0.005], colors=[[255, 255, 255]], labels=["Wrist Center"], show_labels=True))
                    break
                except Exception:
                    pass

            # 2. Tool Center frame
            tool_bodies = [
                "robot_0/grasp_center_link",
                "robot_0/quick_connect_interface_link",
                "robot_0/tool_attachment_site_link",
            ]
            for tb in tool_bodies:
                try:
                    t_id = model.body(tb).id
                    pos = data.xpos[t_id]
                    mat = data.xmat[t_id].reshape(3, 3)
                    rr.log("world/frames/tool_center", rr.Transform3D(translation=pos, mat3x3=mat))
                    rr.log("world/frames/tool_center/axes", rr.Arrows3D(vectors=axes_vectors, colors=axes_colors, labels=axes_labels, show_labels=True))
                    rr.log("world/frames/tool_center/label", rr.Points3D(positions=[[0.0, 0.0, 0.0]], radii=[0.005], colors=[[255, 255, 255]], labels=["Tool Center"], show_labels=True))
                    break
                except Exception:
                    pass

            # 3. Object frame
            if obj_body_ids:
                # Use root-most body in obj_body_ids
                root_obj_id = min(obj_body_ids)
                pos = data.xpos[root_obj_id]
                mat = data.xmat[root_obj_id].reshape(3, 3)
                rr.log("world/frames/object", rr.Transform3D(translation=pos, mat3x3=mat))
                rr.log("world/frames/object/axes", rr.Arrows3D(vectors=axes_vectors, colors=axes_colors, labels=axes_labels, show_labels=True))
                rr.log("world/frames/object/label", rr.Points3D(positions=[[0.0, 0.0, 0.0]], radii=[0.005], colors=[[255, 255, 255]], labels=[f"Object: {pickup_obj_name}" if pickup_obj_name else "Object"], show_labels=True))

            # 4. Target Grasp frame
            unwrapped_policy = self._unwrap_policy(policy)
            if unwrapped_policy is not None and hasattr(unwrapped_policy, "_grasp"):
                grasp = unwrapped_policy._grasp
                if grasp is not None:
                    pos = grasp.position
                    mat = grasp.rotation
                    grasp_type = "Authored" if getattr(grasp, "authored", False) else "Styled"
                    rr.log("world/frames/target_grasp", rr.Transform3D(translation=pos, mat3x3=mat))
                    rr.log(
                        "world/frames/target_grasp/axes",
                        rr.Arrows3D(vectors=axes_vectors, colors=axes_colors, labels=axes_labels, show_labels=True),
                    )
                    rr.log(
                        "world/frames/target_grasp/label",
                        rr.Points3D(
                            positions=[[0.0, 0.0, 0.0]],
                            radii=[0.005],
                            colors=[[255, 215, 0]],
                            labels=[f"Target Grasp ({grasp_type})"],
                            show_labels=True,
                        ),
                    )

            # 5. Waypoint Text Log and Info Document
            if unwrapped_policy is not None and hasattr(unwrapped_policy, "_plan"):
                plan = unwrapped_policy._plan
                w_idx = getattr(unwrapped_policy, "_waypoint_index", 0)
                steps_in_w = getattr(unwrapped_policy, "_steps_in_waypoint", 0)
                grasp_lost = getattr(unwrapped_policy, "_grasp_lost", False)

                if plan:
                    if 0 <= w_idx < len(plan):
                        curr_wp = plan[w_idx]
                        w_label = curr_wp.label
                        w_pos = curr_wp.position
                        w_pitch = np.degrees(curr_wp.wrist_pitch)
                        w_roll = np.degrees(curr_wp.wrist_roll)
                        w_yaw = (
                            f"{np.degrees(curr_wp.approach_yaw):+.1f}°"
                            if curr_wp.approach_yaw is not None
                            else "Free"
                        )
                        w_grip = "Open" if curr_wp.gripper_open else "Closed"
                        w_width = (
                            f"{curr_wp.grip_width_m:.3f} m"
                            if curr_wp.grip_width_m is not None
                            else "N/A"
                        )

                        # Emit chronological text log on waypoint transitions
                        if w_idx != self._last_logged_waypoint_idx:
                            self._last_logged_waypoint_idx = w_idx
                            log_msg = (
                                f"📍 [Step {step_idx}] Waypoint {w_idx + 1}/{len(plan)}: '{w_label}' | "
                                f"Target: [{w_pos[0]:.3f}, {w_pos[1]:.3f}, {w_pos[2]:.3f}] | "
                                f"Pitch: {w_pitch:+.1f}° | Roll: {w_roll:+.1f}° | Gripper: {w_grip}"
                            )
                            rr.log("logs/waypoints", rr.TextLog(log_msg, level=rr.TextLogLevel.INFO))

                        # Build plan progress list
                        checklist_lines = []
                        for i, wp in enumerate(plan):
                            if i < w_idx:
                                checklist_lines.append(f"- [x] `{wp.label}`")
                            elif i == w_idx:
                                checklist_lines.append(f"- [x] **`{wp.label}`** ◀ *(Active)*")
                            else:
                                checklist_lines.append(f"- [ ] `{wp.label}`")
                        checklist_str = "\n".join(checklist_lines)

                        doc_md = f"""### 🎯 Active Waypoint: `{w_label}` ({w_idx + 1}/{len(plan)})

| Property | Value |
|:---|:---|
| **Label** | `{w_label}` |
| **Index** | {w_idx + 1} of {len(plan)} |
| **Steps in Waypoint** | {steps_in_w} |
| **Target Pos (xyz)** | `[{w_pos[0]:.3f}, {w_pos[1]:.3f}, {w_pos[2]:.3f}]` |
| **Wrist Pitch** | `{curr_wp.wrist_pitch:+.2f} rad ({w_pitch:+.1f}°)` |
| **Wrist Roll** | `{curr_wp.wrist_roll:+.2f} rad ({w_roll:+.1f}°)` |
| **Approach Yaw** | `{w_yaw}` |
| **Gripper** | {w_grip} |
| **Grip Width** | {w_width} |
| **Tolerance** | `{curr_wp.tolerance:.3f} m` |
| **Establishes Grasp** | `{curr_wp.establishes_grasp}` |
| **Verify Grasp** | `{curr_wp.verify_grasp}` |
| **Settle Steps** | {curr_wp.settle_steps} |
| **Grasp Status** | `{'Grasp Lost!' if grasp_lost else ('Held' if (getattr(unwrapped_policy, "_grasp_offset", None) is not None) else 'In Progress')}` |

#### 📋 Plan Progress
{checklist_str}
"""
                    else:
                        if w_idx != self._last_logged_waypoint_idx:
                            self._last_logged_waypoint_idx = w_idx
                            rr.log(
                                "logs/waypoints",
                                rr.TextLog(f"🏁 [Step {step_idx}] All {len(plan)} waypoints completed.", level=rr.TextLogLevel.INFO),
                            )
                        doc_md = f"""### 🏁 Plan Complete ({len(plan)}/{len(plan)} waypoints)

All waypoints finished execution. Holding final posture/grip.
"""
                    rr.log("planner/waypoint", rr.TextDocument(doc_md, media_type=rr.MediaType.MARKDOWN))

                    if grasp_lost and not self._logged_grasp_lost:
                        self._logged_grasp_lost = True
                        rr.log("logs/waypoints", rr.TextLog(f"⚠️ [Step {step_idx}] Grasp lost during lift!", level=rr.TextLogLevel.WARN))

            # 6. Optional Camera feeds
            if observation is not None:
                obs_dict = observation[0] if isinstance(observation, list) and observation else observation
                if isinstance(obs_dict, dict):
                    for cam_name in ["head_camera", "wrist_camera_left", "wrist_camera_right", "head_camera_left", "head_camera_right"]:
                        if cam_name in obs_dict and obs_dict[cam_name] is not None:
                            img = obs_dict[cam_name]
                            if hasattr(img, "ndim") and img.ndim == 3:
                                rr.log(f"world/cameras/{cam_name}", rr.Image(img))
                    for depth_key in [k for k in obs_dict.keys() if k.endswith("_depth")]:
                        depth_img = obs_dict[depth_key]
                        if depth_img is not None and hasattr(depth_img, "ndim") and depth_img.ndim in (2, 3):
                            if depth_img.ndim == 3 and depth_img.shape[-1] == 1:
                                depth_img = depth_img.squeeze(-1)
                            rr.log(f"world/cameras/{depth_key}", rr.DepthImage(depth_img))
        except Exception as e:
            log.debug(f"Error logging to Rerun: {e}")


def snap_free_camera_to_robot(viewer: Any, task: Any) -> None:
    """Configures MuJoCo passive viewer in free camera mode snapped to the robot."""
    if viewer is None:
        return
    try:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.fixedcamid = -1

        robot_pos = None
        if hasattr(task, "env") and hasattr(task.env, "current_model") and hasattr(task.env, "mj_datas"):
            m = task.env.current_model
            d = task.env.mj_datas[task.env.current_batch_index]
            for candidate in ["robot_0/base_link", "robot_0/base", "base_link", "robot_0/lift_link"]:
                try:
                    bid = m.body(candidate).id
                    robot_pos = d.xpos[bid]
                    break
                except Exception:
                    pass

        if robot_pos is None and hasattr(task, "robot") and task.robot:
            base_pose = getattr(task.robot, "base_pose", None)
            if base_pose is not None:
                robot_pos = base_pose[:3]

        if robot_pos is not None:
            viewer.cam.lookat[0] = float(robot_pos[0])
            viewer.cam.lookat[1] = float(robot_pos[1])
            viewer.cam.lookat[2] = float(robot_pos[2]) + 0.6
            viewer.cam.distance = 2.5
            viewer.cam.elevation = -20.0
            viewer.cam.azimuth = 135.0
    except Exception as e:
        log.warning(f"Failed to snap free camera to robot: {e}")


def stretch_house_processing_worker(
    worker_id: int,
    exp_config: Any,
    work_items: list[tuple[int, int, int, int]],
    shutdown_event: Any,
    counter_lock: Any,
    house_counter: Any,
    success_count: Any,
    total_count: Any,
    completed_houses: Any,
    skipped_houses: Any,
    max_allowed_sequential_task_sampler_failures: int = 10,
    max_allowed_sequential_rollout_failures: int = 10,
    max_allowed_sequential_irrecoverable_failures: int = 5,
    preloaded_policy: Any = None,
    filter_for_successful_trajectories: bool = False,
    runner_class: Any = None,
    max_items_per_worker: int | None = 10,
) -> None:
    """Worker function that processes a limited number of work items before exiting to recycle memory."""
    worker_logger = get_worker_logger(worker_id)

    if hasattr(exp_config, "datagen_profiler") and exp_config.datagen_profiler:
        datagen_profiler = DatagenProfiler(logger=worker_logger, enabled=True)
    else:
        datagen_profiler = None

    num_sequential_irrecoverable_failures = 0
    task_sampler = exp_config.task_sampler_config.task_sampler_class(exp_config)
    task_sampler.set_datagen_profiler(datagen_profiler)

    items_processed_by_worker = 0
    with worker_stdout_context(worker_logger, worker_id):
        try:
            while True:
                if shutdown_event.is_set():
                    worker_logger.info(
                        f"Worker {worker_id} received shutdown signal, cleaning up..."
                    )
                    break

                with counter_lock:
                    if house_counter.value >= len(work_items):
                        break
                    item_idx = house_counter.value
                    house_counter.value += 1

                current_house_id, batch_samples, batch_num, total_batches = work_items[item_idx]

                worker_logger.info(
                    f"Worker {worker_id} (PID {os.getpid()}) starting house {current_house_id} "
                    f"batch {batch_num}/{total_batches} ({batch_samples} episodes) "
                    f"(item {item_idx + 1}/{len(work_items)})"
                )

                house_success_count, house_total_count, irrecoverable = (
                    runner_class.process_single_house(
                        worker_id,
                        worker_logger,
                        current_house_id,
                        exp_config,
                        batch_samples,
                        shutdown_event,
                        task_sampler,
                        preloaded_policy,
                        max_allowed_sequential_task_sampler_failures,
                        max_allowed_sequential_rollout_failures,
                        filter_for_successful_trajectories=filter_for_successful_trajectories,
                        runner_class=runner_class,
                        batch_num=batch_num,
                        total_batches=total_batches,
                        datagen_profiler=datagen_profiler,
                    )
                )

                with counter_lock:
                    success_count.value += house_success_count
                    total_count.value += house_total_count
                    if house_total_count > 0:
                        completed_houses.value += 1
                    else:
                        skipped_houses.value += 1

                items_processed_by_worker += 1
                trim_memory()
                # Logged after the trim, so a footprint that keeps climbing across
                # work items is a real leak worth chasing, while one that returns
                # to a flat baseline is just the per-episode peak.
                log_memory_usage(
                    worker_logger,
                    prefix=f"Worker {worker_id} after {items_processed_by_worker} "
                    f"work items (house {current_house_id}): ",
                )

                if irrecoverable:
                    num_sequential_irrecoverable_failures += 1
                    if (
                        num_sequential_irrecoverable_failures
                        >= max_allowed_sequential_irrecoverable_failures
                    ):
                        worker_logger.error(
                            f"Worker {worker_id} encountered {num_sequential_irrecoverable_failures} "
                            "sequential irrecoverable failures. Exiting worker."
                        )
                        break
                else:
                    num_sequential_irrecoverable_failures = 0

                # Process recycling check: exit cleanly so kernel frees leaked C/driver memory
                if max_items_per_worker is not None and max_items_per_worker > 0:
                    if items_processed_by_worker >= max_items_per_worker:
                        worker_logger.info(
                            f"Worker {worker_id} (PID {os.getpid()}) completed {items_processed_by_worker} "
                            f"work items (limit: {max_items_per_worker}). Recycling process to free OS/GPU memory."
                        )
                        break

            worker_logger.info(f"Worker {worker_id} finished processing assigned work items")
        finally:
            if datagen_profiler is not None:
                datagen_profiler.log_worker_summary()
            if task_sampler is not None:
                task_sampler.close()
            trim_memory()


class StretchRolloutRunner(ParallelRolloutRunner):
    """ParallelRolloutRunner with free camera snapping, simulation slowdown, Rerun 3D viz, and process recycling."""

    slow_rate: float | None = None
    visualize: bool = False
    rerun_visualizer: StretchRerunVisualizer | None = None
    max_items_per_worker: int = 10

    @staticmethod
    def run_single_rollout(
        episode_seed: int,
        task: Any,
        policy: Any,
        profiler: Any = None,
        viewer: Any = None,
        shutdown_event: Any = None,
        datagen_profiler: Any = None,
        end_on_success: bool = False,
    ) -> bool:
        slow_rate = StretchRolloutRunner.slow_rate
        if slow_rate is None:
            env_val = os.environ.get("STRETCH_DATAGEN_SLOW_RATE")
            if env_val:
                try:
                    slow_rate = float(env_val)
                except ValueError:
                    slow_rate = None

        visualize = StretchRolloutRunner.visualize or viewer is not None
        if not visualize and os.environ.get("STRETCH_DATAGEN_VISUALIZE") == "1":
            visualize = True

        rerun_viz = None
        if visualize:
            if StretchRolloutRunner.rerun_visualizer is None:
                StretchRolloutRunner.rerun_visualizer = StretchRerunVisualizer(spawn=True)
            rerun_viz = StretchRolloutRunner.rerun_visualizer
            rerun_viz.start_episode(episode_seed, task, policy=policy)

        if profiler is not None:
            profiler.start("rollout")
        if datagen_profiler is not None:
            datagen_profiler.start("rollout_total")
            datagen_profiler.start("rollout_reset")

        observation, _info = task.reset()

        if datagen_profiler is not None:
            datagen_profiler.end("rollout_reset")

        if viewer is not None:
            snap_free_camera_to_robot(viewer, task)
            viewer.sync()

        if rerun_viz is not None:
            rerun_viz.log_step(0, task, observation, policy=policy)

        try:
            task.env.current_model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)
            task.env.current_model.opt.sleep_tolerance = 1e-3
        except AttributeError:
            log.debug("Not setting mujoco sleep. Needs version >=mujoco-3.8")

        step_count = 0
        while not task.is_done():
            # Check for shutdown signal
            if shutdown_event is not None and shutdown_event.is_set():
                if datagen_profiler is not None:
                    datagen_profiler.end("rollout_total")
                return False

            if viewer is not None and hasattr(viewer, "is_running") and not viewer.is_running():
                break

            t_step_wall_start = time.perf_counter()
            t_sim_start = (
                task.env.mj_datas[task.env.current_batch_index].time
                if hasattr(task, "env") and hasattr(task.env, "mj_datas") and task.env.mj_datas
                else None
            )

            # Step with policy
            if profiler is not None:
                profiler.start("policy_get_action")
            if datagen_profiler is not None:
                datagen_profiler.start("policy_get_action")
            # An action chunk is a list of actions to be applied open-loop before a
            # new observation is needed.
            action_chunk = policy.get_action_chunk(observation) or [policy.get_action(observation)]
            if profiler is not None:
                profiler.end("policy_get_action")
            if datagen_profiler is not None:
                datagen_profiler.end("policy_get_action")

            # Step the task
            if profiler is not None:
                profiler.start("task_step")
            if datagen_profiler is not None:
                datagen_profiler.start("task_step")
            if action_chunk[0] is None:
                log.info("Policy returned None action, ending episode")
                break
            observation, reward, terminal, truncated, infos = task.step_chunk(
                action_chunk, stop_on_success=end_on_success
            )
            step_count += len(action_chunk)
            if profiler is not None:
                profiler.end("task_step")
            if datagen_profiler is not None:
                datagen_profiler.end("task_step")

            if viewer is not None:
                viewer.sync()

            if rerun_viz is not None:
                rerun_viz.log_step(step_count, task, observation, policy=policy)

            # Add termination if succ
            if end_on_success and "success" in infos[0] and infos[0]["success"]:
                break

            if slow_rate is not None and slow_rate > 0:
                t_sim_end = (
                    task.env.mj_datas[task.env.current_batch_index].time
                    if hasattr(task, "env") and hasattr(task.env, "mj_datas") and task.env.mj_datas
                    else None
                )
                if t_sim_start is not None and t_sim_end is not None and t_sim_end > t_sim_start:
                    sim_dt = t_sim_end - t_sim_start
                else:
                    policy_dt_ms = getattr(getattr(task, "config", None), "policy_dt_ms", 66.0)
                    sim_dt = (policy_dt_ms / 1000.0) * len(action_chunk)

                target_wall_dt = sim_dt * slow_rate
                elapsed_wall = time.perf_counter() - t_step_wall_start
                sleep_time = target_wall_dt - elapsed_wall
                if sleep_time > 0:
                    time.sleep(sleep_time)

        try:
            task.env.current_model.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_SLEEP)
        except AttributeError:
            pass

        # Save profiler summary
        if profiler is not None:
            profiler.end("rollout")
        if datagen_profiler is not None:
            datagen_profiler.end("rollout_total")
            datagen_profiler.record("step_count_indicator", step_count / 1000.0)

        # Check success if method exists
        success = task.judge_success() if hasattr(task, "judge_success") else False
        return success

    @staticmethod
    def process_single_house(
        worker_id: int,
        worker_logger: Any,
        house_id: int,
        exp_config: Any,
        samples_per_house: int,
        shutdown_event: Any,
        task_sampler: Any,
        preloaded_policy: Any = None,
        max_allowed_sequential_task_sampler_failures: int = 10,
        max_allowed_sequential_rollout_failures: int = 10,
        filter_for_successful_trajectories: bool = False,
        runner_class: Any = None,
        batch_num: int | None = None,
        total_batches: int | None = None,
        datagen_profiler: Any = None,
    ) -> tuple[int, int, bool]:
        """Process all episodes for a single house with aggressive memory trimming and cache clearing."""
        house_success_count = 0
        house_total_count = 0
        irrecoverable_failure_in_house = False

        # Setup directories and check for existing output
        house_output_dir, house_debug_dir, batch_suffix, should_skip = setup_house_dirs(
            exp_config, house_id, batch_num, total_batches
        )
        if should_skip:
            worker_logger.info(
                f"SKIPPING HOUSE {house_id} BATCH {batch_num}/{total_batches}: "
                f"Output already exists at {house_output_dir / f'trajectories{batch_suffix}.h5'}"
            )
            return 0, 0, False

        episode_specs, shared_task_sampler = runner_class.load_episodes_for_house(
            exp_config, house_id, batch_suffix, task_sampler, worker_logger
        )

        if not episode_specs:
            worker_logger.warning(f"No episodes to process for house {house_id}")
            return 0, 0, False

        max_attempts = runner_class.get_max_episode_attempts(
            episode_specs, samples_per_house, exp_config
        )

        # Camera-stripped batched tensors, one per kept episode. The raw frames
        # they came from are released as each episode finishes, so this list
        # stays in the tens of megabytes rather than the tens of gigabytes.
        house_prepared_episodes: list[dict] = []
        house_debug_prepared_episodes: list[dict] = []

        num_sequential_task_sampler_failures = 0
        num_sequential_rollout_failures = 0
        viewer = None

        episode_idx = 0
        while episode_idx < max_attempts:
            should_stop = runner_class.should_stop_early(
                len(house_prepared_episodes), samples_per_house, exp_config=exp_config
            )
            if should_stop:
                break

            if shutdown_event.is_set():
                worker_logger.info(f"Worker {worker_id} house {house_id} received shutdown signal")
                irrecoverable_failure_in_house = True
                break

            if num_sequential_task_sampler_failures >= max_allowed_sequential_task_sampler_failures:
                worker_logger.error(
                    f"Worker {worker_id} house {house_id} encountered "
                    f"{num_sequential_task_sampler_failures} consecutive task sampling failures."
                )
                irrecoverable_failure_in_house = True
                break

            if num_sequential_rollout_failures >= max_allowed_sequential_rollout_failures:
                worker_logger.error(
                    f"Worker {worker_id} house {house_id} rollout failed across "
                    f"{num_sequential_rollout_failures} retries."
                )
                irrecoverable_failure_in_house = True
                break

            episode_spec = runner_class.get_episode_spec_at_index(episode_specs, episode_idx)

            task = None
            policy = None
            episode_task_sampler = None
            success = False
            task_sampling_failed = False
            house_invalid = False

            if datagen_profiler is not None:
                datagen_profiler.start("episode_total")

            episode_config = runner_class.prepare_episode_config(
                exp_config, episode_spec, episode_idx
            )

            with cleanup_context():
                if viewer is not None:
                    viewer.close()
                    viewer = None

                task_sampling_start = time.perf_counter()

                try:
                    episode_task_sampler = runner_class.get_episode_task_sampler(
                        episode_config, episode_spec, shared_task_sampler, datagen_profiler
                    )
                    task = runner_class.sample_task_from_spec(
                        episode_task_sampler, house_id, episode_spec, episode_idx
                    )

                    if task is None:
                        worker_logger.info(
                            f"Worker {worker_id} house {house_id} episode {episode_idx}: task sampling returned None"
                        )
                        house_invalid = True
                    else:
                        if datagen_profiler is not None:
                            datagen_profiler.record(
                                "task_sampling", time.perf_counter() - task_sampling_start
                            )
                            task.set_datagen_profiler(datagen_profiler)

                        num_sequential_task_sampler_failures = 0
                        worker_logger.info(
                            f"Worker {worker_id} house {house_id} episode {episode_idx}/{max_attempts} "
                            f"collected={len(house_prepared_episodes)}/{samples_per_house}"
                        )
                except HouseInvalidForTask as e:
                    traceback.print_exc()
                    worker_logger.warning(
                        f"Worker {worker_id} house {house_id} episode {episode_idx} HouseInvalidForTask: {e.reason}"
                    )
                    house_invalid = True
                    if datagen_profiler is not None:
                        datagen_profiler.record(
                            "task_sampling_failed", time.perf_counter() - task_sampling_start
                        )
                except Exception as e:
                    traceback.print_exc()
                    worker_logger.error(
                        f"Worker {worker_id} house {house_id} episode {episode_idx} task sampling error: {str(e)}"
                    )
                    num_sequential_task_sampler_failures += 1
                    task_sampling_failed = True
                    if datagen_profiler is not None:
                        datagen_profiler.record(
                            "task_sampling_failed", time.perf_counter() - task_sampling_start
                        )

                if task is not None and not house_invalid and not task_sampling_failed:
                    try:
                        policy = setup_policy(
                            episode_config, task, preloaded_policy, datagen_profiler
                        )
                        viewer = setup_viewer(episode_config, task, policy, viewer)

                        episode_seed = runner_class.get_episode_seed(
                            episode_idx, episode_spec, episode_task_sampler
                        )

                        success = runner_class.run_single_rollout(
                            episode_seed=episode_seed,
                            task=task,
                            policy=policy,
                            profiler=episode_config.profiler,
                            viewer=viewer,
                            shutdown_event=shutdown_event,
                            datagen_profiler=datagen_profiler,
                            end_on_success=exp_config.end_on_success,
                        )

                        num_sequential_rollout_failures = 0

                        object_name = "unknown"
                        if hasattr(task, "config") and hasattr(task.config, "task_config"):
                            if hasattr(task.config.task_config, "pickup_obj_name"):
                                object_name = task.config.task_config.pickup_obj_name

                        worker_logger.info(
                            f"Worker {worker_id} house {house_id} episode {episode_idx} "
                            f"object {object_name} completed with success={success}"
                        )

                        should_save = success or not filter_for_successful_trajectories
                        history = task.get_history()

                        should_save_debug = not should_save and random.random() < 0.01

                        # Encode and release this episode's frames now rather than
                        # at the end of the house. `history` aliases the task's own
                        # observation cache and `prepare_episode_for_saving` empties
                        # it in place, so the frames are gone before the next
                        # episode's scene is loaded.
                        if should_save or should_save_debug:
                            if should_save:
                                target_dir = house_output_dir
                                target_list = house_prepared_episodes
                            else:
                                target_dir = house_debug_dir
                                target_list = house_debug_prepared_episodes
                                worker_logger.info(
                                    f"Saving failed trajectory for debug (seed: {episode_seed})"
                                )

                            prepared = flush_episode_to_disk(
                                worker_logger,
                                history=history,
                                sensor_suite=task.sensor_suite,
                                save_dir=target_dir,
                                exp_config=exp_config,
                                batch_suffix=batch_suffix,
                                episode_idx=len(target_list),
                                datagen_profiler=(
                                    datagen_profiler if should_save else None
                                ),
                            )
                            if prepared is not None:
                                target_list.append(prepared)

                        del history
                        trim_memory()
                        log_memory_usage(
                            worker_logger,
                            prefix=f"Worker {worker_id} house {house_id} "
                            f"after episode {episode_idx}: ",
                        )

                        house_total_count += 1
                        if success:
                            house_success_count += 1
                        else:
                            asset_uid = task_sampler.get_asset_uid_from_object(
                                task.env, object_name
                            )
                            if asset_uid:
                                task_sampler.report_asset_failure(asset_uid, "rollout failed")

                        if datagen_profiler is not None:
                            datagen_profiler.end("episode_total")
                            datagen_profiler.log_episode_summary(
                                episode_idx=episode_idx,
                                house_id=house_id,
                                success=success,
                            )
                    except Exception as e:
                        worker_logger.error(
                            f"Worker {worker_id} house {house_id} episode {episode_idx} rollout error: {str(e)}"
                        )
                        traceback.print_exc()
                        num_sequential_rollout_failures += 1

                        try:
                            asset_uid = task_sampler.get_asset_uid_from_object(
                                task.env, object_name
                            )
                            if asset_uid:
                                task_sampler.report_asset_failure(
                                    asset_uid, f"rollout exception: {e}"
                                )
                        except Exception:
                            pass

                        if datagen_profiler is not None:
                            datagen_profiler.end("episode_total")
                else:
                    if datagen_profiler is not None:
                        datagen_profiler.end("episode_total")

                cleanup_episode_resources(
                    task=task,
                    policy=policy,
                    task_sampler=episode_task_sampler,
                    preloaded_policy=preloaded_policy,
                    close_task_sampler=runner_class.should_close_episode_task_sampler(),
                )

            if house_invalid:
                irrecoverable_failure_in_house = True
                break

            episode_idx += 1

        if viewer is not None:
            viewer.close()
            viewer = None

        if shutdown_event.is_set():
            worker_logger.info(
                f"Worker {worker_id} house {house_id} shutdown requested, skipping save"
            )
            # The HDF5 is what `setup_house_dirs` resumes off, so not writing it
            # means this house batch gets redone. Videos are now written as each
            # episode finishes rather than alongside the HDF5, so they have to be
            # cleared too -- otherwise the re-run leaves stale MP4s behind
            # whenever it keeps fewer episodes than this attempt did.
            for stale_dir in (house_output_dir, house_debug_dir):
                for stale_mp4 in Path(stale_dir).glob(f"episode_*{batch_suffix}.mp4"):
                    try:
                        stale_mp4.unlink()
                    except OSError as e:
                        worker_logger.warning(f"Could not remove partial video {stale_mp4}: {e}")
            return house_success_count, house_total_count, True

        save_prepared_trajectories(
            worker_logger,
            house_prepared_episodes,
            house_output_dir,
            exp_config,
            batch_suffix,
            datagen_profiler,
            batch_num,
            total_batches,
        )

        save_prepared_trajectories(
            worker_logger,
            house_debug_prepared_episodes,
            house_debug_dir,
            exp_config,
            batch_suffix,
            datagen_profiler=None,
            batch_num=batch_num,
            total_batches=total_batches,
        )

        # Drop the prepared tensors before trimming; the previous version trimmed
        # while they were still in scope, so it could not reclaim them.
        house_prepared_episodes.clear()
        house_debug_prepared_episodes.clear()
        trim_memory()

        worker_logger.info(
            f"Worker {worker_id} completed house {house_id}: "
            f"{house_success_count}/{house_total_count} successful episodes"
        )

        if datagen_profiler is not None:
            datagen_profiler.log_house_summary(
                house_id=house_id,
                success_count=house_success_count,
                total_count=house_total_count,
            )

        return house_success_count, house_total_count, irrecoverable_failure_in_house

    def run(self, preloaded_policy: Any = None) -> tuple[int, int]:
        """Run rollouts using a pool of recycled worker processes to prevent memory leaks."""
        total_expected_episodes = sum(wi[1] for wi in self.work_items)
        self.logger.info(
            f"Starting rollout of {self.total_houses} houses "
            f"split into {len(self.work_items)} work items ({total_expected_episodes} total episodes) "
            f"using {self.config.num_workers} worker processes (recycling every {self.max_items_per_worker} items)"
        )

        self.logger.info("Evaluation configuration:")
        self.logger.info(pprint.pformat(self.config.model_dump()))
        self.config.save_config(output_dir=Path(self.config.output_dir))

        start_time = time.time()

        if self.config.num_workers > 1 or (not self.visualize and self.max_items_per_worker):
            target_workers = self.config.num_workers
            active_processes: dict[int, Any] = {}
            next_worker_id = 0

            def spawn_worker(wid: int) -> Any:
                p = mp_context.Process(
                    target=stretch_house_processing_worker,
                    args=(
                        wid,
                        self.config,
                        self.work_items,
                        self.shutdown_event,
                        self.counter_lock,
                        self.house_counter,
                        self.success_count,
                        self.total_count,
                        self.completed_houses,
                        self.skipped_houses,
                        self.max_allowed_sequential_task_sampler_failures,
                        self.max_allowed_sequential_rollout_failures,
                        self.max_allowed_sequential_irrecoverable_failures,
                        preloaded_policy,
                        self.config.filter_for_successful_trajectories,
                        type(self),
                        self.max_items_per_worker,
                    ),
                )
                p.start()
                return p

            initial_count = min(target_workers, len(self.work_items))
            for _ in range(initial_count):
                active_processes[next_worker_id] = spawn_worker(next_worker_id)
                next_worker_id += 1

            last_log_time = start_time
            log_interval = 60

            while active_processes:
                dead_ids = []
                for wid, p in list(active_processes.items()):
                    if not p.is_alive():
                        p.join()
                        p.close()
                        dead_ids.append(wid)

                for wid in dead_ids:
                    del active_processes[wid]
                    if not self.shutdown_event.is_set():
                        with self.counter_lock:
                            has_more_work = self.house_counter.value < len(self.work_items)
                        if has_more_work and len(active_processes) < target_workers:
                            active_processes[next_worker_id] = spawn_worker(next_worker_id)
                            next_worker_id += 1

                current_time = time.time()
                if self.wandb_enabled and (current_time - last_log_time) >= log_interval:
                    try:
                        elapsed_time = current_time - start_time
                        completed = self.completed_houses.value
                        skipped = self.skipped_houses.value
                        success = self.success_count.value
                        total = self.total_count.value
                        active = sum(1 for p in active_processes.values() if p.is_alive())
                        total_work_items = len(self.work_items)
                        success_rate = success / total if total > 0 else 0.0
                        episodes_per_second = total / elapsed_time if elapsed_time > 0 else 0.0
                        completion_percentage = (completed + skipped) / total_work_items * 100

                        import wandb

                        wandb.log(
                            {
                                "elapsed_time_seconds": elapsed_time,
                                "elapsed_time_hours": elapsed_time / 3600,
                                "completed_houses": completed,
                                "skipped_houses": skipped,
                                "success_count": success,
                                "total_count": total,
                                "success_rate": success_rate,
                                "episodes_per_second": episodes_per_second,
                                "active_workers": active,
                                "completion_percentage": completion_percentage,
                            }
                        )
                        self.logger.info(
                            f"Progress: {completed}/{total_work_items} work items completed "
                            f"({completion_percentage:.1f}%), {success}/{total} successful episodes "
                            f"({success_rate * 100:.1f}%), {active} workers active"
                        )
                        last_log_time = current_time
                    except Exception as e:
                        self.logger.warning(f"WandB periodic logging failed: {e}")

                time.sleep(1)

        else:
            # Single-worker in-process mode (used for --visualize interactive viewer)
            stretch_house_processing_worker(
                worker_id=0,
                exp_config=self.config,
                work_items=self.work_items,
                shutdown_event=self.shutdown_event,
                counter_lock=self.counter_lock,
                house_counter=self.house_counter,
                success_count=self.success_count,
                total_count=self.total_count,
                completed_houses=self.completed_houses,
                skipped_houses=self.skipped_houses,
                max_allowed_sequential_task_sampler_failures=self.max_allowed_sequential_task_sampler_failures,
                max_allowed_sequential_rollout_failures=self.max_allowed_sequential_rollout_failures,
                max_allowed_sequential_irrecoverable_failures=self.max_allowed_sequential_irrecoverable_failures,
                preloaded_policy=preloaded_policy,
                filter_for_successful_trajectories=self.config.filter_for_successful_trajectories,
                runner_class=type(self),
                max_items_per_worker=None,
            )

        success_count_val = self.success_count.value
        total_count_val = self.total_count.value
        completed_houses_val = self.completed_houses.value
        skipped_houses_val = self.skipped_houses.value
        success_rate = success_count_val / total_count_val if total_count_val > 0 else 0.0

        self.logger.info(
            f"Completed {completed_houses_val} work items, skipped {skipped_houses_val} work items"
        )
        self.logger.info(f"Success count: {success_count_val}, Total count: {total_count_val}")
        self.logger.info(f"Success rate: {success_rate * 100:.2f}%")

        if self.wandb_enabled:
            try:
                import wandb

                final_elapsed_time = time.time() - start_time
                wandb.log(
                    {
                        "final_success_count": success_count_val,
                        "final_total_count": total_count_val,
                        "final_success_rate": success_rate,
                        "final_completed_houses": completed_houses_val,
                        "final_skipped_houses": skipped_houses_val,
                        "final_elapsed_time_seconds": final_elapsed_time,
                        "final_elapsed_time_hours": final_elapsed_time / 3600,
                    }
                )
                wandb.finish()
            except Exception as e:
                self.logger.warning(f"WandB final logging failed: {e}")

        return success_count_val, total_count_val


def generate_rollouts(
    task: str,
    output_dir: Path,
    episodes: int | None = None,
    num_workers: int = 1,
    scene_dataset: str | None = None,
    data_split: str | None = None,
    houses: int | None = None,
    seed: int | None = None,
    keep_failures: bool = False,
    visualize: bool = False,
    slow_rate: float | None = None,
    max_items_per_worker: int = 10,
) -> Path:
    """Run the data generation pipeline for one task family.

    Args:
        task: a key of `DATAGEN_CONFIGS`.
        output_dir: where the rollouts go. The pipeline appends its own
            `<ConfigName>/<timestamp>/` beneath this.
        episodes: total episodes to attempt. Spread over `houses` houses.
        num_workers: parallel rollout worker processes.
        scene_dataset: override the config's scene dataset, e.g. `procthor-10k`.
        data_split: `train`, `val` or `test`. Left at the config's default
            (`train`) unless given -- generating fine-tuning data out of `val` is
            how a benchmark score stops meaning anything.
        houses: how many houses to draw from. Defaults to enough that each house
            contributes a handful of episodes rather than hundreds, which is
            what keeps the scene distribution wide.
        seed: task-sampling seed.
        keep_failures: keep failed trajectories in the rollout dataset.
        visualize: watch the rollouts in MuJoCo's passive viewer. Requires
            `num_workers == 1` -- see `main()`.
        slow_rate: slow down simulation by a time factor (e.g. 1.0 for real-time,
            2.0 for 2x slower than real-time).
        max_items_per_worker: number of work items (houses) a worker process handles
            before being recycled to release system and GPU driver memory.

    Returns:
        The directory the pipeline actually wrote to.
    """
    module_name, class_name = qualified_config_name(task).split(":")
    importlib.import_module(module_name)
    config = get_config_class(class_name)()

    if scene_dataset is not None:
        config.scene_dataset = scene_dataset
    if data_split is not None:
        config.data_split = data_split
    if seed is not None:
        config.seed = seed
    config.num_workers = num_workers
    config.use_wandb = False
    config.use_passive_viewer = visualize
    config.filter_for_successful_trajectories = not keep_failures

    if episodes is not None:
        _spread_episodes(config, episodes, houses)

    config.output_dir = Path(output_dir) / task
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.save_config()

    StretchRolloutRunner.visualize = visualize
    StretchRolloutRunner.max_items_per_worker = max_items_per_worker
    if visualize:
        os.environ["STRETCH_DATAGEN_VISUALIZE"] = "1"
    elif "STRETCH_DATAGEN_VISUALIZE" in os.environ:
        del os.environ["STRETCH_DATAGEN_VISUALIZE"]

    if slow_rate is not None:
        StretchRolloutRunner.slow_rate = slow_rate
        os.environ["STRETCH_DATAGEN_SLOW_RATE"] = str(slow_rate)
    elif "STRETCH_DATAGEN_SLOW_RATE" in os.environ:
        del os.environ["STRETCH_DATAGEN_SLOW_RATE"]
        StretchRolloutRunner.slow_rate = None
    else:
        StretchRolloutRunner.slow_rate = None

    log.info(
        f"[datagen] {class_name} | {config.scene_dataset}/{config.data_split} | "
        f"{len(config.task_sampler_config.house_inds)} houses x "
        f"{config.task_sampler_config.samples_per_house} episodes | "
        f"{num_workers} workers (recycling every {max_items_per_worker} items) -> {config.output_dir}"
    )
    successes, total = StretchRolloutRunner(config).run()
    log.info(f"[datagen] {task}: {successes}/{total} episodes succeeded")
    return config.output_dir


def _spread_episodes(config, episodes: int, houses: int | None) -> None:
    """Turn a total episode count into houses x samples-per-house.

    The samplers count in those two numbers rather than in episodes, and how the
    total is split matters: all 2000 episodes in one house is 2000 rollouts of
    one room. Defaults to roughly 4 episodes per house, capped at the 10 the
    sampler will retry a single house for before it gives up on it.
    """
    sampler_config = config.task_sampler_config
    per_house = 4 if houses is None else max(1, episodes // max(houses, 1))
    per_house = min(per_house, 10)
    house_count = houses if houses is not None else max(1, -(-episodes // per_house))

    available = list(sampler_config.house_inds or [])
    if len(available) < house_count:
        # The datagen configs ship a short house list (the first few) for
        # debugging; a real run needs more of the dataset than that.
        available = list(range(house_count))
    sampler_config.house_inds = available[:house_count]
    sampler_config.samples_per_house = per_house
    sampler_config.max_tasks = episodes


@click.command()
@click.option(
    "--task",
    "tasks",
    multiple=True,
    type=click.Choice(sorted(DATAGEN_CONFIGS)),
    default=("pick",),
    help="Task family to generate. Repeatable; several are pooled into one dataset.",
)
@click.option(
    "--episodes",
    type=int,
    default=None,
    help="Episodes to attempt per task. Defaults to the config's own house list.",
)
@click.option("--num-workers", type=int, default=1, help="Parallel rollout workers.")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Root for `rollouts/` and `lerobot/`.",
)
@click.option(
    "--rollouts",
    "rollout_dirs",
    multiple=True,
    type=click.Path(path_type=Path, exists=True),
    help="Skip generation and export these existing rollout directories instead.",
)
@click.option(
    "--action-space",
    type=click.Choice(ACTION_SPACES),
    default="stretch",
    help="Action/state space to export in. Only 'stretch', the native 10-dim "
    "move-group vector. See lerobot_export.py.",
)
@click.option(
    "--scene-dataset",
    type=str,
    default=None,
    help="Override the scene dataset, e.g. procthor-10k for a fast local run.",
)
@click.option(
    "--data-split",
    type=click.Choice(["train", "val", "test"]),
    default=None,
    help="Scene split. Leave unset to use the config's default (train).",
)
@click.option("--houses", type=int, default=None, help="How many houses to draw episodes from.")
@click.option("--seed", type=int, default=None, help="Task-sampling seed.")
@click.option(
    "--keep-failures/--successful-only",
    default=False,
    help="Include episodes the task judged unsuccessful. Off by default: a partial "
    "expert's failures are counter-examples, not demonstrations.",
)
@click.option(
    "--visualize",
    is_flag=True,
    help="Watch each episode live in MuJoCo's passive viewer, from the robot's "
    "chase camera. Forces --num-workers 1.",
)
@click.option(
    "--slow-rate",
    "--slow_rate",
    "slow_rate",
    type=float,
    default=None,
    help="Slow down simulation by a time factor (e.g. 1.0 for real-time, 2.0 for 2x slower than real-time).",
)
@click.option(
    "--max-items-per-worker",
    type=int,
    default=10,
    help="Number of house work items a worker process handles before being recycled to release system/GPU memory.",
)
@click.option("--export/--no-export", "want_export", default=True, help="Run the export stage.")
@click.option(
    "--fps", type=float, default=15.0, help="Frame rate to record in the dataset metadata."
)
def main(
    tasks: tuple[str, ...],
    episodes: int | None,
    num_workers: int,
    output_dir: Path,
    rollout_dirs: tuple[Path, ...],
    action_space: str,
    scene_dataset: str | None,
    data_split: str | None,
    houses: int | None,
    seed: int | None,
    keep_failures: bool,
    visualize: bool,
    slow_rate: float | None,
    max_items_per_worker: int,
    want_export: bool,
    fps: float,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # MolmoSpaces renders off-screen through EGL; without this a headless run
    # fails inside the camera manager rather than at startup. The passive viewer
    # is unaffected -- it is the C++ `simulate` app, which brings its own GLFW
    # window regardless of what the offscreen camera renderer is using.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    if visualize and num_workers != 1:
        # `ParallelRolloutRunner.run()` only stays in the main process for a
        # single worker; above that the rollouts happen in spawned processes,
        # where a viewer window would be launched per worker if it opened at all.
        click.secho("--visualize forces --num-workers 1.", fg="yellow")
        num_workers = 1

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if rollout_dirs:
        directories = [Path(directory) for directory in rollout_dirs]
        click.secho(f"Exporting {len(directories)} existing rollout directories.", fg="cyan")
    else:
        directories = [
            generate_rollouts(
                task=task,
                output_dir=output_dir / "rollouts",
                episodes=episodes,
                num_workers=num_workers,
                scene_dataset=scene_dataset,
                data_split=data_split,
                houses=houses,
                seed=seed,
                keep_failures=keep_failures,
                visualize=visualize,
                slow_rate=slow_rate,
                max_items_per_worker=max_items_per_worker,
            )
            for task in tasks
        ]

    if not want_export:
        click.secho(f"Rollouts under {output_dir / 'rollouts'}", fg="green")
        return

    dataset_dir = output_dir / "lerobot"
    metadata = export_lerobot_dataset(
        rollout_dirs=directories,
        output_dir=dataset_dir,
        action_space=action_space,
        successful_only=not keep_failures,
        fps=fps,
    )
    click.echo("")
    click.secho(
        f"{metadata.num_episodes} episodes / {metadata.num_frames} frames "
        f"in {action_space} action space -> {dataset_dir}",
        fg="green",
    )
    click.echo(
        "Fine-tune with:\n"
        f"  python -m examples.machine_learning.molmospaces.finetuning.finetune "
        f"--dataset {dataset_dir} --dry-run"
    )


if __name__ == "__main__":
    main()

