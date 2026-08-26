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

import importlib
import logging
import os
import time
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
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner

log = logging.getLogger(__name__)


class StretchRerunVisualizer:
    """Streams 3D robot meshes, object meshes, and coordinate frames to Rerun."""

    def __init__(self, spawn: bool = True):
        self._spawn = spawn
        self._initialized = False
        self._logged_meshes: set[str] = set()

    def start_episode(self, episode_seed: int, task: Any) -> None:
        """Starts a new Rerun recording for each episode."""
        try:
            import uuid
            import rerun as rr
            import rerun.blueprint as rrb

            rec_id = f"episode_{episode_seed}_{uuid.uuid4().hex[:8]}"
            app_id = "Stretch4 Datagen"

            blueprint = rrb.Blueprint(
                rrb.Spatial3DView(origin="world", name="3D Scene"),
                collapse_panels=True,
            )

            if not self._initialized:
                rr.init(app_id, recording_id=rec_id, spawn=self._spawn, default_blueprint=blueprint)
                if self._spawn:
                    try:
                        rr.spawn(memory_limit="4GB")
                    except Exception:
                        pass
                self._initialized = True
            else:
                rr.init(app_id, recording_id=rec_id, spawn=False, default_blueprint=blueprint)

            try:
                rr.send_blueprint(blueprint)
            except Exception as e:
                log.debug(f"Could not send Rerun blueprint: {e}")

            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
            self._logged_meshes.clear()

            pickup_obj_name = self._get_pickup_object_name(task)
            if hasattr(task, "env") and hasattr(task.env, "current_model"):
                self.setup_meshes(task.env.current_model, pickup_obj_name)
        except Exception as e:
            log.warning(f"Failed to start Rerun episode recording: {e}")

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

    def log_step(
        self,
        step_idx: int,
        task: Any,
        observation: Any = None,
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

            # Update body transforms for Stretch robot and manipulated object
            for b_id in range(model.nbody):
                b_name = model.body(b_id).name
                b_key = b_name.replace("/", "_")
                if "robot_0" in b_name:
                    pos = data.xpos[b_id]
                    mat = data.xmat[b_id].reshape(3, 3)
                    rr.log(f"world/robot/{b_key}", rr.Transform3D(translation=pos, mat3x3=mat))
                elif b_id in obj_body_ids:
                    pos = data.xpos[b_id]
                    mat = data.xmat[b_id].reshape(3, 3)
                    rr.log(f"world/object/{b_key}", rr.Transform3D(translation=pos, mat3x3=mat))

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

            # 4. Optional Camera feeds
            if observation is not None:
                obs_dict = observation[0] if isinstance(observation, list) and observation else observation
                if isinstance(obs_dict, dict):
                    for cam_name in ["head_camera", "wrist_camera", "chase_camera"]:
                        if cam_name in obs_dict and obs_dict[cam_name] is not None:
                            img = obs_dict[cam_name]
                            if hasattr(img, "ndim") and img.ndim == 3:
                                rr.log(f"world/cameras/{cam_name}", rr.Image(img))
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


class StretchRolloutRunner(ParallelRolloutRunner):
    """ParallelRolloutRunner with free camera snapping, simulation slowdown, and Rerun 3D viz."""

    slow_rate: float | None = None
    visualize: bool = False
    rerun_visualizer: StretchRerunVisualizer | None = None

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
            rerun_viz.start_episode(episode_seed, task)

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
            rerun_viz.log_step(0, task, observation)

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
                rerun_viz.log_step(step_count, task, observation)

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


def generate_rollouts(
    task: str,
    output_dir: Path,
    episodes: int | None = None,
    num_workers: int = 1,
    scene_dataset: str | None = None,
    data_split: str | None = None,
    houses: int | None = None,
    seed: int | None = None,
    visualize: bool = False,
    slow_rate: float | None = None,
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
        visualize: watch the rollouts in MuJoCo's passive viewer. Requires
            `num_workers == 1` -- see `main()`.
        slow_rate: slow down simulation by a time factor (e.g. 1.0 for real-time,
            2.0 for 2x slower than real-time).

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

    if episodes is not None:
        _spread_episodes(config, episodes, houses)

    config.output_dir = Path(output_dir) / task
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.save_config()

    StretchRolloutRunner.visualize = visualize
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
        f"{num_workers} workers -> {config.output_dir}"
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
                visualize=visualize,
                slow_rate=slow_rate,
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
