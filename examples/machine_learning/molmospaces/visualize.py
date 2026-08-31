"""
What `--visualize` shows of a Stretch rollout, shared by datagen and evaluation.

Two views of the same episode, and both halves are shared so a rollout looks the
same whichever pipeline produced it -- `finetuning/generate_dataset.py --visualize`
or `run_benchmarks.py --visualize`:

* MuJoCo's passive viewer shows the scene. `snap_free_camera_to_robot()` aims its
  *free* camera at the robot at the start of each episode, which is the framing
  worth having: MuJoCo's own default frames the whole model, and a benchmark house
  loaded in its "ceiling" variant is then a sealed building seen from ~70m away
  with the robot invisible inside. Free rather than tracking or fixed, so orbiting,
  panning and zooming all still do what you expect once you take the camera over.
* `StretchRerunVisualizer` streams what the policy is working from -- the target
  grasp, the waypoint plan and its progress, the frames the IK solves in, and
  whatever camera images the observation carries.

Datagen drives both directly, from its own `ParallelRolloutRunner` subclass.
Evaluation cannot: `run_evaluation()` constructs and runs `JsonEvalRunner` itself
and takes no `runner_class`, so there is nothing to subclass into the loop.
`install_eval_visualize_hook()` wraps `JsonEvalRunner.run_single_rollout` instead
-- the same seam `added_pickup_repair` patches on the eval path -- and works from
the task's own `reset`/`step_chunk` rather than from a copy of the rollout loop, so
it inherits upstream changes to that loop instead of drifting from them.
"""

from __future__ import annotations

import contextlib
import functools
import logging
from typing import Any

import mujoco
import numpy as np

log = logging.getLogger(__name__)


class StretchRerunVisualizer:
    """Streams 3D robot meshes, object meshes, coordinate frames, target grasp, and waypoints to Rerun."""

    def __init__(self, spawn: bool = True, port: int = 9876, app_id: str = "Stretch4 Datagen"):
        self._spawn = spawn
        self._port = port
        self._app_id = app_id
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
            app_id = self._app_id

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
    """Put MuJoCo's passive viewer on a free camera, aimed at the robot.

    Called once per episode, since the framing is only right for the pose the
    robot resets into. It stays a *free* camera afterwards -- not tracking, not
    fixed -- so the mouse keeps full control from that starting point, and it does
    not swing around as the base turns. Press `[` / `]` in the viewer to cycle to
    the model's own cameras (Stretch mounts a chase camera and its head camera),
    or Esc to come back here.
    """
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


@contextlib.contextmanager
def _visualize_rollout(
    visualizer: StretchRerunVisualizer, task: Any, policy: Any, viewer: Any = None
):
    """Drive both views from `task.reset` and `task.step_chunk`, for one rollout.

    Shadowing the two methods on the task instance, rather than reimplementing
    `run_single_rollout`, is what keeps this hook thin: the step count, the chunk
    length and the observation are all right here, and the rollout loop upstream
    stays the one actually running the episode. `reset` is also the earliest point
    at which the robot is standing where the episode wants it, so it is where the
    viewer camera gets aimed.
    """
    original_reset = task.reset
    original_step_chunk = task.step_chunk
    steps = 0

    def reset(*args: Any, **kwargs: Any) -> Any:
        result = original_reset(*args, **kwargs)
        observation = result[0] if isinstance(result, tuple) else result
        snap_free_camera_to_robot(viewer, task)
        visualizer.log_step(0, task, observation, policy=policy)
        return result

    def step_chunk(action_chunk: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal steps
        result = original_step_chunk(action_chunk, *args, **kwargs)
        steps += len(action_chunk)
        visualizer.log_step(steps, task, result[0], policy=policy)
        return result

    task.reset = reset
    task.step_chunk = step_chunk
    try:
        yield
    finally:
        # Instance attributes shadowing the class methods, so deleting them
        # restores the originals. A task is per-episode and dropped after this,
        # but leaving a closure over it alive would pin the whole MuJoCo model.
        for name in ("reset", "step_chunk"):
            task.__dict__.pop(name, None)


def install_eval_visualize_hook(spawn: bool = True, port: int = 9876) -> None:
    """Give evaluation rollouts the same two views datagen's `--visualize` gets.

    Idempotent: eval configs are re-imported when MolmoSpaces resolves a
    "module:Class" string, and the workers import them again, so this gets called
    more than once per process.
    """
    from molmo_spaces.evaluation.json_eval_runner import JsonEvalRunner

    if getattr(JsonEvalRunner.run_single_rollout, "_stretch_visualize_hook", False):
        return

    original_run_single_rollout = JsonEvalRunner.run_single_rollout
    # One visualizer for the whole process: `start_episode` opens a fresh Rerun
    # recording per episode, but spawning the viewer and connecting to it happen
    # once, on the first episode.
    visualizer = StretchRerunVisualizer(spawn=spawn, port=port, app_id="Stretch4 Benchmark Eval")

    @functools.wraps(original_run_single_rollout)
    def run_single_rollout(episode_seed: int, task: Any, policy: Any, **kwargs: Any) -> bool:
        visualizer.start_episode(episode_seed, task, policy=policy)
        with _visualize_rollout(visualizer, task, policy, viewer=kwargs.get("viewer")):
            return original_run_single_rollout(
                episode_seed=episode_seed, task=task, policy=policy, **kwargs
            )

    run_single_rollout._stretch_visualize_hook = True
    JsonEvalRunner.run_single_rollout = staticmethod(run_single_rollout)
    log.info(f"[visualize] evaluation rollouts stream to Rerun on port {port}")
