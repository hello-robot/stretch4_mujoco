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
  the camera images it is looking at. Its "Robot Closeup" tab opens on the robot
  in its room, from the same place the MuJoCo viewer's camera sits
  (`robot_view_framing` picks that spot for both), and shows the scene's visual
  meshes only -- a benchmark house carries about six collision meshes for every
  one you can see, and `_is_collision_geom` is what tells them apart. The camera
  views are the cameras the policy reads, or whatever `--visualize-camera` names.

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
import math
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from examples.machine_learning.molmospaces.stretch.config import (
    HEAD_CAMERA,
    HEAD_CAMERA_LEFT,
    HEAD_CAMERA_RIGHT,
    WRIST_CAMERA_LEFT,
    WRIST_CAMERA_RIGHT,
    Stretch4RobotConfig,
)
from examples.machine_learning.molmospaces.stretch.robot import STRETCH_ROOT_BODY

log = logging.getLogger(__name__)

CAMERA_NAMES = [
    WRIST_CAMERA_LEFT,
    WRIST_CAMERA_RIGHT,
    HEAD_CAMERA_LEFT,
    HEAD_CAMERA,
    HEAD_CAMERA_RIGHT,
]
"""
Every camera an episode renders an RGB image for, and so every camera that can
be streamed.

These are the observation keys the images arrive under: `get_core_sensors`
registers one `CameraSensor` per `Stretch4CameraSystem` camera with
`uuid=camera_name`. Only the RGB ones are here -- `wrist_camera_stereo` is
configured `record_rgb=False`.

This is the fallback, not the default: a visualizer normally streams the subset
a policy actually looks at. See `StretchRerunVisualizer._resolve_camera_names`.
"""

RERUN_MEMORY_LIMIT = "8GB"
"""
How much the spawned Rerun viewer may hold before it drops the oldest data.

Worth being generous with, because what gets dropped first is the camera feeds,
and the symptom is confusing: the 3D scene plays back over the whole run (its
meshes are logged `static=True`, which is exempt from the viewer's memory
budget) while the 2D views go blank for every episode but the last few. A
640x368 RGB frame is 700KB, so at ~190 logged steps per episode two cameras
cost roughly 250MB an episode -- an eight-episode run fits inside this with
room to spare, where the five cameras a policy is *not* looking at would not.
"""


def _lower_name(value: Any) -> str:
    """A MuJoCo name, lowercased -- or "" for anything that is not a name."""
    return value.lower() if isinstance(value, str) else ""


def _is_collision_geom(model: Any, geom_id: int) -> bool:
    """Whether a geom is there to collide rather than to be seen.

    Two signals, because neither alone covers the models one episode loads:

    * **Contact bits.** A housegen scene declares everything it draws
      `contype="0" conaffinity="0"` (its `__VISUAL_MJT__` default class) and
      everything structural or dynamic with contact bits set. That is the test
      that tells a house's 340 visual meshes from its ~2100 collision meshes,
      and it is the only one that can: a wall's visual geom and its collision
      geom reference the *same* mesh, `wall_4_0`, so the mesh name says nothing.
    * **The name.** Stretch's own MJCF needs the other test. Some of its
      collision geoms are deliberately contactless -- `grasp_center_collision_link`
      and its neighbours, see `models/stretch_4/mjcf_generator.py` -- but every
      one of them is named `*_collision_link`.
    """
    if "collision" in _lower_name(_geom_label(model, geom_id)):
        return True
    if "collision" in _lower_name(_mesh_label(model, geom_id)):
        return True
    try:
        return bool(int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id]))
    except (TypeError, ValueError, IndexError, KeyError):
        # A model that will not say is drawn rather than dropped.
        return False


def _geom_label(model: Any, geom_id: int) -> Any:
    try:
        return model.geom(geom_id).name
    except Exception:  # noqa: BLE001 - an unnamed geom is not an error
        return None


def _mesh_label(model: Any, geom_id: int) -> Any:
    try:
        mesh_id = int(model.geom_dataid[geom_id])
        return model.mesh(mesh_id).name if mesh_id >= 0 else None
    except Exception:  # noqa: BLE001 - a primitive has no mesh
        return None


def _renderable_geoms(model: Any, geom_ids: Iterable[int]) -> list[int]:
    """Which of `geom_ids` to draw: a body's visual geoms, or all of them if it has none.

    Judged per body rather than per model, because "collision only" is a
    property of a body. Every body a benchmark house draws has at least one
    visual geom, so its collision hulls all drop out here -- but the world
    body's `floor` plane is collision geometry with no visual counterpart, and
    dropping that would take the ground out from under the robot.
    """
    by_body: dict[int, list[int]] = {}
    for geom_id in geom_ids:
        by_body.setdefault(int(model.geom_bodyid[geom_id]), []).append(geom_id)

    renderable: list[int] = []
    for geoms in by_body.values():
        visual = [g for g in geoms if not _is_collision_geom(model, g)]
        renderable.extend(visual or geoms)
    return sorted(renderable)


class StretchRerunVisualizer:
    """Streams 3D robot meshes, object meshes, coordinate frames, target grasp, and waypoints to Rerun."""

    def __init__(
        self,
        spawn: bool = True,
        port: int = 9876,
        app_id: str = "Stretch4 Datagen",
        camera_names: Sequence[str] | None = None,
    ):
        self._spawn = spawn
        self._port = port
        self._app_id = app_id
        self._initialized = False
        self._logged_meshes: set[str] = set()
        self._last_logged_waypoint_idx = -1
        self._logged_grasp_lost = False
        # The caller's choice of cameras, and the choice in force for the
        # episode running now -- which is the policy's own set when the caller
        # named none. See `_resolve_camera_names`.
        self._requested_cameras = list(camera_names) if camera_names else None
        self._camera_names: list[str] = list(camera_names) if camera_names else list(CAMERA_NAMES)
        self._announced_cameras: list[str] | None = None

    def start_episode(self, episode_seed: int, task: Any, policy: Any = None) -> None:
        """Starts a new Rerun recording for each episode."""
        try:
            import uuid
            import rerun as rr

            rec_id = f"episode_{episode_seed}_{uuid.uuid4().hex[:8]}"
            app_id = self._app_id

            self._camera_names = self._resolve_camera_names(policy)
            blueprint = self._build_blueprint()

            if not self._initialized:
                rr.init(app_id, recording_id=rec_id, spawn=False, default_blueprint=blueprint)
                if self._spawn:
                    try:
                        rr.spawn(port=self._port, memory_limit=RERUN_MEMORY_LIMIT)
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

    def focus_on_robot(self, task: Any) -> None:
        """Re-send the blueprint with the closeup tab's eye aimed at the robot.

        Called from the rollout's `reset`, because that is the first moment the
        robot is standing where the episode wants it: `start_episode` runs
        before it, so the blueprint sent there can only carry Rerun's own
        default eye, which frames the whole house.

        The eye comes from `robot_view_framing`, the same occlusion-checked
        framing the MuJoCo viewer's free camera gets, so the two views of the
        episode look at the robot from the same place.
        """
        if not self._initialized:
            return
        framing = robot_view_framing(task)
        if framing is None:
            return
        try:
            import rerun as rr

            rr.send_blueprint(self._build_blueprint(framing))
        except Exception as e:  # noqa: BLE001 - a stale camera is not worth an exception
            log.debug(f"Could not aim the Rerun 3D eye at the robot: {e}")

    def _resolve_camera_names(self, policy: Any) -> list[str]:
        """The cameras to stream this episode: the caller's choice, else the policy's.

        A fine-tuned checkpoint records the cameras it was trained on, and
        `StretchMolmoBotPolicy` serves exactly those rather than its config's
        (see `policies/molmobot_checkpoint.py`), so the policy already knows the
        answer -- and it is the interesting one, since a camera the policy never
        looks at tells you nothing about why it did what it did. Streaming the
        others is also what pushes the images the policy *does* see out of the
        viewer's memory budget; see `RERUN_MEMORY_LIMIT`.

        `CAMERA_NAMES` is the last resort, for a policy that names no cameras
        at all -- the waypoint experts, which read poses rather than pixels.
        """
        if self._requested_cameras:
            return list(self._requested_cameras)

        for holder in (
            policy,
            # MolmoBot's own `SynthVLAPolicy`, which `StretchMolmoBotPolicy` wraps.
            getattr(policy, "_inner", None),
            self._unwrap_policy(policy),
            getattr(getattr(policy, "config", None), "policy_config", None),
        ):
            names = getattr(holder, "camera_names", None)
            # Checked rather than trusted: `camera_names` is read off objects
            # this module does not own, and a test double answers every
            # attribute with something truthy that is not a list of strings.
            if not isinstance(names, (list, tuple)) or not names:
                continue
            if all(isinstance(name, str) for name in names):
                resolved = list(names)
                break
        else:
            resolved = list(CAMERA_NAMES)

        if resolved != self._announced_cameras:
            self._announced_cameras = list(resolved)
            log.info(f"[visualize] streaming cameras {resolved}")
        return resolved

    def _build_blueprint(self, framing: tuple[np.ndarray, float, float] | None = None) -> Any:
        """The viewer layout, for the cameras and robot framing now in force.

        Rebuilt rather than mutated because a Rerun blueprint is sent whole, and
        it is sent twice per episode: once from `start_episode` and once from
        `focus_on_robot`, when the robot's reset pose is known.
        """
        import rerun.blueprint as rrb

        right_column = [
            rrb.Grid(
                contents=[
                    rrb.Spatial2DView(origin=f"world/cameras/{cam_name}", name=cam_name)
                    for cam_name in self._camera_names
                ]
            ),
            rrb.Horizontal(
                rrb.TextDocumentView(origin="planner/waypoint", name="Current Waypoint"),
                rrb.TextLogView(origin="logs/waypoints", name="Waypoint Log"),
            ),
        ]
        # 3D scene > cameras > text, so the scene reads at a glance and the text
        # panes stay a strip along the bottom of the right column. With no
        # cameras to show, the text gets the column to itself.
        row_shares = [3, 1] if self._camera_names else None
        if not self._camera_names:
            right_column = right_column[1:]

        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Tabs(
                    rrb.Spatial3DView(
                        origin="world",
                        # The whole scene, house included, minus the camera
                        # images -- they are 2D, so a 3D view can only list
                        # them. `setup_meshes` logs a body's collision geometry
                        # only where it has no visual geometry, so "everything"
                        # here is already the collision-free scene: the ~2100
                        # collision meshes of a benchmark house are not logged
                        # at all, rather than hidden by this filter.
                        contents=["+ $origin/**", "- $origin/cameras/**"],
                        name="Robot Closeup",
                        eye_controls=self._closeup_eye(framing),
                    ),
                    rrb.Spatial3DView(
                        origin="world",
                        contents=["+ $origin/**", "- $origin/scene_objects/**"],
                        name="Robot and Object",
                    ),
                    rrb.Spatial3DView(
                        origin="world",
                        contents=["+ $origin/**"],
                        name="3D Scene (Complete)",
                    ),
                    active_tab=0,
                ),
                rrb.Vertical(*right_column, row_shares=row_shares),
                column_shares=[3, 2],
            ),
            collapse_panels=True,
        )

    @staticmethod
    def _closeup_eye(framing: tuple[np.ndarray, float, float] | None) -> Any:
        """The closeup tab's 3D eye, put where the viewer's free camera goes.

        None until the robot's pose is known, which leaves Rerun's own eye in
        place -- and None as well on a `rerun-sdk` without `EyeControls3D`,
        which is a blueprint archetype Rerun still marks unstable, so the tab
        is worth having without it rather than not at all.
        """
        if framing is None:
            return None
        lookat, distance, azimuth = framing
        try:
            import rerun.blueprint as rrb

            position = lookat + _eye_direction(azimuth, FREE_CAMERA_ELEVATION) * distance
            return rrb.EyeControls3D(
                kind="Orbital",
                position=[float(v) for v in position],
                look_target=[float(v) for v in lookat],
                eye_up=[0.0, 0.0, 1.0],
            )
        except Exception as e:  # noqa: BLE001 - see the docstring
            log.debug(f"Rerun blueprint eye unavailable, keeping the default: {e}")
            return None

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
        robot_geoms = _renderable_geoms(
            model,
            (
                g
                for g in range(model.ngeom)
                if "robot_0" in model.body(model.geom_bodyid[g]).name
                and model.geom_dataid[g] >= 0
            ),
        )

        for g in robot_geoms:
            b_id = model.geom_bodyid[g]
            b_name = model.body(b_id).name
            g_mesh = model.geom_dataid[g]

            geom_key = f"robot/{b_name}/{g}"
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
        obj_geoms = _renderable_geoms(
            model, (g for g in range(model.ngeom) if model.geom_bodyid[g] in obj_body_ids)
        )

        for g in obj_geoms:
            b_id = model.geom_bodyid[g]
            b_name = model.body(b_id).name
            g_type = model.geom_type[g]
            g_mesh = model.geom_dataid[g]

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
        scene_geoms = _renderable_geoms(
            model,
            (
                g
                for g in range(model.ngeom)
                if model.geom_bodyid[g] not in obj_body_ids
                and "robot_0" not in model.body(model.geom_bodyid[g]).name
            ),
        )

        for g in scene_geoms:
            b_id = model.geom_bodyid[g]
            b_name = model.body(b_id).name
            g_type = model.geom_type[g]
            g_mesh = model.geom_dataid[g] if hasattr(model, "geom_dataid") else -1

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
            #
            # Only the cameras the blueprint lays out -- normally the ones the
            # policy is looking at. An episode renders every camera in
            # `Stretch4CameraSystem` whether or not anything reads it, and
            # logging the unread ones costs the viewer's whole memory budget in
            # images nobody can see. See `RERUN_MEMORY_LIMIT`.
            if observation is not None:
                obs_dict = observation[0] if isinstance(observation, list) and observation else observation
                if isinstance(obs_dict, dict):
                    for cam_name in self._camera_names:
                        img = obs_dict.get(cam_name)
                        if img is not None and hasattr(img, "ndim") and img.ndim == 3:
                            rr.log(f"world/cameras/{cam_name}", rr.Image(img))

                        depth_img = obs_dict.get(f"{cam_name}_depth")
                        if depth_img is not None and hasattr(depth_img, "ndim") and depth_img.ndim in (2, 3):
                            if depth_img.ndim == 3 and depth_img.shape[-1] == 1:
                                depth_img = depth_img.squeeze(-1)
                            rr.log(f"world/cameras/{cam_name}_depth", rr.DepthImage(depth_img))
        except Exception as e:
            log.debug(f"Error logging to Rerun: {e}")


# Framing for the viewer's free camera, all measured against benchmark episodes
# rather than guessed -- see `StretchRobot._add_chase_camera`, which found that
# anything 1.5m or more behind the robot was inside a wall in every episode
# tried, and that everything within about a metre had a clear view in five of
# six. So the offsets here start close and behind-right, which is where the
# chase camera sits, and only back off if that view is blocked.
FREE_CAMERA_LOOKAT_HEIGHT = 0.85
"""
Metres above the base body to aim at.

Halfway up the robot, so the base and the head are both in frame: `head_link`
sits 1.64m above the floor with the lift down, and `robot_0/base` is on it.
"""

FREE_CAMERA_ELEVATION = -20.0

FREE_CAMERA_DISTANCES = (2.2, 1.8, 1.4)
"""
Camera distances to try, in order of preference.

At the model's 45 degree fovy the visible half-height at the lookat plane is
`distance * tan(fovy/2) / cos(elevation)`, so 2.2m puts the top of the frame at
z=1.82 -- the head, at 1.64, with room above it. The fallbacks step *inwards*
rather than out: per `_add_chase_camera`'s measurements the wall risk grows with
distance, so when the robot is boxed in the way out is to close in and lose the
head, not to back off into the wall.
"""

FREE_CAMERA_BEARINGS = (-155.0, -115.0, 155.0, 115.0, -65.0, 65.0, 180.0, 0.0)
"""
Directions to place the camera, in degrees to the left of the base's forward axis.

-155 is the chase camera's own bearing (`pos = [-1.0, -0.45, 1.9]`, so behind and
to the robot's right); the rest sweep outwards from it, keeping a view over the
robot's shoulder wherever one is available.
"""

FREE_CAMERA_CLEARANCE = 0.45
"""
Where the occlusion ray starts, in metres from the lookat point.

The ray is cast outwards from the robot rather than inwards from the camera, so
it has to clear Stretch's own mast and base first or the robot would occlude
every candidate view of itself.
"""


def _base_pose_of(task: Any) -> tuple[np.ndarray, float] | None:
    """The robot's world position and yaw at reset, or None if it cannot be read.

    Tries the live MuJoCo state first, since that is the pose the episode is
    actually standing in, and falls back to the pose the episode spec asked for.
    """
    env = getattr(task, "env", None)
    if env is not None:
        try:
            model, data = env.current_model, env.current_data
        except Exception:  # noqa: BLE001 - an env mid-reset simply has no state yet
            model = data = None
        if model is not None and data is not None:
            # `robot_0/base` is the holonomic base body `StretchRobot` adds to the
            # worldbody; `robot_0/stretch4` is the robot root parented under it.
            prefix = getattr(getattr(task, "robot", None), "namespace", None) or (
                Stretch4RobotConfig.model_fields["robot_namespace"].default
            )
            for name in (f"{prefix}base", f"{prefix}{STRETCH_ROOT_BODY}"):
                try:
                    body = model.body(name)
                except KeyError:
                    continue
                position = np.array(data.xpos[body.id], dtype=float)
                # Column 0 of the body frame is its forward axis, in world coordinates.
                forward = np.array(data.xmat[body.id], dtype=float).reshape(3, 3)[:, 0]
                return position, float(np.arctan2(forward[1], forward[0]))

    # The base pose the episode was authored with, as a 4x4 or a 7-vector.
    pose = getattr(getattr(task, "config", None), "task_config", None)
    pose = getattr(pose, "robot_base_pose", None)
    if pose is not None:
        pose = np.asarray(pose, dtype=float)
        if pose.shape == (4, 4):
            return pose[:3, 3], float(np.arctan2(pose[1, 0], pose[0, 0]))
        if pose.shape[-1] >= 3:
            return pose[:3], 0.0
    return None


def _camera_azimuth(bearing_deg: float, base_yaw: float) -> float:
    """MuJoCo azimuth that puts the camera `bearing_deg` off the base's forward axis.

    MuJoCo's azimuth names the direction the camera *looks along*, and its eye sits
    on the opposite side of the lookat point, so the bearing has to be flipped.
    """
    angle = base_yaw + math.radians(bearing_deg)
    return math.degrees(math.atan2(-math.sin(angle), -math.cos(angle)))


def _eye_direction(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Unit vector from the lookat point towards the camera's eye.

    Verified against `mjv_updateScene`: for a free camera the eye sits at
    `lookat + distance * this`, so a negative elevation puts it overhead.
    """
    azimuth, elevation = math.radians(azimuth_deg), math.radians(elevation_deg)
    return np.array(
        [
            -math.cos(elevation) * math.cos(azimuth),
            -math.cos(elevation) * math.sin(azimuth),
            -math.sin(elevation),
        ]
    )


def _view_is_clear(task: Any, lookat: np.ndarray, eye_direction: np.ndarray, distance: float) -> bool:
    """Whether anything stands between the camera's eye and the robot.

    Casts one ray outwards from the robot along the sightline. Static geometry is
    included, because a wall is exactly what this is looking for.
    """
    span = distance - FREE_CAMERA_CLEARANCE
    if span <= 0.0:
        return True
    try:
        model, data = task.env.current_model, task.env.current_data
        geom_id = np.zeros(1, dtype=np.int32)
        hit = mujoco.mj_ray(
            model,
            data,
            lookat + eye_direction * FREE_CAMERA_CLEARANCE,
            eye_direction,
            None,
            1,
            -1,
            geom_id,
        )
    except Exception as error:  # noqa: BLE001 - an untestable view is not a fatal one
        log.debug(f"Occlusion ray failed, accepting the view as-is: {error}")
        return True
    return hit < 0.0 or hit > span


def robot_view_framing(task: Any) -> tuple[np.ndarray, float, float] | None:
    """Where to put a camera for a clear, close view of the robot.

    Returns `(lookat, distance, azimuth)` -- MuJoCo free-camera terms, because
    that is what the passive viewer takes -- or None if the robot's pose cannot
    be read at all. Every candidate view is tested against the scene's own
    geometry, so what comes back is a sightline that is actually clear; see the
    constants above for where the candidates come from.

    Shared with the Rerun blueprint's 3D eye
    (`StretchRerunVisualizer._closeup_eye`), so the two views of an episode look
    at the robot from the same place rather than each having their own idea of
    where it is.
    """
    pose = _base_pose_of(task)
    if pose is None:
        return None
    robot_position, base_yaw = pose

    lookat = np.array(
        [robot_position[0], robot_position[1], robot_position[2] + FREE_CAMERA_LOOKAT_HEIGHT]
    )

    # Preferred distance first, so a clear view is also a well-framed one.
    for distance in FREE_CAMERA_DISTANCES:
        for bearing in FREE_CAMERA_BEARINGS:
            azimuth = _camera_azimuth(bearing, base_yaw)
            direction = _eye_direction(azimuth, FREE_CAMERA_ELEVATION)
            if _view_is_clear(task, lookat, direction, distance):
                return lookat, distance, azimuth

    # Blocked from every side -- the robot is boxed in. Sit as close as
    # possible on the chase camera's bearing: inside the wall beats 70m
    # outside it, and the near clipping plane hides what the eye is buried in.
    log.info("[visualize] every candidate view of the robot is occluded; using the closest")
    return lookat, FREE_CAMERA_DISTANCES[0], _camera_azimuth(FREE_CAMERA_BEARINGS[0], base_yaw)


def snap_free_camera_to_robot(viewer: Any, task: Any) -> None:
    """Put MuJoCo's passive viewer on a free camera, aimed at the robot.

    Called once per episode, since the framing is only right for the pose the
    robot resets into. It stays a *free* camera afterwards -- not tracking, not
    fixed -- so the mouse keeps full control from that starting point, and it does
    not swing around as the base turns. Press `[` / `]` in the viewer to cycle to
    the model's own cameras (Stretch mounts a chase camera and its head camera),
    or Esc to come back here.

    The camera is only taken over once the robot's pose is in hand. MuJoCo's
    default free camera frames the whole model, and a benchmark house is loaded in
    its sealed "ceiling" variant, so that default is a building shot from ~70m
    out -- switching to it and then failing to aim is strictly worse than leaving
    the fixed chase camera `setup_viewer` configured.
    """
    if viewer is None:
        return
    try:
        framing = robot_view_framing(task)
        if framing is None:
            log.warning(
                "Could not read the robot's base pose, so the viewer keeps the chase "
                "camera. Press Esc for the free camera."
            )
            return
        lookat, distance, azimuth = framing

        # The viewer renders from its own thread, so the camera is written under
        # the handle's lock -- otherwise a frame can be drawn from a half-updated
        # one. A viewer stub without `lock()` just gets the plain writes.
        lock = getattr(viewer, "lock", None)
        with lock() if callable(lock) else contextlib.nullcontext():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.fixedcamid = -1
            viewer.cam.lookat[0] = float(lookat[0])
            viewer.cam.lookat[1] = float(lookat[1])
            viewer.cam.lookat[2] = float(lookat[2])
            viewer.cam.distance = distance
            viewer.cam.elevation = FREE_CAMERA_ELEVATION
            viewer.cam.azimuth = azimuth
        log.info(
            f"[visualize] free camera on the robot at {np.round(lookat, 2).tolist()}, "
            f"{distance}m out, azimuth {azimuth:.0f}"
        )
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
        visualizer.focus_on_robot(task)
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


_HOUSE_SCENE_STEM = re.compile(r"^(?:train|val|test)_\d+(?:_ceiling)?$")
"""
Filenames housegen gives a house scene, as `<split>_<index>[_ceiling]`.

Documented at `molmo_spaces_constants.py:392` and parsed the same way there. This
is what tells a house apart from the receptacles, pickup objects and robot that
`MjSpec.from_file` also loads.
"""


def name_viewer_window_after(task_name: str) -> None:
    """Make the viewer's title bar name the task instead of the house.

    MuJoCo titles its window `"MuJoCo : " + <model name>`; the string is composed
    in C++ inside `_simulate`, so the prefix cannot be dropped from here. The name
    after it comes from the house scene, which `housegen/exporter.py` stamps with
    `f"{split}_{house_index}"` -- so the title reads "MuJoCo : val_0" and changes
    with every episode.

    A compiled model cannot be renamed (`MjModel.names` is `bytes` in Python, and
    the viewer has to be handed the very model the simulation steps), so the rename
    happens on the spec, before `task_sampler.setup_robot_scene` compiles it.

    Narrowed to house scenes by filename on purpose: the same `MjSpec.from_file`
    loads receptacles, pickup objects and the robot, and those specs get attached
    into the scene, where their names are not ours to rename.
    """
    if getattr(mujoco.MjSpec.from_file, "_stretch_scene_name", None) is not None:
        # Already installed -- just retarget it, so a sweep over several
        # benchmarks titles each one rather than keeping the first.
        mujoco.MjSpec.from_file._stretch_scene_name = task_name
        return

    original_from_file = mujoco.MjSpec.from_file

    @functools.wraps(original_from_file)
    def from_file(filename: Any, *args: Any, **kwargs: Any) -> Any:
        spec = original_from_file(filename, *args, **kwargs)
        name = from_file._stretch_scene_name
        if name and _HOUSE_SCENE_STEM.match(Path(str(filename)).stem):
            with contextlib.suppress(Exception):
                # Cosmetic: a spec that will not take a name is not worth an
                # exception on the scene-loading path.
                spec.modelname = name
        return spec

    from_file._stretch_scene_name = task_name
    mujoco.MjSpec.from_file = staticmethod(from_file)
    log.info(f"[visualize] viewer window title: MuJoCo : {task_name}")


def _hide_viewer_panels() -> None:
    """Open MuJoCo's passive viewer with both side panels collapsed.

    `launch_passive` takes `show_left_ui` / `show_right_ui`, but the call that
    creates the viewer is MolmoSpaces' `setup_viewer`, which does not pass either
    -- so wrapping the function is the way in. `setup_viewer` looks the name up on
    the module at call time, so replacing the attribute is enough, and it re-opens
    the viewer once per episode, so this has to hold for the whole process rather
    than just the first window.

    Tab and Shift+Tab still bring the panels back once the window is open.
    """
    import mujoco.viewer

    if getattr(mujoco.viewer.launch_passive, "_stretch_hidden_panels", False):
        return

    original_launch_passive = mujoco.viewer.launch_passive

    @functools.wraps(original_launch_passive)
    def launch_passive(*args: Any, **kwargs: Any) -> Any:
        # setdefault, not an override: an explicit caller still wins.
        kwargs.setdefault("show_left_ui", False)
        kwargs.setdefault("show_right_ui", False)
        return original_launch_passive(*args, **kwargs)

    launch_passive._stretch_hidden_panels = True
    mujoco.viewer.launch_passive = launch_passive


def install_eval_visualize_hook(
    spawn: bool = True, port: int = 9876, camera_names: Sequence[str] | None = None
) -> None:
    """Give evaluation rollouts the same two views datagen's `--visualize` gets.

    `camera_names` is `--visualize-camera`: the cameras to stream and lay out.
    Left None, each episode streams the cameras its policy reads, which for a
    fine-tuned checkpoint is the set it was trained on -- see
    `StretchRerunVisualizer._resolve_camera_names`.

    Idempotent: eval configs are re-imported when MolmoSpaces resolves a
    "module:Class" string, and the workers import them again, so this gets called
    more than once per process.
    """
    from molmo_spaces.evaluation.json_eval_runner import JsonEvalRunner

    _hide_viewer_panels()

    if getattr(JsonEvalRunner.run_single_rollout, "_stretch_visualize_hook", False):
        return

    original_run_single_rollout = JsonEvalRunner.run_single_rollout
    # One visualizer for the whole process: `start_episode` opens a fresh Rerun
    # recording per episode, but spawning the viewer and connecting to it happen
    # once, on the first episode.
    visualizer = StretchRerunVisualizer(
        spawn=spawn, port=port, app_id="Stretch4 Benchmark Eval", camera_names=camera_names
    )

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
