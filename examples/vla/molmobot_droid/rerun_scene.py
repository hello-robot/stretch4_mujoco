"""
Rerun view of a MolmoBot-DROID run: the whole MolmoSpaces scene in 3D, built from the same
MuJoCo model the simulation runs (so every mesh comes straight from molmospaces), the robot
and target object moving in it, the ghost Franka when there is one, the cameras the policy
sees, and the TCP target vs. where the tool actually is.

The geometry is drawn by `examples/digital_twin.py`'s `RerunMujocoRobot`.
"""

from __future__ import annotations

import mujoco
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from examples.digital_twin import RerunMujocoRobot

SCENE_PATH = "world/scene"
CAMERAS_PATH = "cameras"
LOG_PATH = "log"


_session: dict = {}
"""The app, `init_rerun()`'s layout and whether a viewer was spawned, for `new_recording()`."""


def init_rerun(
    app_name: str,
    camera_names: list[str],
    spawn: bool = True,
    text_log: bool = False,
    main_cameras: list[str] | None = None,
    eye: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """
    The scene camera, when there is one, is the big view; the 3D scene sits in the column with
    the other cameras and the metrics, and with `text_log`, the messages under LOG_PATH.

    With `main_cameras`, that layout is the Diagnostics tab, and the tab shown first is the 3D
    scene with only those cameras beside it.

    `eye` (position, look target, in the world frame) is where the 3D scene views start, e.g.
    `scene_camera_pose()`, instead of zoomed out on everything logged.
    """
    rr.init(app_name, spawn=False)
    if spawn:
        rr.spawn(memory_limit="5GB")
    _session.update(
        app_name=app_name, spawned=spawn, camera_names=camera_names, text_log=text_log, main_cameras=main_cameras
    )
    rr.send_blueprint(_blueprint(eye))


def _blueprint(eye: tuple[np.ndarray, np.ndarray] | None) -> rrb.Blueprint:
    """`init_rerun()`'s layout, its 3D scene views starting at `eye`."""
    camera_names, main_cameras = _session["camera_names"], _session["main_cameras"]
    eye_controls = (
        rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital, position=eye[0], look_target=eye[1], eye_up=[0, 0, 1])
        if eye is not None
        else None
    )

    def scene_3d() -> rrb.Spatial3DView:
        return rrb.Spatial3DView(origin="world", name="3D scene", eye_controls=eye_controls)

    big = "scene" if "scene" in camera_names else None
    main_view = rrb.Spatial2DView(origin=f"{CAMERAS_PATH}/{big}", name=big) if big else scene_3d()
    column = ([scene_3d()] if big else []) + [
        rrb.Spatial2DView(origin=f"{CAMERAS_PATH}/{name}", name=name) for name in camera_names if name != big
    ]
    layout = rrb.Horizontal(
        main_view,
        rrb.Vertical(
            *column,
            rrb.TimeSeriesView(origin="metrics", name="Metrics"),
            *([rrb.TextLogView(origin=LOG_PATH, name="Log")] if _session["text_log"] else []),
        ),
        column_shares=[3, 2],
        name="Diagnostics",
    )
    if main_cameras:
        main = rrb.Horizontal(
            scene_3d(),
            rrb.Vertical(*[rrb.Spatial2DView(origin=f"{CAMERAS_PATH}/{name}", name=name) for name in main_cameras]),
            column_shares=[3, 2],
            name="Main",
        )
        layout = rrb.Tabs(main, layout, active_tab=0)
    return rrb.Blueprint(layout, collapse_panels=True)


def new_recording(name: str, eye: tuple[np.ndarray, np.ndarray] | None = None) -> None:
    """
    Log to a fresh recording from now on, in the same viewer and with the same layout (one per
    benchmark episode, `name`d in the viewer's list), its 3D scene starting at `eye` (see
    `init_rerun()`): each keeps its own static scene geometry, so an earlier one still replays
    with its scene, where clearing and relogging one recording's would have replaced it.
    """
    recording = rr.RecordingStream(_session["app_name"], make_default=True)
    if _session["spawned"]:
        recording.connect_grpc()
    rr.send_recording_name(name, recording=recording)
    rr.send_blueprint(_blueprint(eye), recording=recording)


class RerunScene:
    """
    Logs every visual geom of `model` once, with the ones that never move placed for good
    (static), then only the ones under `moving_bodies` (the robot, the target object, the
    ghost) on each `log()`.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, moving_bodies: list[str]):
        self.model = model
        self.scene = RerunMujocoRobot(rr, model, "world", SCENE_PATH, tint=None)
        moving = {model.body(name).id for name in moving_bodies if _has_body(model, name)}
        self.moving_geoms = [
            geom for geom in self.scene.geoms if _ancestor_in(model, int(model.geom_bodyid[geom]), moving)
        ]
        moving_set = set(self.moving_geoms)
        for geom in self.scene.geoms:
            if geom not in moving_set:
                self._log_pose(geom, data, static=True)
        self.log(data)

    def _log_pose(self, geom: int, data: mujoco.MjData, static: bool = False) -> None:
        rr.log(
            self.scene._paths[geom],
            rr.Transform3D(translation=data.geom_xpos[geom], mat3x3=data.geom_xmat[geom].reshape(3, 3)),
            static=static,
        )

    def log(self, data: mujoco.MjData) -> None:
        for geom in self.moving_geoms:
            self._log_pose(geom, data)


def set_step(step: int, sim_time: float | None = None) -> None:
    """`step` on the `step` timeline and, with `sim_time` (s), when it was in the simulation on
    `sim_time`: replayed on that one, the motion plays at the speed it happened, pauses and
    start pose moves included."""
    rr.set_time("step", sequence=step)
    if sim_time is not None:
        rr.set_time("sim_time", duration=sim_time)


def log_cameras(images: dict[str, np.ndarray]) -> None:
    """RGB images, downscaled so the viewer stays responsive."""
    import cv2

    for name, image in images.items():
        if image is None:
            continue
        scale = 960 / max(image.shape[:2])
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        rr.log(f"{CAMERAS_PATH}/{name}", rr.Image(image).compress(jpeg_quality=85))


TARGET_COLOR = [255, 160, 0]
"""Orange: where the policy wants the tool."""
ACTUAL_COLOR = [0, 200, 255]
"""Blue: where the tool is."""
LABEL_HEIGHTS = {"target": 0.12, "actual": 0.06}
"""m above each tool point its label sits, on a stalk: clear of the gripper, and of each other
when the two points meet."""


def log_tool_poses(
    target_world: np.ndarray | None, actual_world: np.ndarray | None, actual_label: str = "Stretch tool"
) -> None:
    """
    Where the policy wants the tool (orange, from the Franka) vs. where it is (blue, `actual_label`),
    each a point with its axes. The target's point is the larger, see-through one, so on target
    reads as a blue dot inside an orange ball; a line joins them when both are known, and the
    blue label says how far apart they are. The labels are under `world/tool/labels`, to hide
    them all at once.
    """
    error = (
        np.linalg.norm(actual_world[:3, 3] - target_world[:3, 3])
        if target_world is not None and actual_world is not None
        else None
    )
    markers = {
        "target": (target_world, "policy target", TARGET_COLOR + [140], 0.014),
        "actual": (
            actual_world,
            actual_label + (f" · {error * 1000:.0f} mm off" if error is not None else ""),
            ACTUAL_COLOR,
            0.009,
        ),
    }
    for name, (pose, label, color, radius) in markers.items():
        if pose is None:
            rr.log(f"world/tool/{name}", rr.Clear(recursive=True))
            rr.log(f"world/tool/labels/{name}", rr.Clear(recursive=True))
            continue
        position = pose[:3, 3]
        rr.log(f"world/tool/{name}", rr.Transform3D(translation=position, mat3x3=pose[:3, :3]), rr.TransformAxes3D(0.08))
        rr.log(f"world/tool/{name}/point", rr.Points3D([[0, 0, 0]], colors=[color], radii=radius))
        tip = position + [0, 0, LABEL_HEIGHTS[name]]
        rr.log(
            f"world/tool/labels/{name}",
            rr.Points3D([tip], colors=[color[:3]], radii=0.003, labels=[label], show_labels=True),
            rr.LineStrips3D([[position, tip]], colors=[color[:3] + [120]], radii=0.0008),
        )
    if error is None:
        rr.log("world/tool/error", rr.Clear(recursive=False))
    else:
        rr.log(
            "world/tool/error",
            rr.LineStrips3D([[target_world[:3, 3], actual_world[:3, 3]]], colors=[[230, 230, 230, 200]], radii=0.002),
        )


def log_metrics(values: dict[str, float]) -> None:
    for name, value in values.items():
        rr.log(f"metrics/{name}", rr.Scalars(float(value)))


def log_text(path: str, text: str, level: str = "INFO") -> None:
    """A message, at `path` under LOG_PATH for `init_rerun(text_log=True)`'s view to show it."""
    rr.log(path, rr.TextLog(text, level=level))


def _has_body(model: mujoco.MjModel, name: str) -> bool:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) != -1


def _ancestor_in(model: mujoco.MjModel, body: int, bodies: set[int]) -> bool:
    while body != 0:
        if body in bodies:
            return True
        body = int(model.body_parentid[body])
    return False
