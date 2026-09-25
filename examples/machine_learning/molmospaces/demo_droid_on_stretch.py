"""
Run the released MolmoBot-DROID checkpoint on Stretch 4, in one MolmoSpaces kitchen.

The standalone demo: one house, one task string, one MP4. It is the smallest
thing that exercises the whole retargeting chain, so it is the place to start
when something about `--policy molmobot_droid` looks wrong -- there is no
benchmark, no evaluation pipeline and no worker process between you and the
robot, and the rollout loop below is fifteen lines you can read.

    # the released checkpoint, fetched from the Hub, on the default task
    python -m examples.machine_learning.molmospaces.demo_droid_on_stretch

    # a different task, longer episode, and the real fisheye head camera
    python -m examples.machine_learning.molmospaces.demo_droid_on_stretch \\
        --task "place the salt shaker in the bowl" --episode-dur 20 --fisheye

    # what the arm alone can do, with the base held still
    python -m examples.machine_learning.molmospaces.demo_droid_on_stretch \\
        --no-include-base

For scores over a benchmark rather than one video, use
`run_benchmarks.py --policy molmobot_droid`, which runs the same retargeting
through `policies/molmobot_droid_policy.py`.

Two things to keep in mind when reading a rollout, both of them properties of the
experiment rather than bugs in it:

* Stretch's lift, arm and wrist are five DOFs against the Franka's seven, and on
  their own they reach a corridor too narrow for these tasks -- so the holonomic
  base joins the IK, on a leash. The printout at the end says how far the
  commanded tool poses fell outside what the robot could reach.
* The policy is looking at a Stretch arm through a Stretch camera, which is not
  what it was trained on. Retargeting fixes the action interface, not the visual
  domain gap. `--fisheye` decides how wide that gap is: the real right head
  camera is a 123-degree fisheye mounted sideways, so the exo frame comes out
  barrel-distorted and portrait where the Franka's was a 640x360 pinhole. The
  default keeps an upright pinhole at the same mounting point, which separates
  "can the retargeting drive this arm" from "can the policy cope with a fisheye".
"""

from __future__ import annotations

import logging
import math
import os
import sys

# MuJoCo binds the backend named by MUJOCO_GL when `mujoco` is first imported,
# which the imports below trigger -- so this has to come before them. Off-screen
# rendering through EGL needs no display, which is what makes this runnable over
# ssh; the passive viewer is not used here at all.
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import click  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from mujoco import MjData, MjModel, MjSpec  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from examples.machine_learning.molmospaces import hdf5_layout  # noqa: E402
from examples.machine_learning.molmospaces.policies import franka_retarget as fr  # noqa: E402
from examples.machine_learning.molmospaces.policies.molmobot_droid_policy import (  # noqa: E402
    DROID_ACTION_SPEC,
    DROID_EXO_CAMERA_KEY,
    DROID_WRIST_CAMERA_KEY,
    StretchMolmoBotDroidPolicy,
    StretchMolmoBotDroidPolicyConfig,
)
from examples.machine_learning.molmospaces.stretch.config import Stretch4RobotConfig  # noqa: E402
from examples.machine_learning.molmospaces.stretch.robot import Stretch4Robot  # noqa: E402
from examples.machine_learning.molmospaces.stretch.robot_view import (  # noqa: E402
    Stretch4RobotView,
)
from molmo_spaces.molmo_spaces_constants import get_procthor_10k_houses  # noqa: E402
from molmo_spaces.utils.function_utils import make_lenient  # noqa: E402
from molmo_spaces.utils.lazy_loading_utils import (  # noqa: E402
    install_scene_with_objects_and_grasps_from_path,
    install_uid,
)
from stretch4_mujoco.enums.stretch_cameras import StretchCameras  # noqa: E402

log = logging.getLogger(__name__)

# The kitchen the notebook this was ported from worked in: house 0 of the
# ProcTHOR-10k validation split, with a bowl dropped on the counter to place
# things into. Fixed rather than a flag because the base pose, the receptacle
# position and the task strings below were all chosen against this one room --
# `run_benchmarks.py` is the way to run somewhere else.
HOUSE_SPLIT = "val"
HOUSE_INDEX = 0

# Stretch's own base: set back from the counter, whose near edge is at y = 10.2,
# so a 34cm-radius chassis is on open floor. Its arm telescopes forward along the
# base's +x, so this yaw points it at the counter. The x is the midpoint of the
# objects these tasks span.
STRETCH_BASE_XY = (6.73, 9.7)
STRETCH_BASE_YAW_DEG = 90.0

# Where the bowl goes: on the counter, within reach of that standing position.
RECEPTACLE_UID = "Bowl_3"
RECEPTACLE_POS = (7.1, 10.2, 1.01)

# 15Hz, the rate every other Stretch policy here runs at, and the rate the
# benchmark eval configs use -- so a rollout from this file and one from
# `run_benchmarks.py` are stepping the same way.
POLICY_DT_MS = 66


# =============================================================================
# Rendering Stretch's cameras
# =============================================================================


def configure_stretch_cameras(spec: MjSpec, namespace: str, cameras: dict) -> None:
    """Give each MJCF camera the intrinsics and render size its hardware has.

    `cameras` maps an output name to a `StretchCameras` member. MuJoCo's default
    `fovy` is 45 degrees -- what a `<camera>` with no `fovy` attribute gets, and
    `mjcf_generator.py` writes none -- which is not any Stretch camera; the head
    fisheyes are over 123.

    That 123 is, strictly, the *horizontal* field of view of the 1920-wide sensor
    stored under a name that says vertical. It is a quirk of folding a
    calibration into one MuJoCo number, and it is reproduced rather than
    corrected here for the same reason `Stretch4CameraSystem` reproduces it: the
    simulator feeds the identical value to `cam_fovy`, and the two agreeing is
    the point.

    Also raises the offscreen framebuffer to fit the largest render. A scene's
    `<visual><global offwidth=.../>` is whatever its author wrote -- the default
    640x480 is too narrow for the gripper cameras' 656-pixel width -- and MuJoCo
    refuses to build a larger renderer than it, as a hard error.
    """
    width = int(spec.visual.global_.offwidth)
    height = int(spec.visual.global_.offheight)
    for camera in cameras.values():
        settings = camera.initial_camera_settings
        mjcf_camera = spec.camera(namespace + camera.camera_name_in_mjcf)
        render_size = hdf5_layout.camera_render_size(camera)
        mjcf_camera.fovy = float(settings.field_of_view_vertical_in_degrees)
        mjcf_camera.resolution = list(render_size)
        width = max(width, render_size[0])
        height = max(height, render_size[1])

    spec.visual.global_.offwidth = width
    spec.visual.global_.offheight = height


class StretchCameraRig:
    """Renders Stretch's cameras the way the robot's own stack delivers them.

    A `mujoco.Renderer` gives you the raw pinhole render; what every consumer of
    the real robot actually sees has been through two more steps, and this
    applies both -- the same two `install_stretch_camera_hooks()` patches into
    the MolmoSpaces datagen path, which is where this is taken from.

    * **The fisheye warp.** The head cameras are fisheyes, and MuJoCo cannot
      render one. The simulator renders a wide pinhole and warps it with the
      camera's real distortion coefficients
      (`MujocoServerCameraManagerSync._render_camera`), then crops away the
      surround the pinhole could not fill and rescales what is left back to size.
      Skip it and straight lines stay straight in a view where the hardware bends
      them, which is most of what a fisheye view looks like.
    * **The quarter turn.** Stretch's head cameras are physically mounted
      sideways, and `StatusStretchCamera.get_camera_data()` undoes it with
      `np.rot90(frame, rotate_number_of_times)` before anyone sees pixels --
      `auto_rotate` defaults to True. For the right head camera that is -1, a
      quarter turn clockwise, and it swaps the frame's width and height.

    One renderer per distinct render size, because these cameras do not share
    one: `mujoco.Renderer` is fixed-size at construction, and each camera is
    rendered at *its own* aspect ratio near a shared pixel budget, because the
    aspect is what decides how much of the scene is in frame.

    The headlight needs no fixing here, unlike in that hook: it misplaces the
    light only on MolmoSpaces' *free* camera path, where `mjv_updateScene` is
    handed a default camera and the pose is overwritten afterwards. Rendering
    through a fixed MJCF camera, MuJoCo pins the headlight to it itself.
    """

    def __init__(self, model: MjModel, namespace: str, cameras: dict) -> None:
        self._cameras = dict(cameras)
        self._mjcf_names = {
            name: namespace + camera.camera_name_in_mjcf for name, camera in self._cameras.items()
        }
        self._render_sizes = {
            name: hdf5_layout.camera_render_size(camera) for name, camera in self._cameras.items()
        }
        self._renderers = {
            size: mujoco.Renderer(model, size[1], size[0])
            for size in set(self._render_sizes.values())
        }
        self._scene_option = mujoco.MjvOption()
        self._scene_option.sitegroup = 0

    @property
    def output_sizes(self) -> dict[str, tuple[int, int]]:
        """Output (width, height) per camera, after the quarter turn."""
        return {
            name: hdf5_layout.camera_output_size(camera) for name, camera in self._cameras.items()
        }

    def describe(self) -> dict[str, str]:
        """One human-readable line per camera, for printing after setup."""
        sizes = self.output_sizes
        lines = {}
        for name, camera in self._cameras.items():
            settings = camera.initial_camera_settings
            width, height = sizes[name]
            lines[name] = (
                f"{camera.camera_name_in_mjcf} (hardware)  {width}x{height}"
                f"  fovy={settings.field_of_view_vertical_in_degrees:.1f}"
                f"  fisheye={camera.applies_fisheye_distortion}"
                f"  quarter_turns={settings.rotate_number_of_times}"
            )
        return lines

    def _postprocess(self, camera: Any, frame: np.ndarray) -> np.ndarray:
        post_processing = camera.post_processing_callback
        if post_processing is not None and not camera.is_depth:
            frame = post_processing(frame)
        quarter_turns = camera.initial_camera_settings.rotate_number_of_times
        if quarter_turns:
            frame = np.rot90(frame, quarter_turns)
        return np.ascontiguousarray(frame)

    def render(self, data: MjData) -> dict[str, np.ndarray]:
        """One RGB frame per configured camera, keyed by output name."""
        frames = {}
        for name, camera in self._cameras.items():
            renderer = self._renderers[self._render_sizes[name]]
            renderer.update_scene(
                data, camera=self._mjcf_names[name], scene_option=self._scene_option
            )
            frames[name] = self._postprocess(camera, renderer.render())
        return frames


# The exo camera the Franka demo reconstructs by hand, as mounted on Stretch: the
# true `camera_right_link` position relative to `base_link` ([0.0788, -0.075,
# 1.5432], less that body's own 0.0845 offset), with an upright pinhole basis
# looking along the same direction.
RECONSTRUCTED_EXO_CAMERA = {
    "pos": [0.0788406, -0.075, 1.4587],
    "euler_xyz_deg": [43.0, 0.0, -90.0],
    "fovy": 71.0,
    "resolution": [640, 360],
}


def add_reconstructed_exo_camera(spec: MjSpec, namespace: str, name: str = "exo_camera_1") -> None:
    """Mount an upright pinhole exo view on Stretch's base.

    Not a camera this robot has. It is the view the Franka demo built -- where
    only the position and the view direction of Stretch's head camera could be
    carried across -- and it is much closer to the policy's training distribution
    than a 123-degree fisheye is, which is the only reason to keep it available.

    `camera_right_link`'s own raw orientation is not usable directly as a camera
    orientation: on the real robot that link's local axes are not aligned to
    "camera looks down -z" -- that remap happens via the URDF's
    `camera_right_optical_link` fixed joint, and even that frame is mounted
    rolled ~90 degrees, because the real camera is physically mounted sideways.
    So the view direction is taken from `base_link -> camera_right_optical_link`
    (forward, tilted 43 degrees down) and an upright basis with zero roll against
    the world is rebuilt around it, which reduces to the euler triple below.
    """
    spec.body(namespace + "base_link").add_camera(
        pos=RECONSTRUCTED_EXO_CAMERA["pos"],
        quat=R.from_euler("xyz", RECONSTRUCTED_EXO_CAMERA["euler_xyz_deg"], degrees=True).as_quat(
            scalar_first=True
        ),
        fovy=RECONSTRUCTED_EXO_CAMERA["fovy"],
        resolution=list(RECONSTRUCTED_EXO_CAMERA["resolution"]),
        name=namespace + name,
    )


class PinholeCameraRig:
    """Plain MuJoCo renders of named MJCF cameras, with no hardware post-processing.

    The counterpart to `StretchCameraRig`, and interchangeable with it: same
    `render` / `output_sizes` / `describe`. Use it for cameras that are not
    modelling a real Stretch sensor -- `add_reconstructed_exo_camera`'s upright
    pinhole -- where there is no distortion to reproduce and no sideways mount to
    undo.
    """

    def __init__(self, model: MjModel, cameras: dict, width: int = 640, height: int = 360) -> None:
        self._mjcf_names = dict(cameras)
        self._size = (width, height)
        self._renderer = mujoco.Renderer(model, height, width)
        self._scene_option = mujoco.MjvOption()
        self._scene_option.sitegroup = 0

    @property
    def output_sizes(self) -> dict[str, tuple[int, int]]:
        return {name: self._size for name in self._mjcf_names}

    def describe(self) -> dict[str, str]:
        return {
            name: f"{mjcf_name} (pinhole)  {self._size[0]}x{self._size[1]}"
            for name, mjcf_name in self._mjcf_names.items()
        }

    def render(self, data: MjData) -> dict[str, np.ndarray]:
        frames = {}
        for name, mjcf_name in self._mjcf_names.items():
            self._renderer.update_scene(data, camera=mjcf_name, scene_option=self._scene_option)
            frames[name] = self._renderer.render()
        return frames


# =============================================================================
# The scene
# =============================================================================


class StretchKitchen:
    """The house, the robot in it, the receptacle on the counter, and the cameras.

    Built once and reset per rollout. Keeping the compiled model rather than
    rebuilding it is not just speed: `FrankaOnStretchView` holds an IK scratch
    `MjData` over this very model, so a rebuild would need the proxy rebuilt with
    it.
    """

    def __init__(self, fisheye: bool) -> None:
        houses = get_procthor_10k_houses(split=HOUSE_SPLIT)
        house_xml_path = houses[HOUSE_SPLIT][HOUSE_INDEX]["base"]
        install_scene_with_objects_and_grasps_from_path(house_xml_path)

        spec = MjSpec.from_file(house_xml_path)
        self.robot_config = Stretch4RobotConfig()
        self.namespace = self.robot_config.robot_namespace

        Stretch4Robot.add_robot_to_scene(
            self.robot_config,
            spec,
            prefix=self.namespace,
            pos=list(STRETCH_BASE_XY),
            quat=R.from_euler("z", STRETCH_BASE_YAW_DEG, degrees=True).as_quat(scalar_first=True),
        )
        Stretch4Robot.apply_control_overrides(spec, self.robot_config)

        if fisheye:
            cameras = {
                DROID_EXO_CAMERA_KEY: StretchCameras.cam_nav_rgb_se4_right,
                DROID_WRIST_CAMERA_KEY: StretchCameras.cam_gripper_se4_right_rgb,
            }
            configure_stretch_cameras(spec, self.namespace, cameras)
        else:
            add_reconstructed_exo_camera(spec, self.namespace)
            spec.camera(self.namespace + "gripper_camera_right_rgb").resolution = [640, 360]

        bowl_spec = MjSpec.from_file(str(install_uid(RECEPTACLE_UID)))
        receptacle_frame = spec.worldbody.add_frame(
            pos=list(RECEPTACLE_POS),
            quat=R.from_euler("x", 90, degrees=True).as_quat(scalar_first=True),
        )
        receptacle_frame.attach_body(bowl_spec.worldbody.first_body(), prefix="place_receptacle/")

        self.model: MjModel = spec.compile()
        self.data = MjData(self.model)
        self.view = Stretch4RobotView(self.data, self.namespace)

        # `init_qpos["base"]` is the origin, which would drag the robot out of the
        # kitchen: the base slide/hinge joints carry `ref` = the spawn pose, so
        # holding position means holding the spawn values, not zero.
        self.init_qpos = dict(self.robot_config.init_qpos)
        self.init_qpos["base"] = [
            STRETCH_BASE_XY[0],
            STRETCH_BASE_XY[1],
            math.radians(STRETCH_BASE_YAW_DEG),
        ]

        if fisheye:
            self.camera_rig: Any = StretchCameraRig(self.model, self.namespace, cameras)
        else:
            self.camera_rig = PinholeCameraRig(
                self.model,
                {
                    DROID_EXO_CAMERA_KEY: self.namespace + DROID_EXO_CAMERA_KEY,
                    DROID_WRIST_CAMERA_KEY: self.namespace + "gripper_camera_right_rgb",
                },
            )

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.view.set_qpos_dict(self.init_qpos)
        # Before the base group can report a hold-still command it has to know
        # where it is, and `mj_resetData` leaves every site pose zeroed until the
        # first forward pass.
        mujoco.mj_forward(self.model, self.data)
        for mg_id in self.view.move_group_ids():
            move_group = self.view.get_move_group(mg_id)
            move_group.ctrl = move_group.noop_ctrl
        mujoco.mj_forward(self.model, self.data)

    def render(self) -> dict[str, np.ndarray]:
        return self.camera_rig.render(self.data)


def load_droid_policy(checkpoint: str | None) -> Any:
    """MolmoBot's `RealRobotVLAPolicy`, driving the released DROID checkpoint.

    Constructed directly rather than through `StretchMolmoBotDroidPolicy`,
    because that class is the *MolmoSpaces* adapter: it expects an experiment
    config, a task and an evaluation pipeline to apply its actions, none of which
    exist here. What is shared with it is everything that decides how the model
    behaves -- the import search, the checkpoint resolution, the action spec, the
    camera keys and the field values -- so the two paths cannot drift into
    driving the same checkpoint differently.
    """
    module = StretchMolmoBotDroidPolicy._import_molmobot()
    policy_config = StretchMolmoBotDroidPolicyConfig(checkpoint_path=checkpoint)
    checkpoint_path = StretchMolmoBotDroidPolicy._resolve_checkpoint(policy_config)

    inner_config = module.RealRobotVLAPolicyConfig(
        checkpoint_path=checkpoint_path,
        # Required by this repository's `molmo_spaces` on every policy config,
        # and not declared by MolmoBot's class; see the same fields in
        # `molmobot_droid_policy._build_inner_policy`. Never used here either.
        policy_cls=module.RealRobotVLAPolicy,
        policy_factory=make_lenient(module.RealRobotVLAPolicy),
        camera_names=[DROID_EXO_CAMERA_KEY, DROID_WRIST_CAMERA_KEY],
        action_move_group_names=list(DROID_ACTION_SPEC),
        action_spec=dict(DROID_ACTION_SPEC),
        action_type=policy_config.action_type,
        action_keys={"arm": policy_config.action_type, "gripper": "joint_pos"},
        action_horizon=policy_config.action_horizon,
        execute_horizon=policy_config.execute_horizon,
        relative_max_joint_delta=[policy_config.max_relative_arm_delta] * DROID_ACTION_SPEC["arm"],
    )

    # `RealRobotVLAPolicy` reads `config.policy_config.*` throughout and nothing
    # else off the experiment config, so this stands in for one. The same shim
    # MolmoBot's own demo notebook uses.
    class _ExpConfigShim:
        def __init__(self, policy_config: Any) -> None:
            self.policy_config = policy_config

    return module.RealRobotVLAPolicy(config=_ExpConfigShim(inner_config), task_type="manipulation")


@click.command()
@click.option(
    "--task",
    default="Pick up the bowl",
    help="The instruction handed to the policy, verbatim. Also names the output file.",
)
@click.option(
    "--checkpoint",
    default=None,
    help="A local DROID checkpoint. Defaults to fetching the released one from the Hub.",
)
@click.option("--episode-dur", type=float, default=20.0, help="Episode length in seconds.")
@click.option(
    "--fisheye/--pinhole",
    default=False,
    help="Give the policy Stretch's real 123-degree fisheye head camera, or the "
    "upright pinhole mounted at the same place. See this module's docstring.",
)
@click.option(
    "--include-base/--no-include-base",
    default=True,
    help="Let the holonomic base join the IK. Off scores what the arm alone can reach.",
)
@click.option(
    "--z-offset",
    type=float,
    default=None,
    help="Metres to raise every retargeted target by. Defaults to half the shortfall "
    "measured on this robot in this scene; see `FrankaOnStretchView`.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("eval_output") / "droid_on_stretch",
    help="Where to write the MP4 and the tool-frame image.",
)
@click.option("--fps", type=int, default=15, help="Frame rate of the written MP4.")
@click.option(
    "--tool-frame-image/--no-tool-frame-image",
    default=False,
    help="Also write a close-up of the gripper at the episode's start pose, with the "
    "pose the retargeting asked for and the one Stretch reached drawn in. The "
    "picture that answers 'is the target being remapped onto the right link'.",
)
def main(
    task: str,
    checkpoint: str | None,
    episode_dur: float,
    fisheye: bool,
    include_base: bool,
    z_offset: float | None,
    output_dir: Path,
    fps: int,
    tool_frame_image: bool,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    from PIL import Image
    from tqdm import tqdm

    kitchen = StretchKitchen(fisheye=fisheye)
    kitchen.reset()
    click.echo(f"move groups: {kitchen.view.move_group_ids()}")
    for name, description in kitchen.camera_rig.describe().items():
        click.echo(f"{name}: {description}")

    # The virtual Franka stands on its pedestal at Stretch's own feet, facing the
    # way Stretch faces -- which is the frame the policy's actions are read in.
    proxy = fr.FrankaOnStretchView(
        kitchen.view,
        kitchen.namespace,
        fr.franka_mount_pose_from_base(kitchen.view.get_move_group("base").joint_pos),
        include_base=include_base,
    )
    # Measured from where the robot is standing after the reset, and without
    # moving it, so this has to come before the snap.
    proxy.target_z_offset = (
        proxy.measure_tool_height_offset() * 0.5 if z_offset is None else z_offset
    )
    click.echo(f"target z offset: {proxy.target_z_offset:+.4f} m")

    policy = load_droid_policy(checkpoint)

    kitchen.reset()
    snap_residual = proxy.snap_to_franka_joint_pos()
    proxy.reset()
    policy.reset()
    click.echo(f"snap residual (dx dy dz | drx dry drz): {np.round(snap_residual, 4).tolist()}")
    click.echo(f"reported Franka arm state: {proxy.get_move_group('arm').joint_pos.round(3)}")
    click.echo(f"Franka home, for comparison: {proxy.franka.init_qpos.round(3)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    start_tool_pose = proxy.arm_ik.tool_pose().copy()
    start_target_pose = proxy.franka_tool_pose_to_world(proxy.franka.fk(proxy.franka.init_qpos))

    frames = []
    residuals = []
    steps = round(episode_dur * 1000 / POLICY_DT_MS)
    for _ in tqdm(range(steps), desc=task):
        obs = {
            "task": task,
            "qpos": {
                "arm": proxy.get_move_group("arm").joint_pos,
                "gripper": proxy.get_move_group("gripper").joint_pos,
            },
            **kitchen.render(),
        }
        frames.append(fr.side_by_side(obs[DROID_EXO_CAMERA_KEY], obs[DROID_WRIST_CAMERA_KEY]))

        action = policy.get_action(obs)
        for mg_id in action:
            proxy.get_move_group(mg_id).ctrl = action[mg_id]
        residuals.append((proxy.last_position_error, proxy.last_orientation_error))

        mujoco.mj_step(
            kitchen.model,
            kitchen.data,
            nstep=POLICY_DT_MS // round(kitchen.model.opt.timestep * 1000),
        )

    # How far the commanded Franka tool poses fell outside what Stretch can
    # reach. A few centimetres is the retargeting working; a persistent 10cm+
    # means the policy is asking for somewhere this robot cannot go from where it
    # is standing.
    position_error, orientation_error = np.array(residuals).T
    click.echo(
        f"tool position residual: mean {position_error.mean():.3f}m, "
        f"max {position_error.max():.3f}m"
    )
    click.echo(
        f"tool orientation residual: mean {orientation_error.mean():.3f}rad, "
        f"max {orientation_error.max():.3f}rad"
    )
    click.echo(f"base ended at: {kitchen.view.get_move_group('base').joint_pos.round(3)}")

    video_path = output_dir / (task.replace(" ", "_") + ".mp4")
    ImageSequenceClip(frames, fps=fps).write_videofile(str(video_path), audio=False)
    click.secho(f"Wrote {video_path}", fg="green")

    if tool_frame_image:
        image_path = output_dir / "tool_frames.png"
        # Both markers in one image, from one viewpoint, close in: at a couple of
        # metres the balls are a few pixels and the picture proves nothing. Green
        # is the pose the retargeting asked for, orange is where Stretch's grasp
        # centre actually went; the disk carries the target's height across, so
        # whichever ball hangs below it is the lower one.
        image = fr.render_tool_frame_view(
            kitchen.model,
            kitchen.data,
            start_tool_pose[:3, 3],
            markers=[
                (start_target_pose, fr.FRANKA_TOOL_COLOR, "franka target"),
                (start_tool_pose, fr.STRETCH_TOOL_COLOR, "stretch grasp_center"),
            ],
            reference_height=float(start_target_pose[2, 3]),
            distance=0.7,
        )
        Image.fromarray(image).save(image_path)
        height_gap = float(start_target_pose[2, 3] - start_tool_pose[2, 3])
        click.echo(
            f"height gap at the start pose: {height_gap:+.4f} m "
            "(positive means Stretch's tool is lower than the target)"
        )
        click.secho(f"Wrote {image_path}", fg="green")


if __name__ == "__main__":
    main()
