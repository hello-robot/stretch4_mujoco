"""
What a gripper command should mean on Stretch, measured against the Robotiq it means something on.

The policy emits one number for its hand: a Robotiq 2F-85 command in 0..255.
`franka_retarget.retarget_robotiq_ctrl` turns that into two Stretch finger
angles by linear interpolation between "shut" and an "open" angle that
`match_robotiq_aperture` picks so the two hands are the same width apart at the
fingertips. That mapping has one free number in it -- `ROBOTIQ_MAX_APERTURE_M` --
and it was chosen by matching *tip separation*, which is the wrong invariant:

    tips 87mm apart  ->  the jaw is 74mm wide at the grasp point, 28mm four
                         centimetres in, and shut at six
    tips 120mm apart ->  110mm, 55mm, 15mm
    tips 188mm apart ->  189mm, 123mm, 65mm

Stretch's fingers curve inwards behind their tips and meet about 7cm back, so
the width available to an object depends on how deep into the jaw the object
sits -- which is exactly what `grasp_offset_m` decides. The Robotiq's pads are
nearly parallel over their working range and its grasp site is between them, so
*its* object sits at one depth and one width.

So the right question is not "how wide do the tips open" but **"how wide is the
jaw where the object will be"**, and this module answers it by putting both
robots in one scene and measuring:

    for each Robotiq command
      -> settle the Robotiq's linkage and measure its clear width at its grasp site
      -> find the Stretch finger angle whose clear width at `grasp_offset_m`
         past its grasp centre is the same
      -> report the tip separation at that angle, which is the number
         `ROBOTIQ_MAX_APERTURE_M` wants to be

    python -m examples.machine_learning.molmospaces.retargetting.aperture

`--overlay` additionally drives Stretch's gripper onto the Franka's grasp site
through the retargeting and renders the two hands on top of each other, at a few
commands, so the match can be looked at rather than taken on trust. Both robots
are in one compiled scene for the same reason: two numbers measured in two
scenes agree by construction, and two hands rendered in one scene do not.

The measurement is a ray cast across the jaw rather than a finger-to-finger
distance, because the fingers are meshes and their inward curve is not in any
single number the model exposes. See `jaw_width`.
"""

from __future__ import annotations

import logging
import math
import os
import sys

import click

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from mujoco import MjData, MjSpec  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from examples.machine_learning.molmospaces.policies import franka_retarget as fr  # noqa: E402
from examples.machine_learning.molmospaces.retargetting import mini_benchmark  # noqa: E402
from examples.machine_learning.molmospaces.stretch.config import (  # noqa: E402
    Stretch4RobotConfig,
)
from examples.machine_learning.molmospaces.stretch.robot import Stretch4Robot  # noqa: E402
from examples.machine_learning.molmospaces.stretch.robot_view import (  # noqa: E402
    Stretch4RobotView,
)

log = logging.getLogger(__name__)

STRETCH_NAMESPACE = "stretch_0/"
FRANKA_NAMESPACE = "franka_0/"
"""Separate namespaces, because one scene holds both and `robot_0/` is what each picks alone."""

ROBOTIQ_SETTLE_STEPS = 300
"""
How many physics steps the Robotiq needs to reach a commanded opening.

The 2F-85 is a linkage -- hinges held together by equality constraints and a
tendon -- so `mj_forward` on a written driver angle does not move the pads;
only stepping does. 300 at 2ms is what `tests/test_retargeting.py` measured as
enough, and this is the same recipe: settle from the model's own rest
configuration in scratch data, with the arm held in free space so nothing sags
into the counter and adds contacts to what should be a free-space motion.
"""

RAY_MAX_REACH_M = 0.25
"""
How far a jaw-crossing ray may travel before it counts as having missed the hand.

A ray fired across an open jaw that clears the fingers carries on until it hits
the kitchen, and a "width" of two metres is a miss reported as a measurement.
A quarter of a metre is comfortably wider than either hand opens and comfortably
narrower than anything else in the room.
"""

PROFILE_DEPTHS_MM = (0, -10, -20, -30, -40, -50, -60)
"""Depths past the grasp frame to report a jaw profile at, in millimetres."""


def build_franka_only():
    """The kitchen with just a Franka on its pedestal, for measuring the Robotiq.

    Measured on its own model rather than in the two-robot scene, and the
    difference is not cosmetic: the 2F-85's pads are held parallel by equality
    constraints and a tendon, and in the combined scene -- where the other robot
    is being stepped with its actuators at zero and collapsing -- that system
    rings instead of settling. Measured, at Robotiq 128: 86mm and still opening
    after 4000 steps in the combined scene, against 46.1mm after 300 here. Both
    hands are still put in one scene, by `build_overlay_scene`, for the picture;
    the numbers come from each hand alone.

    Returns `(model, data, view, namespace, config)`.
    """
    from molmo_spaces.configs.robot_configs import FrankaRobotConfig
    from molmo_spaces.robots.franka import FrankaRobot
    from molmo_spaces.robots.robot_views.franka_droid_view import FrankaDroidRobotView
    from molmo_spaces.utils.lazy_loading_utils import (
        install_scene_with_objects_and_grasps_from_path,
    )

    from examples.machine_learning.molmospaces.retargetting.setups import FRANKA_LINK0_HEIGHT

    scene = mini_benchmark.house_scene_path()
    install_scene_with_objects_and_grasps_from_path(str(scene))
    spec = MjSpec.from_file(str(scene))

    base_xy = mini_benchmark.ROBOT_BASE_XY
    target_xy = mini_benchmark.TARGET_XY
    yaw = math.atan2(target_xy[1] - base_xy[1], target_xy[0] - base_xy[0])
    mount = fr.franka_mount_pose_from_base([base_xy[0], base_xy[1], yaw])

    config = FrankaRobotConfig()
    pedestal = float(config.base_size[2]) if config.base_size else 0.0
    FrankaRobot.add_robot_to_scene(
        config,
        spec,
        prefix=config.robot_namespace,
        pos=[float(mount[0, 3]), float(mount[1, 3]), FRANKA_LINK0_HEIGHT - pedestal],
        quat=list(R.from_matrix(mount[:3, :3]).as_quat(scalar_first=True)),
    )
    model = spec.compile()
    model.opt.timestep = 0.002
    data = MjData(model)
    view = FrankaDroidRobotView(data, config.robot_namespace)
    for group, values in config.init_qpos.items():
        if group in view.move_group_ids():
            view.get_move_group(group).joint_pos = values
    mujoco.mj_forward(model, data)
    return model, data, view, config.robot_namespace, config


def build_overlay_scene():
    """The mini benchmark's kitchen with a Stretch *and* a Franka standing in it.

    Both at the pose the study puts them at -- the Franka on its pedestal where
    `franka_mount_pose_from_base` imagines it, Stretch on the floor at the same
    xy. For the picture only; see `build_franka_only` for why the numbers are
    not taken here.

    Returns `(model, data, stretch_view, franka_view, franka_config)`.
    """
    from molmo_spaces.configs.robot_configs import FrankaRobotConfig
    from molmo_spaces.robots.franka import FrankaRobot
    from molmo_spaces.robots.robot_views.franka_droid_view import FrankaDroidRobotView
    from molmo_spaces.utils.lazy_loading_utils import (
        install_scene_with_objects_and_grasps_from_path,
    )

    from examples.machine_learning.molmospaces.retargetting.setups import FRANKA_LINK0_HEIGHT

    scene = mini_benchmark.house_scene_path()
    install_scene_with_objects_and_grasps_from_path(str(scene))
    spec = MjSpec.from_file(str(scene))

    base_xy = mini_benchmark.ROBOT_BASE_XY
    target_xy = mini_benchmark.TARGET_XY
    yaw = math.atan2(target_xy[1] - base_xy[1], target_xy[0] - base_xy[0])
    quat = list(R.from_euler("z", yaw).as_quat(scalar_first=True))

    stretch_config = Stretch4RobotConfig(robot_namespace=STRETCH_NAMESPACE)
    Stretch4Robot.add_robot_to_scene(
        stretch_config, spec, prefix=STRETCH_NAMESPACE, pos=list(base_xy), quat=quat
    )
    Stretch4Robot.apply_control_overrides(spec, stretch_config)

    franka_config = FrankaRobotConfig(robot_namespace=FRANKA_NAMESPACE)
    pedestal = float(franka_config.base_size[2]) if franka_config.base_size else 0.0
    mount = fr.franka_mount_pose_from_base([base_xy[0], base_xy[1], yaw])
    FrankaRobot.add_robot_to_scene(
        franka_config,
        spec,
        prefix=FRANKA_NAMESPACE,
        pos=[float(mount[0, 3]), float(mount[1, 3]), FRANKA_LINK0_HEIGHT - pedestal],
        quat=list(R.from_matrix(mount[:3, :3]).as_quat(scalar_first=True)),
    )

    model = spec.compile()
    model.opt.timestep = 0.002
    data = MjData(model)
    stretch_view = Stretch4RobotView(data, STRETCH_NAMESPACE)
    franka_view = FrankaDroidRobotView(data, FRANKA_NAMESPACE)

    stretch_view.set_qpos_dict(
        {
            "base": [base_xy[0], base_xy[1], yaw],
            "lift": [1.0],
            "arm": [0.2],
            "wrist": [0.0, 0.0, 0.0],
            "gripper": [0.0, 0.0],
        }
    )
    for group, values in franka_config.init_qpos.items():
        if group in franka_view.move_group_ids():
            franka_view.get_move_group(group).joint_pos = values
    mujoco.mj_forward(model, data)
    return model, data, stretch_view, franka_view, franka_config


def jaw_width(model, data, pose: np.ndarray, approach: int, across: int, depth_m: float) -> float:
    """Clear width across a jaw, `depth_m` along its approach axis from `pose`'s origin.

    Two rays from the jaw's centre line, one each way along the axis the fingers
    separate on; the sum of what they travel before hitting something is the gap
    an object at that depth has to fit into. Returns NaN when either ray escapes
    (see `RAY_MAX_REACH_M`), because "wider than the hand" is not a width.

    `approach` and `across` are column indices into `pose`'s rotation -- the two
    hands do not share a convention (the Robotiq reaches along +z, Stretch along
    +x) and this takes them as given rather than assuming either.
    """
    origin = pose[:3, 3] + pose[:3, approach] * depth_m
    total = 0.0
    for sign in (1.0, -1.0):
        hit = mujoco.mj_ray(
            model, data, origin, sign * pose[:3, across], None, 1, -1, np.zeros(1, dtype=np.int32)
        )
        if hit < 0 or hit > RAY_MAX_REACH_M:
            return float("nan")
        total += hit
    return total


def settle_robotiq(model, data, view, namespace: str, franka_config, command: float) -> None:
    """Drive the Robotiq to `command` and step until its linkage has followed.

    Settled in scratch data from the model's own rest configuration and copied
    back, which is `tests/test_retargeting.py`'s recipe and is load-bearing for
    the same two reasons: the hand is a linkage that only moves under stepping,
    and settling *from wherever it happens to be* is path dependent, so a hand
    already shut stays shut when told to open half way. The arm is held at the
    Franka's free-space home while the hand settles, so nothing sags into the
    counter and adds contacts to what should be a free-space motion.
    """
    gripper_qposadr = np.array(
        [
            model.jnt_qposadr[joint]
            for joint in range(model.njnt)
            if model.joint(joint).name.startswith(namespace)
            and "gripper" in model.joint(joint).name
        ]
    )
    scratch = MjData(model)
    scratch.qpos[:] = data.qpos
    scratch.qpos[gripper_qposadr] = model.qpos0[gripper_qposadr]
    scratch.qvel[:] = 0.0
    scratch.ctrl[:] = 0.0

    scratch_view = type(view)(scratch, namespace)
    home = list(franka_config.init_qpos["arm"])
    scratch_view.get_move_group("arm").joint_pos = home
    scratch_view.get_move_group("arm").ctrl = home
    scratch_view.get_move_group("gripper").ctrl = [float(command)]
    for _ in range(ROBOTIQ_SETTLE_STEPS):
        mujoco.mj_step(model, scratch)

    data.qpos[gripper_qposadr] = scratch.qpos[gripper_qposadr]
    mujoco.mj_forward(model, data)


def stretch_width_at(model, data, stretch_view, angle: float, depth_m: float) -> float:
    """Stretch's clear jaw width `depth_m` past its grasp centre, at finger `angle`."""
    stretch_view.get_move_group("gripper").joint_pos = [angle, angle]
    mujoco.mj_forward(model, data)
    pose = np.asarray(stretch_view.get_move_group("wrist").leaf_frame_to_world, dtype=float)
    return jaw_width(model, data, pose, 0, 1, depth_m)


def stretch_angle_for_width(
    model, data, stretch_view, width_m: float, depth_m: float, steps: int = 40
) -> float:
    """The finger angle at which Stretch's jaw is `width_m` wide at `depth_m`.

    A bisection, because the relation is monotone in the angle at any fixed depth
    and the model is the only thing that knows it -- the same argument, and the
    same method, as `franka_retarget.stretch_finger_for_aperture`, which this
    replaces for the one question that matters. Returns `STRETCH_FINGER_OPEN`
    when the hand cannot open that wide at that depth, rather than a midpoint
    that would read as a match.
    """
    low, high = fr.STRETCH_FINGER_CLOSED, fr.STRETCH_FINGER_OPEN
    widest = stretch_width_at(model, data, stretch_view, high, depth_m)
    if not np.isfinite(widest) or widest <= width_m:
        return float(high)
    for _ in range(steps):
        middle = 0.5 * (low + high)
        here = stretch_width_at(model, data, stretch_view, middle, depth_m)
        if not np.isfinite(here) or here < width_m:
            low = middle
        else:
            high = middle
    return float(0.5 * (low + high))


def tip_separation(model, data, stretch_view, angle: float) -> float:
    """Stretch's fingertip separation at `angle` -- the units `ROBOTIQ_MAX_APERTURE_M` is in."""
    stretch_view.get_move_group("gripper").joint_pos = [angle, angle]
    mujoco.mj_forward(model, data)
    return float(stretch_view.get_move_group("gripper").inter_finger_dist)


def calibrate(grasp_offset_m: float, commands=(0.0, 64.0, 128.0, 192.0, 255.0)) -> list[dict]:
    """The Robotiq's jaw width at each command, and the Stretch hand that matches it.

    `grasp_offset_m` is where the object ends up in Stretch's jaw, and the whole
    point of the exercise is that the answer depends on it -- so it is an
    argument rather than a constant. The object sits `grasp_offset_m` *behind*
    the grasp centre, because the retargeting drives the grasp centre that far
    past it, which is why the depth handed to the ray cast is negative. See
    `RetargetParams.grasp_offset_m`.
    """
    from examples.machine_learning.molmospaces.retargetting import diagnose

    franka_model, franka_data, franka_view, namespace, franka_config = build_franka_only()
    stretch_model, stretch_data, stretch_view, _ = diagnose.build_standing_robot()
    depth = -abs(float(grasp_offset_m))

    rows = []
    for command in commands:
        settle_robotiq(franka_model, franka_data, franka_view, namespace, franka_config, command)
        franka_pose = np.asarray(
            franka_view.get_move_group("arm").leaf_frame_to_world, dtype=float
        )
        # The Robotiq reaches along +z and separates along +-y; its object sits at
        # its grasp site, which is depth 0. Its pads are held parallel by the
        # linkage, so the profile below is nearly flat -- which is the contrast
        # the whole module is about.
        robotiq = jaw_width(franka_model, franka_data, franka_pose, 2, 1, 0.0)
        profile = [
            jaw_width(franka_model, franka_data, franka_pose, 2, 1, d / 1000.0)
            for d in PROFILE_DEPTHS_MM
        ]
        angle = (
            stretch_angle_for_width(stretch_model, stretch_data, stretch_view, robotiq, depth)
            if np.isfinite(robotiq)
            else float("nan")
        )
        matched = stretch_width_at(stretch_model, stretch_data, stretch_view, angle, depth)
        rows.append(
            {
                "command": command,
                "robotiq_width_m": robotiq,
                "robotiq_profile_m": profile,
                "stretch_angle_rad": angle,
                "stretch_width_m": matched,
                "stretch_tips_m": tip_separation(
                    stretch_model, stretch_data, stretch_view, angle
                ),
                "reachable": bool(np.isfinite(matched) and matched >= robotiq - 5e-4),
            }
        )
    return rows


def report(rows: list[dict], grasp_offset_m: float) -> None:
    """Print the calibration, and the one number it is really about."""
    click.secho(
        f"\n== what a gripper command is worth on each hand, "
        f"with the object {grasp_offset_m * 1000:.0f}mm into Stretch's jaw ==",
        bold=True,
    )
    click.echo(
        "\n  cmd | robotiq clear width | stretch finger | stretch clear width | stretch tips"
    )
    for row in rows:
        click.echo(
            f"  {row['command']:4.0f} | {row['robotiq_width_m'] * 1000:15.1f} mm |"
            f" {row['stretch_angle_rad']:11.4f} rad |"
            f" {row['stretch_width_m'] * 1000:15.1f} mm |"
            f" {row['stretch_tips_m'] * 1000:8.1f} mm"
            + ("" if row["reachable"] else "   <- Stretch cannot open this wide here")
        )

    click.echo("\n  the Robotiq's own jaw, by depth past its grasp site:")
    click.echo("       cmd |" + "".join(f"{d:7d}mm" for d in PROFILE_DEPTHS_MM))
    for row in rows:
        click.echo(
            f"      {row['command']:4.0f} |"
            + "".join(f"{w * 1000:7.1f}  " for w in row["robotiq_profile_m"])
        )

    wide = next((row for row in rows if row["command"] == 0.0), None)
    if wide is None or not np.isfinite(wide["stretch_tips_m"]):
        return
    click.secho(
        f"\n  ROBOTIQ_MAX_APERTURE_M = {wide['stretch_tips_m']:.4f}"
        f" for grasp_offset_m={grasp_offset_m:.3f}   (it is {fr.ROBOTIQ_MAX_APERTURE_M:.4f} now)",
        fg="green" if wide["reachable"] else "yellow",
    )
    click.echo(
        "  That is the tip separation at which Stretch's jaw is as wide, where the object\n"
        "  actually sits, as the Robotiq's is when the policy says 'open'. Re-run with a\n"
        "  different --grasp-offset and it moves, which is the finding: the aperture is not\n"
        "  a property of the hand alone, it is a property of the hand and the grasp depth."
    )
    if not wide["reachable"]:
        click.secho(
            f"  At this depth Stretch cannot match it at all -- wide open it manages\n"
            f"  {wide['stretch_width_m'] * 1000:.0f}mm against the Robotiq's"
            f" {wide['robotiq_width_m'] * 1000:.0f}mm, so the number above is just the hand's\n"
            f"  limit. Either the object goes shallower into the jaw or it does not fit.",
            fg="yellow",
        )


# =============================================================================
# Looking at it
# =============================================================================

OVERLAY_SIZE = (1280, 960)


def render_overlay(rows: list[dict], grasp_offset_m: float, output_dir) -> list:
    """Render the two hands on top of each other, one PNG per command.

    Stretch is driven onto the Franka's grasp site through the retargeting the
    policy uses -- `FrankaOnStretchView`, with the study's tool correction -- so
    what the picture shows is the alignment a rollout actually gets, not a
    hand-placed one. The camera looks along the jaw line, because that is the
    axis the question is about: two hands that agree about the aperture have
    their four pads in two pairs at the same two distances from the centre.
    """
    import cv2

    from examples.machine_learning.molmospaces.retargetting.setups import apply_tool_correction

    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, data, stretch_view, franka_view, franka_config = build_overlay_scene()
    namespace = FRANKA_NAMESPACE
    base = np.asarray(stretch_view.get_move_group("base").joint_pos, dtype=float)
    proxy = fr.FrankaOnStretchView(
        stretch_view, STRETCH_NAMESPACE, fr.franka_mount_pose_from_base(base), include_base=False
    )
    apply_tool_correction(proxy, wrist_tilt_deg=0.0, grasp_offset_m=grasp_offset_m)

    # The model's own offscreen buffer is whatever the scene declared, and a
    # `Renderer` larger than it fails at construction rather than resizing.
    from examples.machine_learning.molmospaces.retargetting.replay import (
        _ensure_offscreen_buffer,
    )

    _ensure_offscreen_buffer(model, OVERLAY_SIZE)
    renderer = mujoco.Renderer(model, height=OVERLAY_SIZE[1], width=OVERLAY_SIZE[0])
    camera = mujoco.MjvCamera()
    camera.distance = 0.42
    camera.elevation = -12.0
    written = []
    try:
        for row in rows:
            command = row["command"]
            settle_robotiq(model, data, franka_view, namespace, franka_config, command)

            # Stretch onto the Franka's own grasp site, through the retargeting.
            targets = proxy.retarget_franka_joint_pos(franka_config.init_qpos["arm"])
            for group, value in targets.items():
                stretch_view.get_move_group(group).joint_pos = value
            stretch_view.get_move_group("gripper").joint_pos = [
                row["stretch_angle_rad"],
                row["stretch_angle_rad"],
            ]
            mujoco.mj_forward(model, data)

            pose = np.asarray(franka_view.get_move_group("arm").leaf_frame_to_world, dtype=float)
            camera.lookat[:] = pose[:3, 3]
            # Down the jaw line, so the two hands' pads separate left and right.
            camera.azimuth = float(np.degrees(np.arctan2(pose[1, 1], pose[0, 1])) + 90.0)
            renderer.update_scene(data, camera)
            frame = renderer.render()

            label = (
                f"robotiq {command:.0f}  |  robotiq {row['robotiq_width_m'] * 1000:.0f}mm"
                f"  stretch {row['stretch_width_m'] * 1000:.0f}mm"
                f"  tips {row['stretch_tips_m'] * 1000:.0f}mm"
            )
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(
                frame, label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            path = output_dir / f"aperture_robotiq_{int(command):03d}.png"
            cv2.imwrite(str(path), frame)
            written.append(path)
    finally:
        renderer.close()
    return written


def sweep(offsets, commands=(0.0, 64.0, 128.0, 192.0, 255.0)) -> dict[float, list[dict]]:
    """`calibrate` at several grasp depths, on one pair of compiled scenes.

    Compiling the two robots is nearly all of the cost and nothing about either
    depends on the depth, so a sweep builds them once. The table it produces is
    the one worth reading: it says how deep into Stretch's jaw an object can go
    before the hand simply cannot open around what the Robotiq could.
    """
    return {float(offset): calibrate(float(offset), commands) for offset in offsets}


def report_sweep(by_offset: dict[float, list[dict]]) -> None:
    """One row per grasp depth: the aperture that matches, and whether it exists."""
    click.secho("\n== the matched aperture, against how deep the object sits ==", bold=True)
    click.echo("\n  grasp_offset | tips to match 'open' | widest Stretch manages | robotiq")
    for offset, rows in sorted(by_offset.items()):
        wide = next(row for row in rows if row["command"] == 0.0)
        verdict = "" if wide["reachable"] else "   <- out of reach, this is just the hand's limit"
        click.echo(
            f"  {offset * 1000:9.0f} mm | {wide['stretch_tips_m'] * 1000:16.1f} mm |"
            f" {wide['stretch_width_m'] * 1000:18.1f} mm |"
            f" {wide['robotiq_width_m'] * 1000:6.1f} mm{verdict}"
        )
    click.echo(
        "\n  'tips to match open' is what ROBOTIQ_MAX_APERTURE_M should be at that depth.\n"
        "  Where it is out of reach the hand cannot be opened far enough for the object the\n"
        "  Robotiq would have swallowed, and the only remedy is a shallower grasp_offset_m."
    )


@click.command()
@click.option(
    "--grasp-offset",
    "grasp_offsets",
    type=float,
    multiple=True,
    help="Where the object sits in Stretch's jaw, in metres past its grasp centre. "
    "Repeatable, to sweep. Defaults to the Stretch setups' own STRETCH_GRASP_OFFSET_M. "
    "The matched aperture depends on it, which is the whole finding.",
)
@click.option(
    "--overlay",
    is_flag=True,
    help="Also render the two hands on top of each other, one PNG per command.",
)
@click.option(
    "--output-dir",
    default="eval_output/aperture",
    help="Where --overlay writes its PNGs.",
)
def main(grasp_offsets: tuple[float, ...], overlay: bool, output_dir: str) -> None:
    """Measure what a gripper command should mean on Stretch. See the module docstring."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    from examples.machine_learning.molmospaces.retargetting.setups import STRETCH_GRASP_OFFSET_M

    offsets = list(grasp_offsets) or [STRETCH_GRASP_OFFSET_M]
    click.secho("Measuring both hands, no policy involved.", bold=True)
    by_offset = sweep(offsets)
    for offset in offsets:
        report(by_offset[float(offset)], float(offset))
    if len(offsets) > 1:
        report_sweep(by_offset)
    if overlay:
        deepest = float(offsets[-1])
        written = render_overlay(by_offset[deepest], deepest, output_dir)
        click.secho(f"\nWrote {len(written)} overlay frames to {output_dir}", fg="green")


if __name__ == "__main__":
    main()
