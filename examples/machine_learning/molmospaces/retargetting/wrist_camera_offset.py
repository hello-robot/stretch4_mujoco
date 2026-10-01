"""
Where each hand's wrist camera sits, and the `tool_offset_x_m` / `tool_offset_y_m` that line them up.

The policy aims through the wrist view, and the two robots do not carry that view
in the same place: relative to the point each gripper closes on, the Franka's
`gripper/wrist_camera` sits nearer the fingers and further to one side than
Stretch's gripper cameras do. This measures the difference, on the compiled
models with the retargeted tool frames coincident, and prints the tool offsets
that put Stretch's camera where the Franka's is.

    python -m examples.machine_learning.molmospaces.retargetting.wrist_camera_offset
    python -m examples.machine_learning.molmospaces.retargetting.wrist_camera_offset --parallel_gripper

Everything is in Stretch's tool axes, from each hand's own grasp point: x along
the approach, y along the jaw line, z across the hand towards the camera's side.
The offset is the Franka camera minus Stretch's. Only x and y are offered as
knobs; z is printed so the gap they leave is visible.

The offsets are then applied through `setups.apply_tool_correction`, which is
exactly what `run_on_real_stretch.py --tool-offset-x-m/-y-m` and
`params_search_side_by_side.py --stretch4-tool-offset-x/-y` do, and the two
cameras measured again -- what is left afterwards should be the z term alone,
with the IK still tracking.

What this does not say is that these are the offsets to *run* with. The cameras
are also aimed differently (the Franka's is pitched in towards the jaw axis,
Stretch's looks straight down the approach), and the policy re-closes its loop
through the view, so how much of the camera offset shows up as a grasp error is
a rollout measurement. See `cameras.RetargetParams.tool_offset_x_m`.
"""

from __future__ import annotations

import logging
import os
import sys

import click

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from examples.machine_learning.molmospaces.policies import franka_retarget as fr  # noqa: E402
from examples.machine_learning.molmospaces.retargetting import aperture  # noqa: E402
from examples.machine_learning.molmospaces.retargetting import (  # noqa: E402
    grasp_center_alignment as alignment,
)
from examples.machine_learning.molmospaces.stretch.config import (  # noqa: E402
    PG4_TOOL_NAME,
    SG4_TOOL_NAME,
    publish_stretch4_tool,
)

log = logging.getLogger(__name__)

STRETCH_WRIST_CAMERAS = ("gripper_camera_right_rgb", "gripper_camera_left_rgb")
"""Stretch's two gripper cameras, right first: the side the policy reads by default."""

IK_TRACKING_MM = 1.0
"""An IK residual above this means the lift saturated and the pose is not the one asked for."""


def _camera_in_tool(model, data, tool_pose: np.ndarray, camera: str) -> tuple[np.ndarray, np.ndarray]:
    """A camera's position and optical axis, in `tool_pose`'s axes, from its origin."""
    camera_id = model.camera(camera).id
    rotation = tool_pose[:3, :3]
    position = rotation.T @ (data.cam_xpos[camera_id] - tool_pose[:3, 3])
    # MuJoCo cameras look down their own -z.
    looks = rotation.T @ -data.cam_xmat[camera_id].reshape(3, 3)[:, 2]
    return position, looks


def measure(drop_m: float = alignment.FRANKA_DROP_M) -> dict:
    """Both hands' wrist cameras in their own tool frames, and the offsets between them.

    One compiled scene with a Stretch and a Franka in it, the Franka commanded to
    its home tool pose lowered `drop_m` (under Stretch's lift ceiling -- see
    `grasp_center_alignment`) and Stretch retargeted onto it with the bare tool
    correction, so the two tool frames coincide.
    """
    from examples.machine_learning.molmospaces.retargetting.setups import (
        FRANKA_WRIST_MJCF,
        apply_tool_correction,
    )

    model, data, stretch_view, franka_view, franka_config = aperture.build_overlay_scene()
    base = np.asarray(stretch_view.get_move_group("base").joint_pos, dtype=float)
    mount = fr.franka_mount_pose_from_base(base)
    proxy = fr.FrankaOnStretchView(stretch_view, aperture.STRETCH_NAMESPACE, mount)
    command = alignment.reachable_franka_command(proxy, franka_config, mount, drop_m)
    franka_camera = aperture.FRANKA_NAMESPACE + FRANKA_WRIST_MJCF

    def pose_for(tool_offset_x_m: float, tool_offset_y_m: float) -> float:
        """Retarget `command` with these offsets, write both arms, return the IK residual in mm."""
        apply_tool_correction(
            proxy,
            wrist_tilt_deg=0.0,
            grasp_offset_m=0.0,
            tool_offset_x_m=tool_offset_x_m,
            tool_offset_y_m=tool_offset_y_m,
        )
        for group, value in proxy.retarget_franka_joint_pos(command).items():
            stretch_view.get_move_group(group).joint_pos = value
        franka_view.get_move_group("arm").joint_pos = list(command)
        mujoco.mj_forward(model, data)
        return proxy.last_position_error * 1000.0

    def cameras_apart_mm(stretch_camera: str) -> float:
        franka = data.cam_xpos[model.camera(franka_camera).id]
        stretch = data.cam_xpos[model.camera(aperture.STRETCH_NAMESPACE + stretch_camera).id]
        return float(np.linalg.norm(franka - stretch) * 1000.0)

    residual_mm = pose_for(0.0, 0.0)
    # Stretch's own tool frame, and the Franka's grasp site turned into the same
    # axes by the bare correction -- the frame the retargeting says they share.
    stretch_tool = np.asarray(stretch_view.get_move_group("gripper").leaf_frame_to_world, dtype=float)
    franka_tool = (
        np.asarray(franka_view.get_move_group("arm").leaf_frame_to_world, dtype=float)
        @ proxy._tool_correction
    )
    franka_position, franka_looks = _camera_in_tool(model, data, franka_tool, franka_camera)
    stretch = {}
    for camera in STRETCH_WRIST_CAMERAS:
        position, looks = _camera_in_tool(
            model, data, stretch_tool, aperture.STRETCH_NAMESPACE + camera
        )
        stretch[camera] = {
            "position_m": position,
            "looks": looks,
            "offset_m": franka_position - position,
            "apart_before_mm": cameras_apart_mm(camera),
        }

    # The check: apply each camera's offset the way a run would, and measure again.
    for camera, row in stretch.items():
        row["residual_after_mm"] = pose_for(float(row["offset_m"][0]), float(row["offset_m"][1]))
        row["apart_after_mm"] = cameras_apart_mm(camera)

    return {
        "tool": stretch_view.get_move_group("gripper").kind.name,
        "residual_mm": residual_mm,
        "franka": {"name": FRANKA_WRIST_MJCF, "position_m": franka_position, "looks": franka_looks},
        "stretch": stretch,
    }


def _pitch_in_deg(looks: np.ndarray) -> float:
    """How far an optical axis is turned off the approach (+x), in degrees."""
    return float(np.degrees(np.arccos(np.clip(looks[0] / np.linalg.norm(looks), -1.0, 1.0))))


def report(result: dict) -> None:
    """The table, the offsets, and whether applying them lined the cameras up."""
    click.secho(
        f"\n== where each wrist camera sits, {result['tool']} against the Franka ==", bold=True
    )
    click.echo(
        "   (mm from each hand's own grasp point, in Stretch's tool axes:\n"
        "    x along the approach, y along the jaw line, z towards the camera's side)\n"
    )
    if result["residual_mm"] > IK_TRACKING_MM:
        click.secho(
            f"  The IK is {result['residual_mm']:.1f}mm off, so the two tool frames do not\n"
            "  coincide and nothing below is a measurement. Raise --drop until it tracks.",
            fg="red",
        )
        return

    click.echo("   camera                          x mm     y mm     z mm   off the approach")
    rows = [("Franka " + result["franka"]["name"], result["franka"])]
    rows += [(f"{result['tool']} {name}", row) for name, row in result["stretch"].items()]
    for label, row in rows:
        x, y, z = row["position_m"] * 1000.0
        click.echo(f"   {label:30s} {x:+7.1f}  {y:+7.1f}  {z:+7.1f}   {_pitch_in_deg(row['looks']):5.1f} deg")

    click.secho("\n== the tool offsets that put Stretch's camera where the Franka's is ==", bold=True)
    for name, row in result["stretch"].items():
        x, y, z = row["offset_m"]
        left = "left" in name
        click.secho(f"\n  {name}", bold=True)
        click.secho(f"    tool_offset_x_m = {x:+.4f}", fg="green")
        click.secho(f"    tool_offset_y_m = {y:+.4f}", fg="green")
        click.echo(f"    (z, which no knob covers: {z * 1000.0:+.1f}mm)")
        click.echo(
            f"    run_on_real_stretch.py        --tool-offset-x-m {x:.4f} --tool-offset-y-m {y:.4f}"
            # The workstation reads the side off the stream; the robot's sender picks it.
            + ("   (robot streaming --wrist-camera left)" if left else "")
        )
        click.echo(
            f"    params_search_side_by_side.py --stretch4-tool-offset-x {x:.4f} "
            f"--stretch4-tool-offset-y {y:.4f}" + (" --use_left_gripper_camera" if left else "")
        )
        tracked = row["residual_after_mm"] <= IK_TRACKING_MM
        click.secho(
            f"    applied: cameras {row['apart_before_mm']:.1f}mm apart -> "
            f"{row['apart_after_mm']:.1f}mm (the z term is {abs(z) * 1000.0:.1f}mm), "
            f"IK residual {row['residual_after_mm']:.1f}mm",
            fg=None if tracked else "yellow",
        )
        if not tracked:
            click.secho(
                "    the offset pose is past Stretch's reach here, so 'applied' is not a clean check",
                fg="yellow",
            )

    click.echo(
        "\n  These make the two *positions* agree. The cameras are also aimed differently\n"
        "  (the angles above), and the policy re-closes its loop through the view, so\n"
        "  treat them as the starting point for a rollout sweep, not as the answer."
    )


@click.command()
@click.option(
    "--drop",
    type=float,
    default=alignment.FRANKA_DROP_M,
    show_default=True,
    help="Metres to lower the Franka's home tool pose by, to get it under Stretch's lift "
    "ceiling. Too little and the IK saturates and the tool frames do not coincide.",
)
@click.option(
    "--parallel_gripper",
    "parallel_gripper",
    is_flag=True,
    help="Measure the parallel jaw gripper (PG4) instead of the stretch gripper (SG4).",
)
def main(drop: float, parallel_gripper: bool) -> None:
    """Measure the tool offsets that line Stretch's wrist camera up with the Franka's."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    publish_stretch4_tool(PG4_TOOL_NAME if parallel_gripper else SG4_TOOL_NAME)
    report(measure(drop_m=drop))


if __name__ == "__main__":
    main()
