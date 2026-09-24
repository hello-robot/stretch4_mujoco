"""
Where each hand's gripping surface sits, and the `grasp_offset_m` that makes the two coincide.

`STRETCH_GRASP_OFFSET_M` is chosen by replaying episodes and counting picks,
which is the right way to choose it and a poor way to understand it. This is the
geometric half of the same question: with Stretch retargeted onto the Franka in
one scene, how far along the approach does each hand's *gripping surface* sit
from its own grasp frame, and what offset lines the two up?

    python -m examples.machine_learning.molmospaces.retargetting.grasp_center_alignment

Measured on the contact surfaces, not on geom origins
-----------------------------------------------------
`diagnose.report_gripper` takes the furthest-forward finger geom's `geom_xpos`
and reports the grasp centre as sitting 1.5cm past the fingertips. That is the
distance to a mesh's *frame origin*, which for Stretch's fingertip meshes is
nowhere near their front face, and the conclusion it invites -- that an object at
the commanded point is outside the hand -- does not survive measuring the meshes
themselves. This module takes every vertex of the fingertip and pad geoms and
projects it on the hand's own approach axis, which is the surface an object
actually touches. From the Robotiq's grasp site, with the Robotiq open and
Stretch's hand at the 132mm tip separation that matches its 87mm jaw:

    Robotiq pads                         -33.6 ..  +4.0 mm
    Stretch fingertips, grasp_offset 0   -48.8 .. +13.1 mm

Both hands carry their grasp frame near the tip end of their own gripping
surface: the Robotiq 4.0mm behind its pad front, Stretch 13.1mm behind its
fingertip front. The offset that makes those two front edges flush is -0.009 --
not the +0.030 the study ships, and not the +0.106 "to the pads" that the
geom-origin reading implies.

**The front edge is the only landmark the two hands share**, and the reason is
the shape of the jaws. Ray-cast across each at a range of depths from its own
grasp frame:

    depth (mm)          +10    0   -10   -20   -30   -40   -50   -60   -70
    Robotiq pads         --    87    87    87    87    90    68    19    --
    Stretch fingertip   127   124   108   108    87    65    44    23     3

The Robotiq's pads are parallel, so its jaw is one width all the way in and any
point of it means the same thing. Stretch's fingertip is a wedge. So the
*midpoints* of the two spans above are not comparable -- one is the middle of a
uniform pad, the other the middle of a taper -- and this module reports that
figure only to say so.

Nor does "put the object where Stretch's jaw is as wide as the Robotiq's" pick
an offset: `aperture.py` solves for the tip separation that makes that true at
whatever offset it is handed (0.030 -> 132mm tips, 0.015 -> 131mm), so it holds
at every offset and singles out none. Where the object sits and how wide the jaw
is there are two knobs, and `ROBOTIQ_MAX_APERTURE_M` carries the trade.

The lift ceiling, which invalidates the obvious version of this measurement
-------------------------------------------------------------------------
The Franka's *home* tool pose is above Stretch's lift ceiling in this kitchen.
Retarget it and the lift saturates 73mm short, the IK returns the same saturated
answer at every offset, and a sweep reports the two hands as fixed 100mm apart
however far it moves them -- an arm at its end stop, read as a geometry result.
`aperture.render_overlay` poses the pair exactly this way, so its pictures are
worth looking at for the *aperture* and are not an alignment check.

So the command here is the Franka's home pose dropped `--drop` metres, and every
row prints the IK residual next to its measurement: a row whose residual is not
a fraction of a millimetre is the lift talking, not the offset.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from examples.machine_learning.molmospaces.policies import franka_retarget as fr  # noqa: E402
from examples.machine_learning.molmospaces.retargetting import aperture  # noqa: E402

log = logging.getLogger(__name__)

OFFSETS_M = (-0.010, 0.000, 0.0025, 0.015, 0.030, 0.045)
"""The offsets swept by default: either side of the alignment, out to the study's own."""

STRETCH_FINGER_ANGLE_RAD = 0.3466
"""
Stretch's fingers at 132mm tip separation -- the opening `aperture.py` calibrates.

The spans below depend on it, because the fingers splay forward as well as out:
fully open the fingertip surface reaches only +3.8mm past the grasp centre,
at this opening +13.1mm. This is the opening at which Stretch's jaw is as wide
as the Robotiq's where the object sits, so it is the one at which comparing the
two hands' surfaces means anything.
"""

FRANKA_DROP_M = 0.22
"""
How far below its home tool pose to command the Franka, in metres.

Enough to bring the pose under Stretch's lift ceiling -- see the module docstring
-- and not so far that the hands are inside the counter. At 0.22 the retargeting
tracks to a tenth of a millimetre.
"""

RENDER_SIZE = aperture.OVERLAY_SIZE
CAMERA_DISTANCE = 0.34
CAMERA_ELEVATION = -18.0
"""Close enough that the two hands fill the frame, high enough to clear the worktop."""


def reachable_franka_command(proxy, franka_config, mount: np.ndarray, drop_m: float) -> np.ndarray:
    """The Franka's home arm command, lowered `drop_m` and re-solved.

    Lowered in the *world*, then taken back through the mount, so the tool keeps
    its home orientation and only its height changes -- the grasp stays a
    top-down one, which is what the benchmark's objects are picked with.
    """
    home = np.asarray(franka_config.init_qpos["arm"], dtype=float)
    world = mount @ proxy.franka.fk(home)
    world[2, 3] -= float(drop_m)
    return proxy.franka.ik(
        np.linalg.inv(mount) @ world, proxy.franka.default_init_qpos, iterations=300
    )


def surface_span(model, data, pose: np.ndarray, approach_col: int, match) -> tuple[float, float]:
    """(front, back) of the matching geoms' surface, in mm along `pose`'s approach axis.

    Every vertex of every matching mesh, not the geoms' origins: the origin of
    Stretch's fingertip mesh sits some centimetres behind its front face, and
    reading it as the fingertip is where the 1.5cm in `STRETCH_GRASP_OFFSET_M`
    came from. Primitives are measured from their half-sizes for the same reason.
    """
    origin, axis = pose[:3, 3], pose[:3, approach_col]
    low, high = np.inf, -np.inf
    for geom_id in range(model.ngeom):
        name = model.geom(geom_id).name
        if not name or not match(name):
            continue
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = model.geom_dataid[geom_id]
            start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
            vertices = model.mesh_vert[start : start + count].reshape(-1, 3)
        else:
            size = model.geom_size[geom_id]
            vertices = np.array(
                [
                    [x, y, z]
                    for x in (-size[0], size[0])
                    for y in (-size[1], size[1])
                    for z in (-size[2], size[2])
                ]
            )
        points = data.geom_xpos[geom_id] + vertices @ data.geom_xmat[geom_id].reshape(3, 3).T
        along = (points - origin) @ axis * 1000.0
        low, high = min(low, along.min()), max(high, along.max())
    return float(high), float(low)


def is_robotiq_pad(name: str) -> bool:
    return name.startswith(aperture.FRANKA_NAMESPACE) and "pad" in name


def is_stretch_fingertip(name: str) -> bool:
    return (
        name.startswith(aperture.STRETCH_NAMESPACE)
        and "fingertip" in name
        and "collision" in name
        and "aruco" not in name
    )


def sweep(
    offsets=OFFSETS_M,
    finger_angle: float = STRETCH_FINGER_ANGLE_RAD,
    drop_m: float = FRANKA_DROP_M,
    render_to: Path | None = None,
) -> list[dict]:
    """Measure both hands' surfaces at each offset, on one compiled scene.

    Compiling the pair is nearly all of the cost and nothing about the scene
    depends on the offset, so the sweep builds it once and re-retargets. The
    Robotiq is settled once for the same reason: its linkage is internal to the
    hand, so a configuration settled in free space is right at any arm pose.
    """
    model, data, stretch_view, franka_view, franka_config = aperture.build_overlay_scene()
    base = np.asarray(stretch_view.get_move_group("base").joint_pos, dtype=float)
    mount = fr.franka_mount_pose_from_base(base)
    proxy = fr.FrankaOnStretchView(
        stretch_view, aperture.STRETCH_NAMESPACE, mount, include_base=False
    )
    command = reachable_franka_command(proxy, franka_config, mount, drop_m)
    aperture.settle_robotiq(model, data, franka_view, aperture.FRANKA_NAMESPACE, franka_config, 0.0)

    renderer = camera = None
    if render_to is not None:
        from examples.machine_learning.molmospaces.retargetting.replay import (
            _ensure_offscreen_buffer,
        )

        render_to.mkdir(parents=True, exist_ok=True)
        _ensure_offscreen_buffer(model, RENDER_SIZE)
        renderer = mujoco.Renderer(model, height=RENDER_SIZE[1], width=RENDER_SIZE[0])
        camera = mujoco.MjvCamera()
        camera.distance = CAMERA_DISTANCE
        camera.elevation = CAMERA_ELEVATION

    rows = []
    try:
        for offset in offsets:
            from examples.machine_learning.molmospaces.retargetting.setups import (
                apply_tool_correction,
            )

            apply_tool_correction(proxy, wrist_tilt_deg=0.0, grasp_offset_m=float(offset))
            for group, value in proxy.retarget_franka_joint_pos(command).items():
                stretch_view.get_move_group(group).joint_pos = value
            stretch_view.get_move_group("gripper").joint_pos = [finger_angle, finger_angle]
            # Written back each iteration: the retargeting's IK runs on scratch
            # data, but the Robotiq's own arm is in *this* scene and a previous
            # iteration's forward pass is the only thing holding it.
            franka_view.get_move_group("arm").joint_pos = list(command)
            mujoco.mj_forward(model, data)

            # Everything on the Franka's approach axis (+z), from its grasp site:
            # one ruler for two hands. Stretch's own axis is +x and the tool
            # correction has already lined the two up, so this is the same axis
            # measured once.
            franka_pose = np.asarray(
                franka_view.get_move_group("arm").leaf_frame_to_world, dtype=float
            )
            stretch_pose = np.asarray(
                stretch_view.get_move_group("gripper").leaf_frame_to_world, dtype=float
            )
            wanted = proxy.franka_tool_pose_to_world(proxy.franka.fk(command))
            robotiq_front, robotiq_back = surface_span(model, data, franka_pose, 2, is_robotiq_pad)
            stretch_front, stretch_back = surface_span(
                model, data, franka_pose, 2, is_stretch_fingertip
            )
            rows.append(
                {
                    "offset_m": float(offset),
                    "robotiq_mm": (robotiq_back, robotiq_front),
                    "stretch_mm": (stretch_back, stretch_front),
                    "front_gap_mm": robotiq_front - stretch_front,
                    "centre_gap_mm": 0.5 * (robotiq_back + robotiq_front)
                    - 0.5 * (stretch_back + stretch_front),
                    "frames_apart_mm": float(
                        (stretch_pose[:3, 3] - franka_pose[:3, 3]) @ franka_pose[:3, 2] * 1000.0
                    ),
                    "ik_residual_mm": float(
                        np.linalg.norm(stretch_pose[:3, 3] - wanted[:3, 3]) * 1000.0
                    ),
                }
            )

            if renderer is None:
                continue
            camera.lookat[:] = franka_pose[:3, 3]
            # Down the jaw line, so the two hands' surfaces separate left and right.
            camera.azimuth = float(
                np.degrees(np.arctan2(franka_pose[1, 1], franka_pose[0, 1])) + 90.0
            )
            renderer.update_scene(data, camera)
            rows[-1]["frame"] = _write_frame(renderer.render(), rows[-1], render_to)
    finally:
        if renderer is not None:
            renderer.close()
    return rows


def _write_frame(frame, row: dict, output_dir: Path) -> Path:
    """One PNG, captioned with the numbers it is a picture of."""
    import cv2

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.putText(
        frame,
        f"grasp_offset_m = {row['offset_m']:+.4f}",
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"robotiq pads {row['robotiq_mm'][0]:+.0f}..{row['robotiq_mm'][1]:+.0f}mm"
        f"   stretch tips {row['stretch_mm'][0]:+.0f}..{row['stretch_mm'][1]:+.0f}mm"
        f"   (from the robotiq grasp site, along its approach)",
        (16, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )
    path = output_dir / f"offset_{row['offset_m']:+.4f}.png"
    cv2.imwrite(str(path), frame)
    return path


def report(rows: list[dict]) -> None:
    """The table, and the offset it says to use."""
    click.secho(
        "\n== where each hand's gripping surface sits, along the Robotiq's approach ==", bold=True
    )
    click.echo("   (mm from the Robotiq's grasp site; + is past it, away from the wrist)\n")
    click.echo(
        "   offset  |  robotiq pads   | stretch fingertips |  front |"
        " centre | frames | IK residual"
    )
    for row in rows:
        suspect = row["ik_residual_mm"] > 1.0
        click.secho(
            f"  {row['offset_m']:+.4f} | {row['robotiq_mm'][0]:+6.1f} .. {row['robotiq_mm'][1]:+5.1f}"
            f" | {row['stretch_mm'][0]:+8.1f} .. {row['stretch_mm'][1]:+7.1f}"
            f" | {row['front_gap_mm']:+6.1f} | {row['centre_gap_mm']:+6.1f}"
            f" | {row['frames_apart_mm']:+6.1f} | {row['ik_residual_mm']:6.2f}mm"
            + ("   <- the lift is saturated; this row is not a measurement" if suspect else ""),
            fg="yellow" if suspect else None,
        )

    usable = [row for row in rows if row["ik_residual_mm"] <= 1.0]
    if not usable:
        click.secho(
            "\n  Every row saturated. Raise --drop until the retargeting tracks.", fg="red"
        )
        return
    # Each measurement translates Stretch's hand by its own offset, so any row
    # gives the same answer; the median is here only to shrug off IK noise.
    centred = float(np.median([row["offset_m"] + row["centre_gap_mm"] / 1000.0 for row in usable]))
    fronted = float(np.median([row["offset_m"] + row["front_gap_mm"] / 1000.0 for row in usable]))
    click.secho(
        f"\n  grasp_offset_m = {fronted:+.4f} makes the two hands' front edges flush.",
        fg="green",
    )
    click.echo(
        "  That is the landmark to use: it is where each hand first touches anything, and\n"
        "  it is the one thing the two surfaces have in common. The Robotiq's pads are\n"
        "  parallel and Stretch's fingertip is a wedge, so there is no second landmark\n"
        "  that means the same thing on both hands.\n"
        f"\n  (Their midpoints would coincide at {centred:+.4f}, and that number is not worth\n"
        "  much: the Robotiq's midpoint is the middle of a uniform pad, Stretch's is the\n"
        "  middle of a taper running from 127mm wide to 3mm.)\n"
        "\n  Neither is the same question as holding the object in a jaw as wide as the\n"
        "  Robotiq's. That one cannot pick an offset at all -- `aperture.py` solves for the\n"
        "  tip separation that makes it true at whatever offset it is given."
    )


@click.command()
@click.option(
    "--offset",
    "offsets",
    type=float,
    multiple=True,
    help="Grasp offsets to measure, in metres. Repeatable. Defaults to OFFSETS_M.",
)
@click.option(
    "--finger-angle",
    type=float,
    default=STRETCH_FINGER_ANGLE_RAD,
    show_default=True,
    help="Stretch's finger angle, in radians. The surfaces move with it; see "
    "STRETCH_FINGER_ANGLE_RAD.",
)
@click.option(
    "--drop",
    type=float,
    default=FRANKA_DROP_M,
    show_default=True,
    help="Metres to lower the Franka's home tool pose by, to get it under Stretch's lift "
    "ceiling. Too little and every row is the lift at its end stop.",
)
@click.option(
    "--render/--no-render",
    default=True,
    show_default=True,
    help="Write one PNG per offset of the two hands, seen down the jaw line.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("eval_output/grasp_center_alignment"),
    show_default=True,
    help="Where --render writes its PNGs.",
)
def main(
    offsets: tuple[float, ...], finger_angle: float, drop: float, render: bool, output_dir: Path
) -> None:
    """Measure the offset at which the two hands' gripping surfaces coincide."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    rows = sweep(
        offsets=tuple(offsets) or OFFSETS_M,
        finger_angle=finger_angle,
        drop_m=drop,
        render_to=output_dir if render else None,
    )
    report(rows)
    if render:
        click.secho(f"\nWrote {len(rows)} frames to {output_dir}", fg="green")


if __name__ == "__main__":
    main()
