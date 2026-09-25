"""
Where the Franka-to-Stretch retargeting puts a grasp, measured rather than argued about.

When a Stretch row of the report scores worse than the Franka row above it with
the same camera, the difference is the retargeting -- and "the retargeting" is
two numbers and a coordinate frame, all of which can be read off a standing
robot without running a policy at all. This prints them.

    python -m examples.machine_learning.molmospaces.retargetting.diagnose

It answers three questions, in the order they matter:

1. **How far above the object does a commanded grasp land?**
   `FrankaOnStretchView.target_z_offset` raises every retargeted target, to stop
   Stretch's gripper dragging through the countertop where its lift has run out
   of travel. It is a hand-set number and defaults to **0**: it used to be a
   fraction of the shortfall measured below, applied automatically, and that is
   removed -- the shortfall is measured at the Franka's *home* pose, which is
   above Stretch's ceiling, so it corrected targets that needed no correcting.
   The measurement is still printed here, because knowing the ceiling is useful;
   it is just no longer wired to anything. If the offset you set by hand is
   larger than the object is tall, the gripper closes above it.

2. **Where is the grasp centre, relative to the fingers?**
   The retargeting drives `grasp_center_link` to the pose the policy asked for
   its Robotiq's `grasp_site`. Those two are not the same point on their
   respective grippers, and the difference is how far into the gripper the
   object ends up. `grasp_offset_m` is the correction.

3. **What does the gripper actually do when told to close?**
   The Robotiq's 0-255 command is mapped onto Stretch's finger angle by
   `franka_retarget`; this prints what each end of that range is in metres of
   finger separation.

Everything here is measured on the compiled model at the mini benchmark's own
robot pose, so the numbers are the ones that episode runs with.
"""

from __future__ import annotations

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
from examples.machine_learning.molmospaces.retargetting.setups import (  # noqa: E402
    apply_tool_correction,
)
from examples.machine_learning.molmospaces.stretch.config import Stretch4RobotConfig  # noqa: E402
from examples.machine_learning.molmospaces.stretch.robot import Stretch4Robot  # noqa: E402
from examples.machine_learning.molmospaces.stretch.robot_view import (  # noqa: E402
    Stretch4RobotView,
)


def build_standing_robot() -> tuple[mujoco.MjModel, MjData, Stretch4RobotView, str]:
    """Stretch in the mini benchmark's kitchen, at the pose its episodes start from.

    The same house, the same base xy, and the yaw the episode override computes
    -- `+x` pointing at the target spot on the counter -- so every distance
    below is one an episode actually sees.
    """
    from molmo_spaces.utils.lazy_loading_utils import (
        install_scene_with_objects_and_grasps_from_path,
    )

    scene = mini_benchmark.house_scene_path()
    install_scene_with_objects_and_grasps_from_path(str(scene))
    spec = MjSpec.from_file(str(scene))

    config = Stretch4RobotConfig()
    namespace = config.robot_namespace
    base_xy = mini_benchmark.ROBOT_BASE_XY
    target_xy = mini_benchmark.TARGET_XY
    yaw = math.atan2(target_xy[1] - base_xy[1], target_xy[0] - base_xy[0])

    Stretch4Robot.add_robot_to_scene(
        config,
        spec,
        prefix=namespace,
        pos=list(base_xy),
        quat=R.from_euler("z", yaw).as_quat(scalar_first=True),
    )
    Stretch4Robot.apply_control_overrides(spec, config)

    model = spec.compile()
    data = MjData(model)
    view = Stretch4RobotView(data, namespace)
    qpos = dict(config.init_qpos)
    qpos["base"] = [base_xy[0], base_xy[1], yaw]
    view.set_qpos_dict(qpos)
    mujoco.mj_forward(model, data)
    return model, data, view, namespace


def report_gripper(model, data, view, namespace: str) -> float:
    """Print the gripper's geometry, and return the grasp centre's overhang in metres.

    The overhang is how far `grasp_center_link` sits *past* the fingertips along
    the approach axis. It is the quantity `grasp_offset_m` exists to cancel: the
    retargeting puts the grasp centre on the object, so a positive overhang
    means the object ends up beyond the fingertips rather than between them.
    """
    click.secho("\n== the gripper ==", bold=True)
    gripper = view.get_move_group("gripper")
    low, high = gripper.inter_finger_dist_range
    click.echo(f"inter-finger distance range: {low:.4f} .. {high:.4f} m")
    for angle in (fr.STRETCH_FINGER_CLOSED, 0.25, fr.STRETCH_FINGER_OPEN):
        view.set_qpos_dict({"gripper": [angle, angle]})
        mujoco.mj_forward(model, data)
        label = ""
        if angle == fr.STRETCH_FINGER_CLOSED:
            label = f"  (Robotiq {fr.ROBOTIQ_CTRL_RANGE[1]:.0f}, closed)"
        elif angle == fr.STRETCH_FINGER_OPEN:
            label = f"  (Robotiq {fr.ROBOTIQ_CTRL_RANGE[0]:.0f}, open)"
        click.echo(f"  finger {angle:.2f} rad -> {gripper.inter_finger_dist:.4f} m apart{label}")

    view.set_qpos_dict({"gripper": [fr.STRETCH_FINGER_CLOSED, fr.STRETCH_FINGER_CLOSED]})
    mujoco.mj_forward(model, data)

    body_id = model.body(namespace + "base_link").id
    rotation = data.xmat[body_id].reshape(3, 3)
    origin = data.xpos[body_id]

    def in_base(point) -> np.ndarray:
        return rotation.T @ (np.asarray(point, dtype=float) - origin)

    grasp_centre = in_base(np.asarray(gripper.leaf_frame_to_world)[:3, 3])
    click.echo(f"\ngrasp_center_link, in the base frame: {np.round(grasp_centre, 4).tolist()}")
    click.echo("  (the arm telescopes along +x, so x is reach and z is height)")

    # The fingertip pads are what a grasp closes on. Taking the furthest-forward
    # finger geom rather than a named one, because the tip is what matters and
    # the naming differs between the aruco pads and the pad bodies.
    tip_x = -np.inf
    for geom_id in range(model.ngeom):
        name = model.geom(geom_id).name
        if "finger" in name and "collision" in name:
            tip_x = max(tip_x, float(in_base(data.geom_xpos[geom_id])[0]))
    overhang = float(grasp_centre[0]) - tip_x
    click.echo(f"furthest-forward fingertip geom, x = {tip_x:.4f}")
    click.secho(
        f"  -> the grasp centre sits {overhang * 100:+.1f} cm past the fingertips",
        fg="yellow" if overhang > 0.005 else "green",
    )
    if overhang > 0.005:
        click.echo(
            "     A policy that puts its Robotiq's grasp site on an object therefore\n"
            "     puts this point on the object, leaving the object at or beyond the\n"
            f"     fingertips. grasp_offset_m = +{overhang:.3f} brings it back to the tips;\n"
            "     more than that pulls it deeper between the fingers."
        )
    return overhang


JAW_PROFILE_DEPTHS_MM = tuple(range(0, -121, -10))
"""How far past the grasp centre `report_jaw_profile` probes, in millimetres.

Down to 12cm because that is past the point where Stretch's fingers meet -- the
profile has to include the pinch to show it.
"""


def report_jaw_profile(model, data, view, namespace: str) -> None:
    """Print how wide the jaw actually is at each depth, which is not what the tips say.

    `inter_finger_dist` -- the number every other part of this study calls the
    aperture, and the one `ROBOTIQ_MAX_APERTURE_M` is matched against -- is the
    separation of the two *fingertip bodies*. That is the widest the jaw ever is.
    Stretch's fingers curve inwards behind the tips and meet about 7cm back, so
    the width available to an object is a strong function of how deep into the
    jaw the retargeting puts it, and `grasp_offset_m` is exactly the control that
    decides that depth.

    Which makes this the measurement that ties the two parameters together: an
    object at depth `grasp_offset_m` needs the row at that depth to be wider than
    the object is, and the only way to widen it is to open the hand further --
    which `match_robotiq_aperture` is capping. Read the two together before
    moving either.

    Measured by ray-casting across the jaw from its centre line rather than from
    the geometry, because the fingers are meshes and their inward curve is not in
    any single number the model exposes.
    """
    click.secho("\n== how wide the jaw is, at each depth ==", bold=True)
    gripper = view.get_move_group("gripper")
    apertures = [
        ("matched to the Robotiq's 87mm", 0.087),
        (f"matched to ROBOTIQ_MAX_APERTURE_M ({fr.ROBOTIQ_MAX_APERTURE_M * 1000:.0f}mm)", fr.ROBOTIQ_MAX_APERTURE_M),
        ("wide open, unmatched", None),
    ]
    for label, aperture in apertures:
        angle = (
            fr.STRETCH_FINGER_OPEN
            if aperture is None
            else fr.stretch_finger_for_aperture(gripper, model, data, aperture)
        )
        view.set_qpos_dict({"gripper": [angle, angle]})
        mujoco.mj_forward(model, data)
        pose = np.asarray(view.get_move_group("wrist").leaf_frame_to_world, dtype=float)
        origin, approach, across = pose[:3, 3], pose[:3, 0], pose[:3, 1]

        click.echo(
            f"\n{label}: finger {angle:.4f} rad, tips {gripper.inter_finger_dist * 1000:.0f}mm apart"
        )
        for depth_mm in JAW_PROFILE_DEPTHS_MM:
            point = origin + approach * (depth_mm / 1000.0)
            width = 0.0
            for direction in (across, -across):
                # The ray starts between the fingers and the fingers are the
                # nearest thing either side, so whatever it hits first is the
                # jaw. `flg_static` is on because the counter is static and a
                # ray that escapes the jaw should stop at it rather than run on.
                hit = mujoco.mj_ray(
                    model, data, point, direction, None, 1, -1, np.zeros(1, dtype=np.int32)
                )
                width += hit if hit >= 0 else float("nan")
            bar = "" if not np.isfinite(width) else "#" * int(width * 200)
            click.echo(f"  {depth_mm:5d} mm past the grasp centre | {width * 1000:6.1f} mm {bar}")

    click.echo(
        "\nAn object is left at depth `grasp_offset_m` past the grasp centre, so read the\n"
        "row at the offset you are using: that is the width the object has to fit in."
    )


def report_height_offset(view, namespace: str) -> float:
    """Print the lift shortfall at the Franka's home pose, and what offsets would do.

    Reported, not applied. `measure_tool_height_offset()` is no longer used by the
    policy -- see `StretchMolmoBotDroidPolicyConfig.target_z_offset` -- and this is
    the diagnostic it survives as: the number says how far above Stretch's reach
    the Franka's home tool sits, which is worth knowing when a rollout starts
    there.
    """
    click.secho("\n== the target height offset ==", bold=True)
    proxy = fr.FrankaOnStretchView(
        view,
        namespace,
        fr.franka_mount_pose_from_base(view.get_move_group("base").joint_pos),
        include_base=True,
    )
    shortfall = proxy.measure_tool_height_offset()
    click.echo(f"measure_tool_height_offset(): {shortfall:+.4f} m")
    click.echo(
        "nothing multiplies this any more: `target_z_offset` is set by hand and defaults\n"
        "to 0. For reference, were you to apply a fraction of it by hand:"
    )
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        offset = shortfall * fraction
        click.echo(
            f"  target_z_offset_m={offset:.3f} ({fraction:.2f} of the shortfall) "
            f"-> every grasp {offset * 100:+.1f} cm high"
        )
    return shortfall


def report_objects(shortfall: float, offset: float) -> None:
    """Print how each target object's height compares with the offset being applied."""
    click.secho("\n== the objects, against that offset ==", bold=True)
    episodes = mini_benchmark.build_episodes()
    counter = None
    click.echo(
        f"{'object':14s} {'rest z':>8s} {'above counter':>14s}   "
        f"verdict at target_z_offset_m={offset:.3f}"
    )
    for episode, target in zip(episodes, mini_benchmark.TARGETS):
        rest_z = float(episode["task"]["pickup_obj_start_pose"][2])
        if counter is None:
            counter = rest_z  # only used if the ray cast below fails
        height = rest_z - COUNTER_Z
        verdict = (
            click.style("above the object entirely", fg="red")
            if offset > height
            else click.style("still on the object", fg="green")
        )
        click.echo(f"{target.key:14s} {rest_z:8.4f} {height * 100:13.1f}cm   {verdict}")
    click.echo(
        f"\n('above counter' is the object's centre height over the {COUNTER_Z:.3f} m counter top;\n"
        " a grasp raised further than that closes over the object rather than around it)"
    )


COUNTER_Z = 0.9384
"""The counter top in the hand-tuned kitchen, as `mini_benchmark.surface_under` measures it."""


def report_grasp_offset(view, namespace: str, shortfall: float) -> None:
    """Show where a commanded grasp actually lands, for a few `grasp_offset_m`."""
    click.secho("\n== what grasp_offset_m does to a commanded grasp ==", bold=True)
    proxy = fr.FrankaOnStretchView(
        view,
        namespace,
        fr.franka_mount_pose_from_base(view.get_move_group("base").joint_pos),
        include_base=True,
    )
    proxy.target_z_offset = 0.0

    world = np.eye(4)
    world[:3, 3] = [mini_benchmark.TARGET_XY[0], mini_benchmark.TARGET_XY[1], COUNTER_Z + 0.04]
    # The Franka tool pose a policy aiming its Robotiq at that point would emit.
    apply_tool_correction(proxy, wrist_tilt_deg=0.0, grasp_offset_m=0.0)
    commanded = proxy.stretch_tool_pose_to_franka(world)

    click.echo("a policy aiming at an object drives Stretch's grasp centre to:")
    for offset in (0.0, 0.03, 0.06, 0.09):
        apply_tool_correction(proxy, wrist_tilt_deg=0.0, grasp_offset_m=offset)
        landed = proxy.franka_tool_pose_to_world(commanded)[:3, 3]
        moved = float(np.linalg.norm(landed - world[:3, 3]))
        click.echo(f"  grasp_offset_m={offset:+.3f} -> {moved * 100:5.1f} cm further along the approach")


@click.command()
@click.option(
    "--target-z-offset",
    type=float,
    default=0.0,
    show_default=True,
    help="The offset, in metres, to report the objects against. The policy config's "
    "default is 0; raise it to see which objects a hand-set offset would close above.",
)
def main(target_z_offset: float) -> None:
    import logging

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    click.secho("Measuring the retargeting on a standing robot, no policy involved.", bold=True)
    model, data, view, namespace = build_standing_robot()
    report_gripper(model, data, view, namespace)
    report_jaw_profile(model, data, view, namespace)
    shortfall = report_height_offset(view, namespace)
    report_objects(shortfall, target_z_offset)
    report_grasp_offset(view, namespace, shortfall)
    click.secho(
        "\nBoth numbers above are searchable: --dim target_z_offset_m=0:0.05:3 "
        "--dim grasp_offset_m=0:0.09:4",
        fg="green",
    )


if __name__ == "__main__":
    main()
