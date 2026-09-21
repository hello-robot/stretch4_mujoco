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
   of travel. It is `z_offset_fraction` of a shortfall measured per episode, so
   the number that actually gets applied is not written down anywhere -- and if
   it is larger than the object is tall, the gripper closes above it.

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


def report_height_offset(view, namespace: str) -> float:
    """Print the per-episode lift shortfall and what each `z_offset_fraction` does with it."""
    click.secho("\n== the target height offset ==", bold=True)
    proxy = fr.FrankaOnStretchView(
        view,
        namespace,
        fr.franka_mount_pose_from_base(view.get_move_group("base").joint_pos),
        include_base=True,
    )
    shortfall = proxy.measure_tool_height_offset()
    click.echo(f"measure_tool_height_offset(): {shortfall:+.4f} m")
    click.echo("this is multiplied by z_offset_fraction and added to every commanded target:")
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        offset = shortfall * fraction
        click.echo(f"  z_offset_fraction={fraction:.2f} -> every grasp {offset * 100:+.1f} cm high")
    return shortfall


def report_objects(shortfall: float) -> None:
    """Print how each target object's height compares with the offset being applied."""
    click.secho("\n== the objects, against that offset ==", bold=True)
    episodes = mini_benchmark.build_episodes()
    counter = None
    click.echo(f"{'object':14s} {'rest z':>8s} {'above counter':>14s}   verdict at z_offset_fraction=0.5")
    for episode, target in zip(episodes, mini_benchmark.TARGETS):
        rest_z = float(episode["task"]["pickup_obj_start_pose"][2])
        if counter is None:
            counter = rest_z  # only used if the ray cast below fails
        height = rest_z - COUNTER_Z
        offset = shortfall * 0.5
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
    "--z-offset-fraction",
    type=float,
    default=0.5,
    help="The fraction to report the objects against. The policy config's default is 0.5.",
)
def main(z_offset_fraction: float) -> None:
    import logging

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    click.secho("Measuring the retargeting on a standing robot, no policy involved.", bold=True)
    model, data, view, namespace = build_standing_robot()
    report_gripper(model, data, view, namespace)
    shortfall = report_height_offset(view, namespace)
    report_objects(shortfall)
    report_grasp_offset(view, namespace, shortfall)
    click.secho(
        "\nBoth numbers above are searchable: --dim z_offset_fraction=0:0.5:3 "
        "--dim grasp_offset_m=0:0.09:4",
        fg="green",
    )


if __name__ == "__main__":
    main()
