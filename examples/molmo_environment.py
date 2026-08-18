"""
Run Stretch 4 inside a MolmoSpaces scene (https://github.com/allenai/molmospaces).

MolmoSpaces builds a scene as a `mujoco.MjSpec`, attaches a robot into it, and
compiles the result. Their `examples/add_robot` (xarm7) shows the pattern:

    spec = MjSpec.from_file(scene_xml)
    Robot.add_robot_to_scene(robot_config, spec, prefix="robot_0/", pos=..., quat=...)
    model = spec.compile()

This example does the same for the Stretch 4 MJCF that
`stretch4_mujoco/models/stretch_4/mjcf_generator.py` generates, then hands the
compiled `MjModel` to `Stretch4MujocoSimulator(model=...)` so the usual
stretch4_mujoco control/camera/lidar API drives the robot inside the house.

`add_stretch_to_scene()` below is deliberately shaped like MolmoSpaces'
`Robot.add_robot_to_scene()` classmethod, so it can be lifted into a
`StretchRobot(Robot)` subclass if you later want to run Stretch through their
datagen pipeline instead of through `Stretch4MujocoSimulator`.

Usage:
    # A scene from a MolmoSpaces house dataset (downloads assets on first use)
    python -m examples.molmo_environment --dataset procthor-10k --house-index 0

    # Or any scene XML on disk
    python -m examples.molmo_environment --scene /path/to/scene.xml
"""

import re
from pathlib import Path

import click
import cv2
import numpy as np
from mujoco import MjModel, MjSpec, MjsBody

from examples.camera_feeds import show_camera_feeds_sync
from examples.rerun_utils import RerunLogger
from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator

# `mjcf_generator.generate_mjcf()` names the root (free-jointed) body "stretch4",
# and `utils.change_start_pose()` looks that name up to set the spawn pose.
STRETCH_ROOT_BODY = "stretch4"

# The wheel contact pairs in models/stretch_4/contact.xml are written against a
# geom named "floor" (the one in models/room_scene/room_scene.xml). A MolmoSpaces
# house has its own floors, so those pairs get retargeted at attach time.
STRETCH_FLOOR_GEOM = "floor"

# `molmo_spaces/housegen/builder.py`'s `add_room()` names each room's floor
# mesh geom "room_<id>_visual_0".
ROOM_FLOOR_GEOM_PATTERN = re.compile(r"^room_(\d+)_visual_0$")


def find_largest_room_spawn_xy(spec: MjSpec) -> tuple[float, float] | None:
    """
    Find a point inside the largest room of a MolmoSpaces house, in scene xy.

    A procthor house is not generally centered on the scene origin, so
    defaulting the robot's spawn x/y to (0, 0) -- as if the scene origin were
    always inside the house -- can spawn it outside the building entirely.

    housegen's `add_room()` positions each room's floor geom (named
    "room_<id>_visual_0") with `pos=[0, 0, room_z]` relative to a body that is
    otherwise left at the world origin. MuJoCo's mesh compiler recenters a
    mesh around its own centroid and folds the offset into the compiled
    `geom_pos`, so after compiling, `geom_pos[:2]` for a room's floor mesh is
    already a point at that room's centroid, in scene world coordinates.

    Returns None if the scene has no MolmoSpaces room floor meshes (e.g. a
    bare, non-MolmoSpaces scene XML), so callers can fall back to their own
    default.
    """
    model = spec.compile()

    best_xy = None
    best_area = -1.0
    for geom_id in range(model.ngeom):
        name = model.geom(geom_id).name
        if not name or not ROOM_FLOOR_GEOM_PATTERN.match(name):
            continue
        # geom_aabb is [center_x, center_y, center_z, half_x, half_y, half_z]
        # in the geom's own (mesh-inertia-aligned) frame, which generally
        # doesn't line up with world xy -- but since a floor mesh is flat,
        # its two largest half-extents approximate the room's footprint
        # regardless of that frame's orientation.
        half_extents = sorted(model.geom_aabb[geom_id][3:], reverse=True)
        area = 4.0 * half_extents[0] * half_extents[1]
        if area > best_area:
            best_area = area
            best_xy = (float(model.geom_pos[geom_id][0]), float(model.geom_pos[geom_id][1]))
    return best_xy


def resolve_spawn_position(
    spec: MjSpec, x: float | None, y: float | None, z: float
) -> list[float]:
    """
    Fill in unset x/y spawn coordinates from the scene's largest room, if any.

    x and y default to None (rather than 0.0) on the CLI specifically so this
    can tell "the user wants the origin" apart from "let the scene pick a
    spawn point that's actually inside the house".
    """
    if x is not None and y is not None:
        return [x, y, z]

    room_xy = find_largest_room_spawn_xy(spec)
    if room_xy is None:
        if x is None and y is None:
            click.secho(
                "No MolmoSpaces room floor meshes found in the scene; "
                "defaulting spawn x/y to 0.0. Pass --x/--y explicitly if the "
                "robot spawns outside the scene.",
                fg="yellow",
            )
        room_xy = (0.0, 0.0)
    else:
        click.secho(
            f"Defaulting spawn to the largest room's centroid: x={room_xy[0]:.2f}, "
            f"y={room_xy[1]:.2f}. Pass --x/--y to override.",
            fg="yellow",
        )
    return [x if x is not None else room_xy[0], y if y is not None else room_xy[1], z]


def find_floor_geoms(spec: MjSpec) -> list[str]:
    """
    Return the names of the geoms in a MolmoSpaces scene that the robot drives on.

    MolmoSpaces' housegen (`molmo_spaces/housegen/builder.py`) gives a house one
    colliding ground plane, named "floor", built from its `__STRUCTURAL_MJT__`
    class. Conveniently that is the same name the Stretch contact pairs already
    use, so on a stock house this usually finds exactly one geom and the
    retargeting below is a no-op.

    Note that the per-room floor meshes ("room_<id>_visual_0") are deliberately
    *not* matched. Despite being what `molmo_spaces/utils/scene_maps.py` scans to
    render its walkable-area maps, they are built from the `__VISUAL_MJT__` class
    (contype=0, conaffinity=0, mass=1e-8) and are pure decoration. Pairing a
    wheel against one would be worse than useless: an explicit `<pair>` bypasses
    contype/conaffinity filtering, so it would resurrect collisions with a mesh
    the scene intends to be non-physical.
    """
    floor_geoms = []
    for geom in spec.geoms:
        name = geom.name
        if not name or not name.startswith(STRETCH_FLOOR_GEOM):
            continue
        if geom.contype == 0 and geom.conaffinity == 0:
            continue  # visual-only, see above
        floor_geoms.append(name)
    return floor_geoms


def retarget_wheel_contact_pairs(robot_spec: MjSpec, floor_geom_names: list[str]) -> None:
    """
    Point the omniwheel contact pairs at the scene's floors instead of "floor".

    This is what makes the omniwheels behave like omniwheels rather than like
    rigid casters, so it is not optional. The pairs carry an anisotropic sliding
    friction (`friction="0.001 1.0"`, i.e. near-frictionless along the roller
    axis and grippy along the drive direction), and a `<geom friction=...>`
    cannot express that -- geom-level friction applies the same coefficient to
    both tangential directions. MuJoCo only ever derives friction from a geom
    pair when there is no explicit `<pair>`, so dropping these pairs silently
    turns the base into a normal three-wheeled cart.

    Explicit pairs also bypass contype/conaffinity filtering, so the wheels are
    guaranteed contact with every floor listed here.

    Args:
        robot_spec: the Stretch MjSpec, before it is attached to the scene.
        floor_geom_names: scene geoms to generate a pair against, per wheel.
    """
    templates = [pair for pair in robot_spec.pairs if pair.geomname1 == STRETCH_FLOOR_GEOM]
    if not templates:
        raise RuntimeError(
            f"No contact pairs against '{STRETCH_FLOOR_GEOM}' found in the Stretch MJCF. "
            "models/stretch_4/contact.xml is expected to define them for the omniwheels."
        )
    if not floor_geom_names:
        raise RuntimeError(
            "No floor geoms found in the scene, so the omniwheel contact pairs cannot be "
            "retargeted. Pass --floor-geom explicitly (repeatable)."
        )

    for template in templates:
        for floor_geom_name in floor_geom_names:
            pair = robot_spec.add_pair()
            pair.geomname1 = floor_geom_name
            pair.geomname2 = template.geomname2
            pair.condim = template.condim
            pair.friction = template.friction
            pair.solref = template.solref
            pair.solreffriction = template.solreffriction
            pair.solimp = template.solimp
            pair.margin = template.margin
            pair.gap = template.gap
        # The original references a geom name that does not exist in the scene,
        # so leaving it in place fails the compile. MuJoCo moved element deletion
        # from `element.delete()` to `spec.delete(element)` after 3.3, and this
        # repo pins 3.3.1 while MolmoSpaces wants ~=3.5.0, so support both.
        if hasattr(robot_spec, "delete"):
            robot_spec.delete(template)
        else:
            template.delete()


def apply_stretch_solver_options(spec: MjSpec, robot_spec: MjSpec) -> None:
    """
    Copy the solver settings the Stretch model is tuned for onto the scene.

    `MjSpec` attachment merges model elements only -- `<option>` stays whatever
    the scene declared -- but the anisotropic wheel friction is only faithfully
    reproduced under an elliptic friction cone with a high `impratio`. These are
    the values from models/stretch_4/defaults.xml. MolmoSpaces scenes are already
    elliptic with `impratio=10`, so this mostly just raises the impedance ratio.
    """
    spec.option.cone = robot_spec.option.cone
    spec.option.impratio = robot_spec.option.impratio
    spec.option.noslip_iterations = robot_spec.option.noslip_iterations
    spec.option.integrator = robot_spec.option.integrator
    spec.option.enableflags |= robot_spec.option.enableflags


def add_stretch_to_scene(
    spec: MjSpec,
    pos: list[float],
    quat: list[float],
    robot_xml_path: str | None = None,
    floor_geom_names: list[str] | None = None,
    match_solver_options: bool = True,
) -> MjsBody:
    """
    Attach the generated Stretch 4 MJCF to a MolmoSpaces scene spec, in place.

    Mirrors MolmoSpaces' `Robot.add_robot_to_scene()`, with two Stretch-specific
    departures from their xarm7 example:

    1. No mocap base body. xarm7 is a fixed arm that gets teleported by a mocap
       body; Stretch is a mobile base whose root body carries a freejoint and is
       driven by its wheels, and a free-jointed body cannot hang off a mocap
       body. The root is attached straight to a worldbody frame instead.
    2. An empty name prefix, where MolmoSpaces would pass "robot_0/". Both
       `MujocoServer` and `utils.change_start_pose()` resolve model elements by
       their bare names ("stretch4", "base_link", "lift", the lidar sites, ...),
       so prefixing them would break every lookup. The consequence is that the
       scene must not already use those names, and only one Stretch can be
       attached per scene.

    Args:
        spec: the MolmoSpaces scene spec to attach into.
        pos: robot spawn position in the scene, [x, y] or [x, y, z].
        quat: robot spawn orientation as a MuJoCo [w, x, y, z] quaternion.
        robot_xml_path: Stretch MJCF to attach. Defaults to the freshly generated
            one from `Stretch4MujocoSimulator.get_robot_xml_path()`.
        floor_geom_names: scene geoms to treat as drivable floor. Defaults to
            `find_floor_geoms(spec)`.
        match_solver_options: overwrite the scene's `<option>` with the Stretch
            model's, see `apply_stretch_solver_options()`.

    Returns:
        The attached root body.
    """
    if robot_xml_path is None:
        # Regenerates the MJCF from the URDF via mjcf_generator.generate_mjcf().
        robot_xml_path = Stretch4MujocoSimulator.get_robot_xml_path()
    if floor_geom_names is None:
        floor_geom_names = find_floor_geoms(spec)

    pos = list(pos) + [0.0] if len(pos) == 2 else list(pos)

    robot_spec = MjSpec.from_file(str(robot_xml_path))
    retarget_wheel_contact_pairs(robot_spec, floor_geom_names)

    if match_solver_options:
        apply_stretch_solver_options(spec, robot_spec)

    # Both specs call their root default class "main", so the merged spec ends up
    # with two of them. That compiles fine, but serializes to a second unnamed
    # <default> block that MuJoCo then refuses to read back ("empty class name"),
    # which would make --write-to-file produce a scene nobody can open. Renaming
    # is purely cosmetic -- elements hold a pointer to their default, not its
    # name -- and leaves the compiled model bit-for-bit identical.
    robot_spec.default.name = "stretch_main"

    robot_root = robot_spec.body(STRETCH_ROOT_BODY)
    if robot_root is None:
        raise ValueError(f"Body '{STRETCH_ROOT_BODY}' not found in {robot_xml_path}")

    attach_frame = spec.worldbody.add_frame(pos=pos, quat=quat)
    return attach_frame.attach_body(robot_root, "", "")


def build_model(
    scene_xml_path: str,
    x: float | None,
    y: float | None,
    z: float,
    quat: list[float],
    floor_geom_names: list[str] | None = None,
    match_solver_options: bool = True,
    write_to_file: str | None = None,
) -> MjModel:
    """
    Load a MolmoSpaces scene, attach Stretch 4 to it, and compile it.

    Returns a model ready for `Stretch4MujocoSimulator(model=...)`. x/y may be
    None to auto-spawn in the scene's largest room, see
    `resolve_spawn_position()`.
    """
    spec = MjSpec.from_file(str(scene_xml_path))

    pos = resolve_spawn_position(spec, x, y, z)

    add_stretch_to_scene(
        spec,
        pos=pos,
        quat=quat,
        floor_geom_names=floor_geom_names,
        match_solver_options=match_solver_options,
    )

    model = spec.compile()

    if write_to_file is not None:
        Path(write_to_file).write_text(spec.to_xml())
        click.secho(f"Wrote combined scene to {write_to_file}", fg="green")

    return model


def resolve_molmospaces_scene(dataset: str, split: str, house_index: int, variant: str) -> str:
    """
    Look up a house from a MolmoSpaces dataset and install its assets locally.

    Requires molmospaces to be installed; the scene and its object/grasp archives
    are downloaded into the MolmoSpaces resource cache on first use.
    """
    try:
        from molmo_spaces.molmo_spaces_constants import get_scenes
        from molmo_spaces.utils.lazy_loading_utils import (
            install_scene_with_objects_and_grasps_from_path,
        )
    except ImportError as e:
        raise click.ClickException(
            "MolmoSpaces is not installed, so --dataset cannot be resolved. Either install it\n"
            '  pip install "molmospaces[mujoco] @ git+https://github.com/allenai/molmospaces.git"\n'
            "or point --scene at a scene XML on disk."
        ) from e

    houses = get_scenes(dataset, split=split)[split]
    if house_index not in houses or houses[house_index] is None:
        raise click.ClickException(
            f"House {house_index} is not available in {dataset}/{split}. "
            f"Available indices: {sorted(i for i, h in houses.items() if h is not None)[:20]}..."
        )

    house = houses[house_index]
    # Older dataset indices map an index straight to a path, newer ones map it to
    # a {variant: path} dict ("base", "ceiling", "map", ...).
    if isinstance(house, dict):
        if house.get(variant) is None:
            available = [v for v, p in house.items() if p is not None]
            raise click.ClickException(
                f"Variant '{variant}' unavailable for house {house_index}. Available: {available}"
            )
        scene_path = house[variant]
    else:
        scene_path = house

    click.secho(f"Installing MolmoSpaces scene assets for {scene_path}...", fg="yellow")
    install_scene_with_objects_and_grasps_from_path(str(scene_path))
    return str(scene_path)


@click.command()
@click.option("--scene", type=str, default=None, help="Path to a scene XML. Overrides --dataset.")
@click.option("--dataset", type=str, default="procthor-10k", help="MolmoSpaces house dataset")
@click.option("--split", type=str, default="train", help="Dataset split")
@click.option("--house-index", type=int, default=0, help="House index within the split")
@click.option(
    "--variant",
    type=str,
    default="base",
    help="House variant. 'base' omits the ceiling, which keeps the scene viewable from above.",
)
@click.option(
    "--x",
    type=float,
    default=None,
    help="Robot spawn x, in scene coordinates. Defaults to the centroid of the "
    "scene's largest room.",
)
@click.option(
    "--y",
    type=float,
    default=None,
    help="Robot spawn y, in scene coordinates. Defaults to the centroid of the "
    "scene's largest room.",
)
@click.option(
    "--z",
    type=float,
    default=0.0,
    help="Robot spawn z. The MJCF already lifts the wheels onto z=0, so raise this only "
    "if the scene's floor sits above the origin.",
)
@click.option("--yaw", type=float, default=0.0, help="Robot spawn yaw about +z, in radians")
@click.option(
    "--floor-geom",
    type=str,
    multiple=True,
    help="Scene geom to treat as drivable floor for the omniwheel contact pairs. "
    "Repeatable. Defaults to auto-detection by MolmoSpaces naming convention.",
)
@click.option(
    "--match-solver-options/--keep-scene-solver-options",
    default=True,
    help="Overwrite the scene's <option> with the values the Stretch model is tuned for.",
)
@click.option("--write-to-file", type=str, default=None, help="Write the combined scene XML here")
@click.option("--headless", is_flag=True, help="Run without the MuJoCo viewer")
@click.option("--keyboard", is_flag=True, help="Drive the robot with the keyboard (WASDQE, ...)")
@click.option("--gamepad", is_flag=True, help="Drive the robot with an Xbox-style gamepad")
@click.option("--lidar", is_flag=True, help="Show the lidar point cloud in Rerun")
def main(
    scene: str | None,
    dataset: str,
    split: str,
    house_index: int,
    variant: str,
    x: float | None,
    y: float | None,
    z: float,
    yaw: float,
    floor_geom: tuple[str, ...],
    match_solver_options: bool,
    write_to_file: str | None,
    headless: bool,
    keyboard: bool,
    gamepad: bool,
    lidar: bool,
):
    if keyboard and gamepad:
        raise click.UsageError("Pass at most one of --keyboard/--gamepad.")

    scene_xml_path = scene or resolve_molmospaces_scene(dataset, split, house_index, variant)

    model = build_model(
        scene_xml_path=scene_xml_path,
        x=x,
        y=y,
        z=z,
        quat=[np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)],
        floor_geom_names=list(floor_geom) or None,
        match_solver_options=match_solver_options,
        write_to_file=write_to_file,
    )

    rerun_logger = RerunLogger()

    sim = Stretch4MujocoSimulator(
        model=model,
        cameras_to_use=Stretch4MujocoSimulator.get_rgb_cameras(),
        camera_hz=10.0 if lidar else 30.0,
    )
    sim.start(headless=headless)

    if lidar:
        rerun_logger.init_pointcloud_viz(use_stretch_3=False)

    teleop = None
    if keyboard:
        from stretch4_mujoco.sim_teleop import KeyboardTeleop, print_keyboard_help

        print_keyboard_help()
        teleop = KeyboardTeleop(sim)
    elif gamepad:
        from stretch4_mujoco.sim_teleop import GamepadTeleop

        teleop = GamepadTeleop(sim)
    if teleop is not None:
        teleop.start()

    try:
        while sim.is_running():
            show_camera_feeds_sync(sim, True)
            if lidar:
                rerun_logger.update_pointcloud_viz(sim.pull_lidar_points(), "world/lidar_points")
    except KeyboardInterrupt:
        pass
    finally:
        rerun_logger.stop()
        if teleop is not None:
            teleop.stop()
        sim.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
