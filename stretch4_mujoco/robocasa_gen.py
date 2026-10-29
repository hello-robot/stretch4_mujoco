from collections import OrderedDict
from typing import Tuple

import click
import mujoco
import mujoco.viewer
import numpy as np
import robosuite
from robocasa.models.scenes.scene_registry import LayoutType, StyleType
from robosuite import load_part_controller_config
from termcolor import colored

from stretch4_mujoco.utils import (
    insert_line_after_mujoco_tag,
    replace_xml_tag_value,
    xml_modify_body_pos,
    xml_remove_subelement,
    xml_remove_tag_by_name,
    change_start_pose,
    xml_add_floor_geom,
)


def get_styles() -> OrderedDict:
    raw_styles = dict(map(lambda item: (item.value, item.name.lower().capitalize()), StyleType))
    styles = OrderedDict()
    for k in sorted(raw_styles.keys()):
        if k < 0:
            continue
        styles[k] = raw_styles[k]
    return styles


layouts = OrderedDict(
    [
        (0, "One wall"),
        (1, "One wall w/ island"),
        (2, "L-shaped"),
        (3, "L-shaped w/ island"),
        (4, "Galley"),
        (5, "U-shaped"),
        (6, "U-shaped w/ island"),
        (7, "G-shaped"),
        (8, "G-shaped (large)"),
        (9, "Wraparound"),
    ]
)

"""
Modified version of robocasa's kitchen scene generation script
https://github.com/robocasa/robocasa/blob/main/robocasa/demos/demo_kitchen_scenes.py
"""


def choose_option(options, option_name, show_keys=False, default=None, default_message=None):
    """
    Prints out environment options, and returns the selected env_name choice

    Returns:
        str: Chosen environment name
    """
    # get the list of all tasks

    if default is None:
        default = options[0]

    if default_message is None:
        default_message = default

    # Select environment to run
    print("{}s:".format(option_name.capitalize()))

    for i, (k, v) in enumerate(options.items()):
        if show_keys:
            print("[{}] {}: {}".format(i, k, v))
        else:
            print("[{}] {}".format(i, v))
    print()
    try:
        s = input(
            "Choose an option 0 to {}, or any other key for default ({}): ".format(
                len(options) - 1,
                default_message,
            )
        )
        # parse input into a number within range
        k = min(max(int(s), 0), len(options) - 1)
        choice = list(options.keys())[k]
    except Exception:
        if default is None:
            choice = options[0]
        else:
            choice = default
        print("Use {} by default.\n".format(choice))

    # Return the chosen environment name
    return choice


def choose_layout():
    layout = choose_option(layouts, "kitchen layout", default=-1, default_message="random layouts")

    if layout == -1:
        layout = np.random.choice(range(10))
        print(colored(f"Randomly choosing layout... id: {layout}", "yellow"))

    return layout


def choose_style():
    styles = get_styles()
    style = choose_option(styles, "kitchen style", default=-1, default_message="random styles")

    if style == -1:
        style = np.random.choice(range(11))
        print(colored(f"Randomly choosing style... id: {style}", "yellow"))

    return style


def layout_from_str(layout: str) -> int:
    """Returns the index of the layout in the orderedDict"""
    return list(layouts.values()).index(layout)


def style_from_str(style: str) -> int:
    """Returns the index of the style in the orderedDict"""
    return list(get_styles().values()).index(style)


def model_generation_wizard(
    stretch_xml_absolute: str,
    task: str = "PickPlaceCounterToCabinet",
    layout: int = None,
    style: int = None,
    write_to_file: str = None,
    robot_spawn_pose: dict = None,
    objects_list: list = None,
) -> Tuple[mujoco.MjModel, str, dict]:
    """
    Wizard/API to generate a kitchen model for a given task, layout, and style.
    If layout and style are not provided, it will take you through a wizard to choose them in the terminal.
    If robot_spawn_pose is not provided, it will spawn the robot to the default pose from robocasa fixtures.
    You can also write the generated xml model with absolutepaths to a file.
    The Object placements are made based on the robocasa defined Kitchen task and uses the default randomized
    placement distribution
    Args:
        task (str): task name
        layout (int): layout id
        style (int): style id
        write_to_file (str): write to file
        robot_spawn_pose (dict): robot spawn pose {pos: "x y z", quat: "x y z w"}
    Returns:
        Tuple[mujoco.MjModel, str, Dict]: model, xml string and Object placements info
    """

    if layout is None:
        layout = choose_layout()
    else:
        layout = layout

    styles = get_styles()
    if style is None:
        style = choose_style()
    else:
        style = style

    # Create argument configuration
    # TODO: Figure how to get an env without robot arg
    config = {
        "env_name": task,
        "robots": "PandaMobile",
        "controller_configs": load_part_controller_config(default_controller="OSC_POSE"),
        "translucent_robot": False,
        "layout_and_style_ids": [[layout, style]],
    }
    if objects_list is not None:
        config["obj_groups"] = objects_list

    print(colored("Initializing environment...", "yellow"))

    env = robosuite.make(
        **config,
        has_offscreen_renderer=False,
        render_camera=None,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    print(
        colored(
            f"Showing configuration:\n    Layout: {layouts[layout]}\n    Style: {styles[style]}",
            "green",
        )
    )
    print()
    print(
        colored(
            "Spawning environment...\n",
            "yellow",
        )
    )
    env._load_model()
    model = env.model.get_model()
    xml = env.model.get_xml()

    # Add the object placements to the xml
    click.secho(f"\nMaking Object Placements for task [{task}]...\n", fg="yellow")
    object_placements_info = {}
    for i in range(len(env.object_cfgs)):
        obj_name = env.object_cfgs[i]["name"]
        category = env.object_cfgs[i]["info"]["cat"]
        object_placements = env.object_placements
        print(
            f"Placing [Object {i}] (category: {category}, body_name: {obj_name}_main) at "
            f"pos: {np.round(object_placements[obj_name][0],2)} quat: {np.round(object_placements[obj_name][1],2)}"
        )
        xml = xml_modify_body_pos(
            xml,
            "body",
            obj_name + "_main",  # Object name ref in the xml
            pos=object_placements[obj_name][0],
            quat=object_placements[obj_name][1],
        )
        object_placements_info[obj_name + "_main"] = {
            "cat": category,
            "pos": object_placements[obj_name][0],
            "quat": object_placements[obj_name][1],
        }

    xml, robot_base_fixture_pose = custom_cleanups(xml)

    # If the env has an anchor for the robot, use that instead of the dummy pos from the XML
    env_pos = getattr(env, "init_robot_base_pos_anchor", None)
    env_ori = getattr(env, "init_robot_base_ori_anchor", None)
    if env_pos is not None and env_ori is not None:
        from scipy.spatial.transform import Rotation

        quat_xyzw = Rotation.from_euler("xyz", env_ori).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        # Apply a backward offset so the robot doesn't spawn too close to spawned objects
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat_wxyz)
        mat = mat.reshape((3, 3))
        offset = -0.55 * mat[:, 0]  # Pull back to not hit objects
        env_pos = env_pos + offset

        if robot_base_fixture_pose is None:
            robot_base_fixture_pose = {}
        robot_base_fixture_pose["pos"] = " ".join(map(str, env_pos))
        robot_base_fixture_pose["quat"] = " ".join(map(str, quat_wxyz))

    if robot_spawn_pose is not None:
        robot_base_fixture_pose = robot_spawn_pose

    # add stretch to kitchen
    click.secho(
        f"\nAdding Robot to Kitchen at pos: {robot_base_fixture_pose['pos']}, quat: {robot_base_fixture_pose['quat']}\n",
        fg="yellow",
    )
    xml = add_stretch_to_kitchen(
        xml, robot_base_fixture_pose, stretch_xml_absolute=stretch_xml_absolute
    )
    model = mujoco.MjModel.from_xml_string(xml)

    if robot_base_fixture_pose:

        def _parse(s):
            return list(map(float, s.split()))

        translation = _parse(robot_base_fixture_pose["pos"])
        rotation = _parse(robot_base_fixture_pose["quat"])
        change_start_pose(model, translation=translation, rotation_quat=rotation, name="stretch4")

    if write_to_file is not None:
        with open(write_to_file, "w") as f:
            f.write(xml)
        print(colored(f"Model saved to {write_to_file}", "green"))

    return model, xml, object_placements_info


def custom_cleanups(xml: str) -> Tuple[str, dict]:
    """
    Custom cleanups to models from robocasa envs to support
    use with stretch4_mujoco package.
    """

    # make invisible the red/blue boxes around geom/sites of interests found
    xml = replace_xml_tag_value(xml, "geom", "rgba", "0.5 0 0 0.5", "0.5 0 0 0")
    xml = replace_xml_tag_value(xml, "geom", "rgba", "0.5 0 0 1", "0.5 0 0 0")
    xml = replace_xml_tag_value(xml, "site", "rgba", "0.5 0 0 1", "0.5 0 0 0")
    xml = replace_xml_tag_value(xml, "site", "actuator", "0.3 0.4 1 0.5", "0.3 0.4 1 0")
    # remove subelements
    xml = xml_remove_subelement(xml, "actuator")
    xml = xml_remove_subelement(xml, "sensor")

    # remove option tag element
    xml = xml_remove_subelement(xml, "option")
    # xml = xml_remove_subelement(xml, "size")

    # remove robot
    xml, remove_robot_attrib = xml_remove_tag_by_name(xml, "body", "robot0_base")

    # add floor geom
    xml = xml_add_floor_geom(xml)

    return xml, remove_robot_attrib


def add_stretch_to_kitchen(xml: str, robot_pose_attrib: dict, stretch_xml_absolute: str) -> str:
    """
    Add stretch robot to kitchen xml
    """
    print(
        f"Adding stretch to kitchen at pos: {robot_pose_attrib['pos']} quat: {robot_pose_attrib['quat']}"
    )

    # add Stretch xml
    xml = insert_line_after_mujoco_tag(
        xml,
        f' <include file="{pathlib.Path(stretch_xml_absolute).as_posix()}"/>',
    )
    return xml
