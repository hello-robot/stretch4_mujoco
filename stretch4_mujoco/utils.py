import dataclasses
import math
import pathlib
import re
import time
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Callable, Tuple

import cv2
import numpy as np
import importlib.resources
import yourdfpy as urdf_loader


from functools import wraps, lru_cache

import mujoco
import mujoco._functions
import mujoco._callbacks
import mujoco._render
import mujoco._enums
import mujoco.viewer
import numpy as np
from mujoco._structs import MjModel
from mujoco.glfw import GLContext as GlFwContext

import stretch4_mujoco.config as config

if TYPE_CHECKING:
    from stretch4_mujoco.stretch4_mujoco_simulator import StretchMujocoSimulator


models_path = importlib.resources.files("stretch4_mujoco") / "models"


def get_urdf_package_path(package_name="stretch_urdf"):
    return importlib.resources.files(package_name)


def get_urdf_file_name(model_name, tool_name):
    return f"stretch_description_{model_name}_{tool_name}.urdf"


def get_mesh_files_directory(package_name="stretch_urdf"):
    return get_urdf_package_path(package_name) / "meshes"


def require_connection(function):
    """Wraps class methods that need self"""

    def wrapper_function(self: "StretchMujocoSimulator", *args, **kwargs):
        if not self.is_running():
            raise ConnectionError(
                "The Stretch Mujoco Simulator is not running. Use the start() method to start it."
            )
        return function(self, *args, **kwargs)

    return wrapper_function


def compute_K(fovy: float, width: int, height: int) -> np.ndarray:
    """
    Compute camera intrinsic matrix
    """
    f = 0.5 * height / math.tan(fovy * math.pi / 360)
    return np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))


def Rx(theta):
    """
    Rotation matrix about x-axis
    """
    return np.matrix(
        [
            [1, 0, 0],
            [0, math.cos(theta), -math.sin(theta)],
            [0, math.sin(theta), math.cos(theta)],
        ]
    )


def Ry(theta):
    """
    Rotation matrix about y-axis
    """
    return np.matrix(
        [
            [math.cos(theta), 0, math.sin(theta)],
            [0, 1, 0],
            [-math.sin(theta), 0, math.cos(theta)],
        ]
    )


def Rz(theta):
    """
    Rotation matrix about z-axis
    """
    return np.matrix(
        [[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]]
    )


def limit_depth_distance(depth_image_meters: np.ndarray, max_depth: float) -> np.ndarray:
    """
    Limit depth distance
    """
    return np.where(depth_image_meters > max_depth, 0, depth_image_meters)


def H0_from_driving_dir(wheel_dia_m, base_radius, forward_dir):
    """
    H0 is the matrix that transforms the base velocity to wheel velocities
    For three wheeled system:
    u[3x1] = H0[3x3] * Vb[3x1]
    u = wheel velocities
    Vb = Twist velocity of the base frame (vx, vy, wz)
    https://control.ros.org/rolling/doc/ros2_controllers/doc/mobile_robot_kinematics.html#omnidirectional-wheeled-mobile-robots
    """
    forwards = {"basquiat": 30.0, "basquiat+": 30, "calder": 30.0}
    gamma = forwards[forward_dir]
    h0 = np.array(
        [
            [np.sin(np.radians(gamma)), -np.cos(np.radians(gamma)), -base_radius],
            [np.sin(np.radians(gamma + 120.0)), -np.cos(np.radians(gamma + 120.0)), -base_radius],
            [np.sin(np.radians(gamma + 240.0)), -np.cos(np.radians(gamma + 240.0)), -base_radius],
        ]
    )
    _r = wheel_dia_m / 2
    h0 = h0 * (1 / _r)
    return h0


def inverse_3x3_matrix(matrix):
    if matrix.shape != (3, 3):
        raise ValueError("Input must be a 3x3 matrix.")
    determinant = np.linalg.det(matrix)
    if determinant == 0:
        raise ValueError("Matrix is singular and cannot be inverted.")
    return np.linalg.inv(matrix)


def rotation_3x3_matrix(theta):
    return np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    )


def diff_drive_fwd_kinematics(w_left: float, w_right: float) -> tuple:
    """
    Calculate the linear and angular velocity of a differential drive robot.
    """
    wheel_diameter = config.robot_settings["wheel_diameter"]
    wheel_separation = config.robot_settings["wheel_separation"]
    R = wheel_diameter / 2
    L = wheel_separation
    if R <= 0:
        raise ValueError("Radius must be greater than zero.")
    if L <= 0:
        raise ValueError("Distance between wheels must be greater than zero.")

    # Linear velocity (V) is the average of the linear velocities of the two wheels
    V = R * (w_left + w_right) / 2.0

    # Angular velocity (omega) is the difference in linear velocities divided by the distance
    # between the wheels
    omega = R * (w_right - w_left) / L

    return (V, omega)


def diff_drive_inv_kinematics(V: float, omega: float) -> tuple:
    """
    Calculate the rotational velocities of the left and right wheels for a
    differential drive robot.
    """
    wheel_diameter = config.robot_settings["wheel_diameter"]
    wheel_separation = config.robot_settings["wheel_separation"]
    R = wheel_diameter / 2
    L = wheel_separation
    if R <= 0:
        raise ValueError("Radius must be greater than zero.")
    if L <= 0:
        raise ValueError("Distance between wheels must be greater than zero.")

    # Calculate the rotational velocities of the wheels
    w_left = (V - (omega * L / 2)) / R
    w_right = (V + (omega * L / 2)) / R

    return (w_left, w_right)


class FpsCounter:
    def __init__(self):
        self._fps_counter = 0
        self._wall_time = time.perf_counter()

        self.fps = 0
        """The actual fps count"""

        self.sim_to_real_ratio: float | None = None
        """Sim time compared with real time"""

        self._last_sim_time = 0

    def tick(self, sim_time: float | None = None):
        """
        Call this during step() to update the fps counter.

        Pass sim_time to calculate sim-to-real time.
        """
        self._fps_counter += 1

        elapsed = time.perf_counter() - self._wall_time
        # When one second has passed, count:
        if elapsed >= 1.0:
            new_wall_time = time.perf_counter()

            if sim_time:
                self.sim_to_real_ratio = (sim_time - self._last_sim_time) / (
                    new_wall_time - self._wall_time
                )
                self._last_sim_time = sim_time

            self.fps = self._fps_counter / elapsed
            self._wall_time = new_wall_time
            self._fps_counter = 0

    @property
    def sim_to_real_time_ratio_msg(self):
        if self.sim_to_real_ratio is None:
            return "sim_to_real_ratio is not set. Call `tick(sim_time=)` with the sim_time to calculate it."
        return f"Sim is running {self.sim_to_real_ratio:.3f}x as fast as realtime"


class URDFmodel:
    def __init__(self, urdf_file_path: str) -> None:
        """
        Load URDF model
        """
        self.urdf = urdf_loader.URDF.load(urdf_file_path, load_meshes=False)
        self.joint_names = []

        # Copy over joint names that exist in this URDF, excluding
        #  * wheels
        #  * the right finger
        #  * head joints if they are not present
        # Also ensures they are in the same order regardless of URDF parsing
        for joint_name in [
            "joint_wrist_yaw",
            "joint_wrist_pitch",
            "joint_wrist_roll",
            "joint_lift",
            "joint_arm_l0",
            "joint_arm_l1",
            "joint_arm_l2",
            "joint_arm_l3",
            "joint_gripper_finger_left",
            "joint_head_pan",
            "joint_head_tilt",
            
            "wrist_yaw_joint",
            "wrist_pitch_joint",
            "wrist_roll_joint",
            "lift_joint",
            "arm_l4_joint",
            "arm_l3_joint",
            "arm_l2_joint",
            "arm_l1_joint",
            "gripper_finger_left_joint",
            "head_pan_joint",
            "head_tilt_joint",
        ]:
            if joint_name in self.urdf.joint_map:
                self.joint_names.append(joint_name)

    def get_transform(self, cfg: dict, link_name: str) -> np.ndarray:
        """
        Get transformation matrix of the link w.r.t. the base_link
        """
        lk_cfg = {
            "joint_wrist_yaw": cfg["wrist_yaw"],
            "joint_wrist_pitch": cfg["wrist_pitch"],
            "joint_wrist_roll": cfg["wrist_roll"],
            "joint_lift": cfg["lift"],
            "joint_arm_l0": cfg["arm"] / 4,
            "joint_arm_l1": cfg["arm"] / 4,
            "joint_arm_l2": cfg["arm"] / 4,
            "joint_arm_l3": cfg["arm"] / 4,
            
            "wrist_yaw_joint": cfg["wrist_yaw"],
            "wrist_pitch_joint": cfg["wrist_pitch"],
            "wrist_roll_joint": cfg["wrist_roll"],
            "lift_joint": cfg["lift"],
            "arm_l4_joint": cfg["arm"] / 4,
            "arm_l3_joint": cfg["arm"] / 4,
            "arm_l2_joint": cfg["arm"] / 4,
            "arm_l1_joint": cfg["arm"] / 4,
        }
        if (
            "joint_head_pan" in cfg.keys()
        ):  # Assumes is pan is available, joint_head_tilt is also available.
            lk_cfg["joint_head_pan"] = cfg["head_pan"]
            lk_cfg["joint_head_tilt"] = cfg["head_tilt"]
        if "head_pan_joint" in cfg.keys():
            lk_cfg["head_pan_joint"] = cfg["head_pan"]
            lk_cfg["head_tilt_joint"] = cfg["head_tilt"]
        
        if "gripper" in cfg.keys():
            lk_cfg["joint_gripper_finger_left"] = cfg["gripper"]
            lk_cfg["joint_gripper_finger_right"] = cfg["gripper"]
            lk_cfg["gripper_finger_left_joint"] = cfg["gripper"]
            lk_cfg["gripper_finger_right_joint"] = cfg["gripper"]
            
        valid_lk_cfg = {k: v for k, v in lk_cfg.items() if k in self.urdf.joint_map}
        self.urdf.update_cfg(valid_lk_cfg)
        return self.urdf.get_transform(link_name, self.urdf.base_link)


def replace_xml_tag_value(xml_str: str, tag: str, attribute: str, pattern: str, value: str) -> str:
    """
    Replace value of a specific tag in an XML string
    Args:
        xml_str: XML string
        tag: Tag name
        attribute: Attribute name
        pattern: Pattern to match
        value: Value to replace with
    Returns:
        str: Modified XML string
    """
    root = ET.fromstring(xml_str)
    tree = ET.ElementTree(root)
    for elem in tree.iter(tag):
        if attribute in elem.attrib.keys():
            if pattern == elem.attrib[attribute]:
                elem.attrib[attribute] = value
    return ET.tostring(root, encoding="unicode")


def xml_remove_subelement(xml_str: str, subelement: str) -> str:
    """
    Remove actuator subelement from MuJoCo XML string
    Args:
        xml_str: MuJoCo XML string
    Returns:
        str: Modified MuJoCo XML string
    """
    root = ET.fromstring(xml_str)
    tree = ET.ElementTree(root)
    for elem in tree.iter(subelement):
        root.remove(elem)
    return ET.tostring(root, encoding="unicode")


def xml_add_floor_geom(xml_str: str) -> str:
    """
    Add a floor geom to the worldbody of a MuJoCo XML string
    Args:
        xml_str: MuJoCo XML string
    Returns:
        str: Modified MuJoCo XML string
    """
    root = ET.fromstring(xml_str)
    worldbody = root.find("worldbody")
    if worldbody is not None:
        floor_exists = False
        for geom in worldbody.findall("geom"):
            if geom.get("name") == "floor":
                floor_exists = True
                break
        if not floor_exists:
            ET.SubElement(
                worldbody,
                "geom",
                {
                    "name": "floor",
                    "type": "plane",
                    "pos": "0 0 0",
                    "size": "10 10 0.02",
                    "rgba": "0 0 0 0",
                    "contype": "1",
                    "conaffinity": "1",
                },
            )
    return ET.tostring(root, encoding="unicode")


def xml_remove_tag_by_name(xml_string: str, tag: str, name: str) -> Tuple[str, dict | None]:
    """
    Remove a subelement from an XML string with a specified tag and name attribute
    """
    # Parse the XML string into an ElementTree
    root = ET.fromstring(xml_string)

    # Find the parent element of the subelement to be removed
    parent_map = {c: p for p in root.iter() for c in p}

    removed_body_attrib = None
    # Iterate through the subelements to find the one with the specified tag and name attribute
    for elem in root.iter(tag):
        if elem.get("name") == name:
            # Remove the matching subelement
            removed_body_attrib = elem.attrib
            parent = parent_map[elem]
            parent.remove(elem)
            break

    # Convert the ElementTree back to a string
    return ET.tostring(root, encoding="unicode"), removed_body_attrib


def xml_modify_body_pos(
    xml_string: str, tag: str, name: str, pos: np.ndarray, quat: np.ndarray
) -> str:
    """
    Modify the position attribute of a tag with a specified name
    """
    # Parse the XML string into an ElementTree
    root = ET.fromstring(xml_string)
    # Find the element with body tag and name attribute
    for elem in root.iter(tag):
        if elem.get("name") == name:
            elem.set("pos", " ".join(map(str, pos)))
            elem.set("quat", " ".join(map(str, quat)))
    return ET.tostring(root, encoding="unicode")


def insert_line_after_mujoco_tag(xml_string: str, line_to_insert: str) -> str:
    """
    Insert a new line after the mujoco tag in the XML string
    """
    # Define the pattern to match the mujoco tag
    pattern = r'(<mujoco\s+model="base"\s*>)'

    # Use re.sub to insert the new line after the matched tag
    modified_xml = re.sub(pattern, f"\\1\n    {line_to_insert}", xml_string, count=1)

    return modified_xml


def get_absolute_path_stretch_xml(xml_file: str, robot_pose_attrib: dict | None = None) -> str:
    """
    Generates Robot XML with absolute path to files
    Args:
        xml_file: Path to starting xml file
        robot_pose_attrib: Robot pose attributes in form {"pos": "x y z", "quat": "x y z w"}
    Returns:
        str: Path to the generated XML file
    """
    print("DEFAULT XML: {}".format(xml_file))

    with open(xml_file, "r") as f:
        default_robot_xml = f.read()

    parent_folder = pathlib.Path(xml_file).parent
    assets_folder = models_path / "assets"

    default_robot_xml = re.sub(
        'assetdir="assets"', f'assetdir="{assets_folder}"', default_robot_xml
    )

    # find all the line which has the pattern {file="something.type"}
    # and replace the file path with the absolute path
    pattern = r'file="(.+?)"'
    for match in re.finditer(pattern, default_robot_xml):
        file_path = match.group(1)

        for folder_option in [parent_folder, models_path, assets_folder]:
            abs_path = folder_option / file_path
            if abs_path.exists():
                default_robot_xml = default_robot_xml.replace(file_path, str(abs_path))
                break

    if robot_pose_attrib is not None:
        pos = f'pos="{robot_pose_attrib["pos"]}" quat="{robot_pose_attrib["quat"]}"'
        default_robot_xml = re.sub(
            '<body name="base_link" childclass="stretch">',
            f'<body name="base_link" childclass="stretch" {pos}>',
            default_robot_xml,
        )

    # Absosolute path converted streth xml
    output_path = models_path / "stretch_temp_abs.xml"
    with open(output_path, "w") as f:
        f.write(default_robot_xml)
    print("Saving temp abs path xml: {}".format(output_path))
    return str(output_path)


def map_between_ranges(
    value: float, from_min_max: Tuple[float, float], to_min_max: Tuple[float, float]
) -> float:
    """
    Map a value from one range to another
    """
    return (value - from_min_max[0]) * (to_min_max[1] - to_min_max[0]) / (
        from_min_max[1] - from_min_max[0]
    ) + to_min_max[0]


def get_depth_color_map(depth_image, clor_map=cv2.COLORMAP_JET):
    """
    Get depth color map
    """
    depth_min = np.min(depth_image)
    depth_max = np.max(depth_image)

    normalized_depth = (depth_image - depth_min) / (depth_max - depth_min)
    depth_8bit = ((1 - normalized_depth) * 255).astype(np.uint8)
    depth_8bit = cv2.applyColorMap(depth_8bit, clor_map)
    return depth_8bit


def dataclass_from_dict(klass, dict_data: dict):
    # references https://stackoverflow.com/a/54769644
    try:
        fieldtypes = {f.name: f.type for f in dataclasses.fields(klass)}
        return klass(**{f: dataclass_from_dict(fieldtypes[f], dict_data[f]) for f in dict_data})
    except:
        return dict_data  # Not a dataclass field


def block_until_check_succeeds(
    wait_timeout: float | None, check: Callable[[], bool], is_alive: Callable[[], bool], time_fn: Callable[[], float] | None = None
) -> bool:
    """Blocks until the check callback succeeds"""

    if wait_timeout is None:
        while is_alive():
            if check():
                return True
        return False

    t_fn = time_fn if time_fn is not None else time.time
    start_time = t_fn()

    while t_fn() - start_time < wait_timeout:
        if not is_alive():
            return False
        if check():
            return True
        time.sleep(0.01)

    return False


def switch_to_glfw_renderer(mjmodel: MjModel, renderer: mujoco.Renderer):
    """
    On Darwin, the default renderer in `mujoco/gl_context.py` is CGL, which is not compatible with offscreen rendering.

    This function frees the initial display context and creates a new one with GLFW.
    """
    if renderer._gl_context:
        renderer._gl_context.free()
    if renderer._mjr_context:
        renderer._mjr_context.free()

    renderer._gl_context = GlFwContext(480, 640)

    renderer._gl_context.make_current()

    renderer._mjr_context = mujoco._render.MjrContext(
        mjmodel, mujoco._enums.mjtFontScale.mjFONTSCALE_150.value
    )
    mujoco._render.mjr_setBuffer(
        mujoco._enums.mjtFramebuffer.mjFB_OFFSCREEN.value, renderer._mjr_context
    )
    renderer._mjr_context.readDepthMap = mujoco._enums.mjtDepthMap.mjDEPTH_ZEROFAR


def change_start_pose(
    model: MjModel,
    name: str,
    translation: list | None = None,
    rotation_quat: list | None = None,
):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    if body_id == -1:
        raise ValueError(f"Body '{name}' not found in the MjModel.")

    joint_id = -1
    for j in range(model.njnt):
        if model.jnt_bodyid[j] == body_id:
            joint_id = j
            break

    if joint_id == -1:
        raise ValueError(f"Body '{name}' does not have any joints attached. Cannot change start pose.")

    # Since the model has a Free Joint, we must change the default QPOS (qpos0).
    qadr = model.jnt_qposadr[joint_id]

    print(
        "Changing start pose of body '{}' from qpos0 '{}' to '{}' and '{}'".format(
            name, model.qpos0[qadr : qadr + 7], translation, rotation_quat
        )
    )
    if translation is not None:
        model.qpos0[qadr : qadr + 2] = translation[:2]
        model.qpos0[qadr + 2] += translation[2]

    if rotation_quat is not None:
        model.qpos0[qadr + 3 : qadr + 7] = rotation_quat


try:
    # Only Python >12 has override.
    override = __import__("typing").override
except:  # noqa

    def override(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper


FISHEYE_MIN_VALID_FRACTION = 0.85
"""How much of a distorted frame has to be real pixels, after cropping.

The pinhole render a fisheye is warped out of cannot cover the whole fisheye
frame: a 123-degree lens maps its outer ring to angles a pinhole camera would
have to put at infinity, so `_get_fisheye_remap_matrices` marks roughly 45% of
the raw output invalid and it comes back black. That black is not a lens
vignette, it is missing source data, and it survives every downstream resize --
`INTER_AREA` averages it into the surviving pixels rather than dropping it.

So the distortion crops to a window that is mostly real pixels before it
resamples. 0.85 keeps the curved corners a real lens actually vignettes (about
15% of the frame) while cutting the empty surround, at roughly a 1.3x zoom into
the frame. Raising it towards 1.0 trades field of view for a frame with no black
in it at all; lowering it keeps more of the wide view and more of the void.
"""


def _fisheye_source_maps(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    distortion_params: tuple,
    height: int,
    width: int,
    fov_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distorted-frame coordinates -> the pinhole-render coordinates they sample.

    `grid_x`/`grid_y` are positions in the full-size distorted frame, so a crop
    is expressed by handing in a grid that spans only part of it.
    """
    x_d = (grid_x - cx) / fx
    y_d = (grid_y - cy) / fy
    r_d = np.sqrt(x_d**2 + y_d**2)
    r_d_safe = np.maximum(r_d, 1e-8)

    theta_d = r_d
    if len(distortion_params) >= 4:
        k1, k2, k3, k4 = distortion_params[:4]
        theta_d2 = theta_d**2
        distortion_factor = 1.0 + theta_d2 * (k1 + theta_d2 * (k2 + theta_d2 * (k3 + theta_d2 * k4)))
        theta = theta_d * distortion_factor
    else:
        theta = theta_d

    # Tangent mapping from fisheye ray angle theta to pinhole image plane radius r_p
    fovy_rad = np.radians(fov_deg)
    f_pinhole = (height / 2.0) / np.tan(fovy_rad / 2.0)

    r_p = np.tan(np.minimum(theta, np.radians(85.0)))

    scale_x = (f_pinhole / fx) * (r_p / r_d_safe)
    scale_y = (f_pinhole / fy) * (r_p / r_d_safe)

    map_x = cx + (grid_x - cx) * scale_x
    map_y = cy + (grid_y - cy) * scale_y

    valid_mask = (map_x >= 0) & (map_x < width) & (map_y >= 0) & (map_y < height) & (theta <= np.radians(85.0))

    map_x = np.where(valid_mask, map_x, -1.0).astype(np.float32)
    map_y = np.where(valid_mask, map_y, -1.0).astype(np.float32)

    return map_x, map_y, valid_mask


def _best_window(integral: np.ndarray, rect_height: int, rect_width: int) -> tuple[int, int, float]:
    """The `rect_width` x `rect_height` window holding the most valid pixels.

    `integral` is a summed-area table of the validity mask, so every candidate
    position is four lookups and the whole sweep is one array expression.
    Returns `(x, y, valid_fraction)`; ties go to the most central window, which
    keeps the crop from sliding around between two near-identical frames.
    """
    counts = (
        integral[rect_height:, rect_width:]
        - integral[:-rect_height, rect_width:]
        - integral[rect_height:, :-rect_width]
        + integral[:-rect_height, :-rect_width]
    )
    best = counts.max()
    ys, xs = np.nonzero(counts == best)
    middle = len(ys) // 2
    return int(xs[middle]), int(ys[middle]), float(best) / (rect_height * rect_width)


def _largest_valid_crop(
    valid_mask: np.ndarray, min_valid_fraction: float
) -> tuple[int, int, int, int]:
    """The biggest crop of `valid_mask`'s shape and aspect that is mostly valid.

    Returns `(x, y, width, height)`. Binary search over the crop height: a
    window that grows can only reach further into the void around the image
    circle, so its valid fraction falls monotonically and the largest window
    still meeting `min_valid_fraction` is what the search converges on.
    """
    height, width = valid_mask.shape
    aspect = width / float(height)

    integral = np.zeros((height + 1, width + 1), dtype=np.int64)
    np.cumsum(np.cumsum(valid_mask.astype(np.int64), axis=0), axis=1, out=integral[1:, 1:])

    full_frame = (0, 0, width, height)
    if valid_mask.mean() >= min_valid_fraction:
        return full_frame

    best = None
    low, high = 1, height
    while low <= high:
        candidate_height = (low + high) // 2
        candidate_width = int(round(candidate_height * aspect))
        if not 1 <= candidate_width <= width:
            high = candidate_height - 1
            continue
        x, y, fraction = _best_window(integral, candidate_height, candidate_width)
        if fraction >= min_valid_fraction:
            best = (x, y, candidate_width, candidate_height)
            low = candidate_height + 1
        else:
            high = candidate_height - 1

    # No window anywhere is clean enough -- an all-black frame, or a
    # min_valid_fraction the optics cannot meet. Cropping to a sliver of a frame
    # that has nothing in it would be worse than leaving it alone.
    return best or full_frame


@lru_cache(maxsize=16)
def _get_fisheye_remap_matrices(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    distortion_params: tuple,
    width: int,
    height: int,
    fov_deg: float = 123.39,
    min_valid_fraction: float = FISHEYE_MIN_VALID_FRACTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    crop_x, crop_y, crop_width, crop_height = get_fisheye_crop_rect(
        fx, fy, cx, cy, distortion_params, width, height, fov_deg, min_valid_fraction
    )
    if (crop_x, crop_y, crop_width, crop_height) == (0, 0, width, height):
        grid_y, grid_x = np.indices((height, width), dtype=np.float32)
        return _fisheye_source_maps(
            grid_x, grid_y, fx, fy, cx, cy, distortion_params, height, width, fov_deg
        )

    # Derive the maps on a grid that spans only the crop, rather than
    # remapping and then resizing the result: the crop is a resample of the
    # pinhole render either way, and going straight there costs one
    # interpolation instead of two.
    crop_grid_x = np.linspace(
        crop_x + 0.5 * crop_width / width - 0.5,
        crop_x + crop_width - 0.5 * crop_width / width - 0.5,
        width,
        dtype=np.float32,
    )
    crop_grid_y = np.linspace(
        crop_y + 0.5 * crop_height / height - 0.5,
        crop_y + crop_height - 0.5 * crop_height / height - 0.5,
        height,
        dtype=np.float32,
    )
    grid_x, grid_y = np.meshgrid(crop_grid_x, crop_grid_y)

    return _fisheye_source_maps(
        grid_x, grid_y, fx, fy, cx, cy, distortion_params, height, width, fov_deg
    )


@lru_cache(maxsize=16)
def get_fisheye_crop_rect(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    distortion_params: tuple,
    width: int,
    height: int,
    fov_deg: float = 123.39,
    min_valid_fraction: float = FISHEYE_MIN_VALID_FRACTION,
) -> tuple[int, int, int, int]:
    """`(x, y, width, height)` of the window `apply_fisheye_distortion` crops to.

    The distorted frame keeps the size it came in at, so this is what says how
    much of the original view is still in it -- `width / crop_width` is the zoom
    the crop applied, which is the factor the frame's focal length grew by.
    """
    grid_y, grid_x = np.indices((height, width), dtype=np.float32)
    _, _, valid_mask = _fisheye_source_maps(
        grid_x, grid_y, fx, fy, cx, cy, distortion_params, height, width, fov_deg
    )
    return _largest_valid_crop(valid_mask, min_valid_fraction)


def apply_fisheye_distortion(
    image: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    distortion_params: tuple,
    fov_deg: float = 123.39,
) -> np.ndarray:
    """
    Applies fisheye lens distortion to a pinhole image using cached remap matrices.

    The result is cropped to the part of the frame the pinhole render actually
    covers (see `FISHEYE_MIN_VALID_FRACTION`) and returned at the input size, so
    a caller that resizes it afterwards is scaling image rather than void.
    """
    height, width = image.shape[:2]
    map_x, map_y, valid_mask = _get_fisheye_remap_matrices(
        fx, fy, cx, cy, tuple(distortion_params), width, height, fov_deg
    )
    distorted = cv2.remap(
        image,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    distorted[~valid_mask] = 0
    return distorted
