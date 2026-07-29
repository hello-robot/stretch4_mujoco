import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import yourdfpy

from urdf2mjcf.convert import convert_urdf_to_mjcf
from urdf2mjcf.model import ConversionMetadata


def generate_mjcf(urdf_path: str, out_mjcf_path: str=None):
    """
    Generates a Stretch 4 MJCF from the provided URDF using urdf2mjcf and programmatic DOM manipulations.
    Note, this function only works for stretch 4 urdf -> mjcf, and is based on a manual process that used to happen to convert a URDF to an mjcf.
    """
    
    print(f"Generating mjcf from {urdf_path}")
    urdf_path = Path(urdf_path)
    if out_mjcf_path is None: 
        out_mjcf_path = urdf_path.with_suffix(".mjcf")
    else: 
        out_mjcf_path = Path(out_mjcf_path)

    # 1. Update default settings

    metadata = ConversionMetadata(
        # Adjust the height so the wheels do not start below the floor (measure from the base_link)
        height_offset=0.056,
        # Allow the root link to be "unwelded"
        freejoint=True,
        # Do not anchor the root link to a surface
        floating_base=True,
        # Avoid the default "front_camera" and "side_camera" artifacts
        cameras=[],
    )

    convert_urdf_to_mjcf(urdf_path, out_mjcf_path, metadata=metadata)

    # 2. Parse the raw MJCF
    tree = ET.parse(out_mjcf_path)
    root = tree.getroot()

    # Create a new root holding only asset and worldbody
    new_root = ET.Element("mujoco", model="stretch")
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("urdf2mjcf output is missing 'asset' or 'worldbody' tags.")

    new_root.append(asset)
    new_root.append(worldbody)

    # Helper function to find a body by name
    def find_body(name):
        return worldbody.find(f".//body[@name='{name}']")

    # 3. Clean up mesh names (remove .STL suffix) and update geom classes
    for mesh in asset.findall("mesh"):
        name = mesh.get("name")
        if name and name.endswith(".STL"):
            mesh.set("name", name[:-4])

        file_attr = mesh.get("file")
        if file_attr and file_attr.startswith("file://"):
            mesh.set("file", file_attr[7:])

    for mat in asset.findall("material"):
        if mat.get("name") == "":
            asset.remove(mat)

    for geom in worldbody.findall(".//geom"):
        mesh_attr = geom.get("mesh")
        if mesh_attr and mesh_attr.endswith(".STL"):
            geom.set("mesh", mesh_attr[:-4])

        geom_class = geom.get("class")
        if geom_class == "visual":
            geom.set("class", "visualgeom")
            geom.set("density", "0")
            geom.set("group", "1")

            mesh_name = geom.get("mesh", "")
            if "aruco" in mesh_name:
                geom.set("rgba", "1 1 1 1")  # Pure white
            elif "arm_l4" in mesh_name or "finger_" in mesh_name:
                geom.set("rgba", "0.9 0.9 0.88 1")  # Off-white / Cream
            elif any(
                part in mesh_name
                for part in [
                    "wrist",
                    "gripper",
                    "tool",
                    "quick_connect",
                    "grasp",
                    "camera",
                    "lidar",
                    "sensor",
                    "wheel",
                ]
            ):
                geom.set("rgba", "0.15 0.15 0.15 1")  # Dark gray / black
            elif any(part in mesh_name for part in ["mast", "arm"]):
                geom.set("rgba", "0.5 0.5 0.5 1")  # Metallic gray
            elif mesh_name:
                geom.set("rgba", "0.9 0.9 0.88 1")  # Off-white / Cream
        elif geom_class == "collision":
            geom_name = geom.get("name", "")
            if geom_name in [
                "grasp_center_collision_link",
                "quick_connect_interface_collision_link",
                "tool_attachment_site_collision_link",
            ] or geom_name.startswith("aruco__link"):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
            elif geom_name in [
                "gripper_fingertip_left_collision_link",
                "gripper_fingertip_right_collision_link",
            ]:
                geom.set("class", "rubber")

        if "material" in geom.attrib and geom.attrib["material"] == "":
            del geom.attrib["material"]


    # 4. Rename root body to stretch4 and update its geoms
    base_body = worldbody.find("body")
    if base_body is not None:
        if base_body.get("name") != "stretch4":
            base_body.set("name", "stretch4")
        if "childclass" in base_body.attrib:
            del base_body.attrib["childclass"]
            
        ET.SubElement(base_body, "site", name="imu", size="0.01", pos="0 0 0")

    # 5. Replace wheels with omniwheels and include omniwheels.xml
    parent_map = {c: p for p in worldbody.iter() for c in p}
    for old_name in ["wheel_0_link", "wheel_1_link", "wheel_2_link"]:
        w_body = find_body(old_name)
        if w_body is not None:
            p = parent_map.get(w_body)
            if p is not None:
                p.remove(w_body)

    if base_body is not None:
        ET.SubElement(base_body, "include", file="omniwheels.xml")

    # 6. Add lidars, cameras, and laser dynamically from URDF
    urdf = yourdfpy.URDF.load(urdf_path)

    def get_cam_pos_quat(link_name, parent_name, post_rotation:np.ndarray|None = None):
        T_parent_link = urdf.get_transform(link_name, parent_name)
        T_parent_cam = T_parent_link
        if post_rotation is not None:
            T_parent_cam = T_parent_link @ post_rotation

        pos = T_parent_cam[:3, 3]
        quat = Rotation.from_matrix(T_parent_cam[:3, :3]).as_quat() # xyzw
        # MuJoCo quat format: w x y z
        quat_mj = f"{quat[3]} {quat[0]} {quat[1]} {quat[2]}"
        pos_mj = f"{pos[0]} {pos[1]} {pos[2]}"
        return pos_mj, quat_mj


    head = find_body("head_link")
    T_opt_cam = np.eye(4)
    # Note that Mujoco's camera frame coordinate system is different from ROS's: " Forward corresponds to the negative Z axis of the camera frame, while up corresponds to the positive Y axis. " , so in creating the Camera lement in the mjcf, we rotate by 180 degrees. https://mujoco.readthedocs.io/en/stable/programming/visualization.html
    T_opt_cam[:3, :3] = Rotation.from_euler('xyz', [180, 0, 0], degrees=True).as_matrix()
    
    if head is not None:
        # Hesai J128 Lidars
        lidar_left_body = find_body("lidar_left_link")
        if lidar_left_body is not None:
            ET.SubElement(lidar_left_body, "site", name="lidar_left", pos="0 0 0", quat="1 0 0 0")
        else:
            pos_l, quat_l = get_cam_pos_quat("lidar_left_link", "head_link")
            ET.SubElement(head, "site", name="lidar_left", pos=pos_l, quat=quat_l)

        lidar_right_body = find_body("lidar_right_link")
        if lidar_right_body is not None:
            ET.SubElement(lidar_right_body, "site", name="lidar_right", pos="0 0 0", quat="1 0 0 0")
        else:
            pos_r, quat_r = get_cam_pos_quat("lidar_right_link", "head_link")
            ET.SubElement(head, "site", name="lidar_right", pos=pos_r, quat=quat_r)

        # Wide angle cameras
        # Right Camera
        cam_r_body = find_body("camera_right_optical_link")
        if cam_r_body is not None:
            pos_r, quat_r = get_cam_pos_quat("camera_right_optical_link", "camera_right_optical_link", post_rotation=T_opt_cam)
            ET.SubElement(cam_r_body, "camera", name="camera_right_link", pos=pos_r, quat=quat_r)

        # Left Camera
        cam_l_body = find_body("camera_left_optical_link")
        if cam_l_body is not None:
            pos_l, quat_l = get_cam_pos_quat("camera_left_optical_link", "camera_left_optical_link", post_rotation=T_opt_cam)
            ET.SubElement(cam_l_body, "camera", name="camera_left_link", pos=pos_l, quat=quat_l)

        # Center Camera
        cam_c_body = find_body("camera_center_optical_link")
        if cam_c_body is not None:
            pos_c, quat_c = get_cam_pos_quat("camera_center_optical_link", "camera_center_optical_link", post_rotation=T_opt_cam)
            ET.SubElement(cam_c_body, "camera", name="camera_center_link", pos=pos_c, quat=quat_c)

    if base_body is not None:
        old_laser = worldbody.find(".//body[@name='laser']")
        if old_laser is not None:
            parent_map = {c: p for p in worldbody.iter() for c in p}
            p = parent_map.get(old_laser)
            if p is not None:
                p.remove(old_laser)

        laser = ET.SubElement(base_body, "body", name="laser", pos="0.0 0.0 0.2", quat="0 0 0 1")
        rep = ET.SubElement(laser, "replicate", count="360", euler="0 0 0.0174533")
        ET.SubElement(rep, "site", name="lidar", zaxis="1 0 0")

    # 7. Encapsulate gripper camera in gripper_camera_link
    gripper_cam = find_body("gripper_camera_link")
    if gripper_cam is not None:
        pos_g_rgb, quat_g_rgb = get_cam_pos_quat("gripper_left_camera_color_optical_frame", "gripper_camera_link", post_rotation=T_opt_cam)
        ET.SubElement(gripper_cam, "camera", name="gripper_camera_left_rgb", pos=pos_g_rgb, quat=quat_g_rgb)
        pos_g_rgb, quat_g_rgb = get_cam_pos_quat("gripper_right_camera_color_optical_frame", "gripper_camera_link", post_rotation=T_opt_cam)
        ET.SubElement(gripper_cam, "camera", name="gripper_camera_right_rgb", pos=pos_g_rgb, quat=quat_g_rgb)
        pos_g_depth, quat_g_depth = get_cam_pos_quat("gripper_stereo_camera_color_optical_frame", "gripper_camera_link", post_rotation=T_opt_cam)
        ET.SubElement(gripper_cam, "camera", name="gripper_camera_stereo_depth", pos=pos_g_depth, quat=quat_g_depth)

    # 8. Modify Mast to use appropriate collision mass
    mast = find_body("mast_link")
    if mast is not None:
        for geom in mast.findall("geom"):
            mast.remove(geom)
        ET.SubElement(mast, "geom", type="mesh", mesh="mast_link", **{"class": "visualgeom"})
        ET.SubElement(
            mast, "geom", mesh="mast_collision_link", **{"class": "collision"}, mass="4.25"
        )

    # 9. Update arm_l0_link if needed and add gravcomp to lift, arm, wrist, gripper bodies
    for b in worldbody.findall(".//body"):
        name = b.get("name", "")
        if "lift_link" in name or "arm_link" in name or "wrist_link" in name or "gripper" in name:
            b.set("gravcomp", "1")

    # 10. Encapsulate gripper fingers into gripper_slider_link
    f_right = find_body("gripper_finger_right_link")
    f_left = find_body("gripper_finger_left_link")
    if f_right is not None and f_left is not None:
        parent_map = {c: p for p in worldbody.iter() for c in p}
        p = parent_map.get(f_right)
        if p is not None:
            slider = ET.Element("body", name="gripper_slider_link", pos="0 0 0", gravcomp="1")
            ET.SubElement(
                slider,
                "geom",
                type="box",
                size=".03 .005 .005",
                mass=".05",
                **{"class": "visualgeom"},
                rgba="0 0 0 0",
            )
            p.remove(f_right)
            p.remove(f_left)
            slider.append(f_right)
            slider.append(f_left)
            p.append(slider)

    # 11. Add "rubber" class to gripper fingers and fix thin-shell mesh issues
    for geom in worldbody.findall(".//geom"):
        mesh = geom.get("mesh", "")
        if "gripper_finger" in mesh or "gripper_fingertip" in mesh:
            if geom.get("class") != "visualgeom":
                geom.set("class", "rubber")

    # Add compliant passive joints to fingertips to allow surface alignment
    for body in worldbody.findall(".//body"):
        name = body.get("name", "")
        if name in ["gripper_fingertip_right_link", "gripper_fingertip_left_link"]:
            ET.SubElement(
                body,
                "joint",
                name=f"{name}_compliant_x",
                type="hinge",
                axis="1 0 0",
                stiffness="0.1",
                damping="0.002",
                springref="0",
                limited="true",
                range="-0.15 0.15",
            )
            ET.SubElement(
                body,
                "joint",
                name=f"{name}_compliant_y",
                type="hinge",
                axis="0 1 0",
                stiffness="0.1",
                damping="0.002",
                springref="0",
                limited="true",
                range="-0.15 0.15",
            )

    # 12. Update Joint Classes
    for j in worldbody.findall(".//joint"):
        name = j.get("name", "")
        if "lift_joint" in name:
            j.set("class", "lift_stretch4")
        elif "arm_l" in name and "_joint" in name:
            j.set("class", "telescope")
        elif "wrist_yaw_joint" in name:
            j.set("class", "wrist_yaw_stretch4")
        elif "wrist_pitch_joint" in name:
            j.set("class", "wrist_pitch_stretch4")
        elif "wrist_roll_joint" in name:
            j.set("class", "wrist_roll_stretch4")
        elif "gripper_finger_left_joint" in name:
            j.set("class", "gripper_left_finger")
        elif "gripper_finger_right_joint" in name:
            j.set("class", "gripper_right_finger")
        else:
            # urdf2mjcf assigns 'motor' by default. If it wasn't remapped above, it's invalid.
            if j.get("class") == "motor":
                raise ValueError(f"Unexpected joint '{name}' retains invalid default class 'motor'. Please add a dedicated class mapping in mjcf_generator.py")

    # Clean up the generated string and save
    ET.indent(new_root, space="  ", level=0)
    tree = ET.ElementTree(new_root)
    tree.write(out_mjcf_path, encoding="unicode", xml_declaration=True)

    # Inject ctrlrange into actuator_sensor.xml based on the extracted joint ranges
    try:
        import os

        actuator_xml_path = os.path.join(os.path.dirname(out_mjcf_path), "actuator_sensor.xml")
        out_actuator_xml_path = os.path.join(
            os.path.dirname(out_mjcf_path), "autogenerated_actuator_sensor.xml"
        )
        if os.path.exists(actuator_xml_path):
            act_tree = ET.parse(actuator_xml_path)
            act_root = act_tree.getroot()

            ranges_joint = {}
            for j in worldbody.findall(".//joint"):
                name = j.get("name")
                r = j.get("range")
                if name and r:
                    ranges_joint[name] = r

            # The arm tendon "extend" represents the sum of 4 telescope joints
            if "arm_l4_joint" in ranges_joint:
                try:
                    l_min, l_max = map(float, ranges_joint["arm_l4_joint"].split())
                    ranges_joint["tendon_extend"] = f"{l_min * 4} {l_max * 4}"
                except ValueError:
                    pass

            for pos in act_root.findall(".//position"):
                j_name = pos.get("joint")
                t_name = pos.get("tendon")

                r = None
                if j_name and j_name in ranges_joint:
                    r = ranges_joint[j_name]
                elif t_name == "extend" and "tendon_extend" in ranges_joint:
                    r = ranges_joint["tendon_extend"]

                if r:
                    pos.set("ctrlrange", r)
                    pos.set("ctrllimited", "true")

            ET.indent(act_root, space="  ", level=0)
            act_tree.write(out_actuator_xml_path, encoding="unicode", xml_declaration=True)
    except Exception as e:
        print(f"Failed to generate autogenerated_actuator_sensor.xml: {e}")

