import subprocess
import tempfile
import xml.etree.ElementTree as ET


def generate_mjcf(urdf_path: str, out_mjcf_path: str):
    """
    Generates a Stretch 4 MJCF from the provided URDF using urdf2mjcf and programmatic DOM manipulations.
    Note, this function only works for stretch 4 urdf -> mjcf, and is based on a manual process that used to happen to convert a URDF to an mjcf.
    """
    print(f"Using {urdf_path=}")
    # Create a temporary file for the raw urdf2mjcf output
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        raw_mjcf_path = tmp.name

    # 1. Run urdf2mjcf
    try:
        subprocess.run(
            ["urdf2mjcf", urdf_path, "--output", raw_mjcf_path], check=True, capture_output=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"urdf2mjcf failed: {e.stderr.decode()}")

    # 2. Parse the raw MJCF
    tree = ET.parse(raw_mjcf_path)
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
                "link_grasp_center_collision",
                "link_quick_connect_interface_collision",
                "link_tool_attachment_site_collision",
            ] or geom_name.startswith("link_aruco_"):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
            elif geom_name in [
                "link_gripper_fingertip_left_collision",
                "link_gripper_fingertip_right_collision",
            ]:
                geom.set("class", "rubber")

        if "material" in geom.attrib and geom.attrib["material"] == "":
            del geom.attrib["material"]

    # 4. Remove lights, ground, and cameras from worldbody
    for tag in ["light", "camera"]:
        for el in worldbody.findall(tag):
            worldbody.remove(el)
    for geom in worldbody.findall("geom"):
        if geom.get("name") == "ground" or geom.get("type") == "plane":
            worldbody.remove(geom)

    # 5. Rename root body to stretch4 and update its geoms
    base_body = worldbody.find("body")
    if base_body is not None:
        if base_body.get("name") != "stretch4":
            base_body.set("name", "stretch4")
        if "childclass" in base_body.attrib:
            del base_body.attrib["childclass"]

        # Shift the robot up slightly to avoid clipping wheels into the floor at spawn
        base_body.set("pos", "0 0 0.03")

        # Ensure freejoint exists
        if base_body.find("freejoint") is None:
            ET.SubElement(base_body, "freejoint")

        ET.SubElement(base_body, "site", name="imu", size="0.01", pos="0 0 0")

    # 6. Replace wheels with omniwheels and include omniwheels.xml
    parent_map = {c: p for p in worldbody.iter() for c in p}
    for old_name in ["link_wheel_0", "link_wheel_1", "link_wheel_2"]:
        w_body = find_body(old_name)
        if w_body is not None:
            p = parent_map.get(w_body)
            if p is not None:
                p.remove(w_body)

    if base_body is not None:
        ET.SubElement(base_body, "include", file="omniwheels.xml")

    # 7. Add lidars, cameras, and laser
    lidar_r = find_body("link_lidar_right")
    if lidar_r is not None:
        ET.SubElement(lidar_r, "include", file="hemisphere_lidar_cameras_right.xml")

    lidar_l = find_body("link_lidar_left")
    if lidar_l is not None:
        ET.SubElement(lidar_l, "include", file="hemisphere_lidar_cameras_left.xml")

    for cam in ["link_camera_right", "link_camera_left", "link_camera_center"]:
        cam_body = find_body(cam)
        if cam_body is not None:
            ET.SubElement(cam_body, "camera", name=cam, pos="0 0 0.015", euler="0 3.14 0")

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

    # 8. Encapsulate gripper camera in wrist
    wrist_roll = find_body("link_wrist_roll")
    if wrist_roll is not None:
        gripper = ET.SubElement(
            wrist_roll,
            "body",
            name="gripper_camera_mount",
            pos="0 0.0660509 -0.0253083",
            quat="0.5 0.5 0.5 -0.5",
        )
        ET.SubElement(gripper, "include", file="gripper_cameras.xml")

    # 9. Modify Mast to use appropriate collision mass
    mast = find_body("link_mast")
    if mast is not None:
        for geom in mast.findall("geom"):
            mast.remove(geom)
        ET.SubElement(mast, "geom", type="mesh", mesh="link_mast", **{"class": "visualgeom"})
        ET.SubElement(
            mast, "geom", mesh="link_mast_collision", **{"class": "collision"}, mass="4.25"
        )

    # 10. Update link_arm_l4 if needed and add gravcomp to lift, arm, wrist, gripper bodies
    for b in worldbody.findall(".//body"):
        name = b.get("name", "")
        if "link_lift" in name or "link_arm" in name or "link_wrist" in name or "gripper" in name:
            b.set("gravcomp", "1")

    # 11. Encapsulate gripper fingers into link_gripper_slider
    f_right = find_body("link_gripper_finger_right")
    f_left = find_body("link_gripper_finger_left")
    if f_right is not None and f_left is not None:
        parent_map = {c: p for p in worldbody.iter() for c in p}
        p = parent_map.get(f_right)
        if p is not None:
            slider = ET.Element("body", name="link_gripper_slider", pos="0 0 0", gravcomp="1")
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

    # 12. Add "rubber" class to gripper fingers and fix thin-shell mesh issues
    for geom in worldbody.findall(".//geom"):
        mesh = geom.get("mesh", "")
        if "gripper_finger" in mesh or "gripper_fingertip" in mesh:
            if geom.get("class") != "visualgeom":
                geom.set("class", "rubber")

                # # Replace complex gray finger meshes with robust solid primitive boxes to prevent objects from tunneling/getting stuck in the scissor mechanism
                # The are not placed correctly - press 1, 2, and 4 on the mujoco viewer to see them.
                # if mesh == "link_gripper_finger_right":
                #     geom.set("type", "box")
                #     geom.set("size", "0.035 0.005 0.015")
                #     geom.set("pos", "0.033 -0.007 0")
                #     if "mesh" in geom.attrib:
                #         del geom.attrib["mesh"]
                # elif mesh == "link_gripper_finger_left":
                #     geom.set("type", "box")
                #     geom.set("size", "0.035 0.005 0.015")
                #     geom.set("pos", "-0.033 -0.007 0")
                #     if "mesh" in geom.attrib:
                #         del geom.attrib["mesh"]

    # Add compliant passive joints to fingertips to allow surface alignment
    for body in worldbody.findall(".//body"):
        name = body.get("name", "")
        if name in ["link_gripper_fingertip_right", "link_gripper_fingertip_left"]:
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

    # 13. Update Joint Classes
    for j in worldbody.findall(".//joint"):
        name = j.get("name", "")
        if "joint_lift" in name:
            j.set("class", "lift_stretch4")
        elif "joint_arm_l" in name:
            j.set("class", "telescope")
        elif "joint_wrist_yaw" in name:
            j.set("class", "wrist_yaw_stretch4")
            # TODO: remove this temporary fix when the urdf is fixed to switch the negative z-axis direction in ../stretch4_urdf/stretch4_urdf urdf's for the wrist joints
            j.set("axis", "0 0 1")
            j.set("range", "-4.276 1.134")
        elif "joint_wrist_pitch" in name:
            j.set("class", "wrist_pitch_stretch4")
            # TODO: remove this temporary fix when the urdf is fixed to switch the negative z-axis direction in ../stretch4_urdf/stretch4_urdf urdf's for the wrist joints
            j.set("axis", "0 0 1")
            j.set("range", "-4.276 1.134")
        elif "joint_wrist_roll" in name:
            j.set("class", "wrist_roll_stretch4")
            # TODO: remove this temporary fix when the urdf is fixed to switch the negative z-axis direction in ../stretch4_urdf/stretch4_urdf urdf's for the wrist joints
            j.set("axis", "0 0 1")
            j.set("range", "-1.134 4.276")
        elif "joint_gripper_finger_left" in name:
            j.set("class", "gripper_left_finger")
        elif "joint_gripper_finger_right" in name:
            j.set("class", "gripper_right_finger")

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

            joint_ranges = {}
            for j in worldbody.findall(".//joint"):
                name = j.get("name")
                r = j.get("range")
                if name and r:
                    joint_ranges[name] = r

            # The arm tendon "extend" represents the sum of 4 telescope joints
            if "joint_arm_l0" in joint_ranges:
                try:
                    l_min, l_max = map(float, joint_ranges["joint_arm_l0"].split())
                    joint_ranges["tendon_extend"] = f"{l_min * 4} {l_max * 4}"
                except ValueError:
                    pass

            for pos in act_root.findall(".//position"):
                j_name = pos.get("joint")
                t_name = pos.get("tendon")

                r = None
                if j_name and j_name in joint_ranges:
                    r = joint_ranges[j_name]
                elif t_name == "extend" and "tendon_extend" in joint_ranges:
                    r = joint_ranges["tendon_extend"]

                if r:
                    pos.set("ctrlrange", r)
                    pos.set("ctrllimited", "true")

            ET.indent(act_root, space="  ", level=0)
            act_tree.write(out_actuator_xml_path, encoding="unicode", xml_declaration=True)
    except Exception as e:
        print(f"Failed to generate autogenerated_actuator_sensor.xml: {e}")

    # Cleanup temp file
    try:
        import os

        os.remove(raw_mjcf_path)
    except OSError:
        pass
