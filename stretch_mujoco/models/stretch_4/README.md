# Stretch 4 MJCF and URDF

The MJCF is now dynamically generated on-the-fly from the `stretch4_urdf` package using `urdf2mjcf` and programmatic DOM manipulations.

When the simulator is initialized, `get_robot_xml_path()` dynamically invokes `mjcf_generator.py` to convert the appropriate URDF into an MJCF and applies custom modifications required by Mujoco.

## Programmatic Transformations Applied
The `mjcf_generator.py` applies the following changes to the raw `urdf2mjcf` output:

1. Removes all tags except `asset` and `worldbody`.
2. Removes specific tags from worldbody (e.g., `light`, `camera`, `ground` plane).
3. Replaces the default wheel joints with custom bodies (`link_left_wheel`, `link_right_wheel`, `link_back_wheel`) containing `class="wheel"` and associated wheel rollers to support omnibase motion, and includes `omniwheels.xml`.
4. Encapsulates lidar and camera geoms with dedicated bodies and includes `hemisphere_lidar_cameras_right/left.xml`.
5. Replaces `link_mast` geometry with appropriate `collision` and `visualgeom` classes.
6. Renames the root body to `base_link` and adds an `imu` site.
7. Encapsulates gripper fingers into `link_gripper_slider` (if present, e.g. for sg4 tool).
8. Adjusts `class="visualgeom"` and adds `class="rubber"` to gripper fingers.
9. Modifies wrist roll to include `gripper_cameras.xml`.
10. Adds `gravcomp="1"` to lift, arm, and wrist components.
11. Maps joints to appropriate classes defined in `defaults.xml` (`lift_stretch4`, `telescope`, `wrist_yaw_stretch4`, etc.).

These steps have been fully automated and replace the previously documented manual processes.
