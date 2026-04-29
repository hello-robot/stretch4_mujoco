# Stretch 4 MJCF and URDF


The URDF was brought over from `stretch_urdf_ii` and converted to mjcf using https://docs.kscale.dev/docs/urdf2mjcf.

```
pip install urdf2mjcf

python3 batch_decimate_stl.py ./meshes # You will need a Blender executable added to your path.

urdf2mjcf stretch_description_SE4_eoa_wrist_dw4_tool_sg4.urdf
```

## Manual modifications to `stretch_description_SE4_eoa_wrist_dw4_tool_sg4.xml`:
1. Remove all the tags except `assets` and `worldbody`
2. Remove these tags from worldbody:
```xml
    <light directional="true" diffuse="0.4 0.4 0.4" specular="0.1 0.1 0.1" pos="0 0 5.0" dir="0 0 -1" castshadow="false" />
    <light directional="true" diffuse="0.6 0.6 0.6" specular="0.2 0.2 0.2" pos="0 0 4" dir="0 0 -1" />
    <geom name="ground" type="plane" pos="0 0 0" size="100 100 0.001" quat="1 0 0 0" material="matplane" condim="3" conaffinity="15" />
    <camera name="fixed" pos="0 -3.0 0.6150738234348692" xyaxes="1 0 0 0 0 1" />
    <camera name="track" mode="trackcom" pos="0 -3.0 0.6150738234348692" xyaxes="1 0 0 0 0 1" />
```
3. Replace the wheels joints with the following. Make sure to keep the original pos and quat.
Change wheel_1 -> back_wheel, wheel_0 -> left_wheel, wheel_2 -> right_wheel.
```xml
      <body name="link_left_wheel" pos="0.150688 0.087 0.0742" quat="0.612372 -0.612372 0.353553 -0.353553">
        <inertial pos="-1.23869e-05 5.05445e-06 -0.000272307" quat="0.135814 0.693941 -0.135814 0.693941" mass="5.91084" diaginertia="0.0295034 0.0152963 0.0152901" />
        <joint name="joint_left_wheel"  class="wheel" />
        <geom type="mesh" mesh="link_wheel_0"  class="visualgeom" />
        <geom name="link_left_wheel2" class="wheel_collision"/>
        <body>
           <joint class="wheel_roller" />
           <geom name="link_left_wheel"  class="wheel_collision_roller" />
        </body>
      </body>
      <body name="link_back_wheel" pos="-0.150688 0.087 0.0742" quat="0.612372 -0.612372 -0.353553 0.353553">
        <inertial pos="-1.23869e-05 5.05445e-06 -0.000272307" quat="0.135814 0.693941 -0.135814 0.693941" mass="5.91084" diaginertia="0.0295034 0.0152963 0.0152901" />
        <joint name="joint_back_wheel" class="wheel"  />
        <geom type="mesh" mesh="link_wheel_1" class="visualgeom" />
        <geom name="link_back_wheel2"   class="wheel_collision"  />

        <body>
           <joint class="wheel_roller"/>
           <geom name="link_back_wheel"  class="wheel_collision_roller"/>
        </body>
      </body>
      <body name="link_right_wheel" pos="0 -0.174 0.0742" quat="0 0 0.707107 -0.707107">
        <inertial pos="-1.23869e-05 5.05445e-06 -0.000272307" quat="0.135814 0.693941 -0.135814 0.693941" mass="5.91084" diaginertia="0.0295034 0.0152963 0.0152901" />
        <joint name="joint_right_wheel" class="wheel"  />
        <geom type="mesh" mesh="link_wheel_2" class="visualgeom" />
        <geom name="link_right_wheel2"   class="wheel_collision"  />
        <body>
           <joint class="wheel_roller"/>
           <geom name="link_right_wheel"  class="wheel_collision_roller"/>
        </body>
      </body>
```

4. Encapsulate lidar and camera geoms with a body, and an `<include/>` tag as shown below. Note that the body should take the pos and quat of the geom, and the virtual <camera/> position and rotation were manually computed:

For `link_lidar_right` and `link_lidar_left`:
```xml
      <body name="link_lidar_right" pos="0.0387442 -0.124414 1.50375"
        quat="-0.339444 -0.353553 -0.853553 0.176704">
        <geom type="mesh" rgba="0.298039 0.298039 0.298039 1" mesh="link_hesai_right" contype="1"
          conaffinity="0" density="0" group="1" class="visualgeom" />
        <!-- <geom type="mesh" rgba="0.298039 0.298039 0.298039 1" mesh="link_hesai_right_collision"
        class="collision" /> -->

        <include file="hemisphere_lidar_cameras_right.xml" />
      </body>

      <body name="link_lidar_left" pos="0.0387442 0.124414 1.50375"
        quat="0.176704 -0.853553 -0.353553 -0.339444">
        <geom type="mesh" rgba="0.298039 0.298039 0.298039 1" mesh="link_hesai_left" contype="1"
          conaffinity="0" density="0" group="1" class="visualgeom" />
        <!-- <geom type="mesh" rgba="0.298039 0.298039 0.298039 1" mesh="link_hesai_left_collision"
        class="collision" /> -->
        <include file="hemisphere_lidar_cameras_left.xml" />
      </body>
```

For `link_camera_right`, `link_camera_left`, and `link_camera_center`:
```xml
      <body name="link_camera_right" pos="0.0736423 -0.075 1.54612" quat="0.446198 0 0.894934 0">
        <geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="link_camera_right" contype="1"
          conaffinity="0" density="0" group="1" class="visualgeom" />
        <!-- <geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="link_camera_right"  /> -->
        <camera name="link_camera_right" pos="0 0 0.015" euler="0 3.14 0" />
      </body>

      <body name="link_camera_left" pos="0.0736423 0.075 1.54612" quat="0 0.894934 0 0.446198">
        <geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="link_camera_left" contype="1"
          conaffinity="0" density="0" group="1" class="visualgeom" />
        <!-- <geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="link_camera_left"  /> -->
        <camera name="link_camera_left" pos="0 0 0.015" euler="0 3.14 0" />
      </body>

      <body name="link_camera_center" pos="0.0800039 0 1.53817" quat="0 0.809017 0 0.587785">
        <geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="link_camera_center" contype="1"
          conaffinity="0" density="0" group="1" class="visualgeom" />
        <!-- <geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="link_camera_center"/> -->
        <camera name="link_camera_center" pos="0 0 0.015" euler="0 3.14 0" />
      </body>
```

For the wrist D405 camera (inside `link_wrist_roll`):
```xml
      <body name="d405_camera_mount" pos="0 0.0660509 -0.0253083" quat="0.5 0.5 0.5 -0.5">
          <include file="d405_cameras.xml"/>
      </body>
```

Pressing `'q'` in the Mujoco Viewer should allow you to see the virtual cameras:
<image src="head_cameras.png" />

6. Replace mast:
```xml
      <geom pos="-0.0227751 -0.124693 0.036" quat="1 0 0 0" type="mesh" rgba="0.752941 0.752941 0.752941 1" mesh="link_mast" contype="1" conaffinity="0" density="0" group="1" class="visualgeom" />
      <geom type="mesh" rgba="0.752941 0.752941 0.752941 1" mesh="link_mast" pos="-0.0227751 -0.124693 0.036" quat="1 0 0 0" />
```

with:
```xml

      <body name="link_mast" pos="-0.0227751 -0.124693 0.036">
        <geom type="mesh"
          mesh="link_mast" class="visualgeom" />
        <geom mesh="link_mast" class="collision" mass="4.25" />
      </body>
```

7. Replace `<body name="root"` with `<body name="base_link"`

8. Remove these assets:
```xml
    <texture name="texplane" type="2d" builtin="checker" rgb1=".0 .0 .0" rgb2=".8 .8 .8" width="100" height="100" />
    <material name="matplane" reflectance="0." texture="texplane" texrepeat="1 1" texuniform="true" />
    <material name="visualgeom" rgba="0.5 0.9 0.2 1" />
```

9. Encapsulate `  <body name="link_gripper_finger_right"` and ` <body name="link_gripper_finger_left"` with a body `link_gripper_slider` and joint `joint_gripper_slide`:

```xml
    <!-- Gripper Mechanism -->
    <body name="link_gripper_slider" euler="0 0 0" pos="0 0 0" gravcomp="1">
    <joint name="joint_gripper_slide" class="finger_slide" />
    <geom type="box" size=".03 .005 .005" mass=".05" class="visualgeom"
        rgba="0 0 0 0" />

        <!-- <body name="link_gripper_finger_right" ..... -->
        <!-- <body name="link_gripper_finger_left" ..... -->

    </body>
````

10. Add `class="rubber"` to the `geom type="mesh"` for `link_gripper_finger_left/right` and `link_gripper_fingertip_left/right`.

11. Encapsulate body `link_gripper_slider` and geoms `link_gripper_s4_body` with a body `<body name="link_gripper_s4_body" euler="0 0 0" pos="0 0 0" gravcomp="1">`:

```xml  
    <body name="link_gripper_s4_body" euler="0 0 0" pos="0 0 0" gravcomp="1">

        <geom pos="0 0 0.014" quat="0 0 0 -1" type="mesh"
            rgba="0.75294 0.75294 0.75294 1" mesh="link_gripper_s4_body" contype="1"
            conaffinity="0" density="0" group="1" class="visualgeom" />
        <geom type="mesh" rgba="0.75294 0.75294 0.75294 1"
            mesh="link_gripper_s4_body"
            pos="0 0 0.014" quat="0 0 0 -1" />

        <!-- Gripper Mechanism -->
        <!-- <body name="link_gripper_slider" euler="0 0 0" pos="0 0 0" gravcomp="1"> .......-->

    </body>

``` 

11. Replace base_link with

```xml
      <geom type="mesh" mesh="base_link" class="visualgeom" />
      <geom type="mesh" mass="26.8" mesh="base_link" class="collision" />
```

12. All <body> tags starting with lift and arm and their children get the `gravcomp="1"` attribute.

13. Encapsulate the geom for `link_arm_l4` and the body for `link_arm_l3` with `<body name="link_arm_l4" gravcomp="1">`

14. Swap `<joint name="joint_lift" pos="0 0 0" axis="0 0 1" type="slide" range="-0.1 1" />` for `<joint name="joint_lift" class="lift"/>`

15. Swap The four arm joints, e.g. `<joint name="joint_arm_lx" pos="0 0 0" axis="0 0 1" type="slide" range="0 0.13" />` for `<joint name="joint_arm_lx" class="telescope"/>`

16. Change the attributes of the three wrist joints to use the `class` attribute: `<joint name="joint_wrist_yaw/pitch/roll" class="wrist_yaw/pitch/roll"/>`. Double check that the joint limits in [defaults.xml](./defaults.xml) are correct.

17. Change the attributes of `joint_gripper_finger_left/right` to `<joint name="joint_gripper_finger_left/right" class="gripper_left/right_finger"/>`

18. Make sure the collision geoms are using the _collision asset, and the others are not.


