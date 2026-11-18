"""
docs/fisheye_crop_vs_rectify.png: Stretch's left and right head fisheyes beside the DROID exo
camera where the virtual Franka's would be, in Stretch's scene at its spawn pose. Top row cropped
(--head-crop droid), bottom row rectified (--rectify).

    MUJOCO_GL=egl python -m examples.vla.molmobot_droid.franka_retarget.docs.scripts.fisheye_crop_vs_rectify
"""

import mujoco

from examples.vla.molmobot_droid.checkpoint import DROID_IMAGE_SIZE
from examples.vla.molmobot_droid.droid import HEAD_CAMERAS, PinholeRenderer, StretchCameraRenderer
from examples.vla.molmobot_droid.franka_retarget.docs.scripts.figures import label, load_scene, save
from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import (
    HEAD_SIDES,
    LIFT_RANGE,
    SPAWN_ARM_EXTENSION,
    SPAWN_LIFT_FRACTION,
    head_rectifier,
    to_droid_frame,
)
from examples.vla.molmobot_droid.molmospaces.custom_scene import DROID_EXO_IN_STRETCH_SCENE, load_custom_scene_stretch4


def main():
    scene, pose = load_scene()
    model = load_custom_scene_stretch4(scene, pose).model
    data = mujoco.MjData(model)
    # Where spawn_stretch4() starts it: lift up, arm a little out.
    data.joint("lift_joint").qpos = SPAWN_LIFT_FRACTION * LIFT_RANGE[1]
    for i in range(1, 5):
        data.joint(f"arm_l{i}_joint").qpos = SPAWN_ARM_EXTENSION / 4
    mujoco.mj_forward(model, data)

    exo = PinholeRenderer(model, DROID_EXO_IN_STRETCH_SCENE, DROID_IMAGE_SIZE).render(data)
    fisheye = {
        side: StretchCameraRenderer(model, HEAD_CAMERAS[side].camera_name_in_mjcf, HEAD_CAMERAS[side]).render(data)
        for side in HEAD_SIDES
    }
    save([
        [label(to_droid_frame(fisheye["left"]), "left fisheye, cropped (--head-crop droid)"), label(exo, "DROID exo camera"),
         label(to_droid_frame(fisheye["right"]), "right fisheye, cropped")],
        [label(head_rectifier("left")(fisheye["left"]), "left fisheye, cropped + rectified (--rectify)"),
         label(exo, "DROID exo camera"), label(head_rectifier("right")(fisheye["right"]), "right fisheye, cropped + rectified")],
    ], "fisheye_crop_vs_rectify")


if __name__ == "__main__":
    main()
