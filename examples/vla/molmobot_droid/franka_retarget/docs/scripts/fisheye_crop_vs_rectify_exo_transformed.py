"""
docs/fisheye_crop_vs_rectify_exo_transformed.png: each head fisheye on the Franka where it is on
Stretch (`run_franka.py --exo_camera left|right`), beside the DROID exo camera moved to the same
pose. Columns: left fisheye, the DROID exo camera there, right fisheye, the DROID exo camera
there; top row cropped (--head-crop droid), bottom row rectified (--rectify). For the left
fisheye the Franka is LEFT_FISHEYE_LOWERING lower, the camera staying put, as the left fisheye is
otherwise inside its arm.

The rectified fisheye should be the moved DROID camera's image: the script prints how far apart
they are (mean difference, and the shift between them).

    MUJOCO_GL=egl python -m examples.vla.molmobot_droid.franka_retarget.docs.scripts.fisheye_crop_vs_rectify_exo_transformed
"""

import math

import cv2
import mujoco
import numpy as np

from examples.vla.molmobot_droid.checkpoint import DROID_IMAGE_SIZE
from examples.vla.molmobot_droid.droid import FrankaDroidEnv
from examples.vla.molmobot_droid.franka_retarget.docs.scripts.figures import label, load_scene, save
from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import (
    DROID_EXO_FOCAL_PX,
    HEAD_SIDES,
    head_rectifier,
    to_droid_frame,
)
from examples.vla.molmobot_droid.molmospaces.custom_scene import load_custom_scene_franka_droid
from stretch4_mujoco.fisheye_renderer import FISHEYE_LENS_PROTRUSION_M

LEFT_FISHEYE_LOWERING = 0.2
"""How far (m) the Franka is lowered for the left fisheye. Within the ±0.3 m its height is
randomized by in training."""


def droid_exo_at_fisheye(env: FrankaDroidEnv) -> np.ndarray:
    """
    The DROID exo camera at the pose of `env`'s head exo camera: a pinhole of DROID_EXO_FOCAL_PX.
    Rendered in the sensor's (landscape) frame, 368 wide and 640 high there, then turned upright
    as the fisheye is; from FISHEYE_LENS_PROTRUSION_M in front of the camera frame, where the
    fisheye's views are rendered from.
    """
    model, data = env.model, env.data
    cam = model.camera(env.spawn.exo_camera_name).id
    saved = model.cam_fovy[cam].copy(), model.cam_pos[cam].copy(), model.cam_sensorsize[cam].copy()
    width, height = DROID_IMAGE_SIZE
    model.cam_fovy[cam] = math.degrees(2 * math.atan(width / 2 / DROID_EXO_FOCAL_PX))
    model.cam_sensorsize[cam] = 0  # so fovy, not the sensor size, sets the focal length
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, model.cam_quat[cam])
    model.cam_pos[cam] += rotation.reshape(3, 3) @ [0.0, 0.0, -FISHEYE_LENS_PROTRUSION_M]  # it looks down -z
    try:
        # The camera's pose in `data` follows cam_pos only once recomputed.
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)
        with mujoco.Renderer(model, width=height, height=width) as renderer:
            renderer.update_scene(data, camera=env.spawn.exo_camera_name)
            image = renderer.render()
    finally:
        model.cam_fovy[cam], model.cam_pos[cam], model.cam_sensorsize[cam] = saved
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)
    turns = env.exo_renderer.camera.initial_camera_settings.rotate_number_of_times
    return np.ascontiguousarray(np.rot90(image, turns))


def main():
    scene, pose = load_scene()
    fisheye, exo = {}, {}
    for side in HEAD_SIDES:
        env = load_custom_scene_franka_droid(scene, pose, exo_camera=side)
        try:
            if side == "left":
                env.set_pedestal_height(env.spawn.pedestal_height - LEFT_FISHEYE_LOWERING)
            env.reset()
            fisheye[side] = env.exo_renderer.render(env.data)
            exo[side] = droid_exo_at_fisheye(env)
        finally:
            env.close()
        rectified = head_rectifier(side)(fisheye[side])
        gray = [cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY) for image in (exo[side], rectified)]
        (dx, dy), _ = cv2.phaseCorrelate(*gray)
        difference = np.abs(rectified.astype(float) - exo[side]).mean()
        print(f"{side}: rectified vs the DROID exo camera there: mean difference {difference:.2f}/255, "
              f"shifted ({dx:.2f}, {dy:.2f}) px")

    save([
        [label(to_droid_frame(fisheye["left"]), "left fisheye, cropped"), label(exo["left"], "DROID exo camera at the left fisheye"),
         label(to_droid_frame(fisheye["right"]), "right fisheye, cropped"), label(exo["right"], "DROID exo camera at the right fisheye")],
        [label(head_rectifier("left")(fisheye["left"]), "left fisheye, cropped + rectified"),
         label(exo["left"], "DROID exo camera at the left fisheye"),
         label(head_rectifier("right")(fisheye["right"]), "right fisheye, cropped + rectified"),
         label(exo["right"], "DROID exo camera at the right fisheye")],
    ], "fisheye_crop_vs_rectify_exo_transformed")


if __name__ == "__main__":
    main()
