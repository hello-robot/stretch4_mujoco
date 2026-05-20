from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator
from scipy.spatial.transform import Rotation as R



if __name__ == "__main__":

    translation_m = [0.1, 0.5, 0.0 ]

    # Convert euler angles to quaternion
    euler_angles_degrees = [0, 0, -90] # Example: roll, pitch, yaw (x, y, z)
    rotation_obj = R.from_euler('xyz', euler_angles_degrees, degrees=True)
    rotation_quat = rotation_obj.as_quat(scalar_first=True)


    sim = Stretch4MujocoSimulator(
        start_translation=translation_m,
        start_rotation_quat=rotation_quat
    )

    sim.start(headless=False)

    while sim.is_running(): ...
