"""
Fast checks for the Stretch fine-tuning path in
`examples/machine_learning/molmospaces/finetuning/`, and for the native MolmoBot
policy adapter.

These cover the things that are silently wrong when they are wrong: the camera
mapping duplicated out of the MolmoSpaces config, the action spec that has to
agree between the trainer command and the evaluation policy, the trajectory
format's fixed-width fields, and the train/val split. The recording path is
exercised end to end against synthetic frames, because a dataset that is subtly
malformed does not fail -- it trains a worse policy.

Requires the optional MolmoSpaces dependency:  pip install -e ".[molmo]"
"""

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("molmo_spaces")
pytest.importorskip("h5py")

from examples.machine_learning.molmospaces import hdf5_layout  # noqa: E402
from examples.machine_learning.molmospaces.finetuning.live_recorder import (  # noqa: E402
    TRAINED_CAMERA_MJCF_NAMES,
    LiveDatasetRecorder,
    pose7_from_matrix,
    qpos_from_status,
)
from examples.machine_learning.molmospaces.stretch.config import (  # noqa: E402
    HEAD_CAMERA,
    HEAD_CAMERA_MJCF_NAME,
    WRIST_CAMERA,
    WRIST_CAMERA_MJCF_NAME,
    Stretch4RobotConfig,
)

# =============================================================================
# The two duplications that have to stay in step
# =============================================================================


def test_recorder_camera_mapping_matches_the_trained_camera_system():
    """`live_recorder` duplicates the camera mapping to avoid a dependency.

    `examples/molmo_environment.py` runs on a bare `--scene` with MolmoSpaces
    absent, so the recorder cannot import `stretch/config.py`. The copy is only
    safe while this test holds: a drift here would record a dataset under the
    name `head_camera` from a camera the benchmark never uses, which looks
    entirely plausible and trains a policy on the wrong viewpoint.
    """
    assert TRAINED_CAMERA_MJCF_NAMES == {
        HEAD_CAMERA: HEAD_CAMERA_MJCF_NAME,
        WRIST_CAMERA: WRIST_CAMERA_MJCF_NAME,
    }


def test_head_camera_is_the_centre_of_the_head_assembly():
    """Pin which of Stretch 4's three head cameras the datasets and policies use.

    The head is a fixed assembly with a centre camera and a stereo pair either
    side, and they do not see the same thing: measured on the compiled MJCF, the
    centre camera sits 1.62m up looking 35 degrees down, while the pair sit 7.5cm
    to each side looking 47 degrees down. Everything here uses the centre one.
    """
    assert HEAD_CAMERA_MJCF_NAME == "camera_center_link"
    assert WRIST_CAMERA_MJCF_NAME == "gripper_camera_left_rgb"


def test_stretch_action_spec_agrees_across_training_and_evaluation():
    """The trainer command and the eval policy must unpack the same vector.

    `finetune.py` writes the move groups into MolmoBot's `--action_move_groups`
    and `policies/molmobot_policy.py` hands the same spec to `SynthVLAPolicy`,
    which unpacks the model's output across them *in order*. A mismatch would
    assign every joint after the first differing group to the wrong actuator,
    with nothing in the logs to say so.
    """
    from examples.machine_learning.molmospaces.finetuning.finetune import (
        STRETCH_ACTION_SPEC as TRAINING_SPEC,
    )
    from examples.machine_learning.molmospaces.policies.molmobot_policy import (
        STRETCH_ACTION_SPEC as EVAL_SPEC,
    )

    assert TRAINING_SPEC == EVAL_SPEC
    assert list(TRAINING_SPEC) == list(EVAL_SPEC), "order matters: the vector is unpacked in it"
    assert sum(TRAINING_SPEC.values()) == 10


def test_stretch_action_spec_matches_the_move_groups(tmp_path):
    """...and both must match the widths the robot view actually reports."""
    import mujoco
    from mujoco import MjSpec

    from examples.machine_learning.molmospaces.policies.molmobot_policy import (
        STRETCH_ACTION_SPEC,
    )
    from examples.machine_learning.molmospaces.stretch.config import Stretch4RobotConfig
    from examples.machine_learning.molmospaces.stretch.robot import Stretch4Robot
    from examples.machine_learning.molmospaces.stretch.robot_view import Stretch4RobotView

    spec = MjSpec()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, name="floor", size=[10.0, 10.0, 0.1])
    Stretch4Robot.add_robot_to_scene(
        Stretch4RobotConfig(), spec, prefix="robot_0/", pos=[0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0]
    )
    model = spec.compile()
    view = Stretch4RobotView(mujoco.MjData(model), "robot_0/")

    for group, width in STRETCH_ACTION_SPEC.items():
        assert view.get_move_group(group).pos_dim == width, group
    assert list(STRETCH_ACTION_SPEC) == list(Stretch4RobotView.MOVE_GROUP_ORDER)


# =============================================================================
# The trajectory format
# =============================================================================


def test_json_blob_round_trips():
    qpos = {
        "base": [1.5, -2.0, 0.3],
        "lift": [0.6],
        "arm": [0.2],
        "wrist": [0.1, 0.2, 0.3],
        "gripper": [0.4, 0.4],
    }
    assert hdf5_layout.decode_json_blob(hdf5_layout.encode_json_blob(qpos)) == qpos


def test_json_blob_refuses_to_truncate():
    """A clipped blob fails to parse thousands of episodes later, not now."""
    with pytest.raises(ValueError, match="over the .* row width"):
        hdf5_layout.encode_json_blob({"arm": list(range(500))}, width=64)


def test_video_path_field_is_fixed_width():
    """The 100-byte field is part of the format, not a suggestion."""
    row = hdf5_layout.encode_video_path("episode_00000003_head_camera.mp4")
    assert row.shape == (hdf5_layout.VIDEO_PATH_FIELD_BYTES,)
    assert row.dtype == np.uint8
    assert hdf5_layout.decode_video_path(row) == "episode_00000003_head_camera.mp4"


def test_video_filename_matches_the_saver_convention():
    assert (
        hdf5_layout.video_filename(3, "head_camera", "_batch_1_of_1")
        == "episode_00000003_head_camera_batch_1_of_1.mp4"
    )


# =============================================================================
# Recording
# =============================================================================


def _record_episodes(directory, episodes=2, frames=20, fps=15.0):
    recorder = LiveDatasetRecorder(
        directory, list(TRAINED_CAMERA_MJCF_NAMES), fps=fps, task_description="pick up the mug"
    )
    rng = np.random.default_rng(0)
    for _ in range(episodes):
        recorder.start_episode()
        for step in range(frames):
            qpos = {
                "base": [1.0 + 0.01 * step, 2.0, 0.3],
                "lift": [0.6 + 0.005 * step],
                "arm": [0.1],
                "wrist": [0.0, 0.3, 0.0],
                "gripper": [0.5, 0.5],
            }
            recorder.record_step(
                qpos=qpos,
                base_pose7=pose7_from_matrix(np.eye(4)),
                tcp_pose7=pose7_from_matrix(np.eye(4)),
                images={
                    name: rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
                    for name in recorder.camera_names
                },
                frame_time=float(step),
            )
        recorder.finish_episode()
    recorder.close()
    return recorder


def test_recorded_episode_is_readable_and_action_is_the_next_state(tmp_path):
    """The action for a frame is the next frame's state; the last frame is dropped."""
    import h5py

    _record_episodes(tmp_path / "rollouts", episodes=1, frames=10)
    path = tmp_path / "rollouts" / "house_teleop_000" / "trajectories.h5"
    with h5py.File(path, "r") as h5_file:
        trajectory = h5_file["traj_0"]
        assert len(trajectory["obs/agent/qpos"]) == 9, "10 frames -> 9 transitions"

        state = hdf5_layout.decode_json_blob(trajectory["obs/agent/qpos"][0])
        next_state = hdf5_layout.decode_json_blob(trajectory["obs/agent/qpos"][1])
        action = hdf5_layout.decode_json_blob(trajectory["actions/joint_pos"][0])
        assert action == next_state

        relative = hdf5_layout.decode_json_blob(trajectory["actions/joint_pos_rel"][0])
        assert relative["lift"][0] == pytest.approx(
            next_state["lift"][0] - state["lift"][0], abs=1e-9
        )

        # MolmoBot reads the video filename out of this group.
        for camera in TRAINED_CAMERA_MJCF_NAMES:
            recorded = hdf5_layout.decode_video_path(trajectory[f"obs/sensor_data/{camera}"][:])
            assert (path.parent / recorded).exists()

        import json

        scene = json.loads(trajectory["obs_scene"][()])
        assert scene["task_description"] == "pick up the mug"
        assert bool(trajectory["success"][-1])


def test_repeated_camera_timestamps_are_dropped(tmp_path):
    """A render loop spins faster than the cameras; repeats would be blind frames."""
    recorder = LiveDatasetRecorder(tmp_path, list(TRAINED_CAMERA_MJCF_NAMES), fps=15.0)
    recorder.start_episode()
    images = {name: np.zeros((8, 8, 3), np.uint8) for name in recorder.camera_names}
    qpos = {
        "base": [0, 0, 0],
        "lift": [0.6],
        "arm": [0.1],
        "wrist": [0, 0, 0],
        "gripper": [0.5, 0.5],
    }
    pose = pose7_from_matrix(np.eye(4))
    for _ in range(5):
        recorder.record_step(
            qpos=qpos, base_pose7=pose, tcp_pose7=pose, images=images, frame_time=1.0
        )
    assert recorder.current_length == 1
    recorder.discard_episode()


def test_a_frame_with_a_missing_camera_is_skipped_whole(tmp_path):
    """Half a frame would put the cameras out of sync with the state."""
    recorder = LiveDatasetRecorder(tmp_path, list(TRAINED_CAMERA_MJCF_NAMES), fps=15.0)
    recorder.start_episode()
    qpos = {
        "base": [0, 0, 0],
        "lift": [0.6],
        "arm": [0.1],
        "wrist": [0, 0, 0],
        "gripper": [0.5, 0.5],
    }
    pose = pose7_from_matrix(np.eye(4))
    recorder.record_step(
        qpos=qpos,
        base_pose7=pose,
        tcp_pose7=pose,
        images={HEAD_CAMERA: np.zeros((8, 8, 3), np.uint8), WRIST_CAMERA: None},
        frame_time=0.0,
    )
    assert recorder.current_length == 0
    recorder.discard_episode()


def test_a_one_frame_episode_is_discarded(tmp_path):
    """An action is the next frame's state, so one frame contains no action."""
    recorder = LiveDatasetRecorder(tmp_path, list(TRAINED_CAMERA_MJCF_NAMES), fps=15.0)
    recorder.start_episode()
    qpos = {
        "base": [0, 0, 0],
        "lift": [0.6],
        "arm": [0.1],
        "wrist": [0, 0, 0],
        "gripper": [0.5, 0.5],
    }
    pose = pose7_from_matrix(np.eye(4))
    recorder.record_step(
        qpos=qpos,
        base_pose7=pose,
        tcp_pose7=pose,
        images={name: np.zeros((8, 8, 3), np.uint8) for name in recorder.camera_names},
        frame_time=0.0,
    )
    assert recorder.finish_episode() is None
    assert recorder.kept_episodes == 0


def test_qpos_from_status_matches_the_move_groups():
    """The recorded qpos dict must be keyed and sized like the robot view's."""
    from examples.machine_learning.molmospaces.policies.molmobot_policy import (
        STRETCH_ACTION_SPEC,
    )
    from stretch4_mujoco.datamodels.status_stretch_joints import StatusStretchJoints

    qpos = qpos_from_status(StatusStretchJoints.default())
    assert list(qpos) == list(STRETCH_ACTION_SPEC)
    for group, width in STRETCH_ACTION_SPEC.items():
        assert len(qpos[group]) == width, group


# =============================================================================
# Layout and repair
# =============================================================================


def test_sensor_data_paths_are_filled_in_from_the_videos(tmp_path):
    """MolmoSpaces strips camera observations, leaving `obs/sensor_data` empty."""
    import h5py

    house = tmp_path / "house_0"
    house.mkdir(parents=True)
    (house / "episode_00000000_head_camera.mp4").write_bytes(b"not really an mp4")
    with h5py.File(house / "trajectories.h5", "w") as h5_file:
        h5_file.create_group("traj_0/obs/sensor_data")

    counts = hdf5_layout.ensure_sensor_data_paths(tmp_path)
    assert counts == {
        "trajectories": 1,
        "entries": 1,
        "already_present": 0,
        "missing_video": 0,
    }
    with h5py.File(house / "trajectories.h5", "r") as h5_file:
        path = hdf5_layout.decode_video_path(h5_file["traj_0/obs/sensor_data/head_camera"][:])
    assert path == "episode_00000000_head_camera.mp4"

    # Idempotent: a second pass must not duplicate or overwrite.
    assert hdf5_layout.ensure_sensor_data_paths(tmp_path)["already_present"] == 1


def test_train_val_split_never_splits_a_house(tmp_path):
    """A room in both splits makes the validation score meaningless."""
    import h5py

    for index in range(10):
        house = tmp_path / "run" / f"house_{index}"
        house.mkdir(parents=True)
        with h5py.File(house / "trajectories.h5", "w") as h5_file:
            h5_file.create_group("traj_0")

    placed = hdf5_layout.arrange_train_val_split(
        tmp_path / "run", tmp_path / "task", val_fraction=0.2
    )
    train = {path.name for path in placed["train"]}
    val = {path.name for path in placed["val"]}
    assert train and val
    assert not (train & val)
    assert len(train) + len(val) == 10
    assert (tmp_path / "task" / "train").is_dir()
    assert (tmp_path / "task" / "val").is_dir()


def test_a_single_house_yields_no_validation_split(tmp_path):
    """Better an empty val split than the same room in both."""
    import h5py

    house = tmp_path / "run" / "house_0"
    house.mkdir(parents=True)
    with h5py.File(house / "trajectories.h5", "w") as h5_file:
        h5_file.create_group("traj_0")

    placed = hdf5_layout.arrange_train_val_split(tmp_path / "run", tmp_path / "task")
    assert len(placed["train"]) == 1
    assert placed["val"] == []


def test_split_refuses_an_empty_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="No house_"):
        hdf5_layout.arrange_train_val_split(tmp_path, tmp_path / "task")


# =============================================================================
# The MolmoBot adapter
# =============================================================================


def test_molmobot_config_rejects_a_group_mismatch():
    """The spec and the group list are unpacked together; they cannot disagree."""
    from examples.machine_learning.molmospaces.policies.molmobot_policy import (
        StretchMolmoBotPolicyConfig,
    )

    with pytest.raises(ValueError, match="same groups"):
        StretchMolmoBotPolicyConfig(
            action_move_group_names=["base", "lift"], action_spec={"base": 3}
        )


def test_molmobot_policy_says_how_to_install_molmobot():
    """MolmoBot is not a dependency, so the failure has to be actionable."""
    from examples.machine_learning.molmospaces.policies.molmobot_policy import (
        StretchMolmoBotPolicy,
    )

    with pytest.raises(ImportError) as error:
        StretchMolmoBotPolicy._import_molmobot()
    message = str(error.value)
    assert "git clone https://github.com/allenai/MolmoBot" in message
    assert "PYTHONPATH" in message


def test_molmobot_eval_config_defaults_to_the_trained_action_type():
    from examples.machine_learning.molmospaces.configs import StretchMolmoBotEvalConfig
    from examples.machine_learning.molmospaces.finetuning.finetune import MOLMOBOT_ACTION_TYPES

    config = StretchMolmoBotEvalConfig()
    assert config.policy_config.action_type == MOLMOBOT_ACTION_TYPES[0] == "joint_pos_rel"
    assert config.tag == "stretch4_molmobot"


# =============================================================================
# The LeRobot export, and the datagen configs that feed it
# =============================================================================


def test_only_the_native_action_space_is_exportable():
    """The Franka-space export is gone; nothing may quietly accept it again."""
    from examples.machine_learning.molmospaces.finetuning.lerobot_export import ACTION_SPACES

    assert ACTION_SPACES == ("stretch",)


def test_datagen_configs_substitute_stretch_and_widen_the_standoff():
    """Every registered datagen config must place the robot where Stretch works."""
    from examples.machine_learning.molmospaces.finetuning import datagen_configs as configs
    from examples.machine_learning.molmospaces.stretch.episode_overrides import REACH_BAND_M

    for task, class_name in configs.DATAGEN_CONFIGS.items():
        config = getattr(configs, class_name)()
        assert isinstance(config.robot_config, Stretch4RobotConfig), task
        assert type(config.camera_config).__name__ == "Stretch4CameraSystem", task
        assert type(config.policy_config).__name__ == "StretchSimpleIKPolicyConfig", task
        sampler = config.task_sampler_config
        assert tuple(sampler.base_pose_sampling_radius_range) == tuple(REACH_BAND_M), task
        assert sampler.robot_safety_radius == configs.STRETCH_BASE_SAFETY_RADIUS_M, task


def test_export_feature_names_describe_one_gripper():
    """Stretch has one commanded gripper DOF; the names must not imply two."""
    from examples.machine_learning.molmospaces.finetuning.lerobot_export import (
        GRIPPER_CHANNEL_NAMES,
        STRETCH_ACTION_NAMES,
        STRETCH_STATE_NAMES,
    )
    from examples.machine_learning.molmospaces.policies.networks import ACTION_DIM, STATE_DIM

    assert GRIPPER_CHANNEL_NAMES[0] == "stretch_gripper"
    # The names are `policies/networks.py`'s encoding, so they have to be exactly
    # as wide as it is or the dataset metadata lies.
    assert len(STRETCH_STATE_NAMES) == STATE_DIM
    assert len(STRETCH_ACTION_NAMES) == ACTION_DIM


def test_implausible_base_commands_are_replaced_not_encoded():
    """Step 0's recorded no-op base command is the world origin, not "stay put".

    Left alone it encodes a 25-metre tool displacement into the first frame of
    every episode. The threshold has to be well clear of any real command.
    """
    from examples.machine_learning.molmospaces.finetuning.lerobot_export import (
        IMPLAUSIBLE_BASE_COMMAND_M,
    )

    assert IMPLAUSIBLE_BASE_COMMAND_M > 0.35, "must not fire on a real base solve"
    assert IMPLAUSIBLE_BASE_COMMAND_M < 5.0, "must still catch a zeroed no-op"


def test_export_encoder_matches_the_bc_encoding():
    """The export must not fork the encoding the BC trainer uses."""
    from examples.machine_learning.molmospaces.finetuning.lerobot_export import (
        _StretchSpaceEncoder,
    )
    from examples.machine_learning.molmospaces.policies.networks import (
        ACTION_DIM,
        STATE_DIM,
        encode_state,
    )

    encoder = _StretchSpaceEncoder()
    qpos = {
        "base": np.array([1.0, 2.0, 0.3]),
        "lift": np.array([0.6]),
        "arm": np.array([0.2]),
        "wrist": np.array([0.1, 0.2, 0.3]),
        "gripper": np.array([0.4, 0.4]),
    }
    assert encoder.state_dim == STATE_DIM
    assert encoder.action_dim == ACTION_DIM
    assert np.allclose(encoder.state(qpos), encode_state(qpos))
