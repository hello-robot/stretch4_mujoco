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
    HEAD_CAMERA_LEFT,
    HEAD_CAMERA_LEFT_MJCF_NAME,
    HEAD_CAMERA_MJCF_NAME,
    HEAD_CAMERA_RIGHT,
    HEAD_CAMERA_RIGHT_MJCF_NAME,
    WRIST_CAMERA_LEFT,
    WRIST_CAMERA_RIGHT,
    WRIST_CAMERA_STEREO,
    WRIST_LEFT_CAMERA_MJCF_NAME,
    WRIST_RIGHT_CAMERA_MJCF_NAME,
    WRIST_STEREO_CAMERA_MJCF_NAME,
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
        WRIST_CAMERA_LEFT: WRIST_LEFT_CAMERA_MJCF_NAME,
        WRIST_CAMERA_RIGHT: WRIST_RIGHT_CAMERA_MJCF_NAME,
        WRIST_CAMERA_STEREO: WRIST_STEREO_CAMERA_MJCF_NAME,
        HEAD_CAMERA_LEFT: HEAD_CAMERA_LEFT_MJCF_NAME,
        HEAD_CAMERA_RIGHT: HEAD_CAMERA_RIGHT_MJCF_NAME,
    }


def test_head_camera_is_the_centre_of_the_head_assembly():
    """Pin which of Stretch 4's three head cameras the datasets and policies use.

    The head is a fixed assembly with a centre camera and a stereo pair either
    side, and they do not see the same thing: measured on the compiled MJCF, the
    centre camera sits 1.62m up looking 35 degrees down, while the pair sit 7.5cm
    to each side looking 47 degrees down. Everything here uses the centre one.
    """
    assert HEAD_CAMERA_MJCF_NAME == "camera_center_link"
    assert WRIST_LEFT_CAMERA_MJCF_NAME == "gripper_camera_left_rgb"


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
        images={HEAD_CAMERA: np.zeros((8, 8, 3), np.uint8), WRIST_CAMERA_LEFT: None},
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


def _make_run(root, count):
    import h5py

    for index in range(count):
        house = root / f"house_{index}"
        if house.exists():
            continue
        house.mkdir(parents=True)
        with h5py.File(house / "trajectories.h5", "w") as h5_file:
            h5_file.create_group("traj_0")


def _houses_on_disk(task):
    return {
        split: {p.name for p in (task / split).iterdir() if p.name.startswith("house_")}
        for split in ("train", "val")
    }


def test_regrown_run_does_not_leave_a_house_in_both_splits(tmp_path):
    """Growing a rollout directory moves houses across the split boundary.

    Houses are assigned by position, so adding more reassigns some. The previous
    placement has to go: leaving it puts the same trajectories in train and val,
    which is what happened in data/stretch_pick (5 houses, 13% of val frames).
    """
    run, task = tmp_path / "run", tmp_path / "task"
    _make_run(run, 10)
    first_val = {p.name for p in hdf5_layout.arrange_train_val_split(run, task, val_fraction=0.2)["val"]}

    _make_run(run, 25)  # regenerate larger; house_10..house_24 appear
    second_val = {p.name for p in hdf5_layout.arrange_train_val_split(run, task, val_fraction=0.2)["val"]}

    assert first_val != second_val, "test is vacuous unless the assignment actually moved"

    on_disk = _houses_on_disk(task)
    assert not (on_disk["train"] & on_disk["val"]), sorted(on_disk["train"] & on_disk["val"])
    assert on_disk["train"] | on_disk["val"] == {f"house_{i}" for i in range(25)}


def test_a_stray_symlink_into_the_wrong_split_is_repaired(tmp_path):
    """A symlink is something this function could have written, so it cleans it up.

    Repaired rather than raised: the layout it would produce is correct once the
    stale link is gone, and there is nothing a person needs to decide.
    """
    run, task = tmp_path / "run", tmp_path / "task"
    _make_run(run, 10)
    placed = hdf5_layout.arrange_train_val_split(run, task, val_fraction=0.2)

    # Put a train house into val/ as a stray `ln -s` would.
    stray = placed["train"][0].name
    (task / "val" / stray).symlink_to((run / stray).resolve(), target_is_directory=True)
    assert stray in _houses_on_disk(task)["val"]  # the bad state really exists

    hdf5_layout.arrange_train_val_split(run, task, val_fraction=0.2)

    on_disk = _houses_on_disk(task)
    assert not (on_disk["train"] & on_disk["val"])
    assert stray in on_disk["train"] and stray not in on_disk["val"]


def test_split_leaves_a_hand_made_directory_alone_but_still_raises(tmp_path):
    """A real directory is not something to delete on a person's behalf."""
    run, task = tmp_path / "run", tmp_path / "task"
    _make_run(run, 10)
    placed = hdf5_layout.arrange_train_val_split(run, task, val_fraction=0.2)

    stray = placed["val"][0].name
    (task / "train" / stray).mkdir()
    (task / "train" / stray / "notes.txt").write_text("put here on purpose")

    with pytest.raises(hdf5_layout.DuplicateSplitError):
        hdf5_layout.arrange_train_val_split(run, task, val_fraction=0.2)
    assert (task / "train" / stray / "notes.txt").read_text() == "put here on purpose"


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


def test_generate_dataset_cli_supports_slow_rate():
    from click.testing import CliRunner

    from examples.machine_learning.molmospaces.finetuning.generate_dataset import main

    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "--slow-rate" in result.output
    assert "--slow_rate" in result.output


def test_stretch_rollout_runner_slow_rate_timing(monkeypatch):
    import time
    from unittest.mock import MagicMock

    from examples.machine_learning.molmospaces.finetuning.generate_dataset import (
        StretchRolloutRunner,
    )

    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

    # Mock task with mock env and 2 steps
    task = MagicMock()
    step_calls = 0

    def mock_step_chunk(chunk, stop_on_success=False):
        nonlocal step_calls
        step_calls += 1
        # Advance mock sim time by 0.1s each step
        task.env.mj_datas[0].time += 0.1
        return MagicMock(), 0.0, False, False, [{}]

    task.step_chunk.side_effect = mock_step_chunk
    task.is_done.side_effect = lambda: step_calls >= 2
    task.reset.return_value = (MagicMock(), {})
    task.judge_success.return_value = True
    task.env.current_batch_index = 0
    task.env.mj_datas = [MagicMock(time=0.0)]

    policy = MagicMock()
    policy.get_action_chunk.return_value = [{"action": 1}]

    StretchRolloutRunner.slow_rate = 2.0
    success = StretchRolloutRunner.run_single_rollout(
        episode_seed=42,
        task=task,
        policy=policy,
    )
    assert success is True
    assert len(slept) == 2
    # dt_sim = 0.1s, slow_rate = 2.0 -> target_wall_dt = 0.2s. Sleep should be ~0.2s
    for s in slept:
        assert 0.15 <= s <= 0.25


def test_snap_free_camera_to_robot():
    import mujoco
    from unittest.mock import MagicMock

    from examples.machine_learning.molmospaces.finetuning.generate_dataset import (
        snap_free_camera_to_robot,
    )

    viewer = MagicMock()
    viewer.cam.lookat = [0.0, 0.0, 0.0]

    task = MagicMock()
    task.env.current_batch_index = 0
    model = MagicMock()
    data = MagicMock()

    body_mock = MagicMock()
    body_mock.id = 1
    model.body.side_effect = lambda name: body_mock if name == "robot_0/base_link" else (_ for _ in ()).throw(KeyError)
    data.xpos = {1: np.array([1.2, 3.4, 0.1])}
    task.env.current_model = model
    task.env.mj_datas = [data]

    snap_free_camera_to_robot(viewer, task)

    assert viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FREE
    assert viewer.cam.fixedcamid == -1
    assert viewer.cam.lookat[0] == pytest.approx(1.2)
    assert viewer.cam.lookat[1] == pytest.approx(3.4)
    assert viewer.cam.lookat[2] == pytest.approx(0.7)
    assert viewer.cam.distance == 2.5
    assert viewer.cam.elevation == -20.0
    assert viewer.cam.azimuth == 135.0


def test_stretch_rerun_visualizer_extracts_pickup_object_and_logs(monkeypatch):
    import mujoco
    import rerun as rr
    from unittest.mock import MagicMock

    from examples.machine_learning.molmospaces.finetuning.generate_dataset import (
        StretchRerunVisualizer,
    )

    logged_entities = {}
    logged_static = {}

    def mock_log(entity, data, static=False):
        logged_entities[entity] = data
        if static:
            logged_static[entity] = data

    inits = []
    def mock_init(app_id, recording_id=None, **kw):
        inits.append((app_id, recording_id))

    connected_urls = []
    def mock_connect_grpc(url=None, **kw):
        connected_urls.append(url)

    rec_names = []
    def mock_send_recording_name(name, **kw):
        rec_names.append(name)

    monkeypatch.setattr(rr, "log", mock_log)
    monkeypatch.setattr(rr, "init", mock_init)
    monkeypatch.setattr(rr, "spawn", lambda *a, **kw: None)
    monkeypatch.setattr(rr, "set_time", lambda *a, **kw: None)
    monkeypatch.setattr(rr, "connect_grpc", mock_connect_grpc)
    monkeypatch.setattr(rr, "send_recording_name", mock_send_recording_name)

    viz = StretchRerunVisualizer(spawn=True, port=9876)

    # Mock task
    task = MagicMock()
    task.get_task_objects.return_value = {"pickup_obj": "apple_123"}
    task.env.current_batch_index = 0

    model = MagicMock()
    model.ngeom = 3
    model.nbody = 4

    # Robot body 1
    # Object body 2
    # Scene object body 3
    body_0 = MagicMock()
    body_0.name = "world"
    body_1 = MagicMock()
    body_1.name = "robot_0/base_link"
    body_2 = MagicMock()
    body_2.name = "apple_123"
    body_3 = MagicMock()
    body_3.name = "table_0"
    body_map = {0: body_0, 1: body_1, 2: body_2, 3: body_3}
    model.body.side_effect = lambda b_id: body_map[b_id] if isinstance(b_id, int) else (
        body_1 if "robot_0" in str(b_id) else (body_3 if "table" in str(b_id) else body_2)
    )

    mesh_0 = MagicMock()
    mesh_0.name = "base_link"
    mesh_1 = MagicMock()
    mesh_1.name = "apple_mesh"
    mesh_2 = MagicMock()
    mesh_2.name = "table_mesh"
    mesh_map = {0: mesh_0, 1: mesh_1, 2: mesh_2}
    model.mesh.side_effect = lambda m_id: mesh_map[m_id]

    model.geom_bodyid = [1, 2, 3]
    model.geom_type = [mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_MESH]
    model.geom_dataid = [0, 1, 2]
    model.mesh_vertadr = [0, 3, 6]
    model.mesh_vertnum = [3, 3, 3]
    model.mesh_vert = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
        [0.5, 0, 0],
        [0, 0.5, 0],
        [0, 0, 0],
        [2.0, 0, 0],
        [0, 2.0, 0],
    ], dtype=np.float32)
    model.mesh_face = np.array([
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
    ], dtype=np.uint32)
    model.geom_pos = [np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0])]
    model.geom_quat = [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
    model.body_parentid = [0, 0, 0, 0]

    data = MagicMock()
    data.xpos = [np.array([0, 0, 0]), np.array([1, 2, 0]), np.array([3, 4, 0]), np.array([5, 6, 0])]
    data.xmat = [np.eye(3).flatten(), np.eye(3).flatten(), np.eye(3).flatten(), np.eye(3).flatten()]
    data.time = 0.5

    task.env.current_model = model
    task.env.mj_datas = [data]

    pickup_name = StretchRerunVisualizer._get_pickup_object_name(task)
    assert pickup_name == "apple_123"

    # Mock policy with grasp and waypoints
    mock_policy = MagicMock()
    mock_grasp = MagicMock()
    mock_grasp.position = np.array([0.5, 0.2, 0.8])
    mock_grasp.rotation = np.eye(3)
    mock_grasp.authored = True
    mock_grasp.approach_yaw = 0.5
    mock_grasp.wrist_pitch = 0.1
    mock_grasp.wrist_roll = -0.2
    mock_policy._grasp = mock_grasp

    wp1 = MagicMock()
    wp1.label = "raise"
    wp1.position = np.array([0.5, 0.2, 0.8])
    wp1.wrist_pitch = 0.1
    wp1.wrist_roll = -0.2
    wp1.approach_yaw = None
    wp1.gripper_open = True
    wp1.grip_width_m = 0.05
    wp1.tolerance = 0.03
    wp1.establishes_grasp = False
    wp1.verify_grasp = False
    wp1.settle_steps = 0

    wp2 = MagicMock()
    wp2.label = "reach"
    wp2.position = np.array([0.5, 0.2, 0.7])
    wp2.wrist_pitch = 0.1
    wp2.wrist_roll = -0.2
    wp2.approach_yaw = 0.5
    wp2.gripper_open = True
    wp2.grip_width_m = 0.05
    wp2.tolerance = 0.03
    wp2.establishes_grasp = True
    wp2.verify_grasp = True
    wp2.settle_steps = 5

    mock_policy._plan = [wp1, wp2]
    mock_policy._waypoint_index = 0
    mock_policy._steps_in_waypoint = 3
    mock_policy._grasp_lost = False
    mock_policy._grasp_offset = None

    # Start episode 1
    viz.start_episode(101, task, policy=mock_policy)
    assert len(inits) == 1
    assert "episode_101_" in inits[0][1]
    assert len(connected_urls) == 1
    assert "9876" in connected_urls[0]
    assert rec_names == ["Episode 101"]
    assert "world/robot/robot_0_base_link/geom_0" in logged_static
    assert "world/object/apple_123/geom_1" in logged_static
    assert "world/scene_objects/table_0/geom_2" in logged_static

    viz.log_step(0, task, observation=[{"head_camera": np.zeros((10, 10, 3), dtype=np.uint8)}], policy=mock_policy)

    assert "world/robot/robot_0_base_link" in logged_entities
    assert "world/object/apple_123" in logged_entities
    assert "world/scene_objects/table_0" in logged_entities
    assert "world/frames/wrist_center" in logged_entities
    assert "world/frames/wrist_center/axes" in logged_entities
    assert "world/frames/wrist_center/label" in logged_entities
    assert "world/frames/tool_center" in logged_entities
    assert "world/frames/tool_center/axes" in logged_entities
    assert "world/frames/tool_center/label" in logged_entities
    assert "world/frames/object" in logged_entities
    assert "world/frames/object/axes" in logged_entities
    assert "world/frames/object/label" in logged_entities
    assert "world/frames/target_grasp" in logged_entities
    assert "world/frames/target_grasp/axes" in logged_entities
    assert "world/frames/target_grasp/label" in logged_entities
    assert "logs/waypoints" in logged_entities
    assert "planner/waypoint" in logged_entities
    assert "world/cameras/head_camera" in logged_entities

    # Start episode 2 -> verifies new recording is created and connected
    viz.start_episode(102, task, policy=mock_policy)
    assert len(inits) == 2
    assert "episode_102_" in inits[1][1]
    assert inits[0][1] != inits[1][1]
    assert len(connected_urls) == 2
    assert rec_names == ["Episode 101", "Episode 102"]


def test_fisheye_distortion_scaling_arbitrary_resolutions():
    """Verify fisheye distortion applies correctly across different resolutions without clipping."""
    from stretch4_mujoco.enums.stretch_cameras import StretchCameras

    cb_left = StretchCameras.cam_nav_rgb_se4_left.post_processing_callback
    cb_right = StretchCameras.cam_nav_rgb_se4_right.post_processing_callback
    assert cb_left is not None
    assert cb_right is not None

    for shape in [(224, 224, 3), (1200, 1920, 3), (480, 640, 3)]:
        img = np.ones(shape, dtype=np.uint8) * 200
        out_l = cb_left(img)
        out_r = cb_right(img)
        assert out_l.shape == shape
        assert out_r.shape == shape
        assert np.count_nonzero(out_l) > 0, f"Left fisheye output was all black for {shape}"
        assert np.count_nonzero(out_r) > 0, f"Right fisheye output was all black for {shape}"


def test_cpumujocoenv_fisheye_distortion_hook():
    """Verify that CPUMujocoEnv.render_rgb_frame applies fisheye distortion for left and right nav cameras."""
    from molmo_spaces.env.env import CPUMujocoEnv

    class MockCamera:
        pos = np.array([0.0, 0.0, 0.0])
        forward = np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        fov = 120.0

    class MockCameraManager:
        registry = {
            "head_camera": MockCamera(),
            "wrist_camera": MockCamera(),
            "head_camera_left": MockCamera(),
            "head_camera_right": MockCamera(),
        }

    class MockEnv(CPUMujocoEnv):
        def __init__(self):
            self._executor = None
            self._renderer = None
            self.object_managers = []
            self.camera_manager = MockCameraManager()

        def _render_frame(self, pos, forward, up, fov, segmentation=False):
            return np.ones((224, 224, 3), dtype=np.uint8) * 200

    env = MockEnv()
    raw_head = env.render_rgb_frame("head_camera")
    assert np.all(raw_head == 200)  # Untouched

    distorted_left = env.render_rgb_frame("head_camera_left")
    assert distorted_left.shape == (224, 224, 3)
    # Circular aperture leaves edges black (0) while center retains pixels
    assert np.count_nonzero(distorted_left == 0) > 0
    assert np.count_nonzero(distorted_left > 0) > 0

    distorted_right = env.render_rgb_frame("head_camera_right")
    assert distorted_right.shape == (224, 224, 3)
    assert np.count_nonzero(distorted_right == 0) > 0
    assert np.count_nonzero(distorted_right > 0) > 0


def test_camera_feature_names_and_defaults_in_lerobot_export():
    """Verify LeRobot export maps all 4 cameras to observation features."""
    from examples.machine_learning.molmospaces.finetuning.lerobot_export import (
        CAMERA_FEATURE_NAMES,
        DEFAULT_CAMERA_NAMES,
    )

    assert HEAD_CAMERA in CAMERA_FEATURE_NAMES
    assert WRIST_CAMERA_LEFT in CAMERA_FEATURE_NAMES
    assert HEAD_CAMERA_LEFT in CAMERA_FEATURE_NAMES
    assert HEAD_CAMERA_RIGHT in CAMERA_FEATURE_NAMES
    assert CAMERA_FEATURE_NAMES[HEAD_CAMERA_LEFT] == "observation.images.head_left"
    assert set(DEFAULT_CAMERA_NAMES) == {
        HEAD_CAMERA,
        WRIST_CAMERA_LEFT,
        WRIST_CAMERA_RIGHT,
        HEAD_CAMERA_LEFT,
        HEAD_CAMERA_RIGHT,
    }


def test_fisheye_active_fill_coverage_not_overly_vignetted():
    """Verify fisheye distortion does not mask out most of the image with excessive black vignetting."""
    from stretch4_mujoco.enums.stretch_cameras import StretchCameras

    cb_left = StretchCameras.cam_nav_rgb_se4_left.post_processing_callback
    img = np.ones((224, 224, 3), dtype=np.uint8) * 200
    out = cb_left(img)
    non_black_ratio = np.count_nonzero(np.any(out > 0, axis=-1)) / (224 * 224)
    # The active image area should be >= 70% of the frame (corners only are curved out)
    assert non_black_ratio >= 0.70, f"Fisheye active area ratio {non_black_ratio:.2f} is too low (excessive black borders)"


def test_finetune_camera_selection_parsing_and_commands(tmp_path):
    """Verify finetune.py camera selection parsing and command generation."""
    import json
    from examples.machine_learning.molmospaces.finetuning.finetune import (
        DatasetSummary,
        parse_camera_names,
        trainer_command,
        write_trainer_config,
    )

    # 1. Alias parsing
    cams = parse_camera_names("head,wrist,left")
    assert cams == ["head_camera", "wrist_camera_left", "head_camera_left"]

    # 2. MolmoBot command with selected cameras
    summary_molmo = DatasetSummary(
        root=tmp_path / "molmobot_test",
        kind="molmospaces",
        action_space="stretch_move_groups",
        state_dim=10,
        action_dim=10,
        num_episodes=5,
        num_frames=100,
        fps=15.0,
        video_keys=["head_camera", "wrist_camera", "head_camera_left", "head_camera_right"],
    )
    summary_molmo.root.mkdir(parents=True, exist_ok=True)
    cfg_molmo = write_trainer_config(
        datasets=[summary_molmo],
        trainer="molmobot",
        base_checkpoint="8b",
        output_dir=tmp_path / "checkpoints",
        batch_size=16,
        steps=1000,
        learning_rate=1e-5,
        action_type="joint_pos_rel",
        camera_names=["head_camera", "head_camera_left"],
    )
    cmd_molmo = trainer_command(
        datasets=[summary_molmo],
        trainer="molmobot",
        config_path=cfg_molmo,
        base_checkpoint="8b",
        output_dir=tmp_path / "checkpoints",
        batch_size=16,
        steps=1000,
        action_type="joint_pos_rel",
        seq_len=2048,
        camera_names=["head_camera", "head_camera_left"],
    )
    assert "--camera_names" in cmd_molmo
    cam_idx = cmd_molmo.index("--camera_names")
    assert cmd_molmo[cam_idx + 1 : cam_idx + 3] == ["head_camera", "head_camera_left"]

    molmo_json = json.loads(cfg_molmo.read_text())
    assert molmo_json["action"]["camera_names"] == ["head_camera", "head_camera_left"]

    # 3. LeRobot feature filtering with selected cameras
    summary_lerobot = DatasetSummary(
        root=tmp_path / "lerobot_test",
        kind="lerobot",
        action_space="stretch_move_groups",
        state_dim=10,
        action_dim=10,
        num_episodes=5,
        num_frames=100,
        fps=15.0,
        video_keys=[
            "observation.images.head",
            "observation.images.wrist",
            "observation.images.head_left",
            "observation.images.head_right",
        ],
    )
    summary_lerobot.root.mkdir(parents=True, exist_ok=True)
    cfg_lerobot = write_trainer_config(
        datasets=[summary_lerobot],
        trainer="lerobot",
        base_checkpoint="pi05_droid",
        output_dir=tmp_path / "checkpoints",
        batch_size=16,
        steps=1000,
        learning_rate=1e-5,
        action_type="joint_pos_rel",
        camera_names=["head_camera", "head_camera_right"],
    )
    lerobot_json = json.loads(cfg_lerobot.read_text())
    assert lerobot_json["features"]["images"] == [
        "observation.images.head",
        "observation.images.head_right",
    ]


def test_finetune_writes_a_runnable_molmobot_script(tmp_path):
    """The generated script is valid bash, and says the things that are easy to get wrong.

    Every failure this guards against is silent at generation time and loud a
    long way downstream: a mis-quoted `"$DATASET"` sends validate_trajectories at
    a directory literally named `$DATASET`; a `--data_paths` with the split
    already on the end makes MolmoBot look for `train/train`; and a missing
    `val` pass leaves out an index `SynthmanipDataset` raises without.
    """
    import subprocess
    from pathlib import Path

    from examples.machine_learning.molmospaces.finetuning.finetune import (
        VRAM_TIERS,
        DatasetSummary,
        write_launch_script,
        write_trainer_config,
    )
    from examples.machine_learning.molmospaces.finetuning.molmobot_repo import (
        PACKAGE_SUBDIR,
        TRAINER_SCRIPT,
        MolmoBotCheckout,
    )

    dataset = tmp_path / "molmobot" / "pick"
    dataset.mkdir(parents=True)
    checkout = MolmoBotCheckout(
        root=tmp_path / "MolmoBot",
        package_dir=tmp_path / "MolmoBot" / PACKAGE_SUBDIR,
        data_scripts_dir=tmp_path / "MolmoBot" / "data_scripts",
    )
    (checkout.package_dir / TRAINER_SCRIPT).parent.mkdir(parents=True)

    summary = DatasetSummary(
        root=dataset,
        kind="molmospaces",
        action_space="stretch_move_groups",
        state_dim=10,
        action_dim=10,
        num_episodes=8,
        num_frames=0,
        fps=15.0,
        video_keys=["head_camera_right", "wrist_camera_right"],
        splits={"train": 1, "val": 1},
    )
    config_path = write_trainer_config(
        datasets=[summary],
        trainer="molmobot",
        base_checkpoint="8b",
        output_dir=tmp_path / "checkpoints",
        batch_size=16,
        steps=1000,
        learning_rate=1e-5,
        action_type="joint_pos_rel",
    )
    script = write_launch_script(
        datasets=[summary],
        trainer="molmobot",
        config_path=config_path,
        base_checkpoint="8b",
        output_dir=tmp_path / "checkpoints",
        batch_size=16,
        steps=1000,
        action_type="joint_pos_rel",
        seq_len=2048,
        checkout=checkout,
    )
    body = script.read_text()

    # 1. bash parses it, and it is executable
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0
    assert script.stat().st_mode & 0o111

    # 2. shell variables survive quoting rather than being emitted literally
    assert '"$SCRIPTS"/validate_trajectories.py "$DATASET"/train' in body
    assert '"$SCRIPTS"/validate_trajectories.py "$DATASET"/val' in body
    assert "'\"$DATASET\"'" not in body

    # 3. --data_paths takes the task directory; MolmoBot appends the split itself
    assert "--data_paths \"$DATASET\" " in body
    assert '--data_paths "$DATASET"/train' not in body

    # 4. the gripper's min_max lookup needs joint_pos alongside the action type
    assert "--keys actions/joint_pos_rel actions/joint_pos obs/agent/qpos" in body

    # 5. no argument reaches torchrun or the trainer with its $VAR quoted shut
    assert '\'--nproc-per-node="$NPROC"\'' not in body
    assert '--nproc-per-node="$NPROC"' in body

    # 6. the three OmegaConf dotlist fields, none of which is an argparse option:
    #    save_folder is TrainConfig's only mandatory value, wandb interpolates
    #    two environment variables that are normally unset, and max_duration is
    #    otherwise hardcoded to 200000 -- all three are resolved in one call, so
    #    each missing one fails only after the checkpoint and stats are loaded.
    assert '--save_folder="$SAVE_FOLDER"' in body
    assert 'mkdir -p "$SAVE_FOLDER"' in body
    assert "--wandb=null" in body
    assert '--max_duration="$MAX_STEPS"' in body
    assert 'MAX_STEPS="${MAX_STEPS:-1000}"' in body

    # 7. the norm stats stay with the run, not in the shared MolmoBot checkout
    assert '--stats_path "$SAVE_FOLDER"/synthmanip_norm_stats.yaml' in body

    # 7b. the VRAM knobs are overridable variables at the top, so an OOM is
    #     retuned from the environment rather than by regenerating. An explicit
    #     seq_len pins the value; device_batch_size was left to the VRAM table.
    #     device_batch_size must reach the trainer either way -- MolmoBot's own
    #     default is 2, not 1, so omitting the flag silently inherits it.
    assert 'SEQ_LEN="${SEQ_LEN:-2048}"' in body
    assert 'DEVICE_BATCH="${DEVICE_BATCH:-$DEVICE_BATCH_AUTO}"' in body
    assert 'GLOBAL_BATCH="${GLOBAL_BATCH:-16}"' in body
    assert '--seq_len "$SEQ_LEN"' in body
    assert '--device_batch_size "$DEVICE_BATCH"' in body
    assert '--global_batch_size "$GLOBAL_BATCH"' in body

    # 7c. the table always assigns, including its last row: an unset
    #     SEQ_LEN_AUTO under `set -u` would kill the run at the training line
    assert body.count("SEQ_LEN_AUTO=") == len(VRAM_TIERS)
    assert "else SEQ_LEN_AUTO=" in body
    assert '-ge 0 ]' not in body

    # 7d. optional flags are listed commented-out, and wandb is off by default
    #     because its config interpolates two normally-unset environment variables
    assert "EXTRA_ARGS=()" in body
    assert "# EXTRA_ARGS+=(--img_aug)" in body
    assert "EXTRA_ARGS+=(--wandb=null)" in body
    assert '"${EXTRA_ARGS[@]}"' in body

    # 7d-ii. the freezing tiers reach the trainer as real ft_* dotlist fields,
    #        and the per-component learning rates are exposed. There is no LoRA
    #        flag because MolmoBot has no LoRA -- do not invent one.
    assert 'TRAINABLE="${TRAINABLE:-action_expert}"' in body
    assert "EXTRA_ARGS+=(--ft_vit=True)" in body
    assert "--ft_vit=True --ft_llm=True --ft_connector=True" in body
    assert '--optimizer.action_expert_learning_rate="$ACTION_EXPERT_LR"' in body
    # No LoRA flag reaches the trainer -- MolmoBot has no LoRA to turn on. The
    # comments may name it; the executed lines must not.
    executed = [line for line in body.splitlines() if not line.lstrip().startswith("#")]
    assert not [line for line in executed if "lora" in line.lower()]

    # 7e. training runs through the progress filter, and PROGRESS=off skips the
    #     pipe entirely -- both branches must invoke the same trainer command
    assert 'if [ "$PROGRESS" = on ]; then' in body
    assert '2>&1 | "$PYTHON" -u "$PROGRESS_FILTER"' in body
    assert body.count("launch_scripts/train_molmobot.py") == 2
    assert '--log_interval "$LOG_INTERVAL"' in body
    filter_path = Path([
        line.split("=", 1)[1] for line in body.splitlines() if line.startswith("PROGRESS_FILTER=")
    ][0])
    assert filter_path.exists(), "the generated script points at a filter that is not there"

    # 8. an empty split is not validated -- there is no directory to index
    solo = DatasetSummary(**{**summary.__dict__, "splits": {"train": 1, "val": 0}})
    solo_body = write_launch_script(
        datasets=[solo],
        trainer="molmobot",
        config_path=config_path,
        base_checkpoint="8b",
        output_dir=tmp_path / "checkpoints",
        batch_size=16,
        steps=1000,
        action_type="joint_pos_rel",
        seq_len=2048,
        checkout=checkout,
    ).read_text()
    assert '"$DATASET"/val' not in solo_body


def test_generated_script_never_rewarps_an_already_fisheye_camera(tmp_path):
    """`--cameras_to_warp` must stay empty for the cameras that render distorted.

    `stretch/config.py` gives `head_camera_left` and `head_camera_right` a ~123
    degree FOV and a barrel-distortion callback that runs before the MP4 is
    written, so the lens is already in the data. MolmoBot's `--cameras_to_warp`
    applies a second GoPro-style distortion -- and the result still looks like a
    plausible wide-angle photo, so a mistake here is invisible in every artefact
    a person would inspect.
    """
    from examples.machine_learning.molmospaces.finetuning.finetune import (
        FISHEYE_CAMERA_NAMES,
        DatasetSummary,
        write_launch_script,
        write_trainer_config,
    )
    from examples.machine_learning.molmospaces.finetuning.molmobot_repo import (
        PACKAGE_SUBDIR,
        TRAINER_SCRIPT,
        MolmoBotCheckout,
    )
    from examples.machine_learning.molmospaces.stretch.config import (
        HEAD_CAMERA,
        HEAD_CAMERA_LEFT,
        HEAD_CAMERA_RIGHT,
        WRIST_CAMERA_LEFT,
        WRIST_CAMERA_RIGHT,
    )

    # The set has to match the cameras config.py actually distorts, or the
    # warning lands on the wrong camera name.
    assert FISHEYE_CAMERA_NAMES == {HEAD_CAMERA_LEFT, HEAD_CAMERA_RIGHT}
    assert HEAD_CAMERA not in FISHEYE_CAMERA_NAMES
    assert not FISHEYE_CAMERA_NAMES & {WRIST_CAMERA_LEFT, WRIST_CAMERA_RIGHT}

    checkout = MolmoBotCheckout(
        root=tmp_path / "MolmoBot",
        package_dir=tmp_path / "MolmoBot" / PACKAGE_SUBDIR,
        data_scripts_dir=tmp_path / "MolmoBot" / "data_scripts",
    )
    (checkout.package_dir / TRAINER_SCRIPT).parent.mkdir(parents=True)
    root = tmp_path / "molmobot" / "pick"
    root.mkdir(parents=True)
    summary = DatasetSummary(
        root=root,
        kind="molmospaces",
        action_space="stretch_move_groups",
        state_dim=10,
        action_dim=10,
        num_episodes=8,
        num_frames=0,
        fps=15.0,
        video_keys=[HEAD_CAMERA_RIGHT, WRIST_CAMERA_RIGHT],
        splits={"train": 3, "val": 1},
    )

    def build(cameras):
        config_path = write_trainer_config(
            datasets=[summary],
            trainer="molmobot",
            base_checkpoint="allenai/MolmoBot-DROID",
            output_dir=tmp_path / "checkpoints",
            batch_size=16,
            steps=1000,
            learning_rate=1e-5,
            action_type="joint_pos_rel",
        )
        return write_launch_script(
            datasets=[summary],
            trainer="molmobot",
            config_path=config_path,
            base_checkpoint="allenai/MolmoBot-DROID",
            output_dir=tmp_path / "checkpoints",
            batch_size=16,
            steps=1000,
            action_type="joint_pos_rel",
            seq_len=528,
            camera_names=cameras,
            checkout=checkout,
        ).read_text()

    # 1. a fisheye camera is never warped by default, and the script says why
    wide = build([HEAD_CAMERA_RIGHT, WRIST_CAMERA_RIGHT])
    assert 'WARP_CAMERAS="${WARP_CAMERAS:-}"' in wide
    assert f"{HEAD_CAMERA_RIGHT} already carries the wide lens" in wide
    assert f"{WRIST_CAMERA_RIGHT} is rectilinear" in wide

    # 2. the wide lens is also the argument for unfreezing the vision tower,
    #    which TRAINABLE=vision does
    assert 'TRAINABLE="${TRAINABLE:-action_expert}"' in wide
    assert "vision         + the vision tower" in wide

    # 3. an all-rectilinear selection gets the opposite advice, and is still
    #    warned off the two cameras that must never be listed
    narrow = build([HEAD_CAMERA, WRIST_CAMERA_RIGHT])
    assert "render rectilinear" in narrow
    assert "must never" in narrow
    assert "already carries the wide lens" not in narrow


def test_stretch_presets_are_registered_in_a_pristine_checkout(tmp_path):
    """A fresh clone has no Stretch preset, and MolmoBot cannot be told one.

    `--action_dim` and `--action_move_groups` give MolmoBot the width and the
    names, but the *per group* widths come only from
    `ACTION_SPECS[args.action_preset]`. With nothing matched, `train_molmobot.py`
    raises `Action spec must be specified via --action_preset` -- and it does so
    after the data paths validate, so it looks like a data problem. This is what
    made a working machine and a fresh one behave differently.
    """
    from examples.machine_learning.molmospaces.finetuning.finetune import STRETCH_ACTION_SPEC
    from examples.machine_learning.molmospaces.finetuning.molmobot_repo import (
        PRESETS_MODULE,
        STRETCH_PRESET_NAMES,
        MolmoBotCheckout,
        MolmoBotSetupError,
        ensure_stretch_presets,
    )

    package = tmp_path / "MolmoBot"
    presets = package / PRESETS_MODULE
    presets.parent.mkdir(parents=True)
    # The shape of MolmoBot's table, minus everything that does not matter here.
    # STATE_SPECS is part of that shape: Stretch's state is a gripper narrower
    # than its action, and this is the only place MolmoBot can be told so.
    presets.write_text(
        "from typing import Dict, List, Optional, Union\n\n"
        'ACTION_SPECS: Dict[str, Dict[str, int]] = {\n'
        '    "franka_joint": {"arm": 7, "gripper": 1},\n'
        "}\n\n"
        'ACTION_DATASET_KEYS: Dict[str, Union[str, Dict[str, str]]] = {\n'
        '    "franka_joint": "joint_pos",\n'
        "}\n\n"
        'STATE_SPECS: Dict[str, Dict[str, int]] = {\n'
        '    "RBY1_multitask": {"base": 3, "torso": 3},\n'
        "}\n"
    )
    checkout = MolmoBotCheckout(
        root=tmp_path,
        package_dir=package,
        data_scripts_dir=tmp_path / "data_scripts",
    )

    # 1. the presets are added, and adding them twice is a no-op
    assert ensure_stretch_presets(checkout, STRETCH_ACTION_SPEC) == [
        "stretch_joint",
        "stretch_jointdelta",
    ]
    assert ensure_stretch_presets(checkout, STRETCH_ACTION_SPEC) == []

    # 2. the result is importable, keeps MolmoBot's own presets, and carries the
    #    move-group order the evaluation policy unpacks in
    namespace: dict = {}
    exec(compile(presets.read_text(), str(presets), "exec"), namespace)  # noqa: S102
    specs, keys = namespace["ACTION_SPECS"], namespace["ACTION_DATASET_KEYS"]
    assert specs["franka_joint"] == {"arm": 7, "gripper": 1}, "upstream presets survive"
    for action_type, name in STRETCH_PRESET_NAMES.items():
        assert specs[name] == dict(STRETCH_ACTION_SPEC)
        assert list(specs[name]) == list(STRETCH_ACTION_SPEC), "order is the unpack order"
        assert sum(specs[name].values()) == 10
        assert keys[name] == action_type, "the preset must read the action type it is named for"

    # 3. a preset table this cannot recognise is a clear error, not a silent skip
    broken = tmp_path / "broken" / PRESETS_MODULE
    broken.parent.mkdir(parents=True)
    broken.write_text("ACTION_SPECS = dict(franka_joint={})\n")
    with pytest.raises(MolmoBotSetupError, match="ACTION_SPECS"):
        ensure_stretch_presets(
            MolmoBotCheckout(
                root=tmp_path / "broken",
                package_dir=tmp_path / "broken" / "MolmoBot",
                data_scripts_dir=tmp_path / "broken" / "data_scripts",
            ),
            STRETCH_ACTION_SPEC,
        )


def test_train_progress_passes_the_log_through_and_draws_a_bar():
    """The progress filter must never eat a line, whatever it does with the bar.

    It sits on the end of the training pipeline, so it is the only thing between
    a crashing trainer and the terminal. A regex that swallowed unmatched lines,
    or an exception on a line it did not expect, would turn a real stack trace
    into silence.
    """
    import io
    import subprocess
    import sys
    from contextlib import redirect_stdout

    from examples.machine_learning.molmospaces.finetuning import train_progress

    header = "[2026-01-01 00:00:00] INFO [olmo.train.trainer:1306, rank=0] "
    lines = [
        "Set up torchrun environment",
        f"{header}[step=15000/30000, eta=1 hour, 2 minutes]",
        "    train/CrossEntropyLoss=1.2345",
        # ANSI colour from the logging handler must not break the match
        f"\033[36m{header}[step=30000/30000, eta=1 minute]\033[0m",
        "Traceback (most recent call last):",
    ]

    captured = io.StringIO()
    stdin = sys.stdin
    try:
        sys.stdin = io.StringIO("\n".join(lines) + "\n")
        with redirect_stdout(captured):
            assert train_progress.main() == 0
    finally:
        sys.stdin = stdin
    out = captured.getvalue()

    # 1. every input line survives, including the traceback after the last header
    for line in lines:
        assert line in out
    assert "Traceback (most recent call last):" in out

    # 2. the bar reflects the trainer's own step counter and carries its eta
    assert "50.0%  step 15,000/30,000" in out
    assert "1 hour, 2 minutes" in out
    assert "100.0%  step 30,000/30,000" in out
    assert train_progress.render_bar(1.0).strip("#") == ""
    assert train_progress.render_bar(0.0).strip("-") == ""

    # 3. a failing trainer still fails the pipeline -- the filter is last in it
    failing = "python -c 'print(\"boom\"); raise SystemExit(3)'"
    piped = subprocess.run(
        ["bash", "-c", f"set -euo pipefail; {failing} 2>&1 | python -u {train_progress.__file__}"],
        capture_output=True,
        text=True,
    )
    assert piped.returncode == 3
    assert "boom" in piped.stdout


def test_finetune_trains_several_tasks_as_one_mixture(tmp_path):
    """Several `--rollouts` become one policy, not one run each.

    MolmoBot conditions on each trajectory's task description, so pick and pnp
    share a checkpoint. The failure modes are all quiet: validation silently
    covering only the first task, a mixture weighted by accident, or two tasks
    whose paths are indistinguishable in the emitted command.
    """
    import subprocess

    from examples.machine_learning.molmospaces.finetuning.finetune import (
        DatasetSummary,
        write_launch_script,
        write_trainer_config,
    )
    from examples.machine_learning.molmospaces.finetuning.molmobot_repo import (
        PACKAGE_SUBDIR,
        TRAINER_SCRIPT,
        MolmoBotCheckout,
    )

    checkout = MolmoBotCheckout(
        root=tmp_path / "MolmoBot",
        package_dir=tmp_path / "MolmoBot" / PACKAGE_SUBDIR,
        data_scripts_dir=tmp_path / "MolmoBot" / "data_scripts",
    )
    (checkout.package_dir / TRAINER_SCRIPT).parent.mkdir(parents=True)

    def task(name: str) -> DatasetSummary:
        (tmp_path / "molmobot" / name).mkdir(parents=True)
        return DatasetSummary(
            root=tmp_path / "molmobot" / name,
            kind="molmospaces",
            action_space="stretch_move_groups",
            state_dim=10,
            action_dim=10,
            num_episodes=8,
            num_frames=0,
            fps=15.0,
            video_keys=["head_camera_right"],
            splits={"train": 3, "val": 1},
        )

    datasets = [task("pick"), task("pnp")]
    rates = [0.6, 0.4]
    config_path = write_trainer_config(
        datasets=datasets,
        trainer="molmobot",
        base_checkpoint="allenai/MolmoBot-DROID",
        output_dir=tmp_path / "checkpoints",
        batch_size=16,
        steps=1000,
        learning_rate=1e-5,
        action_type="joint_pos_rel",
        sample_rates=rates,
    )
    script = write_launch_script(
        datasets=datasets,
        trainer="molmobot",
        config_path=config_path,
        base_checkpoint="allenai/MolmoBot-DROID",
        output_dir=tmp_path / "checkpoints",
        batch_size=16,
        steps=1000,
        action_type="joint_pos_rel",
        seq_len=2048,
        sample_rates=rates,
        checkout=checkout,
    )
    body = script.read_text()

    # 1. the config and script live above the tasks, since the run spans them
    assert script.parent == tmp_path / "molmobot"
    assert config_path.parent == tmp_path / "molmobot"

    # 2. one variable per task, named after it
    assert "DATASET_PICK=" in body and "DATASET_PNP=" in body

    # 3. every split of every task is indexed, not just the first task's
    for name in ("PICK", "PNP"):
        for split in ("train", "val"):
            assert f'validate_trajectories.py "$DATASET_{name}"/{split}' in body

    # 4. one training command over both, weighted, with validation spanning both
    assert '--data_paths "$DATASET_PICK" "$DATASET_PNP"' in body
    assert '--val_data_paths "$DATASET_PICK" "$DATASET_PNP"' in body
    assert "--dataset_sample_rates 0.6 0.4" in body
    assert "--exp_name=stretch4_pick_pnp" in body

    # 4a. exactly one training invocation per PROGRESS branch, and the two must be
    #     the same command -- a knob added to one branch only is silently ignored
    #     for whoever set PROGRESS=off
    invocations = [
        line.strip().split(" 2>&1 |")[0]
        for line in body.splitlines()
        if "launch_scripts/train_molmobot.py" in line
    ]
    assert len(invocations) == 2
    assert invocations[0] == invocations[1]

    # 4b. the mixture gets its own save folder and its own normalisation stats,
    #     so a pick-only run and a pick+pnp run cannot overwrite each other
    assert body.count("checkpoints/stretch4_pick_pnp") >= 1
    assert '--stats_path "$SAVE_FOLDER"/synthmanip_norm_stats.yaml' in body

    # 5. the base checkpoint is downloaded, because the trainer resolves no names
    assert "hf download allenai/MolmoBot-DROID" in body
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0

    # 6. a single task keeps the plain DATASET and passes no mixture flags
    single = write_launch_script(
        datasets=[datasets[0]],
        trainer="molmobot",
        config_path=config_path,
        base_checkpoint="allenai/MolmoBot-DROID",
        output_dir=tmp_path / "checkpoints",
        batch_size=16,
        steps=1000,
        action_type="joint_pos_rel",
        seq_len=2048,
        checkout=checkout,
    ).read_text()
    assert '--data_paths "$DATASET" ' in single
    assert "--val_data_paths" not in single
    assert "--dataset_sample_rates" not in single


def test_stretch_camera_system_exposes_onboard_cameras_at_640x368(tmp_path):
    """Stretch 4 camera system exposes the onboard cameras at 640x368 resolution."""
    from examples.machine_learning.molmospaces.stretch.config import (
        CHASE_CAMERA,
        HEAD_CAMERA,
        HEAD_CAMERA_LEFT,
        HEAD_CAMERA_RIGHT,
        WRIST_CAMERA_LEFT,
        WRIST_CAMERA_RIGHT,
        WRIST_CAMERA_STEREO,
        Stretch4CameraSystem,
    )
    from examples.machine_learning.molmospaces import hdf5_layout
    from examples.machine_learning.molmospaces.finetuning import finetune, lerobot_export

    # 1. Stretch4CameraSystem includes the onboard cameras at (640, 368)
    cam_system = Stretch4CameraSystem()
    assert cam_system.img_resolution == (640, 368)
    cam_names = [c.name for c in cam_system.cameras]
    assert cam_names == [
        HEAD_CAMERA,
        WRIST_CAMERA_LEFT,
        WRIST_CAMERA_RIGHT,
        WRIST_CAMERA_STEREO,
        HEAD_CAMERA_LEFT,
        HEAD_CAMERA_RIGHT,
    ]

    # 2. Defaults for trainers and exports match
    assert finetune.DEFAULT_CAMERA_NAMES == [
        HEAD_CAMERA,
        WRIST_CAMERA_LEFT,
        WRIST_CAMERA_RIGHT,
        HEAD_CAMERA_LEFT,
        HEAD_CAMERA_RIGHT,
    ]
    assert CHASE_CAMERA not in finetune.DEFAULT_CAMERA_NAMES
    assert CHASE_CAMERA not in lerobot_export.DEFAULT_CAMERA_NAMES

    # 3. hdf5_layout._cameras_beside filters out debug/non-training camera MP4s if present
    house_dir = tmp_path / "house_1"
    house_dir.mkdir(parents=True, exist_ok=True)
    (house_dir / "episode_00000000_head_camera.mp4").write_bytes(b"dummy")
    (house_dir / "episode_00000000_chase_camera.mp4").write_bytes(b"dummy")

    detected = hdf5_layout._cameras_beside(house_dir, 0, "")
    assert "head_camera" in detected
    assert "chase_camera" not in detected



