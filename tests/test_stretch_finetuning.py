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

import os
from pathlib import Path

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


def test_molmobot_setup_error_says_how_to_clone(tmp_path):
    """MolmoBot is not a dependency, so a missing checkout has to be actionable."""
    from examples.machine_learning.molmospaces.finetuning.molmobot_repo import (
        MolmoBotSetupError,
        ensure_importable,
    )

    with pytest.raises(MolmoBotSetupError) as error:
        ensure_importable(tmp_path / "no-checkout-here")
    message = str(error.value)
    assert "git clone" in message
    assert "allenai/MolmoBot" in message


def test_molmobot_import_failure_names_what_it_tried(monkeypatch):
    """The ImportError has to distinguish "not cloned" from "cloned and broken"."""
    from examples.machine_learning.molmospaces.policies import molmobot_policy

    def no_checkout(*args, **kwargs):
        raise molmobot_policy.MolmoBotSetupError("no checkout at third_party/MolmoBot")

    monkeypatch.setattr(molmobot_policy, "ensure_importable", no_checkout)
    monkeypatch.setattr(molmobot_policy, "MOLMOBOT_MODULES", ("stretch4_not_molmobot",))

    with pytest.raises(ImportError) as error:
        molmobot_policy.StretchMolmoBotPolicy._import_molmobot()
    message = str(error.value)
    assert "no checkout at third_party/MolmoBot" in message
    assert "stretch4_not_molmobot" in message


def test_molmobot_import_puts_the_checkout_on_the_path(monkeypatch, tmp_path):
    """`--policy molmobot` must work without anyone exporting PYTHONPATH."""
    import os
    import sys

    from examples.machine_learning.molmospaces.finetuning import molmobot_repo

    package_dir = tmp_path / "third_party" / "MolmoBot" / "MolmoBot"
    (package_dir / "launch_scripts").mkdir(parents=True)
    (package_dir / molmobot_repo.TRAINER_SCRIPT).write_text("")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("PYTHONPATH", "")

    returned = molmobot_repo.ensure_importable(tmp_path / "third_party" / "MolmoBot", patch=False)

    assert returned == package_dir.resolve()
    # sys.path for this interpreter and the forkserver children that inherit it;
    # PYTHONPATH for spawn, which starts a fresh one that reads only the
    # environment.
    assert str(package_dir.resolve()) in sys.path
    assert str(package_dir.resolve()) in os.environ["PYTHONPATH"].split(os.pathsep)


def test_molmobot_policy_factory_patch_is_scoped_and_idempotent(tmp_path):
    """Every BasePolicyConfig subclass in MolmoBot's eval module needs the default.

    The module builds several of its own eval configs at import time, so one
    unpatched class fails the import for all of them -- which surfaces as
    "MolmoBot is not importable" when it is.
    """
    from examples.machine_learning.molmospaces.finetuning.molmobot_repo import (
        EVAL_MODULE,
        MolmoBotCheckout,
        ensure_policy_factory_default,
    )

    package_dir = tmp_path / "MolmoBot"
    module = package_dir / EVAL_MODULE
    module.parent.mkdir(parents=True)
    module.write_text(
        'class SynthVLAPolicyConfig(BasePolicyConfig):\n'
        '    """Doc."""\n'
        '    policy_type: str = "learned"\n'
        "\n"
        "\n"
        "class SynthVLARBY1PolicyConfig(BasePolicyConfig):\n"
        '    policy_type: str = "learned"\n'
        "\n"
        "\n"
        "class NotAPolicyConfig(JsonBenchmarkEvalConfig):\n"
        "    policy_dt_ms: float = 66.0\n"
    )
    checkout = MolmoBotCheckout(
        root=tmp_path, package_dir=package_dir, data_scripts_dir=tmp_path / "data_scripts"
    )

    assert ensure_policy_factory_default(checkout) is True
    patched = module.read_text()
    assert patched.count("policy_factory: object = None") == 2
    # The docstring stays a docstring: the field goes after it, not before.
    assert patched.index('"""Doc."""') < patched.index("policy_factory: object = None")
    assert "NotAPolicyConfig(JsonBenchmarkEvalConfig):\n    policy_dt_ms" in patched

    assert ensure_policy_factory_default(checkout) is False
    assert module.read_text() == patched


# =============================================================================
# What a checkpoint says about how it was trained
# =============================================================================


def _write_checkpoint_config(directory, args, action_dim=10, states_mode="cross_attn"):
    """A `config.yaml` shaped like the one `train_molmobot.py` saves."""
    import yaml

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {"action_dim": action_dim, "states_mode": states_mode},
                "runtime_data": {"args": args, "hostname": "test"},
            }
        )
    )
    return directory


def test_reads_the_cameras_and_action_type_a_checkpoint_was_trained_with(tmp_path):
    from examples.machine_learning.molmospaces.policies.molmobot_checkpoint import (
        read_training_args,
    )

    checkpoint = _write_checkpoint_config(
        tmp_path / "step9500_bestfit",
        "launch_scripts/train_molmobot.py /base --data_paths /data --seq_len 528 "
        "--action_dim 10 --action_preset stretch_jointdelta --camera_names "
        "head_camera_right wrist_camera_right --action_type joint_pos_rel "
        "--exp_name=stretch4_pick --ft_vit=True",
    )

    training = read_training_args(checkpoint)

    # The list flag stops at the next `--`, and the scalar one before it does not
    # swallow its neighbour.
    assert training.camera_names == ["head_camera_right", "wrist_camera_right"]
    assert training.action_type == "joint_pos_rel"
    assert training.action_preset == "stretch_jointdelta"
    assert training.action_dim == 10
    assert training.states_mode == "cross_attn"
    assert bool(training) is True


def test_a_checkpoint_that_records_nothing_is_not_an_error(tmp_path):
    """Every field is optional: "did not say" is not "said no"."""
    from examples.machine_learning.molmospaces.policies.molmobot_checkpoint import (
        read_training_args,
    )

    (tmp_path / "empty").mkdir()
    assert bool(read_training_args(tmp_path / "empty")) is False
    assert bool(read_training_args(tmp_path / "does-not-exist")) is False

    (tmp_path / "unparseable").mkdir()
    (tmp_path / "unparseable" / "config.yaml").write_text("model: [unclosed\n")
    assert bool(read_training_args(tmp_path / "unparseable")) is False


def _adapter(**policy_config_kwargs):
    """A `StretchMolmoBotPolicy` with only the fields the reconciliation reads.

    Constructed without `__init__`, which would build MolmoBot's policy and load
    a checkpoint. What is under test is the part that runs before that.
    """
    from types import SimpleNamespace

    from examples.machine_learning.molmospaces.policies.molmobot_policy import (
        StretchMolmoBotPolicy,
        StretchMolmoBotPolicyConfig,
    )

    policy = object.__new__(StretchMolmoBotPolicy)
    policy.config = SimpleNamespace(
        policy_config=StretchMolmoBotPolicyConfig(**policy_config_kwargs)
    )
    return policy


def test_the_checkpoints_cameras_win_over_the_configured_ones():
    """Serving a VLA the wrong cameras raises nothing; it just stops working."""
    from examples.machine_learning.molmospaces.policies.molmobot_checkpoint import TrainingArgs

    trained_on = ["head_camera_right", "wrist_camera_right"]
    policy = _adapter(camera_names=[HEAD_CAMERA, WRIST_CAMERA_LEFT])

    assert policy._resolve_camera_names(TrainingArgs(camera_names=trained_on)) == trained_on
    # Including the order, which is part of the model's input layout.
    assert policy._resolve_camera_names(TrainingArgs(camera_names=trained_on[::-1])) == (
        trained_on[::-1]
    )
    # Nothing recorded, or explicitly switched off, leaves the config in force.
    assert policy._resolve_camera_names(TrainingArgs()) == [HEAD_CAMERA, WRIST_CAMERA_LEFT]
    pinned = _adapter(camera_names=[HEAD_CAMERA], configure_from_checkpoint=False)
    assert pinned._resolve_camera_names(TrainingArgs(camera_names=trained_on)) == [HEAD_CAMERA]


def test_an_explicit_action_type_outranks_the_checkpoint():
    from examples.machine_learning.molmospaces.policies.molmobot_checkpoint import TrainingArgs

    trained = TrainingArgs(action_type="joint_pos_rel")
    assert _adapter(action_type="joint_pos")._resolve_action_type(trained) == "joint_pos_rel"
    named = _adapter(action_type="joint_pos", action_type_explicit=True)
    assert named._resolve_action_type(trained) == "joint_pos"


def test_a_checkpoint_of_the_wrong_action_width_is_refused():
    """MolmoBot-DROID emits eight numbers; unpacking them into ten shifts every joint."""
    from examples.machine_learning.molmospaces.policies.molmobot_checkpoint import TrainingArgs

    policy = _adapter()
    with pytest.raises(ValueError, match="misassign"):
        policy._check_action_width(TrainingArgs(action_dim=8))
    policy._check_action_width(TrainingArgs(action_dim=10))
    policy._check_action_width(TrainingArgs())


def test_relative_actions_become_the_absolute_targets_stretch_is_commanded_with():
    """`qpos + delta` inverts how `actions/joint_pos_rel` was recorded."""
    policy = _adapter()
    policy._action_type = "joint_pos_rel"
    reference = {
        "base": np.array([1.0, 2.0, 0.5]),
        "lift": np.array([0.6]),
        "arm": np.array([0.2]),
        "wrist": np.array([3.14, -0.4, 0.0]),
        "gripper": np.array([0.1, 0.1]),
    }
    action = {
        "base": np.array([0.01, 0.0, -0.02]),
        "lift": np.array([0.05]),
        "arm": np.array([-0.01]),
        "wrist": np.array([0.0, 0.1, 0.0]),
        "gripper": np.array([-0.3, -0.3]),
        "done": False,
    }

    absolute = policy._to_absolute_targets(action, reference)

    assert np.allclose(absolute["base"], [1.01, 2.0, 0.48])
    assert np.allclose(absolute["lift"], [0.65])
    assert np.allclose(absolute["arm"], [0.19])
    # The gripper squeeze: a delta past where the fingers can go is what holds a
    # grasp, and it only survives as an absolute target.
    assert np.allclose(absolute["gripper"], [-0.2, -0.2])
    assert absolute["done"] is False


def test_a_relative_action_is_measured_from_the_previous_step(monkeypatch):
    """`joint_pos_rel[t]` is `commanded[t] - qpos[t-1]`, and the off-by-one is not free.

    `LastCommandedRelativeJointPosSensor` keeps the qpos from its previous call
    and differences the newly commanded position against *that*. Over a real
    pick episode, inverting it against this step's qpos instead misses the
    recorded command by up to 0.23 rad of lift.
    """
    import types

    policy = _adapter()
    policy._action_type = "joint_pos_rel"
    policy._previous_qpos = None
    policy._clipper = types.SimpleNamespace(clip_action=lambda action: action)

    states = [
        {"lift": np.array([0.50]), "arm": np.array([0.10])},
        {"lift": np.array([0.62]), "arm": np.array([0.13])},
        {"lift": np.array([0.71]), "arm": np.array([0.15])},
    ]
    policy.task = types.SimpleNamespace(
        env=types.SimpleNamespace(
            current_robot=types.SimpleNamespace(
                robot_view=types.SimpleNamespace(get_qpos_dict=lambda: states[0])
            )
        )
    )
    deltas = [
        {"lift": np.array([0.15]), "arm": np.array([0.04])},
        {"lift": np.array([0.10]), "arm": np.array([0.03])},
        {"lift": np.array([0.05]), "arm": np.array([0.01])},
    ]
    policy._inner = types.SimpleNamespace(
        camera_names=[], get_action=lambda observation: deltas.pop(0)
    )

    targets = []
    for state in states:
        policy.task.env.current_robot.robot_view.get_qpos_dict = lambda state=state: state
        targets.append(policy.get_action({}))

    # First step: no previous observation, so this one stands in -- which is also
    # the step whose recorded action is zeros.
    assert np.allclose(targets[0]["lift"], [0.65])
    # Second and third: the *previous* state, not the one just measured.
    assert np.allclose(targets[1]["lift"], [0.60])  # 0.50 + 0.10, not 0.62 + 0.10
    assert np.allclose(targets[2]["arm"], [0.14])  # 0.13 + 0.01, not 0.15 + 0.01


def test_absolute_checkpoints_are_passed_through_untouched():
    policy = _adapter(action_type="joint_pos")
    policy._action_type = "joint_pos"
    action = {"lift": np.array([0.6])}
    assert policy._to_absolute_targets(action, {"lift": np.array([0.2])}) is action


def test_molmobot_eval_config_defaults_to_the_trained_action_type():
    from examples.machine_learning.molmospaces.configs import StretchMolmoBotEvalConfig
    from examples.machine_learning.molmospaces.finetuning.finetune import MOLMOBOT_ACTION_TYPES

    config = StretchMolmoBotEvalConfig()
    assert config.policy_config.action_type == MOLMOBOT_ACTION_TYPES[0] == "joint_pos_rel"
    assert config.tag == "stretch4_molmobot"


def test_a_run_of_symlinked_houses_is_still_a_run(tmp_path):
    """Picking a few houses out of a big run is how the overfit check is built.

    `rglob`'s `**` does not descend into symlinks, so the obvious way to make a
    tiny dataset -- symlink four houses into a directory of their own -- used to
    look like an empty run, and `arrange_train_val_split` symlinks houses too.
    """
    from examples.machine_learning.molmospaces.hdf5_layout import trajectory_files

    source = tmp_path / "pick"
    for name in ("house_0", "house_1"):
        (source / name).mkdir(parents=True)
        (source / name / "trajectories_batch_1_of_1.h5").write_bytes(b"")

    assert len(trajectory_files(source)) == 2

    tiny = tmp_path / "tiny"
    tiny.mkdir()
    for name in ("house_0", "house_1"):
        (tiny / name).symlink_to(source / name, target_is_directory=True)

    assert len(trajectory_files(tiny)) == 2
    assert [path.parent.name for path in trajectory_files(tiny)] == ["house_0", "house_1"]


# =============================================================================
# Metrics: recording them, reporting on them, sweeping over them
# =============================================================================


def _patched_trainer_module(tmp_path):
    """A synthetic MolmoBot trainer, patched, with its module namespace exec'd.

    Returns `(namespace, patched_source, checkout)`. The stub carries the
    attributes the recorder reads off a real `Trainer` -- and the two save
    methods, since the patch wraps them.
    """
    from examples.machine_learning.molmospaces.finetuning.molmobot_repo import (
        TRAINER_MODULE,
        MolmoBotCheckout,
        ensure_metrics_log,
    )

    package_dir = tmp_path / "MolmoBot"
    trainer = package_dir / TRAINER_MODULE
    trainer.parent.mkdir(parents=True)
    trainer.write_text(
        "class Trainer:\n"
        "    def log_metrics_to_console(self, prefix: str, metrics):\n"
        "        print(prefix, metrics)\n"
        "\n"
        "    def save_checkpoint(self, checkpoint_type, optim=True):\n"
        "        return (self.destination, None)\n"
        "\n"
        "    def save_bestfit_checkpoint(self):\n"
        "        return self.destination\n"
    )
    checkout = MolmoBotCheckout(
        root=tmp_path, package_dir=package_dir, data_scripts_dir=tmp_path / "data_scripts"
    )
    assert ensure_metrics_log(checkout) is True

    namespace: dict = {}
    exec(trainer.read_text(), namespace)  # noqa: S102
    return namespace, trainer, checkout


def test_metrics_patch_is_inserted_once_and_records_what_the_console_drops(tmp_path):
    """The trainer's metrics dict is captured before `log_metrics_to_console` filters it.

    That filter is the whole reason for the patch: it drops every `optim/` key
    but `optim/total_grad_norm`, which this trainer never emits, so the learning
    rates and gradient norms exist only in W&B -- which is off by default.
    """
    import json

    from examples.machine_learning.molmospaces.finetuning.molmobot_repo import (
        METRICS_ENV_VAR,
        ensure_metrics_log,
    )

    namespace, trainer_file, checkout = _patched_trainer_module(tmp_path)
    patched = trainer_file.read_text()
    assert ensure_metrics_log(checkout) is False
    assert trainer_file.read_text() == patched

    # The call is the first statement of the method, and the recorder is at
    # module level -- name resolution happens at call time, so its position
    # after the class is fine and cannot disturb a line of the trainer.
    assert (
        "    def log_metrics_to_console(self, prefix: str, metrics):\n"
        "        _stretch4_record_metrics(self, prefix, metrics)\n" in patched
    )

    destination = tmp_path / "metrics.jsonl"
    os.environ[METRICS_ENV_VAR] = str(destination)
    try:
        fake = namespace["Trainer"]()
        fake.global_step, fake.max_steps = 120, 10000
        namespace["_stretch4_record_metrics"](
            fake,
            "[step=120/10000, eta=1 hour]",
            {"train/action_flow_loss": 0.042, "optim/action_expert_lr": 1e-4, "label": "x"},
        )
        namespace["_stretch4_record_metrics"](fake, "val", {"action_flow_loss": 0.051})
    finally:
        del os.environ[METRICS_ENV_VAR]

    records = [json.loads(line) for line in destination.read_text().splitlines()]
    assert [record["split"] for record in records] == ["train", "eval"]
    assert records[0]["step"] == 120
    assert records[0]["metrics"]["optim/action_expert_lr"] == 1e-4
    # Non-numeric values are dropped rather than breaking the line.
    assert "label" not in records[0]["metrics"]
    assert records[1]["label"] == "val"


def test_a_saved_checkpoint_carries_the_state_training_was_in(tmp_path):
    """`step<N>_bestfit/` on its own says which step it came from and nothing else.

    The summary is what makes an old checkpoint interpretable: whether the run
    had converged there or was still improving, and at what loss.
    """
    import types

    from examples.machine_learning.molmospaces.finetuning.molmobot_repo import (
        CHECKPOINT_METRICS_FILENAME,
    )
    from examples.machine_learning.molmospaces.policies.molmobot_checkpoint import (
        read_training_state,
    )

    namespace, _, _ = _patched_trainer_module(tmp_path)
    checkpoint = tmp_path / "step9500_bestfit"
    checkpoint.mkdir()

    trainer = namespace["Trainer"]()
    trainer.destination = str(checkpoint)
    trainer.global_step, trainer.max_steps = 9500, 10000
    trainer.cur_train_loss, trainer.min_train_loss = 0.0243, 0.0241
    trainer.best_eval_loss, trainer.best_eval_step = 0.0591, 9500
    trainer.evals_since_improvement = 0
    trainer.optim = types.SimpleNamespace(
        param_groups=[{"group_name": "action_expert", "lr": 2.3e-6}]
    )
    namespace["_stretch4_record_metrics"](
        trainer, "[step=9500/10000]", {"train/action_flow_loss": 0.0243}
    )
    namespace["_stretch4_record_metrics"](trainer, "synthmanip_val", {"action_flow_loss": 0.0591})

    # Saving is what writes it -- both kinds of save, since both are wrapped.
    trainer.save_bestfit_checkpoint()
    assert (checkpoint / CHECKPOINT_METRICS_FILENAME).is_file()

    state = read_training_state(checkpoint)
    assert state.step == 9500
    assert state.kind == "bestfit"
    assert state.train_loss == 0.0243
    assert state.best_loss == 0.0591
    assert state.eval_losses == {"synthmanip_val": 0.0591}
    assert state.learning_rates == {"action_expert": 2.3e-6}
    # The best loss is this very step's, so the run had not stopped improving.
    assert state.converged is False
    assert "still improving when saved" in state.summary()

    periodic = tmp_path / "step10000"
    periodic.mkdir()
    trainer.destination = str(periodic)
    trainer.global_step = 10000
    trainer.evals_since_improvement = 2
    trainer.save_checkpoint("sharded")
    later = read_training_state(periodic)
    assert later.kind == "periodic"
    assert later.converged is True
    assert later.best_step == 9500  # the good weights are in the other directory


def test_a_checkpoint_without_a_summary_is_not_an_error(tmp_path):
    """Checkpoints saved before the patch existed still evaluate; they just say less."""
    from examples.machine_learning.molmospaces.policies.molmobot_checkpoint import (
        read_training_state,
    )

    (tmp_path / "step100").mkdir()
    assert not read_training_state(tmp_path / "step100")
    assert not read_training_state(tmp_path / "does-not-exist")

    (tmp_path / "step100" / "training_metrics.json").write_text("{not json")
    assert not read_training_state(tmp_path / "step100")


def test_the_metrics_patch_upgrades_an_older_copy_of_itself(tmp_path):
    """The block is generated, so a stale one in a clone is a bug that hides."""
    from examples.machine_learning.molmospaces.finetuning.molmobot_repo import (
        GENERATED_MARKER,
        TRAINER_MODULE,
        ensure_metrics_log,
    )

    _, trainer_file, checkout = _patched_trainer_module(tmp_path)
    current = trainer_file.read_text()

    # An older patch: the call line in place, a different block at the end.
    stale = current[: current.index(GENERATED_MARKER)] + GENERATED_MARKER + (
        "\ndef _stretch4_record_metrics(trainer, prefix, metrics):\n    pass\n"
    )
    trainer_file.write_text(stale)

    assert ensure_metrics_log(checkout) is True
    assert trainer_file.read_text() == current
    assert ensure_metrics_log(checkout) is False
    # MolmoBot's own code is untouched by the round trip.
    assert "    def save_bestfit_checkpoint(self):" in trainer_file.read_text()
    assert (checkout.package_dir / TRAINER_MODULE).read_text().count(GENERATED_MARKER) == 1


def _write_metrics(path, points, val_points=(), max_steps=1000):
    """A metrics.jsonl in the shape the patched trainer writes."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for step, loss, grad_norm, lr in points:
        lines.append(
            json.dumps(
                {
                    "time": f"2026-08-28T11:{step % 60:02d}:00",
                    "step": step,
                    "max_steps": max_steps,
                    "split": "train",
                    "label": None,
                    "metrics": {
                        "train/action_flow_loss": loss,
                        "optim/action_expert_grad_norm": grad_norm,
                        "optim/action_expert_lr": lr,
                    },
                }
            )
        )
    for step, loss in val_points:
        lines.append(
            json.dumps(
                {
                    "time": f"2026-08-28T11:{step % 60:02d}:30",
                    "step": step,
                    "max_steps": max_steps,
                    "split": "eval",
                    "label": "val",
                    "metrics": {"action_flow_loss": loss},
                }
            )
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def test_training_report_names_the_best_checkpoint_and_the_overfit(tmp_path):
    """Validation turning back up is the finding the whole report exists for."""
    from examples.machine_learning.molmospaces.finetuning.training_report import (
        diagnose,
        format_run,
        read_run,
    )

    run = tmp_path / "stretch4_pick"
    _write_metrics(
        run / "metrics.jsonl",
        [(step, 0.4 - step * 0.00035, 1.0, 1e-4) for step in range(20, 1001, 20)],
        val_points=[(100, 0.20), (200, 0.14), (300, 0.11), (400, 0.13), (500, 0.15), (600, 0.18)],
    )

    loaded = read_run(run)
    assert loaded.steps == 1000
    assert "val" in loaded.evals

    text = format_run(loaded)
    assert "best 0.11000 @ step 300" in text

    findings = [observation for observation, _ in diagnose(loaded)]
    assert any("bottomed out at 0.11000 (step 300)" in finding for finding in findings)


def test_the_report_lists_the_checkpoints_and_points_at_the_best(tmp_path):
    """A run's checkpoints are the thing you act on; the report says which one."""
    import json

    from examples.machine_learning.molmospaces.finetuning import training_report
    from examples.machine_learning.molmospaces.finetuning.molmobot_repo import (
        CHECKPOINT_METRICS_FILENAME,
    )
    from examples.machine_learning.molmospaces.policies import molmobot_checkpoint

    # The three copies of this filename have to agree: the trainer patch writes
    # it, the evaluation reads it, and the report -- which runs under MolmoBot's
    # interpreter, where this repository cannot be imported -- spells it out.
    assert (
        CHECKPOINT_METRICS_FILENAME
        == training_report.CHECKPOINT_METRICS_FILENAME
        == molmobot_checkpoint.TRAINING_METRICS_FILENAME
    )

    run = tmp_path / "stretch4_pick"
    _write_metrics(
        run / "metrics.jsonl",
        [(step, 0.4 - step * 0.0003, 1.0, 1e-4) for step in range(20, 1001, 20)],
        val_points=[(500, 0.12), (1000, 0.15)],
    )
    for name, step, since in (("step500_bestfit", 500, 0), ("step1000", 1000, 1)):
        directory = run / name
        directory.mkdir()
        (directory / CHECKPOINT_METRICS_FILENAME).write_text(
            json.dumps(
                {
                    "step": step,
                    "max_steps": 1000,
                    "kind": "bestfit" if step == 500 else "periodic",
                    "train": {"loss": 0.1 if step == 500 else 0.05},
                    "best": {
                        "metric": "action_flow_loss",
                        "loss": 0.12,
                        "step": 500,
                        "evals_since_improvement": since,
                    },
                }
            )
        )

    text = training_report.format_run(training_report.read_run(run))

    assert "step500_bestfit" in text and "step1000" in text
    assert "(saved on an improvement)" in text
    assert "(1 evals without improvement by then)" in text
    # The newest checkpoint is not the one to evaluate, and the report says so.
    assert f"-> evaluate {run / 'step500_bestfit'}" in text


def test_training_report_says_when_nothing_was_validated(tmp_path):
    """A run with no val loss cannot be tuned, and should say so rather than look fine."""
    from examples.machine_learning.molmospaces.finetuning.training_report import (
        diagnose,
        read_run,
    )

    run = tmp_path / "no_val"
    _write_metrics(
        run / "metrics.jsonl", [(step, 0.4 - step * 0.0003, 1.0, 1e-4) for step in range(20, 401, 20)]
    )

    findings = diagnose(read_run(run))
    assert any("No validation loss was recorded." == observation for observation, _ in findings)


def test_training_report_reads_a_run_that_is_still_going(tmp_path):
    """The file is appended to while training; a half-written last line is normal."""
    from examples.machine_learning.molmospaces.finetuning.training_report import read_run

    run = tmp_path / "partial"
    path = _write_metrics(
        run / "metrics.jsonl", [(step, 0.4, 1.0, 1e-4) for step in range(20, 201, 20)]
    )
    path.write_text(path.read_text() + '{"step": 220, "split": "train", "metr')

    assert read_run(run).steps == 200


def test_training_report_orders_a_sweep_by_validation_loss(tmp_path):
    """What a probe is for: which value won, and by how much."""
    from examples.machine_learning.molmospaces.finetuning.training_report import (
        format_comparison,
        read_run,
    )

    for name, best in (("lr_3e-5", 0.20), ("lr_1e-4", 0.09), ("lr_3e-4", 0.31)):
        _write_metrics(
            tmp_path / name / "metrics.jsonl",
            [(step, 0.4, 1.0, 1e-4) for step in range(20, 201, 20)],
            val_points=[(100, best + 0.05), (200, best)],
        )
    runs = [read_run(tmp_path / name) for name in ("lr_3e-5", "lr_1e-4", "lr_3e-4")]

    table = format_comparison(runs)
    rows = [line.split()[0] for line in table.splitlines()[2:5]]
    assert rows == ["lr_1e-4", "lr_3e-5", "lr_3e-4"]


def test_the_plot_names_the_move_groups_the_action_dimensions_belong_to(tmp_path):
    """`flow_loss_dim_7` means nothing; "wrist" means something."""
    from examples.machine_learning.molmospaces.finetuning import training_report
    from examples.machine_learning.molmospaces.policies.molmobot_policy import (
        STRETCH_ACTION_SPEC,
    )

    # The report runs under MolmoBot's interpreter, where this repository is not
    # importable, so it carries its own copy of the layout. It has to be the
    # same layout, or the panel labels a curve with the wrong joints.
    assert dict(training_report.MOVE_GROUP_DIMENSIONS) == STRETCH_ACTION_SPEC
    assert list(dict(training_report.MOVE_GROUP_DIMENSIONS)) == list(STRETCH_ACTION_SPEC)

    assert training_report.move_group_dimensions() == {
        "base": [0, 1, 2],
        "lift": [3],
        "arm": [4],
        "wrist": [5, 6, 7],
        "gripper": [8, 9],
    }


def test_smoothing_follows_the_series_without_shortening_it():
    """The raw flow loss is noisy per step; the eye needs a line, not a cloud."""
    from examples.machine_learning.molmospaces.finetuning.training_report import smoothed

    points = [(step, 1.0 if step % 2 else 0.0) for step in range(100)]
    curve = smoothed(points, span=10)

    assert [step for step, _ in curve] == [step for step, _ in points]
    # The average of an alternating series settles near its mean, and the noise
    # is gone: no smoothed value sits at either extreme by the end.
    assert 0.3 < curve[-1][1] < 0.7
    assert all(0.0 < value < 1.0 for _, value in curve[20:])


def test_the_plot_is_written_whole_and_beside_the_metrics(tmp_path):
    """An image viewer left open on the file must never catch a half-written PNG."""
    pytest.importorskip("matplotlib")

    from examples.machine_learning.molmospaces.finetuning import training_report

    run = tmp_path / "stretch4_pick"
    _write_metrics(
        run / "metrics.jsonl",
        [(step, 0.4 - step * 0.0003, 1.0, 1e-4) for step in range(20, 401, 20)],
        val_points=[(100, 0.2), (200, 0.15), (300, 0.14), (400, 0.16)],
    )

    # A bare --plot puts it beside the metrics, which is where a run in progress
    # wants it: in the save folder, at a path that does not change.
    assert training_report.main([str(run), "--plot"]) == 0
    written = run / "training.png"
    assert written.is_file()
    assert written.stat().st_size > 10_000
    # The temporary it was renamed from is not left behind.
    assert not list(run.glob(".training.png*"))


def test_watching_redraws_when_the_metrics_grow(tmp_path, monkeypatch):
    """What runs beside a fine-tune: redraw on change, and say what it drew."""
    pytest.importorskip("matplotlib")

    from examples.machine_learning.molmospaces.finetuning import training_report

    run = tmp_path / "stretch4_pick"
    _write_metrics(
        run / "metrics.jsonl",
        [(step, 0.4 - step * 0.0003, 1.0, 1e-4) for step in range(20, 201, 20)],
        val_points=[(100, 0.2), (200, 0.15)],
    )
    plot = tmp_path / "training.png"

    # One pass, then out: the loop is otherwise deliberately endless.
    def stop(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(training_report.time, "sleep", stop)
    assert training_report.watch_plot([run], plot, interval=0.01) == 0
    assert plot.is_file()

    first = plot.stat().st_mtime_ns
    _write_metrics(
        run / "metrics.jsonl",
        [(step, 0.4 - step * 0.0003, 1.0, 1e-4) for step in range(20, 401, 20)],
        val_points=[(100, 0.2), (200, 0.15), (300, 0.13), (400, 0.12)],
    )
    assert training_report.watch_plot([run], plot, interval=0.01) == 0
    assert plot.stat().st_mtime_ns != first


def _fake_run_script(path):
    """A run_molmobot.sh with the knobs a probe drives, and nothing else."""
    path.write_text(
        "#!/usr/bin/env bash\n"
        'SAVE_FOLDER="${SAVE_FOLDER:-/tmp/checkpoints}"\n'
        'METRICS="${METRICS:-$SAVE_FOLDER/metrics.jsonl}"\n'
        'STATS_PATH="${STATS_PATH:-$SAVE_FOLDER/stats.yaml}"\n'
        'PREPARE="${PREPARE:-on}"\n'
        'MAX_STEPS="${MAX_STEPS:-10000}"\n'
        'EVAL_INTERVAL="${EVAL_INTERVAL:-500}"\n'
        'ACTION_EXPERT_LR="${ACTION_EXPERT_LR:-1e-4}"\n'
    )
    return path


def test_hparam_probe_writes_one_run_per_value_over_the_real_script(tmp_path):
    """A probe is the generated script with one variable changed -- not a copy of it."""
    import subprocess

    from examples.machine_learning.molmospaces.finetuning.hparam_probe import probe_script

    script = _fake_run_script(tmp_path / "run_molmobot.sh")
    body = probe_script(
        script=script,
        variable="ACTION_EXPERT_LR",
        values=["3e-5", "1e-4"],
        steps=600,
        eval_interval=100,
        output_dir=tmp_path / "probe",
        report=Path("training_report.py"),
    )
    probe = tmp_path / "probe" / "run_probe.sh"
    probe.parent.mkdir()
    probe.write_text(body)
    assert subprocess.run(["bash", "-n", str(probe)], capture_output=True).returncode == 0

    # Each value gets its own save folder and metrics file, or the comparison
    # would be over one run overwritten twice.
    assert body.count("ACTION_EXPERT_LR=3e-5") >= 1
    assert "action_expert_lr_3e-5/metrics.jsonl" in body
    assert "action_expert_lr_1e-4/metrics.jsonl" in body
    # The data preparation happens once, not once per value.
    assert body.count("PREPARE=on") == 1
    assert body.count("PREPARE=off") == 1
    assert body.count('STATS_PATH="$SHARED_STATS"') == 2
    # The preflight questions are about running fewer steps than a real
    # fine-tune, which is exactly what a probe does on purpose.
    assert body.count("ASSUME_YES=1") == 2
    assert "--compare" in body


def test_hparam_probe_refuses_a_script_it_cannot_drive(tmp_path):
    """Setting an environment variable a script does not read is a silent no-op."""
    from click.testing import CliRunner

    from examples.machine_learning.molmospaces.finetuning.hparam_probe import main

    script = _fake_run_script(tmp_path / "run_molmobot.sh")
    runner = CliRunner()

    unknown = runner.invoke(main, ["--script", str(script), "--vary", "NOT_A_KNOB"])
    assert unknown.exit_code != 0
    assert "NOT_A_KNOB" in unknown.output

    # A script generated before the metrics recording existed: it has the knob
    # being swept, but nothing to compare the runs with afterwards.
    (tmp_path / "old.sh").write_text(
        'SAVE_FOLDER="${SAVE_FOLDER:-/tmp}"\nACTION_EXPERT_LR="${ACTION_EXPERT_LR:-1e-4}"\n'
    )
    stale = runner.invoke(main, ["--script", str(tmp_path / "old.sh")])
    assert stale.exit_code != 0
    assert "Regenerate it" in stale.output


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
    """The distortion crops to the image circle instead of resizing the void.

    A 123 degree fisheye warped out of a pinhole render leaves ~45% of the raw
    frame with no source pixels behind it. That black is not a lens vignette,
    and `INTER_AREA` averages it into every downstream resize, so the distortion
    crops it away before handing the frame back.
    """
    from examples.machine_learning.molmospaces.stretch.config import CAMERA_RENDER_SIZE
    from stretch4_mujoco.enums.stretch_cameras import StretchCameras
    from stretch4_mujoco.utils import FISHEYE_MIN_VALID_FRACTION

    sizes = [(224, 224), (640, 400)]
    sizes += [CAMERA_RENDER_SIZE[name] for name in ("head_camera_left", "head_camera_right")]

    for camera in (StretchCameras.cam_nav_rgb_se4_left, StretchCameras.cam_nav_rgb_se4_right):
        distort = camera.post_processing_callback
        assert distort is not None
        for width, height in sizes:
            out = distort(np.full((height, width, 3), 200, dtype=np.uint8))
            assert out.shape == (height, width, 3), "the crop must not change the frame size"
            non_black_ratio = np.count_nonzero(np.any(out > 0, axis=-1)) / float(width * height)
            assert non_black_ratio >= FISHEYE_MIN_VALID_FRACTION, (
                f"{camera.name} at {width}x{height} keeps only {non_black_ratio:.2f} of the "
                "frame as real pixels; the crop should hold it to FISHEYE_MIN_VALID_FRACTION"
            )
            # Some black is expected and wanted: the curved corners are what a
            # real lens vignettes. A frame with none of it has been cropped past
            # the image circle and thrown away field of view for nothing.
            assert non_black_ratio < 1.0


def test_fisheye_crop_is_a_zoom_into_the_same_view_at_every_resolution():
    """The crop is the same window of the scene whatever size the frame is.

    Both stacks render these cameras at their own sizes -- the simulator at the
    sensor's 1920x1200, the datagen path at a macroblock-sized 640x400 -- and a
    crop that differed between them would mean a policy trained on one seeing a
    different field of view at the other.
    """
    from stretch4_mujoco.enums.stretch_cameras import StretchCameras

    for camera in (StretchCameras.cam_nav_rgb_se4_left, StretchCameras.cam_nav_rgb_se4_right):
        settings = camera.initial_camera_settings
        native = camera.fisheye_crop_rect(settings.width, settings.height)
        scaled = camera.fisheye_crop_rect(640, 400)
        scale = 640.0 / settings.width

        assert camera.fisheye_crop_zoom(settings.width, settings.height) > 1.0
        for native_edge, scaled_edge in zip(native, scaled):
            # Two pixels of slack: the crop is searched on each frame's own
            # integer grid, and one step of the 640-wide one is three native
            # pixels.
            assert abs(native_edge * scale - scaled_edge) <= 2.0


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
        METRICS_ENV_VAR,
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

    # 7. the norm stats stay with the run, not in the shared MolmoBot checkout --
    #    through a variable, so a sweep over one dataset can share one file
    assert '--stats_path "$STATS_PATH"' in body
    assert 'STATS_PATH="${STATS_PATH:-$SAVE_FOLDER/synthmanip_norm_stats.yaml}"' in body

    # 7a-ii. the run records its own metrics, because with W&B off nothing else
    #        keeps the validation loss, the learning rates or the gradient norms
    #        -- log_metrics_to_console filters the last two out entirely.
    assert 'METRICS="${METRICS:-$SAVE_FOLDER/metrics.jsonl}"' in body
    assert f'export {METRICS_ENV_VAR}="$METRICS"' in body
    assert '"$PYTHON" "$TRAINING_REPORT" "$METRICS"' in body

    # 7a-iii. and everything a second run over the same data would repeat is
    #         skippable, which is what makes a sweep cheap
    assert 'PREPARE="${PREPARE:-on}"' in body
    assert 'SAVE_FOLDER="${SAVE_FOLDER:-' in body

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
    # Defaults to `vision`: the ~123 degree distorted head cameras are far enough
    # from MolmoBot's rectilinear DROID pretraining that the tower has to adapt.
    assert 'TRAINABLE="${TRAINABLE:-vision}"' in body
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
    assert 'TRAINABLE="${TRAINABLE:-vision}"' in wide
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
    assert 'STATS_PATH="${STATS_PATH:-$SAVE_FOLDER/synthmanip_norm_stats.yaml}"' in body

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


def test_datagen_cameras_frame_the_scene_like_the_simulator():
    """Each MolmoSpaces camera sees exactly what `Stretch4MujocoSimulator` sees.

    The simulator renders through the MJCF camera with `cam_fovy` and the
    viewport both taken from `StretchCameras.initial_camera_settings`
    (`MujocoServerCameraManagerSync.set_camera_params`). MolmoSpaces renders a
    *free* camera instead, whose vertical FOV is `MjcfCameraConfig.fov` and whose
    horizontal FOV falls out of the render viewport's aspect ratio. So the two
    stacks agree on the view only if both numbers are carried across -- and
    getting it wrong is silent: a policy trains on a narrower or wider world than
    the one it is later deployed into, and simply does worse.
    """
    from examples.machine_learning.molmospaces.stretch.config import (
        CAMERA_OUTPUT_SIZE,
        CAMERA_RENDER_SIZE,
        STRETCH_CAMERA_FOR_CAMERA,
        Stretch4CameraSystem,
    )

    for spec in Stretch4CameraSystem().cameras:
        settings = STRETCH_CAMERA_FOR_CAMERA[spec.name].initial_camera_settings

        assert spec.fov is not None, (
            f"{spec.name} has no FOV, so MolmoSpaces falls back to the MJCF's -- and "
            "mjcf_generator.py writes no fovy attribute, which means MuJoCo's 45 degree "
            "default rather than any Stretch camera"
        )
        assert spec.fov == pytest.approx(settings.field_of_view_vertical_in_degrees)

        width, height = CAMERA_RENDER_SIZE[spec.name]
        # 0.5%: both dimensions have to be multiples of 16 for the video writers
        # not to resize the frame, which is coarse enough that most cameras have
        # no exactly-correct size. See `hdf5_layout.camera_render_size`.
        assert width / height == pytest.approx(settings.width / settings.height, rel=5e-3), (
            f"{spec.name} renders at {width}x{height}, a different aspect ratio from the "
            f"hardware's {settings.width}x{settings.height}; with fovy fixed that changes "
            "how much of the scene lands in frame horizontally"
        )
        assert width % 16 == 0 and height % 16 == 0, (
            f"{spec.name} renders at {width}x{height}; imageio's ffmpeg writer resizes "
            "anything that is not a multiple of 16, so the MP4 would stop being the frame "
            "that was rendered"
        )

        expected = (height, width) if settings.rotate_number_of_times % 2 else (width, height)
        assert CAMERA_OUTPUT_SIZE[spec.name] == expected


def test_recorded_intrinsics_describe_the_recorded_frame():
    """`intrinsic_cv` matches the frame that is actually saved, rotation included.

    MolmoSpaces builds K from the shared buffer resolution and knows nothing
    about the quarter turn the head cameras get, so unpatched it describes an
    image no consumer ever sees.
    """
    from molmo_spaces.env.sensors_cameras import CameraParameterSensor

    from examples.machine_learning.molmospaces.stretch.config import (
        CAMERA_OUTPUT_SIZE,
        CAMERA_RENDER_SIZE,
        HEAD_CAMERA_LEFT,
        STRETCH_CAMERA_FOR_CAMERA,
        WRIST_CAMERA_LEFT,
    )

    class _Camera:
        def __init__(self, fov):
            self.fov = fov

        def get_pose(self):
            return np.eye(4)

    class _Env:
        def __init__(self, name, fov):
            self.camera_manager = type("_M", (), {"registry": {name: _Camera(fov)}})()

    for name in (WRIST_CAMERA_LEFT, HEAD_CAMERA_LEFT):
        camera = STRETCH_CAMERA_FOR_CAMERA[name]
        settings = camera.initial_camera_settings
        fov = settings.field_of_view_vertical_in_degrees
        sensor = CameraParameterSensor(camera_name=name, img_resolution=(648, 422))
        params = sensor.get_observation(_Env(name, fov), task=None)

        k = np.array(params["intrinsic_cv"])
        width, height = CAMERA_OUTPUT_SIZE[name]
        render_width, render_height = CAMERA_RENDER_SIZE[name]
        # A 123 degree lens has a far shorter focal length than a 58 degree one;
        # the pre-patch code reported the same 444px for every camera.
        unrotated_height = min(CAMERA_OUTPUT_SIZE[name]) if settings.rotate_number_of_times % 2 else height
        focal = (unrotated_height / 2.0) / np.tan(np.radians(fov / 2.0))

        if camera.applies_fisheye_distortion:
            # The distortion crops into the render before handing the frame
            # back at its original size, so the saved frame is zoomed in on a
            # window of it and K has to say so.
            zoom = camera.fisheye_crop_zoom(render_width, render_height)
            assert zoom > 1.0
            assert k[0, 0] == pytest.approx(focal * zoom, rel=1e-3)
            # The crop tracks the image circle rather than the frame centre, so
            # the principal point moves off centre -- but only by a few percent
            # of the frame, not into another part of it.
            assert abs(k[0, 2] - width / 2.0) < 0.1 * width
            assert abs(k[1, 2] - height / 2.0) < 0.1 * height
            continue

        # The principal point sits at the centre of the frame that gets written,
        # which for a rotated camera is the portrait one.
        assert k[0, 2] == pytest.approx(width / 2.0, abs=1.0)
        assert k[1, 2] == pytest.approx(height / 2.0, abs=1.0)
        assert k[0, 0] == pytest.approx(focal)


def test_stretch_camera_system_exposes_onboard_cameras(tmp_path):
    """Stretch 4 camera system exposes the onboard cameras into a big enough buffer."""
    from examples.machine_learning.molmospaces.stretch.config import (
        CAMERA_RENDER_SIZE,
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

    # 1. img_resolution is the shared offscreen buffer, not an image size: every
    #    camera renders into its own sub-rectangle of it, so it has to enclose
    #    all of them.
    cam_system = Stretch4CameraSystem()
    buffer_width, buffer_height = cam_system.img_resolution
    for name, (width, height) in CAMERA_RENDER_SIZE.items():
        assert width <= buffer_width and height <= buffer_height, (
            f"{name} renders {width}x{height}, which does not fit the "
            f"{buffer_width}x{buffer_height} buffer MolmoSpaces allocates"
        )
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



