"""
Teaching a policy to speak Stretch, instead of translating for it.

The alternative was to take a Franka-trained model as given and rewrite its
numbers. That is what `../franka_remapping/` did, and it is gone: the arm
retargeted to a few millimetres but a third of the intermediate poses were out of
reach, and it could never touch the other half of the mismatch -- the model has
never seen a Stretch head camera, and no amount of kinematics fixes a viewpoint.
See *Why a Franka-space model is not simply remapped* in `../README.md`.

    datagen_configs.py   Stretch versions of MolmoSpaces' data generation configs
    generate_dataset.py  run them, then export -- one command
    live_recorder.py     record teleop demonstrations instead of generating them
    lerobot_export.py    recorded rollouts -> a LeRobot dataset
    finetune.py          check the dataset, write the trainer config, launch

The recorded-trajectory format is `../hdf5_layout.py`, one level up because
`../training/` reads it too: the rollouts generated here are also what the
behaviour-cloning road clones from.

Everything here exports in Stretch's own 10-dimensional move-group space. A
pretrained checkpoint contributes its vision and language weights and re-learns
its action head, which wants more data than a warm start would -- and buys, in
exchange, actions that mean exactly what they say on the robot they run on.
"""
