"""
Teaching a policy to speak Stretch, instead of translating for it.

`../franka_remapping/` takes a Franka-trained model as given and rewrites its
numbers. That works -- the arm retargets to a few millimetres -- but it cannot
touch the other half of the mismatch: the model has never seen a Stretch head
camera, and no amount of kinematics fixes a viewpoint. This package is the other
road.

    datagen_configs.py   Stretch versions of MolmoSpaces' data generation configs
    generate_dataset.py  run them, then export -- one command
    lerobot_export.py    recorded rollouts -> a LeRobot dataset
    finetune.py          check the dataset, write the trainer config, launch

The two roads meet at `lerobot_export.py --action-space franka`, which encodes
Stretch demonstrations in the *same* 8-dimensional Franka joint space the
pretrained model already emits, by running `../franka_remapping/` backwards. A
model fine-tuned on that keeps its action head and its pretrained weights, and
is evaluated through the same remapper -- so whatever the retarget cannot
express, the training data does not contain either.
"""
