"""
Finding the camera and gripper settings that let MolmoBot-DROID drive Stretch 4.

`params_search.py` is the entry point and the module worth reading first. The
rest of this package is what it needs to run a trial:

    mini_benchmark.py   one kitchen, one robot pose, four target objects
    setups.py           the seven robot/camera setups and their eval configs
    cameras.py          how an exo view is built, warped, rectified and cropped
    franka_droid_policy.py  the same checkpoint driving a Franka, un-retargeted
    scoring.py          what a trial is scored on, and the report it writes

Nothing here is imported by the rest of `molmospaces/`: this is a study of the
retargeting, run beside it rather than inside it.
"""
