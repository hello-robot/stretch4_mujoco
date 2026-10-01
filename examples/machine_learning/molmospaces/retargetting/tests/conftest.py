"""
Put the checkout's root on `sys.path`, the way running these modules as scripts does.

`tests/test_retargeting.py` imports its subject the way every other module in
`examples/` does -- `from examples.machine_learning.molmospaces... import ...` --
which resolves when the checkout root is on `sys.path`, as it is under
`python -m examples....` from the repository root.

Under pytest it is not, and there is no `__init__.py` in this directory
deliberately. With one, pytest's default `prepend` import mode walks up through
`retargetting`, `molmospaces`, `machine_learning`, `examples` and the
`__init__.py` at the root of the checkout, decides this file belongs to a package
rooted at the checkout's *parent*, and puts that parent on `sys.path` instead. The
checkout directory is itself named `stretch4_mujoco`, so `import stretch4_mujoco`
then finds the repository root rather than the package inside it, and the first
thing to ask for a submodule fails with

    ModuleNotFoundError: No module named 'stretch4_mujoco.models.stretch_4.mjcf_generator'

Without an `__init__.py` pytest imports the test module as a top-level one and
puts *this* directory on `sys.path`, which leaves the checkout root to be added
here. See `tests/conftest.py` at the root of the repository for the other half of
this story -- the same package layout shadowing an installed distribution with a
sibling checkout.
"""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[5]

if str(REPOSITORY_ROOT) not in sys.path:
    # Position 0, not appended: the parent directory may already be on the path,
    # and `stretch4_mujoco` has to resolve to the package in this checkout rather
    # than to the checkout itself.
    sys.path.insert(0, str(REPOSITORY_ROOT))

import stretch4_mujoco.stretch4_mujoco_simulator  # noqa: E402,F401  cached for the session
