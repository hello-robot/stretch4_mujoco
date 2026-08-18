"""
Import this repository's own modules before pytest rearranges `sys.path`.

There is an `__init__.py` at the root of this checkout, so pytest's default
`prepend` import mode treats the whole repository as a package and, while
collecting, puts the checkout's *parent* directory on `sys.path` ahead of
site-packages. Any sibling directory then shadows an installed distribution of
the same name. On a development machine with `../stretch4_urdf` beside this repo
that directory has no top-level `__init__.py`, so `import stretch4_urdf` finds it
as an empty namespace package rather than the installed
`hello-robot-stretch4-urdf`, and `from stretch4_urdf import get_urdf` fails with
"cannot import name ... (unknown location)".

Whether it bites depends on collection order -- it does not if some earlier test
module already imported the real package -- which is why running the whole
`tests/` directory passes while running one file on its own can fail. Importing
the simulator here, from a conftest loaded before any collection, resolves it and
its dependencies against the unmodified path and caches them in `sys.modules` for
the rest of the session.
"""

import stretch4_mujoco.stretch4_mujoco_simulator  # noqa: F401  cached for the session
