"""
Give an added-to-scene pickupable the mass procthor would have given it.

A THOR prefab under `objects/thor/` declares its *collider* with a sensible
`density="80"` and its *visual* mesh with no mass or density at all -- and the
visual mesh asset carries `inertia="shell"`, which makes MuJoCo derive a mass
from surface area times the default density of 1000 kg/m^3. Standalone, the
prefab therefore weighs absurdly much. Measured over the 29 potato assets that
have grasps: **16 to 40 kg**, median 23 kg, for an object 10cm across.

procthor's house generator already knows this. Every prefab it inlines into a
house XML gets `mass="1e-08"` on the visual geom:

    <geom name="...Irishpotato..._visual_0"    type="mesh" mass="1e-08" .../>
    <geom name="...Irishpotato..._collision_1" type="mesh" density="80" .../>

which is why a potato that arrives *with the scene* behaves and one added by
`added_pickup_objects` does not: both `PickTaskSampler._add_pickupables_to_scene`
and `JsonEvalTaskSampler.add_auxiliary_objects` attach the raw prefab straight
from `install_uid()`, so nothing applies the correction. This module applies it,
with the same value and for the same reason.

## Why this is one bug and not two

Fixing the mass fixes both symptoms reported of the potato task -- "it rolls
away" and "the gripper cannot lift it". Dropped onto a flat plane at the
orientation the sampler places it, and settled for the sampler's 500 timesteps:

    asset       variant             mass kg   drift   still moving after settle
    Potato_30   as-is (upstream)      19.28   51 mm   yes
    Potato_30   visual mass fixed      0.02    0 mm   no
    Potato_16   as-is (upstream)      39.86   19 mm   yes
    Potato_16   visual mass fixed      0.07    0 mm   no

A 20 kg potato has the momentum to keep rolling through the settle, and nothing
holds it afterwards. Adding rolling friction on top changed nothing measurable
once the mass was right, so there is no friction tweak here -- the fix is the one
value procthor already uses.

End to end, on identical houses (`procthor-10k`, 2 houses, 8 episodes,
`--task potato`): **0/8 solved before, 3/8 after** -- in line with the plain
`pick` baseline on those houses.

## Why it is installed in two places

The repair has to happen after the pickupable is attached to the scene spec and
before it is compiled, which is `add_auxiliary_objects`. Data generation gets
this properly, by subclassing: `StretchAddedPickupTaskSampler` overrides the
method, and the config points `task_sampler_class` at it.

Evaluation has no such seam. `run_evaluation()` hard-codes `JsonEvalRunner`,
`JsonEvalRunner.get_episode_task_sampler()` hard-codes `JsonEvalTaskSampler`,
and neither reads `task_sampler_config.task_sampler_class` -- so there is no
supported way to substitute a subclass on the eval path. Hence
`install_eval_repair()`, which wraps the method on the class.

Leaving evaluation out was not an option: the benchmark's target object *is* an
added pickupable, so without this a potato benchmark scores 0% for every policy
no matter how good it is, and -- worse -- disagrees with the training data about
what a potato weighs. Measured: a benchmark built from three episodes the expert
had just solved replayed at 0/3 before this was installed.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

VISUAL_GEOM_MASS = 1e-8
"""
What procthor writes on an inlined prefab's visual geom.

Not zero: a geom with exactly zero mass and `inertia="shell"` is a degenerate
inertia MuJoCo can reject, and matching procthor's value keeps an added instance
of an asset numerically identical to a scene-native one.
"""


def _added_object_prefix(body_name: str) -> str:
    """`pickup/0_0/Potato_26` -> `pickup/0_0/`, the namespace its geoms carry.

    Both attach paths build this the same way -- the sampler from
    `f"{added_pickup_namespace}{i}_{j}/"`, the eval sampler from
    `"/".join(name_parts[:-1]) + "/"` -- so grouping geoms by it groups them by
    the object they belong to.
    """
    head, _, _ = body_name.rpartition("/")
    return f"{head}/" if head else ""


def repair_added_pickup_masses(spec: Any, added_object_names: list[str]) -> int:
    """Zero the visual-geom mass of each named added object, in place on `spec`.

    Args:
        spec: the `MjSpec` the objects have just been attached to.
        added_object_names: body names of the added objects, i.e. the keys of
            `task_config.added_objects` / `scene_modifications.added_objects`.

    Returns:
        How many visual geoms were repaired.
    """
    prefixes = {_added_object_prefix(name) for name in added_object_names}
    prefixes.discard("")
    if not prefixes:
        return 0

    # Group by object, because whether a visual geom may be zeroed depends on
    # whether that same object has a collider to carry the mass instead.
    visual: dict[str, list[Any]] = {prefix: [] for prefix in prefixes}
    has_collider: dict[str, bool] = {prefix: False for prefix in prefixes}

    for geom in spec.geoms:
        prefix = next((p for p in prefixes if geom.name.startswith(p)), None)
        if prefix is None:
            continue
        # The prefab's `__VISUAL_MJT__` class is exactly contype=conaffinity=0;
        # its collider keeps its own density and is never touched.
        if geom.contype == 0 and geom.conaffinity == 0:
            visual[prefix].append(geom)
        else:
            has_collider[prefix] = True

    repaired = 0
    for prefix, geoms in visual.items():
        if not geoms:
            continue
        if not has_collider[prefix]:
            # An asset whose only geom is its visual mesh has nothing else to
            # carry the mass; zeroing it would leave a massless body with a free
            # joint, which MuJoCo rejects outright. Leave it heavy and say so.
            log.warning(
                f"[added-pickup] {prefix} has no collision geom, so its visual mesh is "
                "load-bearing for mass and is left alone. It may weigh far too much; see "
                "added_pickup_repair."
            )
            continue
        for geom in geoms:
            geom.mass = VISUAL_GEOM_MASS
            repaired += 1

    return repaired


_eval_repair_installed = False


def install_eval_repair() -> None:
    """Make `JsonEvalTaskSampler` apply the repair to the objects it adds.

    A wrapper on the class rather than a subclass because the eval path offers
    nowhere to inject one; see the module docstring. Idempotent, because the
    module that calls this is imported once per evaluation worker process *and*
    again when an eval config is resolved from its "module:Class" string.
    """
    global _eval_repair_installed
    if _eval_repair_installed:
        return

    from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler

    original = JsonEvalTaskSampler.add_auxiliary_objects

    def add_auxiliary_objects(self, spec) -> None:
        original(self, spec)
        added = list(self.episode_spec.scene_modifications.added_objects)
        if not added:
            return
        repaired = repair_added_pickup_masses(spec, added)
        if repaired:
            log.debug(f"[added-pickup] repaired {repaired} visual geoms on {len(added)} objects")
        else:
            # Loud: silently skipping this scores a good policy 0% and looks
            # like a policy problem rather than an asset one.
            log.warning(
                f"[added-pickup] repaired nothing on {added}. If these assets weigh tens "
                "of kilograms the benchmark is unsolvable; see added_pickup_repair."
            )

    JsonEvalTaskSampler.add_auxiliary_objects = add_auxiliary_objects
    _eval_repair_installed = True
