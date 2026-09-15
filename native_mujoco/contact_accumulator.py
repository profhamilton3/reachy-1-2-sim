"""R12-8xx: E1 readiness (assignment 2026-09-14, work item 4) -- arm-link
contacts in the recorded state stream.

`SimState.snapshot_grippers()`'s `grip_force_n` is pad force only; a
forearm or thumb graze against a scene object that displaces nothing is
otherwise invisible in a recording. `simulation_core.contact_records()`
already computes every active MuJoCo contact each step (`snapshot_contacts`,
used only by the episode runner's evaluation snapshots); this module filters
that list down to arm-link/object pairs and accumulates them ACROSS the
physics steps between two state pushes (10 steps at 500 Hz / 50 Hz -- see
`native_mujoco/server.py`'s `_STATE_HZ`), so a contact shorter than one
state-push interval is never missed even though it moves nothing and the
state stream itself only samples at 50 Hz.

What `steps` measures (read this before using it for anything)
-----------------------------------------------------------------
`steps` is the number of PHYSICS STEPS, within the current accumulation
window (since the last `drain()`), during which this exact `(arm_geom,
object_id)` pair had an active MuJoCo contact. It is NOT:

  * wall-clock duration -- convert with the physics rate (500 Hz) if you
    want seconds, and remember the window itself is already bounded to at
    most 10 steps (20 ms at 500 Hz) between one drain and the next.
  * a count of distinct touch events -- a pair that breaks contact and
    re-contacts twice within the SAME window still reports one entry with
    `steps` equal to the total number of contacting steps, not "2 touches".
    `first_sim_step`/`last_sim_step` bound the window the steps fell in,
    not necessarily a single unbroken touch.
  * comparable across drains without care -- `steps` resets to 0 at every
    `drain()` call (every state push). A pair present in three consecutive
    pushes is three separate dict entries, `steps<=10` each; summing them
    (matching by `(arm_geom, object_id)` across the pushed states) is the
    caller's job if a longer-duration total is wanted.

Why there are two ways to feed it (measured, not assumed)
------------------------------------------------------------
`add_step()` takes an already-built `List[ContactRecord]` (e.g. from
`simulation_core.contact_records()`) -- simple to test offline with
synthetic records, and used that way throughout this module's own test
suite. But `contact_records()` calls `mj_id2name` (x4) and
`mj_contactForce` for EVERY active contact, arm-related or not, and this
robot model has far more self-contact than arm/object contact: measured on
a compiled `scenes/e1_boards/B2_foam_r3c3.yaml`, `data.ncon` at the home
keyframe (nothing touching any tracked object) was 125, and
`contact_records()` + `add_step()` together cost ~800 us/call -- ~40% of
the 2000 us (500 Hz) physics-step budget, EVERY step, regardless of
whether the arm is anywhere near an object. `add_step_from_data()` is the
production path `native_mujoco/server.py` actually calls: it filters
`data.contact` by geom/body ID BEFORE paying for any name lookup or force
computation, so a step with no arm/object contact (the common case) costs
only `data.ncon` integer comparisons. See
`docs/reviews/probes-2026-09-14-sim-e1-readiness/README.md` for the full
benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Sequence, Tuple

import mujoco
import numpy as np

from evaluation_snapshot import ContactRecord

#: Collision-pad geoms whose contact against a tracked object is worth
#: recording. Excludes the table, rails, and torso deliberately -- those
#: are self-collision / E2 questions, out of scope here (assignment
#: "out of scope" list).
ARM_CONTACT_GEOMS: Tuple[str, ...] = (
    "r_upper_arm_col", "r_forearm_col", "r_thumb_col", "r_finger_col",
)


@dataclass
class _PairAccumulator:
    steps: int = 0
    max_normal_force_n: float = -1.0  # sentinel below any real force (>=0)
    min_dist_m: float = float("inf")
    first_sim_step: int = -1
    last_sim_step: int = -1
    pos_at_max_force: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class ContactAccumulator:
    """Accumulates arm-link/object contact pairs across physics steps
    between two state pushes. `add_step()` once per physics step (gated,
    in the server, behind `--record`); `drain()` once per state push,
    returning the window's pairs as plain dicts and clearing for the next
    window."""

    def __init__(self, arm_geoms: Iterable[str] = ARM_CONTACT_GEOMS) -> None:
        self._arm_geoms: FrozenSet[str] = frozenset(arm_geoms)
        self._pairs: Dict[Tuple[str, str], _PairAccumulator] = {}
        # (model, arm_geom_ids, geom_name_by_id, body_name_by_id) --
        # resolved once per model and reused; see add_step_from_data().
        self._id_cache: tuple = (None, frozenset(), {}, {})

    def _match(self, c: ContactRecord, object_ids: FrozenSet[str]):
        """(arm_geom, object_id) for a contact that is exactly one arm
        collision geom against exactly one tracked-object body, else
        None -- an arm-arm contact (self-collision) or an arm-fixture
        contact (table/rails/torso, not a tracked object body) is ignored
        here, both deliberately out of scope for this accumulator."""
        g1_arm = c.geom1 in self._arm_geoms
        g2_arm = c.geom2 in self._arm_geoms
        b1_obj = c.body1 in object_ids
        b2_obj = c.body2 in object_ids
        if g1_arm and b2_obj and not (g2_arm and b1_obj):
            return c.geom1, c.body2
        if g2_arm and b1_obj and not (g1_arm and b2_obj):
            return c.geom2, c.body1
        return None

    def add_step(self, records: Sequence[ContactRecord], sim_step: int,
                object_ids: Iterable[str]) -> None:
        """Fold one physics step's contacts into the current window."""
        want = frozenset(object_ids)
        for c in records:
            match = self._match(c, want)
            if match is None:
                continue
            self._fold(match, sim_step, c.dist, c.normal_force, c.pos)

    def _fold(self, match: Tuple[str, str], sim_step: int, dist: float,
             normal_force: float, pos: Tuple[float, float, float]) -> None:
        acc = self._pairs.setdefault(match, _PairAccumulator())
        acc.steps += 1
        if acc.first_sim_step == -1:
            acc.first_sim_step = sim_step
        acc.last_sim_step = sim_step
        acc.min_dist_m = min(acc.min_dist_m, dist)
        if normal_force > acc.max_normal_force_n:
            acc.max_normal_force_n = normal_force
            acc.pos_at_max_force = pos

    def _resolve_ids(self, model) -> Tuple[FrozenSet[int], Dict[int, str], Dict[int, str]]:
        cached_model, arm_geom_ids, geom_name_by_id, body_name_by_id = self._id_cache
        if cached_model is model:
            return arm_geom_ids, geom_name_by_id, body_name_by_id
        geom_name_by_id = {
            gid: (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "")
            for gid in range(model.ngeom)
        }
        body_name_by_id = {
            bid: (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or "")
            for bid in range(model.nbody)
        }
        arm_geom_ids = frozenset(
            gid for gid, name in geom_name_by_id.items() if name in self._arm_geoms)
        self._id_cache = (model, arm_geom_ids, geom_name_by_id, body_name_by_id)
        return arm_geom_ids, geom_name_by_id, body_name_by_id

    def add_step_from_data(self, model, data, sim_step: int,
                           object_ids: Iterable[str]) -> None:
        """Production path (`native_mujoco/server.py`): filters
        `data.contact` by geom/body ID BEFORE building anything -- no
        `mj_id2name` or `mj_contactForce` call for a contact that cannot
        possibly match. Equivalent to
        `add_step(contact_records(model, data), sim_step, object_ids)` for
        the SAME model/data (see the module docstring's benchmark), but
        does not pay per-contact name-lookup/force cost for every
        non-matching (self-)contact."""
        arm_geom_ids, geom_name_by_id, body_name_by_id = self._resolve_ids(model)
        want_bodies = frozenset(object_ids)
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            g1_arm = g1 in arm_geom_ids
            g2_arm = g2 in arm_geom_ids
            if not (g1_arm or g2_arm):
                continue  # cheap skip -- the overwhelming majority of contacts
            b1_name = body_name_by_id[int(model.geom_bodyid[g1])]
            b2_name = body_name_by_id[int(model.geom_bodyid[g2])]
            b1_obj = b1_name in want_bodies
            b2_obj = b2_name in want_bodies
            if g1_arm and b2_obj and not (g2_arm and b1_obj):
                match = (geom_name_by_id[g1], b2_name)
            elif g2_arm and b1_obj and not (g1_arm and b2_obj):
                match = (geom_name_by_id[g2], b1_name)
            else:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            self._fold(match, sim_step, float(c.dist), float(force[0]),
                      (float(c.pos[0]), float(c.pos[1]), float(c.pos[2])))

    def drain(self) -> List[dict]:
        """The current window's pairs as plain dicts, and clear for the
        next window (called once per state push)."""
        out = [
            {
                "arm_geom": arm_geom,
                "object_id": object_id,
                "steps": acc.steps,
                "max_normal_force_n": acc.max_normal_force_n,
                "min_dist_m": acc.min_dist_m,
                "first_sim_step": acc.first_sim_step,
                "last_sim_step": acc.last_sim_step,
                "pos_at_max_force": list(acc.pos_at_max_force),
            }
            for (arm_geom, object_id), acc in self._pairs.items()
        ]
        self._pairs = {}
        return out
