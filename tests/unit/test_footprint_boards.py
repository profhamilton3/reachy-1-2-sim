"""Issue #56/#74 (review 2026-09-12, Part 3, section 4): the fixture table
that motivated the "shells" hand mode, computed here with
`SceneModel.clearance` for both hand modes rather than quoted from the
review's own scratch probes -- re-run 2026-09-14 after the tube's coverage
correction (docs/adr/0003, "Correcting the tube"), and again the same day
(Slice 2) once "shells" gained the `thumb_pad` capsule and a widened
`finger` capsule.

  * TestNamedBoards -- the four named boards in the review's table, worst
    clearance over ANY guarded route (every `FOOTPRINT_LEGS` leg), for both
    hand="tube" (today's default) and hand="shells" (opt-in), plus the two
    4 cm pool objects on r2c3 the tube correction newly refuses (added in
    Slice 2). The tube column changed with the correction; the shells
    column changed again in Slice 2 once the pads were covered; the legacy
    numbers are recorded in the class docstring.

  * TestCorrectionNewlyRefuses -- every scene object on every grid cell,
    every guarded route: which boards the corrected tube refuses that the
    legacy tube allowed. Exactly two, both the 4 cm pool objects on r2c3.

  * TestSweepOrdering -- a 224-position, 6 cm cube sweep on the PLACE_ROUTE
    tail, comparing tube, shells, and the MJCF reference (geomdist.py's
    method, copied in below; ALL hand geoms including the collision pads) at
    each position, all three at the same aperture. The tube never reads more
    clearance than the MJCF geometry (its coverage, on real objects).
    Before Slice 2, "shells" did too, at 62/224 -- the thumb-pad gap pinned
    in test_arm_geometry_mjcf.py::TestShellsMissThumbPad. Since the pad
    coverage (test_arm_geometry_mjcf.py::TestShellsCoverMJCFHand), that
    count is 0/224.

Offline throughout: MuJoCo is used only to `mj_forward` a compiled model and
read back geom frames, never `mj_step`, and no server is started.
"""

import math
import os
import sys
from dataclasses import replace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../src"))
sys.path.insert(0, os.path.join(_HERE, "../../native_mujoco"))

from objects import build_scene_model_xml  # noqa: E402
from placement import ObjectPlacer  # noqa: E402
from scene_io import load_scene  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402
from reachy_ai.motion.kinematics import hand_radius, joint_path, link_capsules  # noqa: E402
from reachy_ai.scene.awareness import SceneModel  # noqa: E402

_SCENE_PATH = os.path.join(_HERE, "../../scenes/FWDCenterLabSivaPool.yaml")


# ---------------------------------------------------------------------------
# TestNamedBoards
# ---------------------------------------------------------------------------

def _board(objects):
    """A SceneModel with `objects` ({oid: (x, y)}) moved to rest on the
    table at (x, y); every other object stays at its YAML pose."""
    model = SceneModel.from_yaml(_SCENE_PATH)
    poses = {oid: model.rest_point(xy, oid) for oid, xy in objects.items()}
    model.update_poses(poses)
    return model


def _worst_over_any_guarded_route(model, oid, hand):
    """Worst clearance for `oid` over every leg of every FOOTPRINT_LEGS
    route -- "any guarded route" the way the review's table computes it,
    not just the one route a particular ability happens to fly."""
    worst = None
    for waypoints in R.FOOTPRINT_LEGS.values():
        for a, b in zip(waypoints, waypoints[1:]):
            qa = [a[j] for j in R.ARM7]
            qb = [b[j] for j in R.ARM7]
            gripper_deg = max(a["r_gripper"], b["r_gripper"], key=hand_radius)
            for q in joint_path(qa, qb, steps=13):
                caps = link_capsules(q, "right", gripper_deg, hand=hand)
                c = model.clearance(caps, ids=[oid])
                if c is not None and (worst is None or c.distance < worst):
                    worst = c.distance
    return worst


_CELL = {  # grid cell -> (x, y), read once per test run via cell_center
}


def _cell_xy(cell_id):
    if cell_id not in _CELL:
        _CELL[cell_id] = SceneModel.from_yaml(_SCENE_PATH).cell_center(cell_id)[:2]
    return _CELL[cell_id]


class TestNamedBoards:
    """The review's fixture table (section 4), re-run under the corrected
    tube, and a third time (Slice 2, 2026-09-14) once "shells" gained the
    `thumb_pad` capsule and a widened `finger` capsule.

    Legacy tube column (aperture-only, centre-swung radius, as the review
    measured it): evidence -1.7, incident -8.0, foam@r3c3 -5.2, soda@r2c3
    -0.4 cm.  All four were already refused; the correction makes each ~4 cm
    more negative (the REST leg's tube radius at -45 deg went 7.5 -> 11.5 cm).

    Pre-Slice-2 "shells" column (visual shells only, no collision pads):
    evidence +4.5, incident -2.4, foam@r3c3 +3.4, soda@r2c3 +4.5 cm -- see
    the now-regression test TestShellsCoversThumbAndFingerPads. With the pads
    covered, `foam_on_r3c3` drops from +3.4 to +2.1 cm (the board sits closer
    to the thumb side than the others); the rest are unchanged to the
    displayed precision.  All shells numbers stay positive except
    `incident_board`, i.e. only `incident_board` remains refused by shells
    geometry -- the guard keeps hand="tube" and FOOTPRINT_MARGIN=0.0
    regardless (this slice changes no default).

    `pool_box_1`/`pool_cyl_1` on r2c3 (added here in Slice 2) are the two
    boards the tube correction newly refuses (TestCorrectionNewlyRefuses);
    both are comfortably positive under shells-with-pads, same as under the
    visual shells before it (+6.3 / +5.9 cm, unchanged to displayed
    precision -- neither board is anywhere near the thumb pad's gap)."""

    @pytest.mark.parametrize("objects,oid,expected_tube_cm,expected_shells_cm", [
        pytest.param(
            {"soda_can": _cell_xy("cell_r1c1"), "foam_block": _cell_xy("cell_r2c3")},
            "foam_block", -5.699, +4.479, id="evidence_board"),
        pytest.param(
            {"foam_block": (_cell_xy("cell_r2c3")[0], _cell_xy("cell_r2c3")[1] - 0.07)},
            "foam_block", -11.944, -2.414, id="incident_board"),
        pytest.param(
            {"foam_block": _cell_xy("cell_r3c3")},
            "foam_block", -9.369, +2.106, id="foam_on_r3c3"),
        pytest.param(
            {"soda_can": _cell_xy("cell_r2c3")},
            "soda_can", -4.379, +4.535, id="soda_can_on_r2c3"),
        pytest.param(
            {"pool_box_1": _cell_xy("cell_r2c3")},
            "pool_box_1", -3.719, +6.258, id="pool_box_1_on_r2c3"),
        pytest.param(
            {"pool_cyl_1": _cell_xy("cell_r2c3")},
            "pool_cyl_1", -3.079, +5.936, id="pool_cyl_1_on_r2c3"),
    ])
    def test_worst_clearance_under_the_corrected_tube(
            self, objects, oid, expected_tube_cm, expected_shells_cm):
        model = _board(objects)
        tube = _worst_over_any_guarded_route(model, oid, "tube")
        shells = _worst_over_any_guarded_route(model, oid, "shells")
        assert tube == pytest.approx(expected_tube_cm / 100.0, abs=0.003)
        assert shells == pytest.approx(expected_shells_cm / 100.0, abs=0.003)


# ---------------------------------------------------------------------------
# TestCorrectionNewlyRefuses -- legacy vs corrected tube, per board, per route
# ---------------------------------------------------------------------------

def _legacy_hand_radius(gripper_deg):
    """The aperture-only, centre-swung radius the guard flew before the
    2026-09-14 correction -- kept here, and only here, to measure what the
    correction changed.  Not a model of anything."""
    if gripper_deg is None:
        gripper_deg = -68.8
    y = -0.037 + 0.038 * math.sin(math.radians(gripper_deg))
    return max(math.hypot(0.025, 0.046), math.hypot(0.012, abs(y) + 0.010))


def _worst_per_route(model, oid, legacy):
    """{route: worst clearance} for `oid` under the tube, optionally with the
    legacy radius swapped in on the hand capsule."""
    out = {}
    for route, waypoints in R.FOOTPRINT_LEGS.items():
        worst = None
        for a, b in zip(waypoints, waypoints[1:]):
            qa = [a[j] for j in R.ARM7]
            qb = [b[j] for j in R.ARM7]
            gripper_deg = max(a["r_gripper"], b["r_gripper"], key=hand_radius)
            for q in joint_path(qa, qb, steps=13):
                caps = link_capsules(q, "right", gripper_deg)
                if legacy:
                    caps = [(n, p0, p1, _legacy_hand_radius(gripper_deg))
                            if n == "hand" else (n, p0, p1, r)
                            for n, p0, p1, r in caps]
                c = model.clearance(caps, ids=[oid])
                if c is not None and (worst is None or c.distance < worst):
                    worst = c.distance
        out[route] = worst
    return out


class TestCorrectionNewlyRefuses:
    """Every newly refused board, named.  Six scene objects (the four named
    ones and one of each 4 cm pool shape) on each of the nine grid cells,
    every guarded route: legacy tube >= 0 and corrected tube < 0.

    Exactly two boards flip -- the 4 cm pool box and pool cylinder on r2c3
    -- on every guarded route (they all share the REST tail).  Both were
    marginal under the legacy tube (+0.3 / +0.9 cm) and are 3-4 cm inside
    the corrected one; the visual-shells clearance at the same positions is
    +5.9 / +6.3 cm, i.e. the refusal is the isotropic tube's
    over-conservatism around an asymmetric hand, not a predicted contact.
    Everything larger on r2c3 and everything on r3c3 was already refused.

    Recorded, not loosened: the tube is now a correct bound and a loose one;
    recovering these boards is Slice 2's job ("shells", once it covers the
    collision pads and E1 has flown), not this radius's.
    """

    OBJECTS = ("red_cube", "blue_cylinder", "soda_can", "foam_block",
               "pool_box_1", "pool_cyl_1")
    CELLS = tuple(f"cell_r{r}c{c}" for r in (1, 2, 3) for c in (1, 2, 3))

    def test_exactly_the_two_pool_objects_on_r2c3_flip(self):
        flips = {}
        already = set()
        for oid in self.OBJECTS:
            for cell in self.CELLS:
                model = _board({oid: _cell_xy(cell)})
                legacy = _worst_per_route(model, oid, legacy=True)
                new = _worst_per_route(model, oid, legacy=False)
                for route in R.FOOTPRINT_LEGS:
                    if legacy[route] < 0:
                        already.add((oid, cell))
                    if legacy[route] >= 0 and new[route] < 0:
                        flips[(oid, cell, route)] = (legacy[route], new[route])
        assert {(o, c) for o, c, _r in flips} == {
            ("pool_box_1", "cell_r2c3"), ("pool_cyl_1", "cell_r2c3")}
        assert {r for _o, _c, r in flips} == set(R.FOOTPRINT_LEGS)
        for (oid, cell, route), (lo, nw) in flips.items():
            if oid == "pool_box_1":
                assert lo == pytest.approx(0.003, abs=0.002) and nw == pytest.approx(-0.037, abs=0.003)
            else:
                assert lo == pytest.approx(0.009, abs=0.002) and nw == pytest.approx(-0.031, abs=0.003)
        # and nothing that was refused stops being refused
        for oid in self.OBJECTS:
            for cell in self.CELLS:
                if (oid, cell) in already:
                    model = _board({oid: _cell_xy(cell)})
                    assert min(_worst_per_route(model, oid, legacy=False).values()) < 0


# ---------------------------------------------------------------------------
# TestSweepOrdering -- geomdist.py's method, copied in per the assignment
# ---------------------------------------------------------------------------

_BOX = int(mujoco.mjtGeom.mjGEOM_BOX)
_CAPSULE = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
_SPHERE = int(mujoco.mjtGeom.mjGEOM_SPHERE)


def _surface_points(m, d, g, n=41):
    """Points on the surface of a BOX geom. `geomdist.py`'s full version
    also samples cylinders and spheres/capsules; this sweep's target is
    always a box (the placed cube), so only that case is needed here."""
    h = np.asarray(m.geom_size[g], dtype=float)
    Rm = d.geom_xmat[g].reshape(3, 3)
    c = d.geom_xpos[g]
    u = np.linspace(-1.0, 1.0, n)
    pts = []
    for i in range(3):
        j, k = [x for x in range(3) if x != i]
        A, B = np.meshgrid(u, u)
        for s in (-1.0, 1.0):
            loc = np.zeros((A.size, 3))
            loc[:, i] = s * h[i]
            loc[:, j] = A.ravel() * h[j]
            loc[:, k] = B.ravel() * h[k]
            pts.append(loc)
    loc = np.concatenate(pts)
    return c + loc @ Rm.T


def _point_to_geom(m, d, pts, g):
    """Closed-form signed distance from world points to primitive geom g.
    `docs/reviews/probes-2026-09-12/geomdist.py`'s method verbatim (box, sphere,
    capsule -- the arm geoms this sweep's reference uses)."""
    t = int(m.geom_type[g])
    h = np.asarray(m.geom_size[g], dtype=float)
    Rm = d.geom_xmat[g].reshape(3, 3)
    c = d.geom_xpos[g]
    loc = (pts - c) @ Rm
    if t == _BOX:
        outside = np.maximum(np.abs(loc) - h, 0.0)
        dout = np.linalg.norm(outside, axis=1)
        inside = np.max(np.abs(loc) - h, axis=1)
        return np.where(dout > 0, dout, inside)
    if t == _SPHERE:
        return np.linalg.norm(loc, axis=1) - h[0]
    if t == _CAPSULE:
        z = np.clip(loc[:, 2], -h[1], h[1])
        seg = loc.copy()
        seg[:, 2] -= z
        return np.linalg.norm(seg, axis=1) - h[0]
    raise ValueError(f"unsupported geom type {t}")


def _mjcf_distance(m, d, arm_geom, target_geom, n=41):
    """Min distance from arm primitive to sampled target-box surface (m).
    `docs/reviews/probes-2026-09-12/geomdist.py`'s `distance()`, copied in."""
    best = float(_point_to_geom(
        m, d, _surface_points(m, d, target_geom, n), arm_geom).min())
    if int(m.geom_type[arm_geom]) == _BOX:
        # box-box: sample both ways, the corners of either can be closest.
        best = min(best, float(_point_to_geom(
            m, d, _surface_points(m, d, arm_geom, n), target_geom).min()))
    return best


@pytest.fixture(scope="module")
def sweep_model():
    doc = load_scene(_SCENE_PATH)
    m = mujoco.MjModel.from_xml_string(build_scene_model_xml(doc))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    for jid in range(m.njnt):
        if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            adr = m.jnt_qposadr[jid]
            d.qpos[adr:adr + 7] = m.qpos0[adr:adr + 7]
    mujoco.mj_forward(m, d)
    placer = ObjectPlacer(m, d, doc)
    return m, d, placer


def _run_sweep(sweep_model):
    m, d, placer = sweep_model
    OID = "pool_box_1"
    placer.reshape(OID, size=[0.06, 0.06, 0.06])
    ob = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, OID)
    og = int(m.body_geomadr[ob])
    oadr = int(m.jnt_qposadr[int(m.body_jntadr[ob])])
    jq = {n: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
          for n in R.R_JOINTS}
    # every arm geom MuJoCo can report contact on or draw: the collision
    # capsules, both visual shells, BOTH collision pads, the wrist ball
    arm_geom_names = ("r_upper_arm_col", "r_forearm_col",
                     "r_thumb_body", "r_finger_body",
                     "r_thumb_col", "r_finger_col", "r_wrist_ball")
    arm_geoms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
                for n in arm_geom_names]

    tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "table_top")
    tg = int(m.body_geomadr[tb])
    tc = d.geom_xpos[tg]
    th = m.geom_size[tg]
    grid_m = 0.04
    xs = np.arange(tc[0] - th[0] + 0.03, tc[0] + th[0] - 0.03 + 1e-9, grid_m)
    ys = np.arange(tc[1] - th[1] + 0.03, tc[1] + th[1] - 0.03 + 1e-9, grid_m)

    scene = SceneModel.from_yaml(_SCENE_PATH)
    scene._objects[OID] = replace(scene._objects[OID], size=(0.06, 0.06, 0.06))

    leg = R.FOOTPRINT_LEGS["PLACE_ROUTE"]
    samples = []
    for a, b in zip(leg, leg[1:]):
        qa = [a[j] for j in R.ARM7]
        qb = [b[j] for j in R.ARM7]
        # One aperture per leg, the same one the footprint guard itself uses
        # (rig_routes A2: the worse of the leg's two commanded apertures), and
        # the SAME one for tube, shells and the MJCF reference below. An
        # earlier draft evaluated the reference at the interpolated commanded
        # aperture instead and read 2/224 "shells > reference" inversions
        # that were only that mismatch: a wider finger can be farther from
        # an object on the thumb side.
        gg = max(a["r_gripper"], b["r_gripper"], key=hand_radius)
        for q in joint_path(qa, qb, 13):
            samples.append((q, gg))

    ts_gaps = []   # tube - shells (positive: tube reads MORE clearance)
    sr_gaps = []   # shells - reference (positive: shells reads MORE clearance)
    tr_gaps = []   # tube - reference (positive: tube reads MORE clearance)
    top_z = scene.table_surface_z
    for x in xs:
        for y in ys:
            d.qpos[oadr:oadr + 3] = (float(x), float(y), top_z + 0.031)
            d.qpos[oadr + 3:oadr + 7] = (1, 0, 0, 0)
            mujoco.mj_forward(m, d)
            scene.update_poses({OID: (float(x), float(y), top_z + 0.031)})
            tube_worst = shells_worst = ref_worst = None
            for q, gg in samples:
                c_tube = scene.clearances(
                    link_capsules(q, "right", gg), ids=[OID])[OID].distance
                c_shells = scene.clearances(
                    link_capsules(q, "right", gg, hand="shells"),
                    ids=[OID])[OID].distance
                tube_worst = c_tube if tube_worst is None else min(tube_worst, c_tube)
                shells_worst = c_shells if shells_worst is None else min(shells_worst, c_shells)
                for name, val in zip(R.ARM7, q):
                    d.qpos[jq[name]] = math.radians(val)
                d.qpos[jq["r_gripper"]] = math.radians(gg)
                mujoco.mj_forward(m, d)
                ref = min(_mjcf_distance(m, d, g, og) for g in arm_geoms)
                ref_worst = ref if ref_worst is None else min(ref_worst, ref)
            ts_gaps.append(tube_worst - shells_worst)
            sr_gaps.append(shells_worst - ref_worst)
            tr_gaps.append(tube_worst - ref_worst)
    return len(xs) * len(ys), ts_gaps, sr_gaps, tr_gaps


class TestSweepOrdering:
    """The 2026-09-12 assignment's sweep ordering, re-measured under the
    corrected tube at the review's resolution (4 cm grid, 224 positions, 13
    samples/leg, n=41 surface sampling), tube / shells / reference all at
    the leg's guarded aperture, reference over ALL MJCF hand geoms
    including the collision pads.  Re-measured again 2026-09-14 (Slice 2)
    once "shells" gained the `thumb_pad` capsule and a widened `finger`
    capsule:

      * tube <= reference at 224/224: the tube never reads more clearance
        than the true MJCF geometry.  This is the coverage claim on real
        objects, and the thing the 2026-09-14 correction exists for (before
        it: tube > shells at 3/224, by up to 2.8 cm).
      * tube <= shells at 224/224: nothing left of the finger poking out.
      * shells > reference at 0/224 (was 62/224 by up to 1.0 cm, before
        Slice 2 covered r_thumb_col and r_finger_col): the pad coverage
        closes the gap `TestShellsMissThumbPad` pinned; the max observed
        gap is now machine precision (~5e-13 m), the coverage bound
        attained rather than approached.  See
        test_arm_geometry_mjcf.py::TestShellsCoverMJCFHand /
        TestShellsCoversThumbAndFingerPads.
      * shells tighter than tube by > 1 cm at 189/224 (unchanged by Slice 2
        -- the pad additions are 0.05 cm at most on this sweep's positions,
        well under the 1 cm threshold): the corrected tube is a correct
        bound and a looser one -- up to 13 cm of over-conservatism against
        the true geometry on this sweep.

    NOT evidence the guard is adequate: it says the tube contains the model
    hand.  E3 (the real gripper vs the MJCF) and E1 (through-the-move) are
    still open.  Recovering the tube's over-conservatism by switching a
    consumer to "shells" is still Slice 2 PR2's job, after E1 -- this PR
    only makes "shells" a correct bound.
    """

    def test_tube_covers_reference_and_the_shells_gap_is_pinned(self, sweep_model):
        n, ts_gaps, sr_gaps, tr_gaps = _run_sweep(sweep_model)
        assert n == 224

        tol = 0.002  # 2 mm: sampling/rounding noise, not a real violation
        tr_violations = sum(1 for g in tr_gaps if g > tol)
        ts_violations = sum(1 for g in ts_gaps if g > tol)
        sr_violations = sum(1 for g in sr_gaps if g > tol)

        # Coverage on real objects: the tube never reads more clearance than
        # the MJCF hand geometry.  A single violation here is a hole in the
        # guard's model.
        assert tr_violations == 0, (
            f"tube read more clearance than the MJCF reference at "
            f"{tr_violations} positions (max {max(tr_gaps) * 100:.2f} cm)")
        assert ts_violations == 0, (
            f"tube read more clearance than shells at {ts_violations} "
            f"positions (max {max(ts_gaps) * 100:.2f} cm)")

        # The shells thumb-pad gap, closed 2026-09-14 (Slice 2): "shells" now
        # covers r_thumb_col and r_finger_col, so it never reads more
        # clearance than the MJCF reference on this sweep -- the small
        # positive residual is floating-point noise at the attained bound,
        # not a real gap (contrast the pre-Slice-2 62/224, up to 1.0 cm).
        assert sr_violations == 0, (
            f"shells read more clearance than the MJCF reference at "
            f"{sr_violations} positions (was 0 after Slice 2's pad coverage "
            "-- a violation here means the thumb-pad/finger-pad gap reopened)")
        assert max(sr_gaps) < 1e-6, (
            f"shells reads {max(sr_gaps) * 100:.4f} cm more clearance than "
            "the MJCF reference somewhere on the sweep -- expected machine "
            "precision now that shells covers the collision pads")

        # shells is a much tighter bound than the corrected tube at most of
        # the board -- the over-conservatism Slice 2 PR2 is meant to recover.
        # Unchanged by the pad coverage: the widened finger/thumb_pad add at
        # most ~0.05 cm here, well under the 1 cm threshold.
        assert sum(1 for g in ts_gaps if g < -0.01) == 189
        assert max(-g for g in tr_gaps) == pytest.approx(0.1296, abs=0.003)
