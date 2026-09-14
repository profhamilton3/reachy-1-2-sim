"""Issue #56/#74 (review 2026-09-12, Part 3, section 4): the fixture table
that motivated the "shells" hand mode, computed here with
`SceneModel.clearance` for both hand modes rather than quoted from the
review's own scratch probes.

Two things are checked, and (as in test_arm_geometry_mjcf.py) they are not
both simply confirmations of the assignment's expectations:

  * TestNamedBoards -- the four named boards in the review's table, worst
    clearance over ANY guarded route (every `FOOTPRINT_LEGS` leg), for both
    hand="tube" (today's default, unchanged) and hand="shells" (new,
    opt-in). Matches the review's recorded numbers to +/- 0.3 cm.

  * TestSweepOrdering -- a 224-position, 6 cm cube sweep on the PLACE_ROUTE
    tail, comparing tube, shells, and the MJCF reference (geomdist.py's
    method, copied in below) at each position, all three at the same
    aperture. The assignment asked for a universal ordering ("shells never
    reads more than tube and never less than the MJCF shell distance"). The
    second half holds everywhere, as the capsule bound predicts. The first
    half inverts at 3 positions -- pinned here rather than asserted away,
    and explained in the class docstring. It is the same underlying cause
    as TestTubeVsShellsAxisDistance in test_arm_geometry_mjcf.py:
    `hand_radius()` swings the finger box's centre, not its far end, so the
    open finger reaches outside the tube.

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
    """The review's fixture table (section 4): "flyable by geometry" rows
    read positive under "shells" despite reading negative under "tube" --
    which is the case for #56/#74 that a shared, better hand model changes
    an outcome, not just a number. The guard itself keeps hand="tube" and
    FOOTPRINT_MARGIN=0.0 regardless (this slice changes no default)."""

    @pytest.mark.parametrize("objects,oid,expected_tube_cm,expected_shells_cm", [
        pytest.param(
            {"soda_can": _cell_xy("cell_r1c1"), "foam_block": _cell_xy("cell_r2c3")},
            "foam_block", -1.7, +4.5, id="evidence_board"),
        pytest.param(
            {"foam_block": (_cell_xy("cell_r2c3")[0], _cell_xy("cell_r2c3")[1] - 0.07)},
            "foam_block", -8.0, -2.4, id="incident_board"),
        pytest.param(
            {"foam_block": _cell_xy("cell_r3c3")},
            "foam_block", -5.2, +3.4, id="foam_on_r3c3"),
        pytest.param(
            {"soda_can": _cell_xy("cell_r2c3")},
            "soda_can", -0.4, +4.5, id="soda_can_on_r2c3"),
    ])
    def test_worst_clearance_matches_the_review(
            self, objects, oid, expected_tube_cm, expected_shells_cm):
        model = _board(objects)
        tube = _worst_over_any_guarded_route(model, oid, "tube")
        shells = _worst_over_any_guarded_route(model, oid, "shells")
        assert tube == pytest.approx(expected_tube_cm / 100.0, abs=0.003)
        assert shells == pytest.approx(expected_shells_cm / 100.0, abs=0.003)


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
    arm_geom_names = ("r_upper_arm_col", "r_forearm_col",
                     "r_thumb_body", "r_finger_body", "r_wrist_ball")
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
    return len(xs) * len(ys), ts_gaps, sr_gaps


class TestSweepOrdering:
    """DISCREPANCY (partial) from the 2026-09-12 assignment, reported for
    Opus rather than forced to pass: the assignment asked this sweep to
    assert "shells" never reads more clearance than "tube" and never less
    than the MJCF reference, as a universal ordering. Measured here at the
    review's own resolution (4 cm grid, 224 positions, 13 samples/leg, n=41
    surface sampling), tube, shells and reference all at the leg's guarded
    aperture:

      * shells <= reference (shells never less cautious than the true MJCF
        shell geometry) at 224/224 -- the capsule-around-box bound holds, as
        it must. An earlier draft read 2/224 inversions here; that was the
        reference being evaluated at a different aperture than the capsules
        (see `_run_sweep`), not geometry.
      * tube > shells (tube less cautious than shells) at 3/224 positions,
        by up to 2.8 cm. THIS is the real finding.

    Same cause as TestTubeVsShellsAxisDistance in test_arm_geometry_mjcf.py:
    `hand_radius()` swings the finger box's centre (3.8 cm below the hinge)
    rather than its far end (7.6 cm), so at the -45 deg aperture the
    PLACE_ROUTE tail is guarded at, the finger shell reaches ~2.4 cm outside
    the tube in its swing direction (the MJCF collision pad `r_finger_col`
    ~1.8 cm). REST/REST_SHUT's r_wrist_roll=30 adds a further 1.6 cm of
    offset. All three violating positions are within the last two columns
    of the sweep grid (the near-right corner strip PLACE_ROUTE's tail is
    already worst on -- review R1), and the magnitudes are small relative to
    the up-to-9.3 cm of over-conservatism "tube" carries against the true
    reference everywhere else on the board.

    NOT evidence the guard is unsafe against any board it has been flown
    over: it still flies "tube" only, at the margin it always has. It IS a
    directional under-conservatism in the model consumers fly today; the
    `hand_radius` lever arm is the first open item in docs/adr/0003 and is
    deliberately not corrected in this slice (it changes which boards the
    guard refuses, so it needs the fixture table re-run alongside).
    """

    def test_ordering_mostly_holds_with_pinned_exceptions(self, sweep_model):
        n, ts_gaps, sr_gaps = _run_sweep(sweep_model)
        assert n == 224

        tol = 0.002  # 2 mm: sampling/rounding noise, not a real violation
        ts_violations = sum(1 for g in ts_gaps if g > tol)
        sr_violations = sum(1 for g in sr_gaps if g > tol)

        # The bound that must hold: a capsule that contains each shell box
        # can never read MORE clearance than the box itself at the same
        # aperture.
        assert sr_violations == 0, (
            f"shells read more clearance than the MJCF reference at "
            f"{sr_violations} positions (max gap {max(sr_gaps) * 100:.2f} cm) "
            "-- the capsule bound is broken, or the reference is being "
            "evaluated at a different aperture than the capsules again")

        assert ts_violations == 3, (
            f"tube read more clearance than shells at {ts_violations} "
            "positions (measured and pinned at 3 -- see class docstring; "
            "if this changed, re-derive the numbers rather than the count)")
        assert max(ts_gaps) == pytest.approx(0.0275, abs=0.001)

        # The property that actually matters for #56/#74: shells is a much
        # tighter (and still safe-in-practice) bound than tube at most board
        # positions (158/224 measured), not just occasionally.
        assert sum(1 for g in ts_gaps if g < -0.01) == 158
