"""Issue #56/#74 (review 2026-09-12, Part 3): validate `link_capsules`'s hand
models against the compiled MJCF -- the model file is the reference, not a
re-implementation of the chain.

  * TestShellsMatchMJCF -- "shells" reproduces the frames MuJoCo places
    r_gripper_thumb / r_gripper_finger / r_wrist_ball at, to 2 mm.

  * TestTubeCoversMJCFHand -- the "tube" every consumer flies CONTAINS every
    corner of every hand box the MJCF hangs under r_wrist2hand (both visual
    shells and both collision pads) and the wrist ball, at every named
    waypoint, every FOOTPRINT_LEGS sample, and a grid over the full
    r_gripper x r_wrist_roll range -- and is tight (the bound is attained).

  * TestDemonstratedMisses -- regression cases for the misses the 2026-09-14
    review measured against the aperture-only, centre-swung radius: the
    collision pad 1.8-3.8 cm outside the tube on guarded poses. Each is now
    inside, checked straight off MuJoCo's own geom frames.

  * TestTubeVsShellsCapsules -- the assignment's "tube bounds the shells"
    inequality, as it actually stands now: the tube contains every shells
    capsule's CENTRELINE; the capsule SURFACES still poke out by the shells
    capsules' own slack over their boxes (<= 1.2 cm), which is not a
    coverage gap against the MJCF and is pinned, not asserted away. Numbers
    re-derived 2026-09-14 (Slice 2) for the widened `finger` capsule and the
    new `thumb_pad` capsule, both of which count toward this margin.

  * TestShellsCoverMJCFHand -- added 2026-09-14 (Slice 2): the exact
    analogue of TestTubeCoversMJCFHand for "shells", now that it carries a
    `thumb_pad` capsule (for `r_thumb_col`) and a widened `finger` capsule
    (for `r_finger_col`). Every corner of all four hand boxes, and the
    wrist ball's far surface, is inside SOME shells capsule at every named
    waypoint, every FOOTPRINT_LEGS sample, and the full r_gripper x
    r_wrist_roll grid -- worst excess machine precision.

  * TestShellsCoversThumbAndFingerPads -- the regression case for the
    DISCREPANCY TestShellsMissThumbPad pinned 2026-09-14: `r_thumb_col` was
    2.05 cm outside every shells capsule (1.25 cm with the finger shut) and
    `r_finger_col` was 0.5 mm outside the finger capsule. Both pads are now
    inside, to machine precision, checked straight off MuJoCo's own geom
    frames -- the old numbers are recorded in the parametrisation.

Offline throughout: MuJoCo is used only to `mj_forward` a compiled model and
read back geom frames, never `mj_step`, and no server is started.
"""

import math
import os
import sys

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../src"))
sys.path.insert(0, os.path.join(_HERE, "../../native_mujoco"))

from objects import build_scene_model_xml  # noqa: E402
from scene_io import load_scene  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402
from reachy_ai.motion.kinematics import (  # noqa: E402
    _GRIPPER_OPEN_LIMIT_DEG,
    _GRIPPER_SHUT_LIMIT_DEG,
    _WRIST_ROLL_LIMIT_DEG,
    hand_radius,
    joint_path,
    link_capsules,
    link_frames,
)

_SCENE_PATH = os.path.join(_HERE, "../../scenes/FWDCenterLabSivaPool.yaml")

#: The full commanded range: the MJCF's own joint limits (-68.8 open, +20.05
#: shut -- see kinematics._GRIPPER_OPEN_LIMIT_DEG) plus two interior points.
GRIPPER_SAMPLES = (-68.8, -45.0, 0.0, 20.0)

#: Every hand box the MJCF hangs under r_wrist2hand.  The pads are what
#: physics contacts with; the shells are the project's description of the
#: hand's volume.  The tube must contain all four.
HAND_BOXES = ("r_thumb_body", "r_finger_body", "r_thumb_col", "r_finger_col")

#: Dense coverage grid over both joints' supported ranges.
ROLL_GRID = tuple(float(r) for r in range(-45, 46, 5))
GRIPPER_GRID = (_GRIPPER_OPEN_LIMIT_DEG, -68.75) + tuple(
    float(g) for g in range(-65, 21, 5)) + (_GRIPPER_SHUT_LIMIT_DEG,)


@pytest.fixture(scope="module")
def compiled():
    doc = load_scene(_SCENE_PATH)
    xml = build_scene_model_xml(doc)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    for jid in range(m.njnt):
        if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            adr = m.jnt_qposadr[jid]
            d.qpos[adr:adr + 7] = m.qpos0[adr:adr + 7]
    mujoco.mj_forward(m, d)
    return m, d


def _joint_qpos_addrs(m):
    return {n: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in R.R_JOINTS}


def _set_arm(m, d, jq, q7, gripper_deg):
    for name, val in zip(R.ARM7, q7):
        d.qpos[jq[name]] = math.radians(val)
    d.qpos[jq["r_gripper"]] = math.radians(gripper_deg)
    mujoco.mj_forward(m, d)


def _box_endpoints(m, d, gname):
    """World endpoints of the capsule that bounds box geom `gname`, along its
    own local Z through its centre -- the same convention `link_capsules`
    uses for "shells", and the one `docs/reviews/probes-2026-09-12/
    probe_hand_candidates.py` validated this construction against."""
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname)
    h = m.geom_size[g]
    Rm = d.geom_xmat[g].reshape(3, 3)
    c = d.geom_xpos[g]
    axis = Rm @ np.array([0.0, 0.0, 1.0])
    return c - axis * h[2], c + axis * h[2]


def _named_poses():
    """Every named waypoint pose in rig_routes (postures + route waypoints),
    plus the wave's two poses."""
    poses = {}
    for name, p in R.POSTURES.items():
        poses[name] = p
    for route_name in ("PLACE_ROUTE", "STOW_ROUTE", "LIFT_TO_PRESENT",
                       "LOWER_TO_REST", "RAISE_TO_SIDE", "STOW_FROM_SIDE"):
        for wp in R.route_named(route_name):
            poses.setdefault(wp.name, wp.pose)
    poses["WAVE_A"] = R.WAVE_A
    poses["WAVE_B"] = R.WAVE_B
    return poses


def _footprint_leg_samples():
    """13 samples of every FOOTPRINT_LEGS leg, as (label, q7) pairs."""
    out = []
    for route_name, waypoints in R.FOOTPRINT_LEGS.items():
        for i, (a, b) in enumerate(zip(waypoints, waypoints[1:])):
            qa = [a[j] for j in R.ARM7]
            qb = [b[j] for j in R.ARM7]
            for k, q7 in enumerate(joint_path(qa, qb, steps=13)):
                out.append((f"{route_name}[{i}]#{k}", q7))
    return out


def _assert_shells_match_mjcf(m, d, jq, q7, gripper_deg, label):
    _set_arm(m, d, jq, q7, gripper_deg)
    caps = {c[0]: c for c in link_capsules(q7, "right", gripper_deg, hand="shells")}

    thumb_p0, thumb_p1 = _box_endpoints(m, d, "r_thumb_body")
    finger_p0, finger_p1 = _box_endpoints(m, d, "r_finger_body")
    wrist_ball = d.geom_xpos[
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "r_wrist_ball")]

    checks = (
        ("thumb", thumb_p0, thumb_p1),
        ("finger", finger_p0, finger_p1),
        ("wrist_ball", wrist_ball, wrist_ball),
    )
    for name, mjcf_p0, mjcf_p1 in checks:
        _n, mine_p0, mine_p1, _r = caps[name]
        for mine, mjcf in ((mine_p0, mjcf_p0), (mine_p1, mjcf_p1)):
            err = max(abs(a - b) for a, b in zip(mine, mjcf))
            assert err < 0.002, (
                f"{label} gripper={gripper_deg}: {name} endpoint off by "
                f"{err * 1000:.2f} mm")


class TestShellsMatchMJCF:
    """`link_capsules(hand="shells")` must reproduce the frames MuJoCo itself
    places r_gripper_thumb / r_gripper_finger / r_wrist_ball at -- not
    approximate them. `upper_arm`/`forearm` are unchanged by `hand="shells"`
    and are already covered by test_arm_clearance.py."""

    def test_named_waypoints(self, compiled):
        m, d = compiled
        jq = _joint_qpos_addrs(m)
        for name, pose in _named_poses().items():
            q7 = [pose[j] for j in R.ARM7]
            for gripper_deg in GRIPPER_SAMPLES:
                _assert_shells_match_mjcf(m, d, jq, q7, gripper_deg, name)

    def test_footprint_leg_samples(self, compiled):
        m, d = compiled
        jq = _joint_qpos_addrs(m)
        for label, q7 in _footprint_leg_samples():
            for gripper_deg in GRIPPER_SAMPLES:
                _assert_shells_match_mjcf(m, d, jq, q7, gripper_deg, label)


def _box_corners(m, d, gname):
    """World positions of the 8 corners of box geom `gname`."""
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname)
    h = m.geom_size[g]
    Rm = d.geom_xmat[g].reshape(3, 3)
    c = d.geom_xpos[g]
    return [c + Rm @ np.array([sx * h[0], sy * h[1], sz * h[2]])
            for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]


def _segment_distance(p, a, b):
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _tube(q7, gripper_deg):
    """(a, b, radius) of the "tube" hand capsule for this pose."""
    for name, p0, p1, r in link_capsules(q7, "right", gripper_deg):
        if name == "hand":
            return np.array(p0), np.array(p1), r
    raise AssertionError("no hand capsule")


def _tube_excess(m, d, jq, q7, gripper_deg):
    """(worst corner excess, worst corner slack) of the MJCF hand geometry
    against the tube at this pose: excess > 0 means some corner of some hand
    box (or the wrist ball's surface) is OUTSIDE the tube; slack is how far
    inside the *farthest* corner sits (0 means the bound is attained)."""
    _set_arm(m, d, jq, q7, gripper_deg)
    a, b, r = _tube(q7, gripper_deg)
    farthest = max(_segment_distance(p, a, b)
                   for gname in HAND_BOXES for p in _box_corners(m, d, gname))
    ball = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "r_wrist_ball")
    ball_reach = _segment_distance(d.geom_xpos[ball], a, b) + float(m.geom_size[ball][0])
    return max(farthest, ball_reach) - r, r - farthest


class TestTubeCoversMJCFHand:
    """The tube every consumer flies must contain the whole MJCF hand.

    "Contain" is checked corner by corner against MuJoCo's own geom frames
    (not against this repo's reconstruction of them): every corner of every
    hand box, and the wrist ball's far surface, is inside the capsule
    `link_capsules` returns for that pose -- including the end caps, which
    is what caught the tube being 4 mm too short for the finger's far corner
    when it hangs straight down.  And the bound is TIGHT: at every pose the
    farthest corner sits on the tube's surface to machine precision, so the
    radius is exactly the geometry and not a padded guess.
    """

    def test_named_waypoints_and_footprint_samples(self, compiled):
        m, d = compiled
        jq = _joint_qpos_addrs(m)
        poses = [(n, [p[j] for j in R.ARM7]) for n, p in _named_poses().items()]
        poses += _footprint_leg_samples()
        for label, q7 in poses:
            for g in GRIPPER_SAMPLES + (_GRIPPER_SHUT_LIMIT_DEG,):
                excess, _slack = _tube_excess(m, d, jq, q7, g)
                assert excess <= 1e-9, (
                    f"{label} gripper={g}: MJCF hand geometry outside the "
                    f"tube by {excess * 1000:.2f} mm")

    def test_full_gripper_and_roll_range(self, compiled):
        """A grid over the whole supported r_gripper x r_wrist_roll range, on
        HOVER (the pose the coverage depends on only through those two
        joints -- everything upstream rotates tube and hand together)."""
        m, d = compiled
        jq = _joint_qpos_addrs(m)
        base = [R.HOVER[j] for j in R.ARM7]
        worst_excess, worst_slack = -1.0, 1.0
        for roll in ROLL_GRID:
            q7 = list(base)
            q7[6] = roll
            for g in GRIPPER_GRID:
                excess, slack = _tube_excess(m, d, jq, q7, g)
                worst_excess = max(worst_excess, excess)
                worst_slack = min(worst_slack, slack)
                assert excess <= 1e-9, (
                    f"roll={roll} gripper={g}: outside by {excess * 1000:.2f} mm")
                assert slack <= 1e-6, (
                    f"roll={roll} gripper={g}: tube padded by {slack * 1000:.2f} mm "
                    "beyond the farthest corner -- radius no longer matches the MJCF")
        # Belt and braces: machine precision across the whole grid.
        assert worst_excess <= 1e-9 and worst_slack >= -1e-9

    def test_unknown_inputs_bound_the_whole_range(self):
        """`None` for either joint must be a bound over that joint's range."""
        both = hand_radius(None, None)
        for roll in ROLL_GRID:
            assert hand_radius(None, roll) <= both + 1e-12
            for g in GRIPPER_GRID:
                exact = hand_radius(g, roll)
                assert exact <= hand_radius(g, None) + 1e-12
                assert exact <= hand_radius(None, roll) + 1e-12
                assert exact <= both + 1e-12
        # and the worst case is the wide-open hand at the roll limit, as the
        # header comment in kinematics.py states
        assert both == pytest.approx(
            hand_radius(_GRIPPER_OPEN_LIMIT_DEG, _WRIST_ROLL_LIMIT_DEG))

    def test_the_tube_is_the_pose_own_roll_not_the_worst(self):
        """link_capsules sizes the tube for the pose's actual wrist roll."""
        hover = [R.HOVER[j] for j in R.ARM7]     # roll 0
        rest = [R.REST[j] for j in R.ARM7]       # roll 30
        assert _tube(hover, -45.0)[2] == pytest.approx(hand_radius(-45.0, 0.0))
        assert _tube(rest, -45.0)[2] == pytest.approx(hand_radius(-45.0, 30.0))
        assert _tube(rest, -45.0)[2] > _tube(hover, -45.0)[2]


class TestDemonstratedMisses:
    """Regression cases: the misses the 2026-09-14 review measured against
    the aperture-only, centre-swung `hand_radius` (docs/adr/0003, "A
    discrepancy this ADR does not paper over"), each now inside the tube.

    `old_miss_cm` is how far the geom's farthest corner sat OUTSIDE the old
    tube, straight off MuJoCo's geom frames; it is recorded so the size of
    what was being flown through is not lost, and is not asserted.
    """

    @pytest.mark.parametrize("posture,gripper_deg,geom,old_miss_cm", [
        ("HOVER", -68.8, "r_finger_col", 2.19),
        ("HOVER", -68.8, "r_finger_body", 2.88),
        ("HOVER", -45.0, "r_finger_col", 1.77),    # PLACE_ROUTE's guarded aperture
        ("HOVER", -45.0, "r_finger_body", 2.37),
        ("REST", -68.8, "r_finger_col", 3.80),
        ("REST", -68.8, "r_finger_body", 4.50),
        ("REST", -45.0, "r_finger_col", 3.38),
        ("REST", -45.0, "r_finger_body", 3.99),
        ("REST", 0.0, "r_finger_col", 1.05),
        ("REST", 0.0, "r_finger_body", 1.20),
        ("REST_SHUT", -68.8, "r_finger_col", 3.80),
        ("PRESENT", -68.8, "r_finger_col", 2.19),
        ("HOME", -68.8, "r_finger_col", 2.19),
    ])
    def test_geom_now_inside_the_tube(self, compiled, posture, gripper_deg,
                                      geom, old_miss_cm):
        m, d = compiled
        jq = _joint_qpos_addrs(m)
        q7 = [getattr(R, posture)[j] for j in R.ARM7]
        _set_arm(m, d, jq, q7, gripper_deg)
        a, b, r = _tube(q7, gripper_deg)
        farthest = max(_segment_distance(p, a, b) for p in _box_corners(m, d, geom))
        assert r - farthest >= -1e-9, (
            f"{posture} gripper={gripper_deg}: {geom} still outside the tube by "
            f"{(farthest - r) * 100:.2f} cm (was {old_miss_cm} cm)")


_SHELLS_HAND_CAPSULE_NAMES = ("thumb", "thumb_pad", "finger", "wrist_ball")


def _shells_excess(m, d, jq, q7, gripper_deg):
    """Worst corner excess of the MJCF hand geometry against the shells
    capsules at this pose: excess > 0 means some corner of some hand box
    (or the wrist ball's surface) is outside EVERY shells capsule.  The
    analogue of `_tube_excess`, but against several capsules instead of
    one: a corner only needs to be inside the nearest one."""
    _set_arm(m, d, jq, q7, gripper_deg)
    caps = [(np.array(p0), np.array(p1), r) for n, p0, p1, r in
            link_capsules(q7, "right", gripper_deg, hand="shells")
            if n in _SHELLS_HAND_CAPSULE_NAMES]

    def excess(p):
        return min((_segment_distance(p, a, b) if np.any(a != b)
                    else float(np.linalg.norm(p - a))) - r
                   for a, b, r in caps)

    farthest = max(excess(p) for gname in HAND_BOXES for p in _box_corners(m, d, gname))
    ball = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "r_wrist_ball")
    ball_excess = excess(d.geom_xpos[ball]) + float(m.geom_size[ball][0])
    return max(farthest, ball_excess)


class TestShellsCoverMJCFHand:
    """The exact analogue of `TestTubeCoversMJCFHand` for the opt-in
    "shells" hand model, now that it carries the `thumb_pad` capsule and a
    widened `finger` capsule (2026-09-14, Slice 2, ADR-0003 item 4 /
    `TestShellsCoversThumbAndFingerPads`): every corner of all four hand
    boxes, and the wrist ball's far surface, must be inside SOME shells
    capsule -- not necessarily the same one -- at every named waypoint,
    every `FOOTPRINT_LEGS` sample, and a grid over the full `r_gripper` x
    `r_wrist_roll` range.

    Unlike the tube, "shells" is several capsules and is not claimed to be
    a *tight* bound in every direction (a corner covered by one capsule's
    slack is not "the bound attained" the way the tube's single isotropic
    radius is) -- only that it is a bound at all, everywhere the tube is
    checked.
    """

    def test_named_waypoints_and_footprint_samples(self, compiled):
        m, d = compiled
        jq = _joint_qpos_addrs(m)
        poses = [(n, [p[j] for j in R.ARM7]) for n, p in _named_poses().items()]
        poses += _footprint_leg_samples()
        for label, q7 in poses:
            for g in GRIPPER_SAMPLES + (_GRIPPER_SHUT_LIMIT_DEG,):
                excess = _shells_excess(m, d, jq, q7, g)
                assert excess <= 1e-9, (
                    f"{label} gripper={g}: MJCF hand geometry outside every "
                    f"shells capsule by {excess * 1000:.2f} mm")

    def test_full_gripper_and_roll_range(self, compiled):
        """A grid over the whole supported r_gripper x r_wrist_roll range, on
        HOVER, matching `TestTubeCoversMJCFHand.test_full_gripper_and_roll_
        range`'s coverage."""
        m, d = compiled
        jq = _joint_qpos_addrs(m)
        base = [R.HOVER[j] for j in R.ARM7]
        worst_excess = -1.0
        for roll in ROLL_GRID:
            q7 = list(base)
            q7[6] = roll
            for g in GRIPPER_GRID:
                excess = _shells_excess(m, d, jq, q7, g)
                worst_excess = max(worst_excess, excess)
                assert excess <= 1e-9, (
                    f"roll={roll} gripper={g}: outside every shells capsule "
                    f"by {excess * 1000:.2f} mm")
        assert worst_excess <= 1e-9


def _shells_axis_margins(posture_name, gripper_deg):
    """(worst surface margin, worst centreline margin) of the shells hand
    capsules against the tube's radius, measured from the tube axis."""
    pose = getattr(R, posture_name)
    q7 = [pose[j] for j in R.ARM7]
    _shoulder, _elbow, wrist, Rm = link_frames(q7, "right")
    axis_dir = Rm @ np.array([0.0, 0.0, -1.0])
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    r_tube = hand_radius(gripper_deg, q7[6])
    surface = centre = None
    for name, p0, p1, r in link_capsules(q7, "right", gripper_deg, hand="shells"):
        if name in ("upper_arm", "forearm"):
            continue
        for p in (np.array(p0), np.array(p1)):
            rel = p - wrist
            perp = float(np.linalg.norm(rel - np.dot(rel, axis_dir) * axis_dir))
            surface = perp + r if surface is None else max(surface, perp + r)
            centre = perp if centre is None else max(centre, perp)
    return r_tube - surface, r_tube - centre


class TestTubeVsShellsCapsules:
    """The 2026-09-12 assignment's inequality "the tube's radius >= every
    shells capsule's distance from the wrist axis", as it stands after the
    tube correction.

    It holds for every shells capsule's CENTRELINE (the centrelines are
    inside the boxes, and the tube contains the boxes).  It does NOT hold
    for the capsule SURFACES, and should not be made to: a shells capsule is
    a cylinder around a box and pokes out past the box's own corners, so its
    surface can sit outside a tube that contains the box exactly.  That
    shortfall is bounded by the capsule's own radius -- it is the shells
    model's slack, not a coverage gap against the MJCF (which
    TestTubeCoversMJCFHand / TestShellsCoverMJCFHand check directly).
    Pinned rather than asserted away.

    Re-derived 2026-09-14 (Slice 2): the widened `finger` capsule (now
    `hypot(0.014, 0.008)`, was `hypot(0.012, 0.010)`) increases the surface
    shortfall by ~0.05 cm wherever the finger sets the widest capsule; the
    new `thumb_pad` capsule (same radius as the widened finger, further
    from the axis than the thumb shell at open apertures) does not change
    which capsule is widest at any of these poses.
    """

    @pytest.mark.parametrize("posture,gripper_deg,expected_surface_cm", [
        ("REST", -68.8, -1.1946),
        ("REST", -45.0, -0.8424),
        ("REST", 0.0, -0.4704),
        ("REST", 20.0, -0.4704),
        ("REST_SHUT", -68.8, -1.1946),
        ("HOVER", -68.8, -1.1864),
        ("HOVER", -45.0, -0.8320),
        ("HOVER", 0.0, -0.3182),
        ("PRESENT", -68.8, -1.1864),
        ("HOME", -68.8, -1.1864),
    ])
    def test_centreline_inside_surface_shortfall_is_capsule_slack(
            self, posture, gripper_deg, expected_surface_cm):
        surface, centreline = _shells_axis_margins(posture, gripper_deg)
        assert centreline >= -1e-9
        assert surface == pytest.approx(expected_surface_cm / 100.0, abs=0.001)
        # the shortfall never exceeds the widest shells capsule's radius
        widest = max(r for n, _p0, _p1, r in
                     link_capsules([getattr(R, posture)[j] for j in R.ARM7],
                                   "right", gripper_deg, hand="shells")
                     if n in _SHELLS_HAND_CAPSULE_NAMES)
        assert -surface <= widest + 1e-9


class TestShellsCoversThumbAndFingerPads:
    """Regression case for the DISCREPANCY `TestShellsMissThumbPad` pinned
    2026-09-14, closed the same day (Slice 2, ADR-0003 item 4).

    Before this PR: the MJCF's r_thumb_col (pos z -0.085, half 0.022 -> z in
    [-0.107, -0.063] below r_gripper_thumb) sat ENTIRELY below the visual
    r_thumb_body box (z in [-0.060, +0.016]); the MJCF comment says the
    pinch pads were placed deliberately rather than derived from the "bulky
    visual box". "shells" was built from the visual boxes only, so it did
    not contain the pad physics actually contacts with: the pad's far
    corner was 2.05 cm outside every shells capsule (1.25 cm with the
    finger shut, when the finger capsule covered part of it), and
    r_finger_col was 0.5 mm outside the finger capsule (it is 2 mm wider in
    X than the shell).

    Now: a dedicated `thumb_pad` capsule covers `r_thumb_col`, and `finger`'s
    radius is widened to `hypot(0.014, 0.008)` (the pad's own corner
    distance, which exceeds the shell's) to cover `r_finger_col`. Both are
    inside to machine precision -- the same construction the tube uses,
    verified corner-for-corner off MuJoCo's own frames, not approximated.
    """

    @pytest.mark.parametrize("posture,gripper_deg,old_thumb_pad_out_cm", [
        ("HOVER", -68.8, 2.05), ("HOVER", -45.0, 2.05), ("HOVER", 0.0, 2.05),
        ("HOVER", 20.0, 1.25),
        ("REST", -68.8, 2.05), ("REST", -45.0, 2.05), ("REST", 0.0, 2.05),
        ("REST", 20.0, 1.25),
    ])
    def test_thumb_and_finger_pads_now_inside_the_shells_capsules(
            self, compiled, posture, gripper_deg, old_thumb_pad_out_cm):
        m, d = compiled
        jq = _joint_qpos_addrs(m)
        q7 = [getattr(R, posture)[j] for j in R.ARM7]
        _set_arm(m, d, jq, q7, gripper_deg)
        caps = [(np.array(p0), np.array(p1), r) for n, p0, p1, r in
                link_capsules(q7, "right", gripper_deg, hand="shells")
                if n in _SHELLS_HAND_CAPSULE_NAMES]

        def outside(p):
            return min(
                (_segment_distance(p, a, b) if np.any(a != b)
                 else float(np.linalg.norm(p - a))) - r
                for a, b, r in caps)

        thumb_pad = max(outside(p) for p in _box_corners(m, d, "r_thumb_col"))
        finger_pad = max(outside(p) for p in _box_corners(m, d, "r_finger_col"))
        assert thumb_pad <= 1e-9, (
            f"{posture} gripper={gripper_deg}: r_thumb_col still "
            f"{thumb_pad * 100:.2f} cm outside every shells capsule "
            f"(was {old_thumb_pad_out_cm} cm before the thumb_pad capsule)")
        assert finger_pad <= 1e-9, (
            f"{posture} gripper={gripper_deg}: r_finger_col still "
            f"{finger_pad * 100:.3f} cm outside the widened finger capsule")
