"""Part D (refs #212, #216, #214): the outer envelope a grasp planner may use for
a compound part.

The jaws close on OUTER faces, so a through-hole does not change the grasp.  But
a compound is only its envelope if its outer surface really is solid all the way
round and over the height; `grasp_envelope` derives that from the parts and
refuses when it cannot.  It is never assumed.  No IK and no physics here: the
planners get an arm that raises at the first IK request, so what is under test
is which shape gates let a part through.
"""

from __future__ import annotations

import math
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src", _ROOT / "native_mujoco"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from reachy_ai.scene.awareness import SceneModel  # noqa: E402
from reachy_ai.scene.grasp_envelope import NoGraspEnvelope, grasp_envelope  # noqa: E402

SCENE = str(_ROOT / "scenes" / "FWDCenterLabSivaPool.yaml")
HEIGHT = {"1": 0.02540, "2": 0.06985}
SIDE, RADIUS = 0.04921, 0.024606
RING_CORNER = math.hypot(RADIUS, RADIUS * math.tan(math.pi / 16))   # 25.088 mm


@pytest.fixture(scope="module")
def scene():
    return SceneModel.from_yaml(SCENE)


def _corners(obj):
    """(x, y) of every footprint corner of every box part, object frame."""
    from scene_io import rotation_matrix
    for p in obj.parts:
        rot = rotation_matrix(p["quat_wxyz"])
        for sx in (-0.5, 0.5):
            for sy in (-0.5, 0.5):
                v = (sx * p["size"][0], sy * p["size"][1])
                yield tuple(p["pos"][a] + rot[a][0] * v[0] + rot[a][1] * v[1] for a in (0, 1))


# ── The PART set ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("oid", ["part111_a", "part111_b", "part121_a", "part121_b"])
def test_a_holed_block_has_a_box_envelope(scene, oid):
    env = scene.grasp_target(oid)
    assert env.kind == "box" and env.envelope_of == "compound"
    assert env.size == pytest.approx((SIDE, SIDE, HEIGHT[oid[5]]), abs=1e-5)
    for x, y in _corners(env):
        assert abs(x) <= env.size[0] / 2 + 1e-6 and abs(y) <= env.size[1] / 2 + 1e-6


@pytest.mark.parametrize("oid", ["part211_a", "part211_b", "part221_a", "part221_b"])
def test_a_holed_cylinder_has_an_enclosing_cylinder_envelope(scene, oid):
    """The sim's ring is 16-sided with corners 0.48 mm proud of the real radius;
    an envelope has to enclose the solid, so it takes the corner radius."""
    env = scene.grasp_target(oid)
    assert env.kind == "cylinder" and env.envelope_of == "compound"
    assert env.size == pytest.approx((2 * RING_CORNER, 2 * RING_CORNER, HEIGHT[oid[5]]), abs=1e-5)
    assert all(math.hypot(x, y) <= env.size[0] / 2 + 1e-6 for x, y in _corners(env))


def test_only_the_shape_changes(scene):
    obj, env = scene.get("part221_a"), scene.grasp_target("part221_a")
    for field in ("id", "center", "quat", "semantic_class", "tags", "dynamic", "tracked", "collides"):
        assert getattr(env, field) == getattr(obj, field), field
    # The scene's own object is untouched: nothing else starts seeing it as solid.
    assert scene.get("part221_a").kind == "compound"
    assert scene.get("part221_a").envelope_of is None


@pytest.mark.parametrize("oid", ["part110_a", "part120_a", "part210_a", "part220_a",
                                 "red_cube", "pool_box_1", "pool_cyl_1", "soda_can"])
def test_a_plain_object_is_returned_as_itself(scene, oid):
    assert scene.grasp_target(oid) is scene.get(oid)


def test_the_envelope_follows_a_live_pose(scene):
    moved = SceneModel.from_yaml(SCENE)
    moved.update_poses({"part121_a": (0.4, -0.1, 0.8)})
    assert moved.grasp_target("part121_a").center == (0.4, -0.1, 0.8)


# ── Shapes with no honest envelope ───────────────────────────────────────────

def _compound(parts):
    doc = {"objects": [{"id": "x", "geometry": {"kind": "compound", "parts": parts},
                        "pose": {"position": [0.4, 0.0, 0.8]}}]}
    return SceneModel.from_doc(doc).get("x")


def _box(size, pos=(0, 0, 0), quat=(1, 0, 0, 0)):
    return {"kind": "box", "size": list(size), "pos": list(pos), "quat_wxyz": list(quat)}


H = 0.05


@pytest.mark.parametrize("parts, why", [
    # an L: half of two sides is air
    ([_box((0.05, 0.015, H), (0, -0.0175, 0)), _box((0.015, 0.035, H), (-0.0175, 0.0075, 0))],
     "outline"),
    # a C: the hull is a square, but one face is open in the middle
    ([_box((0.05, 0.012, H), (0, 0.019, 0)), _box((0.05, 0.012, H), (0, -0.019, 0)),
      _box((0.012, 0.026, H), (-0.019, 0, 0))], "outline"),
    # a plate on a thin stem: solid outline only at the top
    ([_box((0.05, 0.05, 0.01), (0, 0, 0.02)), _box((0.01, 0.01, H), (0, 0, 0))], "outline"),
    # four walls, one of them only half height
    ([_box((0.05, 0.012, H), (0, 0.019, 0)), _box((0.05, 0.012, H / 2), (0, -0.019, 0)),
      _box((0.012, 0.026, H), (-0.019, 0, 0)), _box((0.012, 0.026, H), (0.019, 0, 0))],
     "outline"),
    # tipped 90 degrees about x: not upright
    ([_box((0.05, 0.05, 0.05), quat=(math.sqrt(0.5), math.sqrt(0.5), 0, 0))], "upright"),
])
def test_a_shape_without_a_solid_outer_surface_has_no_envelope(parts, why):
    with pytest.raises(NoGraspEnvelope, match=why):
        grasp_envelope(_compound(parts))


def test_a_ring_with_one_segment_missing_has_no_envelope(scene):
    ring = list(scene.get("part221_a").parts)
    del ring[2]      # a full-height segment, away from the axes so the bounds stay
    with pytest.raises(NoGraspEnvelope, match="outline"):
        grasp_envelope(_compound([dict(p) for p in ring]))


def test_a_cross_is_neither_a_box_nor_a_cylinder():
    with pytest.raises(NoGraspEnvelope, match="outline"):
        grasp_envelope(_compound([_box((0.06, 0.02, H)), _box((0.02, 0.06, H))]))


# ── What the planners now let through ────────────────────────────────────────

class _PastTheGates(BaseException):
    """Not an Exception: the planners catch those around IK."""


class _NoIKArm:
    def inverse_kinematics(self, *_a, **_k):
        raise _PastTheGates

    def forward_kinematics(self, *_a, **_k):
        raise _PastTheGates


def _gate(planner_name, scene, oid):
    """"passes" (reaches IK), or the refusal that came first."""
    from reachy_ai.motion.kinematics import CartesianPlanner
    from reachy_ai.tasks import crane_pick_live as CP
    from reachy_ai.tasks import side_grasp_live as SG
    x, y, _ = scene.cell_center("cell_r2c2")
    pose = CP.Observation((x, y, scene.table_surface_z + scene.get(oid).size[2] / 2),
                          (1.0, 0.0, 0.0, 0.0), 0.0, False, 0.0)
    plan = {"crane": CP.plan_crane_pick, "side": SG.side_candidates}[planner_name]
    events = []

    def on_event(kind, **kw):
        # The signature the live motion worker and the offline harness use: an
        # event field called `kind` would collide with it and raise TypeError.
        events.append((kind, kw))

    try:
        out = plan(CartesianPlanner(_NoIKArm(), scene=scene), scene, oid, pose,
                   on_event=on_event)
    except _PastTheGates:
        return "passes", events
    except CP.CraneRefused as exc:
        return f"{exc.stage}: {exc.reason}", events
    return sorted({c.refusal for c in out if c.refusal})[0], events


@pytest.mark.parametrize("planner, oid", [
    ("crane", "part111_a"), ("crane", "part121_a"), ("side", "part221_a"),
])
def test_a_compound_gate_that_was_only_about_the_kind_now_passes(scene, planner, oid):
    verdict, events = _gate(planner, scene, oid)
    assert verdict == "passes", verdict
    used = [kw for k, kw in events if k == "GRASP_ENVELOPE"]
    assert used and used[0]["object_id"] == oid and used[0]["of"] == "compound"
    assert used[0]["outline"] == ("box" if oid[4] == "1" else "cylinder")


@pytest.mark.parametrize("planner, oid, reason", [
    # Real reasons, kept.  The crane closes on two flat faces.
    ("crane", "part210_a", "two flat faces"), ("crane", "part211_a", "two flat faces"),
    ("crane", "part220_a", "two flat faces"), ("crane", "part221_a", "two flat faces"),
    # Side-on pads sit 34 mm up so the forearm clears the table; these parts are 25 mm.
    ("side", "part210_a", "tall; the pads at 34 mm"),
    ("side", "part211_a", "tall; the pads at 34 mm"),
    # The side-on grasp was measured on cubes and cylinders; blocks are neither.
    ("side", "part110_a", "side-on grasp is for cubes"),
    ("side", "part111_a", "side-on grasp is for cubes"),
    ("side", "part120_a", "side-on grasp is for cubes"),
    ("side", "part121_a", "side-on grasp is for cubes"),
])
def test_a_real_refusal_is_kept(scene, planner, oid, reason):
    verdict, _ = _gate(planner, scene, oid)
    assert reason in verdict, verdict


@pytest.mark.parametrize("planner, oid", [("crane", "part110_a"), ("crane", "part120_a"),
                                          ("side", "part220_a")])
def test_solid_parts_pass_as_before_and_use_no_envelope(scene, planner, oid):
    verdict, events = _gate(planner, scene, oid)
    assert verdict == "passes", verdict
    assert not [k for k, _ in events if k == "GRASP_ENVELOPE"]


@pytest.mark.parametrize("planner", ["crane", "side"])
def test_a_compound_without_an_envelope_is_refused_with_the_reason(scene, planner):
    from reachy_ai.scene.awareness import SceneModel as Model
    bent = Model.from_yaml(SCENE)
    broken = dict(bent.get("part121_a").parts[0])
    broken["size"] = [broken["size"][0] / 3, broken["size"][1], broken["size"][2]]
    import dataclasses
    obj = bent.get("part121_a")
    bent._objects["part121_a"] = dataclasses.replace(obj, parts=(broken,) + obj.parts[1:])
    verdict, _ = _gate(planner, bent, "part121_a")
    assert verdict.startswith("shape:") and "outline" in verdict, verdict


@pytest.mark.parametrize("planner", ["crane", "side"])
def test_a_placement_of_a_compound_without_an_envelope_is_refused_too(planner):
    """The placement planners read the object before they plan the pick; the
    panel calls them directly, so they must refuse, not crash."""
    import dataclasses
    from reachy_ai.motion.kinematics import CartesianPlanner
    from reachy_ai.tasks import crane_pick_live as CP
    from reachy_ai.tasks import crane_place_live as PL
    from reachy_ai.tasks import side_grasp_live as SG
    bent = SceneModel.from_yaml(SCENE)
    obj = bent.get("part121_a")
    part = dict(obj.parts[0])
    part["size"] = [part["size"][0] / 3, part["size"][1], part["size"][2]]
    bent._objects["part121_a"] = dataclasses.replace(obj, parts=(part,) + obj.parts[1:])
    x, y, _ = bent.cell_center("cell_r2c2")
    pose = CP.Observation((x, y, bent.table_surface_z + obj.size[2] / 2),
                          (1.0, 0.0, 0.0, 0.0), 0.0, False, 0.0)
    place = {"crane": PL.plan_crane_place, "side": SG.plan_side_place}[planner]
    with pytest.raises(CP.CraneRefused, match="outline") as caught:
        place(CartesianPlanner(_NoIKArm(), scene=bent), bent, "part121_a", "r1c1", pose)
    assert caught.value.stage == "shape"


def test_a_compound_neighbour_is_still_its_bounding_box(scene):
    """Only the object being grasped gets an envelope; obstacles are unchanged."""
    from reachy_ai.motion import grasp_alignment as G
    box = G.object_box(scene.get("part221_a"))
    assert box.half == pytest.approx((RADIUS, RADIUS, HEIGHT["2"] / 2), abs=1e-5)
