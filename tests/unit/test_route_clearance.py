"""tools/route_clearance (A0b, reachy-tabletop-ai#66): the vendored route hash, the board-distance method, the exact
box distances, the verdict's reasons, conditional contact declarations, the occupancy rule, the target-placement
tolerance, and a deterministic record. Offline MuJoCo only: no server, SDK or Docker.

The full B2_GRIP_TUNE run takes minutes, so it is not repeated here; the record in the PR is its output."""

import copy
import hashlib
import json
import math
import pathlib

import mujoco
import numpy as np
import pytest

from tools.route_clearance import arm_route_hash, evaluate as E, flight, geometry as G, record, route as R
from tools.route_clearance.limits import ARM_LIMITS_DEG
from tools.route_clearance.world import World

ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "route_clearance"
GOLDEN = ROOT / "tests" / "fixtures" / "arm_routes"
ROUTES = ROOT / "tests" / "fixtures" / "route_clearance" / "arm_routes.json"
SCENE = "FWDCenterLabSivaPool"

# SHA-256 of the tabletop files at 7468ccb, so a fixture edited here is caught.
TABLETOP_SHA = {
    GOLDEN / "golden.json": "d2cdee8c70e639e317f0d2991de99d061d115f7b29a69319d5e8f52defe92137",
    GOLDEN / "golden_hashes.json": "fb98466fa893fc6913af978e3647b4985dd02664a779102fd03a499cdb98b05c",
    ROUTES: "c312e0e8290928f3de20261ea2abf4abffd4a279a895afc28caa08b9b3825642",
}
B2_GRIP_TUNE_HASH = "89e62bda86b010e898d7a335eed7f630ee468e9944b08ea9a27f0fc0d81daf04"   # tabletop route_hashes()


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def world():
    return World(ROOT / "scenes" / (SCENE + ".yaml"))


@pytest.fixture
def doc():
    return R.load_doc(ROUTES)


def pose(doc, name, gripper):
    return dict(doc["poses"][name]["joints_deg"], r_gripper=gripper)


# ---- the vendored hash ----------------------------------------------------------------------------------------

def test_fixtures_are_the_tabletop_files_byte_for_byte():
    for path, digest in TABLETOP_SHA.items():
        assert sha(path) == digest, path


def test_vendored_hash_header_names_the_sha_of_its_body():
    text = (TOOL / "arm_route_hash.py").read_text(encoding="utf-8")
    header, marker, body = text.partition("# --- end of vendoring header ---\n")
    assert marker
    claimed = [line.split(":", 1)[1].strip() for line in header.splitlines() if line.startswith("# Source SHA-256:")]
    assert claimed == [hashlib.sha256(body.encode("utf-8")).hexdigest()]
    assert "reachy-tabletop-ai" in header and "# Source commit: 7468ccb" in header


def test_vendored_hash_reproduces_the_golden_hashes():
    golden = json.loads((GOLDEN / "golden.json").read_text(encoding="utf-8"))
    expected = json.loads((GOLDEN / "golden_hashes.json").read_text(encoding="utf-8"))
    limits = {joint: tuple(pair) for joint, pair in expected["limits"].items()}
    got = {name: arm_route_hash.route_hash(name, golden, limits) for name in golden["routes"]}
    assert got == expected["hashes"]


def test_b2_grip_tune_hashes_as_on_the_tabletop(doc):
    assert R.hash_of(doc, "B2_GRIP_TUNE") == B2_GRIP_TUNE_HASH
    changed = copy.deepcopy(doc)
    changed["poses"]["B2_PLACE"]["joints_deg"]["r_wrist_roll"] += 0.01
    assert R.hash_of(changed, "B2_GRIP_TUNE") != B2_GRIP_TUNE_HASH
    limits = dict(ARM_LIMITS_DEG, r_wrist_roll=(-45.0, 45.0))
    assert R.hash_of(doc, "B2_GRIP_TUNE", limits) != B2_GRIP_TUNE_HASH


def test_route_expands_out_and_back_with_the_capped_durations(doc):
    legs = R.expand(doc, "B2_GRIP_TUNE")
    assert [(leg.start, leg.to) for leg in legs] == [
        ("REST", "REST_OUT"), ("REST_OUT", "TRANSPORT"), ("TRANSPORT", "B2_SAFE"), ("B2_SAFE", "B2_PLACE"),
        ("B2_PLACE", "B2_SAFE"), ("B2_SAFE", "TRANSPORT"), ("TRANSPORT", "REST_OUT"), ("REST_OUT", "REST")]
    assert [leg.duration_s for leg in legs] == [4.0, 8.4, 6.0, 5.0, 5.0, 6.0, 8.4, 4.0]
    # the gripper at a pose is the one its arriving leg commands, both ways
    assert [leg.q_to["r_gripper"] for leg in legs] == [20.0, 20.0, -29.0, -29.0, -29.0, 20.0, 20.0, 20.0]


# ---- distances ------------------------------------------------------------------------------------------------

def test_board_distance_step1_cross_check(world, doc):
    """Step 1 board_check.txt: B2_SAFE -> B2_PLACE at -29 is +0.2 cm (forearm, at B2_PLACE) analytic, where
    mj_geomDistance reads a hand box against the slab as 0 mid-leg; PICK_SAFE -> PICK is +2.2 cm (palm)."""
    leg = R.expand(doc, "B2_GRIP_TUNE")[3]
    hands = world.arm[2:]
    analytic, raw = [], []
    for _t, q in flight.leg_samples(leg):
        world.set_arm(q)
        analytic.append(world.board_clearance())
        raw.append(min(mujoco.mj_geomDistance(world.model, world.data, g, world.board, 0.09, None) for g in hands))
    v, geom = min(analytic)
    assert geom == "r_forearm_col" and 0.0015 <= v <= 0.0025 and analytic[-1] == (v, geom)
    assert min(raw) < 0.001                         # the artefact: a hand box "touching" the slab
    assert min(a[0] for a, r in zip(analytic, raw) if r < 0.001) >= 0.0015
    world.set_arm(pose(doc, "PICK", -29.0))
    v, geom = world.board_clearance()
    assert geom == "r_palm_col" and 0.0215 <= v <= 0.0225


def test_exact_box_distance_where_mj_geomdistance_reads_zero(world, doc):
    """A finger box 2.93 cm from a cube on r3c2, mid B2 descent: mj_geomDistance reads 0 at distmax 0.03."""
    leg = R.expand(doc, "B2_GRIP_TUNE")[3]
    world.set_arm(dict(flight.leg_samples(leg)[182][1]))
    xy = world.cell_point("r3c2", (-0.0254, -0.0127))
    v, geom = world.piece_distance_at("pool_box_1", xy)
    assert geom == "r_finger_col" and v == pytest.approx(0.02934, abs=2e-4)
    cube = world.body_geoms("pool_box_1")[0]
    d = world.data
    d.geom_xpos[cube] = (xy[0], xy[1], world.board_top + 0.0254)
    d.geom_xmat[cube] = np.eye(3).ravel()
    assert mujoco.mj_geomDistance(world.model, d, world.arm[-1], cube, 0.03, None) == 0.0


def _rotation(rng):
    q = rng.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _surface(box, n=21):
    lin = np.linspace(-1, 1, n)
    u, v = (a.ravel() for a in np.meshgrid(lin, lin))
    faces = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            p = np.zeros((n * n, 3))
            p[:, axis] = sign
            others = [i for i in range(3) if i != axis]
            p[:, others[0]], p[:, others[1]] = u, v
            faces.append(p)
    return box.c + (np.vstack(faces) * box.h) @ box.r.T


def test_exact_distances_against_brute_force():
    rng = np.random.default_rng(7)
    for _ in range(60):
        a = G.Box(rng.normal(size=3) * 0.05, _rotation(rng), rng.uniform(0.005, 0.04, 3))
        b = G.Box(rng.normal(size=3) * 0.05, _rotation(rng), rng.uniform(0.005, 0.04, 3))
        v = G.box_box(a, b)
        if v > 0:
            brute = min(G.point_box(_surface(a, 61), b).min(), G.point_box(_surface(b, 61), a).min())
            assert v <= brute + 1e-9 and brute - v < 5e-4     # the brute force samples a 61 x 61 grid per face
        else:
            assert G.point_box(_surface(a, 41), b).min() < 2e-3 or G.point_box(_surface(b, 41), a).min() < 2e-3
        p = rng.normal(size=3) * 0.08
        q = p + rng.normal(size=3) * 0.05
        c = G.capsule_box(p, q, 0.02, a)
        if c > 0:
            seg = p + np.linspace(0, 1, 4001)[:, None] * (q - p)
            assert c == pytest.approx(G.point_box(seg, a).min() - 0.02, abs=1e-5)


# ---- the verdict's reasons ------------------------------------------------------------------------------------

def _synthetic(doc, **poses):
    out = copy.deepcopy(doc)
    for name, joints in poses.items():
        out["poses"][name] = {"joints_deg": joints, "provenance": out["poses"]["B2_PLACE"]["provenance"]}
    return out


def test_synthetic_route_outside_the_limits_fails_with_its_reason(doc, tmp_path):
    wide = dict(doc["poses"]["B2_SAFE"]["joints_deg"], r_wrist_roll=60.8)
    bad = _synthetic(doc, B2_WIDE=wide)
    bad["routes"]["WIDE"] = {"purpose": "gripper_calibration", "start": "TRANSPORT", "end": "TRANSPORT",
                             "legs": [{"to": "B2_WIDE", "gripper_deg": 20.0, "min_duration_s": 3.0}],
                             "close_pose": "B2_WIDE", "return": "reverse"}
    path = tmp_path / "routes.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    rec = record.run(path, "WIDE", SCENE, physics=False, cells=[])
    assert rec["verdict"] == "fail"
    assert "waypoint B2_WIDE: r_wrist_roll +60.80 deg is outside the limits [-44.97, 44.97]" in rec["reasons"]


def test_synthetic_leg_through_the_board_fails_with_its_reason(doc, tmp_path):
    sink = dict(doc["poses"]["B2_PLACE"]["joints_deg"])
    sink["r_elbow_pitch"] += 15.0          # the fingers 7.4 cm into the board
    bad = _synthetic(doc, SINK=sink)
    bad["routes"]["SINK_ROUTE"] = {"purpose": "gripper_calibration", "start": "B2_SAFE", "end": "B2_SAFE",
                                   "legs": [{"to": "SINK", "gripper_deg": -29.0, "min_duration_s": 3.0}],
                                   "close_pose": "SINK", "return": "reverse"}
    path = tmp_path / "routes.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    rec = record.run(path, "SINK_ROUTE", SCENE, physics=False, cells=[])
    assert rec["verdict"] == "fail"
    board = [r for r in rec["reasons"] if r.startswith("leg B2_SAFE->SINK: board -")]
    assert board and "needs >= 1 cm" in board[0]
    assert rec["legs"][0]["kinematic"]["board_cm"] < -7.0 and rec["legs"][0]["kinematic"]["board_geom"] == "r_finger_col"


def test_judge_kinematic_relaxes_only_at_near_board_and_skips_declared_zones(doc):
    leg = R.expand(doc, "B2_GRIP_TUNE")[3]                    # B2_SAFE -> B2_PLACE
    worst = {"rig": (0.09, None, 0), "left_arm": (0.09, None, 0)}
    board = [(0.0, 0.05), (4.0, 0.008), (5.0, 0.002)]
    assert E.judge_kinematic(leg, worst, board, {}) == [
        "leg B2_SAFE->B2_PLACE: board +0.2 cm at t=5.00 s (needs >= 1 cm)"]
    assert E.judge_kinematic(leg, worst, board, {"B2_PLACE": ("near_board", (4.0, 5.0))}) == []
    assert E.judge_kinematic(leg, worst, [(0.0, 0.05), (5.0, -0.004)], {"B2_PLACE": ("near_board", (5.0, 5.0))})
    assert E.judge_kinematic(leg, worst, [(0.0, 0.05), (5.0, -0.004)], {"B2_PLACE": ("declared", (5.0, 5.0))}) == []


# ---- conditional declarations ---------------------------------------------------------------------------------

DECLARED = {("REST", "r_forearm_col/table_top"): None, ("REST", "r_palm_col/table_top"): 1.0,
            ("REST", "r_hand_back_col/table_top"): 1.0}


def _rest_exit(doc):
    return R.expand(doc, "B2_GRIP_TUNE")[0]                   # REST -> REST_OUT, 4 s


def test_palm_contact_at_two_degrees_from_rest_fails(doc):
    leg = _rest_exit(doc)
    zones = {"REST": ("declared", (0.0, 1.1))}
    contacts = {"r_palm_col/table_top": [0.20, 0.40, 100, 2.0, 38.0]}
    (c,) = E.classify_contacts(leg, contacts, zones, DECLARED)
    assert c["declared"] is False and c["at"] == "REST" and c["deviation_deg"] == {"REST": 2.0, "REST_OUT": 38.0}
    _result, reasons = E.verdict([leg], [{"kinematic_reasons": [], "physics": {
        "follow_deg": 1.0, "arrival_deg": 0.5, "contacts": [c]}}], [])
    assert "leg REST->REST_OUT: undeclared contact r_palm_col/table_top at 0.20-0.40 s (at REST)" in reasons


def test_palm_contact_within_one_degree_of_rest_is_declared_on_the_way_out_and_back(doc):
    out_leg = _rest_exit(doc)
    back_leg = R.expand(doc, "B2_GRIP_TUNE")[-1]               # REST_OUT -> REST
    (c,) = E.classify_contacts(out_leg, {"r_palm_col/table_top": [0.18, 0.40, 107, 0.15, 39.9]}, {}, DECLARED)
    assert c["declared"] is True and c["at"] == "REST"
    (c,) = E.classify_contacts(back_leg, {"r_palm_col/table_top": [4.2, 5.0, 300, 39.9, 0.3]}, {}, DECLARED)
    assert c["declared"] is True and c["at"] == "REST"
    # the same contact on a leg that neither starts nor ends at REST is not covered
    other = R.expand(doc, "B2_GRIP_TUNE")[1]
    (c,) = E.classify_contacts(other, {"r_palm_col/table_top": [1.0, 1.2, 10, 0.5, 0.5]}, {}, DECLARED)
    assert c["declared"] is False


def test_forearm_declaration_stays_unconditional_but_windowed(doc):
    leg = _rest_exit(doc)
    zones = {"REST": ("declared", (0.0, 1.1))}
    (inside,) = E.classify_contacts(leg, {"r_forearm_col/table_top": [0.0, 0.6, 300, 5.0, 35.0]}, zones, DECLARED)
    (outside,) = E.classify_contacts(leg, {"r_forearm_col/table_top": [2.0, 2.5, 30, 20.0, 20.0]}, zones, DECLARED)
    assert inside["declared"] is True and outside["declared"] is False


def test_settle_contacts_are_judged_with_the_same_declarations():
    contacts = {"r_palm_col/table_top": [0.0, 0.06, 31, 0.03, 0.03], "r_thumb_col/table_top": [0.0, 0.1, 50, 0.0, 0.0]}
    rows = E.classify_settle("REST", contacts, DECLARED)
    assert [(r["pair"], r["declared"]) for r in rows] == [("r_palm_col/table_top", True),
                                                          ("r_thumb_col/table_top", False)]
    _result, reasons = E.verdict([], [], [], settle=rows)
    assert reasons == ["settle at REST: undeclared contact r_thumb_col/table_top at 0.00-0.10 s"]


# ---- occupancy and the target square ------------------------------------------------------------------------

def test_occupancy_rule_on_a_toy_case():
    assert E.occupancy_status(0.025, 0.0) == "may_be_occupied"
    assert E.occupancy_status(0.02, 0.99) == "may_be_occupied"
    assert E.occupancy_status(0.0199, 0.0) == "must_be_empty"        # too close kinematically
    assert E.occupancy_status(0.05, 1.0) == "must_be_empty"          # moved 1 mm in physics
    rows = [{"clear_m": 0.05, "moved_mm": 0.0}, {"clear_m": 0.012, "moved_mm": 0.0}]
    assert E.cell_status(rows) == "must_be_empty"                    # one lab piece fails: the cell is empty
    assert E.cell_status(rows[:1]) == "may_be_occupied"
    grid = E.offsets(0.0635, 0.0254)
    assert len(grid) == 49 and (0.0, 0.0) in grid
    assert max(abs(x) for x, _ in grid) == pytest.approx(0.0381)     # the cube's footprint stays inside the cell


def test_target_placement_tolerance_and_reason():
    assert E.placement_tolerance([(0, 0.1), (1, 0.2), (2, 0.9), (3, 1.4)]) == 2
    assert E.placement_tolerance([(0, 0.0), (1, 1.0), (2, 0.0)]) == 0          # contiguous from the centre
    assert E.placement_tolerance([(0, 2.46)]) is None
    pieces = {"cube": {"tolerance_mm": None, "steps": [{"offset_mm": 0, "moved_mm": 2.46}]},
              "cylinder": {"tolerance_mm": 4, "steps": []}}
    assert E.target_reasons("r2c2", "B2_PLACE", pieces) == [
        "target r2c2: a centred cube moved 2.5 mm on the approach to B2_PLACE, before the pads close"]


def test_close_cell_and_approach_for_b2_grip_tune(world, doc):
    legs = R.expand(doc, "B2_GRIP_TUNE")
    assert E.close_cell(world, legs, "B2_PLACE") == "r2c2"
    assert [(leg.start, leg.to) for leg in E.approach_legs(legs, "B2_PLACE")][-1] == ("B2_SAFE", "B2_PLACE")
    u = E.approach_direction(world, legs, "B2_PLACE")
    assert math.hypot(*u) == pytest.approx(1.0) and u[1] > 0.9      # the hand comes in from +y (column 1's side)


def pick_route(doc):
    """The doc with PICK_GRIP_TUNE added: REST -> REST_OUT -> PICK_SAFE -> PICK, closing at PICK (off every cell)."""
    out = copy.deepcopy(doc)
    out["routes"]["PICK_GRIP_TUNE"] = {
        "purpose": "gripper_calibration", "start": "REST", "end": "REST", "close_pose": "PICK", "return": "reverse",
        "legs": [{"to": "REST_OUT", "gripper_deg": 20.0, "min_duration_s": 4.0},
                 {"to": "PICK_SAFE", "gripper_deg": -29.0, "min_duration_s": 6.0},
                 {"to": "PICK", "gripper_deg": -29.0, "min_duration_s": 5.0, "near_board": True}]}
    return out


def test_close_spot_is_the_pick_tip_beside_r2c3(world, doc):
    legs = R.expand(pick_route(doc), "PICK_GRIP_TUNE")
    assert E.close_cell(world, legs, "PICK") is None                       # the supply spot is off the grid
    x, y = E.close_spot(world, legs, "PICK")
    assert (x, y) == pytest.approx((0.4811, -0.2520), abs=1e-4)
    cx, cy, half = world.cells["r2c3"]
    assert abs(y - cy) > half and abs(x - cx) < half                       # beside r2c3, outside its edge
    assert math.hypot(*E.approach_direction(world, legs, "PICK")) == pytest.approx(1.0)


def test_target_spot_option_refuses_a_close_pose_over_a_cell(doc, tmp_path):
    with pytest.raises(R.RouteError, match="B2_PLACE tip is over r2c2"):
        record.run(ROUTES, "B2_GRIP_TUNE", SCENE, physics=False, cells=["r3c3"], target_spot=True)


def test_target_spot_flies_pieces_from_the_spot_in_1mm_steps(world, doc, monkeypatch):
    """_target with a spot: pieces start at the spot, step toward the approach, tolerance from the steps. The flight is
    replaced by a stand-in that disturbs a piece once it is 3 mm or more from the spot."""
    legs = R.expand(pick_route(doc), "PICK_GRIP_TUNE")
    spot = E.close_spot(world, legs, "PICK")
    seen = []

    def fake_fly(_world, _legs, layout):
        (body, xy), = layout.items()
        off = math.hypot(xy[0] - spot[0], xy[1] - spot[1]) * 1000
        seen.append(off)
        return {"moved_mm": {body: 1.5 if off >= 2.99 else 0.0}, "legs": [{"contacts": {}}] * len(_legs)}

    monkeypatch.setattr(record.flight, "fly", fake_fly)
    u, pieces = record._target(world, legs, "PICK", None, lambda *_: None, spot_xy=spot)
    assert seen[0] == pytest.approx(0.0, abs=1e-6)
    assert {k: v["tolerance_mm"] for k, v in pieces.items()} == {"cube": 2, "cylinder": 2}
    assert [s["offset_mm"] for s in pieces["cube"]["steps"]] == [0, 1, 2, 3]


# ---- the record -------------------------------------------------------------------------------------------------

def test_record_is_deterministic_and_named_by_route_and_hash(doc, tmp_path):
    """A short route flown in physics twice: byte-identical records."""
    short = copy.deepcopy(doc)
    short["routes"]["SHORT"] = {"purpose": "gripper_calibration", "start": "REST", "end": "REST",
                                "legs": [{"to": "REST_OUT", "gripper_deg": 20.0, "min_duration_s": 4.0}],
                                "close_pose": "REST_OUT", "return": "reverse"}
    path = tmp_path / "routes.json"
    path.write_text(json.dumps(short), encoding="utf-8")
    declared = [("REST", "r_forearm_col/table_top", None), ("REST", "r_palm_col/table_top", 1.0),
                ("REST", "r_hand_back_col/table_top", 1.0)]
    first = record.run(path, "SHORT", SCENE, cells=["r3c3"], declared=declared)
    second = record.run(path, "SHORT", SCENE, cells=["r3c3"], declared=declared)
    assert record.dumps(first) == record.dumps(second)
    assert record.file_name(first) == "SHORT@%s.json" % R.hash_of(short, "SHORT")[:12]
    assert first["schema"] == "arm_route_record.v1" and first["target_cell"] is None   # REST_OUT is off the board
    assert first["declared_contacts"][2] == {"pose": "REST", "pair": "r_palm_col/table_top",
                                             "condition": {"max_joint_deviation_deg": 1.0}}
    assert first["rig"]["rig_layout_id"] == "fwd-center-lab" and first["rig"]["mount_id"] == "october-2026"
    assert first["reasons"] == ["occupancy covers only r3c3"] and first["verdict"] == "fail"
    assert first["physics_settle"]["judged"] is True
