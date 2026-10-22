"""One clearance run and its record (design section 3.2), written as `<route>@<hash12>.json`.

The record is deterministic: no clock, no host paths, sorted keys, fixed rounding. The same route document, scene,
model, tool and MuJoCo give byte-identical output.
"""
import hashlib
import json
import pathlib
import subprocess

import mujoco

from . import _REPO, evaluate as E, flight, head
from .limits import ARM_MOUNT_ID, ARM_RIG_LAYOUT_ID
from .route import MAX_AVG_SPEED_DEG_S, expand, hash_of, near_board_poses, waypoints
from .world import CAP_M, MODEL_PATH, PIECES, World

SCHEMA = "arm_route_record.v1"
#: Design section 3.2: the contacts a route is allowed, as (waypoint, "arm geom/other geom", condition). A condition
#: of None is unconditional; a number allows the contact only while the arm's largest joint deviation from the
#: waypoint is within that many degrees (owner decision 2026-10-22: the hand grazes the board only while still at REST).
DECLARED_CONTACTS = {
    "B2_GRIP_TUNE": (("REST", "r_forearm_col/table_top", None), ("B2_PLACE", "r_forearm_col/table_top", None),
                     ("REST", "r_palm_col/table_top", 1.0), ("REST", "r_hand_back_col/table_top", 1.0)),
}
TOOL_DIR = pathlib.Path(__file__).resolve().parent
_INPUT_PATHS = ("tools/route_clearance", "native_mujoco", "scenes", "src")


def _sha256_file(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def tool_sha256():
    """SHA-256 over this tool's source files (sorted by name, each as name NUL bytes NUL)."""
    digest = hashlib.sha256()
    for path in sorted(TOOL_DIR.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def sim_commit():
    """HEAD of this repo, with "-dirty" when a file that can change the result differs from it."""
    try:
        head_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO, capture_output=True, text=True,
                                  check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", *_INPUT_PATHS],
                               cwd=_REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head_sha + ("-dirty" if dirty else "")


def _r(x, nd=2):
    return round(float(x), nd) + 0.0


def _cm(x):
    return _r(x * 100)


def _piece_contacts(legs, flown, bodies):
    """The contacts with the layout's pieces (the arm's own contacts are judged on the empty-board flight)."""
    out = []
    for leg, row in zip(legs, flown["legs"]):
        for c in E.classify_contacts(leg, row["contacts"], {}, {}):
            if set(c["pair"].split("/")) & set(bodies):
                out.append("leg %s->%s: %s at %.2f-%.2f s" % (leg.start, leg.to, c["pair"], c["t_s"][0],
                                                              c["t_s"][1]))
    return out


def _target(world, legs, close_pose, cell, log):
    """The target piece at its intended placement on the close cell: centred, then 1 mm steps toward the approach,
    flown to the close pose and its hold (before the pads close), until a step moves it >= 1 mm."""
    approach = E.approach_legs(legs, close_pose)
    u = E.approach_direction(world, legs, close_pose)
    pieces = {}
    for kind in sorted(PIECES):
        body = PIECES[kind][0]
        reach_mm = int((world.cells[cell][2] - world.piece_half_footprint(body)) * 1000 + 1e-9)
        steps, contacts = [], []
        for k in range(reach_mm + 1):
            centre = world.cell_point(cell)
            xy = (centre[0] + u[0] * k / 1000.0, centre[1] + u[1] * k / 1000.0)
            log("physics: target %s %s at %d mm" % (cell, kind, k))
            flown = flight.fly(world, approach, {body: xy})
            moved = flown["moved_mm"][body]
            steps.append({"offset_mm": k, "moved_mm": _r(moved)})
            if k == 0:
                contacts = _piece_contacts(approach, flown, [body])
            if moved >= E.OCCUPANCY_MOVE_MM:
                break
        pieces[kind] = {"body": body, "steps": steps, "centred_piece_contacts": contacts,
                        "tolerance_mm": E.placement_tolerance([(s["offset_mm"], s["moved_mm"]) for s in steps])}
    return [_r(u[0], 4), _r(u[1], 4)], pieces


def run(routes_path, route_name, scene_name, physics=True, cells=None, declared=None, log=lambda *_: None):
    """The record as a dict. declared: [(pose, pair, None | max_joint_deviation_deg)], default the route's
    DECLARED_CONTACTS."""
    doc_text = pathlib.Path(routes_path).read_bytes()
    doc = json.loads(doc_text)
    legs = expand(doc, route_name)
    close_pose = doc["routes"][route_name]["close_pose"]
    digest = hash_of(doc, route_name)
    declared = sorted(DECLARED_CONTACTS.get(route_name, ()) if declared is None else declared,
                      key=lambda c: (c[0], c[1]))
    bounds = {(pose, pair): bound for pose, pair, bound in declared}
    contact_poses = {pose for pose, _pair, _bound in declared}
    near = set(near_board_poses(legs))
    scene_path = _REPO / "scenes" / (scene_name + ".yaml")
    world = World(scene_path)
    target_cell = E.close_cell(world, legs, close_pose)
    extra = []

    log("kinematic")
    leg_reports, zones_by_leg, leg_rows = [], [], []
    for leg in legs:
        worst, board = E.kinematic_leg(world, leg)
        zones = E.leg_zones(leg, board, contact_poses, near)
        zones_by_leg.append(zones)
        leg_reports.append({"kinematic_reasons": E.judge_kinematic(leg, worst, board, zones)})
        leg_rows.append({
            "from": leg.start, "to": leg.to, "direction": leg.direction, "duration_s": _r(leg.duration_s),
            "gripper_deg": _r(leg.q_to["r_gripper"]),
            "kinematic": {
                "board_cm": _cm(worst["board"][0]), "board_geom": worst["board"][1],
                "board_t_s": _r(worst["board"][2]),
                "rig_cm": _cm(worst["rig"][0]), "left_arm_cm": _cm(worst["left_arm"][0]),
                "self_cm": _cm(worst["self"][0]),
                "zones": {pose: {"kind": kind, "t_s": [_r(t0), _r(t1)]} for pose, (kind, (t0, t1))
                          in sorted(zones.items())},
            },
        })

    layouts, settle = [], None
    if physics:
        log("physics: empty board")
        flown = flight.fly(world, legs)
        for leg, rep, row, zones, flown_leg in zip(legs, leg_reports, leg_rows, zones_by_leg, flown["legs"]):
            contacts = E.classify_contacts(leg, flown_leg["contacts"], zones, bounds)
            rep["physics"] = {"follow_deg": flown_leg["follow_deg"], "arrival_deg": flown_leg["arrival_deg"],
                              "contacts": contacts}
            row["physics"] = {"follow_deg": _r(flown_leg["follow_deg"]), "arrival_deg": _r(flown_leg["arrival_deg"]),
                              "arrival_weak_deg": _r(flown_leg["arrival_weak_deg"]), "contacts": contacts}
        settle = {"at": legs[0].start, "judged": True,
                  "contacts": E.classify_settle(legs[0].start, flown["settle"], bounds)}
        layouts.append({"name": "empty board", "pieces": {}, "moved_mm": {}, "judged": False,
                        "piece_contacts": []})
    else:
        for row in leg_rows:
            row["physics"] = None

    log("occupancy: worst positions")
    map_cells = [c for c in (sorted(world.cells) if cells is None else cells) if c != target_cell]
    worst_pos = E.worst_positions(world, legs, map_cells)
    occupancy, per_cell = {}, {}
    for (cell, kind), (clear, off, geom) in sorted(worst_pos.items()):
        body = PIECES[kind][0]
        xy = world.cell_point(cell, off)
        entry = {"name": "%s %s" % (cell, kind), "cell": cell, "piece": kind, "pieces": {body: cell},
                 "offset_mm": [_r(off[0] * 1000, 1), _r(off[1] * 1000, 1)], "clear_cm": _cm(clear),
                 "clear_geom": geom, "judged": False}
        moved = float("inf")
        if physics:
            log("physics: %s" % entry["name"])
            flown = flight.fly(world, legs, {body: xy})
            moved = flown["moved_mm"][body]
            entry["moved_mm"] = {body: _r(moved)}
            entry["piece_contacts"] = _piece_contacts(legs, flown, [body])
        entry["status"] = E.occupancy_status(clear, moved)
        per_cell.setdefault(cell, []).append({"clear_m": clear, "moved_mm": moved, "kind": kind, "xy": xy})
        layouts.append(entry)
    for cell, rows in sorted(per_cell.items()):
        occupancy[cell] = E.cell_status(rows)

    may = [c for c in sorted(occupancy) if occupancy[c] == "may_be_occupied"]
    if physics and may:
        log("physics: every may_be_occupied cell at once")
        free = {kind: list(bodies) for kind, bodies in PIECES.items()}
        placed, where = {}, {}
        for cell in may:
            for row in sorted(per_cell[cell], key=lambda r: (r["clear_m"], r["kind"])):
                if free[row["kind"]]:
                    body = free[row["kind"]].pop(0)
                    placed[body], where[body] = row["xy"], cell
                    break
        flown = flight.fly(world, legs, placed)
        layouts.append({"name": "every may_be_occupied cell", "pieces": dict(sorted(where.items())),
                        "moved_mm": {b: _r(v) for b, v in sorted(flown["moved_mm"].items())}, "judged": True,
                        "piece_contacts": _piece_contacts(legs, flown, list(placed))})

    approach_xy, target_pieces = None, {}
    if target_cell is not None and physics:
        approach_xy, target_pieces = _target(world, legs, close_pose, target_cell, log)
        extra += E.target_reasons(target_cell, close_pose, target_pieces)
    elif target_cell is not None:
        extra.append("target %s: placement not flown in physics" % target_cell)

    log("head")
    head_rows = {}
    for name, q in waypoints(legs):
        a = head.aim(world, q)
        head_rows[name] = {
            "neck_deg": {k: _r(v) for k, v in a["neck_deg"].items()}, "at_limit": a["at_limit"],
            "off_axis_deg": _r(a["off_axis_deg"]),
            "tip_in_view_left": bool(a["tip_in_view_left"]), "tip_in_view_right": bool(a["tip_in_view_right"]),
            "tip_px_left": [_r(v, 1) for v in a["tip_px_left"]] if a["tip_px_left"] else None,
            "tip_px_right": [_r(v, 1) for v in a["tip_px_right"]] if a["tip_px_right"] else None,
        }

    if not physics:
        extra.append("occupancy not flown in physics")
    if cells is not None and sorted(set(cells) | {target_cell} - {None}) != sorted(world.cells):
        extra.append("occupancy covers only %s" % (", ".join(sorted(map_cells)) or "no cells"))
    result, reasons = E.verdict(legs, leg_reports, layouts, extra, settle["contacts"] if settle else ())
    return {
        "schema": SCHEMA,
        "route": route_name,
        "route_hash": digest,
        "verdict": result,
        "reasons": reasons,
        "rig": {"rig_layout_id": ARM_RIG_LAYOUT_ID, "mount_id": ARM_MOUNT_ID, "scene_name": scene_name,
                "scene_sha256": _sha256_file(scene_path), "scene_revision": world.core.scene_revision},
        "sim": {"commit": sim_commit(), "model_sha256": _sha256_file(MODEL_PATH),
                "mujoco_version": mujoco.__version__, "tool_sha256": tool_sha256()},
        "routes_sha256": hashlib.sha256(doc_text).hexdigest(),
        "legs": leg_rows,
        "physics_settle": settle,
        "declared_contacts": [{"pose": pose, "pair": pair,
                               "condition": None if bound is None else {"max_joint_deviation_deg": _r(bound)}}
                              for pose, pair, bound in declared],
        "occupancy": occupancy,
        "target_cell": target_cell,
        "target_placement_tolerance_mm": {kind: row["tolerance_mm"] for kind, row in sorted(target_pieces.items())},
        "target_placement": {"close_pose": close_pose, "approach_xy": approach_xy,
                             "pieces": dict(sorted(target_pieces.items()))},
        "layouts_flown": layouts,
        "head": head_rows,
        "method": {
            "board_distance": "analytic: lowest box corner / capsule end over the board top (Step 1 board_check)",
            "piece_rig_left_self_distance": "exact for box and capsule pairs (geometry.py); mj_geomDistance for "
                                            "cylinder pairs",
            "cap_cm": _cm(CAP_M), "stream_hz": flight.STREAM_HZ, "settle_s": flight.SETTLE_S,
            "hold_s": flight.HOLD_S, "max_avg_speed_deg_s": MAX_AVG_SPEED_DEG_S,
            "leg_start": "the measured position (design section 4.2)",
            "moved": "largest displacement during the flight",
            "near_board_cm": _cm(E.NEAR_BOARD_M), "contact_lag_s": E.LAG_S,
            "follow_max_deg": E.FOLLOW_MAX_DEG, "arrival_max_deg": E.ARRIVAL_MAX_DEG,
            "follow_joints": list(flight.GROSS), "arrival_joints": list(flight.GUARD),
            "occupancy_clear_cm": _cm(E.OCCUPANCY_CLEAR_M), "occupancy_move_mm": E.OCCUPANCY_MOVE_MM,
            "occupancy_offsets_per_axis": E.OFFSETS_PER_AXIS, "target_step_mm": 1,
            "head_frame_px": list(head.FRAME_PX),
        },
    }


def dumps(record):
    return json.dumps(record, sort_keys=True, indent=1, ensure_ascii=False) + "\n"


def file_name(record):
    return "%s@%s.json" % (record["route"], record["route_hash"][:12])


def write(record, out_dir):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / file_name(record)
    path.write_text(dumps(record), encoding="utf-8")
    return path
