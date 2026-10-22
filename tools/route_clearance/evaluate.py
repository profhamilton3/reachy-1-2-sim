"""Kinematic clearances, contact zones, the occupancy map and the verdict (design section 3.2).

Pure rules (`zone`, `judge_kinematic`, `classify_contacts`, `occupancy_status`, `verdict`) work on plain values so
they can be tested without the model; `kinematic_leg`, `worst_positions`, `close_cell` and `approach_direction` read
the model.
"""
import itertools

import numpy as np

from . import flight
from .route import limit_violations, near_board_poses, waypoints
from .world import CAP_M, PIECES

NEAR_BOARD_M = 0.01       # the 1 cm board rule; also the edge of a waypoint's contact zone
FOLLOW_MAX_DEG = 6.0
ARRIVAL_MAX_DEG = 3.0
LAG_S = 0.25              # a contact may trail a zone by the following window (design section 4.3: 6 deg for 0.25 s)
OCCUPANCY_CLEAR_M = 0.02
OCCUPANCY_MOVE_MM = 1.0
OFFSETS_PER_AXIS = 7      # the worst-position search: a 7 x 7 grid of piece centres inside the cell


# ---- kinematic ---------------------------------------------------------------------------------------------------
def kinematic_leg(world, leg):
    """Worst distance per group along the leg, and the board clearance at every stream sample."""
    worst = {k: (CAP_M, None, 0.0) for k in ("board", "rig", "left_arm", "self")}
    board = []
    for t, q in flight.leg_samples(leg):
        world.set_arm(q)
        for k, (v, geom) in world.clearances().items():
            if v < worst[k][0]:
                worst[k] = (v, geom, t)
        board.append((t, world.board_clearance()[0]))
    return worst, board


def zone(board, at_end):
    """(t_start, t_end) of the run of samples within NEAR_BOARD_M of the board, from one end of the leg (at_end
    False: the start). None when that end is clear."""
    samples = board[::-1] if at_end else board
    run = list(itertools.takewhile(lambda s: s[1] < NEAR_BOARD_M, samples))
    if not run:
        return None
    times = [s[0] for s in run]
    return (min(times), max(times))


def leg_zones(leg, board, contact_poses, near_poses):
    """{pose: (kind, (t0, t1))} for the leg's ends that are declared-contact or near_board waypoints."""
    out = {}
    for pose, at_end in ((leg.start, False), (leg.to, True)):
        if pose in contact_poses or pose in near_poses:
            z = zone(board, at_end)
            if z is not None:
                out[pose] = ("declared" if pose in contact_poses else "near_board", z)
    return out


def judge_kinematic(leg, worst, board, zones):
    """Reasons the leg fails section 3.2's distance rule: board, rig and left arm >= 0, and >= 1 cm except near a
    near_board waypoint. The board is not judged inside a declared-contact zone; a zone that runs past mid-leg is
    itself a failure."""
    reasons = []
    name = "%s->%s" % (leg.start, leg.to)
    half = leg.duration_s / 2.0
    for pose, (kind, (t0, t1)) in sorted(zones.items()):
        if kind == "declared" and (t1 - t0) > half:
            reasons.append("leg %s: the board contact zone at %s runs %.2f-%.2f s, past mid-leg"
                           % (name, pose, t0, t1))

    def inside(t, kinds):
        return any(k in kinds and t0 <= t <= t1 for k, (t0, t1) in zones.values())

    worst_bad = None                         # the deepest sample below its floor
    for t, v in board:
        if inside(t, ("declared",)):
            continue
        floor = 0.0 if inside(t, ("near_board",)) else NEAR_BOARD_M
        if v < floor and (worst_bad is None or v < worst_bad[1]):
            worst_bad = (t, v, floor)
    if worst_bad is not None:
        t, v, floor = worst_bad
        reasons.append("leg %s: board %+.1f cm at t=%.2f s (needs >= %.0f cm)" % (name, v * 100, t, floor * 100))
    near_any = bool([1 for k, _ in zones.values() if k == "near_board"]) or leg.near_board
    for group in ("rig", "left_arm"):
        v = worst[group][0]
        floor = 0.0 if near_any else NEAR_BOARD_M
        if v < floor:
            reasons.append("leg %s: %s %+.1f cm (needs >= %.0f cm)" % (name, group, v * 100, floor * 100))
    return reasons


# ---- physics contacts ---------------------------------------------------------------------------------------------
def classify_contacts(leg, contacts, zones, declared):
    """[{pair, t_s, steps, at, declared, deviation_deg}] for one leg.

    declared: {(pose, pair): None | max_joint_deviation_deg}. An unconditional declaration (None) covers a contact
    whose every step falls in the pose's window: its zone on this leg (widened by LAG_S), or the hold after the leg
    at its `to` pose. A conditional one covers a contact only while the arm's largest joint deviation from the pose
    stays within the bound, on any leg that starts or ends there; outside it the contact is undeclared."""
    def window(pose):
        lo, hi = float("inf"), float("-inf")
        if pose in zones:
            t0, t1 = zones[pose][1]
            lo, hi = t0 - LAG_S, t1 + LAG_S
        if pose == leg.to and (pose not in zones or hi >= leg.duration_s):
            lo, hi = min(lo, leg.duration_s), float("inf")
        return lo, hi

    out = []
    for pair, (t_first, t_last, steps, dev_start, dev_to) in sorted(contacts.items()):
        devs = {leg.start: dev_start, leg.to: dev_to}
        at, ok = None, False
        for pose in (leg.start, leg.to):
            lo, hi = window(pose)
            in_window = lo <= t_first and t_last <= hi
            if (pose, pair) in declared:
                bound = declared[(pose, pair)]
                if (in_window if bound is None else devs[pose] <= bound):
                    at, ok = pose, True
                    break
            if at is None and in_window:
                at = pose
        out.append({"pair": pair, "t_s": [round(t_first, 3), round(t_last, 3)], "steps": steps, "at": at,
                    "declared": ok, "deviation_deg": {p: round(v, 2) + 0.0 for p, v in sorted(devs.items())}})
    return out


def classify_settle(pose, contacts, declared):
    """The settle's contacts, all at the start pose: declared as on a leg, with the whole settle as the window."""
    out = []
    for pair, (t_first, t_last, steps, dev, _dev) in sorted(contacts.items()):
        ok = (pose, pair) in declared and (declared[(pose, pair)] is None or dev <= declared[(pose, pair)])
        out.append({"pair": pair, "t_s": [round(t_first, 3), round(t_last, 3)], "steps": steps, "at": pose,
                    "declared": ok, "deviation_deg": {pose: round(dev, 2) + 0.0}})
    return out


# ---- the target square -----------------------------------------------------------------------------------------
def approach_legs(legs, close_pose):
    """The outbound legs up to the first arrival at the close pose."""
    for i, leg in enumerate(legs):
        if leg.direction == "out" and leg.to == close_pose:
            return legs[:i + 1]
    raise ValueError("no outbound leg reaches the close pose %s" % close_pose)


def close_cell(world, legs, close_pose):
    """The cell under the close pose's tip, or None when the tip is not over a cell."""
    world.set_arm(approach_legs(legs, close_pose)[-1].q_to)
    tip = world.tip_world()
    for cell, (x, y, half) in sorted(world.cells.items()):
        if abs(tip[0] - x) <= half and abs(tip[1] - y) <= half:
            return cell
    return None


def close_spot(world, legs, close_pose):
    """World xy (m) of the close pose's tip: the supply spot a piece is held at when the pose is over no cell."""
    world.set_arm(approach_legs(legs, close_pose)[-1].q_to)
    return tuple(float(v) for v in world.tip_world()[:2])


def approach_direction(world, legs, close_pose):
    """Unit xy vector from the close pose's tip toward the tip of the pose the last approach leg starts from: the
    side the hand comes in from."""
    last = approach_legs(legs, close_pose)[-1]
    world.set_arm(last.q_to)
    to = world.tip_world()[:2]
    world.set_arm(last.q_from)
    v = world.tip_world()[:2] - to
    return v / np.linalg.norm(v)


#: Half-width (m) of the square a supply-spot target may wander over, the same as a grid cell's.
SPOT_HALF_M = 0.0635


def placement_tolerance(steps):
    """steps: [(offset_mm, moved_mm)] from 0 in 1 mm steps. The largest offset up to which every step moved the
    piece < OCCUPANCY_MOVE_MM; None when the centred piece already moved."""
    best = None
    for offset, moved in steps:
        if moved >= OCCUPANCY_MOVE_MM:
            break
        best = offset
    return best


def target_reasons(cell, close_pose, pieces):
    """pieces: {kind: {"tolerance_mm", "steps": [{offset_mm, moved_mm}]}}."""
    out = []
    for kind, row in sorted(pieces.items()):
        if row["tolerance_mm"] is None:
            out.append("target %s: a centred %s moved %.1f mm on the approach to %s, before the pads close"
                       % (cell, kind, row["steps"][0]["moved_mm"], close_pose))
    return out


# ---- occupancy -------------------------------------------------------------------------------------------------
def offsets(cell_half, piece_half, n=OFFSETS_PER_AXIS):
    """Piece-centre offsets (m) that keep the piece's footprint inside the cell, on an n x n grid."""
    reach = max(0.0, cell_half - piece_half)
    axis = np.linspace(-reach, reach, n) if n > 1 else np.array([0.0])
    return [(float(x), float(y)) for x in axis for y in axis]


def worst_positions(world, legs, cells=None):
    """{(cell, kind): (metres, offset_xy, arm geom)}: for each cell and lab piece, the position in the cell that
    comes closest to the arm anywhere along the route (kinematic), ties to the offset nearest the centre."""
    cells = sorted(world.cells) if cells is None else list(cells)
    candidates = []
    for cell in cells:
        for kind in sorted(PIECES):
            body = PIECES[kind][0]
            for off in offsets(world.cells[cell][2], world.piece_half_footprint(body)):
                candidates.append((cell, kind, body, off))
    if not candidates:
        return {}
    centres = np.array([world.cell_point(c, off) + (world.board_top + world.piece_half_height(b),)
                        for c, _k, b, off in candidates])
    radius = np.array([float(world.model.geom_rbound[world.body_geoms(b)[0]]) for _c, _k, b, _o in candidates])
    best = [(CAP_M, None)] * len(candidates)
    for leg in legs:
        if leg.direction == "return":       # the same path as its outbound leg, backwards
            continue
        for _t, q in flight.leg_samples(leg):
            world.set_arm(q)
            gap = np.full(len(candidates), np.inf)
            for pos, r in world.arm_points():
                gap = np.minimum(gap, np.linalg.norm(centres - pos, axis=1) - r - radius)
            for i in np.nonzero(gap < CAP_M)[0]:
                cell, kind, body, off = candidates[i]
                v, geom = world.piece_distance_at(body, world.cell_point(cell, off))
                if v < best[i][0]:
                    best[i] = (v, geom)
    out = {}
    for i, (cell, kind, _body, off) in enumerate(candidates):
        v, geom = best[i]
        key = (cell, kind)
        rank = (round(v, 6), round(off[0] ** 2 + off[1] ** 2, 9), off)
        if key not in out or rank < out[key][0]:
            out[key] = (rank, v, off, geom)
    return {key: (v, off, geom) for key, (_rank, v, off, geom) in out.items()}


def occupancy_status(clear_m, moved_mm):
    """Section 3.2: may_be_occupied only if the piece at its worst position stays >= 2 cm from the arm and moves
    < 1 mm in physics."""
    if clear_m >= OCCUPANCY_CLEAR_M and moved_mm < OCCUPANCY_MOVE_MM:
        return "may_be_occupied"
    return "must_be_empty"


def cell_status(rows):
    """A cell is may_be_occupied only if every lab piece there is."""
    return "may_be_occupied" if all(occupancy_status(r["clear_m"], r["moved_mm"]) == "may_be_occupied"
                                    for r in rows) else "must_be_empty"


# ---- verdict ---------------------------------------------------------------------------------------------------
def verdict(legs, leg_reports, layouts, extra_reasons=(), settle=()):
    """("pass" | "fail", reasons). leg_reports: per leg {"kinematic_reasons", "physics"}; layouts: the flown
    layouts; a judged one (every may_be_occupied cell at once) must not touch or move a piece. settle: the classified
    contacts of the settle at the start pose."""
    reasons = []
    for c in settle:
        if not c["declared"]:
            reasons.append("settle at %s: undeclared contact %s at %.2f-%.2f s" % (c["at"], c["pair"], c["t_s"][0],
                                                                                 c["t_s"][1]))
    for name, q in waypoints(legs):
        reasons += limit_violations(name, q)
    for leg, rep in zip(legs, leg_reports):
        reasons += rep["kinematic_reasons"]
        phys = rep.get("physics")
        if phys is None:
            reasons.append("leg %s->%s: not flown in physics" % (leg.start, leg.to))
            continue
        if phys["follow_deg"] > FOLLOW_MAX_DEG:
            reasons.append("leg %s->%s: following %.1f deg (max %.0f)" % (leg.start, leg.to, phys["follow_deg"],
                                                                         FOLLOW_MAX_DEG))
        if phys["arrival_deg"] > ARRIVAL_MAX_DEG:
            reasons.append("leg %s->%s: arrival %.1f deg (max %.0f)" % (leg.start, leg.to, phys["arrival_deg"],
                                                                       ARRIVAL_MAX_DEG))
        for c in phys["contacts"]:
            if not c["declared"]:
                reasons.append("leg %s->%s: undeclared contact %s at %.2f-%.2f s%s" % (
                    leg.start, leg.to, c["pair"], c["t_s"][0], c["t_s"][1],
                    " (at %s)" % c["at"] if c["at"] else ""))
    for lay in layouts:
        if not lay.get("judged", False):
            continue
        for c in lay.get("piece_contacts", []):
            reasons.append("layout %s: contact %s" % (lay["name"], c))
        for body, mm in sorted(lay.get("moved_mm", {}).items()):
            if mm >= OCCUPANCY_MOVE_MM:
                reasons.append("layout %s: %s moved %.1f mm" % (lay["name"], body, mm))
    reasons += list(extra_reasons)
    return ("pass" if not reasons else "fail"), reasons


__all__ = ["approach_direction", "approach_legs", "cell_status", "classify_contacts", "classify_settle", "close_cell",
           "judge_kinematic", "kinematic_leg", "leg_zones", "near_board_poses", "occupancy_status", "offsets",
           "placement_tolerance", "target_reasons", "verdict", "worst_positions", "zone"]
