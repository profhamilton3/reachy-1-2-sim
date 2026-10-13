"""Placement: pick a box up from one cell and set it down on another (#174).

What it is for
--------------
The crane lift (`crane_pick_live`) can only put an object back where it found
it.  A placement is the same pick, then three things the lift does not have:

  1. A CARRY between two ladders.  The object is lifted on the pickup ladder
     to the carry height, flown across to the same height on a DESTINATION
     ladder, and lowered there.  The carry is checked with the held object
     in it: the box where the grasp put it, against the table, the rails and
     every other object, and against the upper arm and forearm; plus the
     crane's whole-arm and table checks on every segment.
  2. A DESTINATION LADDER, solved like the pickup's: the hand turned about
     the vertical by the change in bearing from the shoulder (so the arm
     keeps the posture it grasped in), and placed so that the object, held
     where it is in the hand, stands on the cell's centre.  At planning time
     the object's place in the hand is PREDICTED (`GRASP_SHIFT_M`); once it
     is held, the ladder is solved again from where it MEASURABLY is, and
     re-checked, before the carry starts.
  3. An ABORT PATH.  Every move after the hold is recorded; a halt during the
     carry or the set-down flies them back in reverse -- every one of them
     was checked -- to the pickup ladder, and the object is set down where it
     was picked up, by the crane's own replacement.  It is never released in
     the air: the hand opens only once the object is measured to rest on the
     table, and with no fresh feedback it does not open at all.

Set-down accuracy (D6: within 1 cm of the cell centre) comes from two places.
At the carry height, and again two rungs above the table, the held object's
MEASURED position is compared with the cell centre and the ladder is shifted
by the difference (bounded, re-solved and re-checked before it is commanded).
And the release is the one the #174 drag study chose: the hand backs away on
the thumb's side while it lets go (`PlaceSpec.release_retreat_m`), which took
the release drag from 9.7 mm to 2.9 mm (mean) over 20 recorded releases
offline, with #173's way out still clear in every one.  The lift itself keeps
#173's release: its put-back relies on the drag undoing the shift the close
gives the object (see the ADR).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from reachy_ai.motion import grasp_alignment as G
from reachy_ai.motion.kinematics import (ArmClearanceError, TargetContact,
                                         within_limits)
from reachy_ai.scene.awareness import SceneModel
from reachy_ai.tasks import crane_pick_live as CP

#: How far the close moves the object toward the thumb, in the hand: the
#: finger pushes it across the open gap and into the thumb pad.  Measured on
#: the 20 recorded crane closes of the drag study (Phase 1.5 and the #173
#: live check): 6.9-9.0 mm.  Used only to PREDICT the hold at planning time;
#: the carry is re-planned from the measured hold.
GRASP_SHIFT_M = 0.008

#: Ability route name (`web/panel_abilities.py`).
ROUTE = "CRANE_PLACE"

#: D6 (#174), the place's success definition: the object's centre within
#: 1 cm (horizontally) of the cell centre, tilted no more than 5 degrees.  The
#: evaluator (`evaluation.panel_routes.evaluate_place`) is the formal one; the
#: crane's own criteria use the same numbers.
D6_POSITION_M = 0.010
D6_TILT_DEG = 5.0

#: The place ability's grasp attitude (#174, step 2), used ONLY by the
#: placement: the lift keeps `CraneSpec()`.  The lift's attitude (tilt 30,
#: lean 59.52) cannot set an object down on r2c1 -- the nearest pose there is
#: 20 mm off the centre.  Searched offline (Reachy-Lab
#: issue174-exit-and-attitude-2026-10-12): every pick, carry, set-down and exit
#: check for r2c2 -> r2c1 passes on x faces for tilt 54-66 / lean 65-70, and
#: flown offline (`tools.place_sweep.fly`) the lean-65 band, tilt 60-66, placed
#: every start 3.8-5.3 mm from the centre with no pad contact after the release.
#: 63 is the middle of that band: 6.8 deg from a stop at the pick (arm yaw),
#: 0.9 deg at the set-down (shoulder roll), no turn about the vertical.  The
#: descent's prediction tolerances and the drag study were measured at tilt
#: 30: the #174 live runs are this attitude's first validation.
PLACE_GRASP = CP.CraneSpec(tilt_deg=63.0, lean_bearing_deg=65.0)


@dataclass(frozen=True)
class PlaceSpec:
    """The placement's parameters, fixed before motion."""
    #: Carry heights tried, in rungs above the grasp, lowest first: the hover
    #: (9 cm), then the via (18 cm) if the hover's carry is not clear.
    carry_rungs: Tuple[int, ...] = (9, 18)
    #: The release the drag study chose (#174): the hand backs 9 mm away along
    #: the jaw, on the thumb's side, while the grip lets go.
    release_retreat_m: float = 0.009
    #: The hand also backs out this far along its own axis while letting go.
    release_backoff_m: float = 0.0
    #: The held object's measured position is corrected toward the cell
    #: centre until it is this close, at most `setdown_corrections` times at
    #: each of the two places it is measured, and never by more than
    #: `max_setdown_correction_m` in all.
    setdown_tolerance_m: float = 0.001
    setdown_corrections: int = 3
    max_setdown_correction_m: float = 0.015
    #: The second measurement, this many rungs above the set-down.
    check_rung: int = 2
    #: The carried object's lowest corner stays at least this far above the
    #: table on the climb and across the carry.
    carry_table_clearance_m: float = 0.01
    #: A carried object that moves this far in the hand stops the carry.
    in_hand_slip_halt_m: float = 0.002
    #: Nothing else may stand closer than this to where the object will stand.
    destination_free_m: float = 0.005
    #: A "placement" shorter than this is not one (it is the lift).
    min_transfer_m: float = 0.02
    #: The whole attempt, pick to PRESENT (the lift's is 300 s).
    max_attempt_s: float = 480.0
    #: Where the set-down is AIMED, against the cell centre D6 measures from:
    #: ``aim_m`` toward the point ``aim_toward_xy`` (world, m).  0 aims at
    #: the centre.  See `PLACE_AIM` for the placement's own.
    #: #174 option 3: the set-down margin from the shoulder-roll stop is under
    #: 2 deg, and the release drag also pushes toward the robot, so aiming
    #: there costs ~0.8 mm of D6 error per mm.  1 mm toward the robot's base
    #: kept the offline D6 error <= 5.7 mm (10 starts) for ~0.2 deg of margin.
    aim_m: float = 0.001
    aim_toward_xy: Tuple[float, float] = (0.0, 0.0)


@dataclass
class PlacePlan:
    object_id: str
    target_cell: str
    target: np.ndarray                 # where the object's centre is aimed to stand
    spec: PlaceSpec
    pick: CP.CranePlan
    dest: CP.CranePlan
    carry_rung: int
    psi_deg: float
    rel: Tuple[np.ndarray, np.ndarray]  # the hold the destination was solved for
    hold_source: str = "predicted"
    checks: Dict[str, object] = field(default_factory=dict)
    #: The cell's centre: what D6 measures the placement from (the aim may
    #: differ from it by `PlaceSpec.aim_m`).
    cell_centre: Optional[np.ndarray] = None

    @property
    def centre(self) -> np.ndarray:
        return self.target if self.cell_centre is None else self.cell_centre

    @property
    def cell(self) -> str:
        return self.target_cell.replace("cell_", "")

    def climb(self) -> List[List[float]]:
        """The pickup ladder from the lift's top rung up to the carry height."""
        s = self.pick.spec
        return [self.pick.rungs[k] for k in range(s.lift_rungs, self.carry_rung + 1)]

    def carry(self) -> List[List[float]]:
        return [self.pick.rungs[self.carry_rung], self.dest.rungs[self.carry_rung]]

    def setdown(self, frm: Optional[int] = None) -> List[List[float]]:
        """Down the destination ladder to its floor (`CranePlan.floor_rung`)."""
        top = self.carry_rung if frm is None else frm
        return [self.dest.rungs[k] for k in range(top, self.dest.floor_rung - 1, -1)]

    def summary(self) -> Dict[str, object]:
        return {"object_id": self.object_id, "target_cell": self.target_cell,
                "target": [round(float(v), 5) for v in self.target],
                "cell_centre": [round(float(v), 5) for v in self.centre[:2]],
                "carry_rung": self.carry_rung, "psi_deg": round(self.psi_deg, 3),
                "hold_source": self.hold_source,
                "hold_in_thumb_mm": (1000 * self.rel[1]).round(2).tolist(),
                "pick": self.pick.summary(), "dest": self.dest.summary(),
                "checks": self.checks}


def _rotz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def bearing_deg(xy: Sequence[float]) -> float:
    """Azimuth of a point from the right shoulder (`crane_pick_live`)."""
    return math.degrees(math.atan2(xy[1] - CP._SHOULDER_XY[1], xy[0] - CP._SHOULDER_XY[0]))


def predicted_hold(pick: CP.CranePlan) -> Tuple[np.ndarray, np.ndarray]:
    """Where the object will sit in the hand once closed: the planned grasp
    (`grasp_transform`), moved `GRASP_SHIFT_M` toward the thumb."""
    h = G.hand_frames(pick.rungs[0], pick.spec.opening_deg)
    rot_rel, t_rel = G.grasp_transform(h, pick.obj)
    return rot_rel, t_rel + GRASP_SHIFT_M * (h.thumb_rot.T @ -h.jaw)


def measured_hold(q: Sequence[float], gripper_deg: float, obj: G.Box
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Where the object measurably sits in the hand (thumb frame)."""
    return G.grasp_transform(G.hand_frames(q, gripper_deg), obj)


def plan_destination(planner, scene: SceneModel, pick: CP.CranePlan,
                     rel: Tuple[np.ndarray, np.ndarray], target_xy: Sequence[float],
                     psi_deg: float, place: PlaceSpec,
                     seeds: Sequence[Sequence[float]] = ()) -> Tuple[CP.CranePlan, np.ndarray]:
    """The destination ladder: rung 0 holds the object standing on the
    target, every rung above it 1 cm higher, the hand turned ``psi_deg``
    about the vertical from its grasp attitude.  Returns (plan, target)."""
    s = replace(pick.spec, release_retreat_m=place.release_retreat_m,
                release_backoff_m=place.release_backoff_m)
    lim = replace(pick.limits, max_lateral_correction_m=place.max_setdown_correction_m,
                  max_attempt_s=place.max_attempt_s)
    arm = planner._arm
    rd = _rotz(psi_deg) @ G.hand_rotation(pick.rungs[0])
    rot_rel, t_rel = rel
    axes = rd @ rot_rel
    half = pick.obj.half
    target = np.array([float(target_xy[0]), float(target_xy[1]),
                       scene.table_surface_z + float(np.abs(axes[2]) @ half)])
    box = G.Box(pick.object_id, target, axes, half)
    # The jaw-gap centre (at the ladder's opening) is rigid in the thumb frame.
    h0 = G.hand_frames(pick.rungs[0], s.opening_deg)
    v = h0.thumb_rot.T @ (h0.gap_mid - h0.thumb.center)
    g0 = target - rd @ t_rel + rd @ v
    tol = lim.ik_position_tol_m
    # The grasp's own posture first (continuity), then the crane's start set.
    q0 = None
    for seed in [pick.rungs[0]] + [list(q) for q in seeds] + CP.GRASP_STARTS:
        got = CP.solve_gap(arm, rd, g0, s.opening_deg, seed, tol)
        if got is not None and CP._rot_err_deg(G.hand_rotation(got[0]), rd) <= 1.0:
            q0 = got[0]
            break
    if q0 is None:
        raise CP.CraneRefused("destination", "IK found no pose that sets the object "
                              "down on the target with the grasp's attitude")
    targets = [g0 + np.array([0.0, 0.0, k * s.rung_m]) for k in range(s.via_rungs + 1)]
    rungs = CP._solve_ladder(arm, rd, targets, s.opening_deg, q0, tol)
    dest = CP.CranePlan(pick.object_id, s, lim, box, rd, targets, rungs, list(pick.present))
    try:
        dest.insertion_rung = CP._insertion_rung(dest)
    except CP.CraneRefused:
        dest.insertion_rung = 0
    return dest, target


def _segments(place: PlacePlan) -> Dict[str, Tuple[List[List[float]], float, TargetContact, bool]]:
    """(poses, gripper deg, target contact, held) per segment after the hold."""
    s = place.dest.spec
    oid = place.object_id
    d, c = place.dest, place.carry_rung
    at = TargetContact(oid, hand_contact=True, center=tuple(place.target))
    held = TargetContact(oid, hand_contact=True, carried=True)
    out = {
        f"carry: climb to +{c} (pickup ladder)": (place.climb(), s.close_deg, held, True),
        f"carry: to {place.cell} +{c}": (place.carry(), s.close_deg, held, True),
        f"set-down: {place.cell} +{c} -> resting": (place.setdown(), s.close_deg, held, True),
        f"retract: {place.cell} resting -> hover": (
            [d.rungs[k] for k in range(0, s.hover_rungs + 1)], s.opening_deg, at, False),
        f"{place.cell} hover -> via": (
            [d.rungs[k] for k in range(s.hover_rungs, s.via_rungs + 1)], s.opening_deg, at, False),
        f"return {place.cell} via -> PRESENT": (
            [d.via, d.present], s.transit_opening_deg,
            TargetContact(oid, hand_contact=False, center=tuple(place.target)), False),
    }
    return out


#: The measured exits from PRESENT (`rig_routes.POSTURE_TRANSITIONS`).
EXITS_FROM_PRESENT: Tuple[Tuple[str, str], ...] = (("rest", "LOWER_TO_REST"),
                                                    ("home", "STOW_FROM_SIDE"))


def exits_open(scene: SceneModel, object_id: str, box: G.Box) -> Dict[str, Dict[str, object]]:
    """Each exit from PRESENT with ``object_id`` standing as ``box``: the
    executor's footprint gate (`footprint.route_footprint_clearances` against
    `rig_routes.FOOTPRINT_MARGIN`, the check `_footprint_refusal` makes before
    it flies the exit).  ``scene`` is not changed."""
    import copy

    from reachy_ai.motion import rig_routes as R
    from reachy_ai.motion.footprint import route_footprint_clearances

    there = copy.copy(scene)
    there._objects = dict(scene._objects)
    there._objects[object_id] = replace(
        scene.get(object_id), center=tuple(float(v) for v in box.center),
        quat=G._rot_to_quat(box.axes))
    out: Dict[str, Dict[str, object]] = {}
    for posture, route in EXITS_FROM_PRESENT:
        c = route_footprint_clearances(there, [route]).get(object_id)
        d = math.inf if c is None else float(c.distance)
        out[posture] = {"route": route, "clearance_m": d,
                        "open": d >= R.FOOTPRINT_MARGIN}
    return out


def check_place(planner, scene: SceneModel, place: PlacePlan,
                offset: Optional[np.ndarray] = None,
                from_rung: Optional[int] = None) -> Dict[str, object]:
    """Every check the carry, the set-down and the way home are held to.
    Raises CraneRefused.  ``offset`` (measured - commanded joints) makes the
    table checks about the predicted realised arm as well; ``from_rung``
    (a destination rung) checks only what is left from there."""
    s, ps = place.dest.spec, place.spec
    off = np.zeros(7) if offset is None else np.asarray(offset, dtype=float)
    oid = place.object_id
    rep: Dict[str, object] = {}
    segs = _segments(place)
    if from_rung is not None:
        segs = {k: v for k, v in segs.items() if not k.startswith("carry:")}
        name = next(k for k in segs if k.startswith("set-down:"))
        _q, g, t, h = segs[name]
        segs[name] = (place.setdown(from_rung), g, t, h)

    # 0. Inside the travel.
    for k, q in enumerate(place.dest.rungs):
        if not within_limits(q):
            raise CP.CraneRefused("joint_limits", f"destination rung +{k} is outside "
                                  "the joint travel", pose=[round(float(v), 3) for v in q])

    # 1. Where the object will stand is free.
    for o in scene.objects.values():
        if not o.collides or o.id == oid or "grid-cell" in o.tags or \
                (scene.table is not None and o.id == scene.table.id):
            continue
        gap = G.obb_separation(place.dest.obj, G.object_box(o))
        if gap < ps.destination_free_m:
            raise CP.CraneRefused("destination", f"{place.cell} is not free: {o.id} is "
                                  f"{1000 * gap:.1f} mm from where {oid} would stand",
                                  other=o.id, gap_m=gap)
    rep["destination_free"] = "pass"

    # 2. The whole-arm preflight, unchanged, on every segment.
    for name, (seq, grip, tgt, _held) in segs.items():
        try:
            planner.check_arm_path(seq, segment=name, gripper_deg=grip, target=tgt)
        except ArmClearanceError as exc:
            raise CP.CraneRefused("check_arm_path", str(exc), segment=name, link=exc.link,
                                  obstacle=exc.obstacle, clearance_m=exc.clearance_m) from exc
    rep["check_arm_path"] = "pass (all segments)"

    # 3. Arm links and pads vs the table, commanded and predicted.
    tab = {}
    for name, (seq, grip, _tgt, held) in segs.items():
        dense = G.densify(seq, 1.0)
        pred = [list(np.asarray(q) + off) for q in dense]
        if held and name.startswith("set-down:"):
            # The lowering stops when the held object meets the table
            # (`execute_crane_pick`'s set-down), so the predicted hand goes no
            # lower than that -- the loaded arm sags, and its rung 0 would.
            for i, q in enumerate(pred):
                b = G.carried_box(G.hand_frames(q, grip), place.pick.obj, place.rel)
                if b.lowest_z <= scene.table_surface_z:
                    pred = pred[:i + 1]
                    break
        # The measured offset is the LOADED arm's: it predicts the segments
        # that carry the object, not the ones after the release (those are
        # judged again from the measured state, by #173's way out).
        checks_ = (("commanded", dense), ("predicted", pred)) if held or offset is None \
            else (("commanded", dense),)
        for label, qs in checks_:
            tr = G.table_clearance(scene, qs, grip)
            tab[f"{name} ({label})"] = round(1000 * tr.worst_m, 1)
            if tr.worst_m < 0.0:
                raise CP.CraneRefused("table", f"{tr.link} {1000 * tr.worst_m:.1f} mm into "
                                      f"the table on '{name}' ({label})", segment=name,
                                      link=tr.link, clearance_m=tr.worst_m,
                                      predicted=(label == "predicted"))
    rep["table_mm"] = tab

    # 4. The HELD object: against the table and every other object, and
    # against the upper arm and forearm.
    held_rep = {}
    for name, (seq, grip, _tgt, held) in segs.items():
        if not held:
            continue
        setting_down = name.startswith("set-down:")
        if setting_down:
            # Down to where the object meets the table, where the set-down
            # stops (its lowest commanded rung is never reached: support is).
            seq = G.densify(seq, 0.5)
            for i, q in enumerate(seq):
                if G.carried_box(G.hand_frames(q, grip), place.pick.obj,
                                 place.rel).lowest_z <= scene.table_surface_z + 1e-4:
                    seq = seq[:i + 1]
                    break
        cr = G.carried_clearance(scene, oid, seq, grip, place.pick.obj, place.rel)
        arm_d, link, _i = G.carried_arm_clearance(seq, grip, place.pick.obj, place.rel)
        held_rep[name] = {**cr.as_dict(), "arm_mm": round(1000 * arm_d, 2), "arm_link": link}
        if cr.others_m < 0.0:
            raise CP.CraneRefused("carried", f"the held object would hit {cr.other_id} "
                                  f"({1000 * cr.others_m:.1f} mm) on '{name}'",
                                  segment=name, carried=cr.as_dict())
        floor = -1e-3 if setting_down else ps.carry_table_clearance_m
        if cr.table_m < floor:
            raise CP.CraneRefused("carried", f"the held object would come "
                                  f"{1000 * cr.table_m:.1f} mm from the table on '{name}' "
                                  f"({1000 * floor:.0f} mm needed)", segment=name,
                                  carried=cr.as_dict())
        if arm_d < 0.0:
            raise CP.CraneRefused("carried", f"the held object would touch the {link} "
                                  f"({1000 * arm_d:.1f} mm) on '{name}'", segment=name)
    rep["held"] = held_rep

    # 5. An exit from PRESENT with the object where it will stand (#174 exit
    # check): the placement ends at PRESENT, and an object left in the path
    # of both measured exits (to REST, to HOME) leaves the arm nowhere to go
    # that the executor's own footprint gate would let it fly.
    exits = exits_open(scene, oid, place.dest.obj)
    rep["exits"] = exits
    if not any(e["open"] for e in exits.values()):
        raise CP.CraneRefused("exit", f"{oid} on {place.cell} would leave no validated exit "
                              "from PRESENT: " + "; ".join(
                                  f"{p} ({e['route']}) {1000 * e['clearance_m']:.0f} mm"
                                  for p, e in exits.items()), exits=exits)

    # 6. The open hand's way out of the placed object, as planned: from the
    # ladder the release retreat leaves it on (the hand backs away along the
    # jaw while it lets go, so the way up starts there).  The live way out is
    # judged again after the release (#173), from where the object is then.
    h0 = G.hand_frames(place.dest.rungs[0], s.close_deg)
    back = -s.release_retreat_m * h0.jaw - s.release_backoff_m * h0.approach
    up = [list(q) for q in place.dest.descent(s.hover_rungs)]
    if back.any():
        moved = CP._shift_ladder(planner, place.dest, back)
        up = [moved[k] for k in range(s.hover_rungs, -1, -1)]
    out_route = G.route_clearance(up, s.opening_deg, place.dest.obj, offset=off)
    rep["pad_route_out"] = {**out_route.as_dict(),
                            "retreat_mm": round(1000 * s.release_retreat_m, 1)}
    # Judged here only for the PREDICTED hold.  A measured hold has the pads
    # squeezed millimetres into the object (the soft contact under the grip,
    # #173), and the way out it really needs is judged after the release,
    # from where the object then is, by #173's check before opening.
    if place.hold_source == "predicted" and out_route.min_m < 0.0:
        raise CP.CraneRefused("pad_route", "the opened hand's way up would cut the placed "
                              f"object ({1000 * out_route.min_m:.2f} mm)",
                              route=out_route.as_dict())
    place.checks = rep
    return rep


def plan_crane_place(planner, scene: SceneModel, object_id: str, target_cell: str,
                     pose: Optional[CP.Observation] = None,
                     spec: CP.CraneSpec = PLACE_GRASP,
                     limits: CP.CraneLimits = CP.CraneLimits(),
                     place: PlaceSpec = PlaceSpec(),
                     on_event: Optional[CP.EventLog] = None,
                     pick_starts: Optional[Sequence[Sequence[float]]] = None,
                     dest_seeds: Sequence[Sequence[float]] = (),
                     headings: Optional[Sequence[float]] = None) -> PlacePlan:
    """Plan and check the whole placement.  IK/FK calls only -- no motion.

    The hand keeps its grasp attitude to the set-down, turned about the
    vertical by a HEADING (the object's own heading does not matter): the
    change in bearing from the shoulder first, then every 15 deg nearest it
    first (`headings_from`).  ``pick_starts`` / ``dest_seeds`` add IK starts
    for an attitude the crane's own start set was not built for.  Raises
    CraneRefused: when no heading passes, the refusal of the first heading
    tried (the bearing change, the one the arm was meant to fly); every other
    heading's is in the event log."""
    ev = on_event or (lambda *_a, **_k: None)
    cell_id = target_cell if target_cell.startswith("cell_") else f"cell_{target_cell}"
    try:
        cell = scene.get(cell_id)
    except KeyError:
        raise CP.CraneRefused("destination", f"there is no cell {target_cell} in this scene")
    if "grid-cell" not in cell.tags:
        raise CP.CraneRefused("destination", f"{target_cell} is not a grid cell")
    if "no-right-arm" in cell.tags:
        raise CP.CraneRefused("destination", f"{target_cell} is out of the right arm's reach")
    cx, cy, _tz = scene.cell_center(cell_id)
    tx, ty = aimed((cx, cy), place)
    # The cell must be free whatever the heading: nothing within the object's
    # footprint circle (plus the margin) of the cell centre.
    held = G.object_box(scene.get(object_id), *(() if pose is None else
                                                   (pose.position, pose.quat_wxyz)))
    reach = float(np.linalg.norm(held.half[:2]))
    square = G.Box(object_id, np.array([tx, ty, held.center[2]]), np.eye(3),
                   np.array([reach, reach, held.half[2]]))
    for o in scene.objects.values():
        if not o.collides or o.id == object_id or "grid-cell" in o.tags or \
                (scene.table is not None and o.id == scene.table.id):
            continue
        gap = G.obb_separation(square, G.object_box(o))
        if gap < place.destination_free_m:
            raise CP.CraneRefused("destination", f"{target_cell} is not free: {o.id} is "
                                  f"{1000 * gap:.1f} mm from where {object_id} would stand",
                                  other=o.id, gap_m=gap)
    limits = replace(limits, max_attempt_s=place.max_attempt_s)
    pick = CP.plan_crane_pick(planner, scene, object_id, pose, spec, limits, on_event,
                              starts=pick_starts)
    moved = math.hypot(tx - pick.obj.center[0], ty - pick.obj.center[1])
    if moved < place.min_transfer_m:
        raise CP.CraneRefused("destination", f"{object_id} is already on {target_cell} "
                              f"({1000 * moved:.0f} mm from its centre)")
    psi0 = bearing_deg((tx, ty)) - bearing_deg(pick.obj.center[:2])
    rel = predicted_hold(pick)
    last: Optional[CP.CraneRefused] = None
    for psi in (headings_from(psi0) if headings is None else headings):
        try:
            dest, target = plan_destination(planner, scene, pick, rel, (tx, ty), psi, place,
                                            seeds=dest_seeds)
        except CP.CraneRefused as exc:
            ev("place_heading_refused", psi_deg=round(psi, 1), stage=exc.stage,
               reason=exc.reason)
            last = last or exc
            continue
        for c in place.carry_rungs:
            try:
                pp = PlacePlan(object_id, cell_id, target, place, pick, dest, c, psi, rel,
                           cell_centre=np.array([cx, cy]))
                check_place(planner, scene, pp)
            except CP.CraneRefused as exc:
                ev("place_carry_refused", carry_rung=c, psi_deg=round(psi, 1),
                   stage=exc.stage, reason=exc.reason)
                last = last or exc
                continue
            ev("place_checked", carry_rung=c, target_cell=cell_id, psi_deg=round(psi, 3),
               target=[round(float(v), 5) for v in target])
            return pp
    assert last is not None
    raise last


def support_rung(dest: CP.CranePlan, rel: Tuple[np.ndarray, np.ndarray], obj: G.Box,
                 offset: Sequence[float], table_z: float) -> int:
    """The highest destination rung at which the held object, on the arm as
    it is predicted to sit (commanded + the measured loaded offset), already
    meets the table: where the set-down's lowering will stop.  0 if none."""
    off = np.asarray(offset, dtype=float)
    g = dest.spec.close_deg
    for j in range(len(dest.rungs) - 1, -1, -1):
        b = G.carried_box(G.hand_frames(np.asarray(dest.rungs[j]) + off, g), obj, rel)
        if b.lowest_z <= table_z:
            return j
    return 0


def replan_setdown(planner, scene: SceneModel, place: PlacePlan, dest: CP.CranePlan,
                   shift: np.ndarray, at_rung: int, offset: Sequence[float]) -> CP.CranePlan:
    """A set-down correction: ``dest`` (the destination ladder as flown)
    moved by ``shift``, re-solved from ``at_rung`` up, and down only to one
    rung past where the held object is predicted to meet the table -- the
    rungs below are never commanded, since the lowering stops at support, and
    the lowest of them stand against the shoulder-roll stop at r2c1 (live run
    3, 2026-10-12: the correction failed there).  If that rung has no pose,
    the contact rung itself.  Raises CraneRefused."""
    contact = support_rung(dest, place.rel, place.pick.obj, offset, scene.table_surface_z)
    tries = sorted({min(at_rung, max(0, contact - 1)), min(at_rung, contact)})
    last: Optional[CP.CraneRefused] = None
    for low in tries:
        try:
            return CP.replan(planner, scene, dest, shift, at_rung, lowest=low)
        except CP.CraneRefused as exc:
            last = exc
    assert last is not None
    raise last


def aimed(centre_xy: Sequence[float], place: PlaceSpec) -> Tuple[float, float]:
    """Where the set-down is aimed: the cell centre moved ``place.aim_m``
    toward ``place.aim_toward_xy``."""
    cx, cy = float(centre_xy[0]), float(centre_xy[1])
    if place.aim_m <= 0.0:
        return cx, cy
    u = np.array([place.aim_toward_xy[0] - cx, place.aim_toward_xy[1] - cy])
    u = u / (np.linalg.norm(u) or 1.0)
    return cx + place.aim_m * float(u[0]), cy + place.aim_m * float(u[1])


def headings_from(psi0: float, step: float = 15.0) -> List[float]:
    """Headings to try for the set-down: ``psi0``, then every ``step``
    degrees around the circle, nearest to ``psi0`` first."""
    out = [psi0]
    k = 1
    while k * step < 180.0 - 1e-9:
        out += [psi0 + k * step, psi0 - k * step]
        k += 1
    if abs(k * step - 180.0) < 1e-9:
        out.append(psi0 + 180.0)             # the opposite heading, once
    return out


def replan_from_hold(planner, scene: SceneModel, place: PlacePlan,
                     rel: Tuple[np.ndarray, np.ndarray],
                     offset: Optional[np.ndarray] = None) -> PlacePlan:
    """The destination re-solved for the MEASURED hold, and re-checked
    (with the measured joint offset).  Raises CraneRefused."""
    dest, target = plan_destination(planner, scene, place.pick, rel,
                                     place.target[:2], place.psi_deg, place.spec,
                                     seeds=[place.dest.rungs[0]])
    pp = replace(place, dest=dest, target=target, rel=rel, hold_source="measured",
                 checks={})
    check_place(planner, scene, pp, offset=offset)
    return pp


def placement_error(position: Sequence[float], target: Sequence[float]) -> float:
    """Horizontal distance (m) of the object's centre from the target."""
    return math.hypot(float(position[0]) - float(target[0]),
                      float(position[1]) - float(target[1]))
