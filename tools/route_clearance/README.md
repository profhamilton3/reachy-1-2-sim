# tools/route_clearance: sim clearance records for tabletop arm routes

A0b of reachy-tabletop-ai `docs/design/arm-motion.md` (sections 3.2 and 10; tabletop #66). Offline MuJoCo only: no
server, SDK, Docker or robot.

```bash
python -m tools.route_clearance --routes <reachy-tabletop-ai>/src/reachy_ai/motion/arm_routes.json \
    --route B2_GRIP_TUNE --scene FWDCenterLabSivaPool --out <dir>
```

It writes `<dir>/<route>@<hash12>.json`, the record the tabletop commits under
`src/reachy_ai/motion/arm_route_records/`. Exit status 0 on pass, 1 on fail, 2 on bad input. The full B2_GRIP_TUNE run
takes about seven minutes on an Apple-Silicon Mac. The record is deterministic: the same inputs give byte-identical output.

## What it does

| Part | Method |
|---|---|
| Route | The route's legs out, then the same legs reversed (`return: reverse`). Leg duration = max(`min_duration_s`, largest arm-joint change / 10 deg/s). A pose's gripper is the one its arriving leg commands. |
| Hash | `arm_route_hash.py`, vendored byte for byte from the tabletop (header: commit and SHA-256), with the tabletop's limits table (`limits.py`). |
| Kinematic | The minimum-jerk path sampled at 50 Hz. Board: analytic, the lowest box corner or capsule end over the board top (Step 1's `board_check`), not `mj_geomDistance`. Rig, left arm, self (reported only) and pieces: exact box/capsule distances (`geometry.py`), because `mj_geomDistance` reads box pairs as 0 whenever they are closer than its distmax. |
| Physics | Motors on, a 1 s settle at the start pose, minimum-jerk goals streamed at 50 Hz from the measured position, a 1 s hold per leg. Following (gross four joints), arrival (the four and wrist pitch), and every right-arm contact per step, with the arm's largest joint deviation from the leg's end poses. |
| Contacts | Declared per route (`record.DECLARED_CONTACTS`). An unconditional declaration covers its pair at its pose: in the pose's near-board zone on an adjacent leg (+ 0.25 s), or in the hold there. A conditional one covers it only while the arm is within N deg (largest joint deviation) of the pose. Anything else fails. |
| Occupancy | Each cell except the target square; each lab piece (cube 50.8 mm, cylinder 45.7 x 71.1 mm) at its worst position in the cell (a 7 x 7 grid of centres, footprint inside). `may_be_occupied` only if it stays >= 2 cm from the arm and moves < 1 mm (largest displacement in flight); otherwise `must_be_empty`. Every `may_be_occupied` cell is then flown at once. |
| Target square | The cell under the close pose. Each lab piece centred there, then in 1 mm steps toward the approach, flown to the close pose and its hold. `target_placement_tolerance_mm` = the largest offset that moves it < 1 mm; a disturbed centred piece fails the verdict. With `--target-spot` (a close pose over no cell, e.g. PICK) the target is the spot under the close pose's tip, the record's `target_cell` is null, `target_placement.spot_xy_m` names the spot, and all nine cells are in the occupancy map. |
| Head | For each waypoint, the neck angles (roll 0, pitch and yaw inside the tabletop limits) that aim the eyes at `r_tip`, from the model's own FK and cameras (not `gaze.project`, #208), and whether the tip is in each 640 x 480 frame. |
| Verdict | Section 3.2: waypoints inside the limits; board, rig and left arm >= 1 cm (>= 0 near a `near_board` pose, not judged in a declared zone); following <= 6 deg, arrival <= 3 deg; no undeclared contact; plus the target square. |

Tests: `python3 -m pytest tests/unit -q -k route_clearance`.
