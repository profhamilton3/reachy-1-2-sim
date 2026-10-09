# ADR-0006: The rig corridor flies at a stated margin, and SWING_1 moved to meet it

- Status: Proposed. D1 and D2 are owner decisions (2026-10-09). The margin is enforced; **D1's realised
  half is NOT met** (§3), so no corridor route is listed as meeting it. Simulator only.
- Date: 2026-10-09
- Decision owners: IITG Reachy 1.2 simulation project
- Relates to: #74; ADR-0002 (whose SWING_1 graze this replaces), ADR-0003 (hand
  models), ADR-0004 (the enforced whole-arm model)

## Context

ADR-0002 accepted a graze. The notebook's `SWING_1` planned **+0.6 cm** past
`rig_rail_outer_right` at the pose and **+0.44 cm** along TUCK → SWING_1 (tube
hand), and eighteen flights realised −0.33 to −1.58 cm through the move. The
acceptance was a judgement about the model, not a repair, and #74 stayed open
for three things: whether SWING_1 can move, what margin the project flies at,
and enough flights to bound the spread rather than observe it.

## Decision

### 1. The margin (D1)

Stated in `rig_routes` and enforced by `check_route()` for every route that
flies SWING_1 (`CORRIDOR_ROUTES`: PLACE_ROUTE, STOW_ROUTE, RAISE_TO_SIDE,
STOW_FROM_SIDE):

| figure | rule | constant |
|---|---|---|
| planned | ≥ 2.0 cm | `CORRIDOR_PLANNED_MARGIN` |
| realised | ≥ 1.0 cm, worst over | `CORRIDOR_REALISED_MARGIN` |
| flights | ≥ 20 | `CORRIDOR_MIN_FLIGHTS` |

Both figures use the model the planner enforces (`check_arm_path`): the tube
hand, against the manipulable objects and the rig rails, not the tabletop
(REST lands on it by design). *Planned* is every leg of the full route, sampled
at ≤ 2°, with each leg's hand sized to its wider endpoint aperture. *Realised*
is the same model on the arm's own 20 Hz joint readings and realised aperture,
through the whole flight, against the object poses the simulator reported.

A corridor row without figures, or with figures short of the margin, is refused
with the shortfall named. The FWDCenterLabMCC corridor rows are therefore now
refused: their evidence is the notebook, which flew the notebook's SWING_1 and
recorded no figures. Nobody flies that scene through the panel today.

The tube is up to 9 cm more conservative than the MJCF hand (ADR-0003). The
margin is stated on it anyway, because it is the model the guard enforces.
Under `shells` the notebook pose already planned +2.25 cm. That is recorded
here so that nobody mistakes the tube for the truth.

### 2. SWING_1 moved (D2)

| | pitch | roll | arm yaw | elbow | wrist pitch |
|---|---|---|---|---|---|
| notebook (`SWING_1_NOTEBOOK`) | 37.5 | −32.5 | 0 | −120 | −45 |
| now (`SWING_1`) | 32.5 | −27.5 | +5 | −125 | −45 |

**How it was found.** By search, not by eye: about 170k nearby poses, each with
TUCK → SWING_1 and SWING_1 → SWING_2 sampled at ≤ 2°. Each was checked against
the objects, the rails, the tabletop, and the URDF torso box. The torso box is
not in `SceneModel`, and MuJoCo has robot↔robot contact off, so nothing else
would see it.

**What does the work.** The elbow fold does, −120 → −125: CURL's own fold. On
its own it takes the crossing from +0.44 to +2.08 cm. Five degrees on pitch,
roll and arm yaw then bring it to +3.48 cm, where SWING_2's own +3.53 cm
(upper arm) is the ceiling. Opening the wrist was rejected: it fouls the
board's edge instead.

**Other clearances on the same legs.**
- Tabletop: −0.25 → +0.53 cm.
- Torso: ≥ +5.5 cm.

**Whole routes.** Every corridor route now plans **+2.57 cm**. The binding
point moves off SWING_1 to the upper arm on SWING_3 → HOVER, which this change
does not touch.

**The notebook.** It is not edited and remains the source of truth for every
other pose. The test that pins the extraction now pins `SWING_1_NOTEBOOK` to
it, and asserts that `SWING_1` differs only on the four named joints.

One fact found on the way, unchanged here: `r_wrist_pitch` ±45° in the notebook
is 0.02° past the joint's travel (±44.977°). The simulator clips it, at TUCK,
SWING_1, SWING_2, SWING_3 and CURL alike.

### 3. The evidence (D1's realised half)

Flown 2026-10-09 in the live FWDCenterLabSivaPool simulator (MuJoCo, no-slip
10), from a worktree at `e9115b4`. The board was as found: all ten objects in
the floor pools. The routes were flown through `rig_motion.ROUTE_RUNNERS`, the
panel's own runners, in forty cycles alternating
`HOME→PLACE→REST→LIFT→PRESENT→WAVE→PRESENT→LOWER→REST→STOW→HOME` and
`HOME→RAISE_TO_SIDE→PRESENT→WAVE→STOW_FROM_SIDE→HOME`. Every flight arrived,
and no object moved (max 0.00 mm). Evidence:
`~/Reachy-Lab/outputs/sim/working/trial-2026-10-09-p2-74-swing1-flights/`.

| route | flights | planned | realised min | p5 | median | D1 |
|---|---|---|---|---|---|---|
| PLACE_ROUTE | 20 | +2.57 | **+0.03** | +1.86 | +2.36 | **fails** |
| STOW_ROUTE | 20 | +2.57 | +1.70 | +2.30 | +2.35 | meets |
| RAISE_TO_SIDE | 20 | +2.57 | +1.82 | +2.32 | +2.36 | meets |
| STOW_FROM_SIDE | 20 | +2.57 | **−1.09** | +1.30 | +2.36 | **fails** |

All figures are in cm, tube hand.

**The spread is not SWING_1's geometry.** Of the 80 corridor flights, 72 kept
`r_forearm_yaw` at 0 through the corridor, and they realised +2.31 to +2.40 cm,
tightly, against +2.57 planned. In the other 8, the forearm yaw swung to
−24…−52° while the route commanded 0, on all four routes. Those 8 hold every
reading below +2.31, including both failures:
- PLACE c2: 46°, +0.03 cm;
- STOW_FROM_SIDE c29: 52°, −1.09 cm, hand against the outer rail.

Forearm yaw is outside `CRITICAL_JOINTS` by design (ADR-0002 §3), so the route
neither waits for it nor stops on it. The cause of the excursion is not
established: a lost or echoed command, and coupling through the ±45° wrist
flip, are both open.

**Not done, by the D1 rule ("stop and report; do not loosen").**
- No figures were added to any row, and the SivaPool corridor rows still carry
  only the 2026-09-10 evidence, so `check_route` refuses all four corridor
  routes in SivaPool.
- STOW_ROUTE and RAISE_TO_SIDE met the margin on their own twenty flights, but
  they cross the same corridor with the same failure mode. Listing them alone
  would certify a corridor that has failed.

## Consequences

- **Merging this as it stands refuses the panel's `rest your arm`, `store your
  arm`, and `wave` or `point` from the pocket in FWDCenterLabSivaPool.** Every
  one of them needs a corridor route, and none has qualifying evidence. That is
  D1 applied. Whether to merge before the forearm-yaw excursion is fixed is the
  owner's decision.
- ADR-0002's §4 graze acceptance is superseded. Its revisit tripwires become
  this ADR's margin.
- `route_version` is **not** bumped. Stored `rest_forearm`/`stow_arm`
  experiences keyed on version 1 describe the old corridor. Bumping is an owner
  decision, raised with this change.
- The torso is still not in `SceneModel` (#74's remaining owner item).

## Revisit conditions

- any corridor flight that realises below 1.0 cm, or that disturbs the board;
- the rig geometry or the scene's rail placement changes;
- the guard moves to the `shells` hand (ADR-0003), which changes what "planned"
  means here;
- physical validation, which this is not.
