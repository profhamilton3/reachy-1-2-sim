# ADR-0003: Arm geometry has a second, verified hand model; the guard keeps the tube

- Status: Accepted (geometry only — no consumer default changes)
- Date: 2026-09-12
- Decision owners: IITG Reachy 1.2 simulation project
- Relates to: #56, #74; ADR-0002's revisit condition on the capsule-vs-physics
  disagreement

## Context

ADR-0002 accepted the SWING_1 graze on the evidence that eighteen flown legs,
with objects on the board, left it undisturbed — and flagged that the
forearm-footprint guard's own capsule model disagrees with that evidence at
r2c3, in a way neither this repo nor #73/#74 had explained. The 2026-09-12
review (`outputs/review-2026-09-12-opus-repair-review-and-56-74-design.md`,
§3.2) resolves it:

- `kinematics.link_capsules`'s `upper_arm` and `forearm` capsules are copied
  directly from the MJCF's own `r_upper_arm_col` / `r_forearm_col` capsules —
  they agree with the physics to sub-millimetre precision, everywhere.
- The **hand** is not a capsule at all in the MJCF. It is two shell bodies —
  `r_thumb_body` (a fixed box) and `r_gripper_finger`'s `r_finger_body` (a box
  that swings with the gripper joint) — plus a `r_wrist_ball` marker. The
  guard's `hand` capsule is one isotropic tube around both, radius 5.2–8.3 cm
  depending on aperture (`hand_radius`).
- Every documented capsule-vs-physics disagreement examined — the evidence
  board, SWING_1's graze, #73's tucked sweep — is the tube reading more
  conservative than the shells by as much as 9 cm. The forearm and upper arm
  were never the disagreement.
- The MJCF's own **collision** hand (`r_thumb_col` / `r_finger_col`) is two
  small fingertip boxes; the palm, wrist ball and shell bodies do not collide
  (`contype=0`). The simulator cannot report contact on the hand's body at
  all — "board undisturbed" over eighteen flights is evidence about the
  fingertips, forearm and upper arm, not the palm.

Neither the tube nor the fingertip-only collision hand is the truth. The
visual shells are the project's own best description of the hand's real
volume and sit between them.

## Decision

### 1. A second, opt-in hand model: `link_capsules(..., hand="shells")`

`kinematics.link_capsules` gains a `hand` parameter, default `"tube"` — no
behaviour change for any existing caller. `hand="shells"` returns five
capsules instead of three: `upper_arm` and `forearm` unchanged, plus `thumb`,
`finger` and `wrist_ball` in place of `hand` — one capsule per MJCF visual
shell, built from `link_frames` plus the finger hinge, placed exactly where
the MJCF places `r_gripper_thumb` → `r_gripper_finger`:

- thumb frame: `wrist2hand` frame, translated `(0, 0, −0.0325)` in the
  **pitch-only** frame (before `r_wrist_roll` — the roll is the thumb body's
  own joint, so the translation to reach it happens in the parent's, not the
  rolled, frame).
- finger frame: thumb frame, translated `(0, −0.037, −0.040)`, then rotated
  about the thumb's local X by the gripper angle.
- capsule axes run along each body's local Z through its shell box centre;
  radii are the box's XY half-diagonal (`thumb` 3.75 cm, `finger` 1.56 cm);
  `wrist_ball` is a zero-length capsule, radius 2.8 cm, at the wrist2hand
  origin.

Verified against the compiled MJCF (`tests/unit/test_arm_geometry_mjcf.py`,
`TestShellsMatchMJCF`), at every named waypoint pose in `rig_routes` and at
13 samples of every `FOOTPRINT_LEGS` leg, over the full gripper range
(`{−68.8, −45, 0, 20}` deg): every capsule endpoint agrees with MuJoCo's own
`geom_xpos`/`geom_xmat` for `r_thumb_body`/`r_finger_body`/`r_wrist_ball` to
machine precision (worst observed error `4e-16` m across a broad joint-space
sweep, and exactly 0 within the routes' own samples).

`hand_radius` and the tube are untouched. Nothing outside the new
`hand="shells"` branch and its tests calls the new mode.

### 2. The fixture table (review §4), reproduced with `SceneModel.clearance`

`tests/unit/test_footprint_boards.py` computes the review's four named
boards, worst clearance over **any** guarded route (every `FOOTPRINT_LEGS`
leg, not only the one a given ability flies), for both hand models:

| fixture | tube | shells-capsules | decision |
|---|---|---|---|
| evidence board (`soda_can`@r1c1, `foam_block`@r2c3) | −1.7 cm | **+4.5 cm** | flyable by geometry; margin undecided |
| incident board (`foam_block` 7 cm right of r2c3) | −8.0 cm | −2.4 cm | refused |
| `foam_block` on r3c3 | −5.2 cm | **+3.4 cm** | flyable by geometry |
| `soda_can` on r2c3 | −0.4 cm | **+4.5 cm** | flyable by geometry |

Matches the review's own numbers to within ±0.3 cm (small differences come
from this repo's exact sample points vs. the review's independent probe
script). "Flyable by geometry" is recorded as exactly that — not as "safe to
fly" — because no margin has been decided (§4 below) and no through-the-move
data exist yet (E1).

### 3. The guard does not change

`panel_executor._footprint_refusal` keeps calling `link_capsules` with its
default (`hand="tube"`), and `rig_routes.FOOTPRINT_MARGIN` stays `0.0`. No
margin, waypoint, speed, or the pick/place `CLEAR_Z`/8 Hz rate changes in
this slice. The rule this ADR writes down is: **arm links are the MJCF
collision capsules; the hand is a bound on the MJCF shells; the pad-point
check (`SceneModel.check_point`) is for the table surface only and is not a
clearance model.**

## A discrepancy this ADR does not paper over

The 2026-09-12 assignment asked the validation tests to also assert that the
tube capsule **contains** the shells capsules unconditionally — "the tube's
radius ≥ every shells capsule's distance from the wrist axis" — and,
separately, that on the review's 224-position sweep "shells" would never
read more clearance than "tube" and never less than the true MJCF
reference. Measured, the first does not hold, and the reason is not the one
first suspected:

- **`hand_radius()` swings the wrong lever arm.** It rotates the finger
  box's *centre* (`_FINGER_ARM = 0.038`, 3.8 cm below the hinge) by the
  gripper angle and then adds the box's *unrotated* Y half-extent (1.0 cm).
  But `r_finger_body` extends 7.6 cm below the hinge, and at −68.8° that far
  end swings out 0.076 · sin 68.8° = 7.1 cm, not 3.5. The finger shell's
  far end therefore sits 10.8 cm (centreline) / 12.4 cm (surface) from the
  wrist axis **at `r_wrist_roll = 0`** — 4.0 cm past the tube's 8.3 cm
  radius at `HOVER`/`PRESENT`/`HOME`, gripper open. `hand_radius`'s own
  comment table ("−68 wide → 8.2 cm from the axis") is wrong for the same
  reason.
- **Wrist roll adds a second, smaller term.** The tube is anchored at the
  wrist frame, but the hand pivots at `r_gripper_thumb`, 3.25 cm below it in
  the pitch-only frame, so the hand's centreline is offset from the tube axis
  by 0.0325 · sin(roll) — 1.6 cm at `REST`/`REST_SHUT`'s `r_wrist_roll = 30`,
  used throughout `FOOTPRINT_LEGS`. That takes the open-hand gap from 4.0 cm
  to 5.6 cm there.
- **The MJCF collision pad is outside the tube too.** This is not only a
  shells-vs-tube abstraction: `r_finger_col` — the geom physics actually
  contacts with — sits 2.2 cm outside the tube at `HOVER` open, 1.8 cm at
  −45° (the aperture `PLACE_ROUTE`'s guarded legs use), and 3.8 cm at `REST`
  open. The tube consumers fly today is under-conservative in the finger's
  swing direction. On the review's own 224-position, 6 cm-cube,
  PLACE_ROUTE-tail sweep this shows up as "tube" reading *more* clearance
  than "shells" at 3/224 positions, by up to 2.8 cm, in the near-right corner
  strip PLACE_ROUTE's tail is already worst on
  (`tests/unit/test_footprint_boards.py::TestSweepOrdering`).

All of the above verified two ways: through `link_capsules("shells")` and
independently straight off MuJoCo's `r_finger_body`/`r_finger_col`/
`r_thumb_body` frames, to rule out a bug in the new code rather than a fact
about the geometry
(`tests/unit/test_arm_geometry_mjcf.py::TestTubeVsShellsAxisDistance`).

Two things that looked like discrepancies and are not:

- At gripper ≥ 0 and `r_wrist_roll = 0` a ~0.3 cm gap remains, from the
  `thumb` capsule. That is the shells capsule's own slack over the thumb box
  (its radius is the box's XY half-diagonal around a centreline 1.8 cm off
  axis), not geometry: `hand_radius`'s `_THUMB_RADIUS = hypot(0.025, 0.046)`
  is the exact corner distance, so the tube already contains the box.
- An earlier draft of the sweep reported "shells" reading *more* clearance
  than the MJCF reference at 2/224 positions. That was a measurement
  artifact: tube and shells were evaluated at the leg's worst-case aperture
  while the reference was evaluated at the interpolated commanded aperture,
  and a wider finger can be farther from an object on the thumb side. With
  the reference at the same aperture the count is 0/224, as the capsule
  bound predicts.

This is **not** evidence the guard is unsafe against any board it has been
flown over — it still flies `"tube"` only, at the margin it always has, and
"shells" is meaningfully tighter than "tube" at most of the board (158/224
positions in the same sweep). It does mean "the tube bounds the shells" is
false as a geometric statement whenever the gripper is open, and the
`hand_radius` lever arm is a defect in the live guard's model, not a
modelling nicety. Correcting it (open-hand radius 8.3 → ~12.4 cm, −45°
7.5 → ~10.7 cm) changes which boards the footprint guard refuses, so it is
deliberately **not** done in this slice: it is the first item under "What is
still open" and needs the fixture table re-run alongside it. Reported here
rather than adjusted to fit, per the assignment's own instruction: a
contradicted inequality is a finding, not a test to make pass.

## What is still open

- **E1** — re-fly the tabletop legs with 20 Hz joint logging, reporting
  per-link realised clearance under both hand models
  (`scripts/measure_route_clearance.py`, committed this slice, offline
  reporting half unit-tested; the recording half needs an operator and a
  live arm). Resolves: the tabletop margin: whether `"shells"` (or a margin
  on `"tube"`) should ever become the guard's default.
- **E2** — the same instrumentation on the SWING_1 rail crossing.
- **E3** — real gripper envelope vs. the MJCF shells (tape measure on the
  physical Reachy 1.2): whether `"shells"` is a bound on the *real* hand,
  not only the model's.
- **The `hand_radius()` lever arm** (first, Slice 2): swing the finger box's
  far end (7.6 cm below the hinge), not its centre, and add the rotated
  cross-section. This raises the open-hand tube from 8.3 to ~12.4 cm and the
  −45° tube from 7.5 to ~10.7 cm, so it must land together with a re-run of
  the fixture table in §2 and a decision on which refusals it newly causes —
  or be superseded outright by switching the guard to `"shells"`, which has
  no such lever-arm approximation to get wrong. The 1.6 cm wrist-roll term
  is a separate, smaller correction (anchor the tube at the thumb pivot, or
  add 0.0325 · sin(roll) to the radius).

## Consequences

- A route or ability wanting `"shells"` clearance can ask for it; nothing
  does yet.
- `docs/adr/0002`'s revisit condition on the capsule-vs-physics disagreement
  is met: the disagreement is the hand tube, explained and quantified, not
  an open question.
- The planning-time question the review raised (`maximise_clearance=True`
  timing, before `plan_segment` could default to it) is unmeasured and not
  part of this slice.

## Revisit conditions

Revisit this ADR when:

- E1 delivers realised per-leg clearance and a margin decision is made for
  Slice 2 (switching the tabletop guard to `"shells"`);
- the `hand_radius()` lever arm is corrected (with the fixture table re-run)
  or the guard moves to `"shells"` and the tube stops being what any consumer
  flies;
- E3 measures the real gripper against the MJCF shells.
