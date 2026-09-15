# ADR-0003: Arm geometry has a second, verified hand model; the guard keeps the tube

- Status: Accepted (geometry only — no consumer default, margin, waypoint
  or speed changes; the tube's radius and length were corrected 2026-09-14,
  see "Correcting the tube"; the Slice 2 precondition on "shells" covering
  the collision pads was met the same day, see "Slice 2: 'shells' covers
  the collision pads")
- Date: 2026-09-12
- Decision owners: IITG Reachy 1.2 simulation project
- Relates to: #56, #74; ADR-0002's revisit condition on the capsule-vs-physics
  disagreement

## Context

ADR-0002 accepted the SWING_1 graze on the evidence that eighteen flown legs,
with objects on the board, left it undisturbed — and flagged that the
forearm-footprint guard's own capsule model disagrees with that evidence at
r2c3, in a way neither this repo nor #73/#74 had explained. The 2026-09-12
review (`docs/reviews/2026-09-12-repair-review-and-56-74-design.md`, §3.2;
its probe scripts and summaries under `docs/reviews/probes-2026-09-12/`)
resolves it:

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

| fixture | tube (legacy, as reviewed) | tube (corrected 2026-09-14) | shells-capsules | decision |
|---|---|---|---|---|
| evidence board (`soda_can`@r1c1, `foam_block`@r2c3) | −1.7 cm | −5.7 cm | **+4.5 cm** | refused by the guard; flyable by visual-shell geometry; margin undecided |
| incident board (`foam_block` 7 cm right of r2c3) | −8.0 cm | −11.9 cm | −2.4 cm | refused |
| `foam_block` on r3c3 | −5.2 cm | −9.4 cm | **+3.4 cm** | refused by the guard; flyable by visual-shell geometry |
| `soda_can` on r2c3 | −0.4 cm | −4.4 cm | **+4.5 cm** | refused by the guard; flyable by visual-shell geometry |

The legacy column matches the review's own numbers to within ±0.3 cm (small
differences come from this repo's exact sample points vs. the review's
independent probe script). All four boards were already refused under the
legacy tube; the corrected tube (below) makes each ~4 cm more negative,
because the REST leg's radius at its guarded −45° aperture went 7.5 → 11.5 cm.
"Flyable by geometry" is recorded as exactly that — not as "safe to fly" —
because no margin has been decided (§4 below), no through-the-move data
exist yet (E1), and — found 2026-09-14 — the shells column is a statement
about the MJCF's *visual* shells only: `"shells"` does not contain the
thumb collision pad (see "Correcting the tube", item 4).

### 3. The guard's model does not change; its tube was corrected

`panel_executor._footprint_refusal` keeps calling `link_capsules` with its
default (`hand="tube"`), and `rig_routes.FOOTPRINT_MARGIN` stays `0.0`. No
margin, waypoint, speed, or the pick/place `CLEAR_Z`/8 Hz rate changes in
this slice. What did change (2026-09-14, "Correcting the tube" below) is the
tube's own radius and length, so that the capsule the guard flies actually
contains the MJCF hand. The rule this ADR writes down is: **arm links are
the MJCF collision capsules; the hand is a bound on every MJCF hand geom —
both visual shells and both collision pads; the pad-point check
(`SceneModel.check_point`) is for the table surface only and is not a
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
about the geometry (`tests/unit/test_arm_geometry_mjcf.py` — originally
`TestTubeVsShellsAxisDistance`, which pinned these negative margins; after
the correction below its cases live on as `TestDemonstratedMisses` and
`TestTubeVsShellsCapsules`).

Two things that looked like discrepancies and are not:

- At gripper ≥ 0 and `r_wrist_roll = 0` a ~0.3 cm gap remains, from the
  `thumb` capsule. That is the shells capsule's own slack over the thumb box
  (its radius is the box's XY half-diagonal around a centreline 1.8 cm off
  axis), not geometry: the legacy `hand_radius`'s thumb term,
  `hypot(0.025, 0.046)`, was the exact corner distance, so the tube already
  contained the box.
- An earlier draft of the sweep reported "shells" reading *more* clearance
  than the MJCF reference at 2/224 positions. That was a measurement
  artifact: tube and shells were evaluated at the leg's worst-case aperture
  while the reference was evaluated at the interpolated commanded aperture,
  and a wider finger can be farther from an object on the thumb side. With
  the reference at the same aperture the count is 0/224 against the
  *visual* shells, as the capsule bound predicts. (Against the collision
  pads as well it is 62/224 — a real gap in `"shells"`, item 4 of
  "Correcting the tube" below.)

The deferral this paragraph originally recorded — leave `hand_radius` for
Slice 2 — was **not approved** (2026-09-14): the demonstrated gap could not
be carried while the affected legs continue to fly. The correction follows.

## Correcting the tube (2026-09-14)

Scope: the tube's geometric coverage only. No consumer switched to
`"shells"`; no waypoint, speed, margin or aperture changed; `"shells"`
itself is untouched. Evidence: `docs/reviews/probes-2026-09-14/` and the
tests named below.

### 1. What changed in `kinematics.py`

- **`hand_radius(gripper_deg, wrist_roll_deg)`** is no longer a formula. It
  is the exact farthest distance from the tube axis of any corner of any
  box the MJCF hangs under `r_wrist2hand` — `r_thumb_body`, `r_finger_body`,
  **`r_thumb_col`, `r_finger_col`** — plus the wrist ball, at the given
  aperture and roll, built term for term from the MJCF chain
  (`_hand_corner_offsets`). Either argument may be `None`, meaning the
  worst case over that joint's whole supported range (−68.8…+20.05° and
  ±45°). The old signature `hand_radius(gripper_deg)` still works and is
  now conservative over roll.
- **`link_capsules(..., hand="tube")`** sizes the tube for the pose's own
  `r_wrist_roll` (the right arm; the left falls back to the worst case).
- **`_HAND_LEN` 0.145 → 0.150 m.** The old length reached the thumb pad's
  far face (0.1395). The finger shell's far corner, with the finger hanging
  straight down (≈ +7.5°), reaches 0.1492 — 4 mm past the old end cap. Found
  by the coverage test, not by inspection.

| `r_gripper` | roll 0 | roll 30 (REST) | roll 45 (limit) | legacy (any roll) |
|---|---|---|---|---|
| +20 (shut) | 5.2 cm | 6.7 | 7.4 | 5.2 |
| 0 | 5.2 | 6.7 | 7.4 | 5.2 |
| −45 (guarded legs) | 9.9 | **11.5** | 12.1 | 7.5 |
| −68.8 (wide) | 11.2 | 12.8 | 13.5 | 8.3 |

A radius-plus-length correction was sufficient: the tube axis is the
rolled wrist frame's Z, so the hand's position relative to it depends on
`r_gripper` and `r_wrist_roll` alone, and a capsule around that axis can
always contain it. What it cannot do is be *tight* in every direction — the
hand is asymmetric (the finger swings one way; the roll offset is one-sided)
and the tube is isotropic, so the corrected tube is a correct bound and a
loose one. That looseness is the cost accounted for in item 3.

### 2. Verification against the MJCF

`tests/unit/test_arm_geometry_mjcf.py`:

- **`TestTubeCoversMJCFHand`** — every corner of all four hand boxes and
  the wrist ball's far surface is inside the tube capsule `link_capsules`
  returns (segment distance ≤ radius, end caps included), at every named
  `rig_routes` waypoint, at 13 samples of every `FOOTPRINT_LEGS` leg, and on
  a grid over the full `r_gripper` (−68.8…+20.05°, 5° steps plus both
  limits) × `r_wrist_roll` (±45°, 5° steps) range. Measured worst excess
  **2 × 10⁻¹⁶ m**. And the bound is **tight**: at every grid point the
  farthest corner sits on the tube's surface to ≤ 10⁻⁶ m, so the radius is
  the geometry and not padding. `None` for either input is verified to bound
  every grid point.
- **`TestDemonstratedMisses`** — regression cases for each miss measured
  against the legacy radius (`r_finger_col` 2.19 / 1.77 / 3.80 / 3.38 /
  1.05 cm outside at HOVER −68.8, HOVER −45, REST −68.8, REST −45, REST 0;
  `r_finger_body` 2.88 / 2.37 / 4.50 / 3.99 / 1.20; plus REST_SHUT, PRESENT,
  HOME wide open). Each is now inside, straight off MuJoCo's geom frames:
  the pads by 0.4–0.7 cm, the finger shell by 0 (it is what sets the radius).
- **`TestTubeVsShellsCapsules`** — the assignment's "tube bounds the
  shells" inequality now holds for every shells capsule's *centreline*. The
  capsule *surfaces* still exceed the tube by up to 1.14 cm (open) / 0.32 cm
  (shut). That is the shells capsules' own slack over their boxes (a
  cylinder around a box pokes past the box's corners), bounded by the
  capsule's radius, not a coverage gap against the MJCF. Pinned.

`tests/unit/test_footprint_boards.py::TestSweepOrdering` (224 positions, 6 cm
cube, PLACE_ROUTE tail, reference over **all** MJCF hand geoms including the
pads, all models at the leg's guarded aperture): **tube ≤ MJCF reference at
224/224** and tube ≤ shells at 224/224 — before the correction, tube read
more than shells at 3/224 by up to 2.8 cm. Over-conservatism of the tube
against the true geometry on this sweep: up to 13.0 cm; shells is tighter
than tube by more than 1 cm at 189/224 (was 158).

### 3. Every board the corrected tube newly refuses

Measured as legacy-tube clearance ≥ 0 and corrected-tube clearance < 0 at
`FOOTPRINT_MARGIN = 0.0` (`docs/reviews/probes-2026-09-14/newly_refused.py`,
pinned by `TestCorrectionNewlyRefuses`):

**Scene objects on grid cells** — `red_cube`, `blue_cylinder`, `soda_can`,
`foam_block`, `pool_box_1`, `pool_cyl_1` on each of the nine cells, every
guarded route (54 boards × 6 routes). Exactly **two boards flip**, on every
guarded route (they all share the REST tail):

| board | legacy tube | corrected tube | visual-shells | why |
|---|---|---|---|---|
| `pool_box_1` (4 cm) on r2c3 | +0.3 cm | −3.7 cm | +6.3 cm | marginal before; the REST-leg radius grew 4 cm |
| `pool_cyl_1` (4 cm) on r2c3 | +0.9 cm | −3.1 cm | +5.9 cm | same |

Everything larger on r2c3 (`red_cube` −1.1, `blue_cylinder` −0.6,
`soda_can` −0.4, `foam_block` −1.7 cm) and everything on r3c3 (−3.4 to
−5.4 cm) was already refused under the legacy tube and stays refused.
Nothing on r1c1–r1c3, r2c1, r2c2, r3c1 or r3c2 is refused under either.

**6 cm cube on the 4 cm / 224-position grid**, worst over any guarded route:
refused **59 → 74**, i.e. **15 newly refused positions**. At **every one of
the 15 the true MJCF clearance is positive** (+3.3 to +11.8 cm): they lie in
a strip at y ≈ −0.08…−0.16 m, x ≈ 0.35…0.67 m, and a second at
x ≈ 0.67…0.71, y ≈ −0.16…−0.32 — the band the +4 cm of REST-leg radius newly
sweeps, on the finger's swing side. Full list with legacy / corrected /
shells / MJCF at each: `probes-2026-09-14/newly_refused.txt`.

**One clearance in the notebook scene** (`FWDCenterLabMCC.yaml`,
`test_arm_clearance.py`): at the notebook's REST pose with the hand wide
open, `red_cube` at its YAML position is now 1.6 cm from the hand (legacy:
3.5 cm, via the forearm — the hand was too small to be the nearest link).
Positive, so not a refusal.

**None of these refusals is loosened.** They are what an isotropic tube
that actually contains the hand costs; the way to recover them is a hand
model that follows the hand's shape (`"shells"`, Slice 2), not a smaller
tube. Interim, no code: those two boards, and the corner strip, are refused
on every guarded route until Slice 2.

### 4. A discrepancy in `"shells"`, found while verifying the tube

Adding the collision pads to the coverage reference showed that the opt-in
`"shells"` model does **not** contain `r_thumb_col`. In the MJCF the thumb
pad sits at z ∈ [−0.107, −0.063] below `r_gripper_thumb` — entirely
**below** the visual thumb box (z ∈ [−0.060, +0.016]); the MJCF's own
comment says the pinch pads were placed deliberately, not derived from the
"bulky visual box". Measured: the pad's far corner is **2.05 cm outside
every shells capsule** (1.25 cm with the finger shut, when the finger
capsule covers part of it); `r_finger_col` is 0.5 mm outside the finger
capsule (2 mm wider in X than its shell). Pinned by
`TestShellsMissThumbPad` and visible in the sweep as shells > reference at
62/224 positions (by up to 1.0 cm).

No consumer flies `"shells"`, so nothing live is affected. It does mean:
the fixture table's shells column describes the visual shells, not the
collision hand; and `"shells"` must gain a thumb-pad capsule (and have its
coverage verified the same way as the tube) **before** it can be a guard.
That is now a Slice 2 precondition. Not corrected here — outside this
correction's scope. **Closed 2026-09-14, same day — see "Slice 2: 'shells'
covers the collision pads" below.**

### 5. What this does and does not establish

It establishes that the capsule the footprint guard flies contains the
MJCF's model of the hand, tightly, over the whole supported aperture and
roll range, and that the previously flown-through gap is closed. It does
**not** establish that the guard is adequate: the MJCF hand is a model of
the real gripper (E3 is open), the guard samples the commanded path and not
the realised one (E1 is open and blocked), and the margin is still 0.0.
Passing tests here mean the tube matches the model; they are not evidence
about the robot.

## Slice 2: "shells" covers the collision pads (2026-09-14)

Scope: `kinematics._shells_hand_capsules` only. No consumer switched to
`"shells"` (`panel_executor._footprint_refusal` keeps `hand="tube"`); no
waypoint, speed, margin, or aperture changed; the tube itself, `hand_radius`,
and E1 execution are untouched; E1 is not unblocked. Evidence:
`docs/reviews/probes-2026-09-14-shells-collision-coverage/` and the tests
named below.

### What changed in `kinematics.py`

- **`thumb_pad`**, a new capsule for `r_thumb_col`: same axis convention as
  every other shells capsule (local Z through the box centre — pos
  `(0, 0.005, -0.085)` in the thumb frame — radius the box's XY
  half-diagonal, `hypot(0.014, 0.008)` ≈ 1.61 cm). It cannot be folded into
  the existing `thumb` capsule because the pad sits entirely below the
  visual thumb box's own end cap (item 4 above).
- **`finger`**'s radius widened** from `hypot(0.012, 0.010)` (the visual
  shell alone) to `hypot(0.014, 0.008)` ≈ 1.61 cm, to also contain
  `r_finger_col`. `r_finger_col`'s z-range already falls inside the visual
  finger box's axis span, and both boxes share the same centreline, so the
  tightest single capsule containing both is the *larger* of their own two
  corner distances — not `hypot(pad_x, shell_y)` ≈ 1.72 cm, which would pad
  the bound past what either box needs. No endpoint change.
- `"tube"` and `hand_radius` are untouched; `"shells"` is still opt-in with
  no consumer.

### Verification against the MJCF

`tests/unit/test_arm_geometry_mjcf.py`:

- **`TestShellsCoverMJCFHand`** — the exact analogue of
  `TestTubeCoversMJCFHand` for `"shells"`: every corner of all four hand
  boxes, and the wrist ball's far surface, is inside SOME shells capsule at
  every named waypoint, every `FOOTPRINT_LEGS` sample, and the full
  `r_gripper` x `r_wrist_roll` grid. Measured worst excess **≤ 3.5 × 10⁻¹⁶
  m** (machine precision).
- **`TestShellsCoversThumbAndFingerPads`** (formerly
  `TestShellsMissThumbPad`, now a regression case) — the pad that was 2.05
  cm (1.25 cm, finger shut) outside every shells capsule, and the finger
  pad that was 0.5 mm outside, are both now inside to machine precision.
  Old numbers recorded in the parametrisation.
- **`TestTubeVsShellsCapsules`** — re-derived: the widened `finger` capsule
  moves the surface shortfall by up to ~0.05 cm wherever it is the widest
  capsule (e.g. REST −68.8°: −1.14 → −1.19 cm); centrelines stay
  non-negative throughout, as before.

`tests/unit/test_footprint_boards.py::TestSweepOrdering` (same 224-position
sweep as the tube correction): **shells > reference goes from 62/224 (up to
1.0 cm) to 0/224** (max observed gap now ~5 × 10⁻¹³ m — floating-point noise
at the attained bound). Everything else the sweep measures — tube ≤
reference at 224/224, tube ≤ shells at 224/224, shells tighter than tube by
> 1 cm at 189/224, tube over-conservatism max 12.96 cm — is unchanged: the
pad additions move the finger/thumb radius by ~0.05 cm at most, well under
this sweep's thresholds.

### The fixture table, a third time

`tests/unit/test_footprint_boards.py::TestNamedBoards`, re-run with
`"shells"` now covering the pads, plus the two 4 cm pool objects on r2c3
(the boards the tube correction newly refuses):

| fixture | tube (corrected) | shells (visual only, pre-Slice-2) | shells-with-pads | flyable-by-shells? |
|---|---|---|---|---|
| evidence board | −5.7 cm | +4.5 cm | +4.48 cm | yes (unchanged) |
| incident board | −11.9 cm | −2.4 cm | −2.41 cm | no (unchanged — the only one refused by shells) |
| `foam_block` on r3c3 | −9.4 cm | +3.4 cm | **+2.11 cm** | yes, but tighter — closest to the thumb side of the six |
| `soda_can` on r2c3 | −4.4 cm | +4.5 cm | +4.54 cm | yes (unchanged) |
| `pool_box_1` on r2c3 | −3.7 cm | +6.3 cm | +6.26 cm | yes (unchanged) |
| `pool_cyl_1` on r2c3 | −3.1 cm | +5.9 cm | +5.94 cm | yes (unchanged) |

No row's flyable-by-shells status flips. `foam_block` on r3c3 is the only
one that moves by more than rounding, because it is the board closest to
the thumb side of the six — but it stays comfortably positive.

Because `TestShellsCoverMJCFHand` now proves `"shells"` bounds the whole
MJCF hand (visual shells and both collision pads, not only the visual
shells), the five positive rows above are relabelled **"flyable by shells
geometry"**, dropping the "visual-only" qualifier item 4 above required.
That is a statement about the model hand, not a safety decision: no margin
is set, E1 has not flown, and E3 (real gripper vs. the MJCF) is still open.

### What this does and does not establish

It establishes that `"shells"` is now a correct bound on the model hand —
both the parts that were already covered and the collision pads that were
not — the same standard `TestTubeCoversMJCFHand` holds the tube to. It does
**not** establish that switching a consumer to `"shells"` is safe: that
still needs a margin decided from simulator E1 (readiness landed
2026-09-14; no flight has been made — see "What is still open" below) and
E3's real-gripper measurement. This PR is the geometry
precondition Slice 2 PR2 was waiting on; PR2 itself — switching
`panel_executor._footprint_refusal` to `hand="shells"`, choosing
`FOOTPRINT_MARGIN`, recovering the refused boards — is still out of scope
here.

## What is still open

- **E1 (simulator)** — realised motion and clearance against the native
  MuJoCo backend, reached through the SDK bridge
  (`scripts/measure_route_clearance.py`), reporting per-link realised
  clearance under both hand models at the aperture the (simulated) hand
  actually had. This measures the SIMULATOR'S actuator model tracking the
  commanded path — `native_mujoco/ACTUATOR_MODEL.md`'s own words, "sim-tuned
  for stable position control, not calibrated to Dynamixel datasheets" —
  never the physical robot's. **Any margin simulator E1 yields is
  simulator-specific and provisional**: it says what this simulator's arm
  does on this simulator's actuator model against these simulator object
  dimensions (decision D6), not what the real Reachy would do. The guard is
  never promoted on simulator evidence alone (see the Slice 2 PR2 entry
  below).

  Readiness for this measurement (assignment 2026-09-14, "simulator-E1
  readiness", `outputs/assignment-2026-09-14-sim-e1-readiness.md`) is
  **implemented, offline, no flight made**: six committed board scene YAMLs
  (`scenes/e1_boards/`, `scripts/make_e1_boards.py`); a verified-identity
  check (`scripts/e1_identity.py`) that the recorder runs itself, before
  recording — never trusting the client's own `REACHY_SIM_BACKEND` or any
  other env var — and refuses to record if the actually-connected simulator
  bridge cannot be established (loopback host, exactly one live physics run
  directory, round-by-round joint/timing correlation against that
  directory's own advancing stream, scene-hash identity, and the container
  bridge's `/status`); automatic linking of scene hash, actual object
  poses, joint/aperture samples, and the server's own recorded stream into
  a `<log>.link.json` sidecar (`scripts/link_e1_flight.py`), including a
  5 mm settled-pose check against each board's own YAML pose; and
  arm-link/object contact recording, including contacts that displace
  nothing (`native_mujoco/contact_accumulator.py`, gated behind `--record`).
  None of this proves simulator identity or the SDK protocol can be made
  cryptographically airtight — see `scripts/e1_identity.py`'s own docstring
  for the residual limitation (the real robot's SDK protocol, which the fake
  server mirrors, carries no simulator-identity field; only joint values are
  reachable through it, and joint-value agreement alone is a correlation
  check, not a proof — this is refused-on-ambiguity, not
  proven-by-cryptography).

  **Pilot protocol, not yet run**: a parked (motionless) recording under
  the standing read-only sim approval, then one separately-approved flight
  (`LOWER_TO_REST` × `B4_pool_box_1_r2c3`, the widest shells margin) before
  any wider campaign — see
  `outputs/e1-operator-checklist-2026-09-14.md`. **Still open**: no flight,
  simulator or physical, has been made.

  **The prerequisite this ADR previously recorded — the recorder must log
  the actual gripper aperture — is now implemented** (2026-09-14,
  aperture-logging PR, no live motion): `record_joint_log` streams all
  eight `rig_routes.R_JOINTS` (`ARM7` plus `r_gripper`), the same read-only
  `getattr(...).present_position` access as the other seven, at the same
  instant. The log schema is now `schema_version: 3` (2026-09-14, sim-E1
  readiness — 2 added aperture logging, 3 adds `wall_time_ns` per sample and
  `t0_wall_ns` on the header for aligning a sample to the server's own
  recorded stream); every value is degrees; `t` is elapsed seconds from
  `time.monotonic()`, spacing best-effort (read each sample's own `t`, never
  assume uniform `1/sample_hz`).

  A recorded sample's aperture is validated, not trusted blindly: missing
  raises `ApertureDataError` unless the caller passes
  `allow_missing_aperture=True` **and** declares the log's `schema_version`
  as one eligible for that rescue (today: only `1`, real schema-1 logs —
  schema 2 is a recognised past shape but was never eligible, since it
  always had `r_gripper`) — a `schema_version` claimed to be the current one
  (`3`) with a missing aperture always raises, the flag notwithstanding,
  because the current schema has no excuse for a gap; the same schema also
  always raises `SampleTimestampError` for a missing `wall_time_ns`. An
  unrecognised `schema_version` (neither current nor a supported legacy
  one) raises `UnsupportedSchemaVersionError` outright, before any
  per-sample check. Every such fallback sample is named in the report
  (`realised_aperture_assumed_samples`) rather than silently treated as
  open; a present-but-invalid reading (non-numeric, non-finite, or outside
  the MJCF's commanded range) always raises, regardless of the flag or
  declared schema — `main()` itself now validates a fresh recording before
  saving or reporting it, so a bad SDK read is caught at record time.

  A recording that fails this validation is not discarded: `main()` saves
  it via `save_invalid_log` (filename ending `_INVALID.json`, body carrying
  `"valid": false`, the failure reason, and every sample actually
  recorded, offending one included) so the flight can be diagnosed instead
  of re-flown blind — and the file is stamped with a `schema_version` this
  module never recognises, so reading it back the normal way always
  refuses it rather than risking it pass for real E1 data.
  `report()`'s result also now names, in plain text, where each half's
  aperture came from (`realised_aperture_source`,
  `planned_aperture_policy` — the latter being the per-leg worst-case
  commanded *endpoint* aperture, applied uniformly along that leg, not a
  per-sample commanded value). `hand_radius` and the tube are untouched by
  this; only the recorder's own schema, validation, and reporting math
  changed.

  What is **not** done by this: no flight, simulator or physical, has been
  made.
- **Physical validation** — E2 and E3 (below), plus a physical re-flight of
  the tabletop legs that nobody has scheduled. The guard is not promoted on
  simulator E1 evidence alone: a candidate margin from simulator E1 says
  what the simulator's actuator model does, and physical validation is what
  says whether the real robot agrees.
- **E2** — the same instrumentation on the SWING_1 rail crossing.
- **E3** — real gripper envelope vs. the MJCF shells (tape measure on the
  physical Reachy 1.2): whether `"shells"` is a bound on the *real* hand,
  not only the model's. Motion-free, like E1 readiness — a caliper and a
  ruler, not a flight.
- ~~**`"shells"` must cover the collision pads** before it can be
  promoted~~ — **done 2026-09-14** (Slice 2 PR1): `thumb_pad` and a widened
  `finger` capsule, verified the way `TestTubeCoversMJCFHand` verifies the
  tube (`TestShellsCoverMJCFHand`). The fixture table's shells column is
  now a statement about the collision hand, not only the visual shells.
- **Slice 2 PR2 — recovering the corrected tube's over-conservatism.** The
  tube still refuses `pool_box_1`/`pool_cyl_1` on r2c3 and 15/224 cube
  positions whose true clearance is +3.3 cm or better; the geometry
  precondition for recovering them (a `"shells"` that covers the pads) is
  now met. Recovering them still means switching the tabletop guard to
  `"shells"`, with a margin decided from E1 — not shrinking the tube — and
  `"shells"` becomes a guard **only** once that margin exists; coverage
  alone is not a promotion decision. Waits on simulator E1 and E3.
- The 1.6 cm wrist-roll term is now inside the radius; a tighter treatment
  (anchoring the tube at the thumb pivot) is possible but would change the
  capsule's endpoints, which every consumer reads, and is not worth it while
  the plan is to move to `"shells"`.

## Consequences

- The footprint guard's tube now contains the MJCF hand at every pose in
  range; it is ~4 cm larger on the REST leg than before, and refuses the
  two 4 cm pool objects on r2c3 and a corner strip of the board that it
  used to allow, all with positive true clearance (item 3 above).
- A route or ability wanting `"shells"` clearance can ask for it, and
  `"shells"` now covers the collision pads (Slice 2 PR1); nothing asks for
  it yet, and nothing should until a margin is decided from E1.
- `docs/adr/0002`'s revisit condition on the capsule-vs-physics disagreement
  is met: the disagreement is the hand tube, explained and quantified, not
  an open question.
- The planning-time question the review raised (`maximise_clearance=True`
  timing, before `plan_segment` could default to it) is unmeasured and not
  part of this slice.

## Revisit conditions

Revisit this ADR when:

- simulator E1 delivers Δ_sim per leg family (realised-vs-planned clearance,
  both hand models, labelled simulator-specific and provisional) and a
  margin decision is made for Slice 2 PR2 (switching the tabletop guard to
  `"shells"`) — **`"shells"` becomes a guard only once that margin is
  decided, and only after physical validation (E3, and the physical
  re-flight nobody has scheduled) agrees with the simulator; coverage
  (done) is a precondition, not a promotion**;
- the guard moves to `"shells"` and the tube stops being what any consumer
  flies;
- E3 measures the real gripper against the MJCF shells.
