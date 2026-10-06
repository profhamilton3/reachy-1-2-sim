# ADR-0004: The pick/place arc is preflighted against the whole arm

- Status: Accepted (implementation; simulator-only, unflown)
- Date: 2026-10-06
- Decision owners: IITG Reachy 1.2 simulation project
- Relates to: #56 (addressed, not closed), #55 (deferred items below), #74;
  ADR-0002, ADR-0003

**This change addresses #56; it does not close it.** It refuses unsafe planned
arcs. It does **not** choose clearance-optimal trajectories, and it makes no
claim of grasp reliability or validated physical safety. Nothing was flown.

## Historical context

The side-arm raise was a crude early solution from the initial table scene,
which had no rails: the SIDE_HIGH hub and the roll -88 sweep. The TLH motion
notebook later supplied the rail-pocket routes: `PLACE_ROUTE` +
`LIFT_TO_PRESENT` = `RAISE_TO_SIDE`, and `STOW_ROUTE` = `STOW_FROM_SIDE`.
Despite their legacy names, `P.raise_to_side` and `P.stow_from_side` now fly
those notebook routes. `P.SIDE_HIGH` survives only as a legacy constant,
numerically equal to PRESENT, and the arc does not use it. Historical use of
the earlier raise is **not** a justification for anything in current motion.

**Priority path.** The priority path is the `FWDCenterLabSivaPool` scene under
MuJoCo, through the panel worker. Its RAISE_TO_SIDE and STOW_FROM_SIDE routes
are recorded as validated for that scene in `rig_routes.ROUTE_COMPATIBILITY`
(2026-09-10), and its default board keeps every manipulable in the pool, so
neither footprint check blocks anything there.

**The demo and the notebook are historical workflows.**
`scripts/demo_pick_place.py` and `notebooks/pick_and_place_training.ipynb` are
historical workflows on `tabletop_demo.yaml`. Both now refuse their default
jobs before any motion: the RAISE_TO_SIDE footprint alone refuses the job
(red_cube, hand, -6.8 cm), and the arcs are refused individually. The outputs
retained in the notebook (code cells showing "episode completed", the task
spec, and so on) describe earlier execution, not the current code. Modernizing
them is separate work; the board was not and will not be re-laid in #56.

## Context

Pick/place planned with the **pad point** only. `pick_place_live.run_segment`
called `CartesianPlanner.plan_segment`, which called `check_collisions`, which
asks only whether the pad point goes below the tabletop surface or inside a
static box. Nothing along a pick/place segment asked about the forearm, the
upper arm, the hand, or any manipulable object. Pointing, escort and the
panel's footprint guard already used the whole-arm model (`link_capsules`,
`path_clearances`); pick/place did not.

Three further defects sat in the same arc:

- **The last step could never work.** The arc ended with `P.raise_to_side(arm)`.
  Since `eb64b83` (2026-09-10) `raise_to_side` requires the arm at HOME
  (`primitives.py`), and at that point the arm is above the place site. The
  worker returned `failed` without stowing ("`r_elbow_pitch` is 100 deg off").
  The same pattern was in `notebooks/pick_and_place_training.ipynb`.
- **No footprint guard.** `PanelExecutor.available()` returns early for
  `pick_place`, so the RAISE_TO_SIDE and STOW_FROM_SIDE legs the job flies were
  never checked against the board.
- **The notebook planned segment by segment**, interleaving an unchecked jaw
  twist (`r_forearm_yaw` to 90 deg) and the broken return.

## Decision

### The rule (ADR-0003, now enforced in code)

- **Objects and rig fixtures** are checked against the **whole arm**:
  `link_capsules(q, "right", gripper_deg, hand="tube")` (upper arm, forearm,
  hand), against `scene.obstacle_ids(include_static=True)` -- the manipulable
  objects plus the rig rails, **not** the tabletop.
- **The tabletop** keeps the existing pad-point rule (`check_collisions` ->
  `SceneModel.validate_path`, `surface_margin=0.005`). That rule covers the
  tabletop surface and static boxes only. **It is not a clearance model**, and
  the arm against the tabletop is still covered by it alone (R7).
- **Tube model, 0 cm margin** (`WHOLE_ARM_MARGIN_M`, pinned equal to
  `rig_routes.FOOTPRINT_MARGIN`). Every call site passes `hand="tube"`
  explicitly and every refusal reports the model. Nothing switches to
  `"shells"`, relaxes the margin, or executes the safe prefix.

### A default, not a flag

`plan_segment` runs `CartesianPlanner.check_arm_path` **whenever there is a
scene**. There is no switch to turn it off. With no scene nothing is checked,
as with every other clearance method.

### Explicit IK policy

`plan_segment(..., ik=IKPolicy.X)` has **no default**: omitting `ik` is a
`TypeError`, and a non-member is a `ValueError`.

- `IKPolicy.FAST` reproduces the old trajectories bit for bit (warm start,
  orientation carried point to point). **Pick/place selects FAST.**
- `IKPolicy.MAX_CLEARANCE` calls the existing `solve(maximise_clearance=True,
  from_joints=q, gripper_deg=...)` per point. **Nothing selects it, and nothing
  selects it automatically.** Measured offline on the demo arc (host Python
  3.14, a copy of the fake server's scipy IK; not the container, gRPC or
  physics): about 37x slower (135 s and 11 070 IK calls against 3.7 s and 409)
  with joint steps up to 86 deg mid-segment, for an improvement of only
  0.4-1.0 cm on the legs that failed.

**Automatic replanning is deferred.** This change refuses unsafe planned arcs;
it does **not** choose clearance-optimal trajectories.

### Sampled, not swept

`check_arm_path` is **sampled** at <= 2 deg of joint spacing along the commanded
joint sequence (`ARM_PATH_MAX_SAMPLE_STEP_DEG`, `n = max(2, ceil(max|dq| / 2)
+ 1)` per consecutive pair). It is **geometry only**: the realised path's
deviation from the commanded one is not covered. That belongs to E1 and the
continuous-monitoring roadmap, and continuous swept checking is deferred.

### The target object: an interim, accepted contact exemption

For this simulator implementation, the target object may overlap the **hand
capsule only** during descent to hover, grasp, carry, and release/retract
(segments 2-7 of 8). The target stays checked against the upper arm and forearm
throughout, and every non-target obstacle stays checked throughout. While
carried, the target's centre follows the planned pad point
(`TargetContact(carried=True)`); after release it is held at the place pose.

**This does not protect against unintended hand-target contact.** The tube
cannot tell the pads from the back of the hand. Tracking the carried target is
how it stays checked against the forearm and upper arm; it is **not**
protection of the carried object against bystanders (see "Deferred to #55").

### Plan, then execute -- and the return

`plan_pick_and_place` plans all eight segments (swing in, descend to hover,
descend to grasp, lift, carry, descend to place, retract, **return to the raised
pose**), sending no motion command, and raises before anything moves if any part
is refused. `execute_arc` flies the plan.

- **Start and end posture are `rig_routes.PRESENT`**, not `P.SIDE_HIGH`
  (PRESENT and SIDE_HIGH are not interchangeable: `stow_from_side` requires
  PRESENT). Every check uses the OPEN aperture (`R.OPEN`, pinned equal to
  `P._GRIPPER_OPEN_DEG`).
- **The return** is a joint-space line from the end of the retract to PRESENT,
  planned and checked with the rest of the arc, and flown with **no
  `converge`, no `smooth_move`, and no `raise_to_side`**. Its last pose equals
  PRESENT exactly.
- **The return guarantee.** Commanding a checked endpoint does not prove the
  arm follows the checked path while converging. So the arm's arrival is tested
  with `P.stow_entry_ok` (the exact posture requirement of the stow: PRESENT on
  the placing joints within 12 deg). If it has not arrived, at most
  `RETURN_CORRECTIONS_MAX = 2` corrective sequences are made, **each checked
  with `check_arm_path` before it is commanded**. If a correction is refused,
  or the attempts run out, `ReturnArrivalError` is raised and nothing else is
  commanded. **The caller must not stow.** There is no general recovery
  planner and no unchecked recovery.
- **The `eb64b83` regression** is pinned: from the post-retract pose
  `raise_to_side` raises (HOME required), `execute_arc` never calls it, and
  `stow_entry_ok` accepts the arm at the plan's end.

### One job-level preflight

`preflight_pick_place(planner, scene, moves, skip_refused=...)` runs, before any
motion: the RAISE_TO_SIDE footprint against the current board; each move's
complete arc on a **copy** of the model updated with each preceding accepted
arc's placed pose; and the STOW_FROM_SIDE footprint against the predicted final
board. A refusal raises `PreflightRefused` (segment, object, link, obstacle,
clearance, margin, model). The worker (`skip_refused=False`), the demo
(`skip_refused=True`) and the notebook all use it, so none of them executes any
of the job before the whole job has been checked.

**What the footprint checks cover, exactly** (`motion/footprint.py`, extracted
from `PanelExecutor._footprint_refusal`, whose text and behaviour are
unchanged):

- Only the legs listed in `rig_routes.FOOTPRINT_LEGS` for each route:
  RAISE_TO_SIDE is HOVER -> REST_SHUT -> REST, and STOW_FROM_SIDE is
  PRESENT -> REST_SHUT -> HOVER.
- Manipulable objects only (`path_clearances` with its default ids): no rig
  fixtures.
- The per-leg wider commanded aperture.
- The rest of each measured route is covered by the route's own rig
  validation, not by this live-object check.

### Compensations: moved, not re-derived

`CLEAR_Z = 1.00`, `CARRY_HZ = 8` and `STEP_HZ = 25` moved to
`reachy_ai/motion/transit.py` with **values unchanged**; `pick_place_live`
re-exports `CLEAR_Z`. `CLEAR_Z` is the transit height every arc inherits, and
lowering it is a behaviour change that needs a simulator measurement nobody has
authorised. `CARRY_HZ` is a tracking compensation (#55 / E1 territory), not
about planner blindness. **#56's "move the compensations" item is not
complete.**

## Offline evidence

Probes in `~/Reachy-Lab/outputs/sim/working/probes-2026-10-06-issue56/` (host
Python 3.14 against a copy of the fake server's scipy IK; not the container,
gRPC or physics -- indicative only):

- The demo board's red_cube move is refused under the tube model (hand against
  blue_cylinder, -0.7 cm on descend to hover and -5.5 cm on retract in the
  prototype), and on `tabletop_demo.yaml` the **RAISE_TO_SIDE footprint alone
  refuses the whole job** (red_cube, hand, -6.8 cm). **The demo board is
  unchanged.**
- Single-object `FWDCenterLabMCC` boards: red_cube and soda_can r2c2 -> r2c3
  and r3c3 -> r2c2 pass the arc (worst +3.6 to +5.2 cm); picks from r1c1 are
  refused (the hand meets `rig_rail_inner_left`). Placing at r2c3 or r3c3 is
  refused at the STOW_FROM_SIDE footprint.

## #56 status

| #56 item | Status |
|---|---|
| Whole-arm check of the pick/place arc | **Done here** (default on, tube, 0 cm, sampled <= 2 deg) |
| Explicit IK policy, no hidden fallback | **Done here** (FAST selected; MAX_CLEARANCE opt-in, unused) |
| Return to the raised pose repaired | **Done here** (checked, PRESENT, arrival tested, <= 2 checked corrections) |
| Worker, demo and notebook preflight the complete arc | **Done here** |
| Footprint checks for the pick/place job | **Done here** (scoped as stated above) |
| Move the compensations (CLEAR_Z, 8 Hz) | **Not complete** -- moved, not re-derived |
| Automatic replanning / clearance-optimal trajectories | **Deferred** |
| Swept (continuous) checking; realised-path deviation | **Deferred** (E1, continuous-protection roadmap) |
| Switch to the `"shells"` hand model | **Not done** (waits on E1 and physical validation) |
| Demo and notebook | **Historical workflows**, now refused before any motion; modernizing them is separate work (the board is not re-laid in #56) |

## Deferred to #55

- **Contact-surface modelling** -- which hand surfaces may touch the target.
  The tube cannot tell the pads from the back of the hand.
- **Carried-object collision protection** -- the held object against
  bystanders.
- **Unintended hand-target contact** during the exemption window above.
- **Jaw orientation** -- the removed notebook twist (`r_forearm_yaw` to 90 deg),
  an unchecked move outside any planned arc.

## Consequences and residuals

- **F1 (HIGH, accepted).** The demo's red_cube move is refused, and on
  `tabletop_demo.yaml` the RAISE footprint refuses the whole job, so the demo
  and the notebook move nothing by default. They are historical workflows;
  modernizing them is separate work and the board is not re-laid in #56.
- **F2 (MEDIUM, accepted).** The tube reads up to ~4 cm too close; rail-side
  picks such as r1c1 are refused. The shells path waits on E1 and physical
  validation.
- **F3 (MEDIUM, accepted interim).** The hand tube is exempt against the target
  from hover to retract.
- **F4 (MEDIUM, unverified physics).** The return streams all joints on a joint
  line. Its behaviour under MuJoCo physics is unverified. Arrival is tested
  immediately after the last setpoint, with at most 2 checked corrections, so a
  lag can produce a reported failed arrival that is safe but possibly spurious.
  A failed arrival is never followed by an unchecked move. An owner-authorised
  simulator trial is needed.
- **F6 (LOW).** Pick/place now runs the footprint checks abilities already run,
  so some boards will be refused for pick/place too.
- **R1** the tube approximates the gripper surfaces; **R2** the return's
  physics is unverified; **R3** sampled, not swept; **R4** no re-validation
  during execution (the arc is planned from nominal PRESENT and not re-checked
  against the realised start); **R5** the carried object against bystanders is
  not modelled (#55); **R6** a cancel before the close leaves the arm mid-arc,
  as before; **R7** the arm against the tabletop is still the pad-point rule
  only.

## Revisit conditions

- E1 delivers the realised-versus-planned deviation and a margin is decided
  (the margin stays 0 cm until then).
- #55 models contact surfaces, at which point the hand-target exemption is
  replaced.
- An owner decision on replanning, or on modernizing the demo and notebook.
