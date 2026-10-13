# ADR-0007: What a lift's success means, the planned route type, and one contact rule table

- Status: Accepted. #172 decisions D5, D7 and option C; the owner's rulings of
  2026-10-10 on the approach, the withdrawal contacts and PRESENT. Simulator only.
- Date: 2026-10-10
- Decision owners: IITG Reachy 1.2 simulation project
- Relates to: #172, #55 (the crane lift), #91 (the panel evaluators), #173
  (withdrawal), #174 (placement, which reuses the table unchanged)

## Context

`lift_object` (route `CRANE_LIFT`) was the only flown panel ability with no
evaluator. That gap was the last failing test on `main`. The crane route is
planned per job, so it cannot be compared with a measured step list.

Before this change, the offline `FWDCenterLabSivaPool` world had no table and
no rails, because `SimulationCore._load_world` ignored `extends:`. With them
added, rest and stow lay the forearm on the board, and a lift's pads close on
a cube that stands on the table. A per-route list of "intended bodies" cannot
say *when* a contact is allowed.

## Decision

1. **Route types (D7).**
   - A `MEASURED` route is a fixed waypoint list. Its integrity is the step
     list.
   - A `PLANNED` route (`CRANE_LIFT`) is solved per job. Its integrity is
     "preflight passed, plan hash recorded" (`check_plan_integrity`). No
     recipe may vary it.
   - The baseline-recipe tests cover measured routes only.
2. **`evaluate_lift_object` (D5).** A lift passes when:
   - the object rises ≥ 4.0 cm (the least rise over the hold);
   - it slips ≤ 2 mm in the hand's frame over the hold;
   - it is put back ≤ 5 mm (xy) from where it was picked up;
   - no contact is unintended;
   - no other object drifts past `OBJECT_DRIFT_TOL`;
   - the arm ends at PRESENT.
3. **The withdrawal is reported, not judged.** It is reported as `completed`
   or `refused`.
   - Ending at PRESENT is part of completing the withdrawal. After a refusal,
     the lift verdict does not check it.
   - `full_cycle` means all of these hold: the lift passed, the withdrawal
     completed, the withdrawal left no unintended contact, and the arm is at
     PRESENT.
4. **Contact rules, option C (`evaluation/contact_rules.py`).** One table of
   roles × phases, shared by every ability.
   - Roles: pad, arm link, target, support (the table and board), rig, other
     object.
   - Phases: approach, grasp, hold, carry, set-down, release, withdraw. The
     abilities tag their own phases; the crane maps its existing segments.
   - Rules:
     - pads on the target are allowed from grasp to release;
     - the target on its support is allowed except while held (hold, carry);
     - an arm link on the board is allowed only at a declared rest pose,
       through a per-route exception;
     - everything else involving the robot or the target is unintended.
   - Exceptions: forearm on the board at REST, for `PLACE_ROUTE` and
     `STOW_ROUTE`.
5. **What the lift's contact verdict covers (owner, 2026-10-10).** It runs
   from the crane's ATTEMPT_START to the release.
   - The approach to the start posture (from HOME: `RAISE_TO_SIDE`) is judged
     under its own route's rules and exceptions and reported. It is not
     skipped, and the lift gets no exception for it.
   - Pad-on-target contact after the release is judged in the withdraw phase
     and reported with the withdrawal, for example
     "refused, residual pad contact 0.20 mm". #173 must drive it to zero.
6. **Run records.** Each panel episode records:
   - `command_poll_fixed`;
   - the scene, model and code hashes (`code.source_sha256`, because the
     container has no `.git`);
   - `seed` (no reset seed; the planner's start set is fixed);
   - `route_version`;
   - the sim steps at either end;
   - every object's measured displacement.

   A lift also records its plan record and contact model, and keeps the
   crane's phase events.

## Validation (offline, no live simulator)

`scripts/replay_lift_evaluation.py` poses each recorded state in the offline
world kinematically and judges the lift. The inputs are the ten Phase 1.5
lifts and the Phase 1.1 diagnostic run.

| observed | count | verdict |
|---|---|---|
| completed full cycles | 6 | 6 full-cycle pass |
| withdrawal refused | 4 + diag | lift pass, withdrawal refused |

The re-measured rise, slip and put-back agree with the crane's own figures.

Flown offline, the measured panel routes make no unintended contact.
Pointing cannot be flown offline.

## Consequences

- `RAISE_TO_SIDE` passes REST with the forearm on the board and has no
  exception. That contact is reported as unintended on the approach leg. It
  is outside the lift verdict.
- A live lift record has no contacts: the server streams them only when it
  records. Live lifts are graded offline from their states, and the
  evaluator refuses to clear a lift with no phase-tagged contacts.
