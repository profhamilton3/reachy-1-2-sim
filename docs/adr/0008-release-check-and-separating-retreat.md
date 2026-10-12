# ADR-0008: The crane checks the way out before it opens, and may separate before it climbs

- Status: Accepted. Covers #173. The option was chosen offline by the owner's
  rule: 0 of 10 refusals and zero residual pad contact first, then the
  smallest drag. Release drag is reported against a 5 mm target and does not
  block. Simulator only.
- Date: 2026-10-12
- Relates to: #173, #172 (ADR-0007: the offline world, the contact rules and
  the reported withdrawal), #55 (the crane lift)

## Context

Four of the ten Phase 1.5 lifts refused their withdrawal after the release:
A1, A4, B4 and B5.

The offline rebuild of those release states shows why:
- **At the grasp, the pads are squeezed into the cube's collision box.** The
  thumb sits about 4–6 mm deep and the finger 9–13 mm, at about 250–270 N.
- **Opening carries the cube.** As the finger opens, the thumb pushes the cube
  after it. The cube travels about 5–12 mm until the finger lets go, near
  −54° to −63°.
- **The cube can come to rest against the finger.** The open hand leaves only
  about 2.8 mm per side.
- **The withdrawal then fails at its first sample.** That sample is the pose
  the hand is already in, which the existing sideways correction of the climb
  cannot move.

Relaxing the grip first, or lifting 2 mm before opening, did not reduce the
drag. Opening while moving the hand along the jaw axis did cut the drag to
2–4 mm, but it needed a 7–10 mm shift, which is past the 6 mm cap, and the
cube still came to rest on the finger.

## Decision: O1 + O2

**O2: check before opening.**
1. `primitives.release_grip` opens only until the grip force reads zero.
2. From the measured state, the way out is judged for the opened hand. The
   wider opening (−68.0°, 0.75° inside the −68.75° travel) is tried before the
   usual −65°. A way out that needs no separating step is preferred.
3. If no opening has a way out, the hand opens no further and the withdrawal
   is refused.
4. After opening, the state is judged again. If a pad is still under the
   allowance, the hand opens to −68°.

**O1: a separating retreat.** `plan_way_out` tries three ways out, in order:
1. straight up the ladder;
2. the existing bounded sideways correction;
3. a separating first leg along the jaw's closing axis, which equalises the
   two pads' start clearances and is capped at 6 mm.

The separating leg may start under the 0.494 mm allowance, but only if:
- no pad starts more than 0.5 mm deep in the object;
- the pad that starts under the allowance gets strictly further away until it
  clears;
- the other pad stays clear;
- every sample of the climb after the leg clears the allowance.

Correcting and separating legs fly with the monitor off, as the existing
correction already did. The climb flies with it on.

## Evidence

The full table is in `Reachy-Lab outputs/sim/working/issue173-replay-<date>.md`.
The integration test `tests/integration/test_withdrawal_after_release.py`
pins the 10 lifts.

| option | refusals (of 10) | lifts with residual contact |
|---|---|---|
| base | 6 | 5 |
| O1 | 0 | 5 |
| O2 | 2 | 2 |
| O1+O2 | 0 | 2 |

- The rebuild is strict. The base option refuses 6 offline where the live
  run refused 4, so the offline release is harsher than the live one.
- O1+O2's two residual contacts are A1 (0.06 mm) and A4 (0.001 mm). The
  separating leg lifts the pad off.
- The new withdrawal also clears all 10 recorded live release states.

Live check (2026-10-12): 10 lifts in one panel worker, with a simulator reset
between lifts. All 10 were lift pass, withdrawal completed and full cycle.
- Ways out: 8 straight up, 1 sideways, 1 separating.
- Release drag: 5.3–10.4 mm, mean 8.1 mm. Phase 1.5 had 4.2–12.2 mm, mean
  7.9 mm.
- Every run record reads `withdrawal: completed`.

## Consequences

- Release drag is unchanged at about 7–12 mm offline and 4–12 mm live. It is
  now reported in the run record, the operator's sentence and the #172
  evaluator, against the 5 mm target, and it does not block. Reducing it needs
  a coordinated release beyond today's 6 mm correction cap; that is a separate
  decision.
- The −68° opening is a simulator figure. The physical gripper's travel must
  be checked before this is ported.
