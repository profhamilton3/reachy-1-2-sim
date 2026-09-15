# Offline geometry probes, 2026-09-14 — "shells" covers the MJCF collision pads

Evidence for ADR-0003's Slice 2 precondition and the PR that adds
`thumb_pad` (for `r_thumb_col`) and widens `finger` (for `r_finger_col`) in
`kinematics._shells_hand_capsules`. All probes: `mj_forward` only on the
MuJoCo scene model compiled from `scenes/FWDCenterLabSivaPool.yaml`.
Nothing stepped, no simulator process, no SDK, no motion. Run from the repo
root, on the commit that carries them:

    PYTHONPATH=src:native_mujoco python3 docs/reviews/probes-2026-09-14-shells-collision-coverage/<probe>.py

| file | what | output |
|---|---|---|
| `shells_coverage.py` | "shells" vs every MJCF hand-box corner (both visual shells and both collision pads) and the wrist ball, over named waypoints + `FOOTPRINT_LEGS` samples and the full `r_gripper` x `r_wrist_roll` grid; the thumb/finger pad regression case (old miss vs. now); `TestTubeVsShellsCapsules`'s re-derived surface/centreline margins; the fixture table (corrected tube / shells-with-pads) for the four named boards plus the two 4 cm pool objects on r2c3 | `shells_coverage.txt` |
| `sweep_ordering_rerun.py` | re-run of `probes-2026-09-14/sweep_ordering.py`'s 224-position PLACE_ROUTE-tail sweep, calling `test_footprint_boards.py`'s own `_run_sweep` directly so the numbers match the test exactly: shells > reference goes from 62/224 (up to 1.0 cm) to 0/224 (machine precision); everything else unchanged | `sweep_ordering_rerun.txt` |

The same numbers are pinned by `tests/unit/test_arm_geometry_mjcf.py`
(`TestShellsCoverMJCFHand`, `TestShellsCoversThumbAndFingerPads`,
`TestTubeVsShellsCapsules`) and `tests/unit/test_footprint_boards.py`
(`TestNamedBoards`, `TestSweepOrdering`).

## What changed and why

`r_thumb_col` (the MJCF's thumb collision pad) sits entirely below the
visual thumb box — no widening of the existing `thumb` capsule can reach it
without extending past the shell's own end cap — so it gets its own
capsule, `thumb_pad`, same axis convention as every other shells capsule
(local Z through the box centre, radius the box's XY half-diagonal).

`r_finger_col`'s z-range already falls inside the visual finger box's axis
span; it is 2 mm wider in X and 2 mm narrower in Y than the shell. Both
boxes share the same centreline, so the tightest single capsule containing
both is the *larger* of their own two corner distances —
`hypot(0.014, 0.008)` ≈ 1.61 cm, vs. the shell's own `hypot(0.012, 0.010)`
≈ 1.56 cm — not `hypot(pad_x, shell_y)` ≈ 1.72 cm, which would pad the
bound past what either box needs. `finger`'s radius is widened to the
former; no new capsule or endpoint change.

## Numbers this PR re-derives (see `shells_coverage.txt` / `sweep_ordering_rerun.txt`)

- **Coverage**: worst excess of any hand-box corner (or the wrist ball's far
  surface) outside every shells capsule, over named waypoints,
  `FOOTPRINT_LEGS` samples, and the full `r_gripper` x `r_wrist_roll` grid:
  **≤ 3.5e-16 m** (machine precision; was up to 2.05 cm before the pads).
- **Sweep**: `shells > reference` **0/224** (was 62/224, up to 1.0 cm).
  Everything else the sweep measures (`tube > reference`, `tube > shells`,
  "shells tighter than tube by > 1 cm" at 189/224, tube over-conservatism
  max 12.96 cm) is **unchanged** — the pad additions move the finger/thumb
  radius by ~0.05 cm at most, well under this sweep's thresholds.
- **`TestTubeVsShellsCapsules`** surface margins move by up to ~0.05 cm
  wherever the widened `finger` capsule sets the widest one (e.g. REST
  −68.8°: −1.14 → −1.19 cm); centrelines stay non-negative throughout.
- **Fixture table** (`TestNamedBoards`): `evidence_board`, `soda_can_on_r2c3`
  and the two pool objects on r2c3 are unchanged to displayed precision;
  `foam_on_r3c3` drops from +3.4 to +2.1 cm (that board sits closer to the
  thumb side than the others) but stays positive. `incident_board` remains
  the only one of the six that shells still refuses. No row's flyable/
  refused status flips.

## What this does not establish

Coverage of the *model* hand, not the real gripper (E3 is still open), and
not a decision to fly "shells" — no consumer is switched by this PR
(`panel_executor._footprint_refusal` keeps `hand="tube"`); that is Slice 2
PR2's job, after E1.
