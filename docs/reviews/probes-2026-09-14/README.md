# Offline geometry probes, 2026-09-14 — the tube correction

Evidence for ADR-0003's "Correcting the tube" section and PR #108. All
probes: `mj_forward` only on the MuJoCo scene model compiled from
`scenes/FWDCenterLabSivaPool.yaml`. Nothing stepped, no simulator process, no
SDK, no motion. Run from the repo root, on the commit that carries them:

    PYTHONPATH=src:native_mujoco python3 docs/reviews/probes-2026-09-14/<probe>.py

| file | what | output |
|---|---|---|
| `tube_coverage.py` | corrected tube vs every MJCF hand-box corner over the full `r_gripper` × `r_wrist_roll` grid (excess and tightness); the 2026-09-14 review's demonstrated misses, now; shells-capsule axis margins; the §2 fixture table under corrected / legacy tube / shells | `tube_coverage.txt` |
| `newly_refused.py` | boards the corrected tube refuses that the legacy tube allowed: (A) six scene objects × nine cells × every guarded route; (B) 6 cm cube on the 224-position grid, worst over any guarded route, with the true MJCF clearance at each flipped position | `newly_refused.txt` (raw `.json` regenerated, not committed) |
| `sweep_ordering.py` | the review's PLACE_ROUTE-tail sweep with the collision pads added to the reference: tube ≤ reference, tube ≤ shells, shells > reference (the `r_thumb_col` gap), over-conservatism | `sweep_ordering.txt` |

The same numbers are pinned by `tests/unit/test_arm_geometry_mjcf.py`
(coverage, regressions, shells gap) and `tests/unit/test_footprint_boards.py`
(fixture table, newly refused boards, sweep). "Legacy" throughout means the
aperture-only, centre-swung `hand_radius` the guard flew before this
correction; it survives only as `_legacy_hand_radius` in the tests and probes.
