# Offline geometry probes, 2026-09-12

All probes: `mj_forward` only on the MuJoCo scene model that
`native_mujoco/server.py` compiles from `scenes/FWDCenterLabSivaPool.yaml`.
Nothing stepped, no simulator process, no SDK, no motion. Host Python 3.14.0,
mujoco 3.11.0, numpy 2.3.5. Originally run from a `git archive` export of
`fix/bounded-review-followups-56-74` at 61fc1be (before the "shells" hand
mode existed in `kinematics.py`; `probe_hand_candidates.py` builds its own).
From the repo root:

    PYTHONPATH=src:native_mujoco python3 docs/reviews/probes-2026-09-12/<probe>.py .

Each probe writes its `.json` next to itself; those are regenerated output,
gitignored here, and not committed — the `.txt` summaries are.

| file | what | output |
|---|---|---|
| `geomdist.py` | sampled closed-form geom distances (mj_geomDistance returned 0.0 on box-box pairs at isolated samples; see review R9) | library |
| `probe_evidence_board.py` | ADR-0002's board (soda r1c1, foam r2c3) on every FOOTPRINT_LEGS leg + REST->PRESENT + the SWING_1 rail legs; capsule (open / A2 / interpolated) vs MJCF collision geoms vs MJCF shells, 13 and 101 samples | `probe_evidence_board.txt/.json` |
| `probe_board_sweep.py` | 4 cm grid of standing 6 cm / 12 cm boxes over the board; descent (A1), lift (A3), checked tail; capsule vs shells; A1/A3 hazard sets | `probe_board_sweep.txt/.json` |
| `probe_hand_candidates.py` | candidate hand models (tube / one capsule per shell / MJCF shells) on evidence, incident, r3c3, r1c3, soda@r2c3 boards and a 224-position sweep | `probe_hand_candidates.txt/.json` |
| (inline, recorded) | WAVE legs vs a 12 cm block over the board | `probe_wave_legs.txt` |
| (not committed) | full offline suite on the 61fc1be export | 1804 passed |

Key numbers are quoted in `../2026-09-12-repair-review-and-56-74-design.md`
§3.2-3.5 and §4, and the fixture table is reproduced by
`tests/unit/test_footprint_boards.py` (ADR-0003 §2).
