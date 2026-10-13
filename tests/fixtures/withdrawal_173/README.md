# Phase 1.5 release states (issue #173)

These are the ten Phase 1.5 re-measure lifts (2026-10-09, red_cube at r2c2, no-slip 10), reduced by
`python -m tools.withdrawal_replay --fixtures`. Each one holds:

- the preflight's object pose;
- the crane's corrections, its grasp-depth MEASURE, and its SUPPORTED, RELEASED and WITHDRAWAL events;
- the state just before the hand opened (`release_start`);
- the state just after it opened (`post_release`);
- what was recorded (whether the withdrawal completed, the release drag, and the judged way out).

Withdrawals refused live: A1, A4, B4, B5. Withdrawals completed live: A2, A3, A5, B1, B2, B3.

Source: `~/Reachy-Lab/outputs/sim/working/trial-2026-10-09-phase15-remeasure/run{A,B}/lift-N`.
