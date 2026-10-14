# Taught board poses vs the sim's arm model (#185)

**Date:** 2026-10-14. **Offline only:** no robot time and no live sim.

**Source:** `profhamilton3/reachy-tabletop-ai` @ `f207378`, `tests/CalibrateBoard.ipynb`. Its SHA-256, the cell indices and the poses are in `taught_poses.json`.

**Sim:** `main` @ `045d508`: `native_mujoco/model/reachy.urdf` and `reachy_1_2.xml`, `fake_reachy_server.py`, `scenes/FWDCenterLabMCC.yaml` (SivaPool's grid is identical).

## Perspective

Siva taught these poses by hand, for his own lab and objects. They are not calculated moves, and his lab is not the MCC table that SivaPool inherits. So they test **frame conventions and reach**, not the sim's board placement. The sim's scenes, gripper range and physics are unchanged by this review.

The notebook's gripper values are Siva's chosen settings for his objects, not limits: open −29.79°, grasp −18.94°, closed +12.08°. The sim's range, −68.8° to +20.05°, is the URDF limit (−1.2 to 0.35 rad).

## Findings

1. **Frame and tool point: verified.** Real `r_arm.forward_kinematics()` is the URDF chain `torso` → `right_tip`, with URDF joint signs. Running the notebook's `B2_SAFE_JOINTS` through that chain reproduces `BOARD_POSES["B2"]` + `SAFE_HEIGHT` (0.15 m) to **0.000 mm / 0.003°**.
   - The sim's fake server stops at the wrist (`r_wrist2hand`). For those joints, the wrist is **0.116 m** from the tip; the gap is 0.116–0.118 m, depending on wrist roll.
   - The sim's own planner compensates internally, but code written on the real robot doesn't match the sim. The fix is tracked in **#187**.
2. **Reach: all 9 taught 6-D tip poses solve within the sim's joint travel.** This is kinematic only, with no collision checks.
   - A1, A3 and B1 sit on or near the `r_shoulder_roll` +10° stop.
   - The robot itself used **+20.3°** shoulder roll at B2's safe height, past the sim's stop.
   - These are notes only; no change is proposed.
3. **Taught grid (Siva's lab), in the torso frame:**
   - centre (0.226, 0.000, −0.377); row step 0.128 m; column step 0.140 m;
   - rows 1.5° off straight ahead, level to within 1°;
   - residual rms 16 mm, max 29 mm. That is teaching scatter: no choice of hand point improves it.
   - The gripper was horizontal at every square, so the height doesn't depend on which point of the hand is used.
   - Square names map as A = far, C = near, columns 1–3 = +y to −y. Against sim cells: A ↔ r3, C ↔ r1, columns agree.
4. **Board comparison: not applicable.** The centre is 0.206 m nearer and 0.117 m lower than the MCC/SivaPool `cell_r2c2`. That reflects a different lab, not a sim error.
   - The MCC rig clearance check and a "real-robot board calibration" follow-up don't apply.
   - If Siva's lab is ever wanted in the sim, a measurement script run on the robot should capture it and hand back a results file. That belongs to the deferred calibration-profile work.

## Files and reproduction

| File | What it is |
|---|---|
| `taught_poses.json` | Extracted poses and the joints/pose pair, with provenance |
| `extract.py` | Notebook → `taught_poses.json`. Parses literals only and executes nothing. |
| `urdf_fk.py`, `ik.py` | URDF chain FK; damped least-squares IK with joint clamping. Numpy only. |
| `analyze.py` | Every number above |
| `analysis.txt` | Raw output of `analyze.py` |

```bash
python3 extract.py <CalibrateBoard.ipynb> taught_poses.json f207378
python3 analyze.py taught_poses.json > analysis.txt
```
