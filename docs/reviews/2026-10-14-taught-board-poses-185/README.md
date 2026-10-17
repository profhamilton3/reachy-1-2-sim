# Taught board poses vs the sim's arm model (#185)

**Date:** 2026-10-14. **Offline only:** no robot time and no live sim.

**Source:** `profhamilton3/reachy-tabletop-ai` @ `f207378`, `tests/CalibrateBoard.ipynb`. Its SHA-256, the cell indices and the poses are in `taught_poses.json`.

**Sim:** `main` @ `045d508`: `native_mujoco/model/reachy.urdf` and `reachy_1_2.xml`, `fake_reachy_server.py`, `scenes/FWDCenterLabMCC.yaml` (SivaPool's grid is identical).

## Perspective

There is one robot and one lab. Siva taught these poses by hand on that robot, at the same board the sim's scenes measure (`FWDCenterLabMCC.yaml`, inherited by the default scene). `CalibrateBoard.ipynb` is the only source of taught board poses.

An earlier version of this review said the poses came from a different lab, and set the board comparison aside. That was wrong and is corrected below (finding 4).

The notebook's gripper values are Siva's chosen settings for his objects, not limits: open −29.79°, grasp −18.94°, closed +12.08°. The sim's range, −68.8° to +20.05°, is the URDF limit (−1.2 to 0.35 rad).

## Findings

1. **Frame and tool point: verified.** Real `r_arm.forward_kinematics()` is the URDF chain `torso` → `right_tip`, with URDF joint signs. Running the notebook's `B2_SAFE_JOINTS` through that chain reproduces `BOARD_POSES["B2"]` + `SAFE_HEIGHT` (0.15 m) to **0.000 mm / 0.003°**.
   - The sim's fake server stops at the wrist (`r_wrist2hand`). For those joints, the wrist is **0.116 m** from the tip; the gap is 0.116–0.118 m, depending on wrist roll.
   - The sim's own planner compensates internally, but code written on the real robot doesn't match the sim. The fix is tracked in **#187**.
2. **Reach: all 9 taught 6-D tip poses solve within the sim's joint travel.** This is kinematic only, with no collision checks.
   - A1, A3 and B1 sit on or near the `r_shoulder_roll` +10° stop.
   - The robot itself used **+20.3°** shoulder roll at B2's safe height, past the sim's stop.
   - These are notes only; no change is proposed.
3. **Taught grid, in the torso frame:**
   - centre (0.226, 0.000, −0.377); row step 0.128 m; column step 0.140 m;
   - rows 1.5° off straight ahead, level to within 1°;
   - residual rms 16 mm, max 29 mm. That is teaching scatter: no choice of hand point improves it.
   - The gripper was horizontal at every square, so the height doesn't depend on which point of the hand is used.
   - Square names map as A = far, C = near, columns 1–3 = +y to −y. Against sim cells: A ↔ r3, C ↔ r1, columns agree.
4. **Measured board vs taught board: they disagree, and the likely cause is the torso mount.**
   - **Positions:** the measured grid centre (`cell_r2c2`, 0.4318 m ahead of the pedestal axis, on the 0.74 m surface) sits at (0.432, 0.000, −0.260) in the model's torso frame, because the model mounts the torso at (0, 0, 1.0) over the pedestal. The taught centre is (0.226, 0.000, −0.377): **0.206 m nearer and 0.117 m lower**.
   - **Rotation:** none to speak of (0.8°), so the difference is almost a pure offset.
   - **Implied torso mount:** if the board measurement is right, the real torso origin sits at about **(0.21, 0.00, 1.12–1.15) m in the pedestal frame**, not (0, 0, 1.0). The height range covers the unknown height of the tip above the surface when taught (0–3 cm). Nothing in the repo measures the torso mount; the scene notes that the mast is not modelled.
   - **Spacing:** taught spacing is 0.838 × (rows) and 0.916 × (columns) the measured 0.1524 m. A mount offset doesn't explain that. It is left open: teaching scatter is 16 mm rms, too small to cause it alone.
   - **Reach:** with the implied mount, the far row comes within reach, as on the robot (A1 and A2 were taught). The sim currently marks `cell_r3c1` and `cell_r3c2` out of right-arm reach.

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
