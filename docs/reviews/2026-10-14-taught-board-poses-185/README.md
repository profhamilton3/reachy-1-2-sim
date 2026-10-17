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
4. **Measured board vs taught board vs camera.**

   **The sources** (board centre, `cell_r2c2`, in the torso frame):

   | Source | Forward x | Left y | Height z |
   |---|---|---|---|
   | Model: tape-measured board, torso mounted at (0, 0, 1.0) | 0.432 | 0.000 | −0.260 |
   | Arm: Siva's taught poses (gripper tip, slightly above the surface) | 0.226 | 0.000 | −0.377 |
   | Camera: `photo_check/`, 17 views, head pitch 12–42° | **0.467** | 0.015 | **−0.416** |

   **How the camera check works.** It solves each photo against the taped grid, using the measured intrinsics (`calibration_measured_2026_08_27.yaml`). The board is level and square to the torso, so its orientation in the image gives the camera's orientation. The camera's position then follows from the URDF neck and camera offsets. No head angles are needed. Spread across views: 6 mm (x), 4 mm (y), 9 mm (z). The left and right eyes differ by 1–2 cm.

   **What it says:**
   - **Height:** the camera and the arm agree that the board surface is about **0.40–0.42 m below the shoulder line**, not the model's 0.26 m. With the measured 0.74 m surface, the shoulder line is about **1.15 m** above the floor, against the model's 1.0 m.
   - **Forward:** the camera agrees with the tape-measured board to within 3.5 cm. The arm's taught positions are the outlier, about 0.24 m short, with spacing compressed to 0.84× / 0.92×. A torso offset does not explain that; the arm's readings or teaching reference do. The earlier "torso about 0.21 m forward" estimate is withdrawn.
   - **Reach:** a higher torso puts the board further below the shoulder. That makes the far row *harder* to reach in the sim, not easier. So the far-row reach shown by the taught A1/A2 is tied up with the arm-side discrepancy, not with the mount.

## Files and reproduction

| File | What it is |
|---|---|
| `taught_poses.json` | Extracted poses and the joints/pose pair, with provenance |
| `extract.py` | Notebook → `taught_poses.json`. Parses literals only and executes nothing. |
| `urdf_fk.py`, `ik.py` | URDF chain FK; damped least-squares IK with joint clamping. Numpy only. |
| `analyze.py` | Every number above |
| `analysis.txt` | Raw output of `analyze.py` |
| `photo_check/` | `board_pnp.py` (board pose from each MCC photo), `results.json` (per view), `summary.txt` |

```bash
python3 extract.py <CalibrateBoard.ipynb> taught_poses.json f207378
python3 analyze.py taught_poses.json > analysis.txt
python3 photo_check/board_pnp.py <reachy-tabletop-ai>/tests/Pictures <out_dir>   # needs opencv
```
