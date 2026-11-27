# sim #212: the PART set — evidence (2026-11-27)

Environment: macOS arm64 (Apple Silicon), Python 3.14.0, MuJoCo 3.11.0, native server under `mjpython`. No Docker.

| File | What it is |
|---|---|
| `contact_sheet.png` | Family, black and one randomised run. Left: the robot's `left_camera` frame as the live server sent it (neck pitch 1.05 rad). Right: a free camera on the same settled poses, **with the robot hidden** because its torso and arm block the view. |
| `*_left_camera_*.jpg` | The unaltered live frames. `family_..._b` is the second copy of each design. |
| `*_top.png` | Free camera from above, robot hidden. |
| `fly_parts.py` → `results.json` | Starts the server for each variant, places parts over the WebSocket, reads poses after ≥ 2 s of sim time and again 0.25 s later. Family: all 16. Black and random (seed 20261127): the 8 `_a` parts. |
| `step_cost.py` → `step_cost.txt` | Why only some parts of a holed object reach its end faces. |
| `realtime_factor.txt` | Live server, sim time against wall time over 20 s, one client receiving state and both cameras: `main`'s Pool scene and this branch's. It compared against a copy of the old scene file and was run from a scratch script, so no script for it is kept here. |

`results.json`, 32 placements: horizontal error ≤ 0.001 mm, height error ≤ 0.045 mm (contact penetration), tilt ≤ 0.0014°, movement over the following 0.25 s ≤ 0.0002 mm.

Cells in the frames, far row first (left to right as seen): PART120, empty, PART221 / PART211, PART121, PART220 / PART110, PART111, PART210.

No STL or STEP file was read to make any of this.
