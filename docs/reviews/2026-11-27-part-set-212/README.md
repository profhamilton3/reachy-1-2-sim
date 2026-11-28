# sim #212: the PART set — evidence (2026-11-27)

Environment: macOS arm64 (Apple Silicon), Python 3.14.0, MuJoCo 3.11.0, native server under `mjpython`. 

Everything here reflects the owner's decisions D1–D6 on PR #219 (round bore in the holed blocks, contact guard, route validations for the colour variants, rendered reference colours).

| File | What it is |
|---|---|
| `contact_sheet.png` | Family, black and one randomised run. Left: the robot's `left_camera` frame as the live server sent it, head at `gaze.BOARD_VIEW`, measured calibration. Right: a free camera on the same settled poses, **with the robot hidden**. Bottom: the holed blocks' round 16-sided bore, close up. |
| `*_left_camera_*.jpg` | The unaltered live frames. `family_..._b` is the second copy of each design. |
| `*_top.png` | Free camera from above, robot hidden. |
| `fly_parts.py` → `results.json` | Starts the server for each variant, places parts over the WebSocket, reads poses after ≥ 2 s of sim time and again 0.25 s later; exits non-zero unless all are still, upright and on their cells. Family: all 16. Black and random (seed 20261127): the 8 `_a` parts. |
| `measure_rendered_colours.py` → `rendered_colours.json` | D6: the rendered reference colours (the table is in `docs/scene-object-dimensions.md`). Two runs gave identical numbers. |
| `hidden_surface_check.py` → `.txt` | No recessed surface of a holed part shows through its full-height top faces in the robot's camera at `board_view`. |
| `closeup_shadows_on_off.png` | The patches on holed tops in close free-camera renders: with shadows (left) and without (right). A shadow-rendering artefact. |
| `step_cost.py` → `step_cost.txt` | Benchmark (not a gate): physics step by which parts reach the end faces. The gate is the contact-count test. |
| `realtime_factor.txt` | Live server, sim time against wall time over 20 s, one client receiving state and both cameras: `main`'s Pool scene and this branch's. It compared against a copy of the old scene file and was run from a scratch script, so no script for it is kept here. |

`results.json`, 32 placements: horizontal error ≤ 0.001 mm, height error ≤ 0.045 mm (contact penetration), tilt ≤ 0.0017°, movement over the following 0.25 s ≤ 0.0003 mm.

`step_cost.txt` and `realtime_factor.txt` were taken while two unrelated test runs were using the machine (load about 3.5).

Cells in the frames, far row first (left to right as seen): PART120, empty, PART221 / PART211, PART121, PART220 / PART110, PART111, PART210.

On macOS the two depth-buffer scripts print `ARB_clip_control unavailable … depth accuracy will be limited`. The accuracy is enough here: on solid parts, which have no bore, every pixel the segmentation gives them was placed on a top or outer side face within 0.4 mm (`max_excluded_fraction_solid` = 0.0).

Container (Docker) results are in `container_checks.md`; `check_rviz_markers.py` runs inside the container, and `container_compare.py` compared the per-file test logs (not committed).

No STL or STEP file was read to make any of this.
