# Perception stage 1 (sim side): the robot team's detector and stereo depth, graded against simulator truth

Refs #186, #182. Robot side: reachy-tabletop-ai#28.

**Status: tooling merged for review; no capture has been run.** The results section at the bottom is empty on purpose. Nothing here was run against the live simulator, the viewer, Docker or the robot: the tools are covered by offline tests on fixture JSON (`tests/unit/test_truth_export.py`, `tests/unit/test_perception_grade.py`). The capture steps below are written from the code and the viewer's own docs and have **not** been clicked through; if a label differs on screen, trust the screen and fix this file.

## What is graded

The tabletop stereo viewer (`StereoLauncher.sh`, source `sim`) takes the sim's left and right camera frames through the v1 `CameraService`, runs the deployed detector (`best.tflite`: `cube`, `cylinder`, `empty`) and stereo depth, and reports 3D positions in the URDF torso frame. The grader compares those with the sim's own object poses.

Why this works without frame metadata: the v1 `CameraService` carries no sim step, so the scene is held **static** and the truth is read before and after each layout's captures. If anything moved more than 1 mm, or the scene revision changed, the whole layout is rejected.

| Piece | File |
|---|---|
| Truth (read-only, `truth.v1`) | `native_mujoco/truth_export.py`, run with `scripts/export_truth.py` |
| Grading | `scripts/perception_grade.py` |
| Fixtures | `tests/fixtures/perception_grade/` |

`scripts/export_truth.py` connects to `ws://127.0.0.1:8765` as a short-lived client, sends only `hello` and `heartbeat_ack`, takes the first fresh state and exits. It cannot move or change anything (a unit test checks the allow-list). It refuses a non-local `--url`.

## Settings (all recorded in `result.json`)

- **Match radius: 40 mm**, on the truth **visible-surface point**: the object's centre moved toward the right camera by the object's half-extent along the view ray. The viewer measures the visible surface, not the centre.
- **In view:** the truth centre projects inside both images (16 px margin), at the recorded neck angles, with the stereo profile's lens model (distortion included when the profile says `distorted`).
- **Frame:** torso. Truth is in the scene frame; the export records `arm_base_in_world = (0.206, 0, 1.15)` (ADR-0010) and the grader subtracts it.
- **Head pose:** `board_view`, neck (roll 0, pitch 64, yaw 0).
- **Camera distance bands** for the error table: <0.45, 0.45-0.50, 0.50-0.55, 0.55-0.60, >=0.60 m (camera to object centre).

## Layouts (scene `FWDCenterLabSivaPool`)

Grid: rows r1 (near the robot, x 0.279) to r3 (far, x 0.584); columns c1 (robot's left, +y) to c3 (right). In the default layout **every object is parked on the floor** and the board is empty.

| Layout | Cube (`red_cube`) | Cylinder (`blue_cylinder`) | Purpose |
|---|---|---|---|
| `A_default` | parked (pool) | parked (pool) | empty board: `empty` precision, false positives |
| `B_corners` | `r1c1` | `r3c3` | near-left cube, far-right cylinder |
| `C_swapped_depth` | `r3c2` | `r1c2` | far cube, near cylinder (the opposite distances) |

`soda_can` and `foam_block` stay parked in all three (the detector has no class for them; if you place one on the board it appears as an OOV hit, not an error).

### Panel requests

On the camera page (`http://localhost:8080`), **SCENE CONTROL**: press **connect** if it is not connected; leave the yaw selector at `0`. Select an object in the drop-down, then click the cell. Each click sends one `place_object` request:

| Layout | Do this | Requests sent (`request_id`) |
|---|---|---|
| `A_default` | press **clear board** | `red_cube@pool`, `blue_cylinder@pool`, ... (recalls every object) |
| `B_corners` | **clear board**; select `red_cube`, click `r1c1`; select `blue_cylinder`, click `r3c3` | `red_cube@r1c1`, `blue_cylinder@r3c3` |
| `C_swapped_depth` | **clear board**; select `red_cube`, click `r3c2`; select `blue_cylinder`, click `r1c2` | `red_cube@r3c2`, `blue_cylinder@r1c2` |

All nine cells are marked reachable, so no "place anyway" override is needed. After each placement, wait about 2 s for the objects to settle before the "before" export; check the cell shows the object.

## Capture (the owner runs this)

Once:

```bash
export RUN=$HOME/reachy-runs/perception-186        # anywhere outside the repo
mkdir -p $RUN/A_default $RUN/B_corners $RUN/C_swapped_depth
./SivaLabLauncher.sh                                # sim, scene FWDCenterLabSivaPool, calibration measured_2026_08_27, --distortion
cd ~/IITG-Reachy-Project && ./StereoLauncher.sh     # viewer on http://localhost:8081, source sim
```

Run `export_truth.py` with the Python that has `native_mujoco/requirements.lock` installed.

Then, for **each layout** in the order A, B, C (`L` = its folder name):

1. Set up the layout in the panel (table above).
2. **Truth before:**
   ```bash
   python3 scripts/export_truth.py --out $RUN/$L/truth_before.json
   ```
3. In the viewer (`http://localhost:8081`), **Data capture**, procedure **Head sweep**: on the `board_view` row tick *I confirm: move head to board_view (0, 64, 0°)* and press **Move head**. Wait for `reached` with a read-back near 0 / 64 / 0.
4. In the top panel, for **each of 2 pairs**: **Capture pair** -> **Analyze captured pair** -> **Measure in 3D** -> **Save pair (zip)**. Save as `$RUN/$L/pair1.zip`, `pair2.zip`. For pair 2, press **Return to live** first and capture again (the sim is deterministic, so the second pair mainly shows detector and matcher repeatability).
5. **Truth after**, with nothing else done to the scene in between:
   ```bash
   python3 scripts/export_truth.py --out $RUN/$L/truth_after.json
   ```

Do not touch the scene, press **recall**/**clear board**, or place anything between steps 2 and 5 of a layout. If you do, start that layout again.

## Grade

Write `$RUN/layouts.json`:

```json
{"layouts": [
  {"name": "A_default",       "truth_before": "A_default/truth_before.json",       "truth_after": "A_default/truth_after.json",       "pairs": ["A_default/pair1.zip", "A_default/pair2.zip"]},
  {"name": "B_corners",       "truth_before": "B_corners/truth_before.json",       "truth_after": "B_corners/truth_after.json",       "pairs": ["B_corners/pair1.zip", "B_corners/pair2.zip"]},
  {"name": "C_swapped_depth", "truth_before": "C_swapped_depth/truth_before.json", "truth_after": "C_swapped_depth/truth_after.json", "pairs": ["C_swapped_depth/pair1.zip", "C_swapped_depth/pair2.zip"]}
]}
```

Paths must stay below the folder holding `layouts.json`. Then:

```bash
python3 scripts/perception_grade.py --manifest $RUN/layouts.json --out-dir $RUN/graded
```

This writes `result.json` (machine-readable, with input SHA-256s, the settings, detector model and stereo profile id) and `report.md`. Keep the raw bundles and truth files next to them. Options: `--match-radius-mm 40`, `--in-view-margin-px 16`, `--score-off-board`.

A rejected layout (scene moved) or an ungradable pair (no `depth.json`, no neck read-back, a bundle that fails its manifest SHA-256) is listed under **Rejected input** in the report and in `rejected_pairs`; it is never silently dropped.

## What each metric means

- **Per class (cube, cylinder), in view:** the number of in-view truth objects of that class across graded pairs. **Recall** = correct / in view. **Precision** = correct / all detections carrying that label, so wrong-class calls, OOV hits and false positives all lower it.
- **Correct:** matched to a truth object within the radius **and** the label equals its class.
- **Class accuracy:** correct / all matched in-vocabulary objects.
- **Confusion:** truth class -> reported label, plus `missed`.
- **Miss:** an in-view cube or cylinder with no detection within the radius. The report notes when a detection box still covers it ("detected, but the 3D position is off").
- **False positive:** a cube or cylinder detection that matched nothing. The report gives the nearest truth object and its distance.
- **OOV hit:** a detection that landed on an object the detector has no class for (can, foam). Not counted correct, not a recall miss, not a false positive.
- **Matched by 2D only:** a detection with no usable 3D position was matched because the truth centre falls inside its box. It counts for recall and class accuracy but not for the 3D errors.
- **`empty` detections, graded against cells:** the box is assigned to the cell whose projected centre is inside it (nearest to the box centre). Correct if that cell holds no object; false positive if it holds one (a can counts); *not on a cell* if no cell centre is inside the box.
- **Per cell:** for each in-view cell, what the truth holds, what the detector reported (`cube`, `cylinder`, `empty`, or `unknown`) and the outcome (`correct`, `wrong`, `unknown`, `oov`). **`unknown` is never `empty`.**
- **3D error, mm, matched objects with a 3D position** (mean, median, p95, max; read p95 with the `n` shown):
  - `gate`: distance to the visible-surface point. At most the match radius by construction.
  - `centre`: distance to the true centre.
  - `horizontal`: error in (x, y) against the true centre. **This is the pick-and-place number.**
  - `along-ray`: signed error along the camera-to-object ray against the true centre; negative = nearer than the centre (expected, because the camera sees the near surface).
  - Grouped by the cell the object stands on and by camera distance.
- **Size error:** not available. The viewer's analysis carries no size estimate; the report says so.
- **Off-board sightings:** the parked floor pool is in view at `board_view` (about 1.4-1.6 m away). Those objects are reported in their own line and are **left out of the headline recall** unless `--score-off-board` is given. This is a deviation from "in view = scored" made so ten floor objects do not swamp the board numbers; the flag restores it.

## Known limits, read before interpreting

- **Horizontal error includes the convention.** The 40 mm gate is applied to the near-surface point (support point along the view ray). If the viewer's median sits on the top face, nearer the object's axis, the `horizontal` error will be small and the gate distance larger; if it sits on the rim, the reverse. A cylinder (r 35 mm) can differ by about 30 mm between the two. `gate` and `horizontal` are both reported so this is visible; the 40 mm radius is one setting if it needs revisiting.
- **The in-view test uses the stereo profile's lens model**, not the renderer's actual warp; an object right at the edge may flip in or out.
- **Pairs of one layout are not independent scenes.** Two pairs measure repeatability, not generalisation.
- **Sim pictures differ from the training photos** (exposure, background, a plain white table: #182 findings 3-5). Low recall or sparse stereo on a plain table is expected and is a finding, not a grader fault.
- **Not covered:** training on sim images, table texture, anything on the physical robot, Docker or Apple-Silicon timing.

## Results

*(to be filled in after the owner's run)*

| Layout | Pairs graded | Rejected | Notes |
|---|---|---|---|
| A_default | | | |
| B_corners | | | |
| C_swapped_depth | | | |

Headline: *(recall / precision per class, class accuracy, 3D error bounds per distance band; post to #182 for its stage 2 gate)*
