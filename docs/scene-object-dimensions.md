# Confirmed pool-object dimensions

## Owner confirmation — 2026-10-06

Terrance confirmed: “The only measurement that is off in the object pool is the soda can. The diameter of 6.6 cm is correct, the height is actually 12.2 cm.” This is owner-supplied confirmation; no new independent measurement or photograph was taken by the coding agent.

| Object | Accepted dimensions |
|---|---|
| `red_cube` | 6 × 6 × 6 cm |
| `blue_cylinder` | Diameter 7 cm; height 10 cm |
| `soda_can` | Diameter **6.6 cm**; height **12.2 cm** (previously 11.5 cm) |
| `foam_block` | 7 × 7 × 5 cm |

Other pool-slot dimensions remain unchanged. The earlier photograph-derived claim that the cube/cylinder were approximately 1.5× oversized is superseded by the owner's confirmation. Do not resize them to the old approximate 4 cm estimate.

## Owner ruling — 2026-10-06

The confirmed dimensions above are authoritative. Photographs of the soda can or foam block are not required: they are not a prerequisite for merging this correction or for grasp work, and no further confirmation is requested. The can and foam-block colours in `FWDCenterLabSiva.yaml` remain modelled values and are unchanged by this ruling. The ruling does not claim that detector accuracy or grasp reliability has been demonstrated.

## Pool pieces: Siva's measurements — 2026-10-19 (#201)

Siva measured the lab's real tic-tac-toe pieces with a tape measure on 2026-10-19, with photos; the owner provided the values in #201. This is owner-provided: the coding agent took no measurement of its own.

| Pieces | Measured | Scene (`FWDCenterLabSivaPool.yaml`) |
|---|---|---|
| `pool_box_1`–`pool_box_5` | cube, 2 in on a side | box 0.0508 × 0.0508 × 0.0508 m |
| `pool_cyl_1`–`pool_cyl_5` | cylinder, 1.8 in across × 2.8 in tall | radius 0.02285 m, length 0.0711 m |

- They replace the 40 mm boxes and 40 × 80 mm cylinders that #193 kept as a stand-in "until Siva measures". All ten change together, so every pool box and every pool cylinder is still one size.
- Both pieces fit the open jaw: it leaves 67.8 mm at −65° (`docs/reviews/2026-10-19-side-on-grasp-180/hand_geometry.txt`).
- Floor-resting pool centres are half the height: 0.0254 m for a box, 0.03555 m for a cylinder. The generated E1 boards B4 and B5 (`scripts/make_e1_boards.py`) take the same half-heights on the board.
- **Not measured, unchanged:** mass (0.05 kg), friction and colour.
- **Unchanged by owner ruling (2026-10-17):** `red_cube`, `blue_cylinder`, `soda_can` and `foam_block` keep the dimensions above.

## Willis's printed parts: the PART set (#212)

Owner decisions: Terrance, 2026-11-27 (#212, part of reachy-tabletop-ai#80). Sixteen bodies in `FWDCenterLabSivaPool.yaml`, two of each of eight designs.

### Source

| | |
|---|---|
| Author | Willis |
| File | `docs/input_3D_printer_files/digital_twin_blocks.stp` (with `.stl`, `.3dm` and a photo of the prints) |
| STEP SHA-256 | `17e1ce8a3f62d394104c095d9e8a36aedffefe7bc1ef277ea6e40cb464083104` |
| STEP units | **INCH** (`CONVERSION_BASED_UNIT('INCH', …)`) |
| Published | **No.** The repository is public and permission to commit these files has not been given. The directory is in `.gitignore` (#217). |
| Licence | None recorded. |

The dimensions and volumes below are those recorded in #212, parsed from the STEP/STL on 2026-11-27. When building the scene the coding agent re-checked only the STEP file's SHA-256 and its unit declaration; it did not re-parse the geometry and measured no physical print. **Nothing in the repository reads the STEP or STL**: the scene is built from primitives, so it needs neither file.

### Parts

`PART<S><H><F>`: S 1 = block, 2 = cylinder; H 1 = short, 2 = tall; F 0 = solid, 1 = through hole. Scene ids are `part<SHF>_a` and `part<SHF>_b`.

| Design | Shape | Size (m) | Hole | Volume (cm³) | Mass (kg), synthetic | Family colour | RGBA |
|---|---|---|---|---|---|---|---|
| PART110 | block | 0.04921 × 0.04921 × 0.02540 | — | 61.5 | 0.175 | Mustard Yellow #CFA023 | 0.812, 0.627, 0.137 |
| PART111 | block | same | Ø0.0254 through, vertical | 48.7 | 0.160 | Mustard Yellow | same |
| PART120 | block | 0.04921 × 0.04921 × 0.06985 | — | 169.2 | 0.270 | Burnt Orange #C85A17 | 0.784, 0.353, 0.090 |
| PART121 | block | same | Ø0.0254 through | 133.8 | 0.240 | Burnt Orange | same |
| PART210 | cylinder | r 0.024606, h 0.02540 | — | 48.3 | 0.160 | Dusty-Blue Grey #6B8299 | 0.420, 0.510, 0.600 |
| PART211 | cylinder | same | Ø0.0254 through | 35.4 | 0.150 | Dusty-Blue Grey | same |
| PART220 | cylinder | r 0.024606, h 0.06985 | — | 132.8 | 0.235 | Grass Green #4E9A34 | 0.306, 0.604, 0.204 |
| PART221 | cylinder | same | Ø0.0254 through | 97.4 | 0.205 | Grass Green | same |

- **Mass is synthetic, not measured.** Owner rule: 1.25× to 2.25× the heaviest object the scene already had (`red_cube`, 0.12 kg), which is 0.150–0.270 kg, mapped linearly on volume (35.4 → 169.2 cm³) and rounded to the nearest 5 g. So similar volumes give similar masses (PART121 ≈ PART220, PART111 ≈ PART210) and a hole always reduces mass. These are not PLA densities. The prints can be weighed later.
- **Friction:** as the pool pieces, not measured: block 0.8, cylinder 0.75 (torsional 0.01, rolling 0.001).
- **Colour** follows the family (shape × height) and never the hole. Source: the owner's palette file, kept with the uncommitted printer inputs. The RGBA values above are the scene's *material* colours. Rendered pixels differ with lighting, so do not compare a camera pixel to these numbers: use the *rendered reference colours* below.
- **Variants:** `FWDCenterLabSivaPoolBlack.yaml` paints every part matte black (0.05, 0.05, 0.05). `FWDCenterLabSivaPoolRandom.yaml` draws each part's colour from the four family colours by a recorded seed (the committed file keeps seed 2026). `scripts/make_part_colour_scene.py` writes one scene per run to `scenes/generated/` (git-ignored, and visible to the container through `./scenes`); the seed is in that file, and the server's recorder manifest and `generate_dataset`'s manifest repeat it with the colours (`color_randomization`).
- **Same geometry** (D5). Both variants compile to the Pool scene's model with only `geom_rgba` different, so `rig_routes.SCENE_GEOMETRY_PARENTS` records the Pool scene as their geometry parent and its route validations apply to them (`tests/unit/test_colour_variant_routes.py`). The experience reuse gate (`check_reuse`) does **not** share trials between them: it requires the same scene and compiled-model hashes (owner decision 2026-11-27).

### Rendered reference colours (owner decision D6, 2026-11-27)

The palette above is the *material* colour. What a camera sees is different, so colour grading (reachy-tabletop-ai#84) compares against **these**, not against the hex palette. The palette is unchanged.

CIELAB (D65) from the sRGB of the decoded JPEG. `left_camera`, 640 × 480, `calibration_measured_2026_08_27`, no distortion or effects; head at `gaze.BOARD_VIEW` (neck 0, 64, 0); arm at REST; the compiled scene's lighting; MuJoCo 3.11.0 on macOS arm64 (another renderer may differ). Pixels on the part's **top and outer side faces** (bore interior and edge pixels left out), one part at a time on each of the nine cells at yaw 0, both designs of the family. Reproduce with `docs/reviews/2026-11-27-part-set-212/measure_rendered_colours.py` (two runs gave identical numbers); raw values are in `rendered_colours.json`.

| Scene | Family | Material | **Reference L\*, a\*, b\*** (top + side) | Top only | Side only | Furthest single placement, ΔE76: combined / top / side | Top share of pixels | Top pixels with a clipped channel |
|---|---|---|---|---|---|---|---|---|
| family | PART11x (block, short) | mustard yellow | **94.9, -17.1, 82.3** | 96.0, -18.8, 83.2 | 41.6, 3.8, 42.9 | 46.2 / 48.2 / 22.6 | 85 % | 60 % |
| family | PART12x (block, tall) | burnt orange | **66.4, 32.3, 63.8** | 69.6, 35.0, 68.4 | 30.8, 25.9, 36.1 | 45.5 / 30.6 / 23.0 | 64 % | 57 % |
| family | PART21x (cylinder, short) | dusty-blue grey | **79.1, -3.5, -17.3** | 80.6, -3.8, -19.6 | 32.0, -1.9, -10.0 | 32.1 / 33.9 / 12.6 | 85 % | 20 % |
| family | PART22x (cylinder, tall) | grass green | **74.3, -51.0, 53.7** | 85.4, -58.8, 62.7 | 33.4, -27.6, 28.7 | 50.9 / 46.6 / 19.8 | 64 % | 20 % |
| black | PART11x (block, short) | matte black | **6.3, 0.0, 0.0** | 6.3, 0.0, 0.0 | 2.2, 0.0, 0.0 | 5.9 / 5.9 / 0.8 | 85 % | 0 % |
| black | PART12x (block, tall) | matte black | **5.1, 0.0, 0.0** | 6.3, 0.0, 0.0 | 2.2, 0.0, 0.0 | 6.2 / 5.4 / 0.8 | 64 % | 0 % |
| black | PART21x (cylinder, short) | matte black | **6.3, 0.0, 0.0** | 6.3, 0.0, 0.0 | 2.2, 0.0, 0.0 | 5.9 / 5.9 / 0.5 | 85 % | 0 % |
| black | PART22x (cylinder, tall) | matte black | **4.7, 0.0, 0.0** | 6.3, 0.0, 0.0 | 2.2, 0.0, 0.0 | 6.6 / 5.4 / 0.8 | 64 % | 0 % |

How far to trust them:

- **In the family scene they separate the families:** 68 of 72 single placements are nearest (ΔE76) to their own family's reference, 70 of 72 using a\*b\* alone. Misses: PART11x 4 (L\*a\*b\*) / 2 (a\*b\*).
- **One placement can be far from its family's reference.** Top and side faces render very differently (L\* often twice as high on top), so the combined median is the median of two clusters and moves with how much side face is in view. Comparing top with top or side with side removes that, but lighting still varies across the board.
- **The lighting clips the top faces:** the last column is the share of top pixels with a channel at 255. Hue there is distorted, which is why mustard renders as a pure yellow.
- **In the black scene all four families are the same colour** (differences are only which cells and faces each shape shows). These rows say what black looks like; they cannot tell families apart.
- These are sim values at one head pose and yaw 0. They say nothing about the printed parts under lab light.

### Where the sim differs from the real part

Holed parts use `geometry.kind: compound` (a mesh would collide as its convex hull, which fills the hole).

| Design | Sim geometry | Difference from the CAD part |
|---|---|---|
| PART111, PART121 | 20 boxes. Four walls (two 49.21 × 11.905 mm, two 11.905 × 25.40 mm) make a square 25.4 mm opening; a ring of 16 boxes from r = 12.7 to 18.0 mm, just past that opening's corners (12.7 × √2 = 17.96), rounds it | The bore is a **16-sided round hole of inradius 12.7 mm** (corners at r = 12.95), the same opening as PART211/PART221 (owner decision D2, 2026-11-27: round, like the printed parts). The ring lies mostly inside the walls. |
| PART211, PART221 | 16 boxes, a 16-sided ring with flats at r = 12.7 and 24.606 mm | The bore and the outside are 16-sided. Outer corners stand 0.48 mm proud of r = 24.606; the bounding box is exactly 49.212 mm. Neighbouring segments overlap near the bore. |

In both, the clear bore is nowhere narrower than Ø25.4 mm.

**End faces** (accepted by the owner, D1). In a holed part only every other segment of a cylinder's ring (8 of 16), or a block's two long walls, reaches the top and bottom faces. All other parts, including a block's whole bore ring, stop **0.1 mm short at each end**. The full-height parts span the whole footprint (the same convex hull as a flat face, so the part stands and tips the same way, either way up), and the envelope, bore and side faces are unchanged. The reason is cost: with all faces flat each ring cylinder carried 64 floor contacts and the Pool scene's physics step rose from about 0.1 ms to 1.7 ms of its 2.0 ms budget, because the no-slip pass (ADR-0005) grows with the square of the contacts on one body. With the recess it is about 0.6 ms (`step_cost.py` and `step_cost.txt` in the review folder; a benchmark, not a test). The no-slip setting is unchanged.

**Faint lines on the tops (note for reachy-tabletop-ai#84).** In close renders the tops of the holed parts show faint lines where the heights change: radial seams on the ring cylinders, and lines on the holed blocks where the recessed ring and short walls meet the long walls. They are artefacts of the sim, not features of the printed parts, and a perception method that looks for 0.1 mm steps or fine edges on a top face will find them. The owner accepts them. Separately, **close** renders with shadows on show small patches on the holed blocks' tops around the bore; they go away with shadows off (`closeup_shadows_on_off.png`), so they are a shadow-rendering artefact. From the robot's `left_camera` at `board_view` no recessed surface shows through a full-height top face (0 pixels in 36 placements of the four holed designs, `hidden_surface_check.txt`).

**Step-cost guard** (D3). `tests/unit/test_part_set_scene.py` places each PART upright on a board cell and bounds its active contacts: solid block 4, holed block 8, solid cylinder 5, holed cylinder 32. It counts rather than times, so it does not depend on the machine's load.

Mass is stated once, on the body's inertial. Its inertia tensor is computed from the parts at one uniform density, so where parts overlap the overlap counts twice. For the ring cylinders it is within 5 % of an ideal annulus. For the holed blocks, where the bore ring lies largely inside the walls and draws mass inward, it is **3.7 % to 9.5 % below** a square prism with a round Ø25.4 bore (both held by tests). The total mass is exact.

For software that reads an object as one primitive: the planner's `SceneModel` treats a compound as its solid bounding box, bore and all (conservative for clearance), except that a grasp planner sees the object it is **grasping** through its checked outer envelope (box for the holed blocks, cylinder for the holed cylinders; `native_mujoco/OBJECT_MODEL.md`), and `truth.v1` reports its outer box or cylinder, chosen by `semantic_class`. `truth.v1` has **no hole field**; the design id carries it (third digit of `part<SHF>`).

## Scene representation

`FWDCenterLabMCC.yaml` is the authoritative inherited geometry: can radius 0.033 m, cylinder length 0.122 m. `FWDCenterLabSiva.yaml` and `FWDCenterLabSivaPool.yaml` inherit it. The can centre on the 0.740 m board is 0.801 m; its floor-resting pool centre is 0.061 m. Generated B1/B3 board placements must use the new half-height too.

The existing 6 cm pointing clearance consequently gives an 18.2 cm hover above the board for this can, rather than 17.5 cm. No clearance threshold, motion rule, friction, mass or gripper geometry is changed.

## Historical results and remaining work

Historical reviews, sealed evidence, pinned worktrees and generated image datasets describe their original geometry and remain unchanged. Reproduce them at their recorded commit; do not interpret their old clearance or hover results as a validation of the taller can. The source-scene change also affects new renders of the detector scene; an old generated dataset remains the old version and is not silently regenerated here.

Issue #42's dimensional uncertainty is resolved by this confirmation; the source correction still needs to land through the repository's normal review process. Issue #55's grasp reliability is separate and remains open. This change does not validate grasping, force calibration, physical accuracy, or return motion, and does not resume the tabled comparison.
