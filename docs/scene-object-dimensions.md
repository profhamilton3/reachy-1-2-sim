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

## Scene representation

`FWDCenterLabMCC.yaml` is the authoritative inherited geometry: can radius 0.033 m, cylinder length 0.122 m. `FWDCenterLabSiva.yaml` and `FWDCenterLabSivaPool.yaml` inherit it. The can centre on the 0.740 m board is 0.801 m; its floor-resting pool centre is 0.061 m. Generated B1/B3 board placements must use the new half-height too.

The existing 6 cm pointing clearance consequently gives an 18.2 cm hover above the board for this can, rather than 17.5 cm. No clearance threshold, motion rule, friction, mass or gripper geometry is changed.

## Historical results and remaining work

Historical reviews, sealed evidence, pinned worktrees and generated image datasets describe their original geometry and remain unchanged. Reproduce them at their recorded commit; do not interpret their old clearance or hover results as a validation of the taller can. The source-scene change also affects new renders of the detector scene; an old generated dataset remains the old version and is not silently regenerated here.

Issue #42's dimensional uncertainty is resolved by this confirmation; the source correction still needs to land through the repository's normal review process. Issue #55's grasp reliability is separate and remains open. This change does not validate grasping, force calibration, physical accuracy, or return motion, and does not resume the tabled comparison.
