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

## Scene representation

`FWDCenterLabMCC.yaml` is the authoritative inherited geometry: can radius 0.033 m, cylinder length 0.122 m. `FWDCenterLabSiva.yaml` and `FWDCenterLabSivaPool.yaml` inherit it. The can centre on the 0.740 m board is 0.801 m; its floor-resting pool centre is 0.061 m. Generated B1/B3 board placements must use the new half-height too.

The existing 6 cm pointing clearance consequently gives an 18.2 cm hover above the board for this can, rather than 17.5 cm. No clearance threshold, motion rule, friction, mass or gripper geometry is changed.

## Historical results and remaining work

Historical reviews, sealed evidence, pinned worktrees and generated image datasets describe their original geometry and remain unchanged. Reproduce them at their recorded commit; do not interpret their old clearance or hover results as a validation of the taller can. The source-scene change also affects new renders of the detector scene; an old generated dataset remains the old version and is not silently regenerated here.

Issue #42's dimensional uncertainty is resolved by this confirmation; the source correction still needs to land through the repository's normal review process. Missing reference photographs/appearance validation and issue #55's grasp reliability are separate. This change does not validate grasping, force calibration, physical accuracy, or return motion, and does not resume the tabled comparison.
