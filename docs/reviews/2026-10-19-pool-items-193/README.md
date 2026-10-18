# Sizing the lab's tic-tac-toe pieces from the existing photos (#193): not settled

`measure_pool_items.py` solves the board pose by PnP on the taped grid (the #188 method, 9 cells, 1.5-1.8 px rms), then fits a cube
(side, height, yaw) or cylinder (radius, height) to each piece's dark silhouette in `annotation/cube` (3 photos, 4 cubes each) and
`annotation/cylinder` (2 photos, 4 cylinders each). Both camera models were tried (the photos do not say which eye took them).
Raw output: `measure_pool_items.json`.

| | fits | width (cm) median, range | height (cm) median, range |
|---|---|---|---|
| cube side | 24 (6 with IoU >= 0.3) | 4.7, 4.1-5.5 | 5.2, 4.0-7.9 |
| cylinder diameter | 20 (4 with IoU >= 0.3) | 5.0, 4.6-6.4 | 6.6, 3.5-9.9 |

**Not settled.** The silhouette fits overlap the pieces poorly (IoU 0.10-0.44), the photos are blurred and the pieces are dark, so edges
are about 3 px (about 5 mm) soft. A single view cannot separate a piece's height from its top-face foreshortening at this blur: the
heights spread over 4-10 cm and the width over 4-5.5 cm, and the cube width sits below the issue's rough 6 cm reading. The results are
therefore not used to size any scene object; per the issue, no value has been guessed.

## Decision (owner, 2026-10-19)

All pool boxes are one size and all pool cylinders are one size. `pool_box_4/5` and `pool_cyl_4/5` therefore copy the existing
six (box 4 cm, cylinder radius 2 cm x 8 cm); the fits above (cube 4.1-5.5 cm, cylinder 4.6-6.4 cm across) neither contradict nor
improve on them. Siva measures the real pieces next, and that value replaces all ten sizes together. `validate_pool_scene.py`
checks the scene (schema, 14 placeable, unique ids), the parked slots settling, one placement of each new item and the render
`default_scene_pool.png`.
