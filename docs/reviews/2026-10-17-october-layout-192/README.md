# The October layout: offline evidence (#192)

**Dates:** 2026-10-17 and 2026-10-18 (after the owner's answers to the second STOP). **Offline:** MuJoCo 3.11 on the host, no live sim. The decision record is [ADR-0010](../../adr/0010-october-layout.md); this folder holds the numbers and the scripts that produced them.

Run everything from the repo root with `PYTHONPATH=src` (the physics scripts add `native_mujoco` themselves). The `.txt` and `.jsonl` files are the raw output, from the model as it is on this branch: the hand collision boxes in, the pedestal stub non-colliding, REST the second search's.

| File | What it measures |
|---|---|
| `reach_cells.py` → `reach_cells.txt` | The per-cell reach table in `scenes/FWDCenterLabMCC.yaml`. `august` = torso (0, 0, 1.0) and the +10° roll stop. No argument = the MJCF as it is. |
| `settle.py` | Motors-off settling of the right arm in the FWDCenterLabSivaPool world (used by the next two). |
| `rest_search.py` | The REST search: random starts on the margin, dropped motors-off, fixed points kept (the gripper still shut), scored against tall objects on column 3. `rig_routes.REST` is the rounded member of a family it and a local search around its best off-stop points found (seeds 41-44 and 51-54). |
| `fly_rest.py` → `fly_rest.txt` | REST motors-off (10 s) and motors-on (5 s), then LOWER_TO_REST and LIFT_TO_PRESENT flown with tall objects standing on column 3. |
| `fly_wave_point.py` → `fly_wave.jsonl`, `fly_point/*.jsonl`, `fly_point.txt` | WAVE and POINT through the panel's own code, from REST and back (`fly_point_summary.py` prints `fly_point.txt`). |
| `grasp_before_after.py` → `.txt` | The R12-502 pinch with and without the hand's collision boxes: robot geoms touching the cube, grip force. |
| `footprint_before_after.py` → `.txt` | The footprint guard's own number, per column-3 cell and object, August vs October. |
| `route_clearance_before_after.py` → `.txt` | Each full route's planned worst clearance to objects, rails and the board, tube and shells hand models. |
| `other_scenes.py` → `.txt` | The start keyframe dropped motors-off in every scene. |
| `bias_fit.py` → `bias_fit.txt` | `gaze.PITCH_BIAS_DEG`'s fit, repeated at the new mount. |
| `render_board_view.py` → `board_view_{left,right}.png` | Both sim cameras at `gaze.BOARD_VIEW`, arm at REST (pinhole, measured fovy). |
| `place_sweep_before.md`, `place_sweep_after.md` | `python -m tools.place_sweep` at `main` (db0c0ef) and on this branch. |

## Headline numbers

- **Reach.**
  - August: r3c1 NO (100.5 mm), r3c2 NO (41.7 mm).
  - October: all nine reach.
  - r3c1 needs shoulder roll of at least 15.3° at 5 cm above the surface and 19.7° at 3 cm, against the new +20.3° stop.
- **REST footprint, column 3** (the guard's number, the worst of six objects up to 12.2 cm tall):

  | | r1c3 | r2c3 | r3c3 |
  |---|---|---|---|
  | August | +2.3 | −5.7 | −9.4 |
  | October | +2.6 | +7.4 | +13.2 |

  Every r2c3 and r3c3 case was refused in August; #192's pool_box_3 on r2c3 reproduces at −3.7 cm.
- **REST in physics.**
  - Motors off for 10 s: at most 1.2° of drift on an arm joint, the gripper 3° off shut, and still.
  - Lower and lift with tall objects on column 3: 0.0 mm moved, worst MJCF distance 2.7 cm.
- **The hand's collision boxes.** Pads-only contact on a 20 mm and a 40 mm cube, grip force identical with and without them (28.0 / 53.6 N at the close). The visual shells at REST: −0.6 and −3.2 mm (contact depth), from −38.6 and −24.6 mm.
- **WAVE:** three layouts, three cycles each, nothing moved, no contact, worst 3.6 cm.
- **POINT:** empty board 9/9 cells (miss 0.4–1.4 cm); one object on the centre cell, 57/60 cells and 3/6 objects (the rest refused or stopped by the guard); nothing moved in any of the 69.
- **Board view.**
  - Neck (0, 64.0, 0): all nine cells inside both frames, with a 66 px worst margin (pinhole).
  - The chest box hides the near tape border and about the bottom quarter of the near row.
- **Pitch bias.** The centring offset is +5.7 to +11.3 depending on the cell, so it was re-fitted from 10.8 to 8.5.

## Notes on the method

- **Box-box distances are capped at 3 cm.** MuJoCo's `mj_geomDistance` returns a spurious 0 for two boxes at a large `distmax`; the flight scripts cap it at 3 cm for those pairs, where it is exact. A 3.0 in the output means "at least 3 cm".
- **The route validation is offline physics,** without the server, the SDK or the bridge. The panel smoke after the image rebuild flies startup, stow, wave and point once each, live.
