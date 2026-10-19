# Offline feasibility sweep — placement (2026-10-19; #174, side-on #180)

reachy-1-2-sim at `worktree`.  Preflight only (`crane_place_live.plan_crane_place`): no motion, no simulator.  IK: the host mirror of the simulator's service.  Scene FWDCenterLabSivaPool, the object upright at the origin cell's centre, nothing else on the board.  Reachable cells: the 9 without the `no-right-arm` tag.

## red_cube

Feasible: **8 of 9** ordered pairs.  r1c1→, r1c2→, r1c3→, r2c1→, r2c2→, r2c3→, r3c2→, r3c3→

| from → to | feasible | grasp used | margin mm (side-on / crane) | refused at | stage | reason |
|---|---|---|---|---|---|---|
| r1c1 → (lift) | **yes** | side-on | 1.00 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c2 → (lift) | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c3 → (lift) | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +1 |
| r2c1 → (lift) | **yes** | side-on | 0.99 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c2 → (lift) | **yes** | side-on | 1.01 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c3 → (lift) | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +10 |
| r3c1 → (lift) | no |  |  | pick | plan | side-on: plan: IK found no level grasp pose for this approach (best of 4 side-on approaches); crane: plan: IK found no grasp pose in the bounded start set |
| r3c2 → (lift) | **yes** | side-on | 1.02 / 0.00 |  |  |  |
| r3c3 → (lift) | **yes** | side-on | 0.97 / — |  |  | crane: straddle: predicted grasp pose does not straddle: ['finger_above_top'] |

Refusals by reason:

- 1 × pick / plan: side-on: plan: IK found no level grasp pose for this approach (best of 4 side-on approaches); crane: plan: IK found no grasp pose in the bounded start set

## foam_block

Feasible: **0 of 9** ordered pairs.

| from → to | feasible | grasp used | margin mm (side-on / crane) | refused at | stage | reason |
|---|---|---|---|---|---|---|
| r1c1 → (lift) | no |  |  | pick | shape | side-on: shape: foam_block is a foam box; the side-on grasp is for cubes (red_cube, the pool boxes) and cylinders (blue_cylinder, the pool cylinders) only; crane: plan: IK found no grasp pose in the bounded start set |
| r1c2 → (lift) | no |  |  | pick | shape | side-on: shape: foam_block is a foam box; the side-on grasp is for cubes (red_cube, the pool boxes) and cylinders (blue_cylinder, the pool cylinders) only; crane: plan: IK found no grasp pose in the bounded start set |
| r1c3 → (lift) | no |  |  | pick | shape | side-on: shape: foam_block is a foam box; the side-on grasp is for cubes (red_cube, the pool boxes) and cylinders (blue_cylinder, the pool cylinders) only; crane: plan: IK found no pose for rung +2 |
| r2c1 → (lift) | no |  |  | pick | shape | side-on: shape: foam_block is a foam box; the side-on grasp is for cubes (red_cube, the pool boxes) and cylinders (blue_cylinder, the pool cylinders) only; crane: plan: IK found no grasp pose in the bounded start set |
| r2c2 → (lift) | no |  |  | pick | shape | side-on: shape: foam_block is a foam box; the side-on grasp is for cubes (red_cube, the pool boxes) and cylinders (blue_cylinder, the pool cylinders) only; crane: plan: IK found no grasp pose in the bounded start set |
| r2c3 → (lift) | no |  |  | pick | shape | side-on: shape: foam_block is a foam box; the side-on grasp is for cubes (red_cube, the pool boxes) and cylinders (blue_cylinder, the pool cylinders) only; crane: plan: IK found no pose for rung +10 |
| r3c1 → (lift) | no |  |  | pick | shape | side-on: shape: foam_block is a foam box; the side-on grasp is for cubes (red_cube, the pool boxes) and cylinders (blue_cylinder, the pool cylinders) only; crane: plan: IK found no grasp pose in the bounded start set |
| r3c2 → (lift) | no |  |  | pick | shape | side-on: shape: foam_block is a foam box; the side-on grasp is for cubes (red_cube, the pool boxes) and cylinders (blue_cylinder, the pool cylinders) only; crane: pad_route: finger pad route clearance -2.50 mm < 0.494 mm from rung +9 |
| r3c3 → (lift) | no |  |  | pick | shape | side-on: shape: foam_block is a foam box; the side-on grasp is for cubes (red_cube, the pool boxes) and cylinders (blue_cylinder, the pool cylinders) only; crane: plan: IK found no grasp pose in the bounded start set |

Refusals by reason:

- 6 × pick / shape: side-on: shape: foam_block is a foam box; the side-on grasp is for cubes (red_cube, the pool boxes) and cylinders (blue_cylinder, the pool cylinders) only; crane: plan: IK found no grasp pose in the bounded start set
- 1 × pick / shape: side-on: shape: foam_block is a foam box; the side-on grasp is for cubes (red_cube, the pool boxes) and cylinders (blue_cylinder, the pool cylinders) only; crane: plan: IK found no pose for rung +2
- 1 × pick / shape: side-on: shape: foam_block is a foam box; the side-on grasp is for cubes (red_cube, the pool boxes) and cylinders (blue_cylinder, the pool cylinders) only; crane: plan: IK found no pose for rung +10
- 1 × pick / shape: side-on: shape: foam_block is a foam box; the side-on grasp is for cubes (red_cube, the pool boxes) and cylinders (blue_cylinder, the pool cylinders) only; crane: pad_route: finger pad route clearance -2.50 mm < 0.494 mm from rung +9

## pool_box_1

Feasible: **8 of 9** ordered pairs.  r1c1→, r1c2→, r1c3→, r2c1→, r2c2→, r2c3→, r3c2→, r3c3→

| from → to | feasible | grasp used | margin mm (side-on / crane) | refused at | stage | reason |
|---|---|---|---|---|---|---|
| r1c1 → (lift) | **yes** | side-on | 0.97 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c2 → (lift) | **yes** | side-on | 0.73 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c3 → (lift) | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +4 |
| r2c1 → (lift) | **yes** | side-on | 0.91 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c2 → (lift) | **yes** | side-on | 1.01 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c3 → (lift) | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +12 |
| r3c1 → (lift) | no |  |  | pick | plan | side-on: plan: IK found no level grasp pose for this approach (best of 4 side-on approaches); crane: plan: IK found no grasp pose in the bounded start set |
| r3c2 → (lift) | **yes** | side-on | 0.98 / — |  |  | crane: table: finger_pad -4.4 mm into the table on 'lift grasp->+lift' (commanded) |
| r3c3 → (lift) | **yes** | side-on | 0.98 / — |  |  | crane: table: thumb_pad -1.9 mm into the table on 'descent hover->grasp' (commanded) |

Refusals by reason:

- 1 × pick / plan: side-on: plan: IK found no level grasp pose for this approach (best of 4 side-on approaches); crane: plan: IK found no grasp pose in the bounded start set

## blue_cylinder

Feasible: **0 of 9** ordered pairs.

| from → to | feasible | grasp used | margin mm (side-on / crane) | refused at | stage | reason |
|---|---|---|---|---|---|---|
| r1c1 → (lift) | no |  |  | pick | opening | side-on: opening: blue_cylinder is 70.0 mm across; the open jaw leaves 67.8 mm at -65 deg (69.0 mm at the widest, -68 deg); crane: plan: blue_cylinder is a cylinder; the crane pick closes on two flat faces |
| r1c2 → (lift) | no |  |  | pick | opening | side-on: opening: blue_cylinder is 70.0 mm across; the open jaw leaves 67.8 mm at -65 deg (69.0 mm at the widest, -68 deg); crane: plan: blue_cylinder is a cylinder; the crane pick closes on two flat faces |
| r1c3 → (lift) | no |  |  | pick | opening | side-on: opening: blue_cylinder is 70.0 mm across; the open jaw leaves 67.8 mm at -65 deg (69.0 mm at the widest, -68 deg); crane: plan: blue_cylinder is a cylinder; the crane pick closes on two flat faces |
| r2c1 → (lift) | no |  |  | pick | opening | side-on: opening: blue_cylinder is 70.0 mm across; the open jaw leaves 67.8 mm at -65 deg (69.0 mm at the widest, -68 deg); crane: plan: blue_cylinder is a cylinder; the crane pick closes on two flat faces |
| r2c2 → (lift) | no |  |  | pick | opening | side-on: opening: blue_cylinder is 70.0 mm across; the open jaw leaves 67.8 mm at -65 deg (69.0 mm at the widest, -68 deg); crane: plan: blue_cylinder is a cylinder; the crane pick closes on two flat faces |
| r2c3 → (lift) | no |  |  | pick | opening | side-on: opening: blue_cylinder is 70.0 mm across; the open jaw leaves 67.8 mm at -65 deg (69.0 mm at the widest, -68 deg); crane: plan: blue_cylinder is a cylinder; the crane pick closes on two flat faces |
| r3c1 → (lift) | no |  |  | pick | opening | side-on: opening: blue_cylinder is 70.0 mm across; the open jaw leaves 67.8 mm at -65 deg (69.0 mm at the widest, -68 deg); crane: plan: blue_cylinder is a cylinder; the crane pick closes on two flat faces |
| r3c2 → (lift) | no |  |  | pick | opening | side-on: opening: blue_cylinder is 70.0 mm across; the open jaw leaves 67.8 mm at -65 deg (69.0 mm at the widest, -68 deg); crane: plan: blue_cylinder is a cylinder; the crane pick closes on two flat faces |
| r3c3 → (lift) | no |  |  | pick | opening | side-on: opening: blue_cylinder is 70.0 mm across; the open jaw leaves 67.8 mm at -65 deg (69.0 mm at the widest, -68 deg); crane: plan: blue_cylinder is a cylinder; the crane pick closes on two flat faces |

Refusals by reason:

- 9 × pick / opening: side-on: opening: blue_cylinder is 70.0 mm across; the open jaw leaves 67.8 mm at -65 deg (69.0 mm at the widest, -68 deg); crane: plan: blue_cylinder is a cylinder; the crane pick closes on two flat faces

## pool_cyl_1

Feasible: **8 of 9** ordered pairs.  r1c1→, r1c2→, r1c3→, r2c1→, r2c2→, r2c3→, r3c2→, r3c3→

| from → to | feasible | grasp used | margin mm (side-on / crane) | refused at | stage | reason |
|---|---|---|---|---|---|---|
| r1c1 → (lift) | **yes** | side-on | 0.97 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r1c2 → (lift) | **yes** | side-on | 0.97 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r1c3 → (lift) | **yes** | side-on | 0.00 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r2c1 → (lift) | **yes** | side-on | 0.96 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r2c2 → (lift) | **yes** | side-on | 0.91 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r2c3 → (lift) | **yes** | side-on | 0.92 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r3c1 → (lift) | no |  |  | pick | plan | side-on: plan: IK found no level grasp pose for this approach (best of 24 side-on approaches); crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r3c2 → (lift) | **yes** | side-on | 0.95 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r3c3 → (lift) | **yes** | side-on | 0.96 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |

Refusals by reason:

- 1 × pick / plan: side-on: plan: IK found no level grasp pose for this approach (best of 24 side-on approaches); crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces

## soda_can

Feasible: **0 of 9** ordered pairs.

| from → to | feasible | grasp used | margin mm (side-on / crane) | refused at | stage | reason |
|---|---|---|---|---|---|---|
| r1c1 → (lift) | no |  |  | pick | shape | side-on: shape: soda_can is a can cylinder; the side-on grasp is for cubes (red_cube, the pool boxes) and cylinders (blue_cylinder, the pool cylinders) only; crane: plan: soda_can is a cylinder; the crane pick closes on two flat faces |
| r1c2 → (lift) | no |  |  | pick | shape | side-on: shape: soda_can is a can cylinder; the side-on grasp is for cubes (red_cube, the pool boxes) and cylinders (blue_cylinder, the pool cylinders) only; crane: plan: soda_can is a cylinder; the crane pick closes on two flat faces |
| r1c3 → (lift) | no |  |  | pick | shape | side-on: shape: soda_can is a can cylinder; the side-on grasp is for cubes (red_cube, the pool boxes) and cylinders (blue_cylinder, the pool cylinders) only; crane: plan: soda_can is a cylinder; the crane pick closes on two flat faces |
| r2c1 → (lift) | no |  |  | pick | shape | side-on: shape: soda_can is a can cylinder; the side-on grasp is for cubes (red_cube, the pool boxes) and cylinders (blue_cylinder, the pool cylinders) only; crane: plan: soda_can is a cylinder; the crane pick closes on two flat faces |
| r2c2 → (lift) | no |  |  | pick | shape | side-on: shape: soda_can is a can cylinder; the side-on grasp is for cubes (red_cube, the pool boxes) and cylinders (blue_cylinder, the pool cylinders) only; crane: plan: soda_can is a cylinder; the crane pick closes on two flat faces |
| r2c3 → (lift) | no |  |  | pick | shape | side-on: shape: soda_can is a can cylinder; the side-on grasp is for cubes (red_cube, the pool boxes) and cylinders (blue_cylinder, the pool cylinders) only; crane: plan: soda_can is a cylinder; the crane pick closes on two flat faces |
| r3c1 → (lift) | no |  |  | pick | shape | side-on: shape: soda_can is a can cylinder; the side-on grasp is for cubes (red_cube, the pool boxes) and cylinders (blue_cylinder, the pool cylinders) only; crane: plan: soda_can is a cylinder; the crane pick closes on two flat faces |
| r3c2 → (lift) | no |  |  | pick | shape | side-on: shape: soda_can is a can cylinder; the side-on grasp is for cubes (red_cube, the pool boxes) and cylinders (blue_cylinder, the pool cylinders) only; crane: plan: soda_can is a cylinder; the crane pick closes on two flat faces |
| r3c3 → (lift) | no |  |  | pick | shape | side-on: shape: soda_can is a can cylinder; the side-on grasp is for cubes (red_cube, the pool boxes) and cylinders (blue_cylinder, the pool cylinders) only; crane: plan: soda_can is a cylinder; the crane pick closes on two flat faces |

Refusals by reason:

- 9 × pick / shape: side-on: shape: soda_can is a can cylinder; the side-on grasp is for cubes (red_cube, the pool boxes) and cylinders (blue_cylinder, the pool cylinders) only; crane: plan: soda_can is a cylinder; the crane pick closes on two flat faces

