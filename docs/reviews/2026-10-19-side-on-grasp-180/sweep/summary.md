# Sweep summary (S = side-on, C = crane, · = refused, — = same cell)

## Lifts (grasp best)

| object | r1c1 | r1c2 | r1c3 | r2c1 | r2c2 | r2c3 | r3c1 | r3c2 | r3c3 | feasible |
|---|---|---|---|---|---|---|---|---|---|---|
| red_cube | S 1.0 | S 0.0 | S 0.0 | S 0.99 | S 1.01 | S 0.0 | · | S 1.02 | S 0.97 | 8/9 |
| pool_box_1 | S 0.97 | S 0.73 | S 0.0 | S 0.91 | S 1.01 | S 0.0 | · | S 0.98 | S 0.98 | 8/9 |
| pool_cyl_1 | S 0.97 | S 0.97 | S 0.0 | S 0.96 | S 0.91 | S 0.92 | · | S 0.95 | S 0.96 | 8/9 |
| blue_cylinder | · | · | · | · | · | · | · | · | · | 0/9 |
| foam_block | · | · | · | · | · | · | · | · | · | 0/9 |
| soda_can | · | · | · | · | · | · | · | · | · | 0/9 |

Lift refusal reasons:

- red_cube: 1 x side-on: plan: IK found no level grasp pose for this approach (best of 4 side-on approaches); crane: plan: IK found no grasp pose in the bounded start set
- pool_box_1: 1 x side-on: plan: IK found no level grasp pose for this approach (best of 4 side-on approaches); crane: plan: IK found no grasp pose in the bounded start set
- pool_cyl_1: 1 x side-on: plan: IK found no level grasp pose for this approach (best of 24 side-on approaches); crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces
- blue_cylinder: 9 x side-on: opening: blue_cylinder is 70.0 mm across; the open jaw leaves 67.8 mm at -65 deg (69.0 mm at the widest, -68 deg); crane: plan: blue_cylinder is a cylinder; the crane pick closes on two flat faces
- foam_block: 6 x side-on: shape: foam_block is a foam box; the side-on grasp is for cubes (red_cube, the pool boxes) and cylinders (blue_cylinder, the pool cylinders) only; crane: plan: IK found no grasp pose in the bounded start set
- foam_block: 1 x side-on: shape: foam_block is a foam box; the side-on grasp is for cubes (red_cube, the pool boxes) and cylinders (blue_cylinder, the pool cylinders) only; crane: plan: IK found no pose for rung +2
- foam_block: 1 x side-on: shape: foam_block is a foam box; the side-on grasp is for cubes (red_cube, the pool boxes) and cylinders (blue_cylinder, the pool cylinders) only; crane: plan: IK found no pose for rung +10
- foam_block: 1 x side-on: shape: foam_block is a foam box; the side-on grasp is for cubes (red_cube, the pool boxes) and cylinders (blue_cylinder, the pool cylinders) only; crane: pad_route: finger pad route clearance -2.50 mm < 0.494 mm from rung +9
- soda_can: 9 x side-on: shape: soda_can is a can cylinder; the side-on grasp is for cubes (red_cube, the pool boxes) and cylinders (blue_cylinder, the pool cylinders) only; crane: plan: soda_can is a cylinder; the crane pick closes on two flat faces

## Placement, grasp best: red_cube

Feasible pairs: **56 of 72**; by grasp: {'side-on': 56}

| from \ to | r1c1 | r1c2 | r1c3 | r2c1 | r2c2 | r2c3 | r3c1 | r3c2 | r3c3 |
|---|---|---|---|---|---|---|---|---|---|
| **r1c1** | — | S | S | S | S | S | · | S | S |
| **r1c2** | S | — | S | S | S | S | · | S | S |
| **r1c3** | S | S | — | S | S | S | · | S | S |
| **r2c1** | S | S | S | — | S | S | · | S | S |
| **r2c2** | S | S | S | S | — | S | · | S | S |
| **r2c3** | S | S | S | S | S | — | · | S | S |
| **r3c1** | · | · | · | · | · | · | — | · | · |
| **r3c2** | S | S | S | S | S | S | · | — | S |
| **r3c3** | S | S | S | S | S | S | · | S | — |

Both grasps feasible (3 pairs): used / margins mm side-on, crane

| from → to | used | side-on | crane |
|---|---|---|---|
| r3c2 → r2c1 | side-on | 0.61 | 0.0 |
| r3c2 → r2c2 | side-on | 1.02 | 0.0 |
| r3c2 → r3c3 | side-on | 1.02 | 0.0 |

Refusal reasons tallied:

- 8 x side-on: plan: IK found no level grasp pose for this approach; crane: plan: IK found no grasp pose in the bounded start set
- 4 x side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no grasp pose in the bounded start set
- 1 x side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no pose for rung +1
- 1 x side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no pose for rung +10
- 1 x side-on: destination: IK found no level pose that sets the object down on the target; crane: destination: IK found no pose that sets the object down on the target with the grasp's attitude
- 1 x side-on: destination: IK found no level pose that sets the object down on the target; crane: straddle: predicted grasp pose does not straddle: ['finger_above_top']

## Placement, grasp best: pool_box_1

Feasible pairs: **56 of 72**; by grasp: {'side-on': 56}

| from \ to | r1c1 | r1c2 | r1c3 | r2c1 | r2c2 | r2c3 | r3c1 | r3c2 | r3c3 |
|---|---|---|---|---|---|---|---|---|---|
| **r1c1** | — | S | S | S | S | S | · | S | S |
| **r1c2** | S | — | S | S | S | S | · | S | S |
| **r1c3** | S | S | — | S | S | S | · | S | S |
| **r2c1** | S | S | S | — | S | S | · | S | S |
| **r2c2** | S | S | S | S | — | S | · | S | S |
| **r2c3** | S | S | S | S | S | — | · | S | S |
| **r3c1** | · | · | · | · | · | · | — | · | · |
| **r3c2** | S | S | S | S | S | S | · | — | S |
| **r3c3** | S | S | S | S | S | S | · | S | — |

Pairs where both grasps were feasible: none.

Refusal reasons tallied:

- 8 x side-on: plan: IK found no level grasp pose for this approach; crane: plan: IK found no grasp pose in the bounded start set
- 4 x side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no grasp pose in the bounded start set
- 1 x side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no pose for rung +4
- 1 x side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no pose for rung +12
- 1 x side-on: destination: IK found no level pose that sets the object down on the target; crane: table: finger_pad -4.4 mm into the table on 'lift grasp->+lift' (commanded)
- 1 x side-on: destination: IK found no level pose that sets the object down on the target; crane: table: thumb_pad -1.9 mm into the table on 'descent hover->grasp' (commanded)

## Placement, grasp best: pool_cyl_1

Feasible pairs: **56 of 72**; by grasp: {'side-on': 56}

| from \ to | r1c1 | r1c2 | r1c3 | r2c1 | r2c2 | r2c3 | r3c1 | r3c2 | r3c3 |
|---|---|---|---|---|---|---|---|---|---|
| **r1c1** | — | S | S | S | S | S | · | S | S |
| **r1c2** | S | — | S | S | S | S | · | S | S |
| **r1c3** | S | S | — | S | S | S | · | S | S |
| **r2c1** | S | S | S | — | S | S | · | S | S |
| **r2c2** | S | S | S | S | — | S | · | S | S |
| **r2c3** | S | S | S | S | S | — | · | S | S |
| **r3c1** | · | · | · | · | · | · | — | · | · |
| **r3c2** | S | S | S | S | S | S | · | — | S |
| **r3c3** | S | S | S | S | S | S | · | S | — |

Pairs where both grasps were feasible: none.

Refusal reasons tallied:

- 8 x side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces
- 8 x side-on: plan: IK found no level grasp pose for this approach; crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces

## Placement, crane only (main): red_cube

Feasible pairs: **3 of 72**; by grasp: {'crane': 3}

| from \ to | r1c1 | r1c2 | r1c3 | r2c1 | r2c2 | r2c3 | r3c1 | r3c2 | r3c3 |
|---|---|---|---|---|---|---|---|---|---|
| **r1c1** | — | · | · | · | · | · | · | · | · |
| **r1c2** | · | — | · | · | · | · | · | · | · |
| **r1c3** | · | · | — | · | · | · | · | · | · |
| **r2c1** | · | · | · | — | · | · | · | · | · |
| **r2c2** | · | · | · | · | — | · | · | · | · |
| **r2c3** | · | · | · | · | · | — | · | · | · |
| **r3c1** | · | · | · | · | · | · | — | · | · |
| **r3c2** | · | · | · | C | C | · | · | — | C |
| **r3c3** | · | · | · | · | · | · | · | · | — |

Pairs where both grasps were feasible: none.

Refusal reasons tallied:

- 40 x plan: IK found no grasp pose in the bounded start set
- 8 x plan: IK found no pose for rung +1
- 8 x plan: IK found no pose for rung +10
- 8 x straddle: predicted grasp pose does not straddle: ['finger_above_top']
- 4 x destination: IK found no pose that sets the object down on the target with the grasp's attitude
- 1 x plan: IK found no pose for rung +2

## Placement, crane only (main): pool_box_1

Feasible pairs: **0 of 72**; by grasp: {}

| from \ to | r1c1 | r1c2 | r1c3 | r2c1 | r2c2 | r2c3 | r3c1 | r3c2 | r3c3 |
|---|---|---|---|---|---|---|---|---|---|
| **r1c1** | — | · | · | · | · | · | · | · | · |
| **r1c2** | · | — | · | · | · | · | · | · | · |
| **r1c3** | · | · | — | · | · | · | · | · | · |
| **r2c1** | · | · | · | — | · | · | · | · | · |
| **r2c2** | · | · | · | · | — | · | · | · | · |
| **r2c3** | · | · | · | · | · | — | · | · | · |
| **r3c1** | · | · | · | · | · | · | — | · | · |
| **r3c2** | · | · | · | · | · | · | · | — | · |
| **r3c3** | · | · | · | · | · | · | · | · | — |

Pairs where both grasps were feasible: none.

Refusal reasons tallied:

- 40 x plan: IK found no grasp pose in the bounded start set
- 8 x plan: IK found no pose for rung +4
- 8 x plan: IK found no pose for rung +12
- 8 x table: finger_pad -4.4 mm into the table on 'lift grasp->+lift' (commanded)
- 8 x table: thumb_pad -1.9 mm into the table on 'descent hover->grasp' (commanded)

