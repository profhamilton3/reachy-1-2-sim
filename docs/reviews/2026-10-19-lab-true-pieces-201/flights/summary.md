# #201 offline flights — summary (pads condim 4; pool box 50.8 mm cube, pool cylinder 45.7 mm dia x 71.1 mm)

Offline physics (`tools/place_sweep/fly.py`: the executor on the live model's physics, host IK mirror).  Scene FWDCenterLabSivaPool, the object upright on the origin cell, nothing else on the board unless stated.  Grasp `best` = the planner choice.  ✓ every #174 criterion true; ✗ flew with a criterion false; H halted or refused in flight; R refused before motion; E crashed.

## Lifts (grasp best)

| object | r1c1 | r1c2 | r1c3 | r2c1 | r2c2 | r2c3 | r3c1 | r3c2 | r3c3 | counts |
|---|---|---|---|---|---|---|---|---|---|---|
| pool_box_1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | R | ✓ | ✓ | ✓ 8, R 1 |
| pool_cyl_1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | R | ✓ | ✓ | ✓ 8, R 1 |
| red_cube | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | R | ✓ | ✓ | ✓ 8, R 1 |

| flight | result | grasp | az / pitch | error mm | tilt ° | slip mm | drag mm | way out | others moved mm | failed / halt / refusal |
|---|---|---|---|---|---|---|---|---|---|---|
| red_cube r1c1 | ✓ | side-on | 90 / 5.0 | 1.97 | 0.00 | 0.003 | 2.91 | direct | 0.0000 |  |
| red_cube r1c2 | ✓ | side-on | 90 / 5.0 | 1.72 | 0.00 | 0.001 | 2.64 | direct | 0.0000 |  |
| red_cube r1c3 | ✓ | side-on | -0 / 5.0 | 2.29 | 0.00 | 0.000 | 2.84 | direct | 0.0000 |  |
| red_cube r2c1 | ✓ | side-on | 90 / 5.0 | 5.80 | 0.00 | 0.002 | 2.56 | direct | 0.0000 |  |
| red_cube r2c2 | ✓ | side-on | -0 / 5.0 | 1.82 | 0.00 | 0.003 | 2.26 | direct | 0.0000 |  |
| red_cube r2c3 | ✓ | side-on | -0 / 5.0 | 1.73 | 0.00 | 0.001 | 2.24 | direct | 0.0000 |  |
| red_cube r3c1 | R |  |  |  |  |  |  |  |  | plan: side-on: plan: IK found no level grasp pose for this approach (best of 4 side-on approaches); crane: plan: IK found no grasp pose in the bounded start set |
| red_cube r3c2 | ✓ | side-on | 0 / 0.0 | 1.43 | 0.00 | 0.001 | 2.73 | direct | 0.0000 |  |
| red_cube r3c3 | ✓ | side-on | 0 / 5.0 | 1.48 | 0.00 | 0.000 | 2.23 | direct | 0.0000 |  |
| pool_box_1 r1c1 | ✓ | side-on | 90 / 5.0 | 4.29 | 0.00 | 0.002 | 1.45 | direct | 0.0000 |  |
| pool_box_1 r1c2 | ✓ | side-on | 90 / 0.0 | 3.44 | 0.00 | 0.000 | 1.07 | direct | 0.0000 |  |
| pool_box_1 r1c3 | ✓ | side-on | 0 / 5.0 | 3.69 | 0.00 | 0.000 | 1.41 | direct | 0.0000 |  |
| pool_box_1 r2c1 | ✓ | side-on | 90 / 0.0 | 3.77 | 0.00 | 0.001 | 0.98 | direct | 0.0000 |  |
| pool_box_1 r2c2 | ✓ | side-on | 0 / 0.0 | 3.38 | 0.00 | 0.001 | 1.38 | direct | 0.0000 |  |
| pool_box_1 r2c3 | ✓ | side-on | 0 / 5.0 | 3.01 | 0.00 | 0.000 | 1.28 | direct | 0.0000 |  |
| pool_box_1 r3c1 | R |  |  |  |  |  |  |  |  | plan: side-on: plan: IK found no level grasp pose for this approach (best of 4 side-on approaches); crane: plan: IK found no grasp pose in the bounded start set |
| pool_box_1 r3c2 | ✓ | side-on | 0 / 0.0 | 4.48 | 0.00 | 0.001 | 1.23 | direct | 0.0000 |  |
| pool_box_1 r3c3 | ✓ | side-on | 0 / 5.0 | 3.28 | 0.00 | 0.002 | 1.42 | direct | 0.0000 |  |
| pool_cyl_1 r1c1 | ✓ | side-on | 90 / 5.0 | 3.08 | 0.00 | 0.004 | 1.38 | direct | 0.0000 |  |
| pool_cyl_1 r1c2 | ✓ | side-on | 30 / 5.0 | 3.64 | 0.00 | 0.004 | 1.21 | direct | 0.0000 |  |
| pool_cyl_1 r1c3 | ✓ | side-on | 0 / 5.0 | 3.33 | 0.00 | 0.003 | 0.78 | direct | 0.0000 |  |
| pool_cyl_1 r2c1 | ✓ | side-on | 90 / 5.0 | 6.34 | 0.00 | 0.005 | 0.84 | direct | 0.0000 |  |
| pool_cyl_1 r2c2 | ✓ | side-on | 75 / 5.0 | 6.72 | 0.00 | 0.004 | 0.48 | direct | 0.0000 |  |
| pool_cyl_1 r2c3 | ✓ | side-on | 30 / 5.0 | 4.18 | 0.00 | 0.003 | 0.08 | direct | 0.0000 |  |
| pool_cyl_1 r3c1 | R |  |  |  |  |  |  |  |  | plan: side-on: plan: IK found no level grasp pose for this approach (best of 24 side-on approaches); crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| pool_cyl_1 r3c2 | ✓ | side-on | 15 / 5.0 | 2.90 | 0.00 | 0.003 | 1.99 | direct | 0.0000 |  |
| pool_cyl_1 r3c3 | ✓ | side-on | 45 / 5.0 | 7.90 | 0.00 | 0.003 | 0.99 | direct | 0.0000 |  |

## Placements: red_cube (grasp best)

All 72 ordered pairs: ✓ 52, ✗ 2, H 2, R 16.  Flown: 56; every criterion: 52.

| from \ to | r1c1 | r1c2 | r1c3 | r2c1 | r2c2 | r2c3 | r3c1 | r3c2 | r3c3 |
|---|---|---|---|---|---|---|---|---|---|
| r1c1 | — | ✓ | ✓ | ✓ | ✓ | ✓ | R | ✓ | ✓ |
| r1c2 | ✓ | — | ✓ | ✓ | ✓ | ✓ | R | ✓ | ✓ |
| r1c3 | ✓ | ✓ | — | ✓ | ✓ | ✓ | R | ✓ | ✓ |
| r2c1 | ✓ | ✓ | ✗ | — | ✓ | ✓ | R | ✓ | ✓ |
| r2c2 | ✓ | ✓ | ✗ | ✓ | — | ✓ | R | ✓ | ✓ |
| r2c3 | ✓ | ✓ | ✓ | ✓ | ✓ | — | R | ✓ | ✓ |
| r3c1 | R | R | R | R | R | R | — | R | R |
| r3c2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | R | — | ✓ |
| r3c3 | ✓ | ✓ | H | ✓ | ✓ | H | R | ✓ | — |

Flown pairs:

| flight | result | grasp | az / pitch | error mm | tilt ° | slip mm | drag mm | way out | others moved mm | failed / halt / refusal |
|---|---|---|---|---|---|---|---|---|---|---|
| r1c1 → r1c2 | ✓ | side-on | 90 / None | 3.24 | 0.00 | 0.003 | 2.83 | direct | 0.0000 |  |
| r1c1 → r1c3 | ✓ | side-on | 90 / None | 2.53 | 0.00 | 0.003 | 2.51 | direct | 0.0000 |  |
| r1c1 → r2c1 | ✓ | side-on | 90 / None | 3.10 | 0.00 | 0.003 | 2.80 | level | 0.0000 |  |
| r1c1 → r2c2 | ✓ | side-on | 90 / None | 3.12 | 0.00 | 0.003 | 2.98 | direct | 0.0000 |  |
| r1c1 → r2c3 | ✓ | side-on | 90 / None | 1.15 | 0.00 | 0.003 | 2.42 | sideways | 0.0000 |  |
| r1c1 → r3c2 | ✓ | side-on | 90 / None | 3.02 | 0.00 | 0.003 | 2.71 | level | 0.0000 |  |
| r1c1 → r3c3 | ✓ | side-on | 90 / None | 2.83 | 0.00 | 0.003 | 2.57 | direct | 0.0000 |  |
| r1c2 → r1c1 | ✓ | side-on | 90 / None | 2.95 | 0.00 | 0.001 | 2.82 | direct | 0.0000 |  |
| r1c2 → r1c3 | ✓ | side-on | 90 / None | 2.57 | 0.00 | 0.001 | 2.62 | direct | 0.0000 |  |
| r1c2 → r2c1 | ✓ | side-on | 90 / None | 3.21 | 0.00 | 0.001 | 2.94 | direct | 0.0000 |  |
| r1c2 → r2c2 | ✓ | side-on | 90 / None | 3.10 | 0.00 | 0.001 | 2.96 | direct | 0.0000 |  |
| r1c2 → r2c3 | ✓ | side-on | 90 / None | 1.24 | 0.00 | 0.001 | 2.72 | sideways | 0.0000 |  |
| r1c2 → r3c2 | ✓ | side-on | 90 / None | 3.25 | 0.00 | 0.001 | 2.96 | direct | 0.0000 |  |
| r1c2 → r3c3 | ✓ | side-on | 90 / None | 3.10 | 0.00 | 0.001 | 2.86 | direct | 0.0000 |  |
| r1c3 → r1c1 | ✓ | side-on | -0 / None | 2.79 | 0.00 | 0.000 | 2.53 | direct | 0.0000 |  |
| r1c3 → r1c2 | ✓ | side-on | -0 / None | 3.64 | 0.00 | 0.000 | 3.05 | direct | 0.0000 |  |
| r1c3 → r2c1 | ✓ | side-on | -0 / None | 2.38 | 0.00 | 0.000 | 2.09 | direct | 0.0000 |  |
| r1c3 → r2c2 | ✓ | side-on | -0 / None | 2.46 | 0.00 | 0.000 | 2.22 | direct | 0.0000 |  |
| r1c3 → r2c3 | ✓ | side-on | -0 / None | 2.83 | 0.00 | 0.000 | 2.69 | direct | 0.0000 |  |
| r1c3 → r3c2 | ✓ | side-on | -0 / None | 2.55 | 0.00 | 0.000 | 2.26 | direct | 0.0000 |  |
| r1c3 → r3c3 | ✓ | side-on | -0 / None | 2.75 | 0.00 | 0.000 | 2.47 | direct | 0.0000 |  |
| r2c1 → r1c1 | ✓ | side-on | 90 / None | 2.62 | 0.00 | 0.002 | 2.53 | level | 0.0000 |  |
| r2c1 → r1c2 | ✓ | side-on | 90 / None | 2.80 | 0.00 | 0.002 | 2.67 | direct | 0.0000 |  |
| r2c1 → r1c3 | ✗ | side-on | 90 / None | 342.98 | 0.00 | 0.002 | 2.56 | direct | 0.0000 | placed_on_target |
| r2c1 → r2c2 | ✓ | side-on | 90 / None | 3.39 | 0.00 | 0.002 | 2.74 | direct | 0.0000 |  |
| r2c1 → r2c3 | ✓ | side-on | 90 / None | 2.75 | 0.00 | 0.002 | 2.62 | direct | 0.0000 |  |
| r2c1 → r3c2 | ✓ | side-on | 90 / None | 2.76 | 0.00 | 0.002 | 2.44 | level | 0.0000 |  |
| r2c1 → r3c3 | ✓ | side-on | 90 / None | 2.74 | 0.00 | 0.002 | 2.51 | level | 0.0000 |  |
| r2c2 → r1c1 | ✓ | side-on | -0 / None | 2.88 | 0.00 | 0.003 | 2.58 | direct | 0.0000 |  |
| r2c2 → r1c2 | ✓ | side-on | -0 / None | 3.14 | 0.00 | 0.003 | 2.42 | direct | 0.0000 |  |
| r2c2 → r1c3 | ✗ | side-on | 90 / None | 214.39 | 0.00 | 0.004 | 2.35 | direct | 0.0000 | placed_on_target |
| r2c2 → r2c1 | ✓ | side-on | -0 / None | 2.83 | 0.00 | 0.003 | 2.49 | direct | 0.0000 |  |
| r2c2 → r2c3 | ✓ | side-on | 90 / None | 2.64 | 0.00 | 0.003 | 2.61 | direct | 0.0000 |  |
| r2c2 → r3c2 | ✓ | side-on | -0 / None | 2.49 | 0.00 | 0.003 | 2.14 | direct | 0.0000 |  |
| r2c2 → r3c3 | ✓ | side-on | 90 / None | 2.30 | 0.00 | 0.003 | 2.58 | direct | 0.0000 |  |
| r2c3 → r1c1 | ✓ | side-on | -0 / None | 2.42 | 0.00 | 0.001 | 2.19 | direct | 0.0000 |  |
| r2c3 → r1c2 | ✓ | side-on | -0 / None | 2.94 | 0.00 | 0.001 | 2.36 | direct | 0.0000 |  |
| r2c3 → r1c3 | ✓ | side-on | -0 / None | 2.21 | 0.00 | 0.001 | 2.34 | sideways | 0.0000 |  |
| r2c3 → r2c1 | ✓ | side-on | -0 / None | 2.77 | 0.00 | 0.001 | 2.43 | direct | 0.0000 |  |
| r2c3 → r2c2 | ✓ | side-on | -0 / None | 2.62 | 0.00 | 0.001 | 2.35 | direct | 0.0000 |  |
| r2c3 → r3c2 | ✓ | side-on | -0 / None | 2.64 | 0.00 | 0.001 | 2.29 | direct | 0.0000 |  |
| r2c3 → r3c3 | ✓ | side-on | -0 / None | 2.43 | 0.00 | 0.001 | 2.13 | direct | 0.0000 |  |
| r3c2 → r1c1 | ✓ | side-on | 0 / None | 2.58 | 0.00 | 0.001 | 2.31 | level | 0.0000 |  |
| r3c2 → r1c2 | ✓ | side-on | 0 / None | 2.76 | 0.00 | 0.001 | 2.10 | direct | 0.0000 |  |
| r3c2 → r1c3 | ✓ | side-on | 0 / None | 2.36 | 0.00 | 0.001 | 2.36 | direct | 0.0000 |  |
| r3c2 → r2c1 | ✓ | side-on | 0 / None | 2.78 | 0.00 | 0.001 | 2.42 | level | 0.0000 |  |
| r3c2 → r2c2 | ✓ | side-on | 0 / None | 2.98 | 0.00 | 0.001 | 2.71 | level | 0.0000 |  |
| r3c2 → r2c3 | ✓ | side-on | 0 / None | 2.83 | 0.00 | 0.001 | 2.60 | direct | 0.0000 |  |
| r3c2 → r3c3 | ✓ | side-on | 0 / None | 2.72 | 0.00 | 0.001 | 2.39 | level | 0.0000 |  |
| r3c3 → r1c1 | ✓ | side-on | 0 / None | 2.53 | 0.00 | 0.000 | 2.34 | direct | 0.0000 |  |
| r3c3 → r1c2 | ✓ | side-on | 0 / None | 2.88 | 0.00 | 0.000 | 2.35 | direct | 0.0000 |  |
| r3c3 → r1c3 | H | side-on | 0 / None | 304.80 | 0.00 |  |  |  | 0.0000 | refused at rung +1 -- table: forearm -0.0 mm into the table on 'descent hover->grasp' (predicted) |
| r3c3 → r2c1 | ✓ | side-on | 0 / None | 2.79 | 0.00 | 0.000 | 2.45 | direct | 0.0000 |  |
| r3c3 → r2c2 | ✓ | side-on | 0 / None | 2.78 | 0.00 | 0.000 | 2.53 | direct | 0.0000 |  |
| r3c3 → r2c3 | H | side-on | 0 / None | 152.40 | 0.00 |  |  |  | 0.0000 | refused at rung +1 -- table: forearm -0.0 mm into the table on 'descent hover->grasp' (predicted) |
| r3c3 → r3c2 | ✓ | side-on | 0 / None | 2.85 | 0.00 | 0.000 | 2.50 | direct | 0.0000 |  |

Refused before motion:

- 8 × plan: side-on: plan: IK found no level grasp pose for this approach; crane: plan: IK found no grasp pose in the bounded start set
- 4 × destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no grasp pose in the bounded start set
- 1 × destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no pose for rung +1
- 1 × destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no pose for rung +10
- 1 × destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: destination: IK found no pose that sets the object down on the target with the grasp's attitude
- 1 × destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: straddle: predicted grasp pose does not straddle: ['finger_above_top']

## Placements: pool_box_1 (grasp best)

All 72 ordered pairs: ✓ 51, ✗ 3, H 2, R 16.  Flown: 56; every criterion: 51.

| from \ to | r1c1 | r1c2 | r1c3 | r2c1 | r2c2 | r2c3 | r3c1 | r3c2 | r3c3 |
|---|---|---|---|---|---|---|---|---|---|
| r1c1 | — | ✓ | ✓ | ✓ | ✓ | ✓ | R | ✓ | ✓ |
| r1c2 | H | — | ✓ | ✓ | ✓ | ✓ | R | ✓ | ✓ |
| r1c3 | ✓ | ✓ | — | ✓ | ✓ | ✓ | R | ✓ | ✓ |
| r2c1 | ✓ | ✓ | ✓ | — | ✓ | ✓ | R | ✓ | ✓ |
| r2c2 | ✓ | ✓ | ✗ | ✓ | — | ✓ | R | ✓ | H |
| r2c3 | ✓ | ✓ | ✓ | ✓ | ✓ | — | R | ✓ | ✓ |
| r3c1 | R | R | R | R | R | R | — | R | R |
| r3c2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | R | — | ✓ |
| r3c3 | ✓ | ✓ | ✗ | ✓ | ✗ | ✓ | R | ✓ | — |

Flown pairs:

| flight | result | grasp | az / pitch | error mm | tilt ° | slip mm | drag mm | way out | others moved mm | failed / halt / refusal |
|---|---|---|---|---|---|---|---|---|---|---|
| r1c1 → r1c2 | ✓ | side-on | 90 / None | 1.72 | 0.00 | 0.002 | 1.37 | direct | 0.0000 |  |
| r1c1 → r1c3 | ✓ | side-on | 90 / None | 1.72 | 0.00 | 0.002 | 1.76 | sideways | 0.0000 |  |
| r1c1 → r2c1 | ✓ | side-on | 90 / None | 1.73 | 0.00 | 0.002 | 1.44 | direct | 0.0000 |  |
| r1c1 → r2c2 | ✓ | side-on | 90 / None | 1.72 | 0.00 | 0.002 | 1.53 | direct | 0.0000 |  |
| r1c1 → r2c3 | ✓ | side-on | 90 / None | 1.90 | 0.00 | 0.002 | 1.67 | direct | 0.0000 |  |
| r1c1 → r3c2 | ✓ | side-on | 90 / None | 1.77 | 0.00 | 0.002 | 1.47 | level | 0.0000 |  |
| r1c1 → r3c3 | ✓ | side-on | 90 / None | 1.88 | 0.00 | 0.002 | 1.62 | direct | 0.0000 |  |
| r1c2 → r1c1 | H | side-on | 90 / None | 1.35 | 0.00 | 0.000 | 1.10 |  | 0.0000 | withdrawal refused before opening: with the grip let go, neither opening has a clear way out, so the hand was not opened further |
| r1c2 → r1c3 | ✓ | side-on | 90 / None | 0.68 | 0.00 | 0.001 | 1.53 | sideways | 0.0000 |  |
| r1c2 → r2c1 | ✓ | side-on | 90 / None | 1.39 | 0.00 | 0.000 | 1.12 | level | 0.0000 |  |
| r1c2 → r2c2 | ✓ | side-on | 90 / None | 2.39 | 0.00 | 0.000 | 1.32 | direct | 0.0000 |  |
| r1c2 → r2c3 | ✓ | side-on | 90 / None | 1.54 | 0.00 | 0.000 | 1.11 | direct | 0.0000 |  |
| r1c2 → r3c2 | ✓ | side-on | 90 / None | 1.53 | 0.00 | 0.000 | 1.19 | level | 0.0000 |  |
| r1c2 → r3c3 | ✓ | side-on | 90 / None | 1.27 | 0.00 | 0.000 | 1.08 | level | 0.0000 |  |
| r1c3 → r1c1 | ✓ | side-on | 0 / None | 1.64 | 0.00 | 0.000 | 1.39 | direct | 0.0000 |  |
| r1c3 → r1c2 | ✓ | side-on | 0 / None | 1.75 | 0.00 | 0.000 | 1.21 | direct | 0.0000 |  |
| r1c3 → r2c1 | ✓ | side-on | 0 / None | 1.64 | 0.00 | 0.000 | 1.35 | direct | 0.0000 |  |
| r1c3 → r2c2 | ✓ | side-on | 0 / None | 1.59 | 0.00 | 0.000 | 1.35 | direct | 0.0000 |  |
| r1c3 → r2c3 | ✓ | side-on | 0 / None | 1.63 | 0.00 | 0.000 | 1.40 | direct | 0.0000 |  |
| r1c3 → r3c2 | ✓ | side-on | 0 / None | 1.94 | 0.00 | 0.000 | 1.63 | direct | 0.0000 |  |
| r1c3 → r3c3 | ✓ | side-on | 0 / None | 1.91 | 0.00 | 0.000 | 1.63 | direct | 0.0000 |  |
| r2c1 → r1c1 | ✓ | side-on | 90 / None | 1.15 | 0.00 | 0.001 | 0.90 | direct | 0.0000 |  |
| r2c1 → r1c2 | ✓ | side-on | 90 / None | 1.22 | 0.00 | 0.001 | 1.12 | direct | 0.0000 |  |
| r2c1 → r1c3 | ✓ | side-on | 90 / None | 1.38 | 0.00 | 0.002 | 1.50 | sideways | 0.0000 |  |
| r2c1 → r2c2 | ✓ | side-on | 90 / None | 2.10 | 0.00 | 0.001 | 1.03 | direct | 0.0000 |  |
| r2c1 → r2c3 | ✓ | side-on | 90 / None | 1.44 | 0.00 | 0.001 | 1.18 | direct | 0.0000 |  |
| r2c1 → r3c2 | ✓ | side-on | 90 / None | 1.45 | 0.00 | 0.001 | 1.11 | level | 0.0000 |  |
| r2c1 → r3c3 | ✓ | side-on | 90 / None | 1.09 | 0.00 | 0.001 | 0.97 | level | 0.0000 |  |
| r2c2 → r1c1 | ✓ | side-on | 0 / None | 1.64 | 0.00 | 0.001 | 1.37 | level | 0.0000 |  |
| r2c2 → r1c2 | ✓ | side-on | 0 / None | 2.14 | 0.00 | 0.001 | 1.63 | direct | 0.0000 |  |
| r2c2 → r1c3 | ✗ | side-on | 90 / None | 213.96 | 0.00 | 0.001 | 1.16 | separating | 0.0000 | placed_on_target |
| r2c2 → r2c1 | ✓ | side-on | 0 / None | 2.04 | 0.00 | 0.001 | 1.76 | direct | 0.0000 |  |
| r2c2 → r2c3 | ✓ | side-on | 0 / None | 2.80 | 0.00 | 0.001 | 1.90 | direct | 0.0000 |  |
| r2c2 → r3c2 | ✓ | side-on | 0 / None | 1.55 | 0.00 | 0.001 | 1.23 | level | 0.0000 |  |
| r2c2 → r3c3 | H | side-on | 90 / None | 1.43 | 0.00 | 0.001 | 1.24 |  | 0.0000 | withdrawal refused before opening: with the grip let go, neither opening has a clear way out, so the hand was not opened further |
| r2c3 → r1c1 | ✓ | side-on | 0 / None | 1.44 | 0.00 | 0.000 | 1.23 | direct | 0.0000 |  |
| r2c3 → r1c2 | ✓ | side-on | 0 / None | 1.59 | 0.00 | 0.000 | 1.21 | direct | 0.0000 |  |
| r2c3 → r1c3 | ✓ | side-on | 0 / None | 0.67 | 0.00 | 0.000 | 1.03 | sideways | 0.0000 |  |
| r2c3 → r2c1 | ✓ | side-on | 0 / None | 1.40 | 0.00 | 0.000 | 1.21 | direct | 0.0000 |  |
| r2c3 → r2c2 | ✓ | side-on | 0 / None | 1.49 | 0.00 | 0.000 | 1.31 | direct | 0.0000 |  |
| r2c3 → r3c2 | ✓ | side-on | 0 / None | 1.47 | 0.00 | 0.000 | 1.23 | direct | 0.0000 |  |
| r2c3 → r3c3 | ✓ | side-on | 0 / None | 1.17 | 0.00 | 0.000 | 1.00 | direct | 0.0000 |  |
| r3c2 → r1c1 | ✓ | side-on | 0 / None | 1.39 | 0.00 | 0.001 | 1.12 | level | 0.0000 |  |
| r3c2 → r1c2 | ✓ | side-on | 0 / None | 1.66 | 0.00 | 0.001 | 1.11 | direct | 0.0000 |  |
| r3c2 → r1c3 | ✓ | side-on | 0 / None | 1.33 | 0.00 | 0.001 | 1.42 | direct | 0.0000 |  |
| r3c2 → r2c1 | ✓ | side-on | 0 / None | 1.46 | 0.00 | 0.001 | 1.15 | level | 0.0000 |  |
| r3c2 → r2c2 | ✓ | side-on | 0 / None | 1.58 | 0.00 | 0.001 | 1.34 | level | 0.0000 |  |
| r3c2 → r2c3 | ✓ | side-on | 0 / None | 1.53 | 0.00 | 0.001 | 1.32 | direct | 0.0000 |  |
| r3c2 → r3c3 | ✓ | side-on | 0 / None | 1.37 | 0.00 | 0.001 | 1.07 | level | 0.0000 |  |
| r3c3 → r1c1 | ✓ | side-on | 0 / None | 1.96 | 0.00 | 0.002 | 1.75 | direct | 0.0000 |  |
| r3c3 → r1c2 | ✓ | side-on | 0 / None | 1.87 | 0.00 | 0.002 | 1.41 | direct | 0.0000 |  |
| r3c3 → r1c3 | ✗ | side-on | 0 / None | 304.26 | 0.00 | 0.001 | 1.76 | direct | 0.0000 | placed_on_target |
| r3c3 → r2c1 | ✓ | side-on | 0 / None | 1.84 | 0.00 | 0.002 | 1.54 | level | 0.0000 |  |
| r3c3 → r2c2 | ✗ | side-on | 0 / None | 213.45 | 0.00 | 0.001 | 1.76 | direct | 0.0000 | placed_on_target |
| r3c3 → r2c3 | ✓ | side-on | 0 / None | 2.22 | 0.00 | 0.002 | 1.61 | direct | 0.0000 |  |
| r3c3 → r3c2 | ✓ | side-on | 0 / None | 1.83 | 0.00 | 0.002 | 1.51 | level | 0.0000 |  |

Refused before motion:

- 8 × plan: side-on: plan: IK found no level grasp pose for this approach; crane: plan: IK found no grasp pose in the bounded start set
- 5 × destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no grasp pose in the bounded start set
- 1 × destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no pose for rung +2
- 1 × destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no pose for rung +11
- 1 × destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: destination: IK found no pose that sets the object down on the target with the grasp's attitude

## Placements: pool_cyl_1 (grasp best)

All 72 ordered pairs: ✓ 51, ✗ 2, H 3, R 16.  Flown: 56; every criterion: 51.

| from \ to | r1c1 | r1c2 | r1c3 | r2c1 | r2c2 | r2c3 | r3c1 | r3c2 | r3c3 |
|---|---|---|---|---|---|---|---|---|---|
| r1c1 | — | ✓ | ✓ | ✓ | ✓ | ✓ | R | H | ✓ |
| r1c2 | ✓ | — | ✓ | ✓ | ✓ | ✓ | R | ✓ | ✓ |
| r1c3 | ✓ | ✓ | — | ✓ | ✓ | ✓ | R | ✓ | ✓ |
| r2c1 | ✓ | ✓ | ✓ | — | ✓ | ✓ | R | ✓ | H |
| r2c2 | ✓ | ✓ | ✓ | ✓ | — | ✗ | R | ✓ | ✓ |
| r2c3 | ✓ | ✓ | ✗ | ✓ | ✓ | — | R | ✓ | ✓ |
| r3c1 | R | R | R | R | R | R | — | R | R |
| r3c2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | R | — | ✓ |
| r3c3 | ✓ | ✓ | ✓ | H | ✓ | ✓ | R | ✓ | — |

Flown pairs:

| flight | result | grasp | az / pitch | error mm | tilt ° | slip mm | drag mm | way out | others moved mm | failed / halt / refusal |
|---|---|---|---|---|---|---|---|---|---|---|
| r1c1 → r1c2 | ✓ | side-on | 90 / None | 1.50 | 0.00 | 0.004 | 1.35 | direct | 0.0000 |  |
| r1c1 → r1c3 | ✓ | side-on | 90 / None | 1.62 | 0.00 | 0.004 | 1.32 | direct | 0.0000 |  |
| r1c1 → r2c1 | ✓ | side-on | 90 / None | 1.05 | 0.00 | 0.004 | 1.06 | direct | 0.0000 |  |
| r1c1 → r2c2 | ✓ | side-on | 90 / None | 0.71 | 0.00 | 0.004 | 1.39 | direct | 0.0000 |  |
| r1c1 → r2c3 | ✓ | side-on | 90 / None | 0.58 | 0.00 | 0.004 | 0.97 | direct | 0.0000 |  |
| r1c1 → r3c2 | H | side-on | 90 / None | 0.89 | 0.00 | 0.004 | 0.98 |  | 0.0000 | withdrawal refused before opening: with the grip let go, neither opening has a clear way out, so the hand was not opened further |
| r1c1 → r3c3 | ✓ | side-on | 90 / None | 1.04 | 0.00 | 0.004 | 1.06 | direct | 0.0000 |  |
| r1c2 → r1c1 | ✓ | side-on | 30 / None | 1.49 | 0.00 | 0.004 | 1.24 | direct | 0.0000 |  |
| r1c2 → r1c3 | ✓ | side-on | 60 / None | 1.02 | 0.00 | 0.051 | 0.79 | direct | 0.0000 |  |
| r1c2 → r2c1 | ✓ | side-on | 30 / None | 1.71 | 0.00 | 0.004 | 1.45 | direct | 0.0000 |  |
| r1c2 → r2c2 | ✓ | side-on | 30 / None | 1.70 | 0.00 | 0.004 | 1.48 | direct | 0.0000 |  |
| r1c2 → r2c3 | ✓ | side-on | 30 / None | 1.84 | 0.00 | 0.004 | 0.85 | direct | 0.0000 |  |
| r1c2 → r3c2 | ✓ | side-on | 60 / None | 0.62 | 0.00 | 0.051 | 0.36 | direct | 0.0000 |  |
| r1c2 → r3c3 | ✓ | side-on | 60 / None | 1.10 | 0.00 | 0.051 | 0.84 | direct | 0.0000 |  |
| r1c3 → r1c1 | ✓ | side-on | 0 / None | 1.59 | 0.00 | 0.003 | 1.36 | direct | 0.0000 |  |
| r1c3 → r1c2 | ✓ | side-on | 0 / None | 1.28 | 0.00 | 0.003 | 0.79 | direct | 0.0000 |  |
| r1c3 → r2c1 | ✓ | side-on | 0 / None | 1.81 | 0.00 | 0.003 | 1.49 | direct | 0.0000 |  |
| r1c3 → r2c2 | ✓ | side-on | 0 / None | 1.59 | 0.00 | 0.003 | 1.37 | direct | 0.0000 |  |
| r1c3 → r2c3 | ✓ | side-on | 0 / None | 1.97 | 0.00 | 0.003 | 1.00 | direct | 0.0000 |  |
| r1c3 → r3c2 | ✓ | side-on | 0 / None | 1.78 | 0.00 | 0.003 | 1.45 | direct | 0.0000 |  |
| r1c3 → r3c3 | ✓ | side-on | 0 / None | 1.67 | 0.00 | 0.003 | 1.40 | direct | 0.0000 |  |
| r2c1 → r1c1 | ✓ | side-on | 90 / None | 0.84 | 0.00 | 0.005 | 1.08 | level | 0.0000 |  |
| r2c1 → r1c2 | ✓ | side-on | 90 / None | 1.24 | 0.00 | 0.005 | 1.33 | direct | 0.0000 |  |
| r2c1 → r1c3 | ✓ | side-on | 90 / None | 1.78 | 0.00 | 0.005 | 1.68 | direct | 0.0000 |  |
| r2c1 → r2c2 | ✓ | side-on | 90 / None | 0.30 | 0.00 | 0.005 | 1.07 | direct | 0.0000 |  |
| r2c1 → r2c3 | ✓ | side-on | 90 / None | 0.36 | 0.00 | 0.003 | 0.63 | separating | 0.0000 |  |
| r2c1 → r3c2 | ✓ | side-on | 90 / None | 0.61 | 0.00 | 0.005 | 0.89 | level | 0.0000 |  |
| r2c1 → r3c3 | H | side-on | 90 / None | 0.63 | 0.00 | 0.005 | 0.77 |  | 0.0000 | withdrawal refused before opening: with the grip let go, neither opening has a clear way out, so the hand was not opened further |
| r2c2 → r1c1 | ✓ | side-on | 75 / None | 1.11 | 0.00 | 0.004 | 0.22 | direct | 0.0000 |  |
| r2c2 → r1c2 | ✓ | side-on | 75 / None | 0.64 | 0.00 | 0.004 | 0.89 | direct | 0.0000 |  |
| r2c2 → r1c3 | ✓ | side-on | 45 / None | 0.73 | 0.00 | 0.003 | 1.04 | direct | 0.0000 |  |
| r2c2 → r2c1 | ✓ | side-on | 75 / None | 0.70 | 0.00 | 0.004 | 0.41 | direct | 0.0000 |  |
| r2c2 → r2c3 | ✗ | side-on | 75 / None | 156.58 | 0.00 | 0.004 | 1.05 | direct | 0.0000 | placed_on_target |
| r2c2 → r3c2 | ✓ | side-on | 75 / None | 0.81 | 0.00 | 0.004 | 0.62 | separating | 0.0000 |  |
| r2c2 → r3c3 | ✓ | side-on | 75 / None | 0.32 | 0.00 | 0.004 | 0.60 | direct | 0.0000 |  |
| r2c3 → r1c1 | ✓ | side-on | 30 / None | 0.39 | 0.00 | 0.003 | 0.34 | direct | 0.0000 |  |
| r2c3 → r1c2 | ✓ | side-on | 0 / None | 0.99 | 0.00 | 0.003 | 0.69 | direct | 0.0000 |  |
| r2c3 → r1c3 | ✗ | side-on | 15 / None | 1.23 | 0.00 | 0.139 | 1.17 | direct | 0.0000 | slip_within_resolution |
| r2c3 → r2c1 | ✓ | side-on | 30 / None | 0.58 | 0.00 | 0.003 | 0.58 | direct | 0.0000 |  |
| r2c3 → r2c2 | ✓ | side-on | 30 / None | 1.29 | 0.00 | 0.003 | 0.47 | direct | 0.0000 |  |
| r2c3 → r3c2 | ✓ | side-on | 30 / None | 0.47 | 0.00 | 0.003 | 0.32 | direct | 0.0000 |  |
| r2c3 → r3c3 | ✓ | side-on | 30 / None | 0.78 | 0.00 | 0.003 | 0.29 | direct | 0.0000 |  |
| r3c2 → r1c1 | ✓ | side-on | 15 / None | 2.43 | 0.00 | 0.003 | 2.27 | direct | 0.0000 |  |
| r3c2 → r1c2 | ✓ | side-on | 45 / None | 1.63 | 0.00 | 0.002 | 0.84 | direct | 0.0000 |  |
| r3c2 → r1c3 | ✓ | side-on | 30 / None | 1.24 | 0.00 | 0.002 | 0.93 | direct | 0.0000 |  |
| r3c2 → r2c1 | ✓ | side-on | 15 / None | 1.90 | 0.00 | 0.003 | 1.60 | direct | 0.0000 |  |
| r3c2 → r2c2 | ✓ | side-on | 15 / None | 1.62 | 0.00 | 0.003 | 1.49 | direct | 0.0000 |  |
| r3c2 → r2c3 | ✓ | side-on | 15 / None | 2.29 | 0.00 | 0.003 | 1.67 | direct | 0.0000 |  |
| r3c2 → r3c3 | ✓ | side-on | 15 / None | 1.86 | 0.00 | 0.003 | 1.61 | direct | 0.0000 |  |
| r3c3 → r1c1 | ✓ | side-on | 45 / None | 1.13 | 0.00 | 0.003 | 1.24 | direct | 0.0000 |  |
| r3c3 → r1c2 | ✓ | side-on | 45 / None | 0.59 | 0.00 | 0.003 | 1.28 | direct | 0.0000 |  |
| r3c3 → r1c3 | ✓ | side-on | 15 / None | 1.08 | 0.00 | 0.004 | 1.17 | direct | 0.0000 |  |
| r3c3 → r2c1 | H | side-on | 45 / None | 0.54 | 0.00 | 0.003 | 0.81 |  | 0.0000 | withdrawal refused before opening: with the grip let go, neither opening has a clear way out, so the hand was not opened further |
| r3c3 → r2c2 | ✓ | side-on | 45 / None | 0.27 | 0.00 | 0.003 | 0.88 | direct | 0.0000 |  |
| r3c3 → r2c3 | ✓ | side-on | 45 / None | 1.08 | 0.00 | 0.003 | 1.26 | direct | 0.0000 |  |
| r3c3 → r3c2 | ✓ | side-on | 45 / None | 0.84 | 0.00 | 0.003 | 1.04 | level | 0.0000 |  |

Refused before motion:

- 8 × destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces
- 8 × plan: side-on: plan: IK found no level grasp pose for this approach; crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces

## Placements: pool_cyl_1 (grasp best) [pads condim 3, attribution]

All 72 ordered pairs: ✓ 8, ✗ 27, H 21, R 16.  Flown: 56; every criterion: 8.

| from \ to | r1c1 | r1c2 | r1c3 | r2c1 | r2c2 | r2c3 | r3c1 | r3c2 | r3c3 |
|---|---|---|---|---|---|---|---|---|---|
| r1c1 | — | ✓ | ✗ | ✓ | ✓ | ✗ | R | ✗ | ✗ |
| r1c2 | ✗ | — | H | ✗ | H | H | R | H | H |
| r1c3 | ✗ | ✗ | — | H | ✗ | H | R | H | H |
| r2c1 | ✗ | H | H | — | ✗ | ✗ | R | ✗ | H |
| r2c2 | ✓ | ✓ | ✗ | ✓ | — | ✗ | R | ✓ | ✓ |
| r2c3 | H | H | ✗ | H | H | — | R | H | H |
| r3c1 | R | R | R | R | R | R | — | R | R |
| r3c2 | H | ✗ | ✗ | H | ✗ | ✗ | R | — | H |
| r3c3 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | R | ✗ | — |

Flown pairs:

| flight | result | grasp | az / pitch | error mm | tilt ° | slip mm | drag mm | way out | others moved mm | failed / halt / refusal |
|---|---|---|---|---|---|---|---|---|---|---|
| r1c1 → r1c2 | ✓ | side-on | 90 / None | 2.31 | 0.00 | 0.011 | 1.92 | direct | 0.0000 |  |
| r1c1 → r1c3 | ✗ | side-on | 90 / None | 301.53 | 0.00 | 0.011 | 2.94 | direct | 0.0000 | placed_on_target |
| r1c1 → r2c1 | ✓ | side-on | 90 / None | 2.58 | 0.00 | 0.011 | 2.24 | direct | 0.0000 |  |
| r1c1 → r2c2 | ✓ | side-on | 90 / None | 2.79 | 0.00 | 0.011 | 2.14 | direct | 0.0000 |  |
| r1c1 → r2c3 | ✗ | side-on | 90 / None | 339.74 | 0.00 | 0.011 | 3.03 | direct | 0.0000 | placed_on_target |
| r1c1 → r3c2 | ✗ | side-on | 90 / None | 343.47 | 0.00 | 0.011 | 2.55 | separating | 0.0000 | placed_on_target |
| r1c1 → r3c3 | ✗ | side-on | 90 / None | 431.76 | 0.00 | 0.011 | 2.46 | direct | 0.0000 | placed_on_target |
| r1c2 → r1c1 | ✗ | side-on | 30 / None | 8.37 | 0.00 | 0.575 | 8.95 | direct | 0.0000 | slip_within_resolution |
| r1c2 → r1c3 | H | side-on | 60 / None | 1.92 | 0.18 | 0.514 | 1.54 |  | 0.0000 | object moved 0.22 mm |
| r1c2 → r2c1 | ✗ | side-on | 30 / None | 218.76 | 0.00 | 0.575 | 2.30 | direct | 0.0000 | placed_on_target, slip_within_resolution |
| r1c2 → r2c2 | H | side-on | 30 / None | 101.71 | 90.07 | 0.575 | 61.35 |  | 0.0000 | object moved 0.63 mm |
| r1c2 → r2c3 | H | side-on | 30 / None | 62.03 | 89.92 | 0.575 | 61.96 |  | 0.0000 | object moved 0.82 mm |
| r1c2 → r3c2 | H | side-on | 60 / None | 2.30 | 1.58 | 0.514 | 2.14 |  | 0.0000 | object moved 0.20 mm |
| r1c2 → r3c3 | H | side-on | 60 / None | 6.20 | 1.80 | 0.514 | 6.10 |  | 0.0000 | object moved 0.21 mm |
| r1c3 → r1c1 | ✗ | side-on | 0 / None | 303.48 | 0.00 | 0.250 | 2.76 | direct | 0.0000 | placed_on_target, slip_within_resolution |
| r1c3 → r1c2 | ✗ | side-on | 0 / None | 152.37 | 0.00 | 0.250 | 3.13 | direct | 0.0000 | placed_on_target, slip_within_resolution |
| r1c3 → r2c1 | H | side-on | 0 / None | 400.56 | 90.00 | 0.250 | 98.59 |  | 0.0000 | object moved 3.89 mm |
| r1c3 → r2c2 | ✗ | side-on | 0 / None | 214.16 | 0.00 | 0.250 | 2.69 | direct | 0.0000 | placed_on_target, slip_within_resolution |
| r1c3 → r2c3 | H | side-on | 0 / None | 102.12 | 90.01 | 0.250 | 72.53 |  | 0.0000 | object moved 1.70 mm |
| r1c3 → r3c2 | H | side-on | 0 / None | 298.20 | 90.01 | 0.250 | 56.80 |  | 0.0000 | object moved 0.36 mm |
| r1c3 → r3c3 | H | side-on | 0 / None | 311.79 | 1.76 | 0.250 | 8.01 |  | 0.0000 | object moved 0.35 mm |
| r2c1 → r1c1 | ✗ | side-on | 90 / None | 3.41 | 0.00 | 0.147 | 3.19 | level | 0.0000 | slip_within_resolution |
| r2c1 → r1c2 | H | side-on | 90 / None | 22.57 | 89.99 | 0.147 |  |  | 0.0000 | stopped while carrying the object back: the grip force read zero while carrying: the object is not held |
| r2c1 → r1c3 | H | side-on | 90 / None | 197.81 | 109.79 | 0.147 |  |  | 0.0000 | stopped while carrying the object back: the grip force read zero while carrying: the object is not held |
| r2c1 → r2c2 | ✗ | side-on | 90 / None | 3.29 | 0.00 | 0.147 | 2.43 | direct | 0.0000 | slip_within_resolution |
| r2c1 → r2c3 | ✗ | side-on | 90 / None | 310.95 | 0.00 | 0.167 | 0.10 | separating | 0.0000 | placed_on_target, slip_within_resolution |
| r2c1 → r3c2 | ✗ | side-on | 90 / None | 3.28 | 0.00 | 0.147 | 3.11 | level | 0.0000 | slip_within_resolution |
| r2c1 → r3c3 | H | side-on | 90 / None | 23.35 | 171.69 | 0.147 |  |  | 0.0000 | stopped while carrying the object back: the grip force read zero while carrying: the object is not held |
| r2c2 → r1c1 | ✓ | side-on | 75 / None | 2.88 | 0.00 | 0.044 | 2.75 | direct | 0.0000 |  |
| r2c2 → r1c2 | ✓ | side-on | 75 / None | 2.84 | 0.00 | 0.044 | 2.60 | direct | 0.0000 |  |
| r2c2 → r1c3 | ✗ | side-on | 45 / None | 1.77 | 0.00 | 0.229 | 1.68 | direct | 0.0000 | slip_within_resolution |
| r2c2 → r2c1 | ✓ | side-on | 75 / None | 3.39 | 0.00 | 0.044 | 3.14 | direct | 0.0000 |  |
| r2c2 → r2c3 | ✗ | side-on | 75 / None | 154.31 | 0.00 | 0.044 | 3.36 | separating | 0.0000 | placed_on_target |
| r2c2 → r3c2 | ✓ | side-on | 75 / None | 3.75 | 0.00 | 0.044 | 3.52 | direct | 0.0000 |  |
| r2c2 → r3c3 | ✓ | side-on | 75 / None | 3.31 | 0.00 | 0.044 | 2.74 | separating | 0.0000 |  |
| r2c3 → r1c1 | H | side-on | 30 / None | 74.29 | 90.02 | 0.008 | 73.36 |  | 0.0000 | object moved 1.69 mm |
| r2c3 → r1c2 | H | side-on | 0 / None | 3.11 | 1.56 | 0.527 | 3.73 |  | 0.0000 | object moved 0.21 mm |
| r2c3 → r1c3 | ✗ | side-on | 15 / None | 1.42 | 0.00 | 0.158 | 2.02 | direct | 0.0000 | slip_within_resolution |
| r2c3 → r2c1 | H | side-on | 30 / None | 76.18 | 89.98 | 0.008 | 75.18 |  | 0.0000 | object moved 1.89 mm |
| r2c3 → r2c2 | H | side-on | 30 / None | 4.39 | 0.26 | 0.008 | 4.45 |  | 0.0000 | object moved 0.21 mm |
| r2c3 → r3c2 | H | side-on | 30 / None | 65.03 | 89.92 | 0.008 | 65.32 |  | 0.0000 | object moved 1.00 mm |
| r2c3 → r3c3 | H | side-on | 30 / None | 12.35 | 3.48 | 0.008 | 11.46 |  | 0.0000 | object moved 0.21 mm |
| r3c2 → r1c1 | H | side-on | 15 / None | 59.14 | 89.91 | 0.278 | 59.41 |  | 0.0000 | object moved 0.30 mm |
| r3c2 → r1c2 | ✗ | side-on | 45 / None | 3.37 | 0.00 | 0.139 | 3.68 | direct | 0.0000 | slip_within_resolution |
| r3c2 → r1c3 | ✗ | side-on | 30 / None | 1.24 | 0.00 | 0.237 | 1.44 | direct | 0.0000 | slip_within_resolution |
| r3c2 → r2c1 | H | side-on | 15 / None | 3.80 | 3.40 | 0.278 | 2.90 |  | 0.0000 | object moved 0.20 mm |
| r3c2 → r2c2 | ✗ | side-on | 15 / None | 0.97 | 0.00 | 0.278 | 1.95 | direct | 0.0000 | slip_within_resolution |
| r3c2 → r2c3 | ✗ | side-on | 15 / None | 1.96 | 0.00 | 0.278 | 1.76 | direct | 0.0000 | slip_within_resolution |
| r3c2 → r3c3 | H | side-on | 15 / None | 146.25 | 89.92 | 0.278 | 147.17 |  | 0.0000 | object moved 5.16 mm |
| r3c3 → r1c1 | ✗ | side-on | 45 / None | 3.03 | 0.00 | 0.147 | 2.83 | separating | 0.0000 | slip_within_resolution |
| r3c3 → r1c2 | ✗ | side-on | 45 / None | 2.22 | 0.00 | 0.147 | 2.08 | direct | 0.0000 | slip_within_resolution |
| r3c3 → r1c3 | ✗ | side-on | 15 / None | 297.87 | 0.00 | 0.360 | 1.41 | direct | 0.0000 | placed_on_target, slip_within_resolution |
| r3c3 → r2c1 | ✗ | side-on | 45 / None | 2.38 | 0.00 | 0.147 | 2.34 | direct | 0.0000 | slip_within_resolution |
| r3c3 → r2c2 | ✗ | side-on | 45 / None | 2.17 | 0.00 | 0.147 | 2.07 | separating | 0.0000 | slip_within_resolution, clean_placement |
| r3c3 → r2c3 | ✗ | side-on | 45 / None | 142.65 | 0.00 | 0.147 | 1.55 | separating | 0.0000 | placed_on_target, slip_within_resolution |
| r3c3 → r3c2 | ✗ | side-on | 45 / None | 2.74 | 0.00 | 0.147 | 2.51 | level | 0.0000 | slip_within_resolution |

Refused before motion:

- 8 × destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces
- 8 × plan: side-on: plan: IK found no level grasp pose for this approach; crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces

## Perturbations (red_cube, grasp best)

Counts: ✓ 24.

| flight | result | grasp | az / pitch | error mm | tilt ° | slip mm | drag mm | way out | others moved mm | failed / halt / refusal |
|---|---|---|---|---|---|---|---|---|---|---|
| r2c2 → r1c1 [best] offset [3.0, 0] mm | ✓ | side-on | -0 / None | 2.74 | 0.00 | 0.003 | 2.41 | level | 0.0000 |  |
| r2c2 → r1c1 [best] offset [-3.0, 0] mm | ✓ | side-on | -0 / None | 2.56 | 0.00 | 0.003 | 2.24 | level | 0.0000 |  |
| r2c2 → r1c1 [best] offset [0, 3.0] mm | ✓ | side-on | -0 / None | 2.90 | 0.00 | 0.003 | 2.60 | direct | 0.0000 |  |
| r2c2 → r1c1 [best] offset [0, -3.0] mm | ✓ | side-on | -0 / None | 2.82 | 0.00 | 0.003 | 2.53 | direct | 0.0000 |  |
| r2c2 → r1c1 [best] yaw 5° | ✓ | side-on | 5 / None | 2.88 | 0.00 | 0.002 | 2.58 | direct | 0.0000 |  |
| r2c2 → r1c1 [best] yaw -5° | ✓ | side-on | -5 / None | 2.90 | 0.00 | 0.001 | 2.68 | level | 0.0000 |  |
| r3c2 → r3c3 [best] offset [3.0, 0] mm | ✓ | side-on | 0 / None | 2.59 | 0.00 | 0.001 | 2.28 | direct | 0.0000 |  |
| r3c2 → r3c3 [best] offset [-3.0, 0] mm | ✓ | side-on | 0 / None | 2.51 | 0.00 | 0.001 | 2.18 | direct | 0.0000 |  |
| r3c2 → r3c3 [best] offset [0, 3.0] mm | ✓ | side-on | 0 / None | 2.90 | 0.00 | 0.001 | 2.57 | level | 0.0000 |  |
| r3c2 → r3c3 [best] offset [0, -3.0] mm | ✓ | side-on | 0 / None | 2.97 | 0.00 | 0.001 | 2.67 | level | 0.0000 |  |
| r3c2 → r3c3 [best] yaw 5° | ✓ | side-on | 5 / None | 2.79 | 0.00 | 0.001 | 2.48 | direct | 0.0000 |  |
| r3c2 → r3c3 [best] yaw -5° | ✓ | side-on | 85 / None | 3.16 | 0.00 | 0.006 | 2.75 | direct | 0.0000 |  |
| r1c3 → r2c1 [best] offset [3.0, 0] mm | ✓ | side-on | 0 / None | 2.30 | 0.00 | 0.000 | 2.00 | direct | 0.0000 |  |
| r1c3 → r2c1 [best] offset [-3.0, 0] mm | ✓ | side-on | -0 / None | 2.56 | 0.00 | 0.000 | 2.25 | direct | 0.0000 |  |
| r1c3 → r2c1 [best] offset [0, 3.0] mm | ✓ | side-on | -0 / None | 3.20 | 0.00 | 0.000 | 2.89 | direct | 0.0000 |  |
| r1c3 → r2c1 [best] offset [0, -3.0] mm | ✓ | side-on | -0 / None | 3.03 | 0.00 | 0.000 | 2.71 | direct | 0.0000 |  |
| r1c3 → r2c1 [best] yaw 5° | ✓ | side-on | 5 / None | 2.96 | 0.00 | 0.000 | 2.65 | direct | 0.0000 |  |
| r1c3 → r2c1 [best] yaw -5° | ✓ | side-on | -5 / None | 3.02 | 0.00 | 0.000 | 2.69 | direct | 0.0000 |  |
| r2c2 → (lift) [best] offset [3.0, 0] mm | ✓ | side-on | -0 / 0.0 | 1.29 | 0.00 | 0.003 | 2.36 | direct | 0.0000 |  |
| r2c2 → (lift) [best] offset [-3.0, 0] mm | ✓ | side-on | -0 / 0.0 | 1.79 | 0.00 | 0.003 | 2.30 | direct | 0.0000 |  |
| r2c2 → (lift) [best] offset [0, 3.0] mm | ✓ | side-on | -0 / 5.0 | 1.90 | 0.00 | 0.003 | 2.24 | direct | 0.0000 |  |
| r2c2 → (lift) [best] offset [0, -3.0] mm | ✓ | side-on | -0 / 5.0 | 1.97 | 0.00 | 0.003 | 2.14 | direct | 0.0000 |  |
| r2c2 → (lift) [best] yaw 5° | ✓ | side-on | 5 / 5.0 | 1.54 | 0.00 | 0.002 | 2.20 | direct | 0.0000 |  |
| r2c2 → (lift) [best] yaw -5° | ✓ | side-on | -5 / 5.0 | 1.41 | 0.00 | 0.002 | 2.41 | direct | 0.0000 |  |

## Before / after: PR #197's pair r3c2 → r3c3

Counts: ✓ 10.

| flight | result | grasp | az / pitch | error mm | tilt ° | slip mm | drag mm | way out | others moved mm | failed / halt / refusal |
|---|---|---|---|---|---|---|---|---|---|---|
| r3c2 → r3c3 [crane]  | ✓ | crane |  | 3.30 | 0.00 | 0.006 | 3.37 | separating | 0.0000 |  |
| r3c2 → r3c3 [crane] offset [3.0, 0] mm | ✓ | crane |  | 3.37 | 0.00 | 0.006 | 3.44 | separating | 0.0000 |  |
| r3c2 → r3c3 [crane] offset [-3.0, 0] mm | ✓ | crane |  | 3.29 | 0.00 | 0.006 | 3.36 | separating | 0.0000 |  |
| r3c2 → r3c3 [crane] offset [0, 3.0] mm | ✓ | crane |  | 4.09 | 0.00 | 0.006 | 4.21 | direct | 0.0000 |  |
| r3c2 → r3c3 [crane] offset [0, -3.0] mm | ✓ | crane |  | 3.07 | 0.00 | 0.006 | 3.18 | direct | 0.0000 |  |
| r3c2 → r3c3 [best]  | ✓ | side-on | 0 / None | 2.72 | 0.00 | 0.001 | 2.39 | level | 0.0000 |  |
| r3c2 → r3c3 [best] offset [3.0, 0] mm | ✓ | side-on | 0 / None | 2.59 | 0.00 | 0.001 | 2.28 | direct | 0.0000 |  |
| r3c2 → r3c3 [best] offset [-3.0, 0] mm | ✓ | side-on | 0 / None | 2.51 | 0.00 | 0.001 | 2.18 | direct | 0.0000 |  |
| r3c2 → r3c3 [best] offset [0, 3.0] mm | ✓ | side-on | 0 / None | 2.90 | 0.00 | 0.001 | 2.57 | level | 0.0000 |  |
| r3c2 → r3c3 [best] offset [0, -3.0] mm | ✓ | side-on | 0 / None | 2.97 | 0.00 | 0.001 | 2.67 | level | 0.0000 |  |

## Neighbours on the board (red_cube)

Counts: ✓ 8, R 2.

| flight | result | grasp | az / pitch | error mm | tilt ° | slip mm | drag mm | way out | others moved mm | failed / halt / refusal |
|---|---|---|---|---|---|---|---|---|---|---|
| r2c2 → (lift) [side] others pool_box_2 on r2c3 | ✓ | side-on | -0 / 5.0 | 1.82 | 0.00 | 0.003 | 2.26 | direct | 0.0000 |  |
| r2c2 → (lift) [side] others pool_box_2 on r2c3, pool_box_3 on r1c2 | R |  |  |  |  |  |  |  |  | check_arm_path: via->hover: hand would come within -2.2 cm of pool_box_2 (boxes (pads, palm, back) hand model, margin 0.0 cm, gripper -45 deg) (best of 4 side-on approaches) |
| r2c2 → (lift) [side] others pool_box_3 on r1c2 | ✓ | side-on | 90 / 5.0 | 2.30 | 0.00 | 0.003 | 2.65 | direct | 0.0000 |  |
| r2c2 → r1c1 [side] others pool_box_2 on r2c3 | ✓ | side-on | -0 / None | 2.88 | 0.00 | 0.003 | 2.58 | direct | 0.0000 |  |
| r2c2 → r1c1 [side] others pool_cyl_2 on r1c2 | ✓ | side-on | 90 / None | 3.32 | 0.00 | 0.003 | 2.94 | direct | 0.0005 |  |
| r2c2 → (lift) [best] others pool_box_2 on r2c3 | ✓ | side-on | -0 / 5.0 | 1.82 | 0.00 | 0.003 | 2.26 | direct | 0.0000 |  |
| r2c2 → (lift) [best] others pool_box_2 on r2c3, pool_box_3 on r1c2 | R |  |  |  |  |  |  |  |  | check_arm_path: side-on: check_arm_path: via->hover: hand would come within -2.2 cm of pool_box_2 (boxes (pads, palm, back) hand model, margin 0.0 cm, gripper -45 deg) (best of 4 side-on approaches); crane: plan: IK found no grasp pose in the bounded start set |
| r2c2 → (lift) [best] others pool_box_3 on r1c2 | ✓ | side-on | 90 / 5.0 | 2.30 | 0.00 | 0.003 | 2.65 | direct | 0.0000 |  |
| r2c2 → r1c1 [best] others pool_box_2 on r2c3 | ✓ | side-on | -0 / None | 2.88 | 0.00 | 0.003 | 2.58 | direct | 0.0000 |  |
| r2c2 → r1c1 [best] others pool_cyl_2 on r1c2 | ✓ | side-on | 90 / None | 3.32 | 0.00 | 0.003 | 2.94 | direct | 0.0005 |  |

