# #180 offline flights — summary

Offline physics (`tools/place_sweep/fly.py`: the executor on the live model's physics, host IK mirror).  Scene FWDCenterLabSivaPool, the object upright on the origin cell, nothing else on the board unless stated.  Grasp `best` = the planner choice.  ✓ every #174 criterion true; ✗ flew with a criterion false; H halted or refused in flight; R refused before motion; E crashed.

## Lifts (grasp best)

| object | r1c1 | r1c2 | r1c3 | r2c1 | r2c2 | r2c3 | r3c1 | r3c2 | r3c3 | counts |
|---|---|---|---|---|---|---|---|---|---|---|
| pool_box_1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | R | ✓ | ✓ | ✓ 8, R 1 |
| pool_cyl_1 | ✓ | ✓ | ✓ | H | ✓ | ✓ | R | H | ✗ | ✓ 5, ✗ 1, H 2, R 1 |
| red_cube | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | R | ✓ | ✓ | ✓ 8, R 1 |

| flight | result | grasp | az / pitch | error mm | tilt ° | slip mm | drag mm | way out | others moved mm | failed / halt / refusal |
|---|---|---|---|---|---|---|---|---|---|---|
| red_cube r1c1 | ✓ | side-on | 90 / 5.0 | 2.30 | 0.00 | 0.006 | 2.40 | direct | 0.0000 |  |
| red_cube r1c2 | ✓ | side-on | 90 / 5.0 | 2.41 | 0.00 | 0.002 | 2.40 | direct | 0.0000 |  |
| red_cube r1c3 | ✓ | side-on | -0 / 5.0 | 1.77 | 0.00 | 0.004 | 2.20 | direct | 0.0000 |  |
| red_cube r2c1 | ✓ | side-on | 90 / 5.0 | 4.92 | 0.00 | 0.014 | 2.36 | direct | 0.0000 |  |
| red_cube r2c2 | ✓ | side-on | -0 / 5.0 | 1.72 | 0.00 | 0.004 | 2.08 | direct | 0.0000 |  |
| red_cube r2c3 | ✓ | side-on | -0 / 5.0 | 1.90 | 0.00 | 0.020 | 1.88 | direct | 0.0000 |  |
| red_cube r3c1 | R |  |  |  |  |  |  |  |  | plan: side-on: plan: IK found no level grasp pose for this approach (best of 4 side-on approaches); crane: plan: IK found no grasp pose in the bounded start set |
| red_cube r3c2 | ✓ | side-on | 0 / 0.0 | 1.90 | 0.00 | 0.004 | 1.94 | direct | 0.0000 |  |
| red_cube r3c3 | ✓ | side-on | 0 / 5.0 | 1.74 | 0.00 | 0.001 | 2.18 | direct | 0.0000 |  |
| pool_box_1 r1c1 | ✓ | side-on | 90 / 5.0 | 3.96 | 0.00 | 0.001 | 1.48 | direct | 0.0000 |  |
| pool_box_1 r1c2 | ✓ | side-on | 90 / 0.0 | 4.08 | 0.00 | 0.006 | 1.28 | direct | 0.0000 |  |
| pool_box_1 r1c3 | ✓ | side-on | -0 / 5.0 | 3.43 | 0.00 | 0.009 | 1.07 | direct | 0.0000 |  |
| pool_box_1 r2c1 | ✓ | side-on | 90 / 5.0 | 6.44 | 0.00 | 0.004 | 1.16 | direct | 0.0000 |  |
| pool_box_1 r2c2 | ✓ | side-on | 0 / 5.0 | 3.58 | 0.00 | 0.002 | 1.00 | direct | 0.0000 |  |
| pool_box_1 r2c3 | ✓ | side-on | -0 / 5.0 | 3.13 | 0.00 | 0.000 | 1.09 | direct | 0.0000 |  |
| pool_box_1 r3c1 | R |  |  |  |  |  |  |  |  | plan: side-on: plan: IK found no level grasp pose for this approach (best of 4 side-on approaches); crane: plan: IK found no grasp pose in the bounded start set |
| pool_box_1 r3c2 | ✓ | side-on | 0 / 5.0 | 3.82 | 0.00 | 0.001 | 1.17 | direct | 0.0000 |  |
| pool_box_1 r3c3 | ✓ | side-on | -0 / 5.0 | 3.20 | 0.00 | 0.001 | 0.94 | direct | 0.0000 |  |
| pool_cyl_1 r1c1 | ✓ | side-on | 90 / 5.0 | 3.93 | 0.00 | 0.002 | 3.27 | direct | 0.0000 |  |
| pool_cyl_1 r1c2 | ✓ | side-on | 30 / 5.0 | 5.21 | 0.00 | 0.052 | 3.07 | direct | 0.0000 |  |
| pool_cyl_1 r1c3 | ✓ | side-on | 0 / 5.0 | 4.56 | 0.00 | 0.044 | 2.97 | direct | 0.0000 |  |
| pool_cyl_1 r2c1 | H | side-on | 75 / 5.0 | 47.52 | 90.04 | 0.056 | 43.71 |  | 0.0000 | object moved 1.29 mm |
| pool_cyl_1 r2c2 | ✓ | side-on | 75 / 5.0 | 7.48 | 0.00 | 0.025 | 3.42 | direct | 0.0000 |  |
| pool_cyl_1 r2c3 | ✓ | side-on | 15 / 5.0 | 5.62 | 0.00 | 0.020 | 3.19 | direct | 0.0000 |  |
| pool_cyl_1 r3c1 | R |  |  |  |  |  |  |  |  | plan: side-on: plan: IK found no level grasp pose for this approach (best of 24 side-on approaches); crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| pool_cyl_1 r3c2 | H | side-on | 45 / 5.0 | 41.80 | 90.08 |  | 46.73 |  | 0.0000 | object did not follow the lift: hand up 4 cm, object up -6.8 mm |
| pool_cyl_1 r3c3 | ✗ | side-on | 45 / 5.0 | 8.21 | 0.00 | 0.048 | 4.81 | separating | 0.0000 | clean_placement |

## Placements: red_cube (grasp best)

All 72 ordered pairs: ✓ 53, ✗ 1, H 2, R 16.  Flown: 56; every criterion: 53.

| from \ to | r1c1 | r1c2 | r1c3 | r2c1 | r2c2 | r2c3 | r3c1 | r3c2 | r3c3 |
|---|---|---|---|---|---|---|---|---|---|
| r1c1 | — | ✓ | ✓ | ✓ | ✓ | ✓ | R | ✓ | ✓ |
| r1c2 | ✓ | — | ✓ | ✓ | ✓ | ✓ | R | ✓ | ✓ |
| r1c3 | ✓ | ✓ | — | ✓ | ✓ | ✓ | R | ✓ | ✓ |
| r2c1 | ✓ | ✓ | ✓ | — | ✓ | ✓ | R | ✓ | ✓ |
| r2c2 | ✓ | ✓ | ✗ | ✓ | — | ✓ | R | ✓ | ✓ |
| r2c3 | ✓ | ✓ | ✓ | ✓ | ✓ | — | R | ✓ | ✓ |
| r3c1 | R | R | R | R | R | R | — | R | R |
| r3c2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | R | — | ✓ |
| r3c3 | ✓ | ✓ | H | ✓ | ✓ | H | R | ✓ | — |

Flown pairs:

| flight | result | grasp | az / pitch | error mm | tilt ° | slip mm | drag mm | way out | others moved mm | failed / halt / refusal |
|---|---|---|---|---|---|---|---|---|---|---|
| r1c1 → r1c2 | ✓ | side-on | 90 / None | 3.05 | 0.00 | 0.006 | 2.53 | direct | 0.0000 |  |
| r1c1 → r1c3 | ✓ | side-on | 90 / None | 2.48 | 0.00 | 0.006 | 2.37 | direct | 0.0000 |  |
| r1c1 → r2c1 | ✓ | side-on | 90 / None | 2.83 | 0.00 | 0.006 | 2.52 | level | 0.0000 |  |
| r1c1 → r2c2 | ✓ | side-on | 90 / None | 2.90 | 0.00 | 0.006 | 2.74 | direct | 0.0000 |  |
| r1c1 → r2c3 | ✓ | side-on | 90 / None | 1.37 | 0.00 | 0.006 | 2.29 | direct | 0.0000 |  |
| r1c1 → r3c2 | ✓ | side-on | 90 / None | 2.79 | 0.00 | 0.006 | 2.47 | level | 0.0000 |  |
| r1c1 → r3c3 | ✓ | side-on | 90 / None | 2.81 | 0.00 | 0.006 | 2.53 | level | 0.0000 |  |
| r1c2 → r1c1 | ✓ | side-on | 90 / None | 2.89 | 0.00 | 0.002 | 2.71 | direct | 0.0000 |  |
| r1c2 → r1c3 | ✓ | side-on | 90 / None | 2.46 | 0.00 | 0.002 | 2.51 | direct | 0.0000 |  |
| r1c2 → r2c1 | ✓ | side-on | 90 / None | 3.02 | 0.00 | 0.002 | 2.72 | direct | 0.0000 |  |
| r1c2 → r2c2 | ✓ | side-on | 90 / None | 3.15 | 0.00 | 0.002 | 3.01 | direct | 0.0000 |  |
| r1c2 → r2c3 | ✓ | side-on | 90 / None | 1.71 | 0.00 | 0.002 | 2.33 | sideways | 0.0000 |  |
| r1c2 → r3c2 | ✓ | side-on | 90 / None | 3.30 | 0.00 | 0.002 | 3.03 | direct | 0.0000 |  |
| r1c2 → r3c3 | ✓ | side-on | 90 / None | 3.20 | 0.00 | 0.002 | 2.98 | direct | 0.0000 |  |
| r1c3 → r1c1 | ✓ | side-on | -0 / None | 2.66 | 0.00 | 0.004 | 2.40 | direct | 0.0000 |  |
| r1c3 → r1c2 | ✓ | side-on | -0 / None | 2.86 | 0.00 | 0.004 | 2.25 | direct | 0.0000 |  |
| r1c3 → r2c1 | ✓ | side-on | -0 / None | 2.83 | 0.00 | 0.004 | 2.52 | direct | 0.0000 |  |
| r1c3 → r2c2 | ✓ | side-on | -0 / None | 2.70 | 0.00 | 0.004 | 2.48 | direct | 0.0000 |  |
| r1c3 → r2c3 | ✓ | side-on | -0 / None | 2.46 | 0.00 | 0.004 | 2.26 | direct | 0.0000 |  |
| r1c3 → r3c2 | ✓ | side-on | -0 / None | 2.72 | 0.00 | 0.004 | 2.42 | direct | 0.0000 |  |
| r1c3 → r3c3 | ✓ | side-on | -0 / None | 2.77 | 0.00 | 0.004 | 2.48 | direct | 0.0000 |  |
| r2c1 → r1c1 | ✓ | side-on | 90 / None | 2.78 | 0.00 | 0.014 | 2.68 | level | 0.0000 |  |
| r2c1 → r1c2 | ✓ | side-on | 90 / None | 2.42 | 0.00 | 0.014 | 2.32 | direct | 0.0000 |  |
| r2c1 → r1c3 | ✓ | side-on | 90 / None | 1.97 | 0.00 | 0.014 | 2.79 | level | 0.0000 |  |
| r2c1 → r2c2 | ✓ | side-on | 90 / None | 3.37 | 0.00 | 0.014 | 2.68 | direct | 0.0000 |  |
| r2c1 → r2c3 | ✓ | side-on | 90 / None | 2.66 | 0.00 | 0.014 | 2.62 | direct | 0.0000 |  |
| r2c1 → r3c2 | ✓ | side-on | 90 / None | 3.06 | 0.00 | 0.014 | 2.82 | level | 0.0000 |  |
| r2c1 → r3c3 | ✓ | side-on | 90 / None | 2.91 | 0.00 | 0.014 | 2.70 | level | 0.0000 |  |
| r2c2 → r1c1 | ✓ | side-on | -0 / None | 2.43 | 0.00 | 0.004 | 2.13 | direct | 0.0000 |  |
| r2c2 → r1c2 | ✓ | side-on | -0 / None | 2.87 | 0.00 | 0.004 | 2.10 | direct | 0.0000 |  |
| r2c2 → r1c3 | ✗ | side-on | 90 / None | 214.04 | 0.00 | 0.000 | 2.05 | direct | 0.0000 | placed_on_target |
| r2c2 → r2c1 | ✓ | side-on | -0 / None | 2.53 | 0.00 | 0.004 | 2.19 | direct | 0.0000 |  |
| r2c2 → r2c3 | ✓ | side-on | 90 / None | 2.53 | 0.00 | 0.004 | 2.41 | direct | 0.0000 |  |
| r2c2 → r3c2 | ✓ | side-on | -0 / None | 2.59 | 0.00 | 0.004 | 2.24 | direct | 0.0000 |  |
| r2c2 → r3c3 | ✓ | side-on | 90 / None | 2.33 | 0.00 | 0.004 | 2.55 | separating | 0.0000 |  |
| r2c3 → r1c1 | ✓ | side-on | -0 / None | 1.95 | 0.00 | 0.020 | 1.72 | direct | 0.0000 |  |
| r2c3 → r1c2 | ✓ | side-on | -0 / None | 2.57 | 0.00 | 0.020 | 2.09 | direct | 0.0000 |  |
| r2c3 → r1c3 | ✓ | side-on | -0 / None | 1.67 | 0.00 | 0.020 | 1.80 | sideways | 0.0000 |  |
| r2c3 → r2c1 | ✓ | side-on | -0 / None | 2.43 | 0.00 | 0.020 | 2.17 | direct | 0.0000 |  |
| r2c3 → r2c2 | ✓ | side-on | -0 / None | 1.86 | 0.00 | 0.020 | 1.67 | direct | 0.0000 |  |
| r2c3 → r3c2 | ✓ | side-on | -0 / None | 2.02 | 0.00 | 0.020 | 1.73 | direct | 0.0000 |  |
| r2c3 → r3c3 | ✓ | side-on | -0 / None | 2.09 | 0.00 | 0.020 | 1.81 | direct | 0.0000 |  |
| r3c2 → r1c1 | ✓ | side-on | 0 / None | 2.69 | 0.00 | 0.004 | 2.42 | level | 0.0000 |  |
| r3c2 → r1c2 | ✓ | side-on | 0 / None | 2.86 | 0.00 | 0.004 | 2.16 | direct | 0.0000 |  |
| r3c2 → r1c3 | ✓ | side-on | 0 / None | 2.47 | 0.00 | 0.009 | 2.54 | direct | 0.0000 |  |
| r3c2 → r2c1 | ✓ | side-on | 0 / None | 2.63 | 0.00 | 0.004 | 2.26 | level | 0.0000 |  |
| r3c2 → r2c2 | ✓ | side-on | 0 / None | 2.62 | 0.00 | 0.004 | 2.34 | level | 0.0000 |  |
| r3c2 → r2c3 | ✓ | side-on | 0 / None | 2.47 | 0.00 | 0.004 | 2.25 | direct | 0.0000 |  |
| r3c2 → r3c3 | ✓ | side-on | 0 / None | 2.45 | 0.00 | 0.004 | 2.13 | level | 0.0000 |  |
| r3c3 → r1c1 | ✓ | side-on | 0 / None | 2.36 | 0.00 | 0.001 | 2.14 | direct | 0.0000 |  |
| r3c3 → r1c2 | ✓ | side-on | 0 / None | 2.69 | 0.00 | 0.001 | 2.13 | direct | 0.0000 |  |
| r3c3 → r1c3 | H | side-on | 0 / None | 304.80 | 0.00 |  |  |  | 0.0000 | refused at rung +1 -- table: forearm -0.0 mm into the table on 'descent hover->grasp' (predicted) |
| r3c3 → r2c1 | ✓ | side-on | 0 / None | 2.56 | 0.00 | 0.001 | 2.22 | direct | 0.0000 |  |
| r3c3 → r2c2 | ✓ | side-on | 0 / None | 2.56 | 0.00 | 0.001 | 2.30 | direct | 0.0000 |  |
| r3c3 → r2c3 | H | side-on | 0 / None | 152.40 | 0.00 |  |  |  | 0.0000 | refused at rung +1 -- table: forearm -0.0 mm into the table on 'descent hover->grasp' (predicted) |
| r3c3 → r3c2 | ✓ | side-on | 0 / None | 2.64 | 0.00 | 0.001 | 2.29 | direct | 0.0000 |  |

Refused before motion:

- 8 × plan: side-on: plan: IK found no level grasp pose for this approach; crane: plan: IK found no grasp pose in the bounded start set
- 4 × destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no grasp pose in the bounded start set
- 1 × destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no pose for rung +1
- 1 × destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no pose for rung +10
- 1 × destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: destination: IK found no pose that sets the object down on the target with the grasp's attitude
- 1 × destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: straddle: predicted grasp pose does not straddle: ['finger_above_top']

## Placements: pool_box_1 (grasp best)

All 72 ordered pairs: ✓ 53, ✗ 2, H 1, R 16.  Flown: 56; every criterion: 53.

| from \ to | r1c1 | r1c2 | r1c3 | r2c1 | r2c2 | r2c3 | r3c1 | r3c2 | r3c3 |
|---|---|---|---|---|---|---|---|---|---|
| r1c1 | — | ✓ | ✓ | ✓ | ✓ | ✓ | R | ✓ | ✓ |
| r1c2 | ✓ | — | ✓ | ✓ | ✓ | ✓ | R | ✓ | ✓ |
| r1c3 | ✓ | ✓ | — | ✓ | ✓ | ✓ | R | ✓ | ✓ |
| r2c1 | H | ✓ | ✓ | — | ✓ | ✓ | R | ✓ | ✓ |
| r2c2 | ✓ | ✓ | ✓ | ✓ | — | ✓ | R | ✓ | ✓ |
| r2c3 | ✓ | ✓ | ✓ | ✓ | ✓ | — | R | ✓ | ✓ |
| r3c1 | R | R | R | R | R | R | — | R | R |
| r3c2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | R | — | ✓ |
| r3c3 | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | R | ✓ | — |

Flown pairs:

| flight | result | grasp | az / pitch | error mm | tilt ° | slip mm | drag mm | way out | others moved mm | failed / halt / refusal |
|---|---|---|---|---|---|---|---|---|---|---|
| r1c1 → r1c2 | ✓ | side-on | 90 / None | 1.93 | 0.00 | 0.001 | 1.54 | direct | 0.0000 |  |
| r1c1 → r1c3 | ✓ | side-on | 90 / None | 0.96 | 0.00 | 0.001 | 1.03 | sideways | 0.0000 |  |
| r1c1 → r2c1 | ✓ | side-on | 90 / None | 1.88 | 0.00 | 0.001 | 1.64 | direct | 0.0000 |  |
| r1c1 → r2c2 | ✓ | side-on | 90 / None | 1.45 | 0.00 | 0.001 | 1.34 | direct | 0.0000 |  |
| r1c1 → r2c3 | ✓ | side-on | 90 / None | 2.05 | 0.00 | 0.001 | 1.41 | direct | 0.0000 |  |
| r1c1 → r3c2 | ✓ | side-on | 90 / None | 1.74 | 0.00 | 0.001 | 1.59 | level | 0.0000 |  |
| r1c1 → r3c3 | ✓ | side-on | 90 / None | 1.64 | 0.00 | 0.001 | 1.50 | direct | 0.0000 |  |
| r1c2 → r1c1 | ✓ | side-on | 90 / None | 1.75 | 0.00 | 0.006 | 1.55 | direct | 0.0000 |  |
| r1c2 → r1c3 | ✓ | side-on | 90 / None | 1.76 | 0.00 | 0.006 | 0.94 | sideways | 0.0000 |  |
| r1c2 → r2c1 | ✓ | side-on | 90 / None | 1.77 | 0.00 | 0.006 | 1.50 | level | 0.0000 |  |
| r1c2 → r2c2 | ✓ | side-on | 90 / None | 2.76 | 0.00 | 0.006 | 1.65 | direct | 0.0000 |  |
| r1c2 → r2c3 | ✓ | side-on | 90 / None | 2.36 | 0.00 | 0.006 | 1.65 | direct | 0.0000 |  |
| r1c2 → r3c2 | ✓ | side-on | 90 / None | 2.27 | 0.00 | 0.006 | 1.96 | level | 0.0000 |  |
| r1c2 → r3c3 | ✓ | side-on | 90 / None | 1.93 | 0.00 | 0.006 | 1.75 | level | 0.0000 |  |
| r1c3 → r1c1 | ✓ | side-on | -0 / None | 1.43 | 0.00 | 0.009 | 1.24 | direct | 0.0000 |  |
| r1c3 → r1c2 | ✓ | side-on | -0 / None | 1.74 | 0.00 | 0.009 | 1.23 | direct | 0.0000 |  |
| r1c3 → r2c1 | ✓ | side-on | -0 / None | 1.66 | 0.00 | 0.009 | 1.45 | direct | 0.0000 |  |
| r1c3 → r2c2 | ✓ | side-on | -0 / None | 1.40 | 0.00 | 0.009 | 1.24 | direct | 0.0000 |  |
| r1c3 → r2c3 | ✓ | side-on | -0 / None | 1.43 | 0.00 | 0.009 | 1.27 | direct | 0.0000 |  |
| r1c3 → r3c2 | ✓ | side-on | -0 / None | 1.46 | 0.00 | 0.009 | 1.26 | direct | 0.0000 |  |
| r1c3 → r3c3 | ✓ | side-on | -0 / None | 1.59 | 0.00 | 0.009 | 1.37 | direct | 0.0000 |  |
| r2c1 → r1c1 | H | side-on | 90 / None | 1.20 | 0.00 | 0.004 | 0.96 |  | 0.0000 | withdrawal refused before opening: with the grip let go, neither opening has a clear way out, so the hand was not opened further |
| r2c1 → r1c2 | ✓ | side-on | 90 / None | 1.60 | 0.00 | 0.004 | 0.94 | direct | 0.0000 |  |
| r2c1 → r1c3 | ✓ | side-on | 90 / None | 0.81 | 0.00 | 0.004 | 0.93 | direct | 0.0000 |  |
| r2c1 → r2c2 | ✓ | side-on | 90 / None | 2.12 | 0.00 | 0.004 | 0.99 | direct | 0.0000 |  |
| r2c1 → r2c3 | ✓ | side-on | 90 / None | 1.50 | 0.00 | 0.004 | 1.31 | direct | 0.0000 |  |
| r2c1 → r3c2 | ✓ | side-on | 90 / None | 1.55 | 0.00 | 0.004 | 1.25 | level | 0.0000 |  |
| r2c1 → r3c3 | ✓ | side-on | 90 / None | 1.27 | 0.00 | 0.004 | 1.03 | level | 0.0000 |  |
| r2c2 → r1c1 | ✓ | side-on | 0 / None | 1.43 | 0.00 | 0.002 | 1.18 | direct | 0.0000 |  |
| r2c2 → r1c2 | ✓ | side-on | 0 / None | 1.59 | 0.00 | 0.002 | 1.17 | direct | 0.0000 |  |
| r2c2 → r1c3 | ✓ | side-on | 90 / None | 0.60 | 0.00 | 0.001 | 1.49 | direct | 0.0000 |  |
| r2c2 → r2c1 | ✓ | side-on | 0 / None | 1.44 | 0.00 | 0.002 | 1.25 | direct | 0.0000 |  |
| r2c2 → r2c3 | ✓ | side-on | 0 / None | 1.87 | 0.00 | 0.002 | 1.10 | direct | 0.0000 |  |
| r2c2 → r3c2 | ✓ | side-on | 0 / None | 1.12 | 0.00 | 0.001 | 0.90 | level | 0.0000 |  |
| r2c2 → r3c3 | ✓ | side-on | 90 / None | 2.56 | 0.00 | 0.001 | 1.50 | direct | 0.0000 |  |
| r2c3 → r1c1 | ✓ | side-on | -0 / None | 1.25 | 0.00 | 0.000 | 1.04 | direct | 0.0000 |  |
| r2c3 → r1c2 | ✓ | side-on | -0 / None | 1.43 | 0.00 | 0.000 | 0.97 | direct | 0.0000 |  |
| r2c3 → r1c3 | ✓ | side-on | -0 / None | 0.52 | 0.00 | 0.000 | 0.87 | sideways | 0.0000 |  |
| r2c3 → r2c1 | ✓ | side-on | -0 / None | 1.18 | 0.00 | 0.000 | 1.04 | direct | 0.0000 |  |
| r2c3 → r2c2 | ✓ | side-on | -0 / None | 1.06 | 0.00 | 0.000 | 0.92 | direct | 0.0000 |  |
| r2c3 → r3c2 | ✓ | side-on | -0 / None | 0.96 | 0.00 | 0.000 | 0.80 | direct | 0.0000 |  |
| r2c3 → r3c3 | ✓ | side-on | -0 / None | 1.32 | 0.00 | 0.000 | 1.12 | direct | 0.0000 |  |
| r3c2 → r1c1 | ✓ | side-on | 0 / None | 1.37 | 0.00 | 0.001 | 1.14 | direct | 0.0000 |  |
| r3c2 → r1c2 | ✓ | side-on | 0 / None | 1.51 | 0.00 | 0.001 | 1.07 | direct | 0.0000 |  |
| r3c2 → r1c3 | ✓ | side-on | 0 / None | 0.76 | 0.00 | 0.001 | 1.06 | direct | 0.0000 |  |
| r3c2 → r2c1 | ✓ | side-on | 0 / None | 1.20 | 0.00 | 0.001 | 1.01 | direct | 0.0000 |  |
| r3c2 → r2c2 | ✓ | side-on | 0 / None | 1.31 | 0.00 | 0.001 | 1.16 | direct | 0.0000 |  |
| r3c2 → r2c3 | ✓ | side-on | 0 / None | 1.20 | 0.00 | 0.005 | 0.96 | direct | 0.0000 |  |
| r3c2 → r3c3 | ✓ | side-on | 0 / None | 1.25 | 0.00 | 0.001 | 1.04 | direct | 0.0000 |  |
| r3c3 → r1c1 | ✓ | side-on | -0 / None | 1.17 | 0.00 | 0.001 | 0.93 | direct | 0.0000 |  |
| r3c3 → r1c2 | ✓ | side-on | -0 / None | 1.20 | 0.00 | 0.001 | 0.75 | direct | 0.0000 |  |
| r3c3 → r1c3 | ✗ | side-on | -0 / None | 304.89 | 0.00 | 0.000 | 0.83 | level | 0.0000 | placed_on_target |
| r3c3 → r2c1 | ✓ | side-on | -0 / None | 1.18 | 0.00 | 0.001 | 0.99 | level | 0.0000 |  |
| r3c3 → r2c2 | ✓ | side-on | -0 / None | 1.21 | 0.00 | 0.001 | 1.05 | direct | 0.0000 |  |
| r3c3 → r2c3 | ✗ | side-on | -0 / None | 152.50 | 0.00 | 0.000 | 0.83 | level | 0.0000 | placed_on_target |
| r3c3 → r3c2 | ✓ | side-on | -0 / None | 1.09 | 0.00 | 0.001 | 0.94 | level | 0.0000 |  |

Refused before motion:

- 8 × plan: side-on: plan: IK found no level grasp pose for this approach; crane: plan: IK found no grasp pose in the bounded start set
- 4 × destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no grasp pose in the bounded start set
- 1 × destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no pose for rung +4
- 1 × destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no pose for rung +12
- 1 × destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: table: finger_pad -4.5 mm into the table on 'lift grasp->+lift' (commanded)
- 1 × destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: table: thumb_pad -1.9 mm into the table on 'descent hover->grasp' (commanded)

## Placements: pool_cyl_1 (grasp best)

All 72 ordered pairs: ✓ 26, ✗ 11, H 19, R 16.  Flown: 56; every criterion: 26.

| from \ to | r1c1 | r1c2 | r1c3 | r2c1 | r2c2 | r2c3 | r3c1 | r3c2 | r3c3 |
|---|---|---|---|---|---|---|---|---|---|
| r1c1 | — | ✓ | ✓ | ✓ | H | ✓ | R | ✓ | ✓ |
| r1c2 | ✓ | — | ✗ | ✗ | ✓ | ✓ | R | ✗ | ✗ |
| r1c3 | ✗ | ✗ | — | ✗ | ✗ | ✓ | R | ✗ | ✗ |
| r2c1 | H | H | H | — | H | H | R | H | H |
| r2c2 | ✓ | ✓ | ✓ | ✗ | — | ✓ | R | ✓ | ✓ |
| r2c3 | ✓ | ✓ | ✓ | ✓ | ✓ | — | R | ✓ | ✓ |
| r3c1 | R | R | R | R | R | R | — | R | R |
| r3c2 | H | H | H | H | ✓ | H | R | — | H |
| r3c3 | H | H | ✓ | H | H | ✓ | R | H | — |

Flown pairs:

| flight | result | grasp | az / pitch | error mm | tilt ° | slip mm | drag mm | way out | others moved mm | failed / halt / refusal |
|---|---|---|---|---|---|---|---|---|---|---|
| r1c1 → r1c2 | ✓ | side-on | 90 / None | 3.74 | 0.00 | 0.002 | 3.48 | direct | 0.0000 |  |
| r1c1 → r1c3 | ✓ | side-on | 90 / None | 3.20 | 0.00 | 0.002 | 3.30 | direct | 0.0000 |  |
| r1c1 → r2c1 | ✓ | side-on | 90 / None | 3.45 | 0.00 | 0.002 | 3.46 | direct | 0.0000 |  |
| r1c1 → r2c2 | H | side-on | 90 / None | 20.87 | 153.54 | 0.002 |  |  | 0.0000 | stopped while carrying the object back: the grip force read zero while carrying: the object is not held |
| r1c1 → r2c3 | ✓ | side-on | 90 / None | 4.11 | 0.00 | 0.002 | 3.58 | direct | 0.0000 |  |
| r1c1 → r3c2 | ✓ | side-on | 90 / None | 5.44 | 0.00 | 0.002 | 4.94 | direct | 0.0000 |  |
| r1c1 → r3c3 | ✓ | side-on | 90 / None | 4.13 | 0.00 | 0.002 | 3.81 | direct | 0.0000 |  |
| r1c2 → r1c1 | ✓ | side-on | 30 / None | 2.91 | 0.00 | 0.052 | 2.18 | direct | 0.0000 |  |
| r1c2 → r1c3 | ✗ | side-on | 60 / None | 148.43 | 0.00 | 0.195 | 1.03 | direct | 0.0000 | placed_on_target, slip_within_resolution |
| r1c2 → r2c1 | ✗ | side-on | 30 / None | 221.06 | 0.00 | 0.052 | 2.34 | direct | 0.0000 | placed_on_target |
| r1c2 → r2c2 | ✓ | side-on | 30 / None | 4.61 | 0.00 | 0.052 | 4.31 | direct | 0.0000 |  |
| r1c2 → r2c3 | ✓ | side-on | 30 / None | 4.63 | 0.00 | 0.052 | 4.38 | direct | 0.0000 |  |
| r1c2 → r3c2 | ✗ | side-on | 60 / None | 312.83 | 0.00 | 0.195 | 2.41 | direct | 0.0000 | placed_on_target, slip_within_resolution |
| r1c2 → r3c3 | ✗ | side-on | 60 / None | 345.32 | 0.00 | 0.195 | 1.74 | direct | 0.0000 | placed_on_target, slip_within_resolution |
| r1c3 → r1c1 | ✗ | side-on | 0 / None | 301.24 | 0.00 | 0.044 | 1.17 | direct | 0.0000 | placed_on_target |
| r1c3 → r1c2 | ✗ | side-on | 0 / None | 148.08 | 0.00 | 0.044 | 2.86 | direct | 0.0000 | placed_on_target |
| r1c3 → r2c1 | ✗ | side-on | 0 / None | 340.37 | 0.00 | 0.044 | 1.48 | direct | 0.0000 | placed_on_target |
| r1c3 → r2c2 | ✗ | side-on | 0 / None | 217.89 | 0.00 | 0.044 | 2.94 | direct | 0.0000 | placed_on_target |
| r1c3 → r2c3 | ✓ | side-on | 0 / None | 4.85 | 0.00 | 0.044 | 4.58 | direct | 0.0000 |  |
| r1c3 → r3c2 | ✗ | side-on | 0 / None | 346.63 | 0.00 | 0.044 | 2.38 | direct | 0.0000 | placed_on_target |
| r1c3 → r3c3 | ✗ | side-on | 0 / None | 312.76 | 0.00 | 0.044 | 2.50 | direct | 0.0000 | placed_on_target |
| r2c1 → r1c1 | H | side-on | 75 / None | 16.81 | 145.58 | 0.056 |  |  | 0.0000 | stopped while carrying the object back: the grip force read zero while carrying: the object is not held |
| r2c1 → r1c2 | H | side-on | 75 / None | 18.20 | 120.70 | 0.056 |  |  | 0.0000 | stopped while carrying the object back: the grip force read zero while carrying: the object is not held |
| r2c1 → r1c3 | H | side-on | 75 / None | 181.39 | 119.29 | 0.056 |  |  | 0.0000 | stopped while carrying the object back: the grip force read zero while carrying: the object is not held |
| r2c1 → r2c2 | H | side-on | 75 / None | 152.40 | 0.00 |  |  |  | 0.0000 | refused at rung +4 -- table: forearm -0.1 mm into the table on 'descent hover->grasp' (predicted) |
| r2c1 → r2c3 | H | side-on | 75 / None | 164.77 | 172.43 | 0.056 |  |  | 0.0000 | stopped while carrying the object back: the grip force read zero while carrying: the object is not held |
| r2c1 → r3c2 | H | side-on | 75 / None | 10.39 | 155.26 | 0.056 |  |  | 0.0000 | stopped while carrying the object back: the grip force read zero while carrying: the object is not held |
| r2c1 → r3c3 | H | side-on | 75 / None | 24.63 | 166.07 | 0.056 |  |  | 0.0000 | stopped while carrying the object back: the grip force read zero while carrying: the object is not held |
| r2c2 → r1c1 | ✓ | side-on | 75 / None | 3.62 | 0.00 | 0.025 | 3.40 | direct | 0.0000 |  |
| r2c2 → r1c2 | ✓ | side-on | 45 / None | 4.00 | 0.00 | 0.069 | 4.02 | direct | 0.0000 |  |
| r2c2 → r1c3 | ✓ | side-on | 45 / None | 0.75 | 0.00 | 0.069 | 1.30 | direct | 0.0000 |  |
| r2c2 → r2c1 | ✗ | side-on | 75 / None | 3.83 | 0.00 | 0.025 | 3.56 | separating | 0.0000 | clean_placement |
| r2c2 → r2c3 | ✓ | side-on | 45 / None | 3.95 | 0.00 | 0.069 | 3.25 | direct | 0.0000 |  |
| r2c2 → r3c2 | ✓ | side-on | 75 / None | 3.78 | 0.00 | 0.025 | 3.63 | direct | 0.0000 |  |
| r2c2 → r3c3 | ✓ | side-on | 75 / None | 3.70 | 0.00 | 0.025 | 3.40 | separating | 0.0000 |  |
| r2c3 → r1c1 | ✓ | side-on | 15 / None | 4.48 | 0.00 | 0.020 | 4.26 | direct | 0.0000 |  |
| r2c3 → r1c2 | ✓ | side-on | 15 / None | 4.68 | 0.00 | 0.020 | 4.21 | direct | 0.0000 |  |
| r2c3 → r1c3 | ✓ | side-on | 15 / None | 3.97 | 0.00 | 0.020 | 3.49 | direct | 0.0000 |  |
| r2c3 → r2c1 | ✓ | side-on | 15 / None | 4.55 | 0.00 | 0.020 | 4.26 | direct | 0.0000 |  |
| r2c3 → r2c2 | ✓ | side-on | 15 / None | 4.59 | 0.00 | 0.020 | 4.40 | direct | 0.0000 |  |
| r2c3 → r3c2 | ✓ | side-on | 15 / None | 3.78 | 0.00 | 0.020 | 3.53 | separating | 0.0000 |  |
| r2c3 → r3c3 | ✓ | side-on | 15 / None | 3.91 | 0.00 | 0.020 | 3.55 | direct | 0.0000 |  |
| r3c2 → r1c1 | H | side-on | 45 / None | 382.56 | 90.08 |  | 46.73 |  | 0.0000 | object did not follow the lift: hand up 4 cm, object up -6.8 mm |
| r3c2 → r1c2 | H | side-on | 45 / None | 342.19 | 90.08 |  | 46.73 |  | 0.0000 | object did not follow the lift: hand up 4 cm, object up -6.8 mm |
| r3c2 → r1c3 | H | side-on | 30 / None | 14.87 | 167.42 | 0.222 |  |  | 0.0000 | stopped while carrying the object back: the grip force read zero while carrying: the object is not held |
| r3c2 → r2c1 | H | side-on | 45 / None | 255.84 | 90.08 |  | 46.73 |  | 0.0000 | object did not follow the lift: hand up 4 cm, object up -6.8 mm |
| r3c2 → r2c2 | ✓ | side-on | 15 / None | 2.65 | 0.00 | 0.064 | 2.26 | direct | 0.0000 |  |
| r3c2 → r2c3 | H | side-on | 45 / None | 231.06 | 90.08 |  | 46.73 |  | 0.0000 | object did not follow the lift: hand up 4 cm, object up -6.8 mm |
| r3c2 → r3c3 | H | side-on | 45 / None | 137.63 | 90.08 |  | 46.73 |  | 0.0000 | object did not follow the lift: hand up 4 cm, object up -6.8 mm |
| r3c3 → r1c1 | H | side-on | 45 / None | 10.24 | 125.63 | 0.048 |  |  | 0.0000 | stopped while carrying the object back: the grip force read zero while carrying: the object is not held |
| r3c3 → r1c2 | H | side-on | 45 / None | 3.66 | 155.13 | 0.048 |  |  | 0.0000 | stopped while carrying the object back: the grip force read zero while carrying: the object is not held |
| r3c3 → r1c3 | ✓ | side-on | 30 / None | 4.26 | 0.00 | 0.012 | 4.36 | direct | 0.0000 |  |
| r3c3 → r2c1 | H | side-on | 45 / None | 6.48 | 124.06 | 0.048 |  |  | 0.0000 | stopped while carrying the object back: the grip force read zero while carrying: the object is not held |
| r3c3 → r2c2 | H | side-on | 45 / None | 6.18 | 135.03 | 0.048 |  |  | 0.0000 | stopped while carrying the object back: the grip force read zero while carrying: the object is not held |
| r3c3 → r2c3 | ✓ | side-on | 45 / None | 4.29 | 0.00 | 0.048 | 4.24 | direct | 0.0000 |  |
| r3c3 → r3c2 | H | side-on | 45 / None | 7.28 | 120.83 | 0.048 |  |  | 0.0000 | stopped while carrying the object back: the grip force read zero while carrying: the object is not held |

Refused before motion:

- 8 × destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces
- 8 × plan: side-on: plan: IK found no level grasp pose for this approach; crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces

## Perturbations (red_cube, grasp best)

Counts: ✓ 24.

| flight | result | grasp | az / pitch | error mm | tilt ° | slip mm | drag mm | way out | others moved mm | failed / halt / refusal |
|---|---|---|---|---|---|---|---|---|---|---|
| r2c2 → r1c1 [best] offset [3.0, 0] mm | ✓ | side-on | -0 / None | 2.48 | 0.00 | 0.002 | 2.15 | level | 0.0000 |  |
| r2c2 → r1c1 [best] offset [-3.0, 0] mm | ✓ | side-on | -0 / None | 2.19 | 0.00 | 0.002 | 1.89 | level | 0.0000 |  |
| r2c2 → r1c1 [best] offset [0, 3.0] mm | ✓ | side-on | -0 / None | 2.54 | 0.00 | 0.004 | 2.23 | direct | 0.0000 |  |
| r2c2 → r1c1 [best] offset [0, -3.0] mm | ✓ | side-on | -0 / None | 2.46 | 0.00 | 0.004 | 2.15 | direct | 0.0000 |  |
| r2c2 → r1c1 [best] yaw 5° | ✓ | side-on | 5 / None | 2.62 | 0.00 | 0.003 | 2.31 | direct | 0.0000 |  |
| r2c2 → r1c1 [best] yaw -5° | ✓ | side-on | -5 / None | 2.15 | 0.00 | 0.003 | 1.96 | level | 0.0000 |  |
| r3c2 → r3c3 [best] offset [3.0, 0] mm | ✓ | side-on | 0 / None | 2.87 | 0.00 | 0.010 | 2.55 | direct | 0.0000 |  |
| r3c2 → r3c3 [best] offset [-3.0, 0] mm | ✓ | side-on | 0 / None | 2.71 | 0.00 | 0.009 | 2.38 | direct | 0.0000 |  |
| r3c2 → r3c3 [best] offset [0, 3.0] mm | ✓ | side-on | 0 / None | 2.59 | 0.00 | 0.004 | 2.26 | level | 0.0000 |  |
| r3c2 → r3c3 [best] offset [0, -3.0] mm | ✓ | side-on | 0 / None | 2.50 | 0.00 | 0.003 | 2.17 | level | 0.0000 |  |
| r3c2 → r3c3 [best] yaw 5° | ✓ | side-on | 5 / None | 2.65 | 0.00 | 0.006 | 2.34 | direct | 0.0000 |  |
| r3c2 → r3c3 [best] yaw -5° | ✓ | side-on | 85 / None | 3.20 | 0.00 | 0.005 | 2.56 | separating | 0.0000 |  |
| r1c3 → r2c1 [best] offset [3.0, 0] mm | ✓ | side-on | 0 / None | 3.07 | 0.00 | 0.002 | 2.75 | direct | 0.0000 |  |
| r1c3 → r2c1 [best] offset [-3.0, 0] mm | ✓ | side-on | -0 / None | 2.94 | 0.00 | 0.002 | 2.64 | direct | 0.0000 |  |
| r1c3 → r2c1 [best] offset [0, 3.0] mm | ✓ | side-on | -0 / None | 2.77 | 0.00 | 0.005 | 2.45 | direct | 0.0000 |  |
| r1c3 → r2c1 [best] offset [0, -3.0] mm | ✓ | side-on | -0 / None | 3.09 | 0.00 | 0.002 | 2.79 | direct | 0.0000 |  |
| r1c3 → r2c1 [best] yaw 5° | ✓ | side-on | 5 / None | 2.91 | 0.00 | 0.008 | 2.60 | direct | 0.0000 |  |
| r1c3 → r2c1 [best] yaw -5° | ✓ | side-on | -5 / None | 2.78 | 0.00 | 0.004 | 2.46 | direct | 0.0000 |  |
| r2c2 → (lift) [best] offset [3.0, 0] mm | ✓ | side-on | -0 / 0.0 | 1.48 | 0.00 | 0.002 | 2.20 | direct | 0.0000 |  |
| r2c2 → (lift) [best] offset [-3.0, 0] mm | ✓ | side-on | -0 / 0.0 | 1.97 | 0.00 | 0.002 | 1.87 | direct | 0.0000 |  |
| r2c2 → (lift) [best] offset [0, 3.0] mm | ✓ | side-on | -0 / 5.0 | 1.70 | 0.00 | 0.004 | 2.09 | direct | 0.0000 |  |
| r2c2 → (lift) [best] offset [0, -3.0] mm | ✓ | side-on | -0 / 5.0 | 1.71 | 0.00 | 0.004 | 2.09 | direct | 0.0000 |  |
| r2c2 → (lift) [best] yaw 5° | ✓ | side-on | 5 / 5.0 | 1.73 | 0.00 | 0.003 | 2.19 | direct | 0.0000 |  |
| r2c2 → (lift) [best] yaw -5° | ✓ | side-on | -5 / 5.0 | 1.34 | 0.00 | 0.004 | 2.26 | direct | 0.0000 |  |

## Before / after: PR #197's pair r3c2 → r3c3

Counts: ✓ 10.

| flight | result | grasp | az / pitch | error mm | tilt ° | slip mm | drag mm | way out | others moved mm | failed / halt / refusal |
|---|---|---|---|---|---|---|---|---|---|---|
| r3c2 → r3c3 [crane]  | ✓ | crane |  | 7.70 | 0.00 | 0.008 | 7.66 | direct | 0.0000 |  |
| r3c2 → r3c3 [crane] offset [3.0, 0] mm | ✓ | crane |  | 7.36 | 0.00 | 0.006 | 7.34 | direct | 0.0000 |  |
| r3c2 → r3c3 [crane] offset [-3.0, 0] mm | ✓ | crane |  | 7.42 | 0.00 | 0.007 | 7.36 | separating | 0.0000 |  |
| r3c2 → r3c3 [crane] offset [0, 3.0] mm | ✓ | crane |  | 7.19 | 0.00 | 0.007 | 7.14 | separating | 0.0000 |  |
| r3c2 → r3c3 [crane] offset [0, -3.0] mm | ✓ | crane |  | 7.93 | 0.00 | 0.008 | 7.88 | direct | 0.0000 |  |
| r3c2 → r3c3 [best]  | ✓ | side-on | 0 / None | 2.45 | 0.00 | 0.004 | 2.13 | level | 0.0000 |  |
| r3c2 → r3c3 [best] offset [3.0, 0] mm | ✓ | side-on | 0 / None | 2.87 | 0.00 | 0.010 | 2.55 | direct | 0.0000 |  |
| r3c2 → r3c3 [best] offset [-3.0, 0] mm | ✓ | side-on | 0 / None | 2.71 | 0.00 | 0.009 | 2.38 | direct | 0.0000 |  |
| r3c2 → r3c3 [best] offset [0, 3.0] mm | ✓ | side-on | 0 / None | 2.59 | 0.00 | 0.004 | 2.26 | level | 0.0000 |  |
| r3c2 → r3c3 [best] offset [0, -3.0] mm | ✓ | side-on | 0 / None | 2.50 | 0.00 | 0.003 | 2.17 | level | 0.0000 |  |

## Neighbours on the board (red_cube)

Counts: ✓ 6, R 4.

| flight | result | grasp | az / pitch | error mm | tilt ° | slip mm | drag mm | way out | others moved mm | failed / halt / refusal |
|---|---|---|---|---|---|---|---|---|---|---|
| r2c2 → (lift) [side] others pool_box_2 on r2c3 | ✓ | side-on | -0 / 5.0 | 1.72 | 0.00 | 0.004 | 2.08 | direct | 0.0000 |  |
| r2c2 → (lift) [side] others pool_box_2 on r2c3, pool_box_3 on r1c2 | R |  |  |  |  |  |  |  |  | check_arm_path: via->hover: hand would come within -1.1 cm of pool_box_2 (boxes (pads, palm, back) hand model, margin 0.0 cm, gripper -45 deg) (best of 4 side-on approaches) |
| r2c2 → (lift) [side] others pool_box_3 on r1c2 | ✓ | side-on | 90 / 5.0 | 2.77 | 0.00 | 0.004 | 2.26 | separating | 0.0000 |  |
| r2c2 → r1c1 [side] others pool_box_2 on r2c3 | ✓ | side-on | -0 / None | 2.43 | 0.00 | 0.004 | 2.13 | direct | 0.0000 |  |
| r2c2 → r1c1 [side] others pool_cyl_2 on r1c2 | R |  |  |  |  |  |  |  |  | check_arm_path: set-down: r1c1 +9 -> resting: forearm would come within -2.8 cm of pool_cyl_2 (boxes (pads, palm, back) hand model, margin 0.0 cm, gripper -35 deg) |
| r2c2 → (lift) [best] others pool_box_2 on r2c3 | ✓ | side-on | -0 / 5.0 | 1.72 | 0.00 | 0.004 | 2.08 | direct | 0.0000 |  |
| r2c2 → (lift) [best] others pool_box_2 on r2c3, pool_box_3 on r1c2 | R |  |  |  |  |  |  |  |  | check_arm_path: side-on: check_arm_path: via->hover: hand would come within -1.1 cm of pool_box_2 (boxes (pads, palm, back) hand model, margin 0.0 cm, gripper -45 deg) (best of 4 side-on approaches); crane: plan: IK found no grasp pose in the bounded start set |
| r2c2 → (lift) [best] others pool_box_3 on r1c2 | ✓ | side-on | 90 / 5.0 | 2.77 | 0.00 | 0.004 | 2.26 | separating | 0.0000 |  |
| r2c2 → r1c1 [best] others pool_box_2 on r2c3 | ✓ | side-on | -0 / None | 2.43 | 0.00 | 0.004 | 2.13 | direct | 0.0000 |  |
| r2c2 → r1c1 [best] others pool_cyl_2 on r1c2 | R |  |  |  |  |  |  |  |  | check_arm_path: side-on: check_arm_path: set-down: r1c1 +9 -> resting: forearm would come within -2.8 cm of pool_cyl_2 (boxes (pads, palm, back) hand model, margin 0.0 cm, gripper -35 deg); crane: plan: IK found no grasp pose in the bounded start set |

