# Issue #174 offline feasibility sweep — placement (2026-10-18)

reachy-1-2-sim at `feat/195 working tree`.  Preflight only (`crane_place_live.plan_crane_place`): no motion, no simulator.  IK: the host mirror of the simulator's service.  Scene FWDCenterLabSivaPool, the object upright at the origin cell's centre, nothing else on the board.  Reachable cells: the 9 without the `no-right-arm` tag.

## red_cube

Feasible: **3 of 72** ordered pairs.  r3c2→r2c1, r3c2→r2c2, r3c2→r3c3

| from → to | feasible | refused at | stage | reason |
|---|---|---|---|---|
| r1c1 → r1c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r1c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r2c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r2c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r2c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r3c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r3c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r3c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r1c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r1c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r2c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r2c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r2c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r3c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r3c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r3c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c3 → r1c1 | no | pick | plan | IK found no pose for rung +1 |
| r1c3 → r1c2 | no | pick | plan | IK found no pose for rung +1 |
| r1c3 → r2c1 | no | pick | plan | IK found no pose for rung +1 |
| r1c3 → r2c2 | no | pick | plan | IK found no pose for rung +1 |
| r1c3 → r2c3 | no | pick | plan | IK found no pose for rung +1 |
| r1c3 → r3c1 | no | pick | plan | IK found no pose for rung +1 |
| r1c3 → r3c2 | no | pick | plan | IK found no pose for rung +1 |
| r1c3 → r3c3 | no | pick | plan | IK found no pose for rung +1 |
| r2c1 → r1c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r1c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r1c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r2c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r2c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r3c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r3c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r3c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r1c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r1c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r1c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r2c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r2c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r3c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r3c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r3c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c3 → r1c1 | no | pick | plan | IK found no pose for rung +10 |
| r2c3 → r1c2 | no | pick | plan | IK found no pose for rung +10 |
| r2c3 → r1c3 | no | pick | plan | IK found no pose for rung +10 |
| r2c3 → r2c1 | no | pick | plan | IK found no pose for rung +10 |
| r2c3 → r2c2 | no | pick | plan | IK found no pose for rung +10 |
| r2c3 → r3c1 | no | pick | plan | IK found no pose for rung +10 |
| r2c3 → r3c2 | no | pick | plan | IK found no pose for rung +10 |
| r2c3 → r3c3 | no | pick | plan | IK found no pose for rung +10 |
| r3c1 → r1c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r1c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r1c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r2c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r2c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r2c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r3c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r3c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c2 → r1c1 | no | destination | destination | IK found no pose that sets the object down on the target with the grasp's attitude |
| r3c2 → r1c2 | no | destination | destination | IK found no pose that sets the object down on the target with the grasp's attitude |
| r3c2 → r1c3 | no | destination | plan | IK found no pose for rung +2 |
| r3c2 → r2c1 | **yes** |  |  |  |
| r3c2 → r2c2 | **yes** |  |  |  |
| r3c2 → r2c3 | no | destination | destination | IK found no pose that sets the object down on the target with the grasp's attitude |
| r3c2 → r3c1 | no | destination | destination | IK found no pose that sets the object down on the target with the grasp's attitude |
| r3c2 → r3c3 | **yes** |  |  |  |
| r3c3 → r1c1 | no | pick | straddle | predicted grasp pose does not straddle: ['finger_above_top'] |
| r3c3 → r1c2 | no | pick | straddle | predicted grasp pose does not straddle: ['finger_above_top'] |
| r3c3 → r1c3 | no | pick | straddle | predicted grasp pose does not straddle: ['finger_above_top'] |
| r3c3 → r2c1 | no | pick | straddle | predicted grasp pose does not straddle: ['finger_above_top'] |
| r3c3 → r2c2 | no | pick | straddle | predicted grasp pose does not straddle: ['finger_above_top'] |
| r3c3 → r2c3 | no | pick | straddle | predicted grasp pose does not straddle: ['finger_above_top'] |
| r3c3 → r3c1 | no | pick | straddle | predicted grasp pose does not straddle: ['finger_above_top'] |
| r3c3 → r3c2 | no | pick | straddle | predicted grasp pose does not straddle: ['finger_above_top'] |

Refusals by reason:

- 40 × pick / plan: IK found no grasp pose in the bounded start set
- 8 × pick / plan: IK found no pose for rung +1
- 8 × pick / plan: IK found no pose for rung +10
- 8 × pick / straddle: predicted grasp pose does not straddle: ['finger_above_top']
- 4 × destination / destination: IK found no pose that sets the object down on the target with the grasp's attitude
- 1 × destination / plan: IK found no pose for rung +2

## foam_block

Feasible: **0 of 72** ordered pairs.

| from → to | feasible | refused at | stage | reason |
|---|---|---|---|---|
| r1c1 → r1c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r1c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r2c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r2c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r2c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r3c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r3c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r3c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r1c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r1c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r2c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r2c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r2c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r3c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r3c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r3c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c3 → r1c1 | no | pick | plan | IK found no pose for rung +2 |
| r1c3 → r1c2 | no | pick | plan | IK found no pose for rung +2 |
| r1c3 → r2c1 | no | pick | plan | IK found no pose for rung +2 |
| r1c3 → r2c2 | no | pick | plan | IK found no pose for rung +2 |
| r1c3 → r2c3 | no | pick | plan | IK found no pose for rung +2 |
| r1c3 → r3c1 | no | pick | plan | IK found no pose for rung +2 |
| r1c3 → r3c2 | no | pick | plan | IK found no pose for rung +2 |
| r1c3 → r3c3 | no | pick | plan | IK found no pose for rung +2 |
| r2c1 → r1c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r1c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r1c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r2c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r2c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r3c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r3c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r3c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r1c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r1c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r1c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r2c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r2c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r3c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r3c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r3c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c3 → r1c1 | no | pick | plan | IK found no pose for rung +10 |
| r2c3 → r1c2 | no | pick | plan | IK found no pose for rung +10 |
| r2c3 → r1c3 | no | pick | plan | IK found no pose for rung +10 |
| r2c3 → r2c1 | no | pick | plan | IK found no pose for rung +10 |
| r2c3 → r2c2 | no | pick | plan | IK found no pose for rung +10 |
| r2c3 → r3c1 | no | pick | plan | IK found no pose for rung +10 |
| r2c3 → r3c2 | no | pick | plan | IK found no pose for rung +10 |
| r2c3 → r3c3 | no | pick | plan | IK found no pose for rung +10 |
| r3c1 → r1c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r1c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r1c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r2c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r2c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r2c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r3c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r3c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c2 → r1c1 | no | pick | pad_route | finger pad route clearance -2.50 mm < 0.494 mm from rung +9 |
| r3c2 → r1c2 | no | pick | pad_route | finger pad route clearance -2.50 mm < 0.494 mm from rung +9 |
| r3c2 → r1c3 | no | pick | pad_route | finger pad route clearance -2.50 mm < 0.494 mm from rung +9 |
| r3c2 → r2c1 | no | pick | pad_route | finger pad route clearance -2.50 mm < 0.494 mm from rung +9 |
| r3c2 → r2c2 | no | pick | pad_route | finger pad route clearance -2.50 mm < 0.494 mm from rung +9 |
| r3c2 → r2c3 | no | pick | pad_route | finger pad route clearance -2.50 mm < 0.494 mm from rung +9 |
| r3c2 → r3c1 | no | pick | pad_route | finger pad route clearance -2.50 mm < 0.494 mm from rung +9 |
| r3c2 → r3c3 | no | pick | pad_route | finger pad route clearance -2.50 mm < 0.494 mm from rung +9 |
| r3c3 → r1c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c3 → r1c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c3 → r1c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c3 → r2c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c3 → r2c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c3 → r2c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c3 → r3c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c3 → r3c2 | no | pick | plan | IK found no grasp pose in the bounded start set |

Refusals by reason:

- 48 × pick / plan: IK found no grasp pose in the bounded start set
- 8 × pick / plan: IK found no pose for rung +2
- 8 × pick / plan: IK found no pose for rung +10
- 8 × pick / pad_route: finger pad route clearance -2.50 mm < 0.494 mm from rung +9

## pool_box_1

Feasible: **0 of 72** ordered pairs.

| from → to | feasible | refused at | stage | reason |
|---|---|---|---|---|
| r1c1 → r1c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r1c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r2c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r2c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r2c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r3c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r3c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r3c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r1c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r1c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r2c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r2c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r2c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r3c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r3c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r3c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r1c3 → r1c1 | no | pick | plan | IK found no pose for rung +4 |
| r1c3 → r1c2 | no | pick | plan | IK found no pose for rung +4 |
| r1c3 → r2c1 | no | pick | plan | IK found no pose for rung +4 |
| r1c3 → r2c2 | no | pick | plan | IK found no pose for rung +4 |
| r1c3 → r2c3 | no | pick | plan | IK found no pose for rung +4 |
| r1c3 → r3c1 | no | pick | plan | IK found no pose for rung +4 |
| r1c3 → r3c2 | no | pick | plan | IK found no pose for rung +4 |
| r1c3 → r3c3 | no | pick | plan | IK found no pose for rung +4 |
| r2c1 → r1c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r1c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r1c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r2c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r2c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r3c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r3c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r3c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r1c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r1c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r1c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r2c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r2c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r3c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r3c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r3c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c3 → r1c1 | no | pick | plan | IK found no pose for rung +12 |
| r2c3 → r1c2 | no | pick | plan | IK found no pose for rung +12 |
| r2c3 → r1c3 | no | pick | plan | IK found no pose for rung +12 |
| r2c3 → r2c1 | no | pick | plan | IK found no pose for rung +12 |
| r2c3 → r2c2 | no | pick | plan | IK found no pose for rung +12 |
| r2c3 → r3c1 | no | pick | plan | IK found no pose for rung +12 |
| r2c3 → r3c2 | no | pick | plan | IK found no pose for rung +12 |
| r2c3 → r3c3 | no | pick | plan | IK found no pose for rung +12 |
| r3c1 → r1c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r1c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r1c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r2c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r2c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r2c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r3c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r3c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c2 → r1c1 | no | pick | table | finger_pad -4.4 mm into the table on 'lift grasp->+lift' (commanded) |
| r3c2 → r1c2 | no | pick | table | finger_pad -4.4 mm into the table on 'lift grasp->+lift' (commanded) |
| r3c2 → r1c3 | no | pick | table | finger_pad -4.4 mm into the table on 'lift grasp->+lift' (commanded) |
| r3c2 → r2c1 | no | pick | table | finger_pad -4.4 mm into the table on 'lift grasp->+lift' (commanded) |
| r3c2 → r2c2 | no | pick | table | finger_pad -4.4 mm into the table on 'lift grasp->+lift' (commanded) |
| r3c2 → r2c3 | no | pick | table | finger_pad -4.4 mm into the table on 'lift grasp->+lift' (commanded) |
| r3c2 → r3c1 | no | pick | table | finger_pad -4.4 mm into the table on 'lift grasp->+lift' (commanded) |
| r3c2 → r3c3 | no | pick | table | finger_pad -4.4 mm into the table on 'lift grasp->+lift' (commanded) |
| r3c3 → r1c1 | no | pick | table | thumb_pad -1.9 mm into the table on 'descent hover->grasp' (commanded) |
| r3c3 → r1c2 | no | pick | table | thumb_pad -1.9 mm into the table on 'descent hover->grasp' (commanded) |
| r3c3 → r1c3 | no | pick | table | thumb_pad -1.9 mm into the table on 'descent hover->grasp' (commanded) |
| r3c3 → r2c1 | no | pick | table | thumb_pad -1.9 mm into the table on 'descent hover->grasp' (commanded) |
| r3c3 → r2c2 | no | pick | table | thumb_pad -1.9 mm into the table on 'descent hover->grasp' (commanded) |
| r3c3 → r2c3 | no | pick | table | thumb_pad -1.9 mm into the table on 'descent hover->grasp' (commanded) |
| r3c3 → r3c1 | no | pick | table | thumb_pad -1.9 mm into the table on 'descent hover->grasp' (commanded) |
| r3c3 → r3c2 | no | pick | table | thumb_pad -1.9 mm into the table on 'descent hover->grasp' (commanded) |

Refusals by reason:

- 40 × pick / plan: IK found no grasp pose in the bounded start set
- 8 × pick / plan: IK found no pose for rung +4
- 8 × pick / plan: IK found no pose for rung +12
- 8 × pick / table: finger_pad -4.4 mm into the table on 'lift grasp->+lift' (commanded)
- 8 × pick / table: thumb_pad -1.9 mm into the table on 'descent hover->grasp' (commanded)

