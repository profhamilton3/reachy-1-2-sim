# Issue #174 offline feasibility sweep — placement (2026-10-18)

reachy-1-2-sim at `feat/192-october-layout`.  Preflight only (`crane_place_live.plan_crane_place`): no motion, no simulator.  IK: the host mirror of the simulator's service.  Scene FWDCenterLabSivaPool, the object upright at the origin cell's centre, nothing else on the board.  Reachable cells: the nine without the `no-right-arm` tag.

## red_cube

Feasible: **7 of 72** ordered pairs.  r3c2→r1c1, r3c2→r1c2, r3c2→r2c1, r3c2→r2c2, r3c2→r2c3, r3c2→r3c1, r3c2→r3c3

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
| r1c2 → r1c1 | no | pick | plan | IK lost the grasp pose while centring it |
| r1c2 → r1c3 | no | pick | plan | IK lost the grasp pose while centring it |
| r1c2 → r2c1 | no | pick | plan | IK lost the grasp pose while centring it |
| r1c2 → r2c2 | no | pick | plan | IK lost the grasp pose while centring it |
| r1c2 → r2c3 | no | pick | plan | IK lost the grasp pose while centring it |
| r1c2 → r3c1 | no | pick | plan | IK lost the grasp pose while centring it |
| r1c2 → r3c2 | no | pick | plan | IK lost the grasp pose while centring it |
| r1c2 → r3c3 | no | pick | plan | IK lost the grasp pose while centring it |
| r1c3 → r1c1 | no | pick | plan | IK found no pose for rung +9 |
| r1c3 → r1c2 | no | pick | plan | IK found no pose for rung +9 |
| r1c3 → r2c1 | no | pick | plan | IK found no pose for rung +9 |
| r1c3 → r2c2 | no | pick | plan | IK found no pose for rung +9 |
| r1c3 → r2c3 | no | pick | plan | IK found no pose for rung +9 |
| r1c3 → r3c1 | no | pick | plan | IK found no pose for rung +9 |
| r1c3 → r3c2 | no | pick | plan | IK found no pose for rung +9 |
| r1c3 → r3c3 | no | pick | plan | IK found no pose for rung +9 |
| r2c1 → r1c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r1c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r1c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r2c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r2c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r3c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r3c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r3c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r1c1 | no | pick | pad_route | finger pad route clearance -5.78 mm < 0.494 mm from rung +9 |
| r2c2 → r1c2 | no | pick | pad_route | finger pad route clearance -5.78 mm < 0.494 mm from rung +9 |
| r2c2 → r1c3 | no | pick | pad_route | finger pad route clearance -5.78 mm < 0.494 mm from rung +9 |
| r2c2 → r2c1 | no | pick | pad_route | finger pad route clearance -5.78 mm < 0.494 mm from rung +9 |
| r2c2 → r2c3 | no | pick | pad_route | finger pad route clearance -5.78 mm < 0.494 mm from rung +9 |
| r2c2 → r3c1 | no | pick | pad_route | finger pad route clearance -5.78 mm < 0.494 mm from rung +9 |
| r2c2 → r3c2 | no | pick | pad_route | finger pad route clearance -5.78 mm < 0.494 mm from rung +9 |
| r2c2 → r3c3 | no | pick | pad_route | finger pad route clearance -5.78 mm < 0.494 mm from rung +9 |
| r2c3 → r1c1 | no | pick | plan | IK found no pose for rung +17 |
| r2c3 → r1c2 | no | pick | plan | IK found no pose for rung +17 |
| r2c3 → r1c3 | no | pick | plan | IK found no pose for rung +17 |
| r2c3 → r2c1 | no | pick | plan | IK found no pose for rung +17 |
| r2c3 → r2c2 | no | pick | plan | IK found no pose for rung +17 |
| r2c3 → r3c1 | no | pick | plan | IK found no pose for rung +17 |
| r2c3 → r3c2 | no | pick | plan | IK found no pose for rung +17 |
| r2c3 → r3c3 | no | pick | plan | IK found no pose for rung +17 |
| r3c1 → r1c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r1c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r1c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r2c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r2c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r2c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r3c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r3c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c2 → r1c1 | **yes** |  |  |  |
| r3c2 → r1c2 | **yes** |  |  |  |
| r3c2 → r1c3 | no | destination | plan | IK found no pose for rung +9 |
| r3c2 → r2c1 | **yes** |  |  |  |
| r3c2 → r2c2 | **yes** |  |  |  |
| r3c2 → r2c3 | **yes** |  |  |  |
| r3c2 → r3c1 | **yes** |  |  |  |
| r3c2 → r3c3 | **yes** |  |  |  |
| r3c3 → r1c1 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r3c3 → r1c2 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r3c3 → r1c3 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r3c3 → r2c1 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r3c3 → r2c2 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r3c3 → r2c3 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r3c3 → r3c1 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r3c3 → r3c2 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |

Refusals by reason:

- 24 × pick / plan: IK found no grasp pose in the bounded start set
- 8 × pick / plan: IK lost the grasp pose while centring it
- 8 × pick / plan: IK found no pose for rung +9
- 8 × pick / pad_route: finger pad route clearance -5.78 mm < 0.494 mm from rung +9
- 8 × pick / plan: IK found no pose for rung +17
- 8 × pick / plan: pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range)
- 1 × destination / plan: IK found no pose for rung +9

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
| r1c2 → r1c1 | no | pick | plan | IK lost the grasp pose while centring it |
| r1c2 → r1c3 | no | pick | plan | IK lost the grasp pose while centring it |
| r1c2 → r2c1 | no | pick | plan | IK lost the grasp pose while centring it |
| r1c2 → r2c2 | no | pick | plan | IK lost the grasp pose while centring it |
| r1c2 → r2c3 | no | pick | plan | IK lost the grasp pose while centring it |
| r1c2 → r3c1 | no | pick | plan | IK lost the grasp pose while centring it |
| r1c2 → r3c2 | no | pick | plan | IK lost the grasp pose while centring it |
| r1c2 → r3c3 | no | pick | plan | IK lost the grasp pose while centring it |
| r1c3 → r1c1 | no | pick | plan | IK found no pose for rung +9 |
| r1c3 → r1c2 | no | pick | plan | IK found no pose for rung +9 |
| r1c3 → r2c1 | no | pick | plan | IK found no pose for rung +9 |
| r1c3 → r2c2 | no | pick | plan | IK found no pose for rung +9 |
| r1c3 → r2c3 | no | pick | plan | IK found no pose for rung +9 |
| r1c3 → r3c1 | no | pick | plan | IK found no pose for rung +9 |
| r1c3 → r3c2 | no | pick | plan | IK found no pose for rung +9 |
| r1c3 → r3c3 | no | pick | plan | IK found no pose for rung +9 |
| r2c1 → r1c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r1c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r1c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r2c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r2c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r3c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r3c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r3c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r1c1 | no | pick | table | finger_pad -1.4 mm into the table on 'descent hover->grasp' (commanded) |
| r2c2 → r1c2 | no | pick | table | finger_pad -1.4 mm into the table on 'descent hover->grasp' (commanded) |
| r2c2 → r1c3 | no | pick | table | finger_pad -1.4 mm into the table on 'descent hover->grasp' (commanded) |
| r2c2 → r2c1 | no | pick | table | finger_pad -1.4 mm into the table on 'descent hover->grasp' (commanded) |
| r2c2 → r2c3 | no | pick | table | finger_pad -1.4 mm into the table on 'descent hover->grasp' (commanded) |
| r2c2 → r3c1 | no | pick | table | finger_pad -1.4 mm into the table on 'descent hover->grasp' (commanded) |
| r2c2 → r3c2 | no | pick | table | finger_pad -1.4 mm into the table on 'descent hover->grasp' (commanded) |
| r2c2 → r3c3 | no | pick | table | finger_pad -1.4 mm into the table on 'descent hover->grasp' (commanded) |
| r2c3 → r1c1 | no | pick | plan | IK found no pose for rung +18 |
| r2c3 → r1c2 | no | pick | plan | IK found no pose for rung +18 |
| r2c3 → r1c3 | no | pick | plan | IK found no pose for rung +18 |
| r2c3 → r2c1 | no | pick | plan | IK found no pose for rung +18 |
| r2c3 → r2c2 | no | pick | plan | IK found no pose for rung +18 |
| r2c3 → r3c1 | no | pick | plan | IK found no pose for rung +18 |
| r2c3 → r3c2 | no | pick | plan | IK found no pose for rung +18 |
| r2c3 → r3c3 | no | pick | plan | IK found no pose for rung +18 |
| r3c1 → r1c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r1c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r1c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r2c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r2c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r2c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r3c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r3c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c2 → r1c1 | no | pick | pad_route | finger pad route clearance -2.22 mm < 0.494 mm from rung +9 |
| r3c2 → r1c2 | no | pick | pad_route | finger pad route clearance -2.22 mm < 0.494 mm from rung +9 |
| r3c2 → r1c3 | no | pick | pad_route | finger pad route clearance -2.22 mm < 0.494 mm from rung +9 |
| r3c2 → r2c1 | no | pick | pad_route | finger pad route clearance -2.22 mm < 0.494 mm from rung +9 |
| r3c2 → r2c2 | no | pick | pad_route | finger pad route clearance -2.22 mm < 0.494 mm from rung +9 |
| r3c2 → r2c3 | no | pick | pad_route | finger pad route clearance -2.22 mm < 0.494 mm from rung +9 |
| r3c2 → r3c1 | no | pick | pad_route | finger pad route clearance -2.22 mm < 0.494 mm from rung +9 |
| r3c2 → r3c3 | no | pick | pad_route | finger pad route clearance -2.22 mm < 0.494 mm from rung +9 |
| r3c3 → r1c1 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r3c3 → r1c2 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r3c3 → r1c3 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r3c3 → r2c1 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r3c3 → r2c2 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r3c3 → r2c3 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r3c3 → r3c1 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r3c3 → r3c2 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |

Refusals by reason:

- 24 × pick / plan: IK found no grasp pose in the bounded start set
- 8 × pick / plan: IK lost the grasp pose while centring it
- 8 × pick / plan: IK found no pose for rung +9
- 8 × pick / table: finger_pad -1.4 mm into the table on 'descent hover->grasp' (commanded)
- 8 × pick / plan: IK found no pose for rung +18
- 8 × pick / pad_route: finger pad route clearance -2.22 mm < 0.494 mm from rung +9
- 8 × pick / plan: pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range)

## pool_box_1

Feasible: **6 of 72** ordered pairs.  r3c2→r1c1, r3c2→r1c2, r3c2→r2c1, r3c2→r2c2, r3c2→r2c3, r3c2→r3c3

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
| r1c2 → r1c1 | no | pick | plan | IK found no pose for rung +13 |
| r1c2 → r1c3 | no | pick | plan | IK found no pose for rung +13 |
| r1c2 → r2c1 | no | pick | plan | IK found no pose for rung +13 |
| r1c2 → r2c2 | no | pick | plan | IK found no pose for rung +13 |
| r1c2 → r2c3 | no | pick | plan | IK found no pose for rung +13 |
| r1c2 → r3c1 | no | pick | plan | IK found no pose for rung +13 |
| r1c2 → r3c2 | no | pick | plan | IK found no pose for rung +13 |
| r1c2 → r3c3 | no | pick | plan | IK found no pose for rung +13 |
| r1c3 → r1c1 | no | pick | plan | IK found no pose for rung +10 |
| r1c3 → r1c2 | no | pick | plan | IK found no pose for rung +10 |
| r1c3 → r2c1 | no | pick | plan | IK found no pose for rung +10 |
| r1c3 → r2c2 | no | pick | plan | IK found no pose for rung +10 |
| r1c3 → r2c3 | no | pick | plan | IK found no pose for rung +10 |
| r1c3 → r3c1 | no | pick | plan | IK found no pose for rung +10 |
| r1c3 → r3c2 | no | pick | plan | IK found no pose for rung +10 |
| r1c3 → r3c3 | no | pick | plan | IK found no pose for rung +10 |
| r2c1 → r1c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r1c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r1c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r2c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r2c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r3c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r3c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r3c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r1c1 | no | pick | table | finger_pad -6.4 mm into the table on 'descent hover->grasp' (commanded) |
| r2c2 → r1c2 | no | pick | table | finger_pad -6.4 mm into the table on 'descent hover->grasp' (commanded) |
| r2c2 → r1c3 | no | pick | table | finger_pad -6.4 mm into the table on 'descent hover->grasp' (commanded) |
| r2c2 → r2c1 | no | pick | table | finger_pad -6.4 mm into the table on 'descent hover->grasp' (commanded) |
| r2c2 → r2c3 | no | pick | table | finger_pad -6.4 mm into the table on 'descent hover->grasp' (commanded) |
| r2c2 → r3c1 | no | pick | table | finger_pad -6.4 mm into the table on 'descent hover->grasp' (commanded) |
| r2c2 → r3c2 | no | pick | table | finger_pad -6.4 mm into the table on 'descent hover->grasp' (commanded) |
| r2c2 → r3c3 | no | pick | table | finger_pad -6.4 mm into the table on 'descent hover->grasp' (commanded) |
| r2c3 → r1c1 | no | pick | plan | IK found no pose for rung +18 |
| r2c3 → r1c2 | no | pick | plan | IK found no pose for rung +18 |
| r2c3 → r1c3 | no | pick | plan | IK found no pose for rung +18 |
| r2c3 → r2c1 | no | pick | plan | IK found no pose for rung +18 |
| r2c3 → r2c2 | no | pick | plan | IK found no pose for rung +18 |
| r2c3 → r3c1 | no | pick | plan | IK found no pose for rung +18 |
| r2c3 → r3c2 | no | pick | plan | IK found no pose for rung +18 |
| r2c3 → r3c3 | no | pick | plan | IK found no pose for rung +18 |
| r3c1 → r1c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r1c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r1c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r2c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r2c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r2c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r3c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r3c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c2 → r1c1 | **yes** |  |  |  |
| r3c2 → r1c2 | **yes** |  |  |  |
| r3c2 → r1c3 | no | destination | plan | IK found no pose for rung +10 |
| r3c2 → r2c1 | **yes** |  |  |  |
| r3c2 → r2c2 | **yes** |  |  |  |
| r3c2 → r2c3 | **yes** |  |  |  |
| r3c2 → r3c1 | no | destination | destination | IK found no pose that sets the object down on the target with the grasp's attitude |
| r3c2 → r3c3 | **yes** |  |  |  |
| r3c3 → r1c1 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r3c3 → r1c2 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r3c3 → r1c3 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r3c3 → r2c1 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r3c3 → r2c2 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r3c3 → r2c3 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r3c3 → r3c1 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r3c3 → r3c2 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |

Refusals by reason:

- 24 × pick / plan: IK found no grasp pose in the bounded start set
- 8 × pick / plan: IK found no pose for rung +13
- 8 × pick / plan: IK found no pose for rung +10
- 8 × pick / table: finger_pad -6.4 mm into the table on 'descent hover->grasp' (commanded)
- 8 × pick / plan: IK found no pose for rung +18
- 8 × pick / plan: pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range)
- 1 × destination / plan: IK found no pose for rung +10
- 1 × destination / destination: IK found no pose that sets the object down on the target with the grasp's attitude

