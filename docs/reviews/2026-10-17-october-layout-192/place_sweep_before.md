# Issue #174 offline feasibility sweep — placement (2026-10-18)

reachy-1-2-sim at `db0c0ef`.  Preflight only (`crane_place_live.plan_crane_place`): no motion, no simulator.  IK: the host mirror of the simulator's service.  Scene FWDCenterLabSivaPool, the object upright at the origin cell's centre, nothing else on the board.  Reachable cells: the seven without the `no-right-arm` tag.

## red_cube

Feasible: **3 of 42** ordered pairs.  r2c2→r1c1, r2c2→r1c2, r2c2→r2c1

| from → to | feasible | refused at | stage | reason |
|---|---|---|---|---|
| r1c1 → r1c2 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r1c1 → r1c3 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r1c1 → r2c1 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r1c1 → r2c2 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r1c1 → r2c3 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r1c1 → r3c3 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r1c2 → r1c1 | no | pick | check_arm_path | descent hover->grasp: hand would come within -2.6 cm of rig_rail_inner_left (tube hand model, margin 0.0 cm, gripper -65 deg) |
| r1c2 → r1c3 | no | pick | check_arm_path | descent hover->grasp: hand would come within -2.6 cm of rig_rail_inner_left (tube hand model, margin 0.0 cm, gripper -65 deg) |
| r1c2 → r2c1 | no | pick | check_arm_path | descent hover->grasp: hand would come within -2.6 cm of rig_rail_inner_left (tube hand model, margin 0.0 cm, gripper -65 deg) |
| r1c2 → r2c2 | no | pick | check_arm_path | descent hover->grasp: hand would come within -2.6 cm of rig_rail_inner_left (tube hand model, margin 0.0 cm, gripper -65 deg) |
| r1c2 → r2c3 | no | pick | check_arm_path | descent hover->grasp: hand would come within -2.6 cm of rig_rail_inner_left (tube hand model, margin 0.0 cm, gripper -65 deg) |
| r1c2 → r3c3 | no | pick | check_arm_path | descent hover->grasp: hand would come within -2.6 cm of rig_rail_inner_left (tube hand model, margin 0.0 cm, gripper -65 deg) |
| r1c3 → r1c1 | no | pick | plan | IK found no pose for rung +9 |
| r1c3 → r1c2 | no | pick | plan | IK found no pose for rung +9 |
| r1c3 → r2c1 | no | pick | plan | IK found no pose for rung +9 |
| r1c3 → r2c2 | no | pick | plan | IK found no pose for rung +9 |
| r1c3 → r2c3 | no | pick | plan | IK found no pose for rung +9 |
| r1c3 → r3c3 | no | pick | plan | IK found no pose for rung +9 |
| r2c1 → r1c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r1c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r1c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r2c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r2c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r3c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r1c1 | **yes** |  |  |  |
| r2c2 → r1c2 | **yes** |  |  |  |
| r2c2 → r1c3 | no | destination | plan | IK found no pose for rung +7 |
| r2c2 → r2c1 | **yes** |  |  |  |
| r2c2 → r2c3 | no | destination | exit | red_cube on r2c3 would leave no validated exit from PRESENT: rest (LOWER_TO_REST) -52 mm; home (STOW_FROM_SIDE) -52 mm |
| r2c2 → r3c3 | no | destination | destination | IK found no pose that sets the object down on the target with the grasp's attitude |
| r2c3 → r1c1 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r2c3 → r1c2 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r2c3 → r1c3 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r2c3 → r2c1 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r2c3 → r2c2 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r2c3 → r3c3 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r3c3 → r1c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c3 → r1c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c3 → r1c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c3 → r2c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c3 → r2c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c3 → r2c3 | no | pick | plan | IK found no grasp pose in the bounded start set |

Refusals by reason:

- 12 × pick / plan: pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range)
- 12 × pick / plan: IK found no grasp pose in the bounded start set
- 6 × pick / check_arm_path: descent hover->grasp: hand would come within -2.6 cm of rig_rail_inner_left (tube hand model, margin 0.0 cm, gripper -65 deg)
- 6 × pick / plan: IK found no pose for rung +9
- 1 × destination / plan: IK found no pose for rung +7
- 1 × destination / exit: red_cube on r2c3 would leave no validated exit from PRESENT: rest (LOWER_TO_REST) -52 mm; home (STOW_FROM_SIDE) -52 mm
- 1 × destination / destination: IK found no pose that sets the object down on the target with the grasp's attitude

## foam_block

Feasible: **0 of 42** ordered pairs.

| from → to | feasible | refused at | stage | reason |
|---|---|---|---|---|
| r1c1 → r1c2 | no | pick | table | finger_pad -10.0 mm into the table on 'descent hover->grasp' (commanded) |
| r1c1 → r1c3 | no | pick | table | finger_pad -10.0 mm into the table on 'descent hover->grasp' (commanded) |
| r1c1 → r2c1 | no | pick | table | finger_pad -10.0 mm into the table on 'descent hover->grasp' (commanded) |
| r1c1 → r2c2 | no | pick | table | finger_pad -10.0 mm into the table on 'descent hover->grasp' (commanded) |
| r1c1 → r2c3 | no | pick | table | finger_pad -10.0 mm into the table on 'descent hover->grasp' (commanded) |
| r1c1 → r3c3 | no | pick | table | finger_pad -10.0 mm into the table on 'descent hover->grasp' (commanded) |
| r1c2 → r1c1 | no | pick | check_arm_path | descent hover->grasp: hand would come within -3.0 cm of rig_rail_inner_left (tube hand model, margin 0.0 cm, gripper -65 deg) |
| r1c2 → r1c3 | no | pick | check_arm_path | descent hover->grasp: hand would come within -3.0 cm of rig_rail_inner_left (tube hand model, margin 0.0 cm, gripper -65 deg) |
| r1c2 → r2c1 | no | pick | check_arm_path | descent hover->grasp: hand would come within -3.0 cm of rig_rail_inner_left (tube hand model, margin 0.0 cm, gripper -65 deg) |
| r1c2 → r2c2 | no | pick | check_arm_path | descent hover->grasp: hand would come within -3.0 cm of rig_rail_inner_left (tube hand model, margin 0.0 cm, gripper -65 deg) |
| r1c2 → r2c3 | no | pick | check_arm_path | descent hover->grasp: hand would come within -3.0 cm of rig_rail_inner_left (tube hand model, margin 0.0 cm, gripper -65 deg) |
| r1c2 → r3c3 | no | pick | check_arm_path | descent hover->grasp: hand would come within -3.0 cm of rig_rail_inner_left (tube hand model, margin 0.0 cm, gripper -65 deg) |
| r1c3 → r1c1 | no | pick | plan | IK found no pose for rung +9 |
| r1c3 → r1c2 | no | pick | plan | IK found no pose for rung +9 |
| r1c3 → r2c1 | no | pick | plan | IK found no pose for rung +9 |
| r1c3 → r2c2 | no | pick | plan | IK found no pose for rung +9 |
| r1c3 → r2c3 | no | pick | plan | IK found no pose for rung +9 |
| r1c3 → r3c3 | no | pick | plan | IK found no pose for rung +9 |
| r2c1 → r1c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r1c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r1c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r2c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r2c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r3c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r1c1 | no | pick | pad_route | thumb pad route clearance -2.06 mm < 0.494 mm from rung +9 |
| r2c2 → r1c2 | no | pick | pad_route | thumb pad route clearance -2.06 mm < 0.494 mm from rung +9 |
| r2c2 → r1c3 | no | pick | pad_route | thumb pad route clearance -2.06 mm < 0.494 mm from rung +9 |
| r2c2 → r2c1 | no | pick | pad_route | thumb pad route clearance -2.06 mm < 0.494 mm from rung +9 |
| r2c2 → r2c3 | no | pick | pad_route | thumb pad route clearance -2.06 mm < 0.494 mm from rung +9 |
| r2c2 → r3c3 | no | pick | pad_route | thumb pad route clearance -2.06 mm < 0.494 mm from rung +9 |
| r2c3 → r1c1 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r2c3 → r1c2 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r2c3 → r1c3 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r2c3 → r2c1 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r2c3 → r2c2 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r2c3 → r3c3 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r3c3 → r1c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c3 → r1c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c3 → r1c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c3 → r2c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c3 → r2c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c3 → r2c3 | no | pick | plan | IK found no grasp pose in the bounded start set |

Refusals by reason:

- 12 × pick / plan: IK found no grasp pose in the bounded start set
- 6 × pick / table: finger_pad -10.0 mm into the table on 'descent hover->grasp' (commanded)
- 6 × pick / check_arm_path: descent hover->grasp: hand would come within -3.0 cm of rig_rail_inner_left (tube hand model, margin 0.0 cm, gripper -65 deg)
- 6 × pick / plan: IK found no pose for rung +9
- 6 × pick / pad_route: thumb pad route clearance -2.06 mm < 0.494 mm from rung +9
- 6 × pick / plan: pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range)

## pool_box_1

Feasible: **3 of 42** ordered pairs.  r2c2→r1c1, r2c2→r1c2, r2c2→r2c1

| from → to | feasible | refused at | stage | reason |
|---|---|---|---|---|
| r1c1 → r1c2 | no | pick | table | finger_pad -15.0 mm into the table on 'descent hover->grasp' (commanded) |
| r1c1 → r1c3 | no | pick | table | finger_pad -15.0 mm into the table on 'descent hover->grasp' (commanded) |
| r1c1 → r2c1 | no | pick | table | finger_pad -15.0 mm into the table on 'descent hover->grasp' (commanded) |
| r1c1 → r2c2 | no | pick | table | finger_pad -15.0 mm into the table on 'descent hover->grasp' (commanded) |
| r1c1 → r2c3 | no | pick | table | finger_pad -15.0 mm into the table on 'descent hover->grasp' (commanded) |
| r1c1 → r3c3 | no | pick | table | finger_pad -15.0 mm into the table on 'descent hover->grasp' (commanded) |
| r1c2 → r1c1 | no | pick | check_arm_path | descent hover->grasp: hand would come within -3.3 cm of rig_rail_inner_left (tube hand model, margin 0.0 cm, gripper -65 deg) |
| r1c2 → r1c3 | no | pick | check_arm_path | descent hover->grasp: hand would come within -3.3 cm of rig_rail_inner_left (tube hand model, margin 0.0 cm, gripper -65 deg) |
| r1c2 → r2c1 | no | pick | check_arm_path | descent hover->grasp: hand would come within -3.3 cm of rig_rail_inner_left (tube hand model, margin 0.0 cm, gripper -65 deg) |
| r1c2 → r2c2 | no | pick | check_arm_path | descent hover->grasp: hand would come within -3.3 cm of rig_rail_inner_left (tube hand model, margin 0.0 cm, gripper -65 deg) |
| r1c2 → r2c3 | no | pick | check_arm_path | descent hover->grasp: hand would come within -3.3 cm of rig_rail_inner_left (tube hand model, margin 0.0 cm, gripper -65 deg) |
| r1c2 → r3c3 | no | pick | check_arm_path | descent hover->grasp: hand would come within -3.3 cm of rig_rail_inner_left (tube hand model, margin 0.0 cm, gripper -65 deg) |
| r1c3 → r1c1 | no | pick | plan | IK found no pose for rung +10 |
| r1c3 → r1c2 | no | pick | plan | IK found no pose for rung +10 |
| r1c3 → r2c1 | no | pick | plan | IK found no pose for rung +10 |
| r1c3 → r2c2 | no | pick | plan | IK found no pose for rung +10 |
| r1c3 → r2c3 | no | pick | plan | IK found no pose for rung +10 |
| r1c3 → r3c3 | no | pick | plan | IK found no pose for rung +10 |
| r2c1 → r1c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r1c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r1c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r2c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r2c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r3c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r1c1 | **yes** |  |  |  |
| r2c2 → r1c2 | **yes** |  |  |  |
| r2c2 → r1c3 | no | destination | plan | IK found no pose for rung +8 |
| r2c2 → r2c1 | **yes** |  |  |  |
| r2c2 → r2c3 | no | destination | exit | pool_box_1 on r2c3 would leave no validated exit from PRESENT: rest (LOWER_TO_REST) -38 mm; home (STOW_FROM_SIDE) -38 mm |
| r2c2 → r3c3 | no | destination | destination | IK found no pose that sets the object down on the target with the grasp's attitude |
| r2c3 → r1c1 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r2c3 → r1c2 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r2c3 → r1c3 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r2c3 → r2c1 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r2c3 → r2c2 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r2c3 → r3c3 | no | pick | plan | pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range) |
| r3c3 → r1c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c3 → r1c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c3 → r1c3 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c3 → r2c1 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c3 → r2c2 | no | pick | plan | IK found no grasp pose in the bounded start set |
| r3c3 → r2c3 | no | pick | plan | IK found no grasp pose in the bounded start set |

Refusals by reason:

- 12 × pick / plan: IK found no grasp pose in the bounded start set
- 6 × pick / table: finger_pad -15.0 mm into the table on 'descent hover->grasp' (commanded)
- 6 × pick / check_arm_path: descent hover->grasp: hand would come within -3.3 cm of rig_rail_inner_left (tube hand model, margin 0.0 cm, gripper -65 deg)
- 6 × pick / plan: IK found no pose for rung +10
- 6 × pick / plan: pads would meet the object's top more than 7 cm below the insertion checkpoint (beyond the recorded prediction range)
- 1 × destination / plan: IK found no pose for rung +8
- 1 × destination / exit: pool_box_1 on r2c3 would leave no validated exit from PRESENT: rest (LOWER_TO_REST) -38 mm; home (STOW_FROM_SIDE) -38 mm
- 1 × destination / destination: IK found no pose that sets the object down on the target with the grasp's attitude

