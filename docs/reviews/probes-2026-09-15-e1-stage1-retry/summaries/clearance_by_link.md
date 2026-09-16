# Planned vs realised clearance by recording, hand model and link (cm, signed; negative = model overlap)

Pure geometry from the recorded joint samples at each sample's own measured aperture, against the B4 YAML poses (`scripts/measure_route_clearance.py` geometry, called one link at a time). `planned` samples the commanded joint-space line between waypoints with the worst-case endpoint aperture. `all` = whole recording; motion recordings are also split into pre-motion, moving, and parked-tail segments by joint velocity (> 2 deg/s). Zero contacts were reported by the server's contact tracker on every recording; a negative number here is the conservative capsule model overlapping, not a physics contact.

## parked_stage0 — LOWER_TO_REST (parked, 60 samples)
start: gripper -37.9°, wrist_roll 39.8°, wrist_pitch -0.0°; end: gripper -38.1°, wrist_roll 39.8°, wrist_pitch -0.0°

| hand | link | object | planned | real all |
|---|---|---|---|---|
| tube | forearm | pool_box_1 (worst) | +6.5 | +38.3 |
| tube | hand | pool_box_1 (worst) | -3.7 | +38.0 |
| tube | upper_arm | pool_box_1 (worst) | +19.3 | +37.7 |
| shells | finger | pool_box_1 (worst) | +11.4 | +52.9 |
| shells | forearm | pool_box_1 (worst) | +6.5 | +38.3 |
| shells | thumb | pool_box_1 (worst) | +7.4 | +47.5 |
| shells | thumb_pad | pool_box_1 (worst) | +7.6 | +52.3 |
| shells | upper_arm | pool_box_1 (worst) | +19.3 | +37.7 |
| shells | wrist_ball | pool_box_1 (worst) | +6.9 | +46.5 |

## parked_S1a — RAISE_TO_SIDE (parked, 60 samples)
start: gripper -40.1°, wrist_roll 40.1°, wrist_pitch -0.0°; end: gripper -40.1°, wrist_roll 40.1°, wrist_pitch -0.0°

| hand | link | object | planned | real all |
|---|---|---|---|---|
| tube | forearm | pool_box_1 (worst) | +6.3 | +38.3 |
| tube | hand | pool_box_1 (worst) | -3.7 | +37.8 |
| tube | upper_arm | pool_box_1 (worst) | +20.9 | +37.7 |
| shells | finger | pool_box_1 (worst) | +10.8 | +52.9 |
| shells | forearm | pool_box_1 (worst) | +6.3 | +38.3 |
| shells | thumb | pool_box_1 (worst) | +7.4 | +47.5 |
| shells | thumb_pad | pool_box_1 (worst) | +7.6 | +52.3 |
| shells | upper_arm | pool_box_1 (worst) | +20.9 | +37.7 |
| shells | wrist_ball | pool_box_1 (worst) | +6.7 | +46.5 |

## setup_a — RAISE_TO_SIDE (motion, 1192 samples)
moving segment: samples 72–730 (t = 3.61–36.5 s); pre-motion 72 samples; parked tail 461 samples
start: gripper -40.1°, wrist_roll 40.1°, wrist_pitch -0.0°; end: gripper -45.0°, wrist_roll -0.0°, wrist_pitch 0.2°

| hand | link | object | planned | real all | real pre_motion | real moving | real parked_tail |
|---|---|---|---|---|---|---|---|
| tube | forearm | pool_box_1 (worst) | +6.3 | +5.5 | +38.3 | +5.5 | +22.3 |
| tube | hand | pool_box_1 (worst) | -3.7 | -3.8 | +37.8 | -3.8 | +26.9 |
| tube | upper_arm | pool_box_1 (worst) | +20.9 | +19.4 | +37.7 | +19.4 | +21.8 |
| shells | finger | pool_box_1 (worst) | +10.8 | +10.4 | +52.9 | +10.4 | +42.6 |
| shells | forearm | pool_box_1 (worst) | +6.3 | +5.5 | +38.3 | +5.5 | +22.3 |
| shells | thumb | pool_box_1 (worst) | +7.4 | +6.7 | +47.5 | +6.7 | +35.2 |
| shells | thumb_pad | pool_box_1 (worst) | +7.6 | +7.3 | +52.3 | +7.3 | +42.3 |
| shells | upper_arm | pool_box_1 (worst) | +20.9 | +19.4 | +37.7 | +19.4 | +21.8 |
| shells | wrist_ball | pool_box_1 (worst) | +6.7 | +5.7 | +46.5 | +5.7 | +34.0 |

## flight_a — LOWER_TO_REST (motion, 748 samples)
moving segment: samples 70–194 (t = 3.5–9.7 s); pre-motion 70 samples; parked tail 553 samples
start: gripper -45.0°, wrist_roll -0.0°, wrist_pitch 0.2°; end: gripper -45.0°, wrist_roll 30.0°, wrist_pitch -9.8°

| hand | link | object | planned | real all | real pre_motion | real moving | real parked_tail |
|---|---|---|---|---|---|---|---|
| tube | forearm | pool_box_1 (worst) | +6.5 | +6.4 | +22.3 | +6.4 | +6.4 |
| tube | hand | pool_box_1 (worst) | -3.7 | -3.7 | +26.9 | -3.7 | -3.7 |
| tube | upper_arm | pool_box_1 (worst) | +19.3 | +19.6 | +21.8 | +19.6 | +20.9 |
| shells | finger | pool_box_1 (worst) | +11.4 | +10.8 | +42.6 | +10.8 | +11.4 |
| shells | forearm | pool_box_1 (worst) | +6.5 | +6.4 | +22.3 | +6.4 | +6.4 |
| shells | thumb | pool_box_1 (worst) | +7.4 | +7.3 | +35.2 | +7.3 | +7.3 |
| shells | thumb_pad | pool_box_1 (worst) | +7.6 | +7.5 | +42.3 | +7.5 | +7.6 |
| shells | upper_arm | pool_box_1 (worst) | +19.3 | +19.6 | +21.8 | +19.6 | +20.9 |
| shells | wrist_ball | pool_box_1 (worst) | +6.9 | +6.8 | +34.0 | +6.8 | +6.9 |

## parked_S1b — PLACE_ROUTE (parked, 60 samples)
start: gripper -0.0°, wrist_roll 0.1°, wrist_pitch -0.0°; end: gripper -0.0°, wrist_roll 0.1°, wrist_pitch -0.0°

| hand | link | object | planned | real all |
|---|---|---|---|---|
| tube | forearm | pool_box_1 (worst) | +6.3 | +38.3 |
| tube | hand | pool_box_1 (worst) | -3.7 | +44.0 |
| tube | upper_arm | pool_box_1 (worst) | +20.9 | +37.7 |
| shells | finger | pool_box_1 (worst) | +10.8 | +52.2 |
| shells | forearm | pool_box_1 (worst) | +6.3 | +38.3 |
| shells | thumb | pool_box_1 (worst) | +7.4 | +46.5 |
| shells | thumb_pad | pool_box_1 (worst) | +7.6 | +53.5 |
| shells | upper_arm | pool_box_1 (worst) | +20.9 | +37.7 |
| shells | wrist_ball | pool_box_1 (worst) | +6.7 | +46.5 |

## flight_b — PLACE_ROUTE (motion, 1282 samples)
moving segment: samples 76–742 (t = 3.8–37.1 s); pre-motion 76 samples; parked tail 539 samples
start: gripper -0.0°, wrist_roll 0.1°, wrist_pitch -0.0°; end: gripper -44.7°, wrist_roll 30.0°, wrist_pitch -9.7°

| hand | link | object | planned | real all | real pre_motion | real moving | real parked_tail |
|---|---|---|---|---|---|---|---|
| tube | forearm | pool_box_1 (worst) | +6.3 | +5.5 | +38.3 | +5.5 | +6.2 |
| tube | hand | pool_box_1 (worst) | -3.7 | -4.0 | +44.0 | -3.9 | -4.0 |
| tube | upper_arm | pool_box_1 (worst) | +20.9 | +20.9 | +37.7 | +20.9 | +20.9 |
| shells | finger | pool_box_1 (worst) | +10.8 | +10.5 | +52.2 | +10.5 | +11.1 |
| shells | forearm | pool_box_1 (worst) | +6.3 | +5.5 | +38.3 | +5.5 | +6.2 |
| shells | thumb | pool_box_1 (worst) | +7.4 | +6.7 | +46.5 | +6.7 | +7.1 |
| shells | thumb_pad | pool_box_1 (worst) | +7.6 | +7.2 | +53.5 | +7.2 | +7.4 |
| shells | upper_arm | pool_box_1 (worst) | +20.9 | +20.9 | +37.7 | +20.9 | +20.9 |
| shells | wrist_ball | pool_box_1 (worst) | +6.7 | +5.7 | +46.5 | +5.7 | +6.6 |

## parked_S1c — PLACE_ROUTE (parked, 60 samples)
start: gripper -0.0°, wrist_roll 0.1°, wrist_pitch 0.0°; end: gripper -0.0°, wrist_roll 0.1°, wrist_pitch 0.0°

| hand | link | object | planned | real all |
|---|---|---|---|---|
| tube | forearm | pool_box_1 (worst) | +6.3 | +38.3 |
| tube | hand | pool_box_1 (worst) | -3.7 | +44.0 |
| tube | upper_arm | pool_box_1 (worst) | +20.9 | +37.7 |
| shells | finger | pool_box_1 (worst) | +10.8 | +52.2 |
| shells | forearm | pool_box_1 (worst) | +6.3 | +38.3 |
| shells | thumb | pool_box_1 (worst) | +7.4 | +46.5 |
| shells | thumb_pad | pool_box_1 (worst) | +7.6 | +53.5 |
| shells | upper_arm | pool_box_1 (worst) | +20.9 | +37.7 |
| shells | wrist_ball | pool_box_1 (worst) | +6.7 | +46.5 |

## setup_c — PLACE_ROUTE (motion, 1282 samples)
moving segment: samples 77–731 (t = 3.85–36.55 s); pre-motion 77 samples; parked tail 550 samples
start: gripper -0.0°, wrist_roll 0.1°, wrist_pitch -0.0°; end: gripper -44.5°, wrist_roll 30.0°, wrist_pitch -9.6°

| hand | link | object | planned | real all | real pre_motion | real moving | real parked_tail |
|---|---|---|---|---|---|---|---|
| tube | forearm | pool_box_1 (worst) | +6.3 | +5.6 | +38.3 | +5.6 | +6.3 |
| tube | hand | pool_box_1 (worst) | -3.7 | -3.9 | +44.0 | -3.9 | -3.9 |
| tube | upper_arm | pool_box_1 (worst) | +20.9 | +20.9 | +37.7 | +20.9 | +20.9 |
| shells | finger | pool_box_1 (worst) | +10.8 | +10.5 | +52.2 | +10.5 | +11.1 |
| shells | forearm | pool_box_1 (worst) | +6.3 | +5.6 | +38.3 | +5.6 | +6.3 |
| shells | thumb | pool_box_1 (worst) | +7.4 | +6.8 | +46.5 | +6.8 | +7.1 |
| shells | thumb_pad | pool_box_1 (worst) | +7.6 | +7.3 | +53.5 | +7.3 | +7.4 |
| shells | upper_arm | pool_box_1 (worst) | +20.9 | +20.9 | +37.7 | +20.9 | +20.9 |
| shells | wrist_ball | pool_box_1 (worst) | +6.7 | +5.9 | +46.5 | +5.9 | +6.7 |

## flight_c — LIFT_TO_PRESENT (motion, 682 samples)
moving segment: samples 70–125 (t = 3.5–6.25 s); pre-motion 70 samples; parked tail 556 samples
start: gripper -44.5°, wrist_roll 30.0°, wrist_pitch -9.6°; end: gripper -45.1°, wrist_roll 0.0°, wrist_pitch 0.2°

| hand | link | object | planned | real all | real pre_motion | real moving | real parked_tail |
|---|---|---|---|---|---|---|---|
| tube | forearm | pool_box_1 (worst) | +6.5 | +6.2 | +6.2 | +6.2 | +22.1 |
| tube | hand | pool_box_1 (worst) | -3.7 | -4.0 | -4.0 | -4.0 | +26.2 |
| tube | upper_arm | pool_box_1 (worst) | +19.3 | +19.4 | +20.9 | +19.4 | +21.6 |
| shells | finger | pool_box_1 (worst) | +11.4 | +11.1 | +11.1 | +11.1 | +41.9 |
| shells | forearm | pool_box_1 (worst) | +6.5 | +6.2 | +6.2 | +6.2 | +22.1 |
| shells | thumb | pool_box_1 (worst) | +7.4 | +7.1 | +7.1 | +7.1 | +34.5 |
| shells | thumb_pad | pool_box_1 (worst) | +7.6 | +7.4 | +7.4 | +7.4 | +41.5 |
| shells | upper_arm | pool_box_1 (worst) | +19.3 | +19.4 | +20.9 | +19.4 | +21.6 |
| shells | wrist_ball | pool_box_1 (worst) | +6.9 | +6.6 | +6.6 | +6.6 | +33.2 |
