# Planned minus realised clearance by route x role x hand x link, per B1 board object, across repetitions (cm)

Pure geometry from the recorded joint samples at each sample's own measured aperture, against the B1 YAML poses (`soda_can` r1c1, `foam_block` r2c3; same method as Stage 1's `clearance_by_link.py` and the B4 session's). `realised` for setups/flights is the **moving** segment (joint velocity > 2 deg/s); parked recordings use the whole 3 s recording. `delta` = planned - realised: positive means the flight came closer than the planned corridor. n = recordings in the group (6 per route/role for a/b shapes, 5 for c-shape routes because S2-B1-c-r6 was not flown; 6 or 11 for parked). Zero physics contacts on every recording; negative clearance is the conservative capsule model overlapping, not a contact. Deterministic simulator: the repetitions of one route are near-identical re-runs of one open-loop trajectory from one verified reset state, so agreement across them is evidence of repeatability of this procedure on this board, NOT independent evidence of broad reliability. No sleep-affected cycle in this session.

## Object `foam_block` -- all flown cycles

| route | role | hand | link | n | planned | realised min..max | delta mean (min..max) | parked-tail min |
|---|---|---|---|---|---|---|---|---|
| LIFT_TO_PRESENT | flight | shells | finger | 5 | +9.42 | +9.14..+9.39 | +0.17 (+0.03..+0.27) | +39.95 |
| LIFT_TO_PRESENT | flight | shells | forearm | 5 | +4.79 | +4.57..+4.78 | +0.13 (+0.01..+0.21) | +19.71 |
| LIFT_TO_PRESENT | flight | shells | thumb | 5 | +5.40 | +5.12..+5.36 | +0.17 (+0.03..+0.28) | +32.50 |
| LIFT_TO_PRESENT | flight | shells | thumb_pad | 5 | +5.53 | +5.27..+5.49 | +0.17 (+0.05..+0.27) | +39.63 |
| LIFT_TO_PRESENT | flight | shells | upper_arm | 5 | +17.08 | +16.99..+17.15 | +0.01 (-0.07..+0.08) | +19.21 |
| LIFT_TO_PRESENT | flight | shells | wrist_ball | 5 | +5.39 | +5.15..+5.37 | +0.15 (+0.02..+0.24) | +31.23 |
| LIFT_TO_PRESENT | flight | tube | forearm | 5 | +4.79 | +4.57..+4.78 | +0.13 (+0.01..+0.21) | +19.71 |
| LIFT_TO_PRESENT | flight | tube | hand | 5 | -5.70 | -5.92..-5.72 | +0.14 (+0.02..+0.22) | +24.15 |
| LIFT_TO_PRESENT | flight | tube | upper_arm | 5 | +17.08 | +16.99..+17.15 | +0.01 (-0.07..+0.08) | +19.21 |
| LOWER_TO_REST | flight | shells | finger | 6 | +9.42 | +8.66..+8.73 | +0.72 (+0.69..+0.76) | +9.18 |
| LOWER_TO_REST | flight | shells | forearm | 6 | +4.79 | +4.48..+4.72 | +0.13 (+0.06..+0.31) | +4.57 |
| LOWER_TO_REST | flight | shells | thumb | 6 | +5.40 | +5.03..+5.30 | +0.17 (+0.10..+0.36) | +5.16 |
| LOWER_TO_REST | flight | shells | thumb_pad | 6 | +5.53 | +5.20..+5.43 | +0.16 (+0.11..+0.33) | +5.33 |
| LOWER_TO_REST | flight | shells | upper_arm | 6 | +17.08 | +17.32..+17.47 | -0.32 (-0.39..-0.24) | +19.02 |
| LOWER_TO_REST | flight | shells | wrist_ball | 6 | +5.39 | +5.05..+5.32 | +0.15 (+0.08..+0.34) | +5.16 |
| LOWER_TO_REST | flight | tube | forearm | 6 | +4.79 | +4.48..+4.72 | +0.13 (+0.06..+0.31) | +4.57 |
| LOWER_TO_REST | flight | tube | hand | 6 | -5.70 | -5.94..-5.72 | +0.10 (+0.02..+0.24) | -5.93 |
| LOWER_TO_REST | flight | tube | upper_arm | 6 | +17.08 | +17.32..+17.47 | -0.32 (-0.39..-0.24) | +19.02 |
| PLACE_ROUTE | flight | shells | finger | 6 | +8.63 | +8.32..+8.48 | +0.24 (+0.15..+0.32) | +9.07 |
| PLACE_ROUTE | flight | shells | forearm | 6 | +4.48 | +3.78..+3.89 | +0.64 (+0.58..+0.70) | +4.51 |
| PLACE_ROUTE | flight | shells | thumb | 6 | +5.40 | +4.85..+5.06 | +0.43 (+0.34..+0.54) | +5.05 |
| PLACE_ROUTE | flight | shells | thumb_pad | 6 | +5.53 | +5.03..+5.24 | +0.39 (+0.29..+0.50) | +5.21 |
| PLACE_ROUTE | flight | shells | upper_arm | 6 | +19.04 | +19.03..+19.05 | +0.00 (-0.01..+0.01) | +19.02 |
| PLACE_ROUTE | flight | shells | wrist_ball | 6 | +5.05 | +4.17..+4.32 | +0.81 (+0.73..+0.88) | +5.08 |
| PLACE_ROUTE | flight | tube | forearm | 6 | +4.48 | +3.78..+3.89 | +0.64 (+0.58..+0.70) | +4.51 |
| PLACE_ROUTE | flight | tube | hand | 6 | -5.70 | -5.97..-5.81 | +0.18 (+0.11..+0.27) | -5.99 |
| PLACE_ROUTE | flight | tube | upper_arm | 6 | +19.04 | +19.03..+19.05 | +0.00 (-0.01..+0.01) | +19.02 |
| PLACE_ROUTE | parked | shells | finger | 11 | +8.63 | +50.95..+50.95 | -42.32 (-42.32..-42.32) | n/a |
| PLACE_ROUTE | parked | shells | forearm | 11 | +4.48 | +36.73..+36.73 | -32.25 (-32.25..-32.25) | n/a |
| PLACE_ROUTE | parked | shells | thumb | 11 | +5.40 | +45.23..+45.23 | -39.84 (-39.84..-39.84) | n/a |
| PLACE_ROUTE | parked | shells | thumb_pad | 11 | +5.53 | +52.34..+52.34 | -46.80 (-46.80..-46.80) | n/a |
| PLACE_ROUTE | parked | shells | upper_arm | 11 | +19.04 | +36.18..+36.18 | -17.14 (-17.14..-17.14) | n/a |
| PLACE_ROUTE | parked | shells | wrist_ball | 11 | +5.05 | +45.20..+45.20 | -40.15 (-40.15..-40.15) | n/a |
| PLACE_ROUTE | parked | tube | forearm | 11 | +4.48 | +36.73..+36.73 | -32.25 (-32.25..-32.25) | n/a |
| PLACE_ROUTE | parked | tube | hand | 11 | -5.70 | +42.76..+42.76 | -48.46 (-48.46..-48.46) | n/a |
| PLACE_ROUTE | parked | tube | upper_arm | 11 | +19.04 | +36.18..+36.18 | -17.14 (-17.14..-17.14) | n/a |
| PLACE_ROUTE | setup | shells | finger | 5 | +8.63 | +8.40..+8.49 | +0.19 (+0.14..+0.24) | +9.17 |
| PLACE_ROUTE | setup | shells | forearm | 5 | +4.48 | +3.74..+3.98 | +0.62 (+0.50..+0.73) | +4.58 |
| PLACE_ROUTE | setup | shells | thumb | 5 | +5.40 | +4.87..+5.09 | +0.39 (+0.31..+0.53) | +5.14 |
| PLACE_ROUTE | setup | shells | thumb_pad | 5 | +5.53 | +5.05..+5.25 | +0.35 (+0.28..+0.48) | +5.30 |
| PLACE_ROUTE | setup | shells | upper_arm | 5 | +19.04 | +19.03..+19.05 | +0.00 (-0.00..+0.01) | +19.04 |
| PLACE_ROUTE | setup | shells | wrist_ball | 5 | +5.05 | +4.07..+4.40 | +0.79 (+0.65..+0.98) | +5.16 |
| PLACE_ROUTE | setup | tube | forearm | 5 | +4.48 | +3.74..+3.98 | +0.62 (+0.50..+0.73) | +4.58 |
| PLACE_ROUTE | setup | tube | hand | 5 | -5.70 | -5.88..-5.83 | +0.15 (+0.13..+0.19) | -5.90 |
| PLACE_ROUTE | setup | tube | upper_arm | 5 | +19.04 | +19.03..+19.05 | +0.00 (-0.00..+0.01) | +19.04 |
| RAISE_TO_SIDE | parked | shells | finger | 6 | +8.63 | +50.95..+50.95 | -42.32 (-42.32..-42.32) | n/a |
| RAISE_TO_SIDE | parked | shells | forearm | 6 | +4.48 | +36.73..+36.73 | -32.25 (-32.25..-32.25) | n/a |
| RAISE_TO_SIDE | parked | shells | thumb | 6 | +5.40 | +45.23..+45.23 | -39.84 (-39.84..-39.84) | n/a |
| RAISE_TO_SIDE | parked | shells | thumb_pad | 6 | +5.53 | +52.34..+52.34 | -46.80 (-46.80..-46.80) | n/a |
| RAISE_TO_SIDE | parked | shells | upper_arm | 6 | +19.04 | +36.18..+36.18 | -17.14 (-17.14..-17.14) | n/a |
| RAISE_TO_SIDE | parked | shells | wrist_ball | 6 | +5.05 | +45.20..+45.20 | -40.15 (-40.15..-40.15) | n/a |
| RAISE_TO_SIDE | parked | tube | forearm | 6 | +4.48 | +36.73..+36.73 | -32.25 (-32.25..-32.25) | n/a |
| RAISE_TO_SIDE | parked | tube | hand | 6 | -5.70 | +42.76..+42.76 | -48.46 (-48.46..-48.46) | n/a |
| RAISE_TO_SIDE | parked | tube | upper_arm | 6 | +19.04 | +36.18..+36.18 | -17.14 (-17.14..-17.14) | n/a |
| RAISE_TO_SIDE | setup | shells | finger | 6 | +8.63 | +8.37..+8.48 | +0.20 (+0.15..+0.26) | +40.68 |
| RAISE_TO_SIDE | setup | shells | forearm | 6 | +4.48 | +3.79..+3.96 | +0.58 (+0.52..+0.69) | +19.88 |
| RAISE_TO_SIDE | setup | shells | thumb | 6 | +5.40 | +4.98..+5.08 | +0.38 (+0.31..+0.42) | +33.19 |
| RAISE_TO_SIDE | setup | shells | thumb_pad | 6 | +5.53 | +5.14..+5.24 | +0.35 (+0.29..+0.39) | +40.44 |
| RAISE_TO_SIDE | setup | shells | upper_arm | 6 | +19.04 | +16.96..+17.11 | +2.03 (+1.94..+2.09) | +19.38 |
| RAISE_TO_SIDE | setup | shells | wrist_ball | 6 | +5.05 | +4.14..+4.37 | +0.78 (+0.68..+0.90) | +31.92 |
| RAISE_TO_SIDE | setup | tube | forearm | 6 | +4.48 | +3.79..+3.96 | +0.58 (+0.52..+0.69) | +19.88 |
| RAISE_TO_SIDE | setup | tube | hand | 6 | -5.70 | -5.94..-5.80 | +0.18 (+0.10..+0.24) | +24.86 |
| RAISE_TO_SIDE | setup | tube | upper_arm | 6 | +19.04 | +16.96..+17.11 | +2.03 (+1.94..+2.09) | +19.38 |

## Object `soda_can` -- all flown cycles

| route | role | hand | link | n | planned | realised min..max | delta mean (min..max) | parked-tail min |
|---|---|---|---|---|---|---|---|---|
| LIFT_TO_PRESENT | flight | shells | finger | 5 | +44.86 | +44.59..+44.82 | +0.17 (+0.04..+0.27) | +57.04 |
| LIFT_TO_PRESENT | flight | shells | forearm | 5 | +33.76 | +33.62..+33.76 | +0.08 (+0.00..+0.14) | +39.80 |
| LIFT_TO_PRESENT | flight | shells | thumb | 5 | +40.76 | +40.49..+40.71 | +0.17 (+0.05..+0.27) | +50.13 |
| LIFT_TO_PRESENT | flight | shells | thumb_pad | 5 | +40.93 | +40.65..+40.89 | +0.17 (+0.04..+0.28) | +55.36 |
| LIFT_TO_PRESENT | flight | shells | upper_arm | 5 | +33.61 | +33.52..+33.63 | +0.05 (-0.01..+0.10) | +37.40 |
| LIFT_TO_PRESENT | flight | shells | wrist_ball | 5 | +38.50 | +38.24..+38.45 | +0.17 (+0.05..+0.26) | +48.70 |
| LIFT_TO_PRESENT | flight | tube | forearm | 5 | +33.76 | +33.62..+33.76 | +0.08 (+0.00..+0.14) | +39.80 |
| LIFT_TO_PRESENT | flight | tube | hand | 5 | +29.66 | +29.45..+29.63 | +0.14 (+0.03..+0.21) | +41.63 |
| LIFT_TO_PRESENT | flight | tube | upper_arm | 5 | +33.61 | +33.52..+33.63 | +0.05 (-0.01..+0.10) | +37.40 |
| LOWER_TO_REST | flight | shells | finger | 6 | +44.86 | +43.34..+43.48 | +1.44 (+1.38..+1.52) | +44.63 |
| LOWER_TO_REST | flight | shells | forearm | 6 | +33.76 | +33.56..+33.72 | +0.08 (+0.04..+0.20) | +33.62 |
| LOWER_TO_REST | flight | shells | thumb | 6 | +40.76 | +40.41..+40.66 | +0.16 (+0.10..+0.34) | +40.53 |
| LOWER_TO_REST | flight | shells | thumb_pad | 6 | +40.93 | +40.57..+40.83 | +0.17 (+0.10..+0.36) | +40.70 |
| LOWER_TO_REST | flight | shells | upper_arm | 6 | +33.61 | +33.47..+33.59 | +0.06 (+0.02..+0.15) | +33.50 |
| LOWER_TO_REST | flight | shells | wrist_ball | 6 | +38.50 | +38.17..+38.41 | +0.15 (+0.10..+0.33) | +38.28 |
| LOWER_TO_REST | flight | tube | forearm | 6 | +33.76 | +33.56..+33.72 | +0.08 (+0.04..+0.20) | +33.62 |
| LOWER_TO_REST | flight | tube | hand | 6 | +29.66 | +29.44..+29.65 | +0.09 (+0.02..+0.23) | +29.44 |
| LOWER_TO_REST | flight | tube | upper_arm | 6 | +33.61 | +33.47..+33.59 | +0.06 (+0.02..+0.15) | +33.50 |
| PLACE_ROUTE | flight | shells | finger | 6 | +43.28 | +31.45..+31.54 | +11.79 (+11.73..+11.82) | +44.53 |
| PLACE_ROUTE | flight | shells | forearm | 6 | +33.55 | +33.31..+33.42 | +0.19 (+0.13..+0.24) | +33.58 |
| PLACE_ROUTE | flight | shells | thumb | 6 | +38.79 | +32.32..+32.41 | +6.45 (+6.39..+6.47) | +40.43 |
| PLACE_ROUTE | flight | shells | thumb_pad | 6 | +40.93 | +30.34..+30.39 | +10.58 (+10.54..+10.60) | +40.59 |
| PLACE_ROUTE | flight | shells | upper_arm | 6 | +33.61 | +33.39..+33.50 | +0.16 (+0.11..+0.23) | +33.48 |
| PLACE_ROUTE | flight | shells | wrist_ball | 6 | +37.57 | +33.92..+34.00 | +3.61 (+3.58..+3.65) | +38.18 |
| PLACE_ROUTE | flight | tube | forearm | 6 | +33.55 | +33.31..+33.42 | +0.19 (+0.13..+0.24) | +33.58 |
| PLACE_ROUTE | flight | tube | hand | 6 | +29.66 | +26.58..+26.67 | +3.05 (+2.99..+3.09) | +29.39 |
| PLACE_ROUTE | flight | tube | upper_arm | 6 | +33.61 | +33.39..+33.50 | +0.16 (+0.11..+0.23) | +33.48 |
| PLACE_ROUTE | parked | shells | finger | 11 | +43.28 | +54.00..+54.00 | -10.72 (-10.72..-10.72) | n/a |
| PLACE_ROUTE | parked | shells | forearm | 11 | +33.55 | +37.94..+37.94 | -4.39 (-4.39..-4.39) | n/a |
| PLACE_ROUTE | parked | shells | thumb | 11 | +38.79 | +47.33..+47.33 | -8.54 (-8.54..-8.54) | n/a |
| PLACE_ROUTE | parked | shells | thumb_pad | 11 | +40.93 | +52.94..+52.94 | -12.01 (-12.01..-12.01) | n/a |
| PLACE_ROUTE | parked | shells | upper_arm | 11 | +33.61 | +37.39..+37.39 | -3.78 (-3.78..-3.78) | n/a |
| PLACE_ROUTE | parked | shells | wrist_ball | 11 | +37.57 | +46.20..+46.20 | -8.62 (-8.62..-8.62) | n/a |
| PLACE_ROUTE | parked | tube | forearm | 11 | +33.55 | +37.94..+37.94 | -4.39 (-4.39..-4.39) | n/a |
| PLACE_ROUTE | parked | tube | hand | 11 | +29.66 | +43.76..+43.76 | -14.09 (-14.09..-14.09) | n/a |
| PLACE_ROUTE | parked | tube | upper_arm | 11 | +33.61 | +37.39..+37.39 | -3.78 (-3.78..-3.78) | n/a |
| PLACE_ROUTE | setup | shells | finger | 5 | +43.28 | +31.38..+31.50 | +11.83 (+11.78..+11.89) | +44.62 |
| PLACE_ROUTE | setup | shells | forearm | 5 | +33.55 | +33.34..+33.39 | +0.18 (+0.16..+0.21) | +33.63 |
| PLACE_ROUTE | setup | shells | thumb | 5 | +38.79 | +32.24..+32.34 | +6.48 (+6.45..+6.55) | +40.52 |
| PLACE_ROUTE | setup | shells | thumb_pad | 5 | +40.93 | +30.23..+30.43 | +10.60 (+10.50..+10.70) | +40.68 |
| PLACE_ROUTE | setup | shells | upper_arm | 5 | +33.61 | +33.41..+33.50 | +0.14 (+0.11..+0.21) | +33.51 |
| PLACE_ROUTE | setup | shells | wrist_ball | 5 | +37.57 | +33.88..+33.99 | +3.63 (+3.59..+3.69) | +38.27 |
| PLACE_ROUTE | setup | tube | forearm | 5 | +33.55 | +33.34..+33.39 | +0.18 (+0.16..+0.21) | +33.63 |
| PLACE_ROUTE | setup | tube | hand | 5 | +29.66 | +26.50..+26.62 | +3.09 (+3.04..+3.16) | +29.47 |
| PLACE_ROUTE | setup | tube | upper_arm | 5 | +33.61 | +33.41..+33.50 | +0.14 (+0.11..+0.21) | +33.51 |
| RAISE_TO_SIDE | parked | shells | finger | 6 | +43.28 | +54.00..+54.00 | -10.72 (-10.72..-10.72) | n/a |
| RAISE_TO_SIDE | parked | shells | forearm | 6 | +33.55 | +37.94..+37.94 | -4.39 (-4.39..-4.39) | n/a |
| RAISE_TO_SIDE | parked | shells | thumb | 6 | +38.79 | +47.33..+47.33 | -8.54 (-8.54..-8.54) | n/a |
| RAISE_TO_SIDE | parked | shells | thumb_pad | 6 | +40.93 | +52.94..+52.94 | -12.01 (-12.01..-12.01) | n/a |
| RAISE_TO_SIDE | parked | shells | upper_arm | 6 | +33.61 | +37.39..+37.39 | -3.78 (-3.78..-3.78) | n/a |
| RAISE_TO_SIDE | parked | shells | wrist_ball | 6 | +37.57 | +46.20..+46.20 | -8.62 (-8.62..-8.62) | n/a |
| RAISE_TO_SIDE | parked | tube | forearm | 6 | +33.55 | +37.94..+37.94 | -4.39 (-4.39..-4.39) | n/a |
| RAISE_TO_SIDE | parked | tube | hand | 6 | +29.66 | +43.76..+43.76 | -14.09 (-14.09..-14.09) | n/a |
| RAISE_TO_SIDE | parked | tube | upper_arm | 6 | +33.61 | +37.39..+37.39 | -3.78 (-3.78..-3.78) | n/a |
| RAISE_TO_SIDE | setup | shells | finger | 6 | +43.28 | +31.41..+31.55 | +11.78 (+11.73..+11.87) | +57.41 |
| RAISE_TO_SIDE | setup | shells | forearm | 6 | +33.55 | +33.36..+33.54 | +0.13 (+0.01..+0.19) | +39.94 |
| RAISE_TO_SIDE | setup | shells | thumb | 6 | +38.79 | +32.27..+32.37 | +6.46 (+6.42..+6.53) | +50.46 |
| RAISE_TO_SIDE | setup | shells | thumb_pad | 6 | +40.93 | +30.28..+30.43 | +10.55 (+10.50..+10.66) | +55.77 |
| RAISE_TO_SIDE | setup | shells | upper_arm | 6 | +33.61 | +33.45..+33.51 | +0.14 (+0.10..+0.16) | +37.48 |
| RAISE_TO_SIDE | setup | shells | wrist_ball | 6 | +37.57 | +33.91..+33.99 | +3.61 (+3.58..+3.66) | +49.05 |
| RAISE_TO_SIDE | setup | tube | forearm | 6 | +33.55 | +33.36..+33.54 | +0.13 (+0.01..+0.19) | +39.94 |
| RAISE_TO_SIDE | setup | tube | hand | 6 | +29.66 | +26.53..+26.67 | +3.04 (+2.99..+3.13) | +41.98 |
| RAISE_TO_SIDE | setup | tube | upper_arm | 6 | +33.61 | +33.45..+33.51 | +0.14 (+0.10..+0.16) | +37.48 |

## Per recording: closest shells link to `foam_block` (moving segment, cm)

| recording | role | closest link | planned | realised | delta |
|---|---|---|---|---|---|
| S2-B1-a-r1-flight | flight | forearm | +4.79 | +4.72 | +0.06 |
| S2-B1-a-r1-setup | setup | forearm | +4.48 | +3.79 | +0.69 |
| S2-B1-b-r1-flight | flight | forearm | +4.48 | +3.89 | +0.58 |
| S2-B1-c-r1-flight | flight | forearm | +4.79 | +4.75 | +0.04 |
| S2-B1-c-r1-setup | setup | forearm | +4.48 | +3.86 | +0.62 |
| S2-B1-a-r2-flight | flight | forearm | +4.79 | +4.48 | +0.31 |
| S2-B1-a-r2-setup | setup | forearm | +4.48 | +3.86 | +0.62 |
| S2-B1-b-r2-flight | flight | forearm | +4.48 | +3.78 | +0.70 |
| S2-B1-c-r2-flight | flight | forearm | +4.79 | +4.61 | +0.17 |
| S2-B1-c-r2-setup | setup | forearm | +4.48 | +3.90 | +0.58 |
| S2-B1-a-r3-flight | flight | forearm | +4.79 | +4.66 | +0.12 |
| S2-B1-a-r3-setup | setup | forearm | +4.48 | +3.96 | +0.52 |
| S2-B1-b-r3-flight | flight | forearm | +4.48 | +3.85 | +0.63 |
| S2-B1-c-r3-flight | flight | forearm | +4.79 | +4.57 | +0.21 |
| S2-B1-c-r3-setup | setup | forearm | +4.48 | +3.74 | +0.73 |
| S2-B1-a-r4-flight | flight | forearm | +4.79 | +4.72 | +0.07 |
| S2-B1-a-r4-setup | setup | forearm | +4.48 | +3.95 | +0.53 |
| S2-B1-b-r4-flight | flight | forearm | +4.48 | +3.86 | +0.62 |
| S2-B1-c-r4-flight | flight | forearm | +4.79 | +4.59 | +0.20 |
| S2-B1-c-r4-setup | setup | forearm | +4.48 | +3.83 | +0.65 |
| S2-B1-a-r5-flight | flight | forearm | +4.79 | +4.68 | +0.11 |
| S2-B1-a-r5-setup | setup | forearm | +4.48 | +3.87 | +0.61 |
| S2-B1-b-r5-flight | flight | forearm | +4.48 | +3.84 | +0.64 |
| S2-B1-c-r5-flight | flight | forearm | +4.79 | +4.78 | +0.01 |
| S2-B1-c-r5-setup | setup | forearm | +4.48 | +3.98 | +0.50 |
| S2-B1-a-r6-flight | flight | forearm | +4.79 | +4.67 | +0.11 |
| S2-B1-a-r6-setup | setup | forearm | +4.48 | +3.95 | +0.53 |
| S2-B1-b-r6-flight | flight | forearm | +4.48 | +3.84 | +0.64 |
