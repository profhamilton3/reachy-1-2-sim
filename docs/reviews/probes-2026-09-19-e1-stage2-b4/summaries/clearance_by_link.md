# Planned minus realised clearance to `pool_box_1`, by route x role x hand x link, across repetitions (cm)

Pure geometry from the recorded joint samples at each sample's own measured aperture, against the B4 YAML poses (same method as Stage 1's `clearance_by_link.py`). `realised` for setups/flights is the **moving** segment (joint velocity > 2 deg/s); parked recordings use the whole 3 s recording. `delta` = planned - realised: positive means the flight came closer than the planned corridor. n = recordings in the group (6 per route/role for setups and flights; 6 or 12 for parked). Zero physics contacts on every recording; negative clearance is the conservative capsule model overlapping, not a contact. Deterministic simulator: the six repetitions of one route are near-identical re-runs of one open-loop trajectory from one reset state, so agreement across them is evidence of repeatability of this procedure on this board, NOT independent evidence of broad reliability.

## All 18 cycles

| route | role | hand | link | n | planned | realised min..max | delta mean (min..max) | parked-tail min |
|---|---|---|---|---|---|---|---|---|
| LIFT_TO_PRESENT | flight | shells | finger | 6 | +11.40 | +11.07..+11.35 | +0.14 (+0.04..+0.32) | +42.75 |
| LIFT_TO_PRESENT | flight | shells | forearm | 6 | +6.46 | +6.20..+6.43 | +0.10 (+0.03..+0.26) | +22.31 |
| LIFT_TO_PRESENT | flight | shells | thumb | 6 | +7.38 | +7.05..+7.33 | +0.14 (+0.04..+0.33) | +35.34 |
| LIFT_TO_PRESENT | flight | shells | thumb_pad | 6 | +7.65 | +7.33..+7.59 | +0.14 (+0.05..+0.31) | +42.44 |
| LIFT_TO_PRESENT | flight | shells | upper_arm | 6 | +19.26 | +19.25..+19.48 | -0.09 (-0.22..+0.01) | +21.81 |
| LIFT_TO_PRESENT | flight | shells | wrist_ball | 6 | +6.89 | +6.60..+6.85 | +0.11 (+0.04..+0.29) | +34.05 |
| LIFT_TO_PRESENT | flight | tube | forearm | 6 | +6.46 | +6.20..+6.43 | +0.10 (+0.03..+0.26) | +22.31 |
| LIFT_TO_PRESENT | flight | tube | hand | 6 | -3.72 | -4.00..-3.74 | +0.10 (+0.02..+0.28) | +27.00 |
| LIFT_TO_PRESENT | flight | tube | upper_arm | 6 | +19.26 | +19.25..+19.48 | -0.09 (-0.22..+0.01) | +21.81 |
| LOWER_TO_REST | flight | shells | finger | 6 | +11.40 | +10.82..+10.91 | +0.54 (+0.49..+0.58) | +11.29 |
| LOWER_TO_REST | flight | shells | forearm | 6 | +6.46 | +6.39..+6.45 | +0.04 (+0.01..+0.07) | +6.39 |
| LOWER_TO_REST | flight | shells | thumb | 6 | +7.38 | +7.27..+7.33 | +0.08 (+0.05..+0.11) | +7.26 |
| LOWER_TO_REST | flight | shells | thumb_pad | 6 | +7.65 | +7.52..+7.58 | +0.10 (+0.06..+0.13) | +7.52 |
| LOWER_TO_REST | flight | shells | upper_arm | 6 | +19.26 | +19.51..+19.70 | -0.32 (-0.44..-0.25) | +20.90 |
| LOWER_TO_REST | flight | shells | wrist_ball | 6 | +6.89 | +6.81..+6.87 | +0.05 (+0.02..+0.09) | +6.81 |
| LOWER_TO_REST | flight | tube | forearm | 6 | +6.46 | +6.39..+6.45 | +0.04 (+0.01..+0.07) | +6.39 |
| LOWER_TO_REST | flight | tube | hand | 6 | -3.72 | -3.74..-3.69 | +0.00 (-0.03..+0.02) | -3.78 |
| LOWER_TO_REST | flight | tube | upper_arm | 6 | +19.26 | +19.51..+19.70 | -0.32 (-0.44..-0.25) | +20.90 |
| PLACE_ROUTE | flight | shells | finger | 6 | +10.75 | +10.48..+10.64 | +0.20 (+0.12..+0.28) | +11.18 |
| PLACE_ROUTE | flight | shells | forearm | 6 | +6.26 | +5.36..+6.05 | +0.66 (+0.21..+0.90) | +6.27 |
| PLACE_ROUTE | flight | shells | thumb | 6 | +7.38 | +6.53..+6.99 | +0.60 (+0.38..+0.84) | +7.15 |
| PLACE_ROUTE | flight | shells | thumb_pad | 6 | +7.65 | +7.29..+7.47 | +0.26 (+0.18..+0.36) | +7.44 |
| PLACE_ROUTE | flight | shells | upper_arm | 6 | +20.89 | +20.87..+20.90 | +0.00 (-0.01..+0.01) | +20.89 |
| PLACE_ROUTE | flight | shells | wrist_ball | 6 | +6.67 | +5.57..+6.25 | +0.84 (+0.42..+1.10) | +6.68 |
| PLACE_ROUTE | flight | tube | forearm | 6 | +6.26 | +5.36..+6.05 | +0.66 (+0.21..+0.90) | +6.27 |
| PLACE_ROUTE | flight | tube | hand | 6 | -3.72 | -3.87..-3.82 | +0.13 (+0.10..+0.16) | -3.93 |
| PLACE_ROUTE | flight | tube | upper_arm | 6 | +20.89 | +20.87..+20.90 | +0.00 (-0.01..+0.01) | +20.89 |
| PLACE_ROUTE | parked | shells | finger | 12 | +10.75 | +52.23..+52.23 | -41.47 (-41.47..-41.47) | n/a |
| PLACE_ROUTE | parked | shells | forearm | 12 | +6.26 | +38.27..+38.27 | -32.01 (-32.01..-32.01) | n/a |
| PLACE_ROUTE | parked | shells | thumb | 12 | +7.38 | +46.54..+46.54 | -39.16 (-39.16..-39.16) | n/a |
| PLACE_ROUTE | parked | shells | thumb_pad | 12 | +7.65 | +53.46..+53.46 | -45.82 (-45.82..-45.82) | n/a |
| PLACE_ROUTE | parked | shells | upper_arm | 12 | +20.89 | +37.72..+37.72 | -16.83 (-16.83..-16.83) | n/a |
| PLACE_ROUTE | parked | shells | wrist_ball | 12 | +6.67 | +46.47..+46.47 | -39.80 (-39.80..-39.80) | n/a |
| PLACE_ROUTE | parked | tube | forearm | 12 | +6.26 | +38.27..+38.27 | -32.01 (-32.01..-32.01) | n/a |
| PLACE_ROUTE | parked | tube | hand | 12 | -3.72 | +44.04..+44.04 | -47.75 (-47.75..-47.75) | n/a |
| PLACE_ROUTE | parked | tube | upper_arm | 12 | +20.89 | +37.72..+37.72 | -16.83 (-16.83..-16.83) | n/a |
| PLACE_ROUTE | setup | shells | finger | 6 | +10.75 | +10.43..+10.59 | +0.21 (+0.16..+0.32) | +11.10 |
| PLACE_ROUTE | setup | shells | forearm | 6 | +6.26 | +5.50..+5.73 | +0.65 (+0.52..+0.76) | +6.21 |
| PLACE_ROUTE | setup | shells | thumb | 6 | +7.38 | +6.71..+6.96 | +0.51 (+0.42..+0.67) | +7.08 |
| PLACE_ROUTE | setup | shells | thumb_pad | 6 | +7.65 | +7.23..+7.39 | +0.32 (+0.26..+0.41) | +7.36 |
| PLACE_ROUTE | setup | shells | upper_arm | 6 | +20.89 | +20.87..+20.90 | +0.00 (-0.01..+0.02) | +20.87 |
| PLACE_ROUTE | setup | shells | wrist_ball | 6 | +6.67 | +5.72..+6.01 | +0.81 (+0.66..+0.95) | +6.62 |
| PLACE_ROUTE | setup | tube | forearm | 6 | +6.26 | +5.50..+5.73 | +0.65 (+0.52..+0.76) | +6.21 |
| PLACE_ROUTE | setup | tube | hand | 6 | -3.72 | -3.94..-3.76 | +0.13 (+0.05..+0.22) | -3.98 |
| PLACE_ROUTE | setup | tube | upper_arm | 6 | +20.89 | +20.87..+20.90 | +0.00 (-0.01..+0.02) | +20.87 |
| RAISE_TO_SIDE | parked | shells | finger | 6 | +10.75 | +52.23..+52.23 | -41.47 (-41.47..-41.47) | n/a |
| RAISE_TO_SIDE | parked | shells | forearm | 6 | +6.26 | +38.27..+38.27 | -32.01 (-32.01..-32.01) | n/a |
| RAISE_TO_SIDE | parked | shells | thumb | 6 | +7.38 | +46.54..+46.54 | -39.16 (-39.16..-39.16) | n/a |
| RAISE_TO_SIDE | parked | shells | thumb_pad | 6 | +7.65 | +53.46..+53.46 | -45.82 (-45.82..-45.82) | n/a |
| RAISE_TO_SIDE | parked | shells | upper_arm | 6 | +20.89 | +37.72..+37.72 | -16.83 (-16.83..-16.83) | n/a |
| RAISE_TO_SIDE | parked | shells | wrist_ball | 6 | +6.67 | +46.47..+46.47 | -39.80 (-39.80..-39.80) | n/a |
| RAISE_TO_SIDE | parked | tube | forearm | 6 | +6.26 | +38.27..+38.27 | -32.01 (-32.01..-32.01) | n/a |
| RAISE_TO_SIDE | parked | tube | hand | 6 | -3.72 | +44.04..+44.04 | -47.75 (-47.75..-47.75) | n/a |
| RAISE_TO_SIDE | parked | tube | upper_arm | 6 | +20.89 | +37.72..+37.72 | -16.83 (-16.83..-16.83) | n/a |
| RAISE_TO_SIDE | setup | shells | finger | 6 | +10.75 | +10.43..+10.61 | +0.22 (+0.14..+0.32) | +41.72 |
| RAISE_TO_SIDE | setup | shells | forearm | 6 | +6.26 | +5.45..+5.69 | +0.68 (+0.57..+0.81) | +22.07 |
| RAISE_TO_SIDE | setup | shells | thumb | 6 | +7.38 | +6.69..+7.02 | +0.54 (+0.35..+0.69) | +34.35 |
| RAISE_TO_SIDE | setup | shells | thumb_pad | 6 | +7.65 | +7.27..+7.39 | +0.31 (+0.26..+0.37) | +41.32 |
| RAISE_TO_SIDE | setup | shells | upper_arm | 6 | +20.89 | +19.17..+19.52 | +1.56 (+1.37..+1.72) | +21.57 |
| RAISE_TO_SIDE | setup | shells | wrist_ball | 6 | +6.67 | +5.66..+5.95 | +0.86 (+0.72..+1.01) | +33.08 |
| RAISE_TO_SIDE | setup | tube | forearm | 6 | +6.26 | +5.45..+5.69 | +0.68 (+0.57..+0.81) | +22.07 |
| RAISE_TO_SIDE | setup | tube | hand | 6 | -3.72 | -3.91..-3.82 | +0.14 (+0.10..+0.19) | +26.01 |
| RAISE_TO_SIDE | setup | tube | upper_arm | 6 | +20.89 | +19.17..+19.52 | +1.56 (+1.37..+1.72) | +21.57 |

## Excluding the sleep-affected cycle S2-B4-c-r1

| route | role | hand | link | n | planned | realised min..max | delta mean (min..max) | parked-tail min |
|---|---|---|---|---|---|---|---|---|
| LIFT_TO_PRESENT | flight | shells | finger | 5 | +11.40 | +11.12..+11.35 | +0.10 (+0.04..+0.28) | +42.75 |
| LIFT_TO_PRESENT | flight | shells | forearm | 5 | +6.46 | +6.26..+6.43 | +0.07 (+0.03..+0.20) | +22.31 |
| LIFT_TO_PRESENT | flight | shells | thumb | 5 | +7.38 | +7.10..+7.33 | +0.10 (+0.04..+0.27) | +35.34 |
| LIFT_TO_PRESENT | flight | shells | thumb_pad | 5 | +7.65 | +7.37..+7.59 | +0.10 (+0.05..+0.28) | +42.44 |
| LIFT_TO_PRESENT | flight | shells | upper_arm | 5 | +19.26 | +19.25..+19.36 | -0.06 (-0.11..+0.01) | +21.81 |
| LIFT_TO_PRESENT | flight | shells | wrist_ball | 5 | +6.89 | +6.66..+6.85 | +0.08 (+0.04..+0.23) | +34.05 |
| LIFT_TO_PRESENT | flight | tube | forearm | 5 | +6.46 | +6.26..+6.43 | +0.07 (+0.03..+0.20) | +22.31 |
| LIFT_TO_PRESENT | flight | tube | hand | 5 | -3.72 | -4.00..-3.74 | +0.09 (+0.02..+0.28) | +27.00 |
| LIFT_TO_PRESENT | flight | tube | upper_arm | 5 | +19.26 | +19.25..+19.36 | -0.06 (-0.11..+0.01) | +21.81 |
| LOWER_TO_REST | flight | shells | finger | 6 | +11.40 | +10.82..+10.91 | +0.54 (+0.49..+0.58) | +11.29 |
| LOWER_TO_REST | flight | shells | forearm | 6 | +6.46 | +6.39..+6.45 | +0.04 (+0.01..+0.07) | +6.39 |
| LOWER_TO_REST | flight | shells | thumb | 6 | +7.38 | +7.27..+7.33 | +0.08 (+0.05..+0.11) | +7.26 |
| LOWER_TO_REST | flight | shells | thumb_pad | 6 | +7.65 | +7.52..+7.58 | +0.10 (+0.06..+0.13) | +7.52 |
| LOWER_TO_REST | flight | shells | upper_arm | 6 | +19.26 | +19.51..+19.70 | -0.32 (-0.44..-0.25) | +20.90 |
| LOWER_TO_REST | flight | shells | wrist_ball | 6 | +6.89 | +6.81..+6.87 | +0.05 (+0.02..+0.09) | +6.81 |
| LOWER_TO_REST | flight | tube | forearm | 6 | +6.46 | +6.39..+6.45 | +0.04 (+0.01..+0.07) | +6.39 |
| LOWER_TO_REST | flight | tube | hand | 6 | -3.72 | -3.74..-3.69 | +0.00 (-0.03..+0.02) | -3.78 |
| LOWER_TO_REST | flight | tube | upper_arm | 6 | +19.26 | +19.51..+19.70 | -0.32 (-0.44..-0.25) | +20.90 |
| PLACE_ROUTE | flight | shells | finger | 6 | +10.75 | +10.48..+10.64 | +0.20 (+0.12..+0.28) | +11.18 |
| PLACE_ROUTE | flight | shells | forearm | 6 | +6.26 | +5.36..+6.05 | +0.66 (+0.21..+0.90) | +6.27 |
| PLACE_ROUTE | flight | shells | thumb | 6 | +7.38 | +6.53..+6.99 | +0.60 (+0.38..+0.84) | +7.15 |
| PLACE_ROUTE | flight | shells | thumb_pad | 6 | +7.65 | +7.29..+7.47 | +0.26 (+0.18..+0.36) | +7.44 |
| PLACE_ROUTE | flight | shells | upper_arm | 6 | +20.89 | +20.87..+20.90 | +0.00 (-0.01..+0.01) | +20.89 |
| PLACE_ROUTE | flight | shells | wrist_ball | 6 | +6.67 | +5.57..+6.25 | +0.84 (+0.42..+1.10) | +6.68 |
| PLACE_ROUTE | flight | tube | forearm | 6 | +6.26 | +5.36..+6.05 | +0.66 (+0.21..+0.90) | +6.27 |
| PLACE_ROUTE | flight | tube | hand | 6 | -3.72 | -3.87..-3.82 | +0.13 (+0.10..+0.16) | -3.93 |
| PLACE_ROUTE | flight | tube | upper_arm | 6 | +20.89 | +20.87..+20.90 | +0.00 (-0.01..+0.01) | +20.89 |
| PLACE_ROUTE | parked | shells | finger | 11 | +10.75 | +52.23..+52.23 | -41.47 (-41.47..-41.47) | n/a |
| PLACE_ROUTE | parked | shells | forearm | 11 | +6.26 | +38.27..+38.27 | -32.01 (-32.01..-32.01) | n/a |
| PLACE_ROUTE | parked | shells | thumb | 11 | +7.38 | +46.54..+46.54 | -39.16 (-39.16..-39.16) | n/a |
| PLACE_ROUTE | parked | shells | thumb_pad | 11 | +7.65 | +53.46..+53.46 | -45.82 (-45.82..-45.82) | n/a |
| PLACE_ROUTE | parked | shells | upper_arm | 11 | +20.89 | +37.72..+37.72 | -16.83 (-16.83..-16.83) | n/a |
| PLACE_ROUTE | parked | shells | wrist_ball | 11 | +6.67 | +46.47..+46.47 | -39.80 (-39.80..-39.80) | n/a |
| PLACE_ROUTE | parked | tube | forearm | 11 | +6.26 | +38.27..+38.27 | -32.01 (-32.01..-32.01) | n/a |
| PLACE_ROUTE | parked | tube | hand | 11 | -3.72 | +44.04..+44.04 | -47.75 (-47.75..-47.75) | n/a |
| PLACE_ROUTE | parked | tube | upper_arm | 11 | +20.89 | +37.72..+37.72 | -16.83 (-16.83..-16.83) | n/a |
| PLACE_ROUTE | setup | shells | finger | 5 | +10.75 | +10.54..+10.59 | +0.19 (+0.16..+0.22) | +11.13 |
| PLACE_ROUTE | setup | shells | forearm | 5 | +6.26 | +5.50..+5.73 | +0.68 (+0.52..+0.76) | +6.26 |
| PLACE_ROUTE | setup | shells | thumb | 5 | +7.38 | +6.71..+6.96 | +0.53 (+0.42..+0.67) | +7.11 |
| PLACE_ROUTE | setup | shells | thumb_pad | 5 | +7.65 | +7.31..+7.39 | +0.30 (+0.26..+0.34) | +7.39 |
| PLACE_ROUTE | setup | shells | upper_arm | 5 | +20.89 | +20.87..+20.89 | +0.00 (-0.01..+0.02) | +20.87 |
| PLACE_ROUTE | setup | shells | wrist_ball | 5 | +6.67 | +5.72..+6.01 | +0.84 (+0.66..+0.95) | +6.67 |
| PLACE_ROUTE | setup | tube | forearm | 5 | +6.26 | +5.50..+5.73 | +0.68 (+0.52..+0.76) | +6.26 |
| PLACE_ROUTE | setup | tube | hand | 5 | -3.72 | -3.94..-3.76 | +0.14 (+0.05..+0.22) | -3.98 |
| PLACE_ROUTE | setup | tube | upper_arm | 5 | +20.89 | +20.87..+20.89 | +0.00 (-0.01..+0.02) | +20.87 |
| RAISE_TO_SIDE | parked | shells | finger | 6 | +10.75 | +52.23..+52.23 | -41.47 (-41.47..-41.47) | n/a |
| RAISE_TO_SIDE | parked | shells | forearm | 6 | +6.26 | +38.27..+38.27 | -32.01 (-32.01..-32.01) | n/a |
| RAISE_TO_SIDE | parked | shells | thumb | 6 | +7.38 | +46.54..+46.54 | -39.16 (-39.16..-39.16) | n/a |
| RAISE_TO_SIDE | parked | shells | thumb_pad | 6 | +7.65 | +53.46..+53.46 | -45.82 (-45.82..-45.82) | n/a |
| RAISE_TO_SIDE | parked | shells | upper_arm | 6 | +20.89 | +37.72..+37.72 | -16.83 (-16.83..-16.83) | n/a |
| RAISE_TO_SIDE | parked | shells | wrist_ball | 6 | +6.67 | +46.47..+46.47 | -39.80 (-39.80..-39.80) | n/a |
| RAISE_TO_SIDE | parked | tube | forearm | 6 | +6.26 | +38.27..+38.27 | -32.01 (-32.01..-32.01) | n/a |
| RAISE_TO_SIDE | parked | tube | hand | 6 | -3.72 | +44.04..+44.04 | -47.75 (-47.75..-47.75) | n/a |
| RAISE_TO_SIDE | parked | tube | upper_arm | 6 | +20.89 | +37.72..+37.72 | -16.83 (-16.83..-16.83) | n/a |
| RAISE_TO_SIDE | setup | shells | finger | 6 | +10.75 | +10.43..+10.61 | +0.22 (+0.14..+0.32) | +41.72 |
| RAISE_TO_SIDE | setup | shells | forearm | 6 | +6.26 | +5.45..+5.69 | +0.68 (+0.57..+0.81) | +22.07 |
| RAISE_TO_SIDE | setup | shells | thumb | 6 | +7.38 | +6.69..+7.02 | +0.54 (+0.35..+0.69) | +34.35 |
| RAISE_TO_SIDE | setup | shells | thumb_pad | 6 | +7.65 | +7.27..+7.39 | +0.31 (+0.26..+0.37) | +41.32 |
| RAISE_TO_SIDE | setup | shells | upper_arm | 6 | +20.89 | +19.17..+19.52 | +1.56 (+1.37..+1.72) | +21.57 |
| RAISE_TO_SIDE | setup | shells | wrist_ball | 6 | +6.67 | +5.66..+5.95 | +0.86 (+0.72..+1.01) | +33.08 |
| RAISE_TO_SIDE | setup | tube | forearm | 6 | +6.26 | +5.45..+5.69 | +0.68 (+0.57..+0.81) | +22.07 |
| RAISE_TO_SIDE | setup | tube | hand | 6 | -3.72 | -3.91..-3.82 | +0.14 (+0.10..+0.19) | +26.01 |
| RAISE_TO_SIDE | setup | tube | upper_arm | 6 | +20.89 | +19.17..+19.52 | +1.56 (+1.37..+1.72) | +21.57 |

## Sleep-affected cycle S2-B4-c-r1 vs the other five c-repetitions (shells, closest link to pool_box_1, moving segment, cm)

| recording | role | closest link | planned | realised | delta |
|---|---|---|---|---|---|
| S2-B4-c-r1-flight **(sleep-affected)** | flight | forearm | +6.46 | +6.20 | +0.26 |
| S2-B4-c-r1-setup **(sleep-affected)** | setup | forearm | +6.26 | +5.72 | +0.53 |
| S2-B4-c-r2-flight | flight | forearm | +6.46 | +6.26 | +0.20 |
| S2-B4-c-r2-setup | setup | forearm | +6.26 | +5.73 | +0.52 |
| S2-B4-c-r3-flight | flight | forearm | +6.46 | +6.43 | +0.04 |
| S2-B4-c-r3-setup | setup | forearm | +6.26 | +5.57 | +0.69 |
| S2-B4-c-r4-flight | flight | forearm | +6.46 | +6.43 | +0.03 |
| S2-B4-c-r4-setup | setup | forearm | +6.26 | +5.58 | +0.67 |
| S2-B4-c-r5-flight | flight | forearm | +6.46 | +6.43 | +0.03 |
| S2-B4-c-r5-setup | setup | forearm | +6.26 | +5.51 | +0.75 |
| S2-B4-c-r6-flight | flight | forearm | +6.46 | +6.43 | +0.03 |
| S2-B4-c-r6-setup | setup | forearm | +6.26 | +5.50 | +0.76 |
