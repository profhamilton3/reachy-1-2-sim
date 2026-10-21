# Perception stage 1 grading report (#186)

- Match radius: 40 mm, on the truth visible-surface point; in-view margin 16 px; distance bands [0.45, 0.5, 0.55, 0.6] m.
- Detector model: 60dad10f8e6a29dc09daf1bc11608a1658d70b5b98ffbaa615d88f54e453ff7e; stereo profile: 771392183ddc6c62.

## Overall

Pairs graded: 10. Class accuracy on matched objects: 0.7. Misses: 24. False positives: 4. OOV hits: 0. Matched by 2D only (no 3D): 0.

| class | in view | correct | detections | recall | precision |
|---|---|---|---|---|---|
| cube | 22 | 2 | 2 | 0.091 | 1.0 |
| cylinder | 22 | 12 | 22 | 0.545 | 0.545 |

Off-board truth in view (parked floor pool; not in the headline): 76 sightings, 0 correct, 0 wrong class, 0 OOV, 76 missed.

Confusion (truth → reported): cube: cube 2, cylinder 6, missed 14; cylinder: cylinder 12, missed 10

`empty` detections: 46 (34 correct, 12 false positive, 0 not on a cell). Precision 0.739.

| object | kind | seen | correct | wrong class | missed | recall |
|---|---|---|---|---|---|---|
| blue_cylinder | cylinder | 4 | 0 | 0 | 4 | 0.0 |
| pool_box_1 | cube | 4 | 0 | 2 | 2 | 0.0 |
| pool_box_2 | cube | 4 | 2 | 0 | 2 | 0.5 |
| pool_box_3 | cube | 4 | 0 | 0 | 4 | 0.0 |
| pool_box_4 | cube | 4 | 0 | 0 | 4 | 0.0 |
| pool_box_5 | cube | 2 | 0 | 0 | 2 | 0.0 |
| pool_cyl_1 | cylinder | 4 | 2 | 0 | 2 | 0.5 |
| pool_cyl_2 | cylinder | 4 | 2 | 0 | 2 | 0.5 |
| pool_cyl_3 | cylinder | 4 | 4 | 0 | 0 | 1.0 |
| pool_cyl_4 | cylinder | 4 | 4 | 0 | 0 | 1.0 |
| pool_cyl_5 | cylinder | 2 | 0 | 0 | 2 | 0.0 |
| red_cube | cube | 4 | 0 | 4 | 0 | 0.0 |

Per cell (outcome counts over pairs):

| cell | outcomes |
|---|---|
| r1c1 | correct 6, wrong 4 |
| r1c2 | correct 2, unknown 6, wrong 2 |
| r1c3 | unknown 8, wrong 2 |
| r2c1 | correct 8, unknown 2 |
| r2c2 | correct 8, wrong 2 |
| r2c3 | correct 6, unknown 2, wrong 2 |
| r3c1 | correct 6, unknown 2, wrong 2 |
| r3c2 | correct 6, wrong 4 |
| r3c3 | correct 6, unknown 4 |

3D error in mm (matched objects with a 3D position). `gate` is to the visible-surface point; `centre`, `horizontal` (x, y) and signed `along-ray` (+ = farther than the centre) are to the true centre.

| group | n | gate mean | gate p95 | centre median | horizontal median | horizontal p95 | along-ray median |
|---|---|---|---|---|---|---|---|
| all | 20 | 18.8 | 33.4 | 24.5 | 10.9 | 23.3 | -24.1 |

By cell:

| group | n | gate mean | gate p95 | centre median | horizontal median | horizontal p95 | along-ray median |
|---|---|---|---|---|---|---|---|
| r1c1 | 4 | 12.4 | 13.9 | 31.0 | 8.2 | 10.8 | -30.1 |
| r1c2 | 2 | 16.7 | 16.7 | 24.3 | 8.8 | 8.8 | -22.8 |
| r2c1 | 2 | 14.3 | 14.3 | 24.5 | 8.7 | 8.7 | -24.4 |
| r2c2 | 2 | 17.4 | 17.4 | 23.6 | 7.8 | 7.8 | -23.5 |
| r2c3 | 2 | 12.0 | 12.1 | 30.0 | 11.1 | 11.1 | -29.9 |
| r3c1 | 2 | 18.3 | 18.3 | 24.5 | 18.7 | 18.7 | -23.8 |
| r3c2 | 4 | 25.4 | 30.3 | 22.0 | 18.6 | 23.3 | -19.8 |
| r3c3 | 2 | 33.4 | 33.4 | 11.8 | 11.0 | 11.1 | -9.6 |

By camera distance:

| group | n | gate mean | gate p95 | centre median | horizontal median | horizontal p95 | along-ray median |
|---|---|---|---|---|---|---|---|
| 0.45-0.50 m | 2 | 16.7 | 16.7 | 24.3 | 8.8 | 8.8 | -22.8 |
| 0.50-0.55 m | 10 | 13.7 | 17.4 | 30.0 | 8.7 | 11.1 | -29.1 |
| 0.55-0.60 m | 6 | 28.1 | 33.4 | 15.5 | 13.7 | 23.3 | -13.4 |
| >=0.60 m | 2 | 18.3 | 18.3 | 24.5 | 18.7 | 18.7 | -23.8 |

Size error: not available (the viewer's analysis carries no size estimate).

## Layout A_default

Static bracket: ok (largest move 0.0 mm; scene `FWDCenterLabSivaPool` revision `initial`).

Pairs graded: 2. Class accuracy on matched objects: -. Misses: 0. False positives: 0. OOV hits: 0. Matched by 2D only (no 3D): 0.

| class | in view | correct | detections | recall | precision |
|---|---|---|---|---|---|
| cube | 0 | 0 | 0 | - | - |
| cylinder | 0 | 0 | 0 | - | - |

Off-board truth in view (parked floor pool; not in the headline): 24 sightings, 0 correct, 0 wrong class, 0 OOV, 24 missed.

Confusion (truth → reported): 

`empty` detections: 14 (14 correct, 0 false positive, 0 not on a cell). Precision 1.0.

| object | kind | seen | correct | wrong class | missed | recall |
|---|---|---|---|---|---|---|

Per cell (outcome counts over pairs):

| cell | outcomes |
|---|---|
| r1c1 | correct 2 |
| r1c2 | unknown 2 |
| r1c3 | unknown 2 |
| r2c1 | correct 2 |
| r2c2 | correct 2 |
| r2c3 | correct 2 |
| r3c1 | correct 2 |
| r3c2 | correct 2 |
| r3c3 | correct 2 |

3D error in mm (matched objects with a 3D position). `gate` is to the visible-surface point; `centre`, `horizontal` (x, y) and signed `along-ray` (+ = farther than the centre) are to the true centre.

| group | n | gate mean | gate p95 | centre median | horizontal median | horizontal p95 | along-ray median |
|---|---|---|---|---|---|---|---|
| all | 0 | - | - | - | - | - | - |

By cell:

| group | n | gate mean | gate p95 | centre median | horizontal median | horizontal p95 | along-ray median |
|---|---|---|---|---|---|---|---|

By camera distance:

| group | n | gate mean | gate p95 | centre median | horizontal median | horizontal p95 | along-ray median |
|---|---|---|---|---|---|---|---|

Size error: not available (the viewer's analysis carries no size estimate).

Pair `A_default/pair1.zip`:
- miss: blue_cylinder (cylinder, None); no detection box covers it
- miss: pool_box_1 (cube, None); no detection box covers it
- miss: pool_box_2 (cube, None); no detection box covers it
- miss: pool_box_3 (cube, None); no detection box covers it
- miss: pool_box_4 (cube, None); no detection box covers it
- miss: pool_box_5 (cube, None); no detection box covers it
- miss: pool_cyl_1 (cylinder, None); no detection box covers it
- miss: pool_cyl_2 (cylinder, None); no detection box covers it
- miss: pool_cyl_3 (cylinder, None); no detection box covers it
- miss: pool_cyl_4 (cylinder, None); no detection box covers it
- miss: pool_cyl_5 (cylinder, None); no detection box covers it
- miss: red_cube (cube, None); no detection box covers it

Pair `A_default/pair2.zip`:
- miss: blue_cylinder (cylinder, None); no detection box covers it
- miss: pool_box_1 (cube, None); no detection box covers it
- miss: pool_box_2 (cube, None); no detection box covers it
- miss: pool_box_3 (cube, None); no detection box covers it
- miss: pool_box_4 (cube, None); no detection box covers it
- miss: pool_box_5 (cube, None); no detection box covers it
- miss: pool_cyl_1 (cylinder, None); no detection box covers it
- miss: pool_cyl_2 (cylinder, None); no detection box covers it
- miss: pool_cyl_3 (cylinder, None); no detection box covers it
- miss: pool_cyl_4 (cylinder, None); no detection box covers it
- miss: pool_cyl_5 (cylinder, None); no detection box covers it
- miss: red_cube (cube, None); no detection box covers it

## Layout B_corners

Static bracket: ok (largest move 0.0 mm; scene `FWDCenterLabSivaPool` revision `initial`).

Pairs graded: 2. Class accuracy on matched objects: 0.0. Misses: 2. False positives: 0. OOV hits: 0. Matched by 2D only (no 3D): 0.

| class | in view | correct | detections | recall | precision |
|---|---|---|---|---|---|
| cube | 2 | 0 | 0 | 0.0 | - |
| cylinder | 2 | 0 | 2 | 0.0 | 0.0 |

Off-board truth in view (parked floor pool; not in the headline): 20 sightings, 0 correct, 0 wrong class, 0 OOV, 20 missed.

Confusion (truth → reported): cube: cylinder 2; cylinder: missed 2

`empty` detections: 8 (8 correct, 0 false positive, 0 not on a cell). Precision 1.0.

| object | kind | seen | correct | wrong class | missed | recall |
|---|---|---|---|---|---|---|
| blue_cylinder | cylinder | 2 | 0 | 0 | 2 | 0.0 |
| red_cube | cube | 2 | 0 | 2 | 0 | 0.0 |

Per cell (outcome counts over pairs):

| cell | outcomes |
|---|---|
| r1c1 | wrong 2 |
| r1c2 | unknown 2 |
| r1c3 | unknown 2 |
| r2c1 | correct 2 |
| r2c2 | correct 2 |
| r2c3 | unknown 2 |
| r3c1 | correct 2 |
| r3c2 | correct 2 |
| r3c3 | unknown 2 |

3D error in mm (matched objects with a 3D position). `gate` is to the visible-surface point; `centre`, `horizontal` (x, y) and signed `along-ray` (+ = farther than the centre) are to the true centre.

| group | n | gate mean | gate p95 | centre median | horizontal median | horizontal p95 | along-ray median |
|---|---|---|---|---|---|---|---|
| all | 2 | 13.9 | 13.9 | 30.8 | 5.6 | 5.6 | -29.1 |

By cell:

| group | n | gate mean | gate p95 | centre median | horizontal median | horizontal p95 | along-ray median |
|---|---|---|---|---|---|---|---|
| r1c1 | 2 | 13.9 | 13.9 | 30.8 | 5.6 | 5.6 | -29.1 |

By camera distance:

| group | n | gate mean | gate p95 | centre median | horizontal median | horizontal p95 | along-ray median |
|---|---|---|---|---|---|---|---|
| 0.50-0.55 m | 2 | 13.9 | 13.9 | 30.8 | 5.6 | 5.6 | -29.1 |

Size error: not available (the viewer's analysis carries no size estimate).

Pair `B_corners/pair1.zip`:
- miss: blue_cylinder (cylinder, r3c3); no detection box covers it
- miss: pool_box_1 (cube, None); no detection box covers it
- miss: pool_box_2 (cube, None); no detection box covers it
- miss: pool_box_3 (cube, None); no detection box covers it
- miss: pool_box_4 (cube, None); no detection box covers it
- miss: pool_box_5 (cube, None); no detection box covers it
- miss: pool_cyl_1 (cylinder, None); no detection box covers it
- miss: pool_cyl_2 (cylinder, None); no detection box covers it
- miss: pool_cyl_3 (cylinder, None); no detection box covers it
- miss: pool_cyl_4 (cylinder, None); no detection box covers it
- miss: pool_cyl_5 (cylinder, None); no detection box covers it

Pair `B_corners/pair2.zip`:
- miss: blue_cylinder (cylinder, r3c3); no detection box covers it
- miss: pool_box_1 (cube, None); no detection box covers it
- miss: pool_box_2 (cube, None); no detection box covers it
- miss: pool_box_3 (cube, None); no detection box covers it
- miss: pool_box_4 (cube, None); no detection box covers it
- miss: pool_box_5 (cube, None); no detection box covers it
- miss: pool_cyl_1 (cylinder, None); no detection box covers it
- miss: pool_cyl_2 (cylinder, None); no detection box covers it
- miss: pool_cyl_3 (cylinder, None); no detection box covers it
- miss: pool_cyl_4 (cylinder, None); no detection box covers it
- miss: pool_cyl_5 (cylinder, None); no detection box covers it

## Layout C_swapped_depth

Static bracket: ok (largest move 0.0 mm; scene `FWDCenterLabSivaPool` revision `initial`).

Pairs graded: 2. Class accuracy on matched objects: 0.0. Misses: 2. False positives: 0. OOV hits: 0. Matched by 2D only (no 3D): 0.

| class | in view | correct | detections | recall | precision |
|---|---|---|---|---|---|
| cube | 2 | 0 | 0 | 0.0 | - |
| cylinder | 2 | 0 | 2 | 0.0 | 0.0 |

Off-board truth in view (parked floor pool; not in the headline): 20 sightings, 0 correct, 0 wrong class, 0 OOV, 20 missed.

Confusion (truth → reported): cube: cylinder 2; cylinder: missed 2

`empty` detections: 12 (12 correct, 0 false positive, 0 not on a cell). Precision 1.0.

| object | kind | seen | correct | wrong class | missed | recall |
|---|---|---|---|---|---|---|
| blue_cylinder | cylinder | 2 | 0 | 0 | 2 | 0.0 |
| red_cube | cube | 2 | 0 | 2 | 0 | 0.0 |

Per cell (outcome counts over pairs):

| cell | outcomes |
|---|---|
| r1c1 | correct 2 |
| r1c2 | unknown 2 |
| r1c3 | unknown 2 |
| r2c1 | correct 2 |
| r2c2 | correct 2 |
| r2c3 | correct 2 |
| r3c1 | correct 2 |
| r3c2 | wrong 2 |
| r3c3 | correct 2 |

3D error in mm (matched objects with a 3D position). `gate` is to the visible-surface point; `centre`, `horizontal` (x, y) and signed `along-ray` (+ = farther than the centre) are to the true centre.

| group | n | gate mean | gate p95 | centre median | horizontal median | horizontal p95 | along-ray median |
|---|---|---|---|---|---|---|---|
| all | 2 | 21.0 | 21.0 | 28.1 | 23.3 | 23.3 | -25.6 |

By cell:

| group | n | gate mean | gate p95 | centre median | horizontal median | horizontal p95 | along-ray median |
|---|---|---|---|---|---|---|---|
| r3c2 | 2 | 21.0 | 21.0 | 28.1 | 23.3 | 23.3 | -25.6 |

By camera distance:

| group | n | gate mean | gate p95 | centre median | horizontal median | horizontal p95 | along-ray median |
|---|---|---|---|---|---|---|---|
| 0.55-0.60 m | 2 | 21.0 | 21.0 | 28.1 | 23.3 | 23.3 | -25.6 |

Size error: not available (the viewer's analysis carries no size estimate).

Pair `C_swapped_depth/pair1.zip`:
- miss: blue_cylinder (cylinder, r1c2); no detection box covers it
- miss: pool_box_1 (cube, None); no detection box covers it
- miss: pool_box_2 (cube, None); no detection box covers it
- miss: pool_box_3 (cube, None); no detection box covers it
- miss: pool_box_4 (cube, None); no detection box covers it
- miss: pool_box_5 (cube, None); no detection box covers it
- miss: pool_cyl_1 (cylinder, None); no detection box covers it
- miss: pool_cyl_2 (cylinder, None); no detection box covers it
- miss: pool_cyl_3 (cylinder, None); no detection box covers it
- miss: pool_cyl_4 (cylinder, None); no detection box covers it
- miss: pool_cyl_5 (cylinder, None); no detection box covers it

Pair `C_swapped_depth/pair2.zip`:
- miss: blue_cylinder (cylinder, r1c2); no detection box covers it
- miss: pool_box_1 (cube, None); no detection box covers it
- miss: pool_box_2 (cube, None); no detection box covers it
- miss: pool_box_3 (cube, None); no detection box covers it
- miss: pool_box_4 (cube, None); no detection box covers it
- miss: pool_box_5 (cube, None); no detection box covers it
- miss: pool_cyl_1 (cylinder, None); no detection box covers it
- miss: pool_cyl_2 (cylinder, None); no detection box covers it
- miss: pool_cyl_3 (cylinder, None); no detection box covers it
- miss: pool_cyl_4 (cylinder, None); no detection box covers it
- miss: pool_cyl_5 (cylinder, None); no detection box covers it

## Layout D_full

Static bracket: ok (largest move 0.001 mm; scene `FWDCenterLabSivaPool` revision `initial`).

Pairs graded: 2. Class accuracy on matched objects: 0.75. Misses: 10. False positives: 2. OOV hits: 0. Matched by 2D only (no 3D): 0.

| class | in view | correct | detections | recall | precision |
|---|---|---|---|---|---|
| cube | 10 | 0 | 0 | 0.0 | - |
| cylinder | 8 | 6 | 10 | 0.75 | 0.6 |

Off-board truth in view (parked floor pool; not in the headline): 6 sightings, 0 correct, 0 wrong class, 0 OOV, 6 missed.

Confusion (truth → reported): cube: cylinder 2, missed 8; cylinder: cylinder 6, missed 2

`empty` detections: 6 (0 correct, 6 false positive, 0 not on a cell). Precision 0.0.

| object | kind | seen | correct | wrong class | missed | recall |
|---|---|---|---|---|---|---|
| pool_box_1 | cube | 2 | 0 | 2 | 0 | 0.0 |
| pool_box_2 | cube | 2 | 0 | 0 | 2 | 0.0 |
| pool_box_3 | cube | 2 | 0 | 0 | 2 | 0.0 |
| pool_box_4 | cube | 2 | 0 | 0 | 2 | 0.0 |
| pool_box_5 | cube | 2 | 0 | 0 | 2 | 0.0 |
| pool_cyl_1 | cylinder | 2 | 2 | 0 | 0 | 1.0 |
| pool_cyl_2 | cylinder | 2 | 0 | 0 | 2 | 0.0 |
| pool_cyl_3 | cylinder | 2 | 2 | 0 | 0 | 1.0 |
| pool_cyl_4 | cylinder | 2 | 2 | 0 | 0 | 1.0 |

Per cell (outcome counts over pairs):

| cell | outcomes |
|---|---|
| r1c1 | wrong 2 |
| r1c2 | correct 2 |
| r1c3 | wrong 2 |
| r2c1 | unknown 2 |
| r2c2 | wrong 2 |
| r2c3 | correct 2 |
| r3c1 | wrong 2 |
| r3c2 | correct 2 |
| r3c3 | unknown 2 |

3D error in mm (matched objects with a 3D position). `gate` is to the visible-surface point; `centre`, `horizontal` (x, y) and signed `along-ray` (+ = farther than the centre) are to the true centre.

| group | n | gate mean | gate p95 | centre median | horizontal median | horizontal p95 | along-ray median |
|---|---|---|---|---|---|---|---|
| all | 8 | 19.2 | 30.1 | 24.4 | 12.3 | 18.7 | -23.3 |

By cell:

| group | n | gate mean | gate p95 | centre median | horizontal median | horizontal p95 | along-ray median |
|---|---|---|---|---|---|---|---|
| r1c2 | 2 | 16.7 | 16.7 | 24.3 | 8.8 | 8.8 | -22.8 |
| r2c3 | 2 | 12.0 | 12.1 | 30.0 | 11.1 | 11.1 | -29.9 |
| r3c1 | 2 | 18.3 | 18.3 | 24.5 | 18.7 | 18.7 | -23.8 |
| r3c2 | 2 | 29.9 | 30.5 | 15.5 | 13.7 | 13.8 | -13.4 |

By camera distance:

| group | n | gate mean | gate p95 | centre median | horizontal median | horizontal p95 | along-ray median |
|---|---|---|---|---|---|---|---|
| 0.45-0.50 m | 2 | 16.7 | 16.7 | 24.3 | 8.8 | 8.8 | -22.8 |
| 0.50-0.55 m | 2 | 12.0 | 12.1 | 30.0 | 11.1 | 11.1 | -29.9 |
| 0.55-0.60 m | 2 | 29.9 | 30.5 | 15.5 | 13.7 | 13.8 | -13.4 |
| >=0.60 m | 2 | 18.3 | 18.3 | 24.5 | 18.7 | 18.7 | -23.8 |

Size error: not available (the viewer's analysis carries no size estimate).

Pair `D_full/pair1.zip`:
- miss: blue_cylinder (cylinder, None); no detection box covers it
- miss: pool_box_2 (cube, r3c3); no detection box covers it
- miss: pool_box_3 (cube, r2c2); no detection box covers it
- miss: pool_box_4 (cube, r1c1); no detection box covers it
- miss: pool_box_5 (cube, r1c3); no detection box covers it
- miss: pool_cyl_2 (cylinder, r2c1); a box covers it labelled cylinder
- miss: pool_cyl_5 (cylinder, None); no detection box covers it
- miss: red_cube (cube, None); no detection box covers it
- false positive: cylinder #2, nearest truth pool_cyl_2 at 49.8 mm

Pair `D_full/pair2.zip`:
- miss: blue_cylinder (cylinder, None); no detection box covers it
- miss: pool_box_2 (cube, r3c3); no detection box covers it
- miss: pool_box_3 (cube, r2c2); no detection box covers it
- miss: pool_box_4 (cube, r1c1); no detection box covers it
- miss: pool_box_5 (cube, r1c3); no detection box covers it
- miss: pool_cyl_2 (cylinder, r2c1); a box covers it labelled cylinder
- miss: pool_cyl_5 (cylinder, None); no detection box covers it
- miss: red_cube (cube, None); no detection box covers it
- false positive: cylinder #2, nearest truth pool_cyl_2 at 50.9 mm

## Layout E_full_swapped

Static bracket: ok (largest move 0.001 mm; scene `FWDCenterLabSivaPool` revision `initial`).

Pairs graded: 2. Class accuracy on matched objects: 1.0. Misses: 10. False positives: 2. OOV hits: 0. Matched by 2D only (no 3D): 0.

| class | in view | correct | detections | recall | precision |
|---|---|---|---|---|---|
| cube | 8 | 2 | 2 | 0.25 | 1.0 |
| cylinder | 10 | 6 | 8 | 0.6 | 0.75 |

Off-board truth in view (parked floor pool; not in the headline): 6 sightings, 0 correct, 0 wrong class, 0 OOV, 6 missed.

Confusion (truth → reported): cube: cube 2, missed 6; cylinder: cylinder 6, missed 4

`empty` detections: 6 (0 correct, 6 false positive, 0 not on a cell). Precision 0.0.

| object | kind | seen | correct | wrong class | missed | recall |
|---|---|---|---|---|---|---|
| pool_box_1 | cube | 2 | 0 | 0 | 2 | 0.0 |
| pool_box_2 | cube | 2 | 2 | 0 | 0 | 1.0 |
| pool_box_3 | cube | 2 | 0 | 0 | 2 | 0.0 |
| pool_box_4 | cube | 2 | 0 | 0 | 2 | 0.0 |
| pool_cyl_1 | cylinder | 2 | 0 | 0 | 2 | 0.0 |
| pool_cyl_2 | cylinder | 2 | 2 | 0 | 0 | 1.0 |
| pool_cyl_3 | cylinder | 2 | 2 | 0 | 0 | 1.0 |
| pool_cyl_4 | cylinder | 2 | 2 | 0 | 0 | 1.0 |
| pool_cyl_5 | cylinder | 2 | 0 | 0 | 2 | 0.0 |

Per cell (outcome counts over pairs):

| cell | outcomes |
|---|---|
| r1c1 | correct 2 |
| r1c2 | wrong 2 |
| r1c3 | unknown 2 |
| r2c1 | correct 2 |
| r2c2 | correct 2 |
| r2c3 | wrong 2 |
| r3c1 | unknown 2 |
| r3c2 | wrong 2 |
| r3c3 | correct 2 |

3D error in mm (matched objects with a 3D position). `gate` is to the visible-surface point; `centre`, `horizontal` (x, y) and signed `along-ray` (+ = farther than the centre) are to the true centre.

| group | n | gate mean | gate p95 | centre median | horizontal median | horizontal p95 | along-ray median |
|---|---|---|---|---|---|---|---|
| all | 8 | 19.0 | 33.4 | 24.1 | 9.7 | 11.1 | -24.0 |

By cell:

| group | n | gate mean | gate p95 | centre median | horizontal median | horizontal p95 | along-ray median |
|---|---|---|---|---|---|---|---|
| r1c1 | 2 | 10.9 | 10.9 | 31.3 | 10.8 | 10.8 | -31.1 |
| r2c1 | 2 | 14.3 | 14.3 | 24.5 | 8.7 | 8.7 | -24.4 |
| r2c2 | 2 | 17.4 | 17.4 | 23.6 | 7.8 | 7.8 | -23.5 |
| r3c3 | 2 | 33.4 | 33.4 | 11.8 | 11.0 | 11.1 | -9.6 |

By camera distance:

| group | n | gate mean | gate p95 | centre median | horizontal median | horizontal p95 | along-ray median |
|---|---|---|---|---|---|---|---|
| 0.50-0.55 m | 6 | 14.2 | 17.4 | 24.5 | 8.7 | 10.8 | -24.4 |
| 0.55-0.60 m | 2 | 33.4 | 33.4 | 11.8 | 11.0 | 11.1 | -9.6 |

Size error: not available (the viewer's analysis carries no size estimate).

Pair `E_full_swapped/pair1.zip`:
- miss: blue_cylinder (cylinder, None); no detection box covers it
- miss: pool_box_1 (cube, r3c2); no detection box covers it
- miss: pool_box_3 (cube, r2c3); no detection box covers it
- miss: pool_box_4 (cube, r1c2); no detection box covers it
- miss: pool_box_5 (cube, None); no detection box covers it
- miss: pool_cyl_1 (cylinder, r3c1); a box covers it labelled cylinder
- miss: pool_cyl_5 (cylinder, r1c3); no detection box covers it
- miss: red_cube (cube, None); no detection box covers it
- false positive: cylinder #2, nearest truth pool_cyl_1 at 50.6 mm

Pair `E_full_swapped/pair2.zip`:
- miss: blue_cylinder (cylinder, None); no detection box covers it
- miss: pool_box_1 (cube, r3c2); no detection box covers it
- miss: pool_box_3 (cube, r2c3); no detection box covers it
- miss: pool_box_4 (cube, r1c2); no detection box covers it
- miss: pool_box_5 (cube, None); no detection box covers it
- miss: pool_cyl_1 (cylinder, r3c1); a box covers it labelled cylinder
- miss: pool_cyl_5 (cylinder, r1c3); no detection box covers it
- miss: red_cube (cube, None); no detection box covers it
- false positive: cylinder #2, nearest truth pool_cyl_1 at 50.6 mm

## Not modelled

- Optical effects of the sim camera (exposure, background) differ from the lab's; see #182 findings 3-5.
- The in-view test uses the profile's lens model; objects near the edge may flip in or out.
- Each pair is an independent sample of a static layout; repeated pairs of one layout are not independent scenes.
