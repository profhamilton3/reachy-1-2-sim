# Route clearance fixture

`arm_routes.json` is reachy-tabletop-ai `src/reachy_ai/motion/arm_routes.json` @ `7468ccb` (SHA-256
`c312e0e8290928f3de20261ea2abf4abffd4a279a895afc28caa08b9b3825642`), byte for byte: the MovePieceLogic poses, REST and
REST_OUT, and `B2_GRIP_TUNE`. `tests/unit/test_route_clearance.py` builds its synthetic routes from it, and checks
that `B2_GRIP_TUNE` hashes to the tabletop's `89e62bda86b0…` with the vendored hash and limits.

`../arm_routes/` is the tabletop's golden hash fixture (`tests/fixtures/arm_routes/` @ `7468ccb`), byte for byte.
