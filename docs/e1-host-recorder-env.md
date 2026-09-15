# E1 recorder host environment (2026-09-15 matrix readiness)

Pilot item (concrete failure without this doc): the host's system Python
cannot install `reachy-sdk==0.7.0`, so `scripts/measure_route_clearance.py`
(the recorder) cannot run there directly. Without a recorded recipe, the
next operator either burns time re-deriving it or runs the recorder inside
the Docker container (N8/A2) -- which fails closed via `verify_simulator_identity`'s
container check, but only after wasting a setup cycle, and misleadingly
unless the reason is read carefully.

## Why a separate venv

Host Python on the pilot machine was 3.14.0. `reachy-sdk==0.7.0` requires
`mobile-base-sdk`, whose 1.x pins `protobuf<=4.25.3`, which forces
`grpcio-tools==1.62.2` -- no CPython 3.14 wheel exists for it, and it fails
to build from source under 3.14 (`pkg_resources` missing). This is a
transitive dependency conflict, not anything specific to this repo.

## Recipe (PR #113 `EVIDENCE.md` §2, exact versions from the pilot run)

```bash
uv venv --python 3.12 /path/to/e1venv
uv pip install --python /path/to/e1venv/bin/python \
    reachy-sdk==0.7.0 "numpy<2" pyyaml nbconvert nbformat ipykernel jupyter_client
```

Resolved versions that worked on the pilot machine: Python 3.12.14,
`reachy-sdk` 0.7.0, `reachy-sdk-api` 0.7.3, `grpcio` 1.62.2, `grpcio-tools`
1.62.2, `protobuf` 4.25.3, `numpy` 1.26.4, `pyyaml` 6.0.3, `scipy` 1.17.1,
`opencv-python` 4.11.0.86, `mobile-base-sdk` 1.0.2, `nbconvert` 7.17.1,
`ipykernel` 7.3.0.

**`reachy2-sdk-api` arrives transitively** (pulled in by `mobile-base-sdk`
1.0.2) and is never imported anywhere in this repo or by the recorder --
the CLAUDE.md rule is about imports, not what's installed in the venv. Do
not treat its presence in `pip list`/`uv pip list` as a violation.

## Run the recorder from this venv

```bash
/path/to/e1venv/bin/python -m scripts.measure_route_clearance --route ... --record-root ...
```

or activate the venv and run it directly with `PYTHONPATH=src`.

The linker (`scripts/link_e1_flight.py`) and tail checker
(`scripts/e1_tail_check.py`) do **not** need this venv -- they only need
numpy/yaml and the repo, and ran on host Python 3.14 with `PYTHONPATH=src`
in the pilot.

## Run from a host shell, not the container

`verify_simulator_identity` refuses (N8/A2, 2026-09-15 matrix readiness) if
`/.dockerenv` exists: the recorder shares the host monotonic clock with the
native server (see `scripts/e1_identity.py`'s "Two clocks, deliberately not
conflated" section), and a container shell does not share it. Always run
the recorder -- venv or not -- from a host shell.
