#!/usr/bin/env bash
# The exact entry point of the fixed 7-run harness validation batch (see validation_batch.py).
# Written and dry-run only until the owner authorizes execution; neither the batch nor D1 is
# authorized by the existence of this file.
#
#   run_validation_batch.sh --pin <full 40-hex SHA of the reviewed branch head P> \
#       --out ~/goalfix-cmp-harness-validation-<YYYY-MM-DD> --venv ~/goalfix-venv --runs 7 [--dry-run]
#
# Optional: --repo <path> (source checkout to clone; default ~/reachy-1-2-sim).
set -euo pipefail

die() { echo "REFUSED: $*" >&2; exit 2; }

PIN=""; OUT=""; VENV=""; RUNS=""; DRY=0; REPO="${HOME}/reachy-1-2-sim"
while [ $# -gt 0 ]; do
  case "$1" in
    --pin)     [ $# -ge 2 ] || die "--pin needs a value";  PIN="$2";  shift 2;;
    --out)     [ $# -ge 2 ] || die "--out needs a value";  OUT="$2";  shift 2;;
    --venv)    [ $# -ge 2 ] || die "--venv needs a value"; VENV="$2"; shift 2;;
    --runs)    [ $# -ge 2 ] || die "--runs needs a value"; RUNS="$2"; shift 2;;
    --repo)    [ $# -ge 2 ] || die "--repo needs a value"; REPO="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    -h|--help) sed -n '2,10p' "$0"; exit 0;;
    *) die "unknown argument: $1";;
  esac
done

[ -n "$PIN" ]  || die "--pin is required"
[ -n "$OUT" ]  || die "--out is required"
[ -n "$VENV" ] || die "--venv is required"
[ -n "$RUNS" ] || die "--runs is required"
OUT="${OUT/#\~/$HOME}"; VENV="${VENV/#\~/$HOME}"; REPO="${REPO/#\~/$HOME}"

[[ "$PIN" =~ ^[0-9a-f]{40}$ ]] || die "--pin must be a full 40-hex-digit SHA"
[ "$RUNS" = "7" ] || die "--runs must be exactly 7 (the scope is fixed in advance)"
case "$OUT" in /*) ;; *) die "--out must be an absolute path";; esac
PARENT="$(cd "$(dirname "$OUT")" 2>/dev/null && pwd -P)" || die "the parent directory of --out does not exist"
case "$PARENT/$(basename "$OUT")" in
  /tmp|/tmp/*|/private/tmp|/private/tmp/*) die "--out must be outside /tmp";;
esac
[ ! -e "$OUT" ] && [ ! -L "$OUT" ] || die "--out already exists"
PY="$VENV/bin/python"
[ -x "$PY" ] || die "--venv has no executable bin/python"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

# Pins, tree/blob equalities, output path, and the exact per-run plan (no SDK import, no harness,
# no network).  Refuses (exit 2) on any mismatch.
"$PY" "$HERE/validation_batch.py" preflight --pin "$PIN" --out "$OUT" --repo "$REPO" \
      --runs "$RUNS" --venv "$VENV"

if [ "$DRY" = "1" ]; then
  echo "DRY RUN: nothing was cloned, imported or executed."
  exit 0
fi

# Source: a local CLONE (not a worktree) detached at P.
mkdir "$OUT"
git clone --local --no-checkout "$REPO" "$OUT/src"
git -C "$OUT/src" checkout --detach "$PIN"
mkdir "$OUT/tmp"

# Nothing may land in /tmp; nothing may be imported from the primary checkout.
export TMPDIR="$OUT/tmp"
export REQUIRE_REACHY_SDK=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH || true
cd "$OUT/src"
# `validation_batch.py run` verifies the clone (clean tree, HEAD == P, P:tools/goalfix_cmp == M's,
# the three bridge blobs == B's) and refuses to start on any mismatch, then asserts that
# mujoco_remote_backend, fake_reachy_server and tools.goalfix_cmp are imported from $OUT/src.
exec env PYTHONPATH="$OUT/src/src:$OUT/src/native_mujoco:$OUT/src/scripts:$OUT/src/tests/integration" \
    "$PY" "$OUT/src/tests/fixtures/goalfix_cmp/validation_batch.py" run \
    --pin "$PIN" --out "$OUT" --venv "$VENV"
