"""Issue #89: the gate between a stored success and a moving arm.

Pure and offline — no store, no scene, no SDK.  Every test states a world and
a movement and asks one question: may this recipe be flown?

The point of the module under test is that the answer is NO far more often
than `query_compatible_trials()` would suggest, and that every NO says which
field disagreed.
"""

import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../src"))

from reachy_ai.experience.compatibility import (  # noqa: E402
    IDENTITY_FIELDS,
    PROMOTED,
    CompatibilityError,
    ReuseCandidate,
    ReusePolicy,
    ReuseRequest,
    check_reuse,
    select_reusable,
)
from reachy_ai.experience.models import EpisodeStatus, SimulatorIdentity  # noqa: E402


def an_identity(**kw) -> SimulatorIdentity:
    base = dict(
        model_sha256="model-1", compiled_model_sha256="compiled-1",
        scene_sha256="scene-1", scene_revision="rev-1",
        scene_schema_version="1.0", scene_compiler_version="c-1",
        physics_profile_id="default", calibration_profile_id="default",
        sensor_effect_profile_id="default", backend_name="native_mujoco",
        protocol_version=1, working_tree_dirty=False,
    )
    base.update(kw)
    return SimulatorIdentity(**base)


def a_request(**kw) -> ReuseRequest:
    base = dict(task_type="stow_arm", arm="right", route="STOW_ROUTE",
                route_version=1, start_posture="rest",
                obstacles=frozenset({"soda_can"}))
    base.update(kw)
    return ReuseRequest(**base)


def a_candidate(**kw) -> ReuseCandidate:
    base = dict(
        trial_id="t-1", identity=an_identity(), task_type="stow_arm",
        arm="right", route="STOW_ROUTE", route_version=1,
        start_posture="rest", promotion_state=PROMOTED,
        status=EpisodeStatus.SUCCEEDED.value, success=True,
        live_interactive=False, obstacles=frozenset({"soda_can"}),
    )
    base.update(kw)
    return ReuseCandidate(**base)


def decide(candidate=None, identity=None, request=None, policy=None):
    return check_reuse(candidate or a_candidate(), identity or an_identity(),
                       request or a_request(), policy)


# ---------------------------------------------------------------------------
# The one case that is allowed
# ---------------------------------------------------------------------------

def test_a_promoted_trial_in_the_same_world_and_movement_is_reusable():
    decision = decide()
    assert decision.allowed
    assert decision.trial_id == "t-1"
    assert decision.policy_version == 1
    assert not decision.mismatches
    # Everything it checked is named, so an accepted reuse can be audited as
    # easily as a rejected one.
    for field in IDENTITY_FIELDS:
        assert field in decision.matched
    for field in ("task_type", "arm", "route", "route_version",
                  "start_posture", "obstacles"):
        assert field in decision.matched


# ---------------------------------------------------------------------------
# Every identity field is enforced, not just the two the query compares
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field,other", [
    ("model_sha256", "model-2"),
    ("compiled_model_sha256", "compiled-2"),
    ("scene_sha256", "scene-2"),
    ("scene_revision", "rev-2"),
    ("scene_schema_version", "2.0"),
    ("scene_compiler_version", "c-2"),
    ("physics_profile_id", "stiff"),
    ("calibration_profile_id", "lab-2026-09"),
    ("sensor_effect_profile_id", "noisy"),
    ("backend_name", "sdk_bridge"),
    ("protocol_version", 2),
])
def test_a_different_world_is_refused_field_by_field(field, other):
    """`query_compatible_trials` would return this row for nine of these
    eleven: it compares model_sha256 and scene_sha256 and nothing else."""
    decision = decide(a_candidate(identity=an_identity(**{field: other})))
    assert not decision.allowed
    assert [m.field for m in decision.mismatches] == [field]
    assert field in decision.reason
    assert str(other) in decision.reason


def test_every_disagreeing_field_is_reported_not_just_the_first():
    decision = decide(a_candidate(identity=an_identity(
        calibration_profile_id="lab-2026-09", physics_profile_id="stiff")))
    assert not decision.allowed
    assert {m.field for m in decision.mismatches} == {
        "calibration_profile_id", "physics_profile_id"}


def test_an_unknown_compiled_hash_is_a_mismatch_and_not_a_pass():
    """One side empty and the other set is missing provenance, which is a
    reason to refuse rather than a reason to skip the comparison."""
    decision = decide(a_candidate(identity=an_identity(compiled_model_sha256="")))
    assert not decision.allowed
    assert [m.field for m in decision.mismatches] == ["compiled_model_sha256"]


# ---------------------------------------------------------------------------
# Promotion is what authorises flying, not success
# ---------------------------------------------------------------------------

def test_an_unpromoted_success_is_never_reusable():
    decision = decide(a_candidate(promotion_state="unpromoted"))
    assert not decision.allowed
    assert "unpromoted" in decision.reason
    assert "evidence" in decision.reason


def test_a_failed_trial_is_not_a_candidate_at_all():
    decision = decide(a_candidate(status=EpisodeStatus.FAILED.value,
                                  success=False))
    assert not decision.allowed
    assert "did not succeed" in decision.reason


def test_a_succeeded_row_whose_success_flag_is_false_is_refused():
    """The two disagree only if something wrote them inconsistently, and a
    gate is the wrong place to decide which one to believe."""
    decision = decide(a_candidate(success=False))
    assert not decision.allowed


def test_a_live_interactive_episode_is_never_reusable():
    decision = decide(a_candidate(live_interactive=True,
                                  promotion_state=PROMOTED))
    assert not decision.allowed
    assert "a person was driving" in decision.reason


def test_a_trial_from_a_modified_checkout_is_refused():
    decision = decide(a_candidate(
        identity=an_identity(working_tree_dirty=True)))
    assert not decision.allowed
    assert "modified checkout" in decision.reason


# ---------------------------------------------------------------------------
# The movement, and the board
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field,other", [
    ("task_type", "wave"),
    ("arm", "left"),
    ("route", "PLACE_ROUTE"),
    ("route_version", 2),
    ("start_posture", "home"),
])
def test_a_different_movement_is_refused(field, other):
    decision = decide(a_candidate(**{field: other}))
    assert not decision.allowed
    assert [m.field for m in decision.mismatches] == [field]


def test_a_board_that_was_never_recorded_is_refused():
    """"Not recorded" is not "empty".  A route certified over an unknown board
    is certified over nothing."""
    decision = decide(a_candidate(obstacles=None))
    assert not decision.allowed
    assert "did not record which objects" in decision.reason


def test_a_different_board_is_refused():
    decision = decide(a_candidate(obstacles=frozenset({"soda_can", "red_cube"})))
    assert not decision.allowed
    assert [m.field for m in decision.mismatches] == ["obstacles"]


def test_an_empty_board_matches_an_empty_board():
    decision = decide(a_candidate(obstacles=frozenset()),
                      request=a_request(obstacles=frozenset()))
    assert decision.allowed


# ---------------------------------------------------------------------------
# The gate refuses to answer when it cannot tell worlds apart
# ---------------------------------------------------------------------------

def test_a_degenerate_current_identity_raises_rather_than_allowing():
    """An empty model hash matches everything.  A gate that cannot tell worlds
    apart must refuse to answer, not answer yes."""
    with pytest.raises(CompatibilityError) as exc:
        decide(identity=an_identity(model_sha256=""))
    assert "model_sha256" in str(exc.value)


# ---------------------------------------------------------------------------
# Rows off the store
# ---------------------------------------------------------------------------

def a_row(**kw):
    import json
    base = {
        "trial_id": "t-row",
        "identity_json": an_identity().to_json(),
        "task_type": "stow_arm",
        "task_spec_json": json.dumps({"task_type": "stow_arm",
                                      "arm_policy": "right"}),
        "recipe_json": json.dumps({"kind": "ability_route",
                                   "route": "STOW_ROUTE", "route_version": 1,
                                   "arm": "right",
                                   "expected_start_posture": "rest"}),
        "optimizer_metadata_json": json.dumps({"obstacles": ["soda_can"]}),
        "status": EpisodeStatus.SUCCEEDED.value,
        "success": 1,
        "promotion_state": PROMOTED,
        "live_interactive": 0,
    }
    base.update(kw)
    return base


def test_a_store_row_becomes_a_candidate():
    candidate = ReuseCandidate.from_row(a_row())
    assert candidate.trial_id == "t-row"
    assert candidate.route == "STOW_ROUTE"
    assert candidate.arm == "right"
    assert candidate.obstacles == frozenset({"soda_can"})
    assert check_reuse(candidate, an_identity(), a_request()).allowed


def test_a_row_from_an_older_store_reads_as_the_conservative_value():
    """A missing column is never permission.  `live_interactive` is absent in
    a v1 file, and `obstacles` is absent in anything written before #89."""
    row = a_row(optimizer_metadata_json="{}")
    del row["live_interactive"]
    candidate = ReuseCandidate.from_row(row)
    assert candidate.live_interactive is False
    assert candidate.obstacles is None
    assert not check_reuse(candidate, an_identity(), a_request()).allowed


def test_unreadable_json_does_not_crash_the_gate():
    candidate = ReuseCandidate.from_row(a_row(recipe_json="not json",
                                              identity_json=""))
    decision = check_reuse(candidate, an_identity(), a_request())
    assert not decision.allowed


# ---------------------------------------------------------------------------
# Choosing between rows
# ---------------------------------------------------------------------------

def test_the_first_reusable_row_wins_and_every_row_is_explained():
    rows = [
        a_row(trial_id="unpromoted", promotion_state="unpromoted"),
        a_row(trial_id="wrong-calibration",
              identity_json=an_identity(
                  calibration_profile_id="lab-2026-09").to_json()),
        a_row(trial_id="good"),
        a_row(trial_id="also-good"),
    ]
    chosen, decisions = select_reusable(rows, an_identity(), a_request())
    assert chosen is not None and chosen.trial_id == "good"
    assert [d.trial_id for d in decisions] == [
        "unpromoted", "wrong-calibration", "good", "also-good"]
    assert [d.allowed for d in decisions] == [False, False, True, True]
    # "Nothing was reusable" has to be explainable, so the reasons come back
    # even for the rows that lost.
    assert all(d.reason for d in decisions)


def test_an_empty_store_selects_nothing_and_says_nothing_false():
    chosen, decisions = select_reusable([], an_identity(), a_request())
    assert chosen is None
    assert decisions == []


def test_a_store_with_nothing_promoted_is_the_state_of_the_world_today():
    """The correct answer today, and not a bug: promotion is a reviewed
    process that does not exist yet, so nothing is certified for anything."""
    rows = [a_row(trial_id=f"t{i}", promotion_state="unpromoted")
            for i in range(5)]
    chosen, decisions = select_reusable(rows, an_identity(), a_request())
    assert chosen is None
    assert all("unpromoted" in d.reason for d in decisions)


# ---------------------------------------------------------------------------
# The policy is explicit, versioned, and recorded
# ---------------------------------------------------------------------------

def test_the_decision_records_the_policy_that_made_it():
    policy = ReusePolicy(policy_version=7)
    assert decide(policy=policy).policy_version == 7


def test_relaxing_a_rule_takes_a_named_policy_and_not_an_argument():
    strict = decide(a_candidate(promotion_state="unpromoted"))
    relaxed = decide(a_candidate(promotion_state="unpromoted"),
                     policy=ReusePolicy(require_promoted=False))
    assert not strict.allowed
    assert relaxed.allowed


def test_a_decision_serialises_for_the_provenance_record():
    d = decide(a_candidate(arm="left")).to_dict()
    assert d["allowed"] is False
    assert d["trial_id"] == "t-1"
    assert d["policy_version"] == 1
    assert d["mismatches"][0]["field"] == "arm"
    assert d["mismatches"][0]["wanted"] == "right"
    assert d["mismatches"][0]["found"] == "left"
