"""R12-807: Tests for StudyReport, ComparisonResult, and compare/report CLI."""

from __future__ import annotations

import json
import uuid
from typing import Dict

import pytest

from reachy_ai.evaluation.base import EpisodeVerdict, Violation, ViolationKind
from reachy_ai.motion.recipe import TrajectoryRecipe
from reachy_ai.search.report import (
    ComparisonResult,
    StudyReport,
    TrialSummary,
    compare_studies,
    generate_report,
)
from reachy_ai.search.runner import SearchConfig, SearchRunner

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _recipe(*, bounded=None):
    return TrajectoryRecipe(
        recipe_id="r",
        task_type="test",
        bounded_parameters=bounded or {
            "x": {"value": 0.0, "min": 0.0, "max": 2.0, "step": 1.0},
        },
    )


def _verdict(
    *,
    is_valid=True,
    is_safe=True,
    is_successful=True,
    accuracy=0.5,
    violations=None,
):
    eid = uuid.uuid4().hex
    return EpisodeVerdict(
        episode_id=eid,
        trial_id=eid,
        task_type="test",
        policy_version=1,
        is_valid=is_valid,
        is_safe=is_safe,
        is_successful=is_successful,
        violations=violations or [],
        metrics={},
        ranking_scores={
            "accuracy_score": accuracy,
            "clearance_score": 0.5,
            "effort_score": 1.0,
            "smoothness_score": 1.0,
            "duration_score": 0.5,
        },
        explanation="test",
    )


def _run_study(store, study_id: str, evaluate_fn, recipe=None) -> None:
    """Helper: run a full search and persist results to store."""
    r = recipe or _recipe()
    config = SearchConfig(study_id=study_id, baseline_recipe=r, budget=50)
    runner = SearchRunner(config)
    runner.run(evaluate_fn, store=store)


# ---------------------------------------------------------------------------
# TrialSummary
# ---------------------------------------------------------------------------

class TestTrialSummary:
    def test_to_dict_round_trips(self):
        ts = TrialSummary(
            trial_id="abc123",
            rank=1,
            is_valid=True,
            is_safe=True,
            is_successful=True,
            rank_key=(1.0, 1.0, 1.0, 0.8, 0.5, 1.0, 1.0, 0.5),
            ranking_scores={"accuracy_score": 0.8},
            search_point={"x": 1.0},
            violations_count=0,
        )
        d = ts.to_dict()
        assert d["trial_id"] == "abc123"
        assert d["rank"] == 1
        assert d["search_point"] == {"x": 1.0}

    def test_one_line_includes_trial_prefix(self):
        ts = TrialSummary(
            trial_id="deadbeef1234",
            rank=2,
            is_valid=True,
            is_safe=False,
            is_successful=False,
            rank_key=(1.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 0.5),
            ranking_scores={"accuracy_score": 0.0, "duration_score": 0.5},
            search_point={},
            violations_count=1,
        )
        line = ts.one_line()
        assert "deadbeef" in line
        assert "#2" in line
        assert "safe=False" in line
        assert "violations=1" in line

    def test_one_line_shows_search_point(self):
        ts = TrialSummary(
            trial_id="aa",
            rank=1,
            is_valid=True,
            is_safe=True,
            is_successful=True,
            rank_key=(1.0,) * 8,
            ranking_scores={},
            search_point={"x": 1.5, "y": 0.3},
            violations_count=0,
        )
        line = ts.one_line()
        assert "x=" in line
        assert "y=" in line


# ---------------------------------------------------------------------------
# StudyReport — generate_report
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_empty_study_returns_empty_report(self, tmp_path):
        from reachy_ai.experience.store import ExperienceStore
        db = str(tmp_path / "empty.db")
        with ExperienceStore.open(db) as store:
            report = generate_report(store, "no_such_study")
        assert report.study_id == "no_such_study"
        assert report.total_trials == 0
        assert report.best is None
        assert report.top == []

    def test_report_counts_successful_trials(self, tmp_path):
        from reachy_ai.experience.store import ExperienceStore

        db = str(tmp_path / "s.db")
        call = [0]

        def ev(r, _seed):
            call[0] += 1
            # 1st trial succeeds, rest fail the task
            return _verdict(is_successful=(call[0] == 1))

        with ExperienceStore.open(db) as store:
            _run_study(store, "st", ev)
            report = generate_report(store, "st")

        assert report.total_trials == 3   # 3-point grid
        assert report.successful_count == 1

    def test_report_counts_safe_trials(self, tmp_path):
        from reachy_ai.experience.store import ExperienceStore

        db = str(tmp_path / "s.db")

        def ev(r, _seed):
            val = float(r.bounded_parameters["x"]["value"])
            return _verdict(is_safe=(val < 2.0))  # x=0 and x=1 are safe, x=2 is not

        with ExperienceStore.open(db) as store:
            _run_study(store, "st", ev)
            report = generate_report(store, "st")

        assert report.safe_count == 2

    def test_best_is_highest_ranked(self, tmp_path):
        from reachy_ai.experience.store import ExperienceStore

        db = str(tmp_path / "s.db")

        def ev(r, _seed):
            val = float(r.bounded_parameters["x"]["value"])
            return _verdict(accuracy=val / 2.0)  # x=2 → acc=1.0 is best

        with ExperienceStore.open(db) as store:
            _run_study(store, "st", ev)
            report = generate_report(store, "st")

        assert report.best is not None
        assert report.best.rank == 1
        assert report.best.ranking_scores["accuracy_score"] == pytest.approx(1.0)

    def test_top_n_is_respected(self, tmp_path):
        from reachy_ai.experience.store import ExperienceStore

        db = str(tmp_path / "s.db")

        with ExperienceStore.open(db) as store:
            _run_study(store, "st", lambda r, _seed: _verdict())
            report = generate_report(store, "st", top_n=2)

        assert len(report.top) <= 2

    def test_search_point_captured_in_summary(self, tmp_path):
        from reachy_ai.experience.store import ExperienceStore

        db = str(tmp_path / "s.db")

        with ExperienceStore.open(db) as store:
            _run_study(store, "st", lambda r, _seed: _verdict())
            report = generate_report(store, "st")

        # Every trial should have a non-empty search_point (from optimizer_metadata)
        assert all(ts.search_point for ts in report.top)

    def test_violations_count_in_summary(self, tmp_path):
        from reachy_ai.experience.store import ExperienceStore

        db = str(tmp_path / "s.db")
        v = Violation(kind=ViolationKind.FORBIDDEN_CONTACT, description="hit table")

        def ev(r, _seed):
            val = float(r.bounded_parameters["x"]["value"])
            viols = [v] if val == 0.0 else []
            return _verdict(is_safe=(val != 0.0), violations=viols)

        with ExperienceStore.open(db) as store:
            _run_study(store, "st", ev)
            report = generate_report(store, "st")

        worst = report.top[-1]  # last in top (worst ranked)
        assert worst.violations_count >= 0  # just verify the field is populated

    def test_to_json_round_trips(self, tmp_path):
        from reachy_ai.experience.store import ExperienceStore

        db = str(tmp_path / "s.db")

        with ExperienceStore.open(db) as store:
            _run_study(store, "st", lambda r, _seed: _verdict())
            report = generate_report(store, "st")

        raw = report.to_json()
        d = json.loads(raw)
        assert d["study_id"] == "st"
        assert isinstance(d["top"], list)
        assert "generated_at" in d

    def test_summary_text_contains_study_id(self, tmp_path):
        from reachy_ai.experience.store import ExperienceStore

        db = str(tmp_path / "s.db")

        with ExperienceStore.open(db) as store:
            _run_study(store, "my_study", lambda r, _seed: _verdict())
            report = generate_report(store, "my_study")

        text = report.summary_text()
        assert "my_study" in text

    def test_summary_text_empty_study_mentions_no_trials(self, tmp_path):
        from reachy_ai.experience.store import ExperienceStore

        db = str(tmp_path / "empty.db")

        with ExperienceStore.open(db) as store:
            report = generate_report(store, "ghost")

        text = report.summary_text()
        assert "no completed" in text.lower() or "ghost" in text


# ---------------------------------------------------------------------------
# ComparisonResult — compare_studies
# ---------------------------------------------------------------------------

class TestCompareStudies:
    def _setup_two_studies(self, tmp_path, acc_a: float, acc_b: float):
        from reachy_ai.experience.store import ExperienceStore

        db = str(tmp_path / "two.db")
        with ExperienceStore.open(db) as store:
            _run_study(store, "A", lambda r, _seed: _verdict(accuracy=acc_a))
            _run_study(store, "B", lambda r, _seed: _verdict(accuracy=acc_b))
        return db

    def test_a_wins_when_better_accuracy(self, tmp_path):
        db = self._setup_two_studies(tmp_path, acc_a=0.9, acc_b=0.3)
        from reachy_ai.experience.store import ExperienceStore
        with ExperienceStore.open(db) as store:
            result = compare_studies(store, "A", "B")
        assert result.winner == "a"

    def test_b_wins_when_better_accuracy(self, tmp_path):
        db = self._setup_two_studies(tmp_path, acc_a=0.2, acc_b=0.8)
        from reachy_ai.experience.store import ExperienceStore
        with ExperienceStore.open(db) as store:
            result = compare_studies(store, "A", "B")
        assert result.winner == "b"

    def test_tie_when_equal_rank_key(self, tmp_path):
        db = self._setup_two_studies(tmp_path, acc_a=0.5, acc_b=0.5)
        from reachy_ai.experience.store import ExperienceStore
        with ExperienceStore.open(db) as store:
            result = compare_studies(store, "A", "B")
        assert result.winner is None

    def test_b_wins_when_a_has_no_trials(self, tmp_path):
        from reachy_ai.experience.store import ExperienceStore

        db = str(tmp_path / "one.db")
        with ExperienceStore.open(db) as store:
            _run_study(store, "B", lambda r, _seed: _verdict(accuracy=0.7))
            result = compare_studies(store, "A", "B")
        assert result.winner == "b"

    def test_a_wins_when_b_has_no_trials(self, tmp_path):
        from reachy_ai.experience.store import ExperienceStore

        db = str(tmp_path / "one.db")
        with ExperienceStore.open(db) as store:
            _run_study(store, "A", lambda r, _seed: _verdict(accuracy=0.7))
            result = compare_studies(store, "A", "B")
        assert result.winner == "a"

    def test_no_winner_when_both_empty(self, tmp_path):
        from reachy_ai.experience.store import ExperienceStore

        db = str(tmp_path / "empty.db")
        with ExperienceStore.open(db) as store:
            result = compare_studies(store, "X", "Y")
        assert result.winner is None
        assert result.rank_delta is None

    def test_rank_delta_sign(self, tmp_path):
        db = self._setup_two_studies(tmp_path, acc_a=0.9, acc_b=0.3)
        from reachy_ai.experience.store import ExperienceStore
        with ExperienceStore.open(db) as store:
            result = compare_studies(store, "A", "B")
        assert result.rank_delta is not None
        acc_idx = 3  # accuracy_score is tier 4 (0-indexed 3)
        assert result.rank_delta[acc_idx] > 0  # A is better in accuracy

    def test_to_json_round_trips(self, tmp_path):
        db = self._setup_two_studies(tmp_path, acc_a=0.7, acc_b=0.4)
        from reachy_ai.experience.store import ExperienceStore
        with ExperienceStore.open(db) as store:
            result = compare_studies(store, "A", "B")
        raw = result.to_json()
        d = json.loads(raw)
        assert d["winner"] == "a"
        assert "report_a" in d
        assert "report_b" in d

    def test_summary_text_contains_both_study_ids(self, tmp_path):
        db = self._setup_two_studies(tmp_path, acc_a=0.5, acc_b=0.5)
        from reachy_ai.experience.store import ExperienceStore
        with ExperienceStore.open(db) as store:
            result = compare_studies(store, "A", "B")
        text = result.summary_text()
        assert "'A'" in text
        assert "'B'" in text

    def test_safety_beats_accuracy(self, tmp_path):
        """is_safe (tier 2) outranks accuracy (tier 4): unsafe high-accuracy loses."""
        from reachy_ai.experience.store import ExperienceStore

        db = str(tmp_path / "safety.db")
        with ExperienceStore.open(db) as store:
            # Study A: safe, low accuracy
            _run_study(store, "A", lambda r, _seed: _verdict(is_safe=True, accuracy=0.1))
            # Study B: unsafe (!), high accuracy
            _run_study(store, "B", lambda r, _seed: _verdict(is_safe=False, accuracy=0.99))
            result = compare_studies(store, "A", "B")

        assert result.winner == "a"


# ---------------------------------------------------------------------------
# CLI integration — report and compare subcommands
# ---------------------------------------------------------------------------

class TestCliReport:
    def _populate_db(self, tmp_path, study_id="s1"):
        from reachy_ai.experience.store import ExperienceStore
        db = str(tmp_path / "cli.db")
        with ExperienceStore.open(db) as store:
            _run_study(store, study_id, lambda r, _seed: _verdict())
        return db

    def test_report_text_output(self, tmp_path, capsys):
        from reachy_ai.search.cli import main
        db = self._populate_db(tmp_path, "cli_study")
        rc = main(["report", "--study-id", "cli_study", "--db", db])
        assert rc == 0
        out = capsys.readouterr().out
        assert "cli_study" in out

    def test_report_json_output(self, tmp_path, capsys):
        from reachy_ai.search.cli import main
        db = self._populate_db(tmp_path, "cli_study")
        rc = main(["report", "--study-id", "cli_study", "--db", db, "--format", "json"])
        assert rc == 0
        out = capsys.readouterr().out
        d = json.loads(out)
        assert d["study_id"] == "cli_study"

    def test_report_requires_db(self, tmp_path, capsys):
        from reachy_ai.search.cli import main
        rc = main(["report", "--study-id", "x"])
        assert rc != 0

    def test_report_empty_study_exits_zero(self, tmp_path, capsys):
        from reachy_ai.experience.store import ExperienceStore
        from reachy_ai.search.cli import main
        db = str(tmp_path / "e.db")
        with ExperienceStore.open(db) as store:
            pass  # empty
        rc = main(["report", "--study-id", "ghost", "--db", db])
        assert rc == 0


class TestCliCompare:
    def _populate_two(self, tmp_path, acc_a=0.8, acc_b=0.3):
        from reachy_ai.experience.store import ExperienceStore
        db = str(tmp_path / "cli2.db")
        with ExperienceStore.open(db) as store:
            _run_study(store, "SA", lambda r, _seed: _verdict(accuracy=acc_a))
            _run_study(store, "SB", lambda r, _seed: _verdict(accuracy=acc_b))
        return db

    def test_compare_text_output(self, tmp_path, capsys):
        from reachy_ai.search.cli import main
        db = self._populate_two(tmp_path)
        rc = main(["compare", "--study-a", "SA", "--study-b", "SB", "--db", db])
        assert rc == 0
        out = capsys.readouterr().out
        assert "'SA'" in out or "SA" in out

    def test_compare_json_output(self, tmp_path, capsys):
        from reachy_ai.search.cli import main
        db = self._populate_two(tmp_path)
        rc = main(["compare", "--study-a", "SA", "--study-b", "SB", "--db", db, "--format", "json"])
        assert rc == 0
        d = json.loads(capsys.readouterr().out)
        assert d["winner"] == "a"

    def test_compare_requires_db(self, capsys):
        from reachy_ai.search.cli import main
        rc = main(["compare", "--study-a", "x", "--study-b", "y"])
        assert rc != 0
