"""CPU cross-play reports cup victories separately from leaderboard scores."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from mjlab_microduck.tug_of_war import evaluation


def test_cross_play_uses_official_judge_swaps_sides_and_reports_faults(monkeypatch, tmp_path):
    # Two normal wins, one loss, one draw, opponent fault, and shared reset.
    outcomes = iter([
        (1, "pit", 150.0), (0, "marker", 25.0), (0, "pit", -140.0),
        (None, "draw", 0.0), (1, "fault", -200.0), (None, "reset", -200.0),
    ])
    calls, scored, loaded = [], [], []
    manifest = SimpleNamespace(rating=SimpleNamespace(pit_bonus_cm=100.0))
    model = object()

    def load_policy(path):
        loaded.append(path)
        return path.name

    def run_bout(actual_manifest, west, east, seed, *, model):
        assert actual_manifest is manifest
        calls.append((west, east, seed, model))
        loser, how, score = next(outcomes)
        return SimpleNamespace(loser=loser, how=how, time_s=1.2, score=score)

    def bout_score_cm(outcome, seat, bonus):
        scored.append((seat, bonus))
        return outcome.score

    monkeypatch.setattr(evaluation, "_arena_api", lambda: SimpleNamespace(
        load_event=lambda event: manifest if event == "tug-of-war" else None,
        compile_tug=lambda actual: model if actual is manifest else None,
        load_policy=load_policy, run_bout=run_bout, bout_score_cm=bout_score_cm,
    ))
    output = tmp_path / "nested" / "crossplay.json"
    report = evaluation.evaluate_policy(
        Path("learner.onnx"), {"frozen": Path("opponent.onnx")},
        # Side swapping is based on bout index, independent of seed parity.
        seeds=(1001, 1003, 1005, 1007, 1009, 1011), output=output,
    )
    assert loaded == [Path("learner.onnx"), Path("opponent.onnx")]
    assert [(a, b) for a, b, _, _ in calls] == [
        ("learner.onnx", "opponent.onnx"), ("opponent.onnx", "learner.onnx"),
    ] * 3
    assert all(actual_model is model for _, _, _, actual_model in calls)
    assert scored == [(0, 100.0), (1, 100.0)] * 3
    opponent = report["opponents"]["frozen"]
    assert (opponent["wins"], opponent["losses"], opponent["draws"]) == (3, 1, 2)
    assert report["mean_win_rate"] == opponent["win_rate"] == 0.5
    assert opponent["mean_score_cm"] == pytest.approx(-365 / 6)
    assert opponent["faults"] == 2 and opponent["resets"] == 1
    assert report["fault_rate"] == pytest.approx(1 / 3)
    assert opponent["bouts"][4]["winner"] == "learner"
    assert opponent["bouts"][4]["score_cm"] == -200.0
    assert opponent["bouts"][5]["winner"] is None
    assert report == json.loads(output.read_text())
    assert not output.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize("opponents,seeds", [({}, (1,)), ({"reference": Path("ref.onnx")}, ())])
def test_cross_play_requires_opponents_and_seeds(opponents, seeds):
    with pytest.raises(ValueError, match="at least one opponent and one seed"):
        evaluation.evaluate_policy(Path("learner.onnx"), opponents, seeds)


def test_missing_optional_tug_runner_gives_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "arena.tug", None)
    with pytest.raises(RuntimeError, match="microduck-arena 0.7.0 or newer"):
        evaluation._arena_api()
