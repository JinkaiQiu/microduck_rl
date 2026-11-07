"""Adverse CPU probes retain the official judge and report actual exposure."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("arena.tug")
import mujoco
from arena import runner, tug
from arena.course import compile_tug, duck_prefix, STRING_SITE
from arena.manifest import load_event
from arena.tug import reference_duck

from mjlab_microduck.tug_of_war import evaluation, stress


def test_stress_report_pairs_conditions_and_sanitizes_only_diagnostics(monkeypatch, tmp_path):
    calls = []
    manifest = SimpleNamespace(rating=SimpleNamespace(pit_bonus_cm=100.),
                               start=SimpleNamespace(settle_s=.5), bout=SimpleNamespace(time_limit_s=45.))
    monkeypatch.setattr(evaluation, "_arena_api", lambda: SimpleNamespace(
        load_event=lambda _: manifest, compile_tug=lambda _: object(),
        load_policy=lambda path: Path(path).name,
        bout_score_cm=lambda outcome, seat, bonus: -200. if outcome.how == "reset" else 125.,
    ))

    def bout(actual, west, east, seed, seat, condition, **kwargs):
        assert actual is manifest and kwargs["stress_seed"] == seed
        calls.append((west, east, seed, seat, condition))
        outcome = SimpleNamespace(loser=None if seed == 10 else 1-seat,
                                  how="reset" if seed == 10 else "pit", time_s=.4)
        return outcome, {"bad": np.array([np.nan, np.inf, -np.inf]),
                         "force_nonzero": np.bool_(condition in ("force", "both")),
                         "force_substantial": condition == "both"}

    monkeypatch.setattr(stress, "run_stress_bout", bout)
    target = tmp_path / "stress.json"
    report = evaluation.evaluate_stress_policy(Path("learner.onnx"), {"baseline": Path("old.onnx")},
                                              (9, 10), output=target)
    assert report == json.loads(target.read_text())
    assert report["checkpoint_selection"] is False and report["num_bouts"] == 8
    assert [c[-1] for c in calls] == ["nominal"]*2 + ["force"]*2 + ["spawn"]*2 + ["both"]*2
    assert [(c[0], c[1], c[3]) for c in calls] == [("learner.onnx", "old.onnx", 0), ("old.onnx", "learner.onnx", 1)]*4
    for condition, result in report["conditions"].items():
        assert result["fault_rate"] == .5 and result["mean_score_cm"] == -37.5
        assert result["opponents"]["baseline"]["resets"] == 1
        assert result["opponents"]["baseline"]["bouts"][0]["stress"]["bad"] == [None]*3
        assert result["force_nonzero_fraction"] == (1. if condition in ("force", "both") else 0.)
    with pytest.raises(ValueError, match="conditions"):
        evaluation.evaluate_stress_policy(Path("x"), {"a": Path("b")}, conditions=("invalid",))


@pytest.mark.parametrize("side", [0, 1])
def test_zero_force_parity_and_unilateral_valid_stress(monkeypatch, side):
    manifest = load_event("tug-of-war")
    model = compile_tug(manifest)
    duck = reference_duck(manifest)
    original_step, original_perturb = mujoco.mj_step, runner._perturb
    original_plan = stress._pulse_plan
    official = tug.run_bout(manifest, duck, duck, 19001, model=model)

    def zero_plan(*args):
        plan = original_plan(*args)
        for pulse in plan:
            pulse["peak_n"] = 0.
        return plan

    monkeypatch.setattr(stress, "_pulse_plan", zero_plan)
    disabled, _ = stress.run_stress_bout(manifest, duck, duck, 19001, side, "force", model=model)
    assert (official.how, official.loser, official.time_s) == (disabled.how, disabled.loser, disabled.time_s)
    for name in ("trace", "settle", "string", "string_settle"):
        assert np.array_equal(getattr(official, name), getattr(disabled, name))
    monkeypatch.setattr(stress, "_pulse_plan", original_plan)
    trunk = model.body(duck_prefix(side) + "trunk_base").id
    neck = model.site(duck_prefix(side) + STRING_SITE).id
    other = model.site(duck_prefix(1-side) + STRING_SITE).id
    exposure = []

    def spy(actual_model, data):
        force = data.xfrc_applied[trunk, :3]
        nonzero = np.flatnonzero(np.linalg.norm(data.xfrc_applied, axis=1) > 1e-12)
        if len(nonzero):
            assert nonzero.tolist() == [trunk] and data.time >= .5 and force[2] == 0.
            assert np.dot(force[:2], (data.site_xpos[other]-data.site_xpos[neck])[:2]) > 0.
            arm = data.site_xpos[neck] - data.xipos[trunk]
            assert np.allclose(data.xfrc_applied[trunk, 3:], np.cross(arm, force))
            exposure.append(data)
        return original_step(actual_model, data)

    monkeypatch.setattr(mujoco, "mj_step", spy)
    saved = {name: getattr(model, name).copy() for name in ("dof_damping", "dof_frictionloss")}
    outcome, diagnostics = stress.run_stress_bout(manifest, duck, duck, 19001, side, "both", model=model)
    assert exposure and not np.any(exposure[0].xfrc_applied)
    assert mujoco.mj_step is spy and runner._perturb is original_perturb
    assert outcome.how != "reset" and diagnostics["force_nonzero"]
    assert diagnostics["spawn"]["valid"]
    assert .98 <= diagnostics["spawn"]["neck_chord_m"] <= .995
    assert diagnostics["spawn"]["max_tie_error_m"] <= .01001
    assert np.array_equal(official.settle[1-side, 0, 3:], outcome.settle[1-side, 0, 3:])
    for pulse in diagnostics["planned_pulses"]:
        assert pulse["delivered_impulse_ns"] <= pulse["impulse_cap_ns"] + 1e-15
    for name, value in saved.items():
        assert np.array_equal(getattr(model, name), value)
    json.dumps(diagnostics, allow_nan=False)


def test_exception_restores_scoped_hooks_and_model(monkeypatch):
    manifest = load_event("tug-of-war")
    model = compile_tug(manifest)
    original_step, original_perturb = mujoco.mj_step, runner._perturb
    saved = model.dof_frictionloss.copy()

    def fail(*args, **kwargs):
        model.dof_frictionloss[:] = 5.
        raise RuntimeError("cleanup probe")

    monkeypatch.setattr(tug, "run_bout", fail)
    with pytest.raises(RuntimeError, match="cleanup probe"):
        stress.run_stress_bout(manifest, None, None, 9, 0, "both", model=model)
    assert mujoco.mj_step is original_step and runner._perturb is original_perturb
    assert np.array_equal(model.dof_frictionloss, saved)
