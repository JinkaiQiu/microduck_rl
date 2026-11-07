"""Held-out CPU cross-play using the official Arena tug judge and scoring."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


def _arena_api():
    # Training can use the GPU environment without importing the optional CPU runner.
    try:
        from arena.course import compile_tug
        from arena.manifest import load_event
        from arena.policy import load_policy
        from arena.tug import bout_score_cm, run_bout
    except ImportError as exc:
        raise RuntimeError(
            "CPU tug evaluation requires microduck-arena 0.7.0 or newer with arena.tug installed"
        ) from exc
    return SimpleNamespace(
        compile_tug=compile_tug, load_event=load_event, load_policy=load_policy,
        bout_score_cm=bout_score_cm, run_bout=run_bout,
    )


def _summarize(bouts: list[dict]) -> dict:
    count = len(bouts)
    wins = sum(bout["winner"] == "learner" for bout in bouts)
    losses = sum(bout["winner"] == "opponent" for bout in bouts)
    faults = sum(bout["how"] in ("fault", "reset") for bout in bouts)
    return {
        "num_bouts": count, "wins": wins, "losses": losses, "draws": count - wins - losses,
        "win_rate": wins / count, "mean_score_cm": sum(bout["score_cm"] for bout in bouts) / count,
        "faults": faults, "fault_rate": faults / count,
        "resets": sum(bout["how"] == "reset" for bout in bouts), "bouts": bouts,
    }


def evaluate_policy(policy_path: Path, opponents: dict[str, Path],
                    seeds: tuple[int, ...] = (1001, 1002, 1003, 1004),
                    output: Path | None = None) -> dict:
    """Evaluate exported ONNX policies, alternating sides as in an official cup.

    Win rates count draws and shared physics resets as zero wins. Rating scores
    independently apply Arena's fault/reset penalty, including an opponent fault.
    Opponents receive equal weight in the aggregate checkpoint-selection metric.
    """
    if not opponents or not seeds:
        raise ValueError("cross-play requires at least one opponent and one seed")
    api = _arena_api()
    manifest = api.load_event("tug-of-war")
    model = api.compile_tug(manifest)
    learner = api.load_policy(Path(policy_path))
    results = {}
    for name, opponent_path in opponents.items():
        opponent = api.load_policy(Path(opponent_path))
        bouts = []
        for index, seed in enumerate(seeds):
            seat = index % 2
            west, east = (learner, opponent) if seat == 0 else (opponent, learner)
            outcome = api.run_bout(manifest, west, east, seed, model=model)
            winner = None if outcome.loser is None else (
                "opponent" if outcome.loser == seat else "learner"
            )
            bouts.append({
                "seed": int(seed), "learner_side": "west" if seat == 0 else "east",
                "winner": winner, "how": outcome.how, "time_s": float(outcome.time_s),
                "score_cm": float(api.bout_score_cm(outcome, seat, manifest.rating.pit_bonus_cm)),
            })
        results[name] = {"policy": str(Path(opponent_path).resolve()), **_summarize(bouts)}
    count = len(results)
    report = {
        "schema": 1, "policy": str(Path(policy_path).resolve()), "seeds": list(seeds),
        "num_bouts": sum(item["num_bouts"] for item in results.values()),
        "mean_win_rate": sum(item["win_rate"] for item in results.values()) / count,
        "mean_score_cm": sum(item["mean_score_cm"] for item in results.values()) / count,
        "fault_rate": sum(item["fault_rate"] for item in results.values()) / count,
        "opponents": results,
    }
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(target)
    return report


def evaluate_stress_policy(policy_path: Path, opponents: dict[str, Path],
                           seeds: tuple[int, ...] = (19001, 19002, 19003, 19004),
                           output: Path | None = None, *,
                           conditions: tuple[str, ...] = ("nominal", "force", "spawn", "both")) -> dict:
    """Separate adverse CPU probes; not part of nominal checkpoint selection.

    The learner alone receives adverse force/spawn changes. Each condition
    reuses the same seeds and alternating seats, and force/both share pulse
    plans. Exposure reports distinguish short bouts from delivered stress.
    """
    if not opponents or not seeds:
        raise ValueError("stress cross-play requires at least one opponent and one seed")
    allowed = {"nominal", "force", "spawn", "both"}
    if not conditions or len(set(conditions)) != len(conditions) or set(conditions) - allowed:
        raise ValueError("stress conditions must be unique nominal/force/spawn/both values")
    api = _arena_api()
    from .stress import run_stress_bout, sanitize_diagnostics

    manifest = api.load_event("tug-of-war")
    model = api.compile_tug(manifest)
    learner = api.load_policy(Path(policy_path))
    loaded = {name: api.load_policy(Path(path)) for name, path in opponents.items()}
    condition_results = {}
    for condition in conditions:
        results = {}
        for name, opponent in loaded.items():
            bouts = []
            for index, seed in enumerate(seeds):
                seat = index % 2
                west, east = (learner, opponent) if seat == 0 else (opponent, learner)
                outcome, diagnostics = run_stress_bout(
                    manifest, west, east, seed, seat, condition, model=model, stress_seed=seed,
                )
                winner = None if outcome.loser is None else (
                    "opponent" if outcome.loser == seat else "learner"
                )
                bouts.append({
                    "seed": int(seed), "learner_side": ("west", "east")[seat],
                    "winner": winner, "how": outcome.how, "time_s": float(outcome.time_s),
                    "score_cm": float(api.bout_score_cm(outcome, seat, manifest.rating.pit_bonus_cm)),
                    "stress": sanitize_diagnostics(diagnostics),
                })
            results[name] = {"policy": str(Path(opponents[name]).resolve()), **_summarize(bouts)}
        all_bouts = [bout for data in results.values() for bout in data["bouts"]]
        count = len(results)
        condition_results[condition] = {
            "num_bouts": len(all_bouts),
            "mean_win_rate": sum(data["win_rate"] for data in results.values()) / count,
            "mean_score_cm": sum(data["mean_score_cm"] for data in results.values()) / count,
            "fault_rate": sum(data["fault_rate"] for data in results.values()) / count,
            "invalid_initializations": sum(bool(bout["stress"].get("invalid_initialization")) for bout in all_bouts),
            "force_nonzero_fraction": sum(bool(bout["stress"].get("force_nonzero")) for bout in all_bouts) / len(all_bouts),
            "force_substantial_fraction": sum(bool(bout["stress"].get("force_substantial")) for bout in all_bouts) / len(all_bouts),
            "substantial_force_threshold_delta_v_m_s": .05,
            "opponents": results,
        }
    report = {
        "schema": 1, "policy": str(Path(policy_path).resolve()), "seeds": list(seeds),
        "protocol": "adverse_cpu_v1", "checkpoint_selection": False,
        "protocol_parameters": {
            "force_weight_fraction": [.10, .20], "pulse_duration_s": [.10, .20],
            "first_pulse_onset_s": [0., .04], "pulses_per_bout": [2, 3],
            "per_pulse_impulse_cap_delta_v_m_s": .15,
            "learner_yaw_magnitude_deg": [4., 8.], "learner_joint_noise_rad": .03,
            "learner_lateral_shift_m": .03, "neck_chord_m": [.98, .995],
            "settle_s": float(manifest.start.settle_s),
            "bout_limit_s": float(manifest.bout.time_limit_s),
        },
        "num_bouts": sum(data["num_bouts"] for data in condition_results.values()),
        "conditions": condition_results,
    }
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(target)
    return report
