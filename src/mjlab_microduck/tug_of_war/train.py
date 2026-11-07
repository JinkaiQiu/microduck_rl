"""GPU tug self-play PPO, with the Arena leaderboard score as terminal reward."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from math import isfinite
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import uuid

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner


ACTOR_CFG = {
    "class_name": "MLPModel", "hidden_dims": (512, 256, 128),
    "activation": "elu", "obs_normalization": True,
    "distribution_cfg": {
        "class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar",
    },
}
CRITIC_CFG = {
    "class_name": "MLPModel", "hidden_dims": (512, 256, 128),
    "activation": "elu", "obs_normalization": True,
}
REWARD = "terminal_arena_leaderboard_score_cm / 100; no auxiliary rewards"


def train_config(iterations: int = 6000, seed: int = 1, *, from_scratch: bool = False,
                 rollout_steps: int = 24, save_interval: int = 200,
                 logger: str = "tensorboard") -> dict:
    actor = deepcopy(ACTOR_CFG)
    if not from_scratch:
        actor["class_name"] = "microduck_challenges.tug_of_war.opponent:ReferenceActor"
    return {
        "seed": seed, "num_steps_per_env": rollout_steps, "max_iterations": iterations,
        "save_interval": save_interval, "experiment_name": "tug_of_war_self_play",
        "run_name": f"seed-{seed}", "logger": logger, "wandb_project": "mjlab_microduck",
        "obs_groups": {"actor": ["actor"], "critic": ["critic"]},
        "actor": actor, "critic": deepcopy(CRITIC_CFG),
        "algorithm": {
            "class_name": "mjlab_microduck.football.learning:FootballPPO",
            "num_learning_epochs": 5, "num_mini_batches": 4, "learning_rate": 1e-3,
            "schedule": "adaptive", "gamma": 1.0, "lam": 0.95, "entropy_coef": 0.001,
            "desired_kl": 0.01, "max_grad_norm": 1.0, "value_loss_coef": 1.0,
            "use_clipped_value_loss": True, "clip_param": 0.2,
            "normalize_advantage_per_mini_batch": False, "optimizer": "adam",
            "teacher_loss_coef_start": 0.0, "teacher_loss_coef_end": 0.0,
            "rnd_cfg": None, "symmetry_cfg": None, "diagnostics": True,
        },
    }


class TugRunner(OnPolicyRunner):
    """Save PPO, immutable opponent snapshots, sampler state, and exact update counts.

    Resume restores learning state; physical bouts restart from their initial states.
    """

    def __init__(self, env, train_cfg: dict, log_dir: str, device: str = "cpu", *,
                 snapshot_interval: int = 200, initialization: str = "reference_actor",
                 evaluation_opponents: dict[str, Path] | None = None,
                 evaluation_seeds: tuple[int, ...] = (1001, 1002, 1003, 1004),
                 eval_interval: int = 200):
        if snapshot_interval < 1:
            raise ValueError("snapshot_interval must be positive")
        self.actor_cfg = deepcopy(train_cfg["actor"])
        self.actor_cfg["class_name"] = "MLPModel"
        self.snapshot_interval = snapshot_interval
        self.initialization = initialization
        self._log_dir = Path(log_dir)
        self.evaluation_opponents = dict(evaluation_opponents or {})
        self.evaluation_seeds = tuple(evaluation_seeds)
        self.eval_interval = eval_interval
        self.best_evaluation = (float("-inf"), float("-inf"))
        self.best_evaluation_metric = "anchors_v1"
        self.last_evaluation = None
        self.best_evaluation_context = None
        self.stress_evaluation = False
        self.stress_evaluation_seeds = (50001, 50002, 50003, 50004)
        super().__init__(env, deepcopy(train_cfg), log_dir, device)
        if self.alg.actor.obs_dim != 61 or env.num_actions != 14:
            raise ValueError("tug actor must preserve the Arena 61D observation to 14D action contract")
        self.alg.after_update = self._after_update

    def _after_update(self, completed_iterations: int) -> None:
        if hasattr(self.env, "set_training_update"):
            self.env.set_training_update(completed_iterations)
        evaluation = None
        if self.evaluation_opponents and self.eval_interval and completed_iterations % self.eval_interval == 0:
            evaluation = self.evaluate(completed_iterations)
        if completed_iterations % self.snapshot_interval == 0:
            if evaluation is None:
                self.env.add_snapshot(self.alg.actor, self.actor_cfg, completed_iterations)
            else:
                self.env.add_snapshot(self.alg.actor, self.actor_cfg, completed_iterations,
                                      scores_by_opponent={name: data["win_rate"]
                                                          for name, data in evaluation["opponents"].items()
                                                          if name in ("juicenv", "andoshy", "ctcheung", "reference")})
            self.save(str(self._log_dir / f"snapshot_{completed_iterations}.pt"))

    def evaluate(self, iteration: int) -> dict:
        from .evaluation import evaluate_policy
        from mjlab_microduck.football.pool import OpponentChoice

        directory = self._log_dir / "evaluations"
        policy = export_policy(self, directory / f"policy_{iteration:06d}.onnx")
        opponents = dict(self.evaluation_opponents)
        pool = self.env.pool
        history = pool.history if getattr(pool, "selection", None) == "challenger" else []
        identities = {}
        for item in history:
            uid = item["uid"]
            name = f"{item['group']}_{item['iteration']}_uid{uid}"
            target = directory / "frozen" / f"policy_{item['iteration']}_uid{uid}.onnx"
            if not target.is_file():
                # MLP construction must not advance the learner's CPU RNG stream.
                with torch.random.fork_rng(devices=[]):
                    actor = pool.model(OpponentChoice("joint", uid), "cpu")
                    export_policy(self, target, actor=actor, iteration=item["iteration"])
            opponents[name] = target
            identities[name] = {"uid": uid, "iteration": item["iteration"], "group": item["group"],
                                "protected": item.get("protected", False)}
        report = evaluate_policy(policy, opponents, self.evaluation_seeds)
        for name, identity in identities.items():
            bouts = report["opponents"][name]["bouts"]
            valid = [bout for bout in bouts if bout["how"] not in ("fault", "reset")]
            wins = sum(bout["winner"] == "learner" for bout in valid)
            losses = sum(bout["winner"] == "opponent" for bout in valid)
            invalid = len(bouts) - len(valid)
            pool.set_challenger_scores(identity["uid"], wins, losses, invalid)
            identity.update(wins=wins, losses=losses, draws=len(valid) - wins - losses, invalid=invalid)
        report["historical_opponents"] = identities
        protected = {name: identity for name, identity in identities.items() if identity["protected"]}
        if protected:
            report["protected_historical_opponents"] = protected
        report["iteration"] = iteration
        report["policy_sha256"] = hashlib.sha256(policy.read_bytes()).hexdigest()
        for data in report["opponents"].values():
            data["policy_sha256"] = hashlib.sha256(Path(data["policy"]).read_bytes()).hexdigest()
        selection, metric, fields = evaluation_selection(report)
        report["checkpoint_selection"] = {"metric": metric, "fields": fields, "value": selection}
        context = {"seeds": list(self.evaluation_seeds),
                   "fixed_policy_sha256": {name: data["policy_sha256"] for name, data in report["opponents"].items()
                                           if name not in identities or name in protected}}
        if metric != self.best_evaluation_metric or (self.best_evaluation_context is not None
                                                     and context != self.best_evaluation_context):
            self.best_evaluation = tuple(float("-inf") for _ in selection)
            self.best_evaluation_metric = metric
        self.best_evaluation_context = context
        self.last_evaluation = report
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"evaluation_{iteration:06d}.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if selection > self.best_evaluation:
            self.best_evaluation = selection
            self.save(str(self._log_dir / "model_best.pt"), {"evaluation": report})
        writer = getattr(self.logger, "writer", None)
        if writer is not None:
            writer.add_scalar("Evaluation/crossplay_win_rate", selection[0], iteration)
            writer.add_scalar("Evaluation/fault_rate", report["fault_rate"], iteration)
            for name, data in report["opponents"].items():
                writer.add_scalar(f"Evaluation/{name}_win_rate", data["win_rate"], iteration)
                writer.add_scalar(f"Evaluation/{name}_score_cm", data["mean_score_cm"], iteration)
        print(f"[tug evaluation] update={iteration} selection={metric}:{selection} "
              f"mean_score_cm={report['mean_score_cm']:.2f} fault_rate={report['fault_rate']:.3f}", flush=True)
        if getattr(self, "stress_evaluation", False):
            from .evaluation import evaluate_stress_policy

            names = {"baseline_6000", "baseline_8000"}
            names.update(name for name, identity in protected.items() if identity["iteration"] in (5400, 5800, 7400, 8800))
            stress = evaluate_stress_policy(policy, {name: path for name, path in opponents.items() if name in names},
                                            self.stress_evaluation_seeds,
                                            conditions=("nominal", "force", "spawn", "both"))
            stress.update(iteration=iteration, policy_sha256=report["policy_sha256"])
            (directory / f"stress_evaluation_{iteration:06d}.json").write_text(
                json.dumps(stress, indent=2, allow_nan=False) + "\n")
            if writer is not None:
                for condition, data in stress["conditions"].items():
                    writer.add_scalar(f"StressEvaluation/{condition}_win_rate", data["mean_win_rate"], iteration)
                    writer.add_scalar(f"StressEvaluation/{condition}_score_cm", data["mean_score_cm"], iteration)
                    writer.add_scalar(f"StressEvaluation/{condition}_fault_rate", data["fault_rate"], iteration)
            print(f"[tug stress evaluation] update={iteration} "
                  + " ".join(f"{name}={data['mean_win_rate']:.3f}" for name, data in stress["conditions"].items()), flush=True)
        return report

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if init_at_random_ep_len:
            raise ValueError("tug bouts must start at the official reset state")
        super().learn(num_learning_iterations, init_at_random_ep_len=False)
        self.current_learning_iteration = int(self.alg.update_count)

    def save(self, path: str, infos: dict | None = None) -> None:
        completed = int(self.alg.update_count)
        state = self.alg.save()
        state.update({"iter": completed, "infos": infos})
        state["tug_of_war"] = {
            "schema": 1, "actor_cfg": deepcopy(self.actor_cfg), "reward": REWARD,
            "initialization": self.initialization, "snapshot_interval": self.snapshot_interval,
            "learning_rate": float(self.alg.learning_rate), "learning_rate_schedule": self.alg.schedule,
            "entropy_coef": float(self.alg.entropy_coef),
            "evaluation_opponents": {name: str(path) for name, path in self.evaluation_opponents.items()},
            "evaluation_seeds": self.evaluation_seeds, "eval_interval": self.eval_interval,
            "best_evaluation": self.best_evaluation,
            "best_evaluation_metric": self.best_evaluation_metric, "last_evaluation": self.last_evaluation,
            "best_evaluation_context": self.best_evaluation_context,
            "stress_evaluation": self.stress_evaluation,
            "stress_evaluation_seeds": self.stress_evaluation_seeds,
            "env_state": self.env.state_dict(),
            "common_step_counter": int(self.env.unwrapped.common_step_counter),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, target)
        if getattr(self.logger, "writer", None) is not None:
            self.logger.save_model(str(target), completed)

    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True,
             map_location: str | None = None) -> dict:
        if load_cfg is not None:
            raise ValueError("tug resume requires the complete optimizer, counters, and opponent pool")
        state = torch.load(path, map_location=map_location or self.device, weights_only=False)
        meta = state.get("tug_of_war", {})
        if meta.get("schema") != 1 or meta.get("reward") != REWARD:
            raise ValueError("checkpoint is not from this tug self-play reward configuration")
        if meta["actor_cfg"] != self.actor_cfg:
            raise ValueError("actor configuration differs from the saved tug run")
        self.alg.load(state, None, strict)
        self.alg.learning_rate = float(meta["learning_rate"])
        self.alg.schedule = meta.get("learning_rate_schedule", "adaptive")
        self.alg.entropy_coef = float(meta.get("entropy_coef", 0.001))
        self.current_learning_iteration = int(state["iter"])
        if self.current_learning_iteration != int(self.alg.update_count):
            raise ValueError("checkpoint iteration and PPO update count disagree")
        self.initialization = meta["initialization"]
        self.snapshot_interval = int(meta["snapshot_interval"])
        self.evaluation_opponents = {name: Path(path) for name, path in meta.get("evaluation_opponents", {}).items()}
        self.evaluation_seeds = tuple(meta.get("evaluation_seeds", self.evaluation_seeds))
        self.eval_interval = int(meta.get("eval_interval", self.eval_interval))
        self.best_evaluation = tuple(meta.get("best_evaluation", self.best_evaluation))
        self.best_evaluation_metric = meta.get("best_evaluation_metric", "anchors_v1")
        self.last_evaluation = meta.get("last_evaluation")
        self.best_evaluation_context = meta.get("best_evaluation_context")
        self.stress_evaluation = meta.get("stress_evaluation", False)
        self.stress_evaluation_seeds = tuple(meta.get("stress_evaluation_seeds", self.stress_evaluation_seeds))
        # Loading the wrapper starts new physical bouts, whose reset jitter
        # must use the saved RNG streams rather than construction-time draws.
        torch.set_rng_state(meta["torch_rng"].cpu())
        if meta["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in meta["cuda_rng"]])
        random.setstate(meta["python_rng"])
        np.random.set_state(meta["numpy_rng"])
        if hasattr(self.env, "set_training_update"):
            self.env.set_training_update(self.current_learning_iteration)
        self.env.load_state_dict(meta["env_state"])
        self.env.unwrapped.common_step_counter = int(meta["common_step_counter"])
        if hasattr(self.env, "set_training_update"):
            self.env.set_training_update(self.current_learning_iteration)
        return state.get("infos") or {}


def evaluation_selection(report: dict) -> tuple[tuple[float, ...], str, list[str]]:
    """Fault-inclusive wins; give cup, fixed baseline and history equal group weight."""
    results = report["opponents"]
    history = report.get("historical_opponents", {})
    protected = report.get("protected_historical_opponents", {})
    if protected and "baseline_8000" in results:
        cup = [data for name, data in results.items() if name.startswith("cup_")]
        baselines = [results[name] for name in ("baseline_6000", "baseline_8000") if name in results]
        historical = [results[name] for name in protected]
        groups = [group for group in (cup, baselines, historical) if group]
        score = sum(sum(data["win_rate"] for data in group) / len(group) for group in groups) / len(groups)
        fixed = [data for group in groups for data in group]
        return ((score, sum(data["win_rate"] for data in baselines) / len(baselines),
                 -sum(data["fault_rate"] for data in fixed) / len(fixed),
                 sum(data["mean_score_cm"] for data in fixed) / len(fixed)),
                "cup_baselines_protected_v2",
                ["cup_baselines_protected_group_mean_win_rate", "baselines_mean_win_rate", "negative_fixed_fault_rate", "fixed_mean_score_cm"])
    baseline = results.get("baseline_6000")
    groups = []
    cup = [data["win_rate"] for name, data in results.items() if name.startswith("cup_")]
    if cup:
        groups.append(sum(cup) / len(cup))
    if baseline is not None:
        groups.append(baseline["win_rate"])
    if history:
        groups.append(sum(results[name]["win_rate"] for name in history) / len(history))
    if groups:
        return ((sum(groups) / len(groups), baseline["win_rate"] if baseline else groups[0],
                 -report["fault_rate"], report["mean_score_cm"]), "cup_baseline_history_v1",
                ["cup_baseline_history_group_mean_win_rate", "baseline_win_rate", "negative_fault_rate", "mean_score_cm"])
    competitors = [data for name, data in results.items() if name != "reference"]
    win_rate = (sum(data["win_rate"] for data in competitors) / len(competitors)
                if competitors else report["mean_win_rate"])
    return (win_rate, report["mean_score_cm"]), "anchors_v1", ["crossplay_win_rate", "mean_score_cm"]


def apply_continuation(runner: TugRunner, *, league_config: dict | None = None,
                       entropy_coef: float | None = None, resume_learning_rate: float | None = None,
                       randomization_config: dict | None = None,
                       evaluation_baseline: Path | None = None,
                       evaluation_baseline_8000: Path | None = None,
                       evaluation_roster: dict[str, Path] | None = None) -> dict:
    """Apply explicit changes only after the complete checkpoint state is loaded."""
    learning_rate_changes = {}
    if resume_learning_rate is not None:
        if not isfinite(resume_learning_rate) or resume_learning_rate <= 0:
            raise ValueError("resume learning rate must be finite and positive")
        learning_rate_changes = {"old_learning_rate": float(runner.alg.learning_rate),
                                 "old_learning_rate_schedule": runner.alg.schedule,
                                 "learning_rate": resume_learning_rate, "learning_rate_schedule": "fixed"}
        runner.alg.learning_rate = resume_learning_rate
        runner.alg.schedule = "fixed"
        for group in runner.alg.optimizer.param_groups:
            group["lr"] = resume_learning_rate * group.get("skill_lr_scale", 1.0)
    old_mix = deepcopy(getattr(runner.env.pool, "weights", None))
    old_entropy = float(runner.alg.entropy_coef)
    if entropy_coef is not None:
        if not isfinite(entropy_coef) or entropy_coef < 0:
            raise ValueError("entropy coefficient must be finite and nonnegative")
        runner.alg.entropy_coef = entropy_coef
    if league_config is not None:
        if not hasattr(runner.env.pool, "configure"):
            raise ValueError("--league-config requires a tug league checkpoint or external policies")
        runner.env.pool.configure(league_config)
        runner.env.live_fraction = runner.env.pool.weights["live"]
        runner.env.reference_fraction = runner.env.pool.weights["reference"]
    if randomization_config is not None:
        randomization = getattr(runner.env.arena, "randomization", None)
        if randomization is None:
            raise ValueError("randomization continuation requires a randomized tug environment")
        randomization.configure_mix(randomization_config)
    if league_config is not None or randomization_config is not None:
        runner.env.reset()
    if evaluation_baseline is not None:
        runner.evaluation_opponents["baseline_6000"] = evaluation_baseline.resolve()
    if evaluation_baseline_8000 is not None:
        runner.evaluation_opponents["baseline_8000"] = evaluation_baseline_8000.resolve()
    if evaluation_roster is not None:
        runner.evaluation_opponents = {name: path for name, path in runner.evaluation_opponents.items()
                                       if not name.startswith("cup_")}
        runner.evaluation_opponents.update(evaluation_roster)
    return {**learning_rate_changes, "old_opponent_mix": old_mix, "opponent_mix": deepcopy(getattr(runner.env.pool, "weights", None)),
            "old_entropy_coefficient": old_entropy, "entropy_coefficient": float(runner.alg.entropy_coef)}


def export_policy(runner: TugRunner, path: Path, *, actor=None, iteration: int | None = None) -> Path:
    """Use RSL's exporter so the actor normalizer is baked into the 61→14 graph."""
    import onnx

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if actor is None:
        runner.export_policy_to_onnx(str(path.parent), path.name)
    else:
        frozen = SimpleNamespace(alg=SimpleNamespace(get_policy=lambda: actor))
        OnPolicyRunner.export_policy_to_onnx(frozen, str(path.parent), path.name)
    model = onnx.load(path)
    def shape(value):
        return [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
    if len(model.graph.input) != 1 or shape(model.graph.input[0]) != [1, 61]:
        raise ValueError("exported tug actor must have exactly one (1, 61) input")
    if len(model.graph.output) != 1 or shape(model.graph.output[0]) != [1, 14]:
        raise ValueError("exported tug actor must have exactly one (1, 14) output")
    onnx.helper.set_model_props(model, {
        "task": "tug-of-war", "reward": REWARD,
        "training_iterations": str(runner.alg.update_count if iteration is None else iteration), "normalization": "baked_in",
    })
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1024, help="physical two-duck matches (2N policy rows)")
    parser.add_argument("--iterations", type=int, default=6000, help="additional PPO updates")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--entropy-coef", type=float, help="override entropy; default .001 fresh, saved value on resume")
    parser.add_argument("--league-config", type=Path, help="JSON league configuration applied after full checkpoint load")
    parser.add_argument("--randomization-config", type=Path, help="JSON force/spawn curriculum; saved in resumable environment state")
    parser.add_argument("--stress-eval-seeds", nargs="+", type=int, help="fixed seeds for separate force/spawn evaluation")
    parser.add_argument("--evaluation-baseline", type=Path, help="fixed update-6000 ONNX regression opponent")
    parser.add_argument("--evaluation-baseline-8000", type=Path, help="fixed update-8000 ONNX regression opponent")
    parser.add_argument("--evaluation-roster", type=Path, help="JSON mapping cup opponent names to fixed ONNX paths")
    parser.add_argument("--from-scratch", action="store_true", help="fresh actor instead of the reference warm start")
    parser.add_argument("--init-policy", type=Path, help="plain ONNX actor warm start; critic/optimizer/counters stay fresh")
    parser.add_argument("--action-std", type=float, default=0.3)
    parser.add_argument("--normalizer-count", type=int, default=100_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="fresh-run learning rate; resume restores checkpoint rate")
    parser.add_argument("--resume-learning-rate", type=float,
                        help="override restored learning rate and use a fixed schedule (requires --resume)")
    for name in ("juicenv", "andoshy", "ctcheung"):
        parser.add_argument(f"--{name}-policy", type=Path)
    parser.add_argument("--eval-interval", type=int, help="CPU cross-play cadence; default 200 fresh, saved on resume; zero disables")
    parser.add_argument("--eval-seeds", type=int, nargs="+", help="default 1001..1004 fresh, saved on resume")
    parser.add_argument("--evaluate-on-start", action="store_true", help="evaluate restored actor before learning")
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--snapshot-interval", type=int, default=200)
    parser.add_argument("--pool-capacity", type=int, default=8)
    parser.add_argument("--live-fraction", type=float, default=0.5)
    parser.add_argument("--reference-fraction", type=float, default=0.25)
    parser.add_argument("--logger", choices=("tensorboard", "wandb"), default="tensorboard")
    return parser


def run(args: argparse.Namespace) -> Path:
    for name in ("num_envs", "iterations", "rollout_steps", "save_interval", "snapshot_interval", "pool_capacity"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not (0 <= args.live_fraction <= 1 and 0 <= args.reference_fraction <= 1
            and args.live_fraction + args.reference_fraction <= 1):
        raise ValueError("live/reference fractions must be nonnegative and sum to at most one")
    if args.resume_learning_rate is not None:
        if not args.resume:
            raise ValueError("--resume-learning-rate requires --resume")
        if not isfinite(args.resume_learning_rate) or args.resume_learning_rate <= 0:
            raise ValueError("resume learning rate must be finite and positive")
    if args.resume and args.from_scratch:
        raise ValueError("--resume and --from-scratch are mutually exclusive")
    if args.init_policy and (args.resume or args.from_scratch):
        raise ValueError("--init-policy requires a fresh warm start, without --resume or --from-scratch")
    if args.action_std <= 0 or args.normalizer_count < 1 or args.learning_rate <= 0 or (args.eval_interval is not None and args.eval_interval < 0):
        raise ValueError("invalid learning rate, exploration, normalizer count, or evaluation interval")
    if args.entropy_coef is not None and (not isfinite(args.entropy_coef) or args.entropy_coef < 0):
        raise ValueError("entropy coefficient must be finite and nonnegative")
    league_config = json.loads(args.league_config.read_text()) if args.league_config else None
    randomization_config = json.loads(args.randomization_config.read_text()) if args.randomization_config else None
    evaluation_roster = None
    if args.evaluation_roster:
        raw_roster = json.loads(args.evaluation_roster.read_text())
        if not isinstance(raw_roster, dict) or not raw_roster or any(not isinstance(name, str) or not name for name in raw_roster):
            raise ValueError("evaluation roster must map nonempty opponent names to ONNX paths")
        evaluation_roster = {(name if name.startswith("cup_") else f"cup_{name}"): Path(path).resolve()
                             for name, path in raw_roster.items()}
        if len(evaluation_roster) != len(raw_roster):
            raise ValueError("evaluation roster names collide after cup_ prefixing")
    external = {name: getattr(args, f"{name}_policy") for name in ("juicenv", "andoshy", "ctcheung")}
    if any(external.values()) and not all(external.values()):
        raise ValueError("the agreed league requires all three external policies")
    external = {name: path.resolve() for name, path in external.items() if path is not None}
    for path in [*external.values(), *([args.init_policy] if args.init_policy else []),
                 *([args.evaluation_baseline] if args.evaluation_baseline else []),
                 *([args.evaluation_baseline_8000] if args.evaluation_baseline_8000 else []),
                 *(evaluation_roster or {}).values()]:
        if not path.is_file():
            raise FileNotFoundError(path)
    # Optional challenge and simulator dependencies are intentionally loaded after argument parsing.
    from mjlab.envs import ManagerBasedRlEnv
    from microduck_challenges.tug_of_war.opponent import REFERENCE
    from mjlab_microduck.football.pool import OpponentPool
    from mjlab_microduck.tug_of_war.arena import TugArena, make_tug_env_cfg
    from mjlab_microduck.tug_of_war.wrapper import TugVecEnv

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    run_dir = (args.run_dir or (args.resume.resolve().parent if args.resume else
                               Path("logs/tug_of_war_self_play") / f"{stamp}_seed{args.seed}")).resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise ValueError(f"new training run directory must be empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cfg = train_config(args.iterations, args.seed, from_scratch=args.from_scratch or bool(args.resume) or bool(args.init_policy),
                       rollout_steps=args.rollout_steps, save_interval=args.save_interval, logger=args.logger)
    cfg["algorithm"]["learning_rate"] = args.learning_rate
    if args.entropy_coef is not None:
        cfg["algorithm"]["entropy_coef"] = args.entropy_coef
    saved = saved_randomization = None
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved_randomization = saved["tug_of_war"]["env_state"].get("randomization")
        if randomization_config is None and saved_randomization is not None:
            randomization_config = saved_randomization["config"]
        elif saved_randomization is not None and randomization_config != saved_randomization["config"]:
            without_mix = lambda config: {key: value for key, value in config.items() if key != "mix"}
            if without_mix(randomization_config) != without_mix(saved_randomization["config"]):
                raise ValueError("resuming a randomized run may change only the randomization mix")
        cfg["actor"] = deepcopy(saved["tug_of_war"]["actor_cfg"])
        cfg["algorithm"]["learning_rate"] = (args.resume_learning_rate if args.resume_learning_rate is not None
                                                else saved["tug_of_war"]["learning_rate"])
        cfg["algorithm"]["schedule"] = ("fixed" if args.resume_learning_rate is not None
                                          else saved["tug_of_war"].get("learning_rate_schedule", "adaptive"))
        cfg["algorithm"]["entropy_coef"] = (args.entropy_coef if args.entropy_coef is not None else
                                                 saved["tug_of_war"].get("entropy_coef", .001))
    tag = f"_resume_{stamp}" if args.resume else ""
    (run_dir / f"config{tag}.json").write_text(json.dumps(cfg, indent=2) + "\n")
    sources = sorted(Path(__file__).parent.glob("*.py")) + [REFERENCE, *external.values()]
    if args.init_policy:
        sources.append(args.init_policy.resolve())
    sources.extend(path.resolve() for path in (args.league_config, args.randomization_config, args.evaluation_baseline,
                                              args.evaluation_baseline_8000, args.evaluation_roster) if path)
    sources.extend((evaluation_roster or {}).values())
    sources.extend(path.with_suffix(".onnx.data") for path in list(sources)
                   if path.suffix == ".onnx" and path.with_suffix(".onnx.data").is_file())
    receipt = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "starting",
        "argv": sys.argv, "args": {key: str(value) if isinstance(value, Path) else value
                                    for key, value in vars(args).items()},
        "run_dir": str(run_dir), "reward": REWARD, "gamma": 1.0,
        "initialization": "resume" if args.resume else "scratch_actor_fresh_critic_optimizer_counters"
        if args.from_scratch else "onnx_actor_and_normalizer_fresh_critic_optimizer_counters"
        if args.init_policy else "reference_actor_and_normalizer_fresh_critic_optimizer_counters",
        "resume_semantics": "restore learning/pool/sampler state; restart physical bouts",
        "source_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "diagnostics_dir": str(run_dir / "diagnostics" / stamp),
        "resume_checkpoint_sha256": hashlib.sha256(args.resume.read_bytes()).hexdigest() if args.resume else None,
        "league_config": league_config,
        "randomization_config": randomization_config,
        "initial_iteration": int(saved["iter"]) if saved else 0,
        "target_iteration": (int(saved["iter"]) if saved else 0) + args.iterations,
    }
    receipt_path = run_dir / f"receipt{tag}.json"
    def write_receipt():
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    write_receipt()
    env = base = None
    try:
        base = ManagerBasedRlEnv(cfg=make_tug_env_cfg(args.num_envs, seed=args.seed), device=args.device)
        saved_pool = saved["tug_of_war"]["env_state"]["pool"] if saved else {}
        if external or saved_pool.get("league_version"):
            from .league import TugLeague

            pool = (TugLeague.from_state(saved_pool) if saved_pool.get("league_version") else
                    TugLeague(external, seed=args.seed, recent_capacity=4, archive_capacity=4))
            receipt["opponent_mix"] = dict(pool.weights)
            receipt["external_provenance"] = dict(pool.external_provenance)
        else:
            pool = OpponentPool(seed=args.seed, capacity=args.pool_capacity)
        arena = TugArena(base, diagnostics_dir=receipt["diagnostics_dir"],
                         randomization_config=saved_randomization["config"] if saved_randomization else randomization_config)
        arena.diagnostics.context.update(receipt=str(receipt_path), resume_checkpoint=str(args.resume) if args.resume else None)
        env = TugVecEnv(arena, pool, seed=args.seed,
                        live_fraction=pool.weights["live"] if hasattr(pool, "weights") else args.live_fraction,
                        reference_fraction=pool.weights["reference"] if hasattr(pool, "weights") else args.reference_fraction)
        runner = TugRunner(env, cfg, str(run_dir), args.device, snapshot_interval=args.snapshot_interval,
                           initialization=receipt["initialization"],
                           evaluation_opponents={**external, "reference": REFERENCE} if external else None,
                           evaluation_seeds=tuple(args.eval_seeds or (1001, 1002, 1003, 1004)),
                           eval_interval=args.eval_interval if args.eval_interval is not None else 200)
        if args.resume:
            runner.load(str(args.resume))
        receipt.update(apply_continuation(runner, league_config=league_config,
                                          entropy_coef=args.entropy_coef,
                                          resume_learning_rate=args.resume_learning_rate,
                                          randomization_config=randomization_config,
                                          evaluation_baseline=args.evaluation_baseline,
                                          evaluation_baseline_8000=args.evaluation_baseline_8000,
                                          evaluation_roster=evaluation_roster))
        runner.stress_evaluation = randomization_config is not None
        if args.stress_eval_seeds is not None:
            runner.stress_evaluation_seeds = tuple(args.stress_eval_seeds)
        if args.eval_seeds is not None:
            runner.evaluation_seeds = tuple(args.eval_seeds)
        if args.eval_interval is not None:
            runner.eval_interval = args.eval_interval
        if args.resume:
            runner.save(str(run_dir / "model_initial_resume.pt"))
        if not args.resume:
            if args.init_policy:
                from .initialization import initialize_actor

                receipt["actor_initialization"] = initialize_actor(
                    runner.alg.actor, args.init_policy.resolve(),
                    observation_count=args.normalizer_count, action_std=args.action_std,
                )
            if external and runner.eval_interval:
                runner.evaluate(0)
            if not external:
                env.add_snapshot(runner.alg.actor, runner.actor_cfg, 0)
            env.reset()
            runner.save(str(run_dir / "model_initial.pt"))
        if args.evaluate_on_start and (args.resume or not external or not runner.eval_interval):
            runner.evaluate(int(runner.alg.update_count))
        distribution = runner.alg.actor.distribution
        initial_std = distribution.std_param if distribution.std_type == "scalar" else distribution.log_std_param.exp()
        receipt.update(status="running", initial_iteration=int(runner.alg.update_count),
                       evaluation_opponents={name: str(path) for name, path in runner.evaluation_opponents.items()},
                       evaluation_seeds=runner.evaluation_seeds, eval_interval=runner.eval_interval,
                       checkpoint_selection_metric=runner.best_evaluation_metric,
                       initial_learning_rate=float(runner.alg.learning_rate),
                       learning_rate_schedule=runner.alg.schedule,
                       initial_common_step_counter=int(env.unwrapped.common_step_counter),
                       initial_optimizer_state_entries=len(runner.alg.optimizer.state),
                       entropy_coefficient=float(runner.alg.entropy_coef),
                       initial_action_std=float(initial_std.detach().mean()),
                       initial_critic_normalizer_count=float(runner.alg.critic.obs_normalizer.count),
                       initial_actor_normalizer_count=float(runner.alg.actor.obs_normalizer.count))
        write_receipt()
        runner.learn(args.iterations, init_at_random_ep_len=False)
        final = run_dir / "model_final.pt"
        runner.save(str(final))
        policy = export_policy(runner, run_dir / "policy.onnx")
        receipt.update(status="complete", completed_iterations=int(runner.alg.update_count),
                       checkpoint=str(final), policy=str(policy),
                       common_step_counter=int(env.unwrapped.common_step_counter),
                       recent_scores_cm=list(env.recent_scores),
                       best_evaluation=runner.best_evaluation, best_evaluation_metric=runner.best_evaluation_metric)
    except BaseException as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_receipt()
        if env is not None:
            env.close()
        elif base is not None:
            base.close()
    print(f"Tug run: {run_dir}\nReceipt: {receipt_path}")
    return run_dir


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
