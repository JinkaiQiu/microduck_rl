"""Tug training contracts without starting the simulator."""

from collections import deque
from copy import deepcopy
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pytest
import torch
from tensordict import TensorDict

from mjlab_microduck.football.pool import OpponentPool
from mjlab_microduck.tug_of_war.train import (
    ACTOR_CFG, REWARD, TugRunner, build_parser, export_policy, train_config,
)


class TinyTug:
    """Four policy rows, including a frozen seat, for a real masked PPO update."""

    num_envs = 4
    num_actions = 14
    device = "cpu"
    max_episode_length = 2
    cfg = {}

    def __init__(self):
        self.unwrapped = self
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.pool = OpponentPool(seed=3, capacity=2)
        self.recent_scores = deque(maxlen=10)

    def get_observations(self):
        actor = torch.zeros(self.num_envs, 61)
        actor[:3, 0] = self.common_step_counter
        actor[3] = 1000.0  # Frozen row must never affect normalizer statistics.
        return TensorDict({"actor": actor, "critic": torch.cat((actor, actor), dim=-1)}, [self.num_envs])

    def step(self, actions):
        assert actions.shape == (4, 14)
        self.common_step_counter += 1
        self.episode_length_buf += 1
        done = self.episode_length_buf >= self.max_episode_length
        reward = done.float() * torch.tensor([1.0, -1.0, 0.25, -0.25])
        self.episode_length_buf[done] = 0
        if done.any():
            self.recent_scores.append(100.0)
        return self.get_observations(), reward, done, {
            "learner_mask": torch.tensor([True, True, True, False]),
        }

    def add_snapshot(self, actor, cfg, iteration):
        return self.pool.add("joint", actor, cfg, iteration)

    def state_dict(self):
        return {"pool": self.pool.state_dict(), "scores": list(self.recent_scores)}

    def load_state_dict(self, state):
        self.pool.load_state_dict(state["pool"])
        self.recent_scores.extend(state["scores"])
        self.reset_draws = (torch.rand(1), random.random(), np.random.random())


def tiny_config():
    cfg = train_config(1, from_scratch=True, rollout_steps=2)
    cfg["actor"]["hidden_dims"] = (8,)
    cfg["critic"]["hidden_dims"] = (8,)
    cfg["algorithm"].update(num_learning_epochs=1, num_mini_batches=2)
    return cfg


def test_training_defaults_preserve_leaderboard_reward_and_reference_start():
    cfg = train_config()
    assert cfg["algorithm"]["gamma"] == 1.0
    assert cfg["algorithm"]["entropy_coef"] == 0.001
    assert cfg["algorithm"]["teacher_loss_coef_start"] == 0.0
    assert cfg["algorithm"]["symmetry_cfg"] is None
    assert cfg["actor"]["class_name"].endswith(":ReferenceActor")
    assert cfg["critic"]["class_name"] == "MLPModel"
    assert cfg["actor"]["obs_normalization"] is True
    assert "leaderboard_score_cm / 100" in REWARD
    assert train_config(from_scratch=True)["actor"] == ACTOR_CFG
    assert ACTOR_CFG["class_name"] == "MLPModel"
    args = build_parser().parse_args(["--num-envs", "64", "--iterations", "5"])
    assert args.num_envs == 64 and args.iterations == 5


def test_randomization_cli_and_update_progress(tmp_path):
    args = build_parser().parse_args(["--randomization-config", str(tmp_path / "randomization.json"),
                                      "--evaluation-baseline-8000", str(tmp_path / "8000.onnx"),
                                      "--stress-eval-seeds", "70021", "70022"])
    assert args.randomization_config == tmp_path / "randomization.json"
    assert args.evaluation_baseline_8000 == tmp_path / "8000.onnx"
    assert args.stress_eval_seeds == [70021, 70022]
    updates = []
    runner = TugRunner.__new__(TugRunner)
    runner.env = SimpleNamespace(set_training_update=updates.append)
    runner.evaluation_opponents = {}
    runner.snapshot_interval = 200
    runner._after_update(8001)
    assert updates == [8001]


def test_continuation_changes_randomization_before_starting_new_bouts():
    from mjlab_microduck.tug_of_war.train import apply_continuation

    calls = []
    config = {"mix": {"nominal": .5, "force": .15, "spawn": .15, "both": .2}}
    runner = SimpleNamespace(alg=SimpleNamespace(entropy_coef=.0003),
                             evaluation_opponents={},
                             env=SimpleNamespace(pool=SimpleNamespace(),
                                 arena=SimpleNamespace(randomization=SimpleNamespace(
                                     configure_mix=lambda value: calls.append(("configure", value)))),
                                 reset=lambda: calls.append(("reset",))))
    apply_continuation(runner, randomization_config=config)
    assert calls == [("configure", config), ("reset",)]


def test_protected_selection_uses_fixed_opponents_instead_of_rotating_recent():
    from mjlab_microduck.tug_of_war.train import evaluation_selection

    def result(win_rate):
        return {"win_rate": win_rate, "fault_rate": 0., "mean_score_cm": 10.}

    report = {"opponents": {"cup_a": result(1.), "baseline_6000": result(.75),
                             "baseline_8000": result(.5), "archive_4400": result(.25),
                             "recent_dynamic": result(0.)},
              "historical_opponents": {"archive_4400": {}, "recent_dynamic": {}},
              "protected_historical_opponents": {"archive_4400": {}}}
    selection, metric, _ = evaluation_selection(report)
    assert metric == "cup_baselines_protected_v2"
    assert selection[:2] == pytest.approx(((1. + .625 + .25) / 3, .625))
    report["opponents"]["recent_dynamic"] = result(1.)
    assert evaluation_selection(report)[0] == selection


def test_reference_start_copies_network_and_normalizer_but_keeps_fresh_learning_state(tmp_path: Path):
    reference = pytest.importorskip("microduck_challenges.tug_of_war.opponent")
    runner = TugRunner(TinyTug(), train_config(1), str(tmp_path))
    obs = torch.linspace(-0.1, 0.1, 61).reshape(1, 61)
    with torch.no_grad():
        expected = reference.load_actor(reference.REFERENCE)(obs)
        actual = runner.alg.actor(TensorDict({"actor": obs}, [1]))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert runner.alg.actor.obs_normalizer.count.item() == reference.REFERENCE_SEEN
    assert runner.alg.critic.obs_normalizer.count.item() == 0
    assert runner.alg.optimizer.state_dict()["state"] == {}
    assert runner.alg.update_count == runner.env.common_step_counter == 0


def test_checkpoint_resumes_exact_update_and_pool_with_frozen_normalizer(tmp_path: Path):
    env = TinyTug()
    runner = TugRunner(env, tiny_config(), str(tmp_path), snapshot_interval=1)
    runner.learn(1)
    assert runner.current_learning_iteration == runner.alg.update_count == 1
    assert env.pool.joint_count == 1
    assert runner.alg.actor.obs_normalizer.count.item() == 6  # 3 live rows × 2 steps
    assert runner.alg.actor.obs_normalizer._mean.abs().max() < 2
    snapshot = deepcopy(env.pool.state_dict()["snapshots"])
    with torch.no_grad():
        next(runner.alg.actor.parameters()).add_(0.2)
    saved = tmp_path / "resume.pt"
    runner.save(str(saved), {"test": "restored"})
    expected_reset_draws = (torch.rand(1), random.random(), np.random.random())
    restored = TugRunner(TinyTug(), tiny_config(), str(tmp_path / "resumed"), snapshot_interval=3)
    assert restored.load(str(saved)) == {"test": "restored"}
    assert restored.current_learning_iteration == restored.alg.update_count == 1
    assert restored.snapshot_interval == 1
    torch.testing.assert_close(restored.env.reset_draws[0], expected_reset_draws[0])
    assert restored.env.reset_draws[1:] == expected_reset_draws[1:]
    assert restored.env.common_step_counter == 2
    assert restored.alg.optimizer.state_dict()["state"]
    for key, value in runner.alg.actor.state_dict().items():
        torch.testing.assert_close(value, restored.alg.actor.state_dict()[key])
    for uid, state in restored.env.pool.state_dict()["snapshots"].items():
        for key, value in snapshot[uid]["actor_state_dict"].items():
            torch.testing.assert_close(value, state["actor_state_dict"][key])
    restored.learn(1)
    assert restored.current_learning_iteration == restored.alg.update_count == 2
    assert restored.env.pool.joint_count == 2
    with pytest.raises(ValueError, match="complete optimizer"):
        restored.load(str(saved), load_cfg={"actor": True})


def test_export_matches_actor_with_baked_normalizer_and_exact_arena_shapes(tmp_path: Path):
    runner = TugRunner(TinyTug(), tiny_config(), str(tmp_path))
    actor = runner.alg.actor
    with torch.no_grad():
        actor.obs_normalizer._mean.fill_(2.0)
        actor.obs_normalizer._std.fill_(3.0)
        actor.obs_normalizer._var.fill_(9.0)
    path = export_policy(runner, tmp_path / "policy.onnx")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    obs = torch.linspace(-1.0, 1.0, 61).reshape(1, 61)
    with torch.no_grad():
        expected = actor(TensorDict({"actor": obs}, [1])).numpy()
    actual = session.run(None, {session.get_inputs()[0].name: obs.numpy()})[0]
    assert session.get_inputs()[0].shape == [1, 61]
    assert session.get_outputs()[0].shape == [1, 14]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_snapshot_boundary_uses_completed_updates(tmp_path: Path):
    runner = TugRunner.__new__(TugRunner)
    runner.snapshot_interval = 200
    runner.evaluation_opponents = {}
    calls = []
    runner.env = SimpleNamespace(add_snapshot=lambda *args: calls.append(args))
    runner.alg = SimpleNamespace(actor=torch.nn.Linear(61, 14))
    runner.actor_cfg = ACTOR_CFG
    runner._log_dir = tmp_path
    saved = []
    runner.save = saved.append
    runner._after_update(199)
    runner._after_update(200)
    runner._after_update(201)
    assert len(calls) == 1 and calls[0][2] == 200
    assert saved == [str(tmp_path / "snapshot_200.pt")]


def test_scored_snapshot_receives_crossplay_vector(tmp_path: Path):
    runner = TugRunner.__new__(TugRunner)
    runner.snapshot_interval = runner.eval_interval = 200
    runner.evaluation_opponents = {"juicenv": tmp_path / "anchor.onnx"}
    runner.evaluate = lambda _: {"opponents": {"juicenv": {"win_rate": .75},
                                               "baseline_6000": {"win_rate": .5},
                                               "archive_1000_uid7": {"win_rate": .25},
                                               "cup_user": {"win_rate": 1.0}}}
    calls = []
    runner.env = SimpleNamespace(add_snapshot=lambda *args, **kwargs: calls.append((args, kwargs)))
    runner.alg = SimpleNamespace(actor=torch.nn.Linear(61, 14))
    runner.actor_cfg = ACTOR_CFG
    runner._log_dir = tmp_path
    runner.save = lambda _: None
    runner._after_update(200)
    assert calls[0][0][2] == 200
    assert calls[0][1]["scores_by_opponent"] == {"juicenv": .75}


def test_resume_entropy_and_post_load_league_override_preserve_learning_state(tmp_path: Path):
    from mjlab_microduck.tug_of_war.train import apply_continuation

    class ConfigurablePool(OpponentPool):
        weights = {"live": .5, "reference": .5}

        def state_dict(self):
            return {**super().state_dict(), "weights": self.weights.copy()}

        def load_state_dict(self, state):
            super().load_state_dict(state)
            self.weights = state["weights"].copy()

        def configure(self, config):
            self.weights = config["weights"].copy()

    class LeagueTug(TinyTug):
        def __init__(self):
            super().__init__()
            self.pool = ConfigurablePool()

        def reset(self):
            self.reset_mix = self.pool.weights.copy()

    source = TugRunner(LeagueTug(), tiny_config(), str(tmp_path / "source"))
    source.learn(1)
    source.alg.entropy_coef = .0007
    saved = tmp_path / "resume.pt"
    source.save(str(saved))
    restored = TugRunner(LeagueTug(), tiny_config(), str(tmp_path / "restored"))
    restored.load(str(saved))
    assert restored.alg.entropy_coef == .0007
    mix = {"live": .2, "reference": .8}
    changes = apply_continuation(restored, league_config={"weights": mix}, entropy_coef=.0003,
                                 evaluation_baseline=tmp_path / "baseline.onnx",
                                 evaluation_roster={"cup_user": tmp_path / "cup.onnx"})
    assert changes["old_opponent_mix"] == {"live": .5, "reference": .5}
    assert changes["old_entropy_coefficient"] == .0007
    assert restored.env.reset_mix == restored.env.pool.weights == mix
    assert restored.env.live_fraction == .2 and restored.env.reference_fraction == .8
    assert restored.alg.entropy_coef == .0003
    assert restored.alg.update_count == restored.current_learning_iteration == 1
    for name in ("actor", "critic"):
        for key, tensor in getattr(source.alg, name).state_dict().items():
            torch.testing.assert_close(tensor, getattr(restored.alg, name).state_dict()[key])
    assert restored.alg.optimizer.state_dict()["param_groups"] == source.alg.optimizer.state_dict()["param_groups"]
    assert len(restored.alg.optimizer.state) == len(source.alg.optimizer.state)
    assert restored.evaluation_opponents["baseline_6000"] == (tmp_path / "baseline.onnx").resolve()
    assert restored.evaluation_opponents["cup_user"] == tmp_path / "cup.onnx"
    # Legacy checkpoints retain the original entropy default, even with another constructor setting.
    legacy = torch.load(saved, weights_only=False)
    del legacy["tug_of_war"]["entropy_coef"]
    torch.save(legacy, saved)
    restored.load(str(saved))
    assert restored.alg.entropy_coef == .001


@pytest.mark.parametrize("coefficient", [-.1, float("nan"), float("inf")])
def test_entropy_override_rejects_invalid_coefficients(coefficient):
    from mjlab_microduck.tug_of_war.train import apply_continuation

    runner = SimpleNamespace(env=SimpleNamespace(pool=OpponentPool()), alg=SimpleNamespace(entropy_coef=.001))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        apply_continuation(runner, entropy_coef=coefficient)
    assert runner.alg.entropy_coef == .001


def test_historical_cpu_evaluation_routes_valid_results_and_preserves_rng(tmp_path: Path, monkeypatch):
    from mjlab_microduck.tug_of_war import evaluation, train

    calls = []

    class HistoricalPool:
        selection = "challenger"
        history = [{"uid": 7, "iteration": 1000, "group": "archive"},
                   {"uid": 8, "iteration": 5800, "group": "recent"}]

        def model(self, choice, device):
            assert device == "cpu" and choice.kind == "joint"
            torch.rand(3)  # Discarded initialization draws must be isolated.
            calls.append(("model", choice.uid))
            return choice.uid

        def set_challenger_scores(self, uid, wins, losses, invalid):
            calls.append(("score", uid, wins, losses, invalid))

    def export(runner, path, *, actor=None, iteration=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(str((actor, iteration)).encode())
        return path

    monkeypatch.setattr(train, "export_policy", export)
    fields = {"reference": .5, "juicenv": 1., "baseline_6000": .5, "cup_user": .75}
    runner = TugRunner.__new__(TugRunner)
    runner._log_dir = tmp_path
    runner.env = SimpleNamespace(pool=HistoricalPool())
    runner.evaluation_opponents = {name: tmp_path / f"{name}.onnx" for name in fields}
    for path in runner.evaluation_opponents.values():
        path.write_bytes(b"fixed")
    runner.evaluation_seeds = (1, 2, 3, 4)
    runner.best_evaluation = (1., 100.)  # Legacy saturated anchors cannot block new comparisons.
    runner.best_evaluation_metric = "anchors_v1"
    runner.best_evaluation_context = None
    runner.logger = SimpleNamespace(writer=None)
    saved = []
    runner.save = lambda path, infos: saved.append((path, infos))

    def bout(winner, how):
        return {"winner": winner, "how": how, "score_cm": 0.}

    archive = [bout("learner", "pit"), bout("opponent", "pit"),
               bout("learner", "fault"), bout("opponent", "reset")]
    recent = [bout("opponent", "pit")]

    def evaluate(policy, opponents, seeds):
        assert seeds == (1, 2, 3, 4)
        assert set(opponents) == set(fields) | {"archive_1000_uid7", "recent_5800_uid8"}
        results = {}
        for name, path in opponents.items():
            bouts = archive if name.startswith("archive_") else recent
            data = evaluation._summarize(bouts)
            if name in fields:
                data["win_rate"] = fields[name]
            results[name] = {"policy": str(path), **data}
        return {"opponents": results, "mean_win_rate": .5, "mean_score_cm": 0., "fault_rate": .1}

    monkeypatch.setattr(evaluation, "evaluate_policy", evaluate)
    rng = torch.get_rng_state().clone()
    report = runner.evaluate(6200)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    assert ("score", 7, 1, 1, 2) in calls
    assert ("score", 8, 0, 1, 0) in calls
    assert report["historical_opponents"]["archive_1000_uid7"]["invalid"] == 2
    # History fault-inclusive rate: (.5 + 0)/2; cup and baseline each have equal group weight.
    assert report["checkpoint_selection"]["value"][0] == pytest.approx((.75 + .5 + .25) / 3)
    assert saved[0][0].endswith("model_best.pt")
    assert (tmp_path / "evaluations/evaluation_006200.json").is_file()
    assert report["opponents"]["baseline_6000"]["policy_sha256"]
    runner.evaluate(6400)
    assert [call for call in calls if call[0] == "model"] == [("model", 7), ("model", 8)]


def test_resume_learning_rate_override_is_fixed_and_survives_checkpoint(tmp_path):
    from mjlab_microduck.tug_of_war.train import apply_continuation

    args = build_parser().parse_args(["--resume", "source.pt", "--resume-learning-rate", "5e-6"])
    assert args.resume_learning_rate == 5e-6
    source = TugRunner(TinyTug(), tiny_config(), str(tmp_path / "source"))
    source.learn(1)
    checkpoint = tmp_path / "source.pt"
    source.save(str(checkpoint))
    runner = TugRunner(TinyTug(), tiny_config(), str(tmp_path / "resumed"))
    runner.load(str(checkpoint))
    before = deepcopy(runner.alg.optimizer.state_dict())
    changes = apply_continuation(runner, resume_learning_rate=5e-6)
    assert changes["old_learning_rate"] == source.alg.learning_rate
    assert changes["learning_rate_schedule"] == runner.alg.schedule == "fixed"
    assert runner.alg.learning_rate == 5e-6
    for group in runner.alg.optimizer.param_groups:
        assert group["lr"] == 5e-6
    for key, state in before["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, runner.alg.optimizer.state_dict()["state"][key][name])
    assert runner.alg.update_count == source.alg.update_count == 1
    runner.learn(1)
    assert runner.alg.update_count == 2
    assert runner.alg.learning_rate == 5e-6
    assert all(group["lr"] == 5e-6 for group in runner.alg.optimizer.param_groups)
    runner.save(str(checkpoint))
    restored = TugRunner(TinyTug(), tiny_config(), str(tmp_path / "restored"))
    restored.load(str(checkpoint))
    assert restored.alg.schedule == "fixed" and restored.alg.learning_rate == 5e-6
    assert all(group["lr"] == 5e-6 for group in restored.alg.optimizer.param_groups)


@pytest.mark.parametrize("rate", [0, -1e-5, float("nan"), float("inf")])
def test_resume_learning_rate_override_rejects_invalid_rates(rate):
    from mjlab_microduck.tug_of_war.train import apply_continuation

    with pytest.raises(ValueError, match="finite and positive"):
        apply_continuation(SimpleNamespace(), resume_learning_rate=rate)
