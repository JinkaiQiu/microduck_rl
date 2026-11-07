"""Self-play routing and league lifetime, without expensive physics fixtures."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from mjlab_microduck.football.pool import OpponentChoice, OpponentPool
from mjlab_microduck.tug_of_war.league import DEFAULT_MIX, TugLeague
from mjlab_microduck.tug_of_war.wrapper import TugVecEnv


class Arena:
    device = "cpu"
    num_envs = 2
    max_episode_length = 2275

    def __init__(self):
        self.env = SimpleNamespace(cfg={}, common_step_counter=0, close=lambda: None)
        self.elapsed = torch.zeros(2, dtype=torch.long)
        self.finish = torch.zeros(2, dtype=torch.bool)
        self.resets = []

    def actor_obs(self, side):
        return torch.full((2, 61), float(side))

    def critic_obs(self, side):
        return torch.full((2, 70), float(side))

    def reset(self, ids):
        self.resets.append(ids.clone())
        self.elapsed[ids] = 0

    def step(self, actions):
        self.actions = actions.clone()
        self.elapsed += 1
        rewards = (torch.tensor([0.2, 0.3]), torch.tensor([-0.2, -0.3]))
        return (), rewards, self.finish.clone(), {
            "score_cm": torch.tensor([[20., -20.], [30., -30.]]),
            "winner": torch.tensor([0, 1]), "fault": torch.zeros(2, dtype=torch.bool),
            "numerical_reset": torch.zeros(2, dtype=torch.bool),
            "timed_out": self.finish.clone(), "elapsed_seconds": self.elapsed.float() / 50,
        }


@pytest.fixture
def wrapped(monkeypatch):
    from microduck_challenges.tug_of_war import opponent

    class Reference(torch.nn.Module):
        def forward(self, obs):
            return torch.full((len(obs), 14), 0.125)

    monkeypatch.setattr(opponent, "load_actor", lambda _: Reference())
    result = TugVecEnv(Arena(), OpponentPool(seed=3), seed=3)
    result.learner_west[:] = torch.tensor([True, False])
    result.opponent_choices[:] = [OpponentChoice("live"), OpponentChoice("scripted")]
    return result


def test_two_seats_route_raw_actions_and_mask_frozen_seat(wrapped):
    obs = wrapped.get_observations()
    assert obs["actor"].shape == (4, 61)
    assert obs["actor"][:, 0].tolist() == [0, 1, 1, 0]
    actions = torch.arange(4.).unsqueeze(-1).expand(-1, 14)
    _, rewards, dones, info = wrapped.step(actions)
    assert wrapped.arena.actions[:, 0].tolist() == [0, 0.125]
    assert wrapped.arena.actions[:, 14].tolist() == [2, 1]
    assert info["learner_mask"].tolist() == [True, True, True, False]
    torch.testing.assert_close(rewards, torch.tensor([0.2, -0.3, -0.2, 0.3]))
    assert dones.tolist() == [0, 0, 0, 0]
    assert "time_outs" not in info


def test_terminal_mask_and_scores_precede_new_opponent_assignment(wrapped):
    wrapped.arena.finish[:] = True
    wrapped.live_fraction, wrapped.reference_fraction = 1., 0.
    _, _, dones, info = wrapped.step(torch.zeros(4, 14))
    assert dones.tolist() == [1, 1, 1, 1]
    assert info["learner_mask"].tolist() == [True, True, True, False]
    assert info["completed"]["score_cm"].tolist() == [20, -30]
    assert list(wrapped.recent_scores) == [-30]
    assert wrapped.recent_results["reference"] == wrapped.recent_results["live"]
    assert [c.kind for c in wrapped.opponent_choices] == ["live", "live"]
    assert wrapped.arena.elapsed.tolist() == [0, 0]


def test_pool_pruning_keeps_snapshot_while_bout_is_active(wrapped):
    wrapped.pool.capacity = 1
    old = wrapped.pool.add_state("joint", {"weight": torch.ones(1)}, {}, 0)
    wrapped.opponent_choices[0] = OpponentChoice("joint", old)
    wrapped.add_snapshot(torch.nn.Linear(1, 1), {}, 1)
    assert old in wrapped.pool._snapshots
    wrapped.opponent_choices[0] = OpponentChoice("live")
    wrapped.pool.prune(set())
    assert old not in wrapped.pool._snapshots


def test_sampler_and_benchmark_statistics_survive_checkpoint(wrapped):
    wrapped.recent_scores.extend([1., -2.])
    state = wrapped.state_dict()
    wrapped.reset()
    expected = (wrapped.learner_west.clone(), list(wrapped.opponent_choices))
    wrapped.load_state_dict(state)
    torch.testing.assert_close(wrapped.learner_west, expected[0])
    assert wrapped.opponent_choices == expected[1]
    assert list(wrapped.recent_scores) == [1., -2.]


def test_paired_clocks_and_fraction_validation(wrapped):
    with pytest.raises(ValueError, match="share"):
        wrapped.episode_length_buf = torch.tensor([0, 1, 0, 0])
    wrapped.episode_length_buf = torch.tensor([2, 3, 2, 3])
    assert wrapped.arena.elapsed.tolist() == [2, 3]
    with pytest.raises(ValueError, match="fractions"):
        TugVecEnv(wrapped.arena, wrapped.pool, live_fraction=.8, reference_fraction=.8)


@pytest.fixture
def league(monkeypatch, tmp_path):
    from microduck_challenges.tug_of_war import opponent

    def actor(_):
        return torch.nn.Sequential(
            opponent.Normalize(torch.zeros(61), torch.ones(61)),
            torch.nn.Linear(61, 8), torch.nn.ELU(), torch.nn.Linear(8, 8),
            torch.nn.ELU(), torch.nn.Linear(8, 8), torch.nn.ELU(), torch.nn.Linear(8, 14),
        )

    monkeypatch.setattr(opponent, "load_actor", actor)
    paths = {}
    for name in ("juicenv", "andoshy", "ctcheung"):
        paths[name] = tmp_path / f"{name}.onnx"
        paths[name].write_bytes(name.encode())
    return TugLeague(paths, seed=12)


def test_league_distribution_and_missing_history_redistribution(league):
    assert league.mixture == pytest.approx({**DEFAULT_MIX, "recent": 0, "archive": 0,
                                           "juicenv": .24, "andoshy": .24, "ctcheung": .12})
    for index in range(9):
        league.add_state("joint", {"weight": torch.tensor(index)}, {}, index)
        league.prune(set())
    assert league.mixture == DEFAULT_MIX
    labels = [league.label(choice) for choice in league.choose(20_000)]
    for name, expected in DEFAULT_MIX.items():
        assert labels.count(name) / len(labels) == pytest.approx(expected, abs=.015)
    assert len(league._recent["joint"]) == len(league._archive) == 4
    assert len(league._snapshots) == 11


def test_external_actors_are_frozen_and_checkpoint_contains_their_weights(league):
    uid = league._external["juicenv"]
    choice = OpponentChoice("joint", uid)
    rng = torch.get_rng_state().clone()
    model = league.model(choice, "cpu")
    assert torch.equal(torch.get_rng_state(), rng)
    assert not model.training and not any(param.requires_grad for param in model.parameters())
    state = {key: value.clone() for key, value in model.state_dict().items()}
    observations = TensorDict({"actor": torch.randn(3, 61)}, batch_size=[3])
    expected = model(observations)
    model(TensorDict({"actor": torch.randn(8, 61) * 10}, batch_size=[8]))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key])
    checkpoint = league.state_dict()
    expected_choices = league.choose(100)
    for snapshot in checkpoint["snapshots"].values():
        if snapshot.get("path"):
            from pathlib import Path
            Path(snapshot["path"]).unlink()
    league.load_state_dict(checkpoint)
    assert league.choose(100) == expected_choices
    torch.testing.assert_close(league.model(choice, "cpu")(observations), expected)
    restored = TugLeague.from_state(checkpoint)
    assert restored.choose(100) == expected_choices
    torch.testing.assert_close(restored.model(choice, "cpu")(observations), expected)
    assert len(league._snapshots[uid]["sha256"]) == 64


def test_archive_keeps_strong_and_different_evaluated_opponents(league):
    league.capacity, league.archive_capacity = 1, 2
    actor = torch.nn.Linear(1, 1)
    first = league.add_snapshot(actor, {}, 0, {"a": .9, "b": .9})
    similar = league.add_snapshot(actor, {}, 1, {"a": .8, "b": .8})
    different = league.add_snapshot(actor, {}, 2, {"a": 0., "b": 1.})
    league.add_snapshot(actor, {}, 3)
    league.prune(set())
    assert set(league._archive) == {first, different}
    assert similar not in league._snapshots
    assert set(league._external.values()) <= league._snapshots.keys()
    with pytest.raises(ValueError, match="win rates"):
        league.set_snapshot_scores(first, {"a": float("nan")})


def test_league_pruning_preserves_active_retired_opponent(league):
    league.capacity = league.archive_capacity = 1
    league.add_state("joint", {"weight": torch.ones(1)}, {}, 0)
    active = league.add_state("joint", {"weight": torch.ones(1)}, {}, 1)
    league.add_state("joint", {"weight": torch.ones(1)}, {}, 2)
    assert active not in league._archive + league._recent["joint"]
    league.prune({active})
    assert active in league._snapshots
    league.prune(set())
    assert active not in league._snapshots


def test_league_wrapper_routes_external_actions_and_keeps_bout_assignment(league):
    wrapped = TugVecEnv(Arena(), league, seed=12)
    wrapped.learner_west[:] = True
    external = OpponentChoice("joint", league._external["juicenv"])
    wrapped.opponent_choices[:] = [external, OpponentChoice("live")]
    expected = league.model(external, "cpu")(TensorDict({"actor": torch.ones(1, 61)}, batch_size=[1]))
    _, _, _, info = wrapped.step(torch.zeros(4, 14))
    torch.testing.assert_close(wrapped.arena.actions[:1, 14:], expected)
    assert wrapped.opponent_choices[0] == external
    assert info["learner_mask"].tolist() == [True, True, False, True]
    assert info["log"]["league/match_fraction_juicenv"] == .5
    checkpoint = wrapped.state_dict()
    wrapped.reset()
    expected_choices = list(wrapped.opponent_choices)
    expected_sides = wrapped.learner_west.clone()
    wrapped.load_state_dict(checkpoint)
    assert wrapped.opponent_choices == expected_choices
    torch.testing.assert_close(wrapped.learner_west, expected_sides)


def test_challenger_overrides_and_scores_survive_checkpoint(league):
    for iteration in range(8):
        league.add_state("joint", {"weight": torch.ones(1)}, {}, iteration)
    weights = dict.fromkeys(DEFAULT_MIX, 0.)
    weights.update(live=.2, recent=.2, archive=.5, reference=.1)
    league.configure({"weights": weights, "selection": "challenger",
                      "challenger_uniform_fraction": .25,
                      "challenger_scores": {"0": {"wins": 16, "losses": 14, "invalid": 2}}})
    league.configure({"challenger_scores": {"1": {"wins": 8, "losses": 0}}})
    checkpoint = league.state_dict()
    expected = league.choose(100)
    restored = TugLeague.from_state(checkpoint)
    assert restored.choose(100) == expected
    assert restored.weights == weights
    assert restored.selection == "challenger"
    assert restored.challenger_uniform_fraction == .25
    assert restored.history == league.history
    first = next(row for row in restored.history if row["iteration"] == 0)
    assert first["learner_win_rate"] == pytest.approx(16 / 30)
    assert first["invalid_rate"] == pytest.approx(2 / 32)
    first["challenger_scores"]["wins"] = 999
    assert restored.history[0]["challenger_scores"]["wins"] == 16


def test_old_league_checkpoint_uses_anchor_defaults(league):
    checkpoint = league.state_dict()
    del checkpoint["selection"], checkpoint["challenger_uniform_fraction"]
    restored = TugLeague.from_state(checkpoint)
    assert restored.selection == "anchors"
    assert restored.challenger_uniform_fraction == .2
    assert restored.weights == DEFAULT_MIX
    assert restored.history == []


def test_challenger_sampling_keeps_uniform_coverage_and_recent_uniform(league):
    for iteration in range(8):
        league.add_state("joint", {"weight": torch.ones(1)}, {}, iteration)
    mastered, challenging, unmeasured, invalid = league._archive
    league.set_challenger_scores(mastered, 8, 0)
    league.set_challenger_scores(challenging, 5, 5)
    league.set_challenger_scores(invalid, 0, 0, invalid=10)
    weights = dict.fromkeys(DEFAULT_MIX, 0.)
    weights["archive"] = 1.
    league.configure({"weights": weights, "selection": "challenger"})
    sampled = [choice.uid for choice in league.choose(12_000)]
    assert sampled.count(mastered) / len(sampled) == pytest.approx(.05, abs=.015)
    for uid in (challenging, unmeasured, invalid):
        assert sampled.count(uid) / len(sampled) == pytest.approx(.05 + .8 / 3, abs=.02)
    league.configure({"weights": {**weights, "archive": 0., "recent": 1.}})
    for uid in league._recent["joint"]:
        league.set_challenger_scores(uid, 0 if uid % 2 else 8, 8 if uid % 2 else 0)
    sampled = [choice.uid for choice in league.choose(8_000)]
    for uid in league._recent["joint"]:
        assert sampled.count(uid) / len(sampled) == pytest.approx(.25, abs=.02)


@pytest.mark.parametrize("config", [
    {"weights": {"archive": 1.}},
    {"weights": {**DEFAULT_MIX, "archive": -.15}},
    {"weights": {**DEFAULT_MIX, "archive": float("nan")}},
    {"selection": "unknown"}, {"challenger_uniform_fraction": float("inf")},
    {"challenger_uniform_fraction": -.1}, {"capacity": 100},
    {"challenger_scores": {"0": {"wins": -1, "losses": 1}}},
    {"challenger_scores": {"0": {"wins": 1.5, "losses": 1}}},
    {"challenger_scores": {"0": {"wins": True, "losses": 1}}},
    {"challenger_scores": {"999": {"wins": 1, "losses": 1}}},
])
def test_invalid_league_overrides_are_atomic(league, config):
    league.add_state("joint", {"weight": torch.ones(1)}, {}, 0)
    with pytest.raises(ValueError):
        league.configure({"selection": "challenger", **config})
    assert league.selection == "anchors"
    assert league.weights == DEFAULT_MIX
    assert league.history[0]["challenger_scores"] is None


def test_challenger_retention_prefers_competition_and_preserves_active_uid(league):
    league.capacity, league.archive_capacity = 1, 2
    mastered = league.add_state("joint", {"weight": torch.ones(1)}, {}, 1000)
    league.set_challenger_scores(mastered, 8, 0)
    competitive = league.add_state("joint", {"weight": torch.ones(1)}, {}, 5200)
    league.set_challenger_scores(competitive, 16, 14, 2)
    league.configure({"selection": "challenger"})
    next_uid = league.add_state("joint", {"weight": torch.ones(1)}, {}, 5400)
    league.set_challenger_scores(next_uid, 16, 14, 2)
    league.add_state("joint", {"weight": torch.ones(1)}, {}, 5600)
    assert len(league._archive) == 2
    assert next_uid in league._archive  # most recent measured competitive probe
    retired = ({mastered, competitive, next_uid} - set(league._archive)).pop()
    league.prune({retired})
    assert retired in league._snapshots
    league.prune(set())
    assert retired not in league._snapshots


def _native_actor():
    from rsl_rl.models import MLPModel

    cfg = {"hidden_dims": (8,), "activation": "elu", "obs_normalization": True}
    obs = TensorDict({"actor": torch.zeros(1, 61)}, batch_size=[1])
    return MLPModel(obs, {"actor": ["actor"]}, "actor", 14, **cfg), cfg


def _save_native(path, actor, cfg, iteration):
    torch.save({"iter": iteration, "actor_state_dict": actor.state_dict(),
                "tug_of_war": {"schema": 1, "actor_cfg": cfg}}, path)
    return str(path.resolve())


def test_protected_native_archive_preserves_recent_state_and_survives_retirement(league, tmp_path):
    actor, cfg = _native_actor()
    actor.obs_normalizer.update(torch.arange(183, dtype=torch.float32).reshape(3, 61))
    assert actor.obs_normalizer.count.item() == 3
    recent = league.add_snapshot(actor, cfg, 7400)
    paths = {str(iteration): _save_native(tmp_path / f"{iteration}.pt", actor, cfg, iteration)
             for iteration in (6000, 7400)}
    recent_before = list(league._recent["joint"])
    rng = league._rng.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    league.configure({"archive_capacity": 2, "protected_checkpoints": paths,
                      "selection": "challenger",
                      "challenger_scores": {"6000": {"wins": 1, "losses": 3}}})
    assert league._recent["joint"] == recent_before
    assert torch.equal(league._rng.get_state(), rng)
    assert torch.equal(torch.get_rng_state(), global_rng)
    assert len(league._archive) == len(league._protected) == 2
    assert not set(league._archive) & set(league._recent["joint"])
    assert league.label(OpponentChoice("joint", recent)) == "recent"
    for uid in league._protected:
        assert league.label(OpponentChoice("joint", uid)) == "archive"
        snapshot = league._snapshots[uid]
        assert len(snapshot["checkpoint_sha256"]) == 64
        assert snapshot["checkpoint_path"] == paths[str(snapshot["iteration"])]
        model = league.model(OpponentChoice("joint", uid), "cpu")
        for key, value in actor.state_dict().items():
            assert torch.equal(model.state_dict()[key], value)
    protected = list(league._protected)
    league.configure({"protected_checkpoints": paths})
    assert league._protected == protected
    assert all(row["protected"] == (row["uid"] in protected) for row in league.history)
    with pytest.raises(ValueError, match="cannot evict"):
        league.configure({"archive_capacity": 1})
    changed = torch.load(paths["6000"], weights_only=False)
    changed["infos"] = "same actor with changed checkpoint identity"
    torch.save(changed, paths["6000"])
    with pytest.raises(ValueError, match="differs"):
        league.configure({"protected_checkpoints": paths})
    assert league._protected == protected and league.archive_capacity == 2
    for iteration in range(7600, 9001, 200):
        league.add_snapshot(actor, cfg, iteration)
        league.prune(set())
    assert league._archive == protected
    assert recent not in league._snapshots
    state = league.state_dict()
    for path in tmp_path.glob("*.pt"):
        path.unlink()
    restored = TugLeague.from_state(state)
    assert restored._protected == protected
    assert restored._archive == protected
    assert restored.choose(100) == league.choose(100)
    for uid in protected:
        for key, value in actor.state_dict().items():
            assert torch.equal(restored._snapshots[uid]["actor_state_dict"][key], value)


@pytest.mark.parametrize("bad", ["architecture", "normalizer", "iteration", "capacity", "identity"])
def test_protected_native_validation_is_atomic(league, tmp_path, bad):
    actor, cfg = _native_actor()
    original = league.add_snapshot(actor, cfg, 6000)
    path = tmp_path / "6000.pt"
    filename = _save_native(path, actor, cfg, 6000)
    config = {"archive_capacity": 2, "protected_checkpoints": {"6000": filename}}
    state = torch.load(path, weights_only=False)
    if bad == "architecture":
        state["tug_of_war"]["actor_cfg"] = {**cfg, "hidden_dims": (16,)}
    elif bad == "normalizer":
        state["actor_state_dict"]["obs_normalizer._std"].zero_()
    elif bad == "iteration":
        state["iter"] = 5999
    elif bad == "capacity":
        config["archive_capacity"] = 0
    else:
        state["actor_state_dict"]["mlp.0.weight"].add_(1)
    torch.save(state, path)
    rng = league._rng.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        league.configure({"selection": "challenger", **config})
    assert league.selection == "anchors"
    assert league.archive_capacity == 4 and league._protected == [] and league._archive == []
    assert league._recent["joint"] == [original]
    assert torch.equal(league._rng.get_state(), rng)
    assert torch.equal(torch.get_rng_state(), global_rng)


def test_explicit_protected_replacement_reuses_identity_and_respects_capacity(league, tmp_path):
    actor, cfg = _native_actor()
    recent = league.add_snapshot(actor, cfg, 7400)
    paths = {str(i): _save_native(tmp_path / f"{i}.pt", actor, cfg, i)
             for i in (5000, 6000, 7400, 8800)}
    league.configure({"archive_capacity": 2,
                      "protected_checkpoints": {i: paths[i] for i in ("5000", "6000")}})
    old = {league._snapshots[uid]["iteration"]: uid for uid in league._protected}
    proposed = {i: paths[i] for i in ("6000", "7400", "8800")}
    # Default configuration still adds pins; explicit replacement is required.
    with pytest.raises(ValueError, match="cannot evict"):
        league.configure({"protected_checkpoints": {"8800": paths["8800"]}})
    with pytest.raises(ValueError, match="cannot evict"):
        league.configure({"protected_checkpoints": proposed, "replace_protected_checkpoints": True})
    assert league._protected == list(old.values())
    assert league.archive_capacity == 2
    league.configure({"archive_capacity": 3, "protected_checkpoints": proposed,
                      "replace_protected_checkpoints": True})
    new = {league._snapshots[uid]["iteration"]: uid for uid in league._protected}
    assert new.keys() == {6000, 7400, 8800}
    assert new[6000] == old[6000]
    assert len(league._archive) == 3 and set(league._archive) == set(new.values())
    assert league._recent["joint"] == [recent]
    assert not set(league._archive) & set(league._recent["joint"])
    league.prune({old[5000]})
    assert old[5000] in league._snapshots
    league.prune(set())
    assert old[5000] not in league._snapshots
    assert TugLeague.from_state(league.state_dict())._protected == league._protected
