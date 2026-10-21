"""Opponent league sampling and checkpoint state invariants."""

import torch

from mjlab_microduck.football.pool import OpponentChoice, OpponentPool
from mjlab_microduck.football.skills import FrozenSkillPolicy


def test_bootstrap_rolling_snapshots_and_resume():
    pool = OpponentPool(seed=7, capacity=8)
    bootstrap = pool.sample(1000, "tactics")
    assert 0.03 < sum(c.kind == "scripted" for c in bootstrap) / len(bootstrap) < 0.07
    assert all(c.kind in ("scripted", "live") for c in bootstrap)
    for iteration in range(10):
        pool.add_state("tactical", {"weight": torch.tensor([float(iteration)])}, {}, iteration)
    assert pool.tactical_count == 8
    choices = pool.sample(1000, "tactics")
    assert 0.03 < sum(c.kind == "scripted" for c in choices) / len(choices) < 0.07
    assert 0.55 < sum(c.kind == "live" for c in choices) / len(choices) < 0.65
    assert 0.30 < sum(c.kind == "tactical" for c in choices) / len(choices) < 0.40
    assert len(pool.state_dict()["snapshots"]) == 10
    pool.prune(set())
    assert len(pool.state_dict()["snapshots"]) == 8
    snapshot = pool.state_dict()
    restored = OpponentPool(seed=99)
    restored.load_state_dict(snapshot)
    assert restored.state_dict()["recent"] == snapshot["recent"]
    assert restored.sample(50, "tactics") == pool.sample(50, "tactics")


def test_distill_and_joint_mix_live_seats_with_frozen_opponents():
    pool = OpponentPool(seed=3)
    pool.add_state("tactical", {"weight": torch.zeros(1)}, {}, 200)
    choices = pool.sample(1000, "distill")
    assert 0.46 < sum(c.kind == "live" for c in choices) / len(choices) < 0.54
    assert all(c.kind in ("live", "tactical") for c in choices)
    choices = pool.sample(1000, "joint")
    assert 0.46 < sum(c.kind == "live" for c in choices) / len(choices) < 0.54
    assert all(c.kind in ("live", "tactical") for c in choices)
    pool.add_state("joint", {"weight": torch.ones(1)}, {}, 200)
    choices = pool.sample(4000, "joint")
    shares = {kind: sum(c.kind == kind for c in choices) / len(choices)
              for kind in ("live", "tactical", "joint")}
    assert 0.47 < shares["live"] < 0.53
    assert 0.10 < shares["tactical"] < 0.15
    assert 0.34 < shares["joint"] < 0.41


def skill_policy(value):
    return FrozenSkillPolicy({
        "mlp.0.weight": torch.full((14, 61), value),
        "mlp.0.bias": torch.full((14,), value),
        "obs_normalizer._mean": torch.full((1, 61), value),
        "obs_normalizer._std": torch.full((1, 61), value + 1),
    })


def test_tactical_snapshot_freezes_skills_and_restores_controller():
    base = tuple(skill_policy(0.0) for _ in range(3))
    tuned = tuple(skill_policy(float(i + 1)) for i in range(3))
    states = {name: policy.state_dict() for name, policy in
              zip(("walk", "kick", "recovery"), tuned)}
    pool = OpponentPool()
    uid = pool.add("tactical", torch.nn.Linear(1, 1), {}, 10, states)
    for policy in tuned:
        with torch.no_grad():
            policy.mlp[0].weight.zero_()
            policy.mean.zero_()
    restored = OpponentPool()
    restored.load_state_dict(pool.state_dict())
    choice = OpponentChoice("tactical", uid)
    controller = restored.skill_controller(choice, base, 2, "cpu")
    assert restored.skill_controller(choice, base, 2, torch.device("cpu")) is controller
    assert restored.skill_controller(choice, base, 3, "cpu") is not controller
    for i, policy in enumerate((controller.walk, controller.kick, controller.recover)):
        expected = skill_policy(float(i + 1))
        assert torch.equal(policy(torch.ones(2, 61)), expected(torch.ones(2, 61)))
        assert not policy.training
        assert all(not parameter.requires_grad for parameter in policy.parameters())
    assert torch.count_nonzero(base[0].mean) == 0


def test_old_snapshot_falls_back_to_base_and_prunes_controller_cache():
    base = tuple(skill_policy(float(i)) for i in range(3))
    pool = OpponentPool(capacity=1)
    uid = pool.add_state("tactical", {"weight": torch.zeros(1)}, {}, 0)
    choice = OpponentChoice("tactical", uid)
    controller = pool.skill_controller(choice, base, 2, "cpu")
    assert controller.walk is not base[0]
    assert torch.equal(controller.walk(torch.ones(2, 61)), base[0](torch.ones(2, 61)))
    pool.add_state("tactical", {"weight": torch.zeros(1)}, {}, 1)
    pool.prune({uid})
    assert pool.skill_controller(choice, base, 2, "cpu") is controller
    pool.prune(set())
    assert not pool._skill_cache
    pool.load_state_dict(pool.state_dict())
    assert not pool._skill_cache
