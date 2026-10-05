"""Opponent league sampling and checkpoint state invariants."""

import torch

from mjlab_microduck.football.pool import OpponentPool


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
