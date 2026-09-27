"""Opponent league sampling and checkpoint state invariants."""

import torch

from mjlab_microduck.football.pool import OpponentPool


def test_bootstrap_rolling_snapshots_and_resume():
    pool = OpponentPool(seed=7, capacity=8)
    assert all(choice.kind == "scripted" for choice in pool.sample(20, "tactics"))
    for iteration in range(10):
        pool.add_state("tactical", {"weight": torch.tensor([float(iteration)])}, {}, iteration)
    assert pool.tactical_count == 8
    assert len(pool.state_dict()["snapshots"]) == 10
    pool.prune(set())
    assert len(pool.state_dict()["snapshots"]) == 8
    snapshot = pool.state_dict()
    restored = OpponentPool(seed=99)
    restored.load_state_dict(snapshot)
    assert restored.state_dict()["recent"] == snapshot["recent"]
    assert restored.sample(50, "tactics") == pool.sample(50, "tactics")


def test_joint_pool_samples_tactical_teacher_one_quarter_of_matches():
    pool = OpponentPool(seed=3)
    pool.add_state("tactical", {"weight": torch.zeros(1)}, {}, 200)
    assert all(choice.kind == "tactical" for choice in pool.sample(20, "joint"))
    pool.add_state("joint", {"weight": torch.ones(1)}, {}, 200)
    choices = pool.sample(1000, "joint")
    tactical_fraction = sum(choice.kind == "tactical" for choice in choices) / len(choices)
    assert 0.20 < tactical_fraction < 0.30
