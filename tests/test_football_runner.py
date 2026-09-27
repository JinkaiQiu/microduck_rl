"""Exact iteration boundary for frozen self-play snapshots."""

from pathlib import Path
from types import SimpleNamespace

import torch

from mjlab_microduck.football.pool import OpponentPool
from mjlab_microduck.football.runner import FootballRunner


def test_snapshot_after_exactly_200_updates(tmp_path: Path):
    runner = FootballRunner.__new__(FootballRunner)
    runner.env = SimpleNamespace(stage="joint", pool=OpponentPool())
    runner.alg = SimpleNamespace(actor=torch.nn.Linear(2, 2))
    runner.actor_cfg = {}
    runner._log_dir = tmp_path
    saved = []
    runner.save = saved.append
    runner._after_update(199)
    assert runner.env.pool.joint_count == 0
    runner._after_update(200)
    assert runner.env.pool.joint_count == 1
    assert saved == [str(tmp_path / "snapshot_200.pt")]
    runner._after_update(201)
    assert runner.env.pool.joint_count == 1
