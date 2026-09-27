"""Frame-level trace stays aligned with train video boundaries."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mjlab_microduck.football.pool import OpponentChoice
from mjlab_microduck.football.video_trace import FootballVideoRecorder


def test_trace_matches_captured_frames_and_resume_step(tmp_path: Path, monkeypatch):
    class FakeEnv:
        render_mode = "rgb_array"
        metadata = {"render_fps": 50}
        step_dt = 0.02

        def __init__(self):
            self.tick = 0
            self.sim = SimpleNamespace(data=SimpleNamespace(**{
                name: torch.zeros(1, 3) for name in
                ("qpos", "qvel", "qacc", "qacc_warmstart", "sensordata")
            }))
            self.scene = {name: SimpleNamespace(data=SimpleNamespace(
                root_link_pos_w=torch.zeros(1, 3),
                root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]),
                root_link_lin_vel_w=torch.zeros(1, 3),
                root_link_ang_vel_w=torch.zeros(1, 3),
            )) for name in ("ball", "home", "away")}
            self.scene = dict(self.scene)
            self.scene["ball"].data.root_link_pos_w[0, 2] = 0.035
            self.scene = SimpleScene(self.scene)

        def step(self, action):
            self.tick += 1
            self.sim.data.qpos[0, 0] = float("nan") if self.tick == 2 else 0.0
            self.scene["home"].data.root_link_pos_w[0, 0] = self.tick
            return None, None, torch.tensor([False]), torch.tensor([False]), {}

        def render(self):
            assert torch.isfinite(self.sim.data.qpos).all()
            return np.full((1, 64, 64, 3), self.tick, dtype=np.uint8)

        def close(self):
            pass

    class SimpleScene(dict):
        env_origins = torch.zeros(1, 3)

    written = []

    def fake_write_video(path, frames, fps):
        written.append((path, len(frames), fps))
        Path(path).write_bytes(b"encoded")

    monkeypatch.setattr("mjlab.utils.wrappers.video_recorder.media.write_video", fake_write_video)
    env = FakeEnv()
    recorder = FootballVideoRecorder(env, tmp_path, step_trigger=lambda step: step == 100,
                                     video_length=2, name_prefix="football-tactics-resume-test")
    recorder.football_env = SimpleNamespace(
        arena=SimpleNamespace(elapsed=torch.tensor([5]), scores=torch.zeros(1, 2),
                              attack_sign=torch.tensor([[1., -1.]])),
        learner_home=torch.tensor([True]), opponent_choices=[OpponentChoice("scripted")],
        learner_skills=SimpleNamespace(kick_remaining=torch.tensor([7])),
        opponent_skills=SimpleNamespace(kick_remaining=torch.tensor([3])),
        scripted_kick_guard=SimpleNamespace(
            failed=torch.tensor([False]), stalled=torch.tensor([False]),
        ),
        stage="tactics",
    )
    recorder.step_count = 100
    for _ in range(3):
        recorder.step(torch.zeros(1, 28))
    recorder.close()

    assert len(written) == 1
    assert written[0][1:] == (2, 50)
    rows = [json.loads(line) for line in (tmp_path / "football-tactics-resume-test-step-100.jsonl").read_text().splitlines()]
    assert [row["frame_index"] for row in rows] == [0, 1]
    assert [row["global_control_step"] for row in rows] == [101, 103]
    assert [row["video_time_s"] for row in rows] == [0, 0.02]
    assert [row["home"]["position_w"][0] for row in rows] == [1, 3]
    assert [row["home_out_of_bounds"] for row in rows] == [False, True]
    assert all(row["opponent_kind"] == "scripted" for row in rows)
    assert all(row["scripted_kick_failed"] is False for row in rows)
    assert all(row["scripted_near_ball_stalled"] is False for row in rows)
