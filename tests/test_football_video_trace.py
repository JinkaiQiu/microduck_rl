"""Frame-level trace stays aligned with train video boundaries."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mjlab_microduck.football.arena import FIELD_HALF_LENGTH, FIELD_HALF_WIDTH
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
            self.scene["home"].data.root_link_pos_w[0, 0] = FIELD_HALF_LENGTH - 0.05
            self.scene["home"].data.root_link_pos_w[0, 1] = FIELD_HALF_WIDTH - 0.08 if self.tick == 3 else 0.0
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
        arena=SimpleNamespace(num_players=2, elapsed=torch.tensor([5]), scores=torch.zeros(1, 2),
                              attack_sign=torch.tensor([[1., -1.]]),
                              field_half_length=torch.tensor([FIELD_HALF_LENGTH]),
                              field_half_width=torch.tensor([FIELD_HALF_WIDTH])),
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
    assert [row["home"]["position_w"][0] for row in rows] == pytest.approx([FIELD_HALF_LENGTH - 0.05] * 2)
    assert [row["home_out_of_bounds"] for row in rows] == [False, True]
    assert all(row["opponent_kind"] == "scripted" for row in rows)
    assert all(row["scripted_kick_failed"] is False for row in rows)
    assert all(row["scripted_near_ball_stalled"] is False for row in rows)


def test_solo_trace_records_one_duck_and_ball(tmp_path: Path, monkeypatch):
    from mjlab_microduck.football.video_trace import FootballVideoRecorder

    class SimpleScene(dict):
        env_origins = torch.zeros(1, 3)

    class FakeEnv:
        render_mode = "rgb_array"
        metadata = {"render_fps": 50}
        step_dt = 0.02

        def __init__(self):
            self.sim = SimpleNamespace(data=SimpleNamespace(**{
                name: torch.zeros(1, 3) for name in
                ("qpos", "qvel", "qacc", "qacc_warmstart", "sensordata")
            }))
            self.scene = SimpleScene({name: SimpleNamespace(data=SimpleNamespace(
                root_link_pos_w=torch.zeros(1, 3),
                root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]),
                root_link_lin_vel_w=torch.zeros(1, 3),
                root_link_ang_vel_w=torch.zeros(1, 3),
            )) for name in ("ball", "home")})

        def step(self, action):
            return None, None, torch.tensor([False]), torch.tensor([False]), {}

        def render(self):
            return np.zeros((1, 64, 64, 3), dtype=np.uint8)

        def close(self):
            pass

    def fake_write_video(path, frames, fps):
        Path(path).write_bytes(b"encoded")

    monkeypatch.setattr("mjlab.utils.wrappers.video_recorder.media.write_video", fake_write_video)
    recorder = FootballVideoRecorder(
        FakeEnv(), tmp_path, step_trigger=lambda step: step == 0,
        video_length=1, name_prefix="football-solo-test",
    )
    recorder.football_env = SimpleNamespace(
        arena=SimpleNamespace(
            num_players=1, elapsed=torch.tensor([5]), attack_sign=torch.tensor([[1.]]),
            field_half_length=torch.tensor([FIELD_HALF_LENGTH]),
            field_half_width=torch.tensor([FIELD_HALF_WIDTH]),
        ),
        learner_skills=SimpleNamespace(kick_remaining=torch.tensor([7])),
    )
    recorder.step(torch.ones(1, 14))
    recorder.close()

    rows = [json.loads(line) for line in
            (tmp_path / "football-solo-test-step-0.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert set(("home", "ball", "home_joint_action")).issubset(rows[0])
    assert "away" not in rows[0]
    assert "opponent_kind" not in rows[0]
    assert rows[0]["home_joint_action"] == [1.] * 14
    assert rows[0]["learner_kick_steps_remaining"] == 7
