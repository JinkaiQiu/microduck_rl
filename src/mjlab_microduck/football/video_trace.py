"""World-state sidecars for football training videos.

Each ``.mp4`` has a same-stem ``.jsonl`` file with one row per encoded frame.
Rows describe environment 0 immediately after its physics step and before
FootballArena updates goals or resets a finished match. ``frame_index`` is
zero-based, ``video_time_s`` is its timestamp within the clip, and
``global_control_step`` is one-based (including resumed training steps).
Positions and velocities are in MuJoCo world coordinates. Quaternions use
``[w, x, y, z]`` order. Actions are 14 joint-position offsets per duck.
``match_elapsed_pre_step`` and ``scores_pre_step`` deliberately name the
pre-bookkeeping values; physical positions and velocities are post-step.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch
from mjlab.utils.wrappers import VideoRecorder

from mjlab_microduck.football.arena import FIELD_HALF_LENGTH, FIELD_HALF_WIDTH

if TYPE_CHECKING:
    from mjlab_microduck.football.wrapper import FootballVecEnv


def _values(tensor: torch.Tensor) -> list[float]:
    return tensor[0].detach().cpu().tolist()


class FootballVideoRecorder(VideoRecorder):
    """Record an env-0 JSONL trace at the exact captured-frame boundary."""

    football_env: FootballVecEnv | None = None

    def _start_recording(self) -> None:
        super()._start_recording()
        self._trace_rows: list[dict] = []

    def step(self, action: torch.Tensor):
        # This is the physical 28D action passed by FootballArena, ordered
        # home then away. The parent calls _record_frame before returning.
        # Keep the 28D action on its device. A CPU conversion on every PPO
        # control step would synchronize the GPU even between video clips.
        self._trace_action = action[0].detach().clone()
        return super().step(action)

    def _record_frame(self) -> None:
        # VideoRecorder renders inside the physics step, before FootballArena
        # can reset a failed world. Never send invalid env-0 state to MuJoCo's
        # renderer or serialize it in the JSONL trace.
        data = self._wrapped_env.sim.data
        if any(not torch.isfinite(values[0]).all() for values in
               (data.qpos, data.qvel, data.qacc, data.qacc_warmstart, data.sensordata)):
            return
        count = len(self.current_video_frames)
        super()._record_frame()
        if len(self.current_video_frames) == count:
            return  # A render failure produced no MP4 frame.
        if self.football_env is None:
            raise RuntimeError("FootballVideoRecorder.football_env was not attached")
        football = self.football_env
        arena = football.arena
        scene = self._wrapped_env.scene
        fps = self._wrapped_env.metadata.get("render_fps", 30)
        learner_home = bool(football.learner_home[0].item())
        opponent = football.opponent_choices[0]
        actions = self._trace_action.cpu().tolist()
        home_action, away_action = actions[:14], actions[14:]

        def entity(name: str) -> dict:
            data = scene[name].data
            return {
                "position_w": _values(data.root_link_pos_w),
                "quaternion_wxyz_w": _values(data.root_link_quat_w),
                "linear_velocity_w": _values(data.root_link_lin_vel_w),
                "angular_velocity_w": _values(data.root_link_ang_vel_w),
            }

        origin = _values(scene.env_origins)
        home, away, ball = entity("home"), entity("away"), entity("ball")

        def outside(player: dict) -> bool:
            x, y = player["position_w"][:2]
            return abs(x - origin[0]) > FIELD_HALF_LENGTH or abs(y - origin[1]) > FIELD_HALF_WIDTH

        self._trace_rows.append({
            "frame_index": count,
            "video_time_s": count / fps,
            "global_control_step": self.step_count + 1,
            "simulation_time_s": (self.step_count + 1) * self._wrapped_env.step_dt,
            "env_index": 0,
            "env_origin_w": origin,
            "home_out_of_bounds": outside(home),
            "away_out_of_bounds": outside(away),
            "learner_home": learner_home,
            "opponent_kind": opponent.kind,
            "opponent_uid": opponent.uid,
            "match_elapsed_pre_step": int(arena.elapsed[0].item()),
            "scores_pre_step": _values(arena.scores),
            "home_attack_sign": float(arena.attack_sign[0, 0].item()),
            "away_attack_sign": float(arena.attack_sign[0, 1].item()),
            "home_joint_action": home_action,
            "away_joint_action": away_action,
            "learner_joint_action": home_action if learner_home else away_action,
            "opponent_joint_action": away_action if learner_home else home_action,
            "learner_kick_steps_remaining": (
                int(football.learner_skills.kick_remaining[0].item())
                if football.stage == "tactics" else None
            ),
            "opponent_kick_steps_remaining": (
                int(football.opponent_skills.kick_remaining[0].item())
                if opponent.kind != "joint" else None
            ),
            "scripted_kick_failed": (
                bool(football.scripted_kick_guard.failed[0].item())
                if opponent.kind == "scripted" else None
            ),
            "scripted_near_ball_stalled": (
                bool(football.scripted_kick_guard.stalled[0].item())
                if opponent.kind == "scripted" else None
            ),
            "ball": ball,
            "home": home,
            "away": away,
        })

    def _finish_recording(self) -> None:
        video_path = self.current_video_path
        rows = self._trace_rows
        super()._finish_recording()
        if video_path is not None and rows and video_path.is_file():
            trace_path = video_path.with_suffix(".jsonl")
            with trace_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
        self._trace_rows = []
