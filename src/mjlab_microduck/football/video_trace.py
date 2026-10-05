"""World-state sidecars for one- and two-duck football training videos.

Each MP4 has a same-stem JSONL file. Rows describe environment 0 after its
physics step and before FootballArena updates scores or resets a finished match.
Actions are 14 joint-position offsets per physical duck.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch
from mjlab.utils.wrappers import VideoRecorder

from mjlab_microduck.football.arena import field_out_of_bounds

if TYPE_CHECKING:
    from mjlab_microduck.football.wrapper import FootballVecEnv


def _values(tensor: torch.Tensor) -> list[float]:
    return tensor[0].detach().cpu().tolist()


class FootballVideoRecorder(VideoRecorder):
    """Record env-0 MP4 frames and aligned world-state traces."""

    football_env: FootballVecEnv | None = None

    def _start_recording(self) -> None:
        super()._start_recording()
        self._trace_rows: list[dict] = []

    def step(self, action: torch.Tensor):
        # Defer CPU transfer until a frame is actually captured.
        self._trace_action = action[0].detach().clone()
        return super().step(action)

    def _record_frame(self) -> None:
        # The renderer runs before FootballArena can reset a numerical failure.
        data = self._wrapped_env.sim.data
        if any(not torch.isfinite(values[0]).all() for values in
               (data.qpos, data.qvel, data.qacc, data.qacc_warmstart, data.sensordata)):
            return
        count = len(self.current_video_frames)
        super()._record_frame()
        if len(self.current_video_frames) == count:
            return
        if self.football_env is None:
            raise RuntimeError("FootballVideoRecorder.football_env was not attached")

        football = self.football_env
        arena = football.arena
        scene = self._wrapped_env.scene
        fps = self._wrapped_env.metadata.get("render_fps", 30)

        def entity(name: str) -> dict:
            state = scene[name].data
            return {
                "position_w": _values(state.root_link_pos_w),
                "quaternion_wxyz_w": _values(state.root_link_quat_w),
                "linear_velocity_w": _values(state.root_link_lin_vel_w),
                "angular_velocity_w": _values(state.root_link_ang_vel_w),
            }

        origin = _values(scene.env_origins)

        def outside(player: dict) -> bool:
            x, y = player["position_w"][:2]
            relative = torch.tensor((x - origin[0], y - origin[1]))
            return bool(field_out_of_bounds(
                relative, arena.field_half_length[0], arena.field_half_width[0],
            ).item())

        actions = self._trace_action.cpu().tolist()
        home_action = actions[:14]
        home, ball = entity("home"), entity("ball")
        row = {
            "frame_index": count,
            "video_time_s": count / fps,
            "global_control_step": self.step_count + 1,
            "simulation_time_s": (self.step_count + 1) * self._wrapped_env.step_dt,
            "env_index": 0,
            "env_origin_w": origin,
            "field_half_length_m": float(arena.field_half_length[0].item()),
            "field_half_width_m": float(arena.field_half_width[0].item()),
            "home_out_of_bounds": outside(home),
            "match_elapsed_pre_step": int(arena.elapsed[0].item()),
            "home_attack_sign": float(arena.attack_sign[0, 0].item()),
            "home_joint_action": home_action,
            "ball": ball,
            "home": home,
        }
        if arena.num_players == 1:
            row.update({
                "learner_joint_action": home_action,
                "learner_kick_steps_remaining": int(
                    football.learner_skills.kick_remaining[0].item()
                ),
            })
        else:
            away_action = actions[14:]
            away = entity("away")
            learner_home = bool(football.learner_home[0].item())
            opponent = football.opponent_choices[0]
            row.update({
                "away_out_of_bounds": outside(away),
                "learner_home": learner_home,
                "opponent_kind": opponent.kind,
                "opponent_uid": opponent.uid,
                "scores_pre_step": _values(arena.scores),
                "away_attack_sign": float(arena.attack_sign[0, 1].item()),
                "away_joint_action": away_action,
                "learner_joint_action": home_action if learner_home else away_action,
                "opponent_joint_action": away_action if learner_home else home_action,
                "learner_kick_steps_remaining": (
                    int(football.learner_skills.kick_remaining[0].item())
                    if football.stage == "tactics" else None
                ),
                "opponent_kick_steps_remaining": (
                    int(football.opponent_skills.kick_remaining[0].item())
                    if opponent.kind in ("scripted", "tactical")
                    or (opponent.kind == "live" and football.stage == "tactics") else None
                ),
                "scripted_kick_failed": (
                    bool(football.scripted_kick_guard.failed[0].item())
                    if opponent.kind == "scripted" else None
                ),
                "scripted_near_ball_stalled": (
                    bool(football.scripted_kick_guard.stalled[0].item())
                    if opponent.kind == "scripted" else None
                ),
                "away": away,
            })
        self._trace_rows.append(row)

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
