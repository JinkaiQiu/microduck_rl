"""RSL interface over solo practice and two-player football matches.

Training exposes both seats as rollout rows. In live self-play both actions are
executed and learned from; a frozen opponent's sampled row is masked from PPO.
Evaluation and keyboard play retain their single-seat interface.
"""

from __future__ import annotations

from collections import deque
from typing import Literal

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from mjlab_microduck.football.appearance import FootballAppearance
from mjlab_microduck.football.arena import FootballArena, MATCH_STEPS
from mjlab_microduck.football.pool import OpponentChoice, OpponentPool, actor_input
from mjlab_microduck.football.skills import FootballSkills, KICK_BALL_OFFSET_BODY_XY


Stage = Literal["solo", "tactics", "distill", "joint"]
WIN_RATE_OPPONENTS = ("scripted", "tactical", "joint", "live")
WIN_RATE_WINDOW = 50


def scripted_tactics(
    actor_obs: torch.Tensor, root_speed: torch.Tensor | None = None,
) -> torch.Tensor:
    """Approach the ball at the right-foot kick skill's trained offset."""
    ball = actor_obs[:, 48:50]
    goal = actor_obs[:, 56:58]
    # The frozen kick actor is blind to the ball. Its training ball was placed
    # 9 cm ahead and 4.2 cm to the right of the body, so merely getting near
    # the ball can trap the baseline in repeated kicks without contact.
    offset = ball.new_tensor(KICK_BALL_OFFSET_BODY_XY)
    error = ball - offset
    shot = goal - ball
    angle_to_goal = torch.atan2(shot[:, 1], shot[:, 0])
    angle_to_ball = torch.atan2(ball[:, 1], ball[:, 0])
    yaw = torch.where(ball.norm(dim=-1) < 0.22, angle_to_goal, angle_to_ball)
    aligned = (error[:, 0].abs() < 0.018) & (error[:, 1].abs() < 0.018)
    action = torch.zeros((len(actor_obs), 4), device=actor_obs.device)
    # The walking skill has a low-speed dead zone near the ball. Keep a
    # meaningful command until the blind kick's small placement window is met.
    forward = error[:, 0].sign() * (error[:, 0].abs() * 2.0).clamp(0.20, 0.4)
    lateral = error[:, 1].sign() * (error[:, 1].abs() * 2.0).clamp(0.16, 0.3)
    action[:, 0] = torch.where(error[:, 0].abs() > 0.015, forward, 0.0)
    action[:, 1] = torch.where(error[:, 1].abs() > 0.015, lateral, 0.0)
    action[:, 2] = (yaw * 2.0).clamp(-1.0, 1.0)
    # BallKick was trained to start from standing. Let walking settle before
    # switching skills; a running approach can swing past the ball.
    settled = torch.ones(len(actor_obs), device=actor_obs.device, dtype=torch.bool)
    if root_speed is not None:
        if root_speed.shape != (len(actor_obs),):
            raise ValueError("root_speed must have one value per actor")
        settled = root_speed < 0.06
    action[:, 3] = torch.where(
        aligned & (angle_to_goal.abs() < 0.25) & settled, 1.0, -1.0,
    )
    return action


def scripted_dribble(actor_obs: torch.Tensor) -> torch.Tensor:
    """Walk through the ball after the frozen kick skill fails to move it."""
    ball = actor_obs[:, 48:50]
    action = torch.zeros((len(actor_obs), 4), device=actor_obs.device)
    angle = torch.atan2(ball[:, 1], ball[:, 0])
    action[:, 0] = torch.where(ball[:, 0] > 0.02, 0.4, 0.0)
    action[:, 1] = (ball[:, 1] * 3.0).clamp(-0.3, 0.3)
    action[:, 2] = (angle * 2.0).clamp(-1.0, 1.0)
    action[:, 3] = -1.0
    return action


class ScriptedKickGuard:
    """Switch scripted control to walking after a missed kick or near-ball stall."""

    def __init__(self, num_envs: int, device: torch.device, kick_steps: int) -> None:
        self.kick_steps = kick_steps
        self.timeout_steps = min(50, kick_steps)
        self.origin = torch.zeros((num_envs, 2), device=device)
        self.tracking = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.failed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.stalled = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.stall_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.last_root = torch.zeros((num_envs, 2), device=device)
        self.last_ball = torch.zeros((num_envs, 2), device=device)
        self.seen = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, ids: torch.Tensor) -> None:
        self.tracking[ids] = False
        self.failed[ids] = False
        self.stalled[ids] = False
        self.stall_steps[ids] = 0
        self.seen[ids] = False

    def before_action(
        self, ids: torch.Tensor, ball_xy: torch.Tensor, remaining: torch.Tensor,
        root_xy: torch.Tensor | None = None,
    ) -> None:
        if root_xy is not None:
            if root_xy.shape != ball_xy.shape:
                raise ValueError("root_xy must match ball_xy")
            near = (ball_xy - root_xy).norm(dim=-1) < 0.2
            root_still = (root_xy - self.last_root[ids]).norm(dim=-1) < 0.001
            ball_still = (ball_xy - self.last_ball[ids]).norm(dim=-1) < 0.001
            stuck = self.seen[ids] & near & root_still & ball_still
            self.stall_steps[ids] = torch.where(
                stuck, self.stall_steps[ids] + 1, 0,
            )
            self.stalled[ids] |= self.stall_steps[ids] >= 50
            self.last_root[ids] = root_xy
            self.last_ball[ids] = ball_xy
            self.seen[ids] = True
        tracked = self.tracking[ids]
        moved = (ball_xy - self.origin[ids]).norm(dim=-1) >= 0.015
        timed_out = remaining[ids] == 0
        timed_out |= (self.kick_steps - remaining[ids]) >= self.timeout_steps
        failed = tracked & timed_out & ~moved
        self.failed[ids[failed]] = True
        remaining[ids[failed]] = 0
        self.tracking[ids[tracked & (moved | timed_out)]] = False
        fallback = self.failed[ids] | self.stalled[ids]
        remaining[ids[fallback]] = 0
        self.tracking[ids[fallback]] = False

    def after_action(
        self, ids: torch.Tensor, ball_xy: torch.Tensor,
        previous_remaining: torch.Tensor, remaining: torch.Tensor,
    ) -> None:
        started = (previous_remaining == 0) & (remaining[ids] > 0)
        self.origin[ids[started]] = ball_xy[started]
        self.tracking[ids[started]] = True


class FootballVecEnv(VecEnv):
    """Route solo practice or one or two learning seats per match."""

    def __init__(
        self,
        arena: FootballArena,
        stage: Stage,
        pool: OpponentPool | None,
        skill_policies: tuple,
        teacher_model: torch.nn.Module | None = None,
        seed: int = 42,
        dual_rollouts: bool = False,
    ) -> None:
        expected_players = 1 if stage == "solo" else 2
        if arena.num_players != expected_players:
            raise ValueError(f"{stage} stage requires {expected_players} physical player(s)")
        if stage == "distill" and teacher_model is None:
            raise ValueError("distillation requires a frozen tactical teacher")
        if stage != "solo" and pool is None:
            raise ValueError("two-player stages require an opponent pool")
        if stage == "solo" and dual_rollouts:
            raise ValueError("solo practice has only one rollout seat")
        self.arena = arena
        self.env = arena.env
        self.stage = stage
        self.pool = pool
        self.teacher_model = teacher_model
        self.device = torch.device(arena.device)
        self.num_matches = arena.num_envs
        self.dual_rollouts = dual_rollouts
        self.num_envs = 2 * self.num_matches if dual_rollouts else self.num_matches
        self.num_actions = 4 if stage in ("solo", "tactics") else 14
        self.max_episode_length = MATCH_STEPS
        self.cfg = self.env.cfg
        self.learner_skills = FootballSkills(*skill_policies, self.num_matches, device=self.device)
        if stage != "solo":
            self.learner_home = torch.zeros(self.num_matches, device=self.device, dtype=torch.bool)
            self.opponent_choices = [OpponentChoice("scripted") for _ in range(self.num_matches)]
            self._rng = torch.Generator(device="cpu")
            self._rng.manual_seed(seed)
            self.opponent_skills = FootballSkills(*skill_policies, self.num_matches, device=self.device)
            self.teacher_skills = FootballSkills(*skill_policies, self.num_envs, device=self.device)
            self.scripted_kick_guard = ScriptedKickGuard(
                self.num_matches, self.device, self.opponent_skills.kick_steps,
            )
        self._last_obs: TensorDict | None = None
        self.recent_match_points: deque[float] = deque(maxlen=WIN_RATE_WINDOW)
        if stage != "solo":
            self.recent_points_by_opponent: dict[str, deque[float]] = {
                kind: deque(maxlen=WIN_RATE_WINDOW) for kind in WIN_RATE_OPPONENTS
            }
            self.appearance = FootballAppearance(self.env.sim.mj_model)
        self.reset()

    @property
    def unwrapped(self):
        return self.env

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.arena.elapsed.repeat(2) if self.dual_rollouts else self.arena.elapsed

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        if self.dual_rollouts:
            if value.shape != (self.num_envs,) or not torch.equal(value[:self.num_matches], value[self.num_matches:]):
                raise ValueError("both rollout seats must have the same match clock")
            value = value[:self.num_matches]
        self.arena.elapsed.copy_(value)

    def refresh_appearance(self) -> None:
        """Match environment 0's visual colors to its current policy roles."""
        self.appearance.set_roles(
            bool(self.learner_home[0].item()), self.opponent_choices[0].kind,
        )

    def _choose_matches(self, ids: torch.Tensor) -> None:
        if self.stage == "solo":
            self.learner_skills.reset(ids)
            return
        if ids.numel() == 0:
            return
        sides = torch.randint(0, 2, (len(ids),), generator=self._rng).bool()
        self.learner_home[ids] = sides.to(self.device)
        choices = self.pool.sample(len(ids), self.stage)
        for env_id, choice in zip(ids.tolist(), choices, strict=True):
            # Evaluation and keyboard play expose a single controlled seat.
            self.opponent_choices[env_id] = (
                OpponentChoice("scripted") if choice.kind == "live" and not self.dual_rollouts
                else choice
            )
        if (ids == 0).any():
            self.refresh_appearance()
        self.learner_skills.reset(ids)
        self.opponent_skills.reset(ids)
        teacher_ids = torch.cat((ids, ids + self.num_matches)) if self.dual_rollouts else ids
        self.teacher_skills.reset(teacher_ids)
        self.scripted_kick_guard.reset(ids)
        self.pool.prune({choice.uid for choice in self.opponent_choices if choice.uid >= 0})

    def _select(self, home: torch.Tensor, away: torch.Tensor, learner: bool) -> torch.Tensor:
        use_home = self.learner_home if learner else ~self.learner_home
        return torch.where(use_home[:, None], home, away)

    def _selected_obs61(self, learner: bool) -> torch.Tensor:
        return self._select(self.arena.obs61(0), self.arena.obs61(1), learner)

    def _selected_ball_xy(self, learner: bool) -> torch.Tensor:
        return self._select(self.arena.ball_body_xy(0), self.arena.ball_body_xy(1), learner)

    def get_observations(self) -> TensorDict:
        if self.stage == "solo":
            self._last_obs = TensorDict({
                "actor": self.arena.actor_obs(0),
                "critic": self.arena.critic_obs(0),
            }, batch_size=[self.num_envs])
            return self._last_obs
        first_actor = self._select(self.arena.actor_obs(0), self.arena.actor_obs(1), True)
        first_critic = self._select(self.arena.critic_obs(0), self.arena.critic_obs(1), True)
        if self.dual_rollouts:
            actor = torch.cat((first_actor, self._select(
                self.arena.actor_obs(0), self.arena.actor_obs(1), False,
            )), dim=0)
            critic = torch.cat((first_critic, self._select(
                self.arena.critic_obs(0), self.arena.critic_obs(1), False,
            )), dim=0)
        else:
            actor, critic = first_actor, first_critic
        self._last_obs = TensorDict({"actor": actor, "critic": critic}, batch_size=[self.num_envs])
        return self._last_obs

    def reset(self) -> tuple[TensorDict, dict]:
        self.arena.reset()
        ids = torch.arange(self.num_matches, device=self.device, dtype=torch.long)
        self._choose_matches(ids)
        return self.get_observations(), {}

    @torch.no_grad()
    def teacher_targets(self, pre_action_obs: TensorDict) -> torch.Tensor:
        """Label both student-visited seats; PPO masks frozen-seat labels."""
        if self.teacher_model is None:
            raise RuntimeError("teacher target requested outside distillation")
        tactics = self.teacher_model(TensorDict(
            {"actor": actor_input(self.teacher_model, pre_action_obs["actor"])}, batch_size=[self.num_envs],
        ))
        obs61 = self._selected_obs61(True)
        ball_xy = self._selected_ball_xy(True)
        if self.dual_rollouts:
            obs61 = torch.cat((obs61, self._selected_obs61(False)), dim=0)
            ball_xy = torch.cat((ball_xy, self._selected_ball_xy(False)), dim=0)
        return self.teacher_skills.act(tactics, obs61, ball_xy)

    @torch.no_grad()
    def _opponent_action(self, opponent_obs: torch.Tensor,
                         live_actions: torch.Tensor | None = None) -> torch.Tensor:
        out = torch.empty((self.num_matches, 14), device=self.device)
        tactics = torch.zeros((self.num_matches, 4), device=self.device)
        tactical_ids: list[int] = []
        scripted_ids: list[int] = []
        ball_xy = self.env.scene["ball"].data.root_link_pos_w[:, :2]
        opponent_velocity = self._select(
            self.env.scene["home"].data.root_link_lin_vel_w[:, :2],
            self.env.scene["away"].data.root_link_lin_vel_w[:, :2], False,
        )
        opponent_speed = opponent_velocity.norm(dim=-1)
        opponent_root = self._select(
            self.env.scene["home"].data.root_link_pos_w[:, :2],
            self.env.scene["away"].data.root_link_pos_w[:, :2], False,
        )
        groups: dict[tuple[str, int], list[int]] = {}
        for env_id, choice in enumerate(self.opponent_choices):
            groups.setdefault((choice.kind, choice.uid), []).append(env_id)
        for (kind, uid), members in groups.items():
            ids = torch.as_tensor(members, device=self.device, dtype=torch.long)
            if kind == "live":
                if live_actions is None:
                    raise RuntimeError("a live opponent needs its sampled policy action")
                if self.stage == "tactics":
                    tactics[ids] = live_actions[ids]
                    tactical_ids.extend(members)
                else:
                    out[ids] = live_actions[ids]
                continue
            if kind == "scripted":
                self.scripted_kick_guard.before_action(
                    ids, ball_xy[ids], self.opponent_skills.kick_remaining,
                    opponent_root[ids],
                )
                tactics[ids] = scripted_tactics(opponent_obs[ids], opponent_speed[ids])
                fallback = (
                    self.scripted_kick_guard.failed[ids]
                    | self.scripted_kick_guard.stalled[ids]
                )
                tactics[ids[fallback]] = scripted_dribble(opponent_obs[ids[fallback]])
                tactical_ids.extend(members)
                scripted_ids.extend(members)
                continue
            model = self.pool.model(OpponentChoice(kind, uid), self.device)
            values = model(TensorDict({"actor": actor_input(model, opponent_obs[ids])}, batch_size=[len(ids)]))
            if kind == "tactical":
                tactics[ids] = values
                tactical_ids.extend(members)
            else:
                out[ids] = values
        if tactical_ids:
            ids = torch.as_tensor(sorted(tactical_ids), device=self.device, dtype=torch.long)
            scripted = torch.as_tensor(scripted_ids, device=self.device, dtype=torch.long)
            scripted_previous = self.opponent_skills.kick_remaining[scripted].clone()
            out[ids] = self.opponent_skills.act(
                tactics[ids], self._selected_obs61(False)[ids],
                self._selected_ball_xy(False)[ids], ids,
            )
            self.scripted_kick_guard.after_action(
                scripted, ball_xy[scripted], scripted_previous,
                self.opponent_skills.kick_remaining,
            )
        return out

    def _record_match_results(self, ids: torch.Tensor, points: torch.Tensor) -> None:
        """Update match-level windows before opponents are resampled."""
        results = points[ids].detach().cpu().tolist()
        for env_id, result in zip(ids.tolist(), results, strict=True):
            kind = self.opponent_choices[env_id].kind
            self.recent_points_by_opponent[kind].append(result)
            if kind != "live":
                self.recent_match_points.append(result)

    def _win_rate_logs(self) -> dict[str, float]:
        """Raw win fractions; a draw is not a win, and empty windows have count 0."""
        def rate(results: deque[float]) -> float:
            return sum(result == 1.0 for result in results) / len(results) if results else 0.0

        logs = {
            "info/benchmark_win_rate": rate(self.recent_match_points),
            "info/benchmark_matches": float(len(self.recent_match_points)),
        }
        for kind, results in self.recent_points_by_opponent.items():
            logs[f"info/win_rate_vs_{kind}"] = rate(results)
            logs[f"info/matches_vs_{kind}"] = float(len(results))
        return logs

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(f"expected learner actions {(self.num_envs, self.num_actions)}")
        if self.stage == "solo":
            return self._step_solo(actions)
        actions = actions.to(self.device)
        n = self.num_matches
        primary_actions = actions[:n]
        secondary_actions = actions[n:] if self.dual_rollouts else None
        live_second = torch.as_tensor(
            [choice.kind == "live" for choice in self.opponent_choices],
            device=self.device, dtype=torch.bool,
        )
        live_mask = (
            torch.cat((torch.ones(n, device=self.device, dtype=torch.bool), live_second))
            if self.dual_rollouts else torch.ones(n, device=self.device, dtype=torch.bool)
        )
        opponent_obs = self._select(self.arena.actor_obs(0), self.arena.actor_obs(1), False)
        with torch.no_grad():
            if self.stage == "tactics":
                learner_joint = self.learner_skills.act(
                    primary_actions, self._selected_obs61(True), self._selected_ball_xy(True)
                )
            else:
                learner_joint = primary_actions
            opponent_joint = self._opponent_action(opponent_obs, secondary_actions)
            learner_ready = (
                self.learner_skills.kick_remaining == 0 if self.stage == "tactics"
                else self.teacher_skills.kick_remaining[:n] == 0 if self.stage == "distill"
                else torch.ones(n, device=self.device, dtype=torch.bool)
            )
            opponent_ready = self.opponent_skills.kick_remaining == 0
            direct_opponents = torch.as_tensor(
                [choice.kind == "joint" or (choice.kind == "live" and self.stage != "tactics")
                 for choice in self.opponent_choices],
                device=self.device, dtype=torch.bool,
            )
            opponent_ready |= direct_opponents
            home_ready = torch.where(self.learner_home, learner_ready, opponent_ready)
            away_ready = torch.where(self.learner_home, opponent_ready, learner_ready)
            self.arena.set_kick_available(torch.stack((home_ready, away_ready), dim=1).float())
            home = torch.where(self.learner_home[:, None], learner_joint, opponent_joint)
            away = torch.where(self.learner_home[:, None], opponent_joint, learner_joint)
            _, side_rewards, dones, info = self.arena.step(torch.cat((home, away), dim=-1))
            primary_rewards = self._select(
                side_rewards[0][:, None], side_rewards[1][:, None], True,
            ).squeeze(-1)
            if self.dual_rollouts:
                secondary_rewards = self._select(
                    side_rewards[0][:, None], side_rewards[1][:, None], False,
                ).squeeze(-1)
                rewards = torch.cat((primary_rewards, secondary_rewards))
                rollout_dones = dones.repeat(2)
            else:
                rewards, rollout_dones = primary_rewards, dones
            # Use the pre-action seat assignment. _choose_matches may change it
            # below before PPO receives the mask and logged reward components.
            reward_component_logs = {}
            for name, values in info["reward_components"].items():
                primary = self._select(values[:, :1], values[:, 1:], True).squeeze(-1)
                if self.dual_rollouts:
                    secondary = self._select(values[:, :1], values[:, 1:], False).squeeze(-1)
                    selected = torch.cat((primary, secondary))
                else:
                    selected = primary
                reward_component_logs[f"reward/{name}"] = selected[live_mask].mean()
            reward_component_logs["reward/total"] = rewards[live_mask].mean()
            scores = info["scores"]
            own = self._select(scores[:, :1], scores[:, 1:], True).squeeze(-1)
            other = self._select(scores[:, :1], scores[:, 1:], False).squeeze(-1)
            points = (own > other).float() + 0.5 * (own == other).float()
            out_of_bounds = info["out_of_bounds"]
            own_out = self._select(out_of_bounds[:, :1], out_of_bounds[:, 1:], True).squeeze(-1)
            other_out = self._select(out_of_bounds[:, :1], out_of_bounds[:, 1:], False).squeeze(-1)
            points = torch.where(own_out | other_out,
                                 torch.where(own_out, 0.5 * other_out.float(), 1.0), points)
            numerical_reset = info["numerical_reset"]
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            completed_ids = (dones & ~numerical_reset).nonzero(as_tuple=False).squeeze(-1)
            benchmark_ids = (dones & ~numerical_reset & ~live_second).nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel():
                # Live-vs-live points are tautological; checkpoint selection
                # uses only matches against scripted or frozen opponents.
                self._record_match_results(completed_ids, points)
                self.arena.reset(completed_ids)
                self._choose_matches(done_ids)
            obs = self.get_observations()
        extras = {
            "learner_mask": live_mask,
            "log": {
                "info/learner_goals": own[benchmark_ids].float().mean().item() if benchmark_ids.numel() else 0.0,
                "info/match_points": points[benchmark_ids].mean().item() if benchmark_ids.numel() else 0.0,
                "info/numerical_resets": numerical_reset.float().sum().item(),
                "info/match_timeouts": info["timed_out"].float().sum().item(),
                "info/live_match_fraction": live_second.float().mean(),
                "info/live_rollout_fraction": live_mask.float().mean(),
                **self._win_rate_logs(),
                **reward_component_logs,
            },
            "completed": {
                "env_ids": completed_ids.clone(),
                "points": points[completed_ids].clone(),
                "own_goals": own[completed_ids].clone(),
                "opponent_goals": other[completed_ids].clone(),
                "own_out_of_bounds": own_out[completed_ids].clone(),
                "opponent_out_of_bounds": other_out[completed_ids].clone(),
                "timed_out": info["timed_out"][completed_ids].clone(),
            },
        }
        return obs, rewards, rollout_dones.long(), extras

    def _step_solo(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        with torch.no_grad():
            joint = self.learner_skills.act(
                actions.to(self.device), self.arena.obs61(0), self.arena.ball_body_xy(0),
            )
            self.arena.set_kick_available((self.learner_skills.kick_remaining == 0).float()[:, None])
            _, side_rewards, dones, info = self.arena.step(joint)
            rewards = side_rewards[0]
            numerical = info["numerical_reset"]
            completed = (dones & ~numerical).nonzero(as_tuple=False).squeeze(-1)
            goals = info["goals"][:, 0]
            if completed.numel():
                self.recent_match_points.extend(goals[completed].float().cpu().tolist())
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if completed.numel():
                self.arena.reset(completed)
            if done_ids.numel():
                self.learner_skills.reset(done_ids)
            obs = self.get_observations()
        logs = {
            "info/solo_goal_rate": sum(self.recent_match_points) / len(self.recent_match_points)
            if self.recent_match_points else 0.0,
            "info/solo_episodes": float(len(self.recent_match_points)),
            "info/match_timeouts": info["timed_out"].float().sum().item(),
            "info/numerical_resets": numerical.float().sum().item(),
            "reward/total": rewards.mean(),
        }
        logs.update({f"reward/{name}": value[:, 0].mean() for name, value in info["reward_components"].items()})
        return obs, rewards, dones.long(), {
            "log": logs,
            "completed": {
                "env_ids": completed.clone(),
                "points": goals[completed].float().clone(),
                "timed_out": info["timed_out"][completed].clone(),
            },
        }

    def state_dict(self) -> dict:
        if self.stage == "solo":
            return {"recent_match_points": list(self.recent_match_points)}
        return {
            "pool": self.pool.state_dict(),
            "rng_state": self._rng.get_state().clone(),
            "recent_match_points": list(self.recent_match_points),
            "recent_points_by_opponent": {
                kind: list(results) for kind, results in self.recent_points_by_opponent.items()
            },
            "curriculum": {"tactical_bootstrap_complete": self.pool.tactical_count > 0},
        }

    def load_state_dict(self, state: dict) -> None:
        if self.stage == "solo":
            self.recent_match_points.clear()
            self.recent_match_points.extend(state.get("recent_match_points", ()))
            self.reset()
            return
        self.pool.load_state_dict(state["pool"])
        self._rng.set_state(state["rng_state"].cpu())
        self.recent_match_points.clear()
        self.recent_match_points.extend(state.get("recent_match_points", ()))
        saved_windows = state.get("recent_points_by_opponent", {})
        for kind, results in self.recent_points_by_opponent.items():
            results.clear()
            results.extend(saved_windows.get(kind, ()))
        self.reset()

    def close(self) -> None:
        self.env.close()
