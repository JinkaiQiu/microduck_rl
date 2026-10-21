"""Live-vs-live training collects both physical seats without training frozen seats."""

import torch
from mjlab.envs import ManagerBasedRlEnv

from mjlab_microduck.football.arena import BALL_RADIUS, FIELD_HALF_LENGTH, MATCH_STEPS, FootballArena, make_football_env_cfg
from mjlab_microduck.football.pool import OpponentChoice, OpponentPool
from mjlab_microduck.football.wrapper import FootballVecEnv


def _zero_skill(obs):
    return obs.new_zeros((len(obs), 14))


def _walking_skill(obs):
    actions = obs.new_zeros((len(obs), 14))
    actions[:, 0] = obs[:, 48]
    return actions


def test_live_joint_seats_execute_both_actions_and_frozen_seat_is_masked():
    base = ManagerBasedRlEnv(cfg=make_football_env_cfg(2), device="cpu")
    wrapped = FootballVecEnv(
        FootballArena(base), "joint", OpponentPool(seed=1),
        (_zero_skill, _zero_skill, _zero_skill), seed=1, dual_rollouts=True,
    )
    try:
        wrapped.learner_home[:] = torch.tensor([True, False])
        wrapped.opponent_choices[:] = [OpponentChoice("live"), OpponentChoice("scripted")]
        assert wrapped.num_envs == 4
        assert wrapped.get_observations()["actor"].shape == (4, 66)
        actions = torch.stack((
            torch.full((14,), 0.1), torch.full((14,), 0.2),
            torch.full((14,), -0.3), torch.full((14,), 0.9),
        ))
        obs, rewards, dones, extras = wrapped.step(actions)
        assert obs["actor"].shape == (4, 66)
        assert rewards.shape == dones.shape == (4,)
        assert extras["learner_mask"].tolist() == [True, True, True, False]
        torch.testing.assert_close(dones[:2], dones[2:])
        torch.testing.assert_close(base.action_manager.action[0, :14], actions[0])
        torch.testing.assert_close(base.action_manager.action[0, 14:], actions[2])
        torch.testing.assert_close(base.action_manager.action[1, 14:], actions[1])
        torch.testing.assert_close(base.action_manager.action[1, :14], torch.zeros(14))
        torch.testing.assert_close(rewards[0], wrapped.arena.rewards("home")[0])
        torch.testing.assert_close(rewards[2], wrapped.arena.rewards("away")[0])
    finally:
        wrapped.close()


def test_live_tactical_seats_use_separate_walk_commands():
    base = ManagerBasedRlEnv(cfg=make_football_env_cfg(1), device="cpu")
    wrapped = FootballVecEnv(
        FootballArena(base), "tactics", OpponentPool(seed=2),
        (_walking_skill, _zero_skill, _zero_skill), seed=2, dual_rollouts=True,
    )
    try:
        wrapped.learner_home[0] = True
        wrapped.opponent_choices[0] = OpponentChoice("live")
        actions = torch.tensor([[0.1, 0.0, 0.0, -1.0],
                                [-0.2, 0.0, 0.0, -1.0]])
        _, _, _, extras = wrapped.step(actions)
        assert extras["learner_mask"].tolist() == [True, True]
        torch.testing.assert_close(base.action_manager.action[0, 0], torch.tensor(0.1))
        torch.testing.assert_close(base.action_manager.action[0, 14], torch.tensor(-0.2))
    finally:
        wrapped.close()


def test_distillation_labels_both_student_seats_before_their_actions():
    base = ManagerBasedRlEnv(cfg=make_football_env_cfg(1), device="cpu")

    def teacher(obs):
        return torch.tensor([[0.1, 0.0, 0.0, -1.0],
                             [-0.2, 0.0, 0.0, -1.0]], device=obs.device)

    wrapped = FootballVecEnv(
        FootballArena(base), "distill", OpponentPool(seed=3),
        (_walking_skill, _zero_skill, _zero_skill), teacher_model=teacher,
        seed=3, dual_rollouts=True,
    )
    try:
        wrapped.learner_home[0] = True
        wrapped.opponent_choices[0] = OpponentChoice("live")
        targets = wrapped.teacher_targets(wrapped.get_observations())
        assert targets.shape == (2, 14)
        torch.testing.assert_close(targets[:, 0], torch.tensor([0.1, -0.2]))
        actions = torch.stack((torch.full((14,), 0.3), torch.full((14,), -0.4)))
        _, _, _, extras = wrapped.step(actions)
        assert extras["learner_mask"].tolist() == [True, True]
        torch.testing.assert_close(base.action_manager.action[0, :14], actions[0])
        torch.testing.assert_close(base.action_manager.action[0, 14:], actions[1])
    finally:
        wrapped.close()


def test_live_mask_refers_to_match_before_resampling():
    base = ManagerBasedRlEnv(cfg=make_football_env_cfg(1), device="cpu")
    wrapped = FootballVecEnv(
        FootballArena(base), "joint", OpponentPool(seed=4),
        (_zero_skill, _zero_skill, _zero_skill), seed=4, dual_rollouts=True,
    )
    try:
        wrapped.opponent_choices[0] = OpponentChoice("scripted")
        ball = base.scene["ball"]
        state = ball.data.default_root_state[:1].clone()
        state[:, :3] = base.scene.env_origins[:1]
        state[:, 0] += wrapped.arena.field_half_length[0] + BALL_RADIUS + 0.08
        state[:, 2] += BALL_RADIUS
        ball.write_root_state_to_sim(state, env_ids=torch.tensor([0]))
        base.scene.write_data_to_sim()
        base.sim.forward()
        _, _, dones, extras = wrapped.step(torch.zeros(2, 14))
        assert dones.tolist() == [1, 1]
        assert "time_outs" not in extras  # Goal termination must not bootstrap PPO value.
        assert "reward/match_outcome" not in extras["log"]
        assert extras["learner_mask"].tolist() == [True, False]
        assert extras["log"]["info/benchmark_matches"] == 1
        assert extras["completed"]["own_goals"].item() + extras["completed"]["opponent_goals"].item() == 1
        assert wrapped.arena.scores[0].tolist() == [0, 0]
        assert wrapped.arena.elapsed[0] == 0
        assert extras["log"]["info/matches_vs_scripted"] == 1
        assert extras["log"]["info/matches_vs_live"] == 0
        assert wrapped.opponent_choices[0].kind == "live"
        _, _, _, extras = wrapped.step(torch.zeros(2, 14))
        assert extras["learner_mask"].tolist() == [True, True]
    finally:
        wrapped.close()


def test_timeout_draw_resets_both_rollout_seats_without_value_bootstrap():
    base = ManagerBasedRlEnv(cfg=make_football_env_cfg(1), device="cpu")
    wrapped = FootballVecEnv(
        FootballArena(base), "joint", OpponentPool(seed=7),
        (_zero_skill, _zero_skill, _zero_skill), seed=7, dual_rollouts=True,
    )
    try:
        wrapped.arena.elapsed[0] = MATCH_STEPS - 1
        _, _, dones, extras = wrapped.step(torch.zeros(2, 14))
        assert dones.tolist() == [1, 1]
        assert "time_outs" not in extras
        assert extras["completed"]["env_ids"].tolist() == [0]
        assert extras["completed"]["timed_out"].tolist() == [True]
        assert extras["completed"]["points"].tolist() == [0.5]
        assert extras["completed"]["own_goals"].tolist() == [0]
        assert extras["completed"]["opponent_goals"].tolist() == [0]
        assert extras["log"]["info/match_timeouts"] == 1
        assert wrapped.arena.elapsed[0] == 0
        assert wrapped.arena.scores[0].tolist() == [0, 0]
    finally:
        wrapped.close()


def test_rolling_win_rates_are_match_level_and_survive_resume():
    base = ManagerBasedRlEnv(cfg=make_football_env_cfg(4), device="cpu")
    wrapped = FootballVecEnv(
        FootballArena(base), "joint", OpponentPool(seed=5),
        (_zero_skill, _zero_skill, _zero_skill), seed=5, dual_rollouts=True,
    )
    try:
        wrapped.opponent_choices[:] = [
            OpponentChoice("scripted"), OpponentChoice("tactical", 1),
            OpponentChoice("joint", 2), OpponentChoice("live"),
        ]
        wrapped._record_match_results(torch.arange(4), torch.tensor([1.0, 0.5, 0.0, 1.0]))
        logs = wrapped._win_rate_logs()
        assert logs["info/benchmark_win_rate"] == 1 / 3
        assert logs["info/benchmark_matches"] == 3
        assert logs["info/win_rate_vs_scripted"] == 1
        assert logs["info/win_rate_vs_tactical"] == 0  # A draw is not a win.
        assert logs["info/win_rate_vs_joint"] == 0
        assert logs["info/win_rate_vs_live"] == 1  # Diagnostic, excluded above.
        assert all(logs[f"info/matches_vs_{kind}"] == 1 for kind in
                   ("scripted", "tactical", "joint", "live"))
        snapshot = wrapped.state_dict()
        wrapped.recent_match_points.clear()
        for results in wrapped.recent_points_by_opponent.values():
            results.clear()
        wrapped.load_state_dict(snapshot)
        assert wrapped._win_rate_logs() == logs
    finally:
        wrapped.close()


def test_trainable_skills_share_live_modules_and_record_executed_actions():
    base = ManagerBasedRlEnv(cfg=make_football_env_cfg(2), device="cpu")
    policies = tuple(torch.nn.Linear(61, 14) for _ in range(3))
    for policy in policies:
        torch.nn.init.zeros_(policy.weight)
        torch.nn.init.zeros_(policy.bias)
        policy.requires_grad_(False)

    def sample(mean, obs, branch):
        return mean + 0.125, mean.new_full((len(mean),), -2.0)

    wrapped = FootballVecEnv(
        FootballArena(base), "tactics", OpponentPool(seed=2), policies,
        dual_rollouts=True, train_skills=True, skill_sampler=sample,
    )
    try:
        assert wrapped.learner_skills.walk is not policies[0]
        assert wrapped.live_opponent_skills.walk is wrapped.learner_skills.walk
        assert wrapped.opponent_skills.walk is policies[0]
        assert wrapped.learner_skills.walk.weight.requires_grad
        assert not policies[0].weight.requires_grad
        assert wrapped.live_opponent_skills.kick_remaining is not wrapped.learner_skills.kick_remaining
        wrapped.learner_home[:] = True
        wrapped.opponent_choices[:] = [OpponentChoice("live"), OpponentChoice("scripted")]
        tactics = torch.zeros((4, 4))
        tactics[:, 3] = -1
        _, _, _, extras = wrapped.step(tactics)
        transition = wrapped.consume_skill_transition()
        assert transition["skill_obs"].shape == (4, 61)
        assert extras["learner_mask"].tolist() == [True, True, True, False]
        torch.testing.assert_close(transition["old_log_prob"], torch.tensor([-2., -2., -2., 0.]))
        torch.testing.assert_close(transition["joint_actions"][:3], torch.full((3, 14), 0.125))
        torch.testing.assert_close(base.action_manager.action[0, 14:], transition["joint_actions"][2])
        assert wrapped.consume_skill_transition() is None
    finally:
        wrapped.close()
