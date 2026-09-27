"""Football field construction and two-player match invariants."""

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import Scene
from mjlab_microduck.football.arena import (
    FIELD_HALF_LENGTH,
    FIELD_HALF_WIDTH,
    FORFEIT_REWARD,
    MATCH_STEPS,
    FootballArena,
    make_football_env_cfg,
)


def test_two_player_scene_compiles_with_one_shared_field():
    cfg = make_football_env_cfg(4)
    assert cfg.decimation * cfg.sim.mujoco.timestep == 0.02
    assert cfg.scene.num_envs == 4
    assert cfg.scene.env_spacing == 0
    assert tuple(cfg.actions) == ("home", "away")
    scene = Scene(cfg.scene, "cpu")
    model = scene.compile()
    assert model.nu == 28
    assert model.nq == 49  # 2 * (free root + 14 joints) + ball free root.
    assert sum(g.name.startswith("football_") for g in scene.spec.geoms) == 6

    def collides(first: str, second: str) -> bool:
        a, b = model.geom(first), model.geom(second)
        return bool((a.contype & b.conaffinity) | (b.contype & a.conaffinity))

    wall = "football_0_north"
    assert collides(wall, "ball/ball_geom")
    assert collides("ball/ball_geom", "home/left_foot_collision")
    assert collides("ball/ball_geom", "terrain")
    assert collides(wall, "home/left_foot_collision")
    assert collides(wall, "away/right_foot_collision")
    assert not any("football_wall_guard" in model.geom(i).name for i in range(model.ngeom))


def test_reset_observations_and_goal_restart():
    env = ManagerBasedRlEnv(cfg=make_football_env_cfg(2), device="cpu")
    arena = FootballArena(env)
    home, away = arena.reset()
    assert home.shape == away.shape == (2, 64)
    assert arena.obs61(0).shape == (2, 61)
    assert arena.critic_obs(0).shape == (2, 82)
    assert torch.equal(home, arena.actor_obs("home"))
    assert torch.allclose(home[:, 5], -torch.ones(2))
    assert torch.equal(arena.attack_sign[:, 0], -arena.attack_sign[:, 1])

    ball = env.scene["ball"]
    state = ball.data.default_root_state[:1].clone()
    state[:, :3] = env.scene.env_origins[:1]
    state[:, 0] += FIELD_HALF_LENGTH + 0.08
    state[:, 2] += 0.035
    ball.write_root_state_to_sim(state, env_ids=torch.tensor([0]))
    env.scene.write_data_to_sim()
    env.sim.forward()
    _, rewards, done, info = arena.step(torch.zeros((2, 28)))
    assert not done.any()
    assert arena.scores[0].sum() == 1
    assert info["goals"][0].sum() == 1
    assert torch.allclose(arena._last_actions[0], torch.zeros((2, 14)))
    assert rewards[0].shape == rewards[1].shape == (2,)

    arena.elapsed[1] = MATCH_STEPS - 1
    arena.scores[1] = torch.tensor([2, 1])
    _, rewards, done, _ = arena.step(torch.zeros((2, 28)))
    assert done[1]
    assert rewards[0][1] >= 5
    assert rewards[1][1] <= -5


def test_side_swapped_scoring_and_joint_action_routing():
    env = ManagerBasedRlEnv(cfg=make_football_env_cfg(2), device="cpu")
    arena = FootballArena(env)
    arena.reset()
    arena.attack_sign[:, 0] = torch.tensor([1.0, -1.0])
    arena.attack_sign[:, 1] = -arena.attack_sign[:, 0]
    ball = env.scene["ball"]
    state = ball.data.default_root_state[:2].clone()
    state[:, :3] = env.scene.env_origins[:2]
    state[:, 0] += FIELD_HALF_LENGTH + 0.08
    state[:, 2] += 0.035
    ball.write_root_state_to_sim(state, env_ids=torch.arange(2))
    env.scene.write_data_to_sim()
    env.sim.forward()
    actions = torch.zeros(2, 28)
    actions[:, :14] = 0.1
    actions[:, 14:] = -0.2
    _, _, _, info = arena.step(actions)
    torch.testing.assert_close(info["goals"].long(), torch.tensor([[1, 0], [0, 1]]))
    torch.testing.assert_close(info["scores"], torch.tensor([[1, 0], [0, 1]]))
    # Goal kickoffs clear previous-action history. The action manager itself
    # must still route the 28 controls to distinct 14-joint robot groups.
    torch.testing.assert_close(env.action_manager.action[:, :14], actions[:, :14])
    torch.testing.assert_close(env.action_manager.action[:, 14:], actions[:, 14:])


def test_duck_out_of_bounds_forfeits_without_kickoff():
    env = ManagerBasedRlEnv(cfg=make_football_env_cfg(2), device="cpu")
    arena = FootballArena(env)
    arena.reset()
    arena.scores[:] = torch.tensor([[3, 0], [0, 3]])
    for env_id, side in ((0, "home"), (1, "away")):
        robot = env.scene[side]
        ids = torch.tensor([env_id])
        state = robot.data.default_root_state[ids].clone()
        state[:, :3] = env.scene.env_origins[ids]
        if side == "home":
            state[:, 0] += FIELD_HALF_LENGTH + 0.2
        else:
            state[:, 1] += 0.8
        state[:, 2] += 0.115
        robot.write_root_state_to_sim(state, env_ids=ids)
    env.scene.write_data_to_sim()
    env.sim.forward()

    _, rewards, done, info = arena.step(torch.zeros((2, 28)))
    assert done.tolist() == [True, True]
    torch.testing.assert_close(info["out_of_bounds"],
                               torch.tensor([[True, False], [False, True]]))
    torch.testing.assert_close(info["goals"], torch.zeros((2, 2), dtype=torch.bool))
    torch.testing.assert_close(info["scores"], torch.tensor([[3, 0], [0, 3]]))
    torch.testing.assert_close(rewards[0], torch.tensor([-FORFEIT_REWARD, FORFEIT_REWARD]))
    torch.testing.assert_close(rewards[1], torch.tensor([FORFEIT_REWARD, -FORFEIT_REWARD]))
    assert env.scene["home"].data.root_link_pos_w[0, 0] > FIELD_HALF_LENGTH
    assert env.scene["away"].data.root_link_pos_w[1, 1] > 0.6



def test_debug_numerics_aborts_before_nonfinite_action_reaches_physics():
    env = ManagerBasedRlEnv(cfg=make_football_env_cfg(1), device="cpu")
    arena = FootballArena(env, debug_numerics=True)
    arena.reset()
    action = torch.zeros(1, 28)
    action[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="home_action"):
        arena.step(action)
    assert torch.isfinite(env.sim.data.qpos).all()
    env.close()



def test_ball_still_bounces_off_wall():
    env = ManagerBasedRlEnv(cfg=make_football_env_cfg(1), device="cpu")
    arena = FootballArena(env)
    arena.reset()
    ball = env.scene["ball"]
    state = ball.data.default_root_state[:1].clone()
    state[0, :3] = env.scene.env_origins[0] + torch.tensor([0.0, 0.50, 0.035])
    state[0, 7:10] = torch.tensor([0.0, 1.0, 0.0])
    ball.write_root_state_to_sim(state, env_ids=torch.tensor([0]))
    env.scene.write_data_to_sim()
    env.sim.forward()
    positions = []
    for _ in range(12):
        arena.step(torch.zeros(1, 28))
        positions.append(float(ball.data.root_link_pos_w[0, 1]))
    assert max(positions) < FIELD_HALF_WIDTH
    assert positions[-1] < max(positions)
    env.close()



def test_nonfinite_action_voids_only_affected_world_without_reward():
    env = ManagerBasedRlEnv(cfg=make_football_env_cfg(2), device="cpu")
    arena = FootballArena(env)
    arena.reset()
    actions = torch.zeros(2, 28)
    actions[0, 0] = float("nan")
    _, rewards, done, info = arena.step(actions)
    assert done.tolist() == [True, False]
    assert info["numerical_reset"].tolist() == [True, False]
    assert rewards[0][0] == rewards[1][0] == 0
    assert arena.elapsed.tolist() == [0, 1]
    assert torch.isfinite(arena.actor_obs(0)).all()
    assert torch.isfinite(arena.actor_obs(1)).all()
    assert torch.isfinite(env.sim.data.qpos).all()
    # The reset world is ready for its next match without an external reset.
    arena.step(torch.zeros(2, 28))
    env.close()


def test_nonfinite_physics_voids_only_affected_world_without_reward():
    env = ManagerBasedRlEnv(cfg=make_football_env_cfg(2), device="cpu")
    arena = FootballArena(env)
    arena.reset()
    normal_step = env.step

    def failed_step(actions):
        result = normal_step(actions)
        env.sim.data.qpos[0, 0] = float("nan")
        return result

    env.step = failed_step
    _, rewards, done, info = arena.step(torch.zeros(2, 28))
    assert done.tolist() == [True, False]
    assert info["numerical_reset"].tolist() == [True, False]
    assert rewards[0][0] == rewards[1][0] == 0
    assert arena.elapsed.tolist() == [0, 1]
    assert torch.isfinite(arena.actor_obs(0)).all()
    assert torch.isfinite(arena.critic_obs(0)).all()
    assert torch.isfinite(env.sim.data.qpos).all()
    env.close()
