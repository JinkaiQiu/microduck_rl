"""Football field construction and two-player match invariants."""

import mujoco
import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import Scene
from mjlab_microduck.football.arena import (
    BALL_SPAWN_SPEED,
    BALL_RADIUS,
    SPAWN_CLEARANCE_M,
    _sample_ball_spawn_xy,
    BALL_CONTACT_DAMPING_RANGE,
    BALL_FLOOR_SLIDING_FRICTION_RANGE,
    BALL_APPROACH_REWARD_WEIGHT,
    KICK_SUCCESS_REWARD,
    KICK_REWARD_COOLDOWN_STEPS,
    BOUNDARY_PENALTY_WEIGHT,
    BOUNDARY_PENALTY_DECAY_M,
    BODY_PROXIMITY_PENALTY_WEIGHT,
    BODY_PROXIMITY_DECAY_M,
    PROXIMITY_PENALTY_CUTOFF_M,
    ROBOT_FOOTPRINT_RADIUS_M,
    FIELD_CORNER_RADIUS,
    FIELD_HALF_LENGTH,
    FIELD_HALF_WIDTH,
    FIELD_HALF_LENGTH_RANGE,
    FIELD_HALF_WIDTH_RANGE,
    FLOOR_CONCAVITY_SCALE_RANGE,
    FLOOR_EXIT_MARGIN,
    FLOOR_FLAT_FRACTION,
    FLOOR_MAX_RISE,
    GOAL_HALF_WIDTH,
    GOAL_REWARD,
    FORFEIT_REWARD,
    MATCH_STEPS,
    FootballArena,
    ball_approach_reward,
    boundary_proximity_penalty,
    body_body_proximity_penalty,
    field_boundary_clearance,
    field_out_of_bounds,
    field_goal_exit,
    football_floor_height,
    make_football_env_cfg,
    successful_kick_reward,
)


def test_rounded_rectangle_has_four_straight_edges_and_curved_corners():
    end_straight_y = FIELD_HALF_WIDTH - FIELD_CORNER_RADIUS
    diagonal = FIELD_CORNER_RADIUS / (2 ** 0.5)
    side_straight_x = FIELD_HALF_LENGTH - FIELD_CORNER_RADIUS
    points = torch.tensor([
        [FIELD_HALF_LENGTH, 0.0], [-FIELD_HALF_LENGTH, 0.0],
        [0.0, FIELD_HALF_WIDTH], [0.0, -FIELD_HALF_WIDTH],
        [FIELD_HALF_LENGTH, end_straight_y / 2],
        [side_straight_x / 2, FIELD_HALF_WIDTH],
        [side_straight_x + diagonal, end_straight_y + diagonal],
        [FIELD_HALF_LENGTH - 0.05, FIELD_HALF_WIDTH - 0.05],
        [-FIELD_HALF_LENGTH + 0.05, -FIELD_HALF_WIDTH + 0.05],
    ])
    clearance = field_boundary_clearance(points)
    torch.testing.assert_close(clearance[:7], torch.zeros(7))
    assert (clearance[7:] < 0).all()
    assert field_out_of_bounds(points).tolist() == [False] * 7 + [True] * 2


def test_floor_is_flat_in_center_and_rises_only_near_randomized_edges():
    lengths = torch.tensor([0.8, 2.0])
    widths = torch.tensor([0.8, 1.0])
    scales = torch.tensor([0.625, 1.0])
    centers = torch.zeros((2, 2))
    x_inside = torch.stack((FLOOR_FLAT_FRACTION * (lengths + FLOOR_EXIT_MARGIN) * 0.99,
                            torch.zeros_like(lengths)), dim=-1)
    edge = torch.stack((lengths, torch.zeros_like(lengths)), dim=-1)
    assert torch.equal(football_floor_height(centers, scales, lengths, widths),
                       torch.zeros(2))
    assert torch.equal(football_floor_height(x_inside, scales, lengths, widths),
                       torch.zeros(2))
    height = football_floor_height(edge, scales, lengths, widths)
    assert (height > 0).all()
    assert (height <= FLOOR_MAX_RISE * scales).all()


def test_goal_exit_requires_outward_crossing_inside_door():
    previous = torch.tensor([
        [FIELD_HALF_LENGTH - 0.02, 0.0],
        [-FIELD_HALF_LENGTH + 0.02, 0.0],
        [FIELD_HALF_LENGTH - 0.02, 0.4],  # Beyond the end wall.
        [0.0, FIELD_HALF_WIDTH - 0.02],  # Through a sideline.
        [FIELD_HALF_LENGTH + 0.02, 0.0],  # Moving back into the pitch.
        [FIELD_HALF_LENGTH - 0.02, 0.16],  # Crosses inside, then moves sideways.
        [FIELD_HALF_LENGTH - 0.02, GOAL_HALF_WIDTH],  # Goal post, not opening.
    ])
    current = torch.tensor([
        [FIELD_HALF_LENGTH + 0.02, 0.0],
        [-FIELD_HALF_LENGTH - 0.02, 0.0],
        [FIELD_HALF_LENGTH + 0.02, 0.4],
        [0.0, FIELD_HALF_WIDTH + 0.02],
        [FIELD_HALF_LENGTH - 0.02, 0.0],
        [FIELD_HALF_LENGTH + 0.02, 0.20],
        [FIELD_HALF_LENGTH + 0.02, GOAL_HALF_WIDTH],
    ])
    assert field_goal_exit(previous, current).tolist() == [True, True, False, False, False, False, False]
    current[5, 1] = 0.18
    assert field_goal_exit(previous, current)[5]


def test_boundary_penalty_rises_exponentially_near_rounded_edge():
    positions = torch.tensor([
        [[0.0, 0.0], [0.0, FIELD_HALF_WIDTH - 0.08]],
        [[FIELD_HALF_LENGTH - 0.05, 0.0], [-FIELD_HALF_LENGTH + 0.05, 0.0]],
    ])
    clearance = field_boundary_clearance(positions)
    torch.testing.assert_close(
        clearance, torch.tensor([[FIELD_HALF_WIDTH, 0.08], [0.05, 0.05]]),
    )
    penalty = boundary_proximity_penalty(clearance)
    assert penalty.shape == (2, 2)
    assert torch.isfinite(penalty).all()
    assert penalty[0, 1] < penalty[0, 0]  # Near a straight sideline.
    torch.testing.assert_close(penalty[1, 0], penalty[1, 1])  # End symmetry.
    edge = boundary_proximity_penalty(torch.tensor([0.0, BOUNDARY_PENALTY_DECAY_M]))
    torch.testing.assert_close(
        edge, torch.tensor([-BOUNDARY_PENALTY_WEIGHT,
                            -BOUNDARY_PENALTY_WEIGHT / torch.exp(torch.tensor(1.0))]),
    )
    clearance = torch.tensor([PROXIMITY_PENALTY_CUTOFF_M - 0.001,
                              PROXIMITY_PENALTY_CUTOFF_M,
                              PROXIMITY_PENALTY_CUTOFF_M + 0.001,
                              1.0])
    cutoff_penalty = boundary_proximity_penalty(clearance)
    assert (cutoff_penalty[:2] < 0).all()
    assert torch.equal(cutoff_penalty[2:], torch.zeros(2))


def test_body_footprint_proximity_penalty_is_symmetric_and_exponential():
    contact_distance = 2 * ROBOT_FOOTPRINT_RADIUS_M
    distances = torch.tensor([contact_distance - 0.03,
                              contact_distance,
                              contact_distance + BODY_PROXIMITY_DECAY_M,
                              contact_distance + PROXIMITY_PENALTY_CUTOFF_M - 0.001,
                              contact_distance + PROXIMITY_PENALTY_CUTOFF_M + 0.001,
                              contact_distance + 5 * BODY_PROXIMITY_DECAY_M])
    players = torch.zeros(len(distances), 2, 2)
    players[:, 1, 0] = distances
    penalty = body_body_proximity_penalty(players)
    expected = -BODY_PROXIMITY_PENALTY_WEIGHT * torch.exp(
        -torch.tensor([0.0, 0.0, 1.0, (PROXIMITY_PENALTY_CUTOFF_M - 0.001) / BODY_PROXIMITY_DECAY_M]),
    )
    torch.testing.assert_close(penalty[:4, 0], expected)
    torch.testing.assert_close(penalty[:4, 1], expected)
    assert torch.equal(penalty[4:], torch.zeros_like(penalty[4:]))
    torch.testing.assert_close(body_body_proximity_penalty(players.flip(1)), penalty)


def test_approach_reward_requires_facing_within_45_degrees():
    previous_players = torch.zeros((2, 2, 2))
    current_players = previous_players.clone()
    current_players[..., 0] = 0.1
    previous_ball = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    yaw = torch.tensor([[0.0, torch.pi / 6], [torch.pi / 4 + 0.01, -torch.pi / 4 - 0.01]])
    reward = ball_approach_reward(previous_players, current_players, previous_ball, yaw)
    assert reward[0, 0] > reward[0, 1] > 0
    torch.testing.assert_close(reward[1], torch.zeros(2))
    stationary = ball_approach_reward(previous_players, previous_players, previous_ball, yaw)
    torch.testing.assert_close(stationary, torch.zeros_like(stationary))
    retreat = previous_players.clone()
    retreat[..., 0] = -0.1
    retreat_reward = ball_approach_reward(previous_players, retreat, previous_ball, yaw)
    assert (retreat_reward < 0).all()  # Facing cutoff does not erase retreat cost.


def test_approach_reward_prefers_right_foot_kick_position_over_root_distance():
    previous_players = torch.tensor([[[0.0, 0.0], [0.03, 0.0]]])
    current_players = torch.tensor([[[0.02, 0.0], [0.0, 0.0]]])
    ball = torch.tensor([[0.09, -0.042]])  # Exact trained right-foot placement for player 0.
    yaw = torch.zeros((1, 2))
    reward = ball_approach_reward(previous_players, current_players, ball, yaw)
    assert reward[0, 0] < 0  # Root approaches, but moves its foot past the kickable pose.
    assert reward[0, 1] > 0  # Root backs up into the kickable pose.
    torch.testing.assert_close(reward[0, 0], torch.tensor(-0.02 * BALL_APPROACH_REWARD_WEIGHT))


def test_successful_kick_requires_contact_goalward_impulse_and_cooldown():
    attack = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    contact = torch.tensor([[True, True], [True, False]])
    before = torch.zeros((2, 2))
    after = torch.tensor([[0.6, 0.0], [-0.5, 0.0]])
    cooldown = torch.zeros((2, 2), dtype=torch.long)
    reward = successful_kick_reward(contact, before, after, attack, cooldown)
    torch.testing.assert_close(reward, torch.tensor([[KICK_SUCCESS_REWARD, 0.0],
                                                    [KICK_SUCCESS_REWARD, 0.0]]))
    cooldown[0, 0] = KICK_REWARD_COOLDOWN_STEPS
    assert successful_kick_reward(contact, before, after, attack, cooldown)[0, 0] == 0
    assert successful_kick_reward(contact, after, after, attack, torch.zeros_like(cooldown)).sum() == 0
    assert successful_kick_reward(torch.zeros_like(contact), before, after, attack,
                                  torch.zeros_like(cooldown)).sum() == 0


def test_two_player_scene_compiles_with_one_shared_field():
    cfg = make_football_env_cfg(4)
    assert cfg.decimation * cfg.sim.mujoco.timestep == 0.02
    assert cfg.scene.num_envs == 4
    assert cfg.scene.env_spacing == 0
    assert cfg.viewer.elevation == -90.0
    assert cfg.viewer.distance == 4.8
    assert cfg.viewer.fovy == 45.0
    assert tuple(cfg.actions) == ("home", "away")
    scene = Scene(cfg.scene, "cpu")
    model = scene.compile()
    assert model.nu == 28
    assert model.nq == 49  # 2 * (free root + 14 joints) + ball free root.
    # Four straight sides (the goal sides split at their openings) and four corners.
    assert sum(g.name.startswith("football_") for g in scene.spec.geoms) == 38
    for corner in ("east_north", "east_south", "west_north", "west_south"):
        assert sum(g.name.startswith(f"football_0_{corner}_arc_")
                   for g in scene.spec.geoms) == 8
    for side in ("north", "south"):
        wall = model.geom(f"football_0_{side}")
        assert wall.size[0] == pytest.approx(FIELD_HALF_LENGTH - FIELD_CORNER_RADIUS)
        assert abs(wall.pos[1]) == pytest.approx(FIELD_HALF_WIDTH + 0.025)
    goal_side_span = (FIELD_HALF_WIDTH - FIELD_CORNER_RADIUS - 0.175) / 2
    for side in ("east", "west"):
        for half in ("north", "south"):
            wall = model.geom(f"football_0_{side}_{half}")
            assert wall.size[1] == pytest.approx(goal_side_span)
            assert abs(wall.pos[0]) == pytest.approx(FIELD_HALF_LENGTH + 0.025)

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


def test_ball_contact_randomization_is_per_world_bounded_and_nonaccumulating():
    env = ManagerBasedRlEnv(cfg=make_football_env_cfg(3), device="cpu")
    arena = FootballArena(env)
    arena.reset()
    model = env.sim.model
    compiled = env.sim.mj_model
    ball = mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_GEOM, "ball/ball_geom")
    pair = mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_PAIR, "ball_floor")
    floor = mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    assert ball >= 0 and pair >= 0 and floor >= 0
    assert compiled.pair_geom1[pair] == floor
    assert compiled.pair_geom2[pair] == ball
    assert model.geom_solref.shape[0] == model.pair_solref.shape[0] == 3
    assert model.pair_friction.shape[0] == 3

    def check_ranges():
        damping = model.geom_solref[:, ball, 1]
        friction = model.pair_friction[:, pair, :2]
        assert ((BALL_CONTACT_DAMPING_RANGE[0] <= damping)
                & (damping <= BALL_CONTACT_DAMPING_RANGE[1])).all()
        assert ((BALL_FLOOR_SLIDING_FRICTION_RANGE[0] <= friction)
                & (friction <= BALL_FLOOR_SLIDING_FRICTION_RANGE[1])).all()
        torch.testing.assert_close(model.pair_solref[:, pair, 1], damping)
        torch.testing.assert_close(friction[:, 0], friction[:, 1])
        torch.testing.assert_close(model.geom_solref[:, ball, 0], torch.full((3,), 0.02))

    check_ranges()
    hfield = mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_HFIELD, "football_bowl")
    assert hfield >= 0
    def check_floor_size():
        size = model.hfield_size[:, 4 * hfield:4 * hfield + 4]
        torch.testing.assert_close(size[:, 0], arena.field_half_length + FLOOR_EXIT_MARGIN)
        torch.testing.assert_close(size[:, 1], arena.field_half_width + FLOOR_EXIT_MARGIN)
        torch.testing.assert_close(size[:, 2], FLOOR_MAX_RISE * arena.floor_scale)
        assert ((FLOOR_CONCAVITY_SCALE_RANGE[0] <= arena.floor_scale)
                & (arena.floor_scale <= FLOOR_CONCAVITY_SCALE_RANGE[1])).all()

    check_floor_size()
    old_floor_size = model.hfield_size.clone()
    old_damping = model.geom_solref[:, ball].clone()
    old_friction = model.pair_friction[:, pair].clone()
    arena.reset(torch.tensor([1]))
    check_ranges()
    check_floor_size()
    torch.testing.assert_close(model.hfield_size[[0, 2]], old_floor_size[[0, 2]])
    torch.testing.assert_close(model.geom_solref[[0, 2], ball], old_damping[[0, 2]])
    torch.testing.assert_close(model.pair_friction[[0, 2], pair], old_friction[[0, 2]])
    for _ in range(10):
        arena.reset(torch.tensor([1]))
        check_ranges()
    env.close()


def test_goal_ends_match_without_extra_outcome_reward_or_internal_kickoff():
    env = ManagerBasedRlEnv(cfg=make_football_env_cfg(2), device="cpu")
    arena = FootballArena(env)
    home, away = arena.reset()
    assert home.shape == away.shape == (2, 66)
    assert arena.obs61(0).shape == (2, 61)
    assert arena.critic_obs(0).shape == (2, 84)
    assert torch.equal(home, arena.actor_obs("home"))
    assert torch.allclose(home[:, 5], -torch.ones(2))
    assert torch.equal(arena.attack_sign[:, 0], -arena.attack_sign[:, 1])

    ball = env.scene["ball"]
    state = ball.data.default_root_state[:1].clone()
    state[:, :3] = env.scene.env_origins[:1]
    state[:, 0] += arena.field_half_length[:len(state)] + 0.08
    state[:, 2] += 0.035
    ball.write_root_state_to_sim(state, env_ids=torch.tensor([0]))
    env.scene.write_data_to_sim()
    env.sim.forward()
    _, rewards, done, info = arena.step(torch.zeros((2, 28)))
    assert done.tolist() == [True, False]
    assert arena.scores[0].sum() == 1
    assert arena.elapsed[0] == 1  # The arena leaves terminal state for the wrapper to reset.
    assert info["goals"][0].sum() == 1
    torch.testing.assert_close(info["reward_components"]["goal"][0].abs(),
                               torch.full((2,), GOAL_REWARD))
    assert torch.allclose(arena._last_actions[0], torch.zeros((2, 14)))
    assert rewards[0].shape == rewards[1].shape == (2,)
    components = info["reward_components"]
    assert torch.equal(components["boundary_proximity"], torch.zeros_like(components["boundary_proximity"]))
    assert torch.equal(components["body_body_proximity"], torch.zeros_like(components["body_body_proximity"]))
    assert set(components) == {
        "ball_progress", "ball_approach", "successful_kick", "fall_penalty",
        "boundary_proximity", "body_body_proximity", "goal",
        "boundary_forfeit",
    }
    torch.testing.assert_close(
        torch.stack(tuple(components.values())).sum(dim=0),
        torch.stack(rewards, dim=1),
    )

    arena.reset(torch.tensor([0]))
    assert arena.scores[0].tolist() == [0, 0]
    assert arena.elapsed[0] == 0
    arena.elapsed[1] = MATCH_STEPS - 1
    _, rewards, done, info = arena.step(torch.zeros((2, 28)))
    assert done.tolist() == [False, True]
    assert info["timed_out"].tolist() == [False, True]
    assert arena.elapsed[1] == MATCH_STEPS
    assert not info["goals"][1].any()
    assert arena.scores[1].tolist() == [0, 0]
    assert "match_outcome" not in info["reward_components"]


def test_kickoff_spawns_players_on_opposite_halves_and_moving_ball_in_center():
    env = ManagerBasedRlEnv(cfg=make_football_env_cfg(64), device="cpu")
    arena = FootballArena(env)
    arena.reset()
    origins = env.scene.env_origins
    home_x = env.scene["home"].data.root_link_pos_w[:, 0] - origins[:, 0]
    away_x = env.scene["away"].data.root_link_pos_w[:, 0] - origins[:, 0]
    ball_pos = env.scene["ball"].data.root_link_pos_w[:, :2] - origins[:, :2]
    ball_vel = env.scene["ball"].data.root_link_lin_vel_w[:, :2]

    assert torch.all(home_x * arena.attack_sign[:, 0] < 0)
    assert torch.all(away_x * arena.attack_sign[:, 1] < 0)
    assert torch.all(home_x.abs() <= arena.field_half_length)
    assert torch.all(away_x.abs() <= arena.field_half_length)
    assert torch.all(ball_pos[:, 0].abs() <= torch.minimum(
        torch.full_like(arena.field_half_length, 0.50), 0.25 * arena.field_half_length,
    ))
    assert torch.all(ball_pos[:, 1].abs() <= torch.minimum(
        torch.full_like(arena.field_half_width, 0.45), 0.50 * arena.field_half_width,
    ))
    assert torch.all(
        field_boundary_clearance(ball_pos, arena.field_half_length, arena.field_half_width)
        >= BALL_RADIUS + SPAWN_CLEARANCE_M
    )
    players = torch.stack((
        env.scene["home"].data.root_link_pos_w[:, :2] - origins[:, :2],
        env.scene["away"].data.root_link_pos_w[:, :2] - origins[:, :2],
    ), dim=1)
    assert torch.all(torch.linalg.vector_norm(players - ball_pos[:, None, :], dim=-1)
                     >= ROBOT_FOOTPRINT_RADIUS_M + BALL_RADIUS + SPAWN_CLEARANCE_M)
    speeds = torch.linalg.vector_norm(ball_vel, dim=1)
    assert torch.all(speeds >= BALL_SPAWN_SPEED[0])
    assert torch.all(speeds <= BALL_SPAWN_SPEED[1])
    env.close()


def test_side_swapped_scoring_and_joint_action_routing():
    env = ManagerBasedRlEnv(cfg=make_football_env_cfg(2), device="cpu")
    arena = FootballArena(env)
    arena.reset()
    arena.attack_sign[:, 0] = torch.tensor([1.0, -1.0])
    arena.attack_sign[:, 1] = -arena.attack_sign[:, 0]
    ball = env.scene["ball"]
    state = ball.data.default_root_state[:2].clone()
    state[:, :3] = env.scene.env_origins[:2]
    state[:, 0] += arena.field_half_length[:len(state)] + 0.08
    state[:, 2] += 0.035
    ball.write_root_state_to_sim(state, env_ids=torch.arange(2))
    env.scene.write_data_to_sim()
    env.sim.forward()
    actions = torch.zeros(2, 28)
    actions[:, :14] = 0.1
    actions[:, 14:] = -0.2
    _, _, dones, info = arena.step(actions)
    assert dones.tolist() == [True, True]
    torch.testing.assert_close(info["goals"].long(), torch.tensor([[1, 0], [0, 1]]))
    torch.testing.assert_close(info["scores"], torch.tensor([[1, 0], [0, 1]]))
    # The action manager must route controls to distinct robot groups on the
    # terminal goal step; the next full reset clears action history.
    torch.testing.assert_close(env.action_manager.action[:, :14], actions[:, :14])
    torch.testing.assert_close(env.action_manager.action[:, 14:], actions[:, 14:])
    arena.reset()
    assert torch.equal(arena.scores, torch.zeros_like(arena.scores))
    assert torch.equal(arena._last_actions, torch.zeros_like(arena._last_actions))
    env.close()


def test_duck_exiting_through_either_goal_door_forfeits_without_kickoff():
    env = ManagerBasedRlEnv(cfg=make_football_env_cfg(2), device="cpu")
    arena = FootballArena(env)
    arena.reset()
    arena.scores[:] = torch.tensor([[3, 0], [0, 3]])
    for env_id, side in ((0, "home"), (1, "away")):
        robot = env.scene[side]
        ids = torch.tensor([env_id])
        state = robot.data.default_root_state[ids].clone()
        state[:, :3] = env.scene.env_origins[ids]
        direction = -1 if side == "home" else 1
        state[:, 0] += direction * (arena.field_half_length[ids] - 0.025)
        state[:, 2] += 0.115
        state[:, 7] = direction * 6.0
        robot.write_root_state_to_sim(state, env_ids=ids)
    env.scene.write_data_to_sim()
    env.sim.forward()

    _, rewards, done, info = arena.step(torch.zeros((2, 28)))
    assert done.tolist() == [True, True]
    torch.testing.assert_close(info["out_of_bounds"],
                               torch.tensor([[True, False], [False, True]]))
    torch.testing.assert_close(info["goals"], torch.zeros((2, 2), dtype=torch.bool))
    torch.testing.assert_close(info["scores"], torch.tensor([[3, 0], [0, 3]]))
    torch.testing.assert_close(rewards[0], torch.tensor([-FORFEIT_REWARD, 0.0]))
    torch.testing.assert_close(rewards[1], torch.tensor([0.0, -FORFEIT_REWARD]))
    torch.testing.assert_close(
        info["reward_components"]["boundary_forfeit"],
        torch.tensor([[-FORFEIT_REWARD, 0.0], [0.0, -FORFEIT_REWARD]]),
    )
    for name, component in info["reward_components"].items():
        if name != "boundary_forfeit":
            torch.testing.assert_close(component, torch.zeros_like(component))
    env.close()


def test_outside_a_solid_wall_does_not_forfeit():
    env = ManagerBasedRlEnv(cfg=make_football_env_cfg(2), device="cpu")
    arena = FootballArena(env)
    arena.reset()
    for env_id, side in ((0, "home"), (1, "away")):
        robot = env.scene[side]
        ids = torch.tensor([env_id])
        state = robot.data.default_root_state[ids].clone()
        state[:, :3] = env.scene.env_origins[ids]
        if side == "home":
            state[:, 0] -= arena.field_half_length[ids] - 0.05
            state[:, 1] += arena.field_half_width[ids] - 0.08  # Outside the rounded corner.
        else:
            state[:, 1] += arena.field_half_width[ids] + 0.2  # Beyond the north wall.
        state[:, 2] += 0.115
        robot.write_root_state_to_sim(state, env_ids=ids)
    env.scene.write_data_to_sim()
    env.sim.forward()

    _, rewards, done, info = arena.step(torch.zeros((2, 28)))
    assert not done.any()
    assert not info["out_of_bounds"].any()
    assert torch.all(info["reward_components"]["boundary_forfeit"] == 0)
    assert all(torch.isfinite(reward).all() for reward in rewards)
    env.close()


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



def test_randomized_ball_floor_bounce_does_not_gain_energy():
    env = ManagerBasedRlEnv(cfg=make_football_env_cfg(2), device="cpu")
    arena = FootballArena(env)
    try:
        arena.reset()
        model = env.sim.model
        ball_geom = mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "ball/ball_geom")
        floor_pair = mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_PAIR, "ball_floor")
        damping = torch.tensor(BALL_CONTACT_DAMPING_RANGE)
        model.geom_solref[:, ball_geom, 1] = damping
        model.pair_solref[:, floor_pair, 1] = damping
        ball = env.scene["ball"]
        state = ball.data.default_root_state[:2].clone()
        state[:, :3] = env.scene.env_origins[:2]
        state[:, 2] += 1.0
        state[:, 7:] = 0
        ball.write_root_state_to_sim(state, env_ids=torch.arange(2))
        env.scene.write_data_to_sim()
        env.sim.forward()
        vertical_speed = []
        height = []
        for _ in range(80):
            arena.step(torch.zeros(2, 28))
            vertical_speed.append(ball.data.root_link_lin_vel_w[:, 2].clone())
            height.append(ball.data.root_link_pos_w[:, 2].clone())
        vertical_speed = torch.stack(vertical_speed)
        height = torch.stack(height)
        for world in range(2):
            rebounds = ((vertical_speed[:-1, world] < -1.0)
                        & (vertical_speed[1:, world] > 0.0)).nonzero(as_tuple=False)
            assert rebounds.numel(), "drop never reached the floor"
            step = int(rebounds[0, 0]) + 1
            assert vertical_speed[step, world] < -vertical_speed[step - 1, world]
            assert height[step:, world].max() < 1.0
    finally:
        env.close()


def test_ball_still_bounces_off_wall():
    env = ManagerBasedRlEnv(cfg=make_football_env_cfg(1), device="cpu")
    arena = FootballArena(env)
    arena.reset()
    ball = env.scene["ball"]
    state = ball.data.default_root_state[:1].clone()
    state[0, :3] = env.scene.env_origins[0] + torch.tensor([0.0, arena.field_half_width[0] - 0.10, 0.035])
    state[0, 7:10] = torch.tensor([0.0, 1.0, 0.0])
    ball.write_root_state_to_sim(state, env_ids=torch.tensor([0]))
    env.scene.write_data_to_sim()
    env.sim.forward()
    positions = []
    for _ in range(12):
        arena.step(torch.zeros(1, 28))
        positions.append(float(ball.data.root_link_pos_w[0, 1]))
    assert max(positions) < arena.field_half_width[0]
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


def test_field_size_randomization_updates_per_world_walls_rules_and_observations():
    env = ManagerBasedRlEnv(cfg=make_football_env_cfg(3), device="cpu")
    arena = FootballArena(env)
    try:
        arena.reset()
        lengths = arena.field_half_length.clone()
        widths = arena.field_half_width.clone()
        assert torch.all((lengths >= FIELD_HALF_LENGTH_RANGE[0]) & (lengths <= FIELD_HALF_LENGTH_RANGE[1]))
        assert torch.all((widths >= FIELD_HALF_WIDTH_RANGE[0]) & (widths <= FIELD_HALF_WIDTH_RANGE[1]))
        assert not torch.allclose(lengths, lengths[0].expand_as(lengths))
        assert not torch.allclose(widths, widths[0].expand_as(widths))
        for side in (0, 1):
            actor = arena.actor_obs(side)
            critic = arena.critic_obs(side)
            torch.testing.assert_close(actor[:, -2], lengths / FIELD_HALF_LENGTH)
            torch.testing.assert_close(actor[:, -1], widths / FIELD_HALF_WIDTH)
            torch.testing.assert_close(critic[:, 64], lengths / FIELD_HALF_LENGTH)
            torch.testing.assert_close(critic[:, 65], widths / FIELD_HALF_WIDTH)
        model = env.sim.model
        native = env.sim.mj_model
        north = mujoco.mj_name2id(native, mujoco.mjtObj.mjOBJ_GEOM, "football_0_north")
        east = mujoco.mj_name2id(native, mujoco.mjtObj.mjOBJ_GEOM, "football_0_east_north")
        arc = mujoco.mj_name2id(native, mujoco.mjtObj.mjOBJ_GEOM, "football_0_east_north_arc_0")
        torch.testing.assert_close(model.geom_pos[:, north, 1], widths + 0.025)
        # Static wall poses in Warp data must follow the randomized model.
        # Checking only model.geom_pos missed the original physical-wall bug.
        wall_ids = torch.as_tensor(env._football_wall_ids)
        torch.testing.assert_close(
            env.sim.data.geom_xpos[:, wall_ids], model.geom_pos[:, wall_ids],
        )
        arena.step(torch.zeros(3, 28))
        torch.testing.assert_close(
            env.sim.data.geom_xpos[:, wall_ids], model.geom_pos[:, wall_ids],
        )
        torch.testing.assert_close(model.geom_size[:, north, 0], lengths - FIELD_CORNER_RADIUS)
        torch.testing.assert_close(model.geom_pos[:, east, 0], lengths + 0.025)
        torch.testing.assert_close(
            model.geom_size[:, east, 1], (widths - FIELD_CORNER_RADIUS - GOAL_HALF_WIDTH) / 2,
        )
        torch.testing.assert_close(
            model.geom_pos[:, arc, 0] - model.geom_pos[0, arc, 0], lengths - lengths[0],
        )
        torch.testing.assert_close(model.geom_aabb[:, north, 1], model.geom_size[:, north])
        torch.testing.assert_close(
            model.geom_rbound[:, north], torch.linalg.vector_norm(model.geom_size[:, north], dim=-1),
        )
        assert native.geom_pos[north, 1] == pytest.approx(widths[0].item() + 0.025)
        # The rule boundary uses each world's sampled geometry.
        point = torch.stack((lengths + 0.01, torch.zeros_like(lengths)), dim=-1)
        assert field_out_of_bounds(point, lengths, widths).all()
        before = torch.stack((lengths - 0.01, torch.zeros_like(lengths)), dim=-1)
        assert field_goal_exit(before, point, lengths).all()
        previous = lengths.clone()
        arena.reset(torch.tensor([1]))
        torch.testing.assert_close(arena.field_half_length[[0, 2]], previous[[0, 2]])
        for _ in range(10):
            arena.reset(torch.tensor([1]))
            size = model.geom_size[1, north, 0]
            torch.testing.assert_close(size, arena.field_half_length[1] - FIELD_CORNER_RADIUS)
            assert FIELD_HALF_LENGTH_RANGE[0] - FIELD_CORNER_RADIUS <= size <= FIELD_HALF_LENGTH_RANGE[1] - FIELD_CORNER_RADIUS  # No reset drift.
    finally:
        env.close()


def test_ball_rebounds_from_resized_solid_end_wall(monkeypatch):
    import mjlab_microduck.football.arena as football

    monkeypatch.setattr(football, "FIELD_HALF_LENGTH_RANGE", (1.2, 1.2))
    monkeypatch.setattr(football, "FIELD_HALF_WIDTH_RANGE", (0.9, 0.9))
    env = ManagerBasedRlEnv(cfg=make_football_env_cfg(1), device="cpu")
    arena = FootballArena(env)
    try:
        arena.reset()
        ball = env.scene["ball"]
        state = ball.data.default_root_state[:1].clone()
        state[:, :3] = env.scene.env_origins[:1]
        state[0, 0] = 1.0  # 20 cm inside the randomized end wall.
        state[0, 1] = 0.3  # Solid wall segment, outside the goal mouth.
        state[0, 2] = BALL_RADIUS
        state[:, 7:] = 0
        state[0, 7] = 3.0
        ball.write_root_state_to_sim(state, env_ids=torch.tensor([0]))
        env.scene.write_data_to_sim()
        env.sim.forward()
        x_positions = []
        for _ in range(30):
            arena.step(torch.zeros(1, 28))
            x_positions.append(ball.data.root_link_pos_w[0, 0].item())
        assert max(x_positions) < 1.2
        assert x_positions[-1] < max(x_positions)  # Rebounded from the sampled wall.
    finally:
        env.close()


def test_ball_spawn_sampler_avoids_ducks_across_randomized_field_sizes():
    torch.manual_seed(7)
    count = 4096
    length = torch.empty(count).uniform_(*FIELD_HALF_LENGTH_RANGE)
    width = torch.empty(count).uniform_(*FIELD_HALF_WIDTH_RANGE)
    player_x = torch.empty(count, 2).uniform_(0.20, 0.70)
    player_y = torch.empty(count, 2).uniform_(-0.28, 0.28)
    players = torch.stack((torch.stack((-player_x[:, 0], player_y[:, 0]), dim=-1),
                           torch.stack((player_x[:, 1], player_y[:, 1]), dim=-1)), dim=1)
    ball = _sample_ball_spawn_xy(players, length, width)
    distance = torch.linalg.vector_norm(ball[:, None, :] - players, dim=-1)
    assert torch.all(distance >= ROBOT_FOOTPRINT_RADIUS_M + BALL_RADIUS + SPAWN_CLEARANCE_M)
    assert torch.all(field_boundary_clearance(ball, length, width) >= BALL_RADIUS + SPAWN_CLEARANCE_M)
    assert torch.all(ball[:, 0].abs() <= torch.minimum(torch.full_like(length, 0.50), 0.25 * length))
    assert torch.all(ball[:, 1].abs() <= torch.minimum(torch.full_like(width, 0.45), 0.50 * width))
    corners = torch.tensor([[x, y] for x in (-0.70, 0.70) for y in (-0.28, 0.28)])
    clearance = field_boundary_clearance(corners, 1.0, 0.65)
    assert torch.all(clearance >= ROBOT_FOOTPRINT_RADIUS_M + SPAWN_CLEARANCE_M)
