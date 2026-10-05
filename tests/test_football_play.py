"""Human tactical controls and fixed opponent behavior."""

import os

from copy import deepcopy

import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from mjlab.envs import ManagerBasedRlEnv

from mjlab_microduck.football.appearance import OPPONENT_COLORS
from mjlab_microduck.football.arena import BALL_RADIUS, FootballArena, FIELD_HALF_LENGTH, FORFEIT_REWARD, make_football_env_cfg
from mjlab_microduck.football.cli import ACTOR_CFG
from mjlab_microduck.football.play import FootballPlayEnv, TacticalKeyboard, TerminalKeys
from mjlab_microduck.football.pool import OpponentChoice, OpponentPool


def test_keyboard_commands_and_one_shot_kick():
    controls = TacticalKeyboard()
    for _ in range(10):
        controls.handle("up")
        controls.handle("left")
        controls.handle("a")
    torch.testing.assert_close(controls.action(torch.device("cpu")),
                               torch.tensor([[0.4, 0.3, 1.0, -1.0]]))
    controls.handle("l")
    assert controls.kick_queue_steps == 100
    assert controls.action(torch.device("cpu"))[0, 3] == 1
    controls.kick_started()
    assert controls.action(torch.device("cpu"))[0, 3] == -1
    controls.handle("l")
    controls.handle(" ")
    torch.testing.assert_close(controls.action(torch.device("cpu")), torch.tensor([[0., 0., 0., -1.]]))
    controls.handle("q")
    assert controls.quit_requested


def test_learned_opponent_and_human_seat_survive_match_reset():
    cfg = deepcopy(ACTOR_CFG)
    cfg.pop("class_name")
    obs = TensorDict({"actor": torch.zeros(1, 64)}, batch_size=[1])
    opponent_model = MLPModel(obs, {"actor": ["actor"]}, "actor", 14, **cfg)
    pool = OpponentPool(seed=2)
    uid = pool.add("joint", opponent_model, ACTOR_CFG, iteration=5)
    choice = OpponentChoice("joint", uid)

    def still(obs):
        return obs.new_zeros((len(obs), 14))

    base = ManagerBasedRlEnv(cfg=make_football_env_cfg(1), device="cpu")
    play = FootballPlayEnv(FootballArena(base), pool, (still, still, still),
                           choice, "away", seed=2)
    try:
        assert play.opponent_choices[0] == choice
        assert not bool(play.learner_home[0])
        palette = play.appearance
        np.testing.assert_allclose(
            base.sim.mj_model.mat_rgba[palette.material_ids["away"]],
            palette.original_rgba["away"],
        )
        np.testing.assert_allclose(
            base.sim.mj_model.mat_rgba[palette.material_ids["home"], :3],
            np.broadcast_to(OPPONENT_COLORS["joint"], (len(palette.material_ids["home"]), 3)),
        )
        ball = base.scene["ball"]
        ball_state = ball.data.default_root_state[:1].clone()
        ball_state[:, :3] = base.scene.env_origins[:1]
        ball_state[:, 0] += play.arena.field_half_length[0] + BALL_RADIUS + 0.08
        ball_state[:, 2] += BALL_RADIUS
        ball.write_root_state_to_sim(ball_state, env_ids=torch.tensor([0]))
        base.scene.write_data_to_sim()
        base.sim.forward()
        _, _, done, extras = play.step(torch.zeros(1, 4))
        assert done[0]
        assert extras["completed"]["env_ids"].tolist() == [0]
        assert play.opponent_choices[0] == choice
        assert not bool(play.learner_home[0])
        assert play.arena.elapsed[0] == 0
        # A learner forfeit ends the fresh match without a goal.
        assert play.arena.scores[0].tolist() == [0, 0]
        robot = base.scene["away"]
        ids = torch.tensor([0])
        state = robot.data.default_root_state[ids].clone()
        state[:, :3] = base.scene.env_origins[ids]
        state[:, 0] += play.arena.field_half_length[0] - 0.025
        state[:, 2] += 0.115
        state[:, 7] = 6.0  # Cross the east goal opening during this control step.
        robot.write_root_state_to_sim(state, env_ids=ids)
        base.scene.write_data_to_sim()
        base.sim.forward()
        _, reward, done, extras = play.step(torch.zeros(1, 4))
        assert done[0]
        assert reward[0] == -FORFEIT_REWARD
        assert extras["completed"]["points"].tolist() == [0.0]
        assert extras["completed"]["own_out_of_bounds"].tolist() == [True]
        assert extras["completed"]["opponent_out_of_bounds"].tolist() == [False]
    finally:
        play.close()


def test_terminal_arrow_and_letter_parsing():
    read_fd, write_fd = os.pipe()
    terminal = TerminalKeys()
    terminal.fd = read_fd
    try:
        os.write(write_fd, b"\x1b[A\x1b[Dleq")
        assert terminal.poll() == ["up", "left", "l", "e", "q"]
    finally:
        os.close(read_fd)
        os.close(write_fd)
