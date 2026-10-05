"""The opening football phase has one physical duck and transferable observations."""

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.football.arena import FootballArena, MATCH_STEPS, make_football_env_cfg
from mjlab_microduck.football.cli import build_parser


def test_solo_scene_has_one_duck_and_terminal_30_second_episode(monkeypatch):
    cfg = make_football_env_cfg(2, seed=7, solo=True)
    assert set(cfg.scene.entities) == {"home", "ball"}
    assert set(cfg.actions) == {"home"}
    assert [sensor.name for sensor in cfg.scene.sensors] == ["home_ball_contact"]
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        arena = FootballArena(env)
        actor = arena.reset()[0]
        assert env.action_manager.total_action_dim == 14
        assert actor.shape == (2, 66)
        assert arena.critic_obs(0).shape == (2, 84)
        assert torch.count_nonzero(actor[:, 52:56]) == 0  # Absent opponent motion.
        assert torch.count_nonzero(actor[:, 60:62]) == 0  # Absent opponent heading.
        origin_x = env.scene.env_origins[:, 0]
        own_x = env.scene["home"].data.root_link_pos_w[:, 0] - origin_x
        assert torch.all(own_x * arena.attack_sign[:, 0] < 0)

        arena.elapsed[:] = MATCH_STEPS - 1
        _, _, done, info = arena.step(torch.zeros((2, 14)))
        assert done.all()
        assert info["timed_out"].all()
        arena.reset(done.nonzero(as_tuple=False).flatten())
        assert torch.all(arena.elapsed == 0)

        class ZeroSkills:
            def __init__(self, *args, device):
                self.kick_remaining = torch.zeros(args[-1], device=device, dtype=torch.long)

            def reset(self, ids):
                self.kick_remaining[ids] = 0

            def act(self, actions, proprio, ball_xy):
                return torch.zeros((len(actions), 14), device=actions.device)

        from mjlab_microduck.football import wrapper
        monkeypatch.setattr(wrapper, "FootballSkills", ZeroSkills)
        learner_env = wrapper.FootballVecEnv(arena, "solo", None, ())
        obs, reward, done, extras = learner_env.step(torch.zeros((2, 4)))
        assert obs["actor"].shape == (2, 66)
        assert obs["critic"].shape == (2, 84)
        assert reward.shape == done.shape == (2,)
        assert "reward/body_body_proximity" in extras["log"]
    finally:
        env.close()


def test_solo_warmup_is_an_internal_tactics_phase():
    parser = build_parser()
    args = parser.parse_args(["tactics", "--iterations", "5"])
    assert args.stage == "tactics"
    assert args.solo_iterations == 1000
    assert args.iterations == 5
    assert parser.parse_args(
        ["tactics", "--solo-iterations", "3", "--iterations", "7"]
    ).solo_iterations == 3
    with pytest.raises(SystemExit):
        parser.parse_args(["solo"])
