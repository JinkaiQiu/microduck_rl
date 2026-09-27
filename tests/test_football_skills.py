"""The football controller must label student states in raw joint-action units."""

import torch

from mjlab_microduck.football.skills import FootballSkills, FrozenSkillPolicy, teacher_targets
from mjlab_microduck.football.wrapper import ScriptedKickGuard, scripted_dribble, scripted_tactics


class RecordingSkill:
    def __init__(self, value: float):
        self.value = value
        self.inputs = []

    def __call__(self, obs):
        self.inputs.append(obs.clone())
        return obs.new_full((len(obs), 14), self.value)


def _controller(num_envs=3):
    walk, kick, recover = RecordingSkill(1), RecordingSkill(2), RecordingSkill(3)
    return FootballSkills(walk, kick, recover, num_envs), walk, kick, recover


def _obs(n):
    obs = torch.zeros(n, 61)
    obs[:, 5] = -1
    return obs


def test_action_routing_and_command_bounds():
    controller, walk, kick, recover = _controller()
    obs = _obs(3)
    obs[2, 5] = 0  # fallen: gravity no longer points down
    obs[2, 3] = 1
    original_obs = obs.clone()
    actions = torch.tensor([[5., -5., 5., 0.], [0., 0., 0., 1.], [0., 0., 0., 1.]])
    ball = torch.tensor([[0.09, -0.04], [0.09, -0.04], [0.09, -0.04]])
    result = controller.act(actions, obs, ball)
    assert result[:, 0].tolist() == [1, 2, 3]
    torch.testing.assert_close(walk.inputs[0][:, 48:51], torch.tensor([[0.4, -0.3, 1.0]]))
    assert torch.all(kick.inputs[0][:, 48:51] == 0)
    assert torch.all(recover.inputs[0][:, 48:51] == 0)
    torch.testing.assert_close(obs, original_obs)


def test_scripted_baseline_positions_ball_before_requesting_kick():
    obs = torch.zeros(2, 64)
    obs[:, 56] = 1.0  # Opponent's attacking goal is straight ahead.
    obs[0, 48:50] = torch.tensor([0.13, 0.0])
    obs[1, 48:50] = torch.tensor([0.09, -0.042])
    tactics = scripted_tactics(obs)
    assert tactics[0, 0] > 0 and tactics[0, 1] > 0
    assert tactics[0, 3] < 0
    assert tactics[1, 3] > 0
    moving = scripted_tactics(obs[1:], torch.tensor([0.24]))
    settled = scripted_tactics(obs[1:], torch.tensor([0.03]))
    assert moving[0, 3] < 0 and settled[0, 3] > 0

    controller, _, _, _ = _controller(num_envs=2)
    skill_obs = _obs(2)
    # Even a learned tactical actor's early request must wait for the
    # blind kick skill's trained ball placement.
    request = torch.tensor([[0., 0., 0., 1.], [0., 0., 0., 1.]])
    routed = controller.act(request, skill_obs, obs[:, 48:50])
    assert routed[:, 0].tolist() == [1, 2]



def test_scripted_failed_kick_falls_back_to_walking_until_kickoff():
    guard = ScriptedKickGuard(2, torch.device("cpu"), kick_steps=150)
    ids = torch.tensor([1])
    ball = torch.tensor([[0.0, 0.0]])
    remaining = torch.tensor([0, 149])
    guard.after_action(ids, ball, torch.tensor([0]), remaining)
    assert guard.tracking[1] and not guard.failed[1]

    remaining[1] = 100  # One second of kick actions made no ball contact.
    guard.before_action(ids, ball, remaining)
    assert guard.failed[1] and remaining[1] == 0
    obs = torch.zeros(1, 64)
    obs[0, 48:50] = torch.tensor([0.081, -0.043])
    fallback = scripted_dribble(obs)
    assert fallback[0, 0] > 0 and fallback[0, 3] < 0
    controller, walk, kick, _ = _controller(num_envs=2)
    controller.kick_remaining[1] = remaining[1]
    assert controller.act(fallback, _obs(1), obs[:, 48:50], ids)[0, 0] == 1
    assert len(walk.inputs) == 1 and not kick.inputs

    guard.reset(ids)
    assert not guard.failed[1] and not guard.tracking[1]



def test_scripted_near_ball_stall_enters_dribble_fallback_without_a_kick():
    guard = ScriptedKickGuard(1, torch.device("cpu"), kick_steps=150)
    ids = torch.tensor([0])
    remaining = torch.tensor([0])
    ball = torch.tensor([[0.0, 0.0]])
    root = torch.tensor([[-0.10, 0.02]])
    for _ in range(51):
        guard.before_action(ids, ball, remaining, root)
    assert guard.stalled[0] and not guard.failed[0]
    guard.reset(ids)
    assert not guard.stalled[0] and guard.stall_steps[0] == 0

def test_scripted_kick_guard_keeps_a_kick_that_moves_the_ball():
    guard = ScriptedKickGuard(1, torch.device("cpu"), kick_steps=150)
    ids = torch.tensor([0])
    remaining = torch.tensor([149])
    guard.after_action(ids, torch.tensor([[0.0, 0.0]]), torch.tensor([0]), remaining)
    remaining[0] = 100
    guard.before_action(ids, torch.tensor([[0.02, 0.0]]), remaining)
    assert not guard.failed[0]
    assert remaining[0] == 100
    assert not guard.tracking[0]

def test_kick_gate_commitment_and_per_env_reset():
    controller, _, _, _ = _controller(num_envs=2)
    obs = _obs(1)
    ids = torch.tensor([1])
    request = torch.tensor([[0., 0., 0., 1.]])
    out_of_reach = torch.tensor([[0.3, -0.04]])
    in_reach = torch.tensor([[0.09, -0.04]])
    assert controller.act(request, obs, out_of_reach, ids)[0, 0] == 1
    assert controller.act(request, obs, in_reach, ids)[0, 0] == 2
    assert controller.kick_remaining.tolist() == [0, 149]
    assert controller.act(torch.zeros_like(request), obs, out_of_reach, ids)[0, 0] == 2
    controller.reset(ids)
    assert controller.act(torch.zeros_like(request), obs, out_of_reach, ids)[0, 0] == 1
    assert controller.kick_remaining.tolist() == [0, 0]


def test_teacher_labels_student_state_without_touching_other_controller():
    teacher, _, _, _ = _controller(num_envs=1)
    live, _, _, _ = _controller(num_envs=1)
    obs = _obs(1)
    obs64 = torch.cat((obs[:, :48], torch.zeros(1, 16)), dim=1)
    target = teacher_targets(lambda _: torch.tensor([[0., 0., 0., 1.]]), teacher,
                             obs64, obs, torch.tensor([[0.09, -0.04]]))
    assert target.shape == (1, 14)
    assert target[0, 0] == 2
    assert teacher.kick_remaining.item() == 149
    assert live.kick_remaining.item() == 0
    resumed, _, _, _ = _controller(num_envs=1)
    resumed.load_state_dict(teacher.state_dict())
    assert resumed.kick_remaining.item() == 149


def test_checkpoint_loader_uses_actor_normalizer_and_elu(tmp_path):
    state = {
        "mlp.0.weight": torch.zeros(2, 61),
        "mlp.0.bias": torch.tensor([-1., 1.]),
        "mlp.2.weight": torch.ones(14, 2),
        "mlp.2.bias": torch.zeros(14),
        "obs_normalizer._mean": torch.zeros(1, 61),
        "obs_normalizer._std": torch.ones(1, 61),
    }
    state["mlp.0.weight"][0, 0] = 1
    path = tmp_path / "skill.pt"
    torch.save({"actor_state_dict": state}, path)
    actor = FrozenSkillPolicy.from_checkpoint(path, "cpu")
    obs = torch.zeros(2, 61)
    obs[:, 0] = torch.tensor([1.01, 2.02])
    result = actor(obs)
    assert result.shape == (2, 14)
    assert result[1, 0] > result[0, 0]
    assert not any(parameter.requires_grad for parameter in actor.parameters())
