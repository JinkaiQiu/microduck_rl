"""CPU-only invariants for student-driven football PPO and walk warm start."""

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

from mjlab_microduck.football.learning import FootballPPO, warm_start_joint_actor_from_walk


def _observations(value=0.0, count=4, width=64):
    return TensorDict(
        {"actor": torch.full((count, width), value), "critic": torch.full((count, width), value)},
        batch_size=[count],
    )


def _actor(obs):
    return MLPModel(
        obs, {"actor": ["actor"]}, "actor", 14, hidden_dims=(32, 16),
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )


def _algorithm(teacher_coef=1.0):
    obs = _observations()
    critic = MLPModel(
        obs, {"critic": ["critic"]}, "critic", 1,
        hidden_dims=(32, 16), obs_normalization=True,
    )
    return FootballPPO(
        _actor(obs), critic, RolloutStorage("rl", 4, 2, obs, [14]),
        teacher_loss_coef_start=teacher_coef,
        num_learning_epochs=1,
        num_mini_batches=2,
        desired_kl=None,
    )


def _step(algorithm, obs):
    student_action = algorithm.act(obs)
    algorithm.process_env_step(
        _observations(99.0), torch.ones(4), torch.zeros(4, dtype=torch.bool), {}
    )
    return student_action


def test_teacher_labels_student_pre_action_state_in_raw_14d_units():
    torch.manual_seed(1)
    algorithm = _algorithm()
    calls = []

    def teacher(pre_action_obs):
        calls.append(pre_action_obs["actor"].clone())
        return torch.full((4, 14), 3.25) + pre_action_obs["actor"][:, :1]

    algorithm.teacher_target_fn = teacher
    completed_updates = []
    algorithm.after_update = completed_updates.append
    first = _step(algorithm, _observations(0.0))
    second = _step(algorithm, _observations(1.0))

    assert first.shape == second.shape == (4, 14)
    assert len(calls) == 2
    assert torch.all(calls[0] == 0.0)
    assert torch.all(calls[1] == 1.0)
    assert torch.all(algorithm.storage.teacher_targets[0] == 3.25)
    assert torch.all(algorithm.storage.teacher_targets[1] == 4.25)
    # Targets remain in policy output units; the trainer never clips to [-1, 1].
    algorithm.compute_returns(_observations(2.0))
    losses = algorithm.update()
    assert losses["teacher"] > 0
    assert losses["teacher_coef"] == 1.0
    assert completed_updates == [1]


def test_distillation_requires_teacher_target_every_step():
    algorithm = _algorithm()
    algorithm.act(_observations())
    try:
        algorithm.process_env_step(_observations(), torch.ones(4), torch.zeros(4), {})
    except RuntimeError as exc:
        assert "teacher targets" in str(exc)
    else:
        raise AssertionError("distillation accepted an unlabeled student transition")


def test_walk_warm_start_preserves_proprio_weights_and_zeros_game_columns():
    torch.manual_seed(2)
    walk = _actor(_observations(width=61))
    student = _actor(_observations(width=64))
    with torch.no_grad():
        walk.mlp[0].weight.fill_(0.5)
        walk.obs_normalizer._mean.fill_(2.0)
        walk.obs_normalizer._var.fill_(4.0)
        walk.obs_normalizer._std.fill_(2.0)
        walk.obs_normalizer.count.fill_(7)
    warm_start_joint_actor_from_walk(student, {"actor_state_dict": walk.state_dict()})
    assert torch.all(student.mlp[0].weight[:, :48] == 0.5)
    assert torch.all(student.mlp[0].weight[:, 48:] == 0)
    assert torch.equal(student.mlp[-1].weight, walk.mlp[-1].weight)
    assert torch.all(student.obs_normalizer._mean[..., :48] == 2)
    assert torch.all(student.obs_normalizer._mean[..., 48:] == 0)
    assert torch.all(student.obs_normalizer._std[..., :48] == 2)
    assert torch.all(student.obs_normalizer._std[..., 48:] == 1)
    assert student.obs_normalizer.count.item() == 7


def test_checkpoint_restores_optimizer_and_guidance_schedule():
    torch.manual_seed(3)
    algorithm = _algorithm()
    algorithm.teacher_target_fn = lambda obs: torch.zeros(4, 14)
    _step(algorithm, _observations())
    _step(algorithm, _observations())
    algorithm.compute_returns(_observations())
    algorithm.update()
    assert algorithm.update_count == 1
    saved = algorithm.save()
    restored = _algorithm()
    restored.load(saved, None, True)
    assert restored.update_count == 1
    assert abs(restored.teacher_loss_coef - (1.0 - 0.9 / 2000)) < 1e-7
    assert restored.optimizer.state_dict()["state"]
    restored.set_iteration(2000)
    assert abs(restored.teacher_loss_coef - 0.1) < 1e-7
    unguided = _algorithm(teacher_coef=0)
    unguided.set_iteration(2000)
    assert unguided.teacher_loss_coef == 0.0
