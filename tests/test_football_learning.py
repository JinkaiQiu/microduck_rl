"""CPU-only invariants for student-driven football PPO and walk warm start."""

import math
import torch
from types import SimpleNamespace
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

from mjlab_microduck.football.learning import FootballPPO, warm_start_joint_actor_from_walk


def _observations(value=0.0, count=4, width=66):
    return TensorDict(
        {"actor": torch.full((count, width), value), "critic": torch.full((count, width), value)},
        batch_size=[count],
    )


def _actor(obs, action_dim=14):
    return MLPModel(
        obs, {"actor": ["actor"]}, "actor", action_dim, hidden_dims=(32, 16),
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )


def _algorithm(teacher_coef=1.0, action_dim=14, **kwargs):
    obs = _observations()
    critic = MLPModel(
        obs, {"critic": ["critic"]}, "critic", 1,
        hidden_dims=(32, 16), obs_normalization=True,
    )
    return FootballPPO(
        _actor(obs, action_dim), critic, RolloutStorage("rl", 4, 2, obs, [action_dim]),
        teacher_loss_coef_start=teacher_coef,
        num_learning_epochs=1,
        num_mini_batches=2,
        desired_kl=None,
        **kwargs,
    )


def test_tactical_ppo_updates_executed_skill_weights_and_restores_them():
    torch.manual_seed(17)
    algorithm = _algorithm(teacher_coef=0.0, action_dim=4)
    algorithm.num_mini_batches = 1  # Include every routed branch in the first PPO update.
    controller = SimpleNamespace(**{
        name: torch.nn.Linear(61, 14, bias=False)
        for name in ("walk", "kick", "recover")
    })
    algorithm.configure_trainable_skills(controller, std=0.05)
    pending = []
    algorithm.skill_transition_fn = lambda: pending.pop(0)
    initial = [skill.weight.detach().clone() for skill in algorithm.trainable_skills]
    for step in range(2):
        obs = _observations(float(step))
        algorithm.act(obs)
        tactical_log_prob = algorithm.transition.actions_log_prob.clone()
        skill_obs = torch.randn(4, 61)
        branches = torch.tensor([0, 1, 2, 0])
        means = torch.stack([
            algorithm.trainable_skills[int(branch)](skill_obs[row])
            for row, branch in enumerate(branches)
        ])
        joint, joint_log_prob = algorithm.sample_skills(means, skill_obs, branches)
        pending.append({
            "skill_obs": skill_obs, "branch": branches,
            "joint_actions": joint.detach(), "old_log_prob": joint_log_prob.detach(),
        })
        algorithm.process_env_step(
            obs, torch.tensor([1.0, -0.5, 0.2, 2.0]) * (step + 1),
            torch.zeros(4, dtype=torch.bool), {},
        )
        torch.testing.assert_close(
            algorithm.storage.actions_log_prob[step, :, 0],
            tactical_log_prob.squeeze(-1) + joint_log_prob,
        )
    algorithm.compute_returns(_observations(2.0))
    algorithm.update()
    assert all(not torch.equal(before, skill.weight)
               for before, skill in zip(initial, algorithm.trainable_skills, strict=True))
    saved = algorithm.save()
    restored = _algorithm(teacher_coef=0.0, action_dim=4)
    restored_controller = SimpleNamespace(**{
        name: torch.nn.Linear(61, 14, bias=False)
        for name in ("walk", "kick", "recover")
    })
    restored.configure_trainable_skills(restored_controller)
    restored.load(saved, None, True)
    for expected, actual in zip(algorithm.trainable_skills, restored.trainable_skills, strict=True):
        torch.testing.assert_close(expected.weight, actual.weight)
    assert len(restored.optimizer.param_groups) == 2


def _step(algorithm, obs):
    student_action = algorithm.act(obs)
    algorithm.process_env_step(
        _observations(99.0), torch.ones(4), torch.zeros(4, dtype=torch.bool), {}
    )
    return student_action


def test_ppo_diagnostics_are_opt_in_finite_and_preserve_updates():
    default_keys = {"value", "surrogate", "entropy", "teacher", "teacher_coef"}
    diagnostic_keys = {
        "kl", "clip_fraction", "actor_grad_norm", "critic_grad_norm",
        "entropy_coefficient", "entropy_bonus",
    }
    algorithms = []
    for diagnostics in (False, True):
        torch.manual_seed(23)
        algorithm = _algorithm(teacher_coef=0.0, diagnostics=diagnostics, entropy_coef=0.02)
        for step in range(2):
            _step(algorithm, _observations(float(step)))
        algorithm.compute_returns(_observations(2.0))
        losses = algorithm.update()
        assert set(losses) == default_keys | (diagnostic_keys if diagnostics else set())
        if diagnostics:
            assert all(math.isfinite(losses[key]) for key in diagnostic_keys)
            assert 0 <= losses["clip_fraction"] <= 1
            assert losses["actor_grad_norm"] >= 0
            assert losses["critic_grad_norm"] >= 0
            assert math.isclose(losses["entropy_coefficient"], 0.02, rel_tol=1e-6)
            assert math.isclose(losses["entropy_bonus"], 0.02 * losses["entropy"], rel_tol=1e-6)
        algorithms.append(algorithm)
    for model_name in ("actor", "critic"):
        for expected, actual in zip(
            getattr(algorithms[0], model_name).parameters(),
            getattr(algorithms[1], model_name).parameters(), strict=True,
        ):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)


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



def test_masked_rollout_trains_only_executed_seats_with_or_without_teacher():
    mask = torch.tensor([True, True, True, False])
    for teacher_coef in (0.0, 1.0):
        torch.manual_seed(10)
        algorithm = _algorithm(teacher_coef)
        if teacher_coef:
            def teacher(obs):
                target = torch.zeros(4, 14)
                target[3] = 1000.0  # The frozen seat must not enter Huber loss.
                return target
            algorithm.teacher_target_fn = teacher
        for _ in range(2):
            algorithm.act(_observations())
            algorithm.process_env_step(
                _observations(1.0), torch.ones(4),
                torch.zeros(4, dtype=torch.bool), {"learner_mask": mask},
            )
        assert algorithm.storage.valid.tolist() == [mask.tolist(), mask.tolist()]
        algorithm.compute_returns(_observations())
        batches = list(algorithm.storage.mini_batch_generator(2, 1))
        assert sum(len(batch.actions) for batch in batches) == 6
        assert all((batch.privileged_actions.abs() < 100).all() for batch in batches)
        losses = algorithm.update()
        if teacher_coef:
            assert losses["teacher"] < 1.0
        else:
            assert losses["teacher"] == 0.0


def test_gae_stops_at_match_done_before_next_match_reward():
    algorithm = _algorithm(teacher_coef=0.0)
    storage = algorithm.storage
    storage.values.zero_()
    storage.rewards.zero_()
    storage.dones.fill_(1)
    storage.rewards[0, :2, 0] = 1.0
    storage.rewards[1, :2, 0] = 100.0
    storage.dones[0, 1] = 0  # Comparison seat continues in the same match.
    algorithm.compute_returns(_observations())
    torch.testing.assert_close(storage.returns[0, 0, 0], torch.tensor(1.0))
    assert storage.returns[0, 1, 0] > 1.0


def test_walk_warm_start_preserves_proprio_weights_and_zeros_game_columns():
    torch.manual_seed(2)
    walk = _actor(_observations(width=61))
    student = _actor(_observations(width=66))
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


def test_old_football_checkpoint_expands_actor_critic_and_optimizer_inputs():
    from mjlab_microduck.football.learning import expand_football_checkpoint_observations

    old_actor = _actor(_observations(width=64))
    new_actor = _actor(_observations(width=66))
    old_critic = MLPModel(
        _observations(width=82), {"critic": ["critic"]}, "critic", 1,
        hidden_dims=(32, 16), obs_normalization=True,
    )
    new_critic = MLPModel(
        _observations(width=84), {"critic": ["critic"]}, "critic", 1,
        hidden_dims=(32, 16), obs_normalization=True,
    )
    state = {
        "actor_state_dict": old_actor.state_dict(),
        "critic_state_dict": old_critic.state_dict(),
        "optimizer_state_dict": {"state": {
            0: {"exp_avg": torch.ones(32, 64)},
            1: {"exp_avg": torch.ones(32, 82)},
        }},
    }
    expanded = expand_football_checkpoint_observations(state, new_actor, new_critic)
    new_actor.load_state_dict(expanded["actor_state_dict"])
    new_critic.load_state_dict(expanded["critic_state_dict"])
    torch.testing.assert_close(new_actor.mlp[0].weight[:, :64], old_actor.mlp[0].weight)
    assert torch.all(new_actor.mlp[0].weight[:, 64:] == 0)
    torch.testing.assert_close(new_critic.mlp[0].weight[:, :64], old_critic.mlp[0].weight[:, :64])
    assert torch.all(new_critic.mlp[0].weight[:, 64:66] == 0)
    torch.testing.assert_close(new_critic.mlp[0].weight[:, 66:], old_critic.mlp[0].weight[:, 64:])
    assert torch.all(new_actor.obs_normalizer._std[:, 64:] == 1)
    assert torch.all(new_critic.obs_normalizer._std[:, 64:66] == 1)
    assert state["optimizer_state_dict"]["state"][0]["exp_avg"].shape == (32, 66)
    assert state["optimizer_state_dict"]["state"][1]["exp_avg"].shape == (32, 84)
