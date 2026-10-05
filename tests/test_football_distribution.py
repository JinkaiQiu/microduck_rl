"""The tactical actor emits controller-unit actions with matching PPO density."""

import torch
from rsl_rl.modules.distribution import GaussianDistribution

from mjlab_microduck.football.cli import ACTOR_CFG, TACTICAL_ACTOR_CFG, train_config
from mjlab_microduck.football.distribution import TanhGaussianDistribution


LIMITS = torch.tensor((0.4, 0.3, 1.0, 1.0))


def test_tactical_actions_and_deterministic_export_stay_inside_skill_limits():
    torch.manual_seed(7)
    distribution = TanhGaussianDistribution(4, tuple(LIMITS.tolist()))
    raw = torch.tensor([[100.0, -100.0, 2.0, -2.0], [-5.0, 5.0, 0.0, 0.0]])
    distribution.update(raw)
    sampled = torch.stack([distribution.sample() for _ in range(100)])
    assert (sampled.abs() <= LIMITS).all()
    expected = distribution.deterministic_output(raw)
    torch.testing.assert_close(distribution.as_deterministic_output_module()(raw), expected)
    torch.testing.assert_close(distribution.mean, expected)
    assert (expected.abs() <= LIMITS).all()


def test_squashed_log_probability_entropy_and_kl_have_finite_gradients():
    torch.manual_seed(3)
    distribution = TanhGaussianDistribution(4, tuple(LIMITS.tolist()))
    raw = torch.tensor([[0.2, -0.7, 1.2, 0.1]], requires_grad=True)
    distribution.update(raw)
    old_params = tuple(param.detach().clone() for param in distribution.params)
    actions = distribution.sample().detach()
    scaled = actions / LIMITS
    latent = torch.atanh(scaled)
    normal = torch.distributions.Normal(*distribution.params)
    expected = (normal.log_prob(latent) - torch.log(LIMITS * (1 - scaled.square()))).sum(-1)
    torch.testing.assert_close(distribution.log_prob(actions), expected, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(distribution.entropy).all()
    loss = distribution.log_prob(actions).sum() + distribution.entropy.sum()
    loss.backward()
    assert torch.isfinite(raw.grad).all()
    assert torch.isfinite(distribution.std_param.grad).all()
    distribution.update(raw + 0.3)
    expected_kl = torch.distributions.kl_divergence(
        torch.distributions.Normal(*old_params),
        torch.distributions.Normal(*distribution.params),
    ).sum(-1)
    torch.testing.assert_close(distribution.kl_divergence(old_params, distribution.params), expected_kl)


def test_old_tactical_actor_parameters_load_and_only_tactics_switch_distribution():
    old = GaussianDistribution(4, init_std=1.0)
    new = TanhGaussianDistribution(4, tuple(LIMITS.tolist()))
    new.load_state_dict(old.state_dict(), strict=True)
    torch.testing.assert_close(new.std_param, old.std_param)
    assert train_config("tactics", 1, 1, "tensorboard")["actor"] == TACTICAL_ACTOR_CFG
    for stage in ("distill", "joint"):
        assert train_config(stage, 1, 1, "tensorboard")["actor"] == ACTOR_CFG


def test_bounded_tactical_snapshot_reloads_for_frozen_opponent():
    from copy import deepcopy

    from tensordict import TensorDict
    from rsl_rl.models import MLPModel

    from mjlab_microduck.football.pool import OpponentChoice, OpponentPool

    obs = TensorDict({"actor": torch.zeros(2, 66)}, batch_size=[2])
    cfg = deepcopy(TACTICAL_ACTOR_CFG)
    cfg["hidden_dims"] = (16, 8)
    source = MLPModel(obs, {"actor": ["actor"]}, "actor", 4,
                      **{key: value for key, value in deepcopy(cfg).items() if key != "class_name"})
    pool = OpponentPool()
    uid = pool.add("tactical", source, cfg, iteration=3)
    frozen = pool.model(OpponentChoice("tactical", uid), "cpu")
    torch.testing.assert_close(source(obs, stochastic_output=False), frozen(obs, stochastic_output=False))
    assert (frozen(obs, stochastic_output=False).abs() <= LIMITS).all()
