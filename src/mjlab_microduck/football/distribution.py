"""Bounded tactical action distribution for football PPO.

The walking controller accepts finite velocity/yaw ranges. A plain Gaussian
followed by controller-side clipping gives many different PPO actions the same
physical effect, so the distribution applies a smooth, invertible transform
before actions are stored or executed.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from rsl_rl.modules.distribution import GaussianDistribution


class _DeterministicTanhAction(nn.Module):
    def __init__(self, limits: torch.Tensor, latent_mean_limit: float):
        super().__init__()
        self.register_buffer("limits", limits.clone())
        self.latent_mean_limit = latent_mean_limit

    def forward(self, raw_mean: torch.Tensor) -> torch.Tensor:
        latent_mean = self.latent_mean_limit * torch.tanh(raw_mean / self.latent_mean_limit)
        return self.limits * torch.tanh(latent_mean)


class TanhGaussianDistribution(GaussianDistribution):
    """Gaussian in latent space, scaled tanh actions in controller units."""

    def __init__(
        self, output_dim: int, limits: tuple[float, ...], init_std: float = 0.6,
        std_type: str = "scalar", latent_mean_limit: float = 2.0,
    ) -> None:
        super().__init__(output_dim, init_std=init_std, std_type=std_type)
        if len(limits) != output_dim or any(limit <= 0 for limit in limits):
            raise ValueError("one positive action limit is required per output")
        if latent_mean_limit <= 0:
            raise ValueError("latent_mean_limit must be positive")
        # Keep the old Gaussian parameter names/load path usable on resume.
        self.register_buffer("limits", torch.tensor(limits, dtype=torch.float32), persistent=False)
        self.latent_mean_limit = float(latent_mean_limit)

    def _bounded_latent_mean(self, raw_mean: torch.Tensor) -> torch.Tensor:
        return self.latent_mean_limit * torch.tanh(raw_mean / self.latent_mean_limit)

    def update(self, mlp_output: torch.Tensor) -> None:
        super().update(self._bounded_latent_mean(mlp_output))

    def _squash(self, latent: torch.Tensor) -> torch.Tensor:
        return self.limits.to(latent) * torch.tanh(latent)

    def _log_jacobian(self, latent: torch.Tensor) -> torch.Tensor:
        # log(1 - tanh(z)^2) in a form that stays finite for large |z|.
        log_tanh_derivative = 2 * (math.log(2) - latent - F.softplus(-2 * latent))
        return torch.log(self.limits.to(latent)) + log_tanh_derivative

    def sample(self) -> torch.Tensor:
        return self._squash(self._distribution.sample())

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return self._squash(self._bounded_latent_mean(mlp_output))

    def as_deterministic_output_module(self) -> nn.Module:
        return _DeterministicTanhAction(self.limits, self.latent_mean_limit)

    @property
    def mean(self) -> torch.Tensor:
        return self._squash(self._distribution.mean)

    @property
    def params(self) -> tuple[torch.Tensor, torch.Tensor]:
        # KL is invariant under the same invertible action transform.
        return self._distribution.mean, self._distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        latent = self._distribution.rsample()
        return self._distribution.entropy().sum(dim=-1) + self._log_jacobian(latent).sum(dim=-1)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        scaled = outputs / self.limits.to(outputs)
        margin = torch.finfo(scaled.dtype).eps
        latent = torch.atanh(scaled.clamp(-1 + margin, 1 - margin))
        return (self._distribution.log_prob(latent) - self._log_jacobian(latent)).sum(dim=-1)
