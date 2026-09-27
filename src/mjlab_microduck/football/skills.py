"""Batched, GPU-native adapters for the existing 61D Microduck skills.

Skill actions are the same 14 joint-position offsets used by the original
training environments.  The caller supplies the current (pre-action) 61D
observation, including its true previous action; this module changes only the
three twist-command slots before querying a frozen skill.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import torch
from torch import nn


class FrozenSkillPolicy(nn.Module):
    """Deterministic PyTorch inference from a standard rsl_rl MLP checkpoint.

    The repository's 61D policies use an ELU MLP and a Gaussian whose
    deterministic output is its mean. Normalizer buffers live in the actor
    state dict, so no ONNX export or CPU inference is needed.
    """

    def __init__(self, state: dict[str, torch.Tensor], input_dim: int = 61, output_dim: int = 14):
        super().__init__()
        weights = []
        for key, value in state.items():
            if key.startswith("mlp.") and key.endswith(".weight"):
                index = int(key.split(".")[1])
                weights.append((index, value))
        weights.sort()
        if not weights or weights[0][1].shape[1] != input_dim or weights[-1][1].shape[0] != output_dim:
            raise ValueError(f"Expected an {input_dim}D to {output_dim}D MLP actor checkpoint")
        modules: list[nn.Module] = []
        previous = input_dim
        for position, (index, weight) in enumerate(weights):
            if index != 2 * position or weight.ndim != 2 or weight.shape[1] != previous:
                raise ValueError("Unsupported skill MLP layer layout")
            layer = nn.Linear(previous, weight.shape[0])
            layer.weight.data.copy_(weight)
            bias = state.get(f"mlp.{index}.bias")
            if bias is None or bias.shape != layer.bias.shape:
                raise ValueError(f"Missing or invalid mlp.{index}.bias")
            layer.bias.data.copy_(bias)
            modules.append(layer)
            if position < len(weights) - 1:
                modules.append(nn.ELU())
            previous = weight.shape[0]
        self.mlp = nn.Sequential(*modules)
        mean = state.get("obs_normalizer._mean")
        std = state.get("obs_normalizer._std")
        if (mean is None) != (std is None):
            raise ValueError("Incomplete actor observation normalizer")
        if mean is None:
            mean = torch.zeros(1, input_dim)
            std = torch.ones(1, input_dim)
            self.normalized = False
        else:
            if mean.shape != (1, input_dim) or std.shape != (1, input_dim):
                raise ValueError("Skill observation normalizer must have shape (1, 61)")
            self.normalized = True
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("std", std.detach().clone())
        self.eval()
        self.requires_grad_(False)

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, device: torch.device | str,
        input_dim: int = 61, output_dim: int = 14,
    ) -> "FrozenSkillPolicy":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("actor_state_dict", checkpoint)
        if not isinstance(state, dict):
            raise ValueError(f"No actor state dict in {path}")
        return cls(state, input_dim, output_dim).to(device)

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 2 or obs.shape[1] != self.mean.shape[1]:
            raise ValueError(f"Expected (N, {self.mean.shape[1]}) skill observations")
        if obs.device != self.mean.device:
            raise ValueError("Skill observation and model must be on the same device")
        if self.normalized:
            obs = (obs - self.mean) / (self.std + 1e-2)
        return self.mlp(obs)


class FootballSkills:
    """Convert 4D tactical commands into raw 14D joint offset actions.

    ``act`` accepts a selected batch of environment rows, and ``env_ids`` maps
    those rows to the controller's persistent kick timers. The caller should
    invoke it exactly once per control step and call ``reset`` after a kickoff
    or whole-match reset. A separate instance is needed for each player.
    """

    def __init__(
        self, walk: Callable[[torch.Tensor], torch.Tensor],
        kick: Callable[[torch.Tensor], torch.Tensor],
        recover: Callable[[torch.Tensor], torch.Tensor],
        num_envs: int, dt: float = 0.02, device: torch.device | str = "cpu",
    ) -> None:
        self.walk, self.kick, self.recover = walk, kick, recover
        self.num_envs = num_envs
        self.dt = dt
        self.device = torch.device(device)
        self.kick_steps = max(1, math.ceil(3.0 / dt))
        self.kick_remaining = torch.zeros(num_envs, dtype=torch.long, device=device)

    @classmethod
    def from_checkpoints(
        cls, walk_path: str | Path, kick_path: str | Path,
        recover_path: str | Path, num_envs: int,
        device: torch.device | str, dt: float = 0.02,
    ) -> "FootballSkills":
        return cls(
            FrozenSkillPolicy.from_checkpoint(walk_path, device),
            FrozenSkillPolicy.from_checkpoint(kick_path, device),
            FrozenSkillPolicy.from_checkpoint(recover_path, device),
            num_envs=num_envs, dt=dt, device=device,
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.kick_remaining.zero_()
        else:
            self.kick_remaining[env_ids] = 0

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"kick_remaining": self.kick_remaining.clone()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        remaining = state["kick_remaining"].to(self.device)
        if remaining.shape != self.kick_remaining.shape:
            raise ValueError("Incompatible football skill timer shape")
        self.kick_remaining.copy_(remaining)

    @torch.no_grad()
    def act(
        self, tactic_actions: torch.Tensor, obs61: torch.Tensor,
        ball_body_xy: torch.Tensor, env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n = tactic_actions.shape[0]
        if tactic_actions.shape != (n, 4) or obs61.shape != (n, 61) or ball_body_xy.shape != (n, 2):
            raise ValueError("Expected tactical (N,4), skill obs (N,61), ball body XY (N,2)")
        if tactic_actions.device != self.device or obs61.device != self.device or ball_body_xy.device != self.device:
            raise ValueError("Football skill inputs must be on controller device")
        if env_ids is None:
            if n != self.num_envs:
                raise ValueError("env_ids are required for a partial batch")
            env_ids = torch.arange(n, device=self.device)
        if env_ids.shape != (n,) or env_ids.dtype != torch.long or torch.unique(env_ids).numel() != n:
            raise ValueError("env_ids must be distinct long indices, one per input row")
        if n == 0:
            return obs61.new_empty((0, 14))

        action = tactic_actions.nan_to_num().clone()
        action[:, 0].clamp_(-0.4, 0.4)
        action[:, 1].clamp_(-0.3, 0.3)
        action[:, 2].clamp_(-1.0, 1.0)
        # Projected gravity is [3:6]; upright means gravity points down.
        gravity = obs61[:, 3:6]
        cosine = -gravity[:, 2] / gravity.norm(dim=1).clamp_min(1e-6)
        fallen = cosine < math.cos(math.radians(35.0))
        ball_x, ball_y = ball_body_xy.unbind(-1)
        # The blind right-foot kick actor was trained with the ball centered at
        # (0.09, -0.042) m in body coordinates, with ±0.015 m placement noise.
        # A broad gate started long 3-second kick sequences when the foot could
        # not reach the ball, leaving the scripted opponent seemingly frozen.
        ready = (ball_x - 0.09).abs() <= 0.018
        ready &= (ball_y + 0.042).abs() <= 0.018
        start = (action[:, 3] > 0.0) & ready & ~fallen & (self.kick_remaining[env_ids] == 0)
        self.kick_remaining[env_ids[start]] = self.kick_steps
        active_kick = self.kick_remaining[env_ids] > 0
        walking = ~fallen & ~active_kick
        kicking = ~fallen & active_kick
        result = obs61.new_empty((n, 14))
        if walking.any():
            walk_obs = obs61[walking].clone()
            walk_obs[:, 48:51] = action[walking, :3]
            result[walking] = self.walk(walk_obs)
        if kicking.any():
            kick_obs = obs61[kicking].clone()
            kick_obs[:, 48:51] = 0
            result[kicking] = self.kick(kick_obs)
        if fallen.any():
            recover_obs = obs61[fallen].clone()
            recover_obs[:, 48:51] = 0
            result[fallen] = self.recover(recover_obs)
        self.kick_remaining[env_ids] = (self.kick_remaining[env_ids] - 1).clamp_min_(0)
        return result


@torch.no_grad()
def teacher_targets(
    tactical_policy: Callable[[torch.Tensor], torch.Tensor],
    controller: FootballSkills,
    actor_obs64: torch.Tensor,
    obs61: torch.Tensor,
    ball_body_xy: torch.Tensor,
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Label the student's current pre-action states; no physics is advanced."""
    if actor_obs64.ndim != 2 or actor_obs64.shape != (obs61.shape[0], 64):
        raise ValueError("Expected 64D football actor observations")
    tactics = tactical_policy(actor_obs64)
    return controller.act(tactics, obs61, ball_body_xy, env_ids)
