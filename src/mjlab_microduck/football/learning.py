"""PPO with student-visited, pre-action teacher targets for football.

The teacher only supplies an auxiliary target.  The sampled student action is
always the action sent to the environment and match reward remains PPO's reward.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage


class TeacherTargetStorage(RolloutStorage):
    """RL rollout storage that keeps a joint target beside each transition."""

    def __init__(self, num_envs, num_steps, obs, action_shape, device="cpu"):
        super().__init__("rl", num_envs, num_steps, obs, action_shape, device)
        self.teacher_targets = torch.zeros_like(self.actions)

    def add_transition(self, transition):
        if transition.privileged_actions is None:
            raise ValueError("set_teacher_targets must be called before process_env_step")
        if transition.privileged_actions.shape != self.teacher_targets[self.step].shape:
            raise ValueError("teacher target shape does not match learner actions")
        self.teacher_targets[self.step].copy_(transition.privileged_actions)
        super().add_transition(transition)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """Yield the same shuffled indices for PPO data and teacher targets."""
        batch_size = self.num_envs * self.num_transitions_per_env
        if batch_size % num_mini_batches:
            raise ValueError("rollout size must be divisible by num_mini_batches")
        mini_batch_size = batch_size // num_mini_batches
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_params = tuple(p.flatten(0, 1) for p in self.distribution_params)
        targets = self.teacher_targets.flatten(0, 1)
        for _ in range(num_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for i in range(num_mini_batches):
                idx = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
                yield RolloutStorage.Batch(
                    observations=observations[idx],
                    actions=actions[idx],
                    values=values[idx],
                    advantages=advantages[idx],
                    returns=returns[idx],
                    old_actions_log_prob=old_log_prob[idx],
                    old_distribution_params=tuple(p[idx] for p in old_params),
                    privileged_actions=targets[idx],
                )


class FootballPPO(PPO):
    """PPO with optional Huber guidance on the actor's mean joint action.

    Set the iteration before collection.  For distillation, call
    ``set_teacher_targets`` after ``act`` and before ``process_env_step`` at
    every environment step.  Teacher targets must use the same raw action units
    as the student's 14D output, before actuator scaling or clipping.
    """

    def __init__(
        self,
        actor,
        critic,
        storage,
        teacher_loss_coef_start: float = 0.0,
        teacher_loss_coef_end: float | None = None,
        teacher_decay_iterations: int = 2000,
        **kwargs,
    ):
        super().__init__(actor, critic, storage, **kwargs)
        if teacher_loss_coef_end is None:
            teacher_loss_coef_end = 0.1 if teacher_loss_coef_start else 0.0
        if teacher_loss_coef_start < 0 or teacher_loss_coef_end < 0:
            raise ValueError("teacher loss coefficients must be nonnegative")
        if teacher_decay_iterations < 1:
            raise ValueError("teacher_decay_iterations must be positive")
        self.teacher_loss_coef_start = teacher_loss_coef_start
        self.teacher_loss_coef_end = teacher_loss_coef_end
        self.teacher_decay_iterations = teacher_decay_iterations
        self.iteration = 0
        self.update_count = 0
        self.teacher_loss_coef = teacher_loss_coef_start
        self.teacher_target_fn = None
        self.after_update = None
        self.storage = TeacherTargetStorage(
            storage.num_envs,
            storage.num_transitions_per_env,
            storage.observations[0],
            storage.actions_shape,
            storage.device,
        )
        if teacher_loss_coef_start and (self.actor.is_recurrent or self.critic.is_recurrent or self.symmetry or self.rnd):
            raise ValueError("teacher-guided football PPO supports feedforward actors without RND or symmetry")

    def set_iteration(self, iteration: int) -> None:
        if iteration < 0:
            raise ValueError("iteration must be nonnegative")
        self.iteration = iteration
        self.update_count = iteration
        fraction = min(iteration / self.teacher_decay_iterations, 1.0)
        self.teacher_loss_coef = self.teacher_loss_coef_start + fraction * (
            self.teacher_loss_coef_end - self.teacher_loss_coef_start
        )

    def set_teacher_targets(self, targets: torch.Tensor) -> None:
        if self.transition.actions is None:
            raise RuntimeError("call act before setting teacher targets")
        if targets.shape != self.transition.actions.shape:
            raise ValueError(f"expected teacher targets {tuple(self.transition.actions.shape)}, got {tuple(targets.shape)}")
        if not torch.isfinite(targets).all():
            raise ValueError("teacher targets must be finite")
        self.transition.privileged_actions = targets.detach().to(self.device).clone()

    def act(self, obs):
        """Sample the student action, then label that same pre-action state."""
        actions = super().act(obs)
        if self.teacher_loss_coef and self.teacher_target_fn is not None:
            with torch.no_grad():
                self.set_teacher_targets(self.teacher_target_fn(obs))
        return actions

    def process_env_step(self, obs, rewards, dones, extras):
        if self.transition.privileged_actions is None:
            if self.teacher_loss_coef:
                raise RuntimeError("teacher targets are required for every distillation step")
            self.transition.privileged_actions = torch.zeros_like(self.transition.actions)
        super().process_env_step(obs, rewards, dones, extras)

    def update(self) -> dict[str, float]:
        self.set_iteration(self.update_count)
        if self.teacher_loss_coef == 0:
            result = super().update()
            result["teacher"] = 0.0
            result["teacher_coef"] = 0.0
            self.update_count += 1
            if self.after_update is not None:
                self.after_update(self.update_count)
            return result

        if self.actor.is_recurrent or self.critic.is_recurrent or self.symmetry or self.rnd:
            raise ValueError("teacher-guided football PPO supports feedforward actors without RND or symmetry")
        means = {"value": 0.0, "surrogate": 0.0, "entropy": 0.0, "teacher": 0.0}
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
            self.actor(batch.observations, stochastic_output=True)
            log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations)
            entropy = self.actor.output_entropy.mean()
            new_params = self.actor.output_distribution_params
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.no_grad():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, new_params).mean()
                    if kl > 2 * self.desired_kl:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif 0 < kl < self.desired_kl / 2:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate
            ratio = torch.exp(log_prob - batch.old_actions_log_prob.squeeze(-1))
            advantage = batch.advantages.squeeze(-1)
            surrogate = -advantage * ratio
            clipped = -advantage * ratio.clamp(1 - self.clip_param, 1 + self.clip_param)
            surrogate_loss = torch.maximum(surrogate, clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.maximum((values - batch.returns).square(), (value_clipped - batch.returns).square()).mean()
            else:
                value_loss = (values - batch.returns).square().mean()
            teacher_loss = F.huber_loss(self.actor.output_mean, batch.privileged_actions)
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            loss = loss + self.teacher_loss_coef * teacher_loss
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            for key, value in (("value", value_loss), ("surrogate", surrogate_loss), ("entropy", entropy), ("teacher", teacher_loss)):
                means[key] += value.item()
        n = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        self.update_count += 1
        if self.after_update is not None:
            self.after_update(self.update_count)
        return {key: value / n for key, value in means.items()} | {"teacher_coef": self.teacher_loss_coef}

    def save(self):
        result = super().save()
        result["football_iteration"] = self.iteration
        result["football_update_count"] = self.update_count
        return result

    def load(self, loaded_dict, load_cfg, strict):
        restore_iteration = super().load(loaded_dict, load_cfg, strict)
        if restore_iteration:
            self.update_count = loaded_dict.get("football_update_count", loaded_dict.get("iter", 0))
            self.set_iteration(self.update_count)
        return restore_iteration


def warm_start_joint_actor_from_walk(student_actor, walk_checkpoint: str | Path | dict) -> None:
    """Transfer 48 proprio weights into a 64D joint actor and zero game weights.

    The walk checkpoint must have matching hidden and 14D output layers.  Its
    first 48 observation columns are the same ordered proprioception contract.
    This function intentionally leaves the student's critic untouched.
    """
    checkpoint = (
        torch.load(walk_checkpoint, map_location="cpu", weights_only=False)
        if isinstance(walk_checkpoint, (str, Path)) else walk_checkpoint
    )
    source = checkpoint.get("actor_state_dict", checkpoint)
    destination = student_actor.state_dict()
    first_weight = "mlp.0.weight"
    old = source[first_weight]
    new = destination[first_weight]
    if old.shape[0] != new.shape[0] or old.shape[1] < 48 or new.shape[1] != 64:
        raise ValueError("walk/student first layers must have matching width and at least 48/64 input columns")
    new[:, :48] = old[:, :48].to(device=new.device, dtype=new.dtype)
    new[:, 48:] = 0
    for key, value in destination.items():
        if key == first_weight or key.startswith("obs_normalizer."):
            continue
        if key not in source or source[key].shape != value.shape:
            raise ValueError(f"walk/student actor mismatch at {key}")
        value.copy_(source[key].to(device=value.device, dtype=value.dtype))
    for key, value in destination.items():
        if not key.startswith("obs_normalizer."):
            continue
        if key == "obs_normalizer.count":
            if key in source:
                value.copy_(source[key])
        elif key in source:
            if value.shape[-1] != 64 or source[key].shape[-1] < 48:
                raise ValueError(f"walk/student normalizer mismatch at {key}")
            value[..., :48] = source[key][..., :48].to(device=value.device, dtype=value.dtype)
            value[..., 48:] = 0 if key == "obs_normalizer._mean" else 1
    student_actor.load_state_dict(destination)
