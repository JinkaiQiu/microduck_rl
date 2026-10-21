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
        self.valid = torch.ones((num_steps, num_envs), device=device, dtype=torch.bool)
        self.skill_obs = None
        self.skill_branches = None
        self.skill_actions = None

    def enable_skill_training(self) -> None:
        shape = (self.num_transitions_per_env, self.num_envs)
        self.skill_obs = torch.zeros((*shape, 61), device=self.device)
        self.skill_branches = torch.zeros(shape, device=self.device, dtype=torch.long)
        self.skill_actions = torch.zeros((*shape, 14), device=self.device)

    def add_transition(self, transition):
        if transition.privileged_actions is None:
            raise ValueError("set_teacher_targets must be called before process_env_step")
        if transition.privileged_actions.shape != self.teacher_targets[self.step].shape:
            raise ValueError("teacher target shape does not match learner actions")
        self.teacher_targets[self.step].copy_(transition.privileged_actions)
        valid = getattr(transition, "valid_mask", None)
        if valid is None:
            valid = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        if valid.shape != (self.num_envs,):
            raise ValueError("live-seat mask must have one entry per rollout row")
        self.valid[self.step].copy_(valid)
        if self.skill_obs is not None:
            detail = getattr(transition, "skill_transition", None)
            if detail is None:
                raise ValueError("skill transition is required when training skills")
            for target, key in (
                (self.skill_obs, "skill_obs"),
                (self.skill_branches, "branch"),
                (self.skill_actions, "joint_actions"),
            ):
                if detail[key].shape != target[self.step].shape:
                    raise ValueError(f"invalid {key} shape in skill transition")
                target[self.step].copy_(detail[key])
        super().add_transition(transition)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """Shuffle only executed live-policy actions, keeping targets aligned."""
        indices = self.valid.flatten().nonzero(as_tuple=False).squeeze(-1)
        if not indices.numel():
            raise ValueError("a football rollout has no live-policy transitions")
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_params = tuple(p.flatten(0, 1) for p in self.distribution_params)
        targets = self.teacher_targets.flatten(0, 1)
        for _ in range(num_epochs):
            shuffled = indices[torch.randperm(len(indices), device=self.device)]
            for idx in torch.tensor_split(shuffled, num_mini_batches):
                if not idx.numel():
                    continue
                batch = RolloutStorage.Batch(
                    observations=observations[idx],
                    actions=actions[idx],
                    values=values[idx],
                    advantages=advantages[idx],
                    returns=returns[idx],
                    old_actions_log_prob=old_log_prob[idx],
                    old_distribution_params=tuple(p[idx] for p in old_params),
                    privileged_actions=targets[idx],
                )
                if self.skill_obs is not None:
                    batch.skill_obs = self.skill_obs.flatten(0, 1)[idx]
                    batch.skill_branches = self.skill_branches.flatten(0, 1)[idx]
                    batch.skill_actions = self.skill_actions.flatten(0, 1)[idx]
                yield batch


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
        self.trainable_skills = None
        self.skill_transition_fn = None
        self.skill_std = 0.1
        self.storage = TeacherTargetStorage(
            storage.num_envs,
            storage.num_transitions_per_env,
            storage.observations[0],
            storage.actions_shape,
            storage.device,
        )
        if self.actor.is_recurrent or self.critic.is_recurrent or self.symmetry or self.rnd:
            raise ValueError("football PPO supports feedforward actors without RND or symmetry")

    def configure_trainable_skills(self, controller, *, std: float = 0.05,
                                   learning_rate_scale: float = 0.01) -> None:
        """Make sampled joint actions part of the tactical PPO likelihood."""
        if self.trainable_skills is not None or std <= 0 or learning_rate_scale <= 0:
            raise ValueError("skill training already configured or invalid skill hyperparameter")
        self.skill_std = float(std)
        self.trainable_skills = (controller.walk, controller.kick, controller.recover)
        for skill in self.trainable_skills:
            skill.requires_grad_(True)
        self.optimizer.add_param_group({
            "params": [parameter for skill in self.trainable_skills for parameter in skill.parameters()],
            "lr": self.learning_rate * learning_rate_scale,
            "skill_lr_scale": learning_rate_scale,
        })
        self.storage.enable_skill_training()

    def sample_skills(self, mean, skill_obs, branch):
        """Rollout sample; the skill log probability is stored with the tactical one."""
        if self.trainable_skills is None:
            raise RuntimeError("skill training has not been configured")
        if mean.shape != (len(branch), 14) or skill_obs.shape != (len(branch), 61):
            raise ValueError("invalid skill sampler inputs")
        distribution = torch.distributions.Normal(mean, self.skill_std)
        action = distribution.sample()
        return action, distribution.log_prob(action).sum(dim=-1)

    def _skill_log_prob(self, skill_obs, branches, joint_actions):
        mean = torch.zeros_like(joint_actions)
        for branch, skill in enumerate(self.trainable_skills):
            selected = branches == branch
            if selected.any():
                mean[selected] = skill(skill_obs[selected])
        if ((branches < 0) | (branches > 2)).any():
            raise ValueError("invalid football skill branch")
        return torch.distributions.Normal(mean, self.skill_std).log_prob(joint_actions).sum(dim=-1)

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
        valid = extras.get("learner_mask")
        if valid is None:
            valid = torch.ones(len(rewards), device=self.device, dtype=torch.bool)
        valid = valid.to(device=self.device, dtype=torch.bool)
        if valid.shape != (len(rewards),):
            raise ValueError("live-seat mask must match the rollout batch")
        self.transition.valid_mask = valid
        if self.trainable_skills is not None:
            if self.skill_transition_fn is None:
                raise RuntimeError("skill transition callback is missing")
            detail = self.skill_transition_fn()
            if detail is None:
                raise RuntimeError("executed joint actions are missing from the skill rollout")
            if detail["old_log_prob"].shape != (len(rewards),):
                raise ValueError("skill log probabilities must match rollout rows")
            if not torch.isfinite(detail["old_log_prob"][valid]).all():
                raise ValueError("nonfinite skill log probabilities")
            self.transition.skill_transition = detail
            self.transition.actions_log_prob += detail["old_log_prob"]
        # Frozen seats are observed for action routing but never train the
        # actor, critic, their normalizers, or the teacher-guidance loss.
        live_obs = self.transition.observations[valid]
        self.actor.update_normalization(live_obs)
        self.critic.update_normalization(live_obs)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1,
            )
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs):
        """Compute GAE per seat, normalizing using executed actions only."""
        st = self.storage
        last_values = self.critic(obs).detach()
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            not_done = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + not_done * self.gamma * next_values - st.values[step]
            advantage = delta + not_done * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        st.advantages = st.returns - st.values
        if not self.normalize_advantage_per_mini_batch:
            live = st.advantages[st.valid]
            st.advantages = (st.advantages - live.mean()) / (live.std(unbiased=False) + 1e-8)

    def update(self) -> dict[str, float]:
        self.set_iteration(self.update_count)
        if self.actor.is_recurrent or self.critic.is_recurrent or self.symmetry or self.rnd:
            raise ValueError("football PPO supports feedforward actors without RND or symmetry")
        means = {"value": 0.0, "surrogate": 0.0, "entropy": 0.0, "teacher": 0.0}
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        batches = 0
        for batch in generator:
            batches += 1
            if self.normalize_advantage_per_mini_batch:
                batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
            self.actor(batch.observations, stochastic_output=True)
            log_prob = self.actor.get_output_log_prob(batch.actions)
            if self.trainable_skills is not None:
                log_prob = log_prob + self._skill_log_prob(
                    batch.skill_obs, batch.skill_branches, batch.skill_actions,
                )
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
                        group["lr"] = self.learning_rate * group.get("skill_lr_scale", 1.0)
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
            teacher_loss = (
                F.huber_loss(self.actor.output_mean, batch.privileged_actions)
                if self.teacher_loss_coef else self.actor.output_mean.new_zeros(())
            )
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            loss = loss + self.teacher_loss_coef * teacher_loss
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            if self.trainable_skills is not None:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for skill in self.trainable_skills for parameter in skill.parameters()],
                    self.max_grad_norm,
                )
            self.optimizer.step()
            for key, value in (("value", value_loss), ("surrogate", surrogate_loss), ("entropy", entropy), ("teacher", teacher_loss)):
                means[key] += value.item()
        n = batches
        self.storage.clear()
        self.update_count += 1
        if self.after_update is not None:
            self.after_update(self.update_count)
        return {key: value / n for key, value in means.items()} | {"teacher_coef": self.teacher_loss_coef}

    def save(self):
        result = super().save()
        if self.trainable_skills is not None:
            result["skill_state_dicts"] = {
                name: skill.state_dict() for name, skill in zip(
                    ("walk", "kick", "recovery"), self.trainable_skills, strict=True,
                )
            }
            result["skill_action_std"] = self.skill_std
        result["football_iteration"] = self.iteration
        result["football_update_count"] = self.update_count
        return result

    def load(self, loaded_dict, load_cfg, strict):
        if self.trainable_skills is not None:
            states = loaded_dict.get("skill_state_dicts")
            if states is None:
                raise ValueError("checkpoint has no trainable skill state")
            for name, skill in zip(("walk", "kick", "recovery"), self.trainable_skills, strict=True):
                skill.load_state_dict(states[name], strict=True)
            self.skill_std = float(loaded_dict["skill_action_std"])
        restore_iteration = super().load(loaded_dict, load_cfg, strict)
        if restore_iteration:
            self.update_count = loaded_dict.get("football_update_count", loaded_dict.get("iter", 0))
            self.set_iteration(self.update_count)
        return restore_iteration


def warm_start_joint_actor_from_walk(student_actor, walk_checkpoint: str | Path | dict) -> None:
    """Transfer 48 proprio weights into a 66D joint actor and zero game weights.

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
    if old.shape[0] != new.shape[0] or old.shape[1] < 48 or new.shape[1] != 66:
        raise ValueError("walk/student first layers must have matching width and at least 48/66 input columns")
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
            if value.shape[-1] != 66 or source[key].shape[-1] < 48:
                raise ValueError(f"walk/student normalizer mismatch at {key}")
            value[..., :48] = source[key][..., :48].to(device=value.device, dtype=value.dtype)
            value[..., 48:] = 0 if key == "obs_normalizer._mean" else 1
    student_actor.load_state_dict(destination)


def expand_football_checkpoint_observations(
    checkpoint: dict, actor, critic=None,
) -> dict:
    """Load 64D/82D football networks into the size-aware 66D/84D layout.

    The two field-size features are appended to actor observations and inserted
    between game and velocity features in critic observations. New first-layer
    columns start at zero, so a previous policy initially behaves as before.
    """
    actor_source = checkpoint["actor_state_dict"]
    actor_target = actor.state_dict()
    old_actor_width = actor_source["mlp.0.weight"].shape[1]
    new_actor_width = actor_target["mlp.0.weight"].shape[1]
    if old_actor_width == new_actor_width:
        return checkpoint
    if (old_actor_width, new_actor_width) != (64, 66):
        raise ValueError(f"unsupported football actor observation change: {old_actor_width} -> {new_actor_width}")

    def expand(state: dict, target: dict, *, insert_at: int) -> None:
        for key in ("mlp.0.weight", "obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"):
            old = state[key]
            new = target[key].clone().to(device=old.device, dtype=old.dtype)
            if key in ("obs_normalizer._var", "obs_normalizer._std"):
                new.fill_(1)
            else:
                new.zero_()
            new[..., :insert_at] = old[..., :insert_at]
            new[..., insert_at + 2:] = old[..., insert_at:]
            state[key] = new

    expand(actor_source, actor_target, insert_at=64)
    if critic is not None:
        critic_source = checkpoint["critic_state_dict"]
        critic_target = critic.state_dict()
        if (critic_source["mlp.0.weight"].shape[1], critic_target["mlp.0.weight"].shape[1]) != (82, 84):
            raise ValueError("old football critic must expand from 82D to 84D")
        expand(critic_source, critic_target, insert_at=64)
        optimizer = checkpoint.get("optimizer_state_dict")
        if optimizer is not None:
            actor_shape = (actor_target["mlp.0.weight"].shape[0], 64)
            critic_shape = (critic_target["mlp.0.weight"].shape[0], 82)
            for param_state in optimizer["state"].values():
                for key, value in param_state.items():
                    if not isinstance(value, torch.Tensor):
                        continue
                    if tuple(value.shape) == tuple(actor_shape):
                        expanded = value.new_zeros((value.shape[0], 66))
                        expanded[:, :64] = value
                        param_state[key] = expanded
                    elif tuple(value.shape) == tuple(critic_shape):
                        expanded = value.new_zeros((value.shape[0], 84))
                        expanded[:, :64] = value[:, :64]
                        expanded[:, 66:] = value[:, 64:]
                        param_state[key] = expanded
    return checkpoint
