"""Checkpoint-aware PPO runner for the three football stages."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab_microduck.football.wrapper import FootballVecEnv


class FootballRunner(OnPolicyRunner):
    """Persist the opponent league and curriculum with ordinary PPO state."""

    env: FootballVecEnv

    def __init__(
        self,
        env: FootballVecEnv,
        train_cfg: dict,
        log_dir: str,
        device: str,
        actor_cfg: dict,
        skill_hashes: dict[str, str],
        source_checkpoint: str | None = None,
    ) -> None:
        self.actor_cfg = deepcopy(actor_cfg)
        self.skill_hashes = dict(skill_hashes)
        self.source_checkpoint = source_checkpoint
        self.best_score = float("-inf")
        for key in ("actor", "critic"):
            for optional in ("cnn_cfg", "distribution_cfg"):
                if train_cfg[key].get(optional) is None:
                    train_cfg[key].pop(optional, None)
            if train_cfg[key].get("rnn_type") is None:
                for optional in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
                    train_cfg[key].pop(optional, None)
        super().__init__(env, train_cfg, log_dir, device)
        if env.stage == "distill":
            self.alg.teacher_target_fn = env.teacher_targets
        self.alg.after_update = self._after_update
        self._log_dir = Path(log_dir)

    def _after_update(self, completed_iterations: int) -> None:
        if completed_iterations % 200 or self.env.stage not in ("tactics", "joint"):
            return
        kind = "tactical" if self.env.stage == "tactics" else "joint"
        self.env.pool.add(kind, self.alg.actor, self.actor_cfg, completed_iterations)
        self.save(str(self._log_dir / f"snapshot_{completed_iterations}.pt"))

    def save(self, path: str, infos: dict | None = None) -> None:
        iteration = int(self.alg.update_count)
        recent = self.env.recent_match_points
        score = sum(recent) / len(recent) if recent else None
        promote = score is None or score > self.best_score
        if score is not None and score > self.best_score:
            self.best_score = score
        state = self.alg.save()
        state["iter"] = iteration
        state["infos"] = infos
        state["football"] = {
            "schema": 1,
            "stage": self.env.stage,
            "actor_cfg": deepcopy(self.actor_cfg),
            "skill_hashes": dict(self.skill_hashes),
            "source_checkpoint": self.source_checkpoint,
            "env_state": self.env.state_dict(),
            "common_step_counter": self.env.unwrapped.common_step_counter,
            "best_score": self.best_score,
        }
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path_obj)
        if self.logger.writer is not None:
            self.logger.save_model(path, iteration)
        # Use the newest checkpoint until at least one full match can be
        # scored. Fixed-seed evaluation remains the final selection gate.
        if promote or not (path_obj.parent / "model_best.pt").exists():
            torch.save(state, path_obj.parent / "model_best.pt")

    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None) -> dict:
        state = torch.load(path, map_location=map_location or self.device, weights_only=False)
        meta = state.get("football")
        if meta is None or meta.get("stage") != self.env.stage:
            raise ValueError("checkpoint is not from the requested football stage")
        if meta["skill_hashes"] != self.skill_hashes:
            raise ValueError("skill checkpoints differ from those in the saved football run")
        self.alg.load(state, load_cfg, strict)
        self.current_learning_iteration = int(state["iter"])
        self.best_score = float(meta.get("best_score", float("-inf")))
        self.env.load_state_dict(meta["env_state"])
        self.env.unwrapped.common_step_counter = int(meta.get("common_step_counter", 0))
        return state.get("infos") or {}
