"""Two PPO seats per tug world, with frozen opponents excluded from learning."""

from collections import deque

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from mjlab_microduck.football.pool import OpponentChoice, OpponentPool
from mjlab_microduck.tug_of_war.league import TugLeague


class TugVecEnv(VecEnv):
    """Rollout order is [primary learners in all worlds, secondary seats].

    Only live secondary actions enter PPO. Frozen policies retain their own
    normalizers and remain assigned until the physical bout ends.
    """

    def __init__(self, arena, pool: OpponentPool, *, seed: int = 1,
                 live_fraction: float = 0.5, reference_fraction: float = 0.25):
        if not (0 <= live_fraction <= 1 and 0 <= reference_fraction <= 1
                and live_fraction + reference_fraction <= 1):
            raise ValueError("opponent fractions must be nonnegative and sum to at most one")
        if pool.capacity < 1:
            raise ValueError("opponent pool capacity must be positive")
        from microduck_challenges.tug_of_war.opponent import REFERENCE, load_actor

        self.arena, self.env, self.pool = arena, arena.env, pool
        self.device = torch.device(arena.device)
        self.num_matches, self.num_envs = arena.num_envs, 2 * arena.num_envs
        self.num_actions, self.max_episode_length = 14, arena.max_episode_length
        self.cfg = self.env.cfg
        self.live_fraction, self.reference_fraction = (
            (pool.weights["live"], pool.weights["reference"]) if isinstance(pool, TugLeague)
            else (live_fraction, reference_fraction)
        )
        self.reference = load_actor(REFERENCE).to(self.device).eval().requires_grad_(False)
        self.learner_west = torch.zeros(self.num_matches, dtype=torch.bool, device=self.device)
        self.opponent_choices = [OpponentChoice("live") for _ in range(self.num_matches)]
        self._rng = torch.Generator(device="cpu").manual_seed(seed)
        self.recent_scores = deque(maxlen=200)
        groups = tuple(pool.weights) if isinstance(pool, TugLeague) else ("live", "reference", "snapshot")
        self.recent_results = {kind: deque(maxlen=200) for kind in groups}
        if getattr(arena, "diagnostics", None) is not None:
            arena.diagnostics.context_provider = self._diagnostic_context
        self.reset()

    def _diagnostic_context(self, ids):
        return {"matches": [{"env_id": index, "learner_west": bool(self.learner_west[index]),
                             "opponent": self._opponent_label(self.opponent_choices[index]),
                             "opponent_uid": self.opponent_choices[index].uid} for index in ids]}

    @property
    def unwrapped(self):
        return self.env

    @property
    def episode_length_buf(self):
        return self.arena.elapsed.repeat(2)

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        n = self.num_matches
        if value.shape != (2 * n,) or not torch.equal(value[:n], value[n:]):
            raise ValueError("both rollout seats must share their physical bout clock")
        self.arena.elapsed.copy_(value[:n])

    def _choose_matches(self, ids):
        if isinstance(self.pool, TugLeague):
            indices = ids.cpu().tolist()
            for index, choice in zip(indices, self.pool.choose(len(indices)), strict=True):
                self.learner_west[index] = bool(torch.randint(2, (), generator=self._rng))
                self.opponent_choices[index] = choice
            self.pool.prune({c.uid for c in self.opponent_choices if c.uid >= 0})
            return
        # Reuse football's snapshot storage, device cache and stable IDs. Tug
        # has only joint policies, with a fixed reference replacing its script.
        recent = self.pool._recent["joint"]
        for index in ids.cpu().tolist():
            self.learner_west[index] = bool(torch.randint(2, (), generator=self._rng))
            draw = float(torch.rand((), generator=self._rng))
            if draw < self.live_fraction:
                choice = OpponentChoice("live")
            elif draw < self.live_fraction + self.reference_fraction or not recent:
                choice = OpponentChoice("scripted")
            else:
                uid = recent[int(torch.randint(len(recent), (), generator=self._rng))]
                choice = OpponentChoice("joint", uid)
            self.opponent_choices[index] = choice
        self.pool.prune({c.uid for c in self.opponent_choices if c.uid >= 0})

    def add_snapshot(self, actor, actor_cfg, iteration, scores_by_opponent=None):
        if isinstance(self.pool, TugLeague):
            uid = self.pool.add_snapshot(actor, actor_cfg, iteration, scores_by_opponent)
        else:
            uid = self.pool.add("joint", actor, actor_cfg, iteration)
        self.pool.prune({c.uid for c in self.opponent_choices if c.uid >= 0})
        return uid

    def set_training_update(self, update):
        randomization = getattr(self.arena, "randomization", None)
        if randomization is not None:
            randomization.set_training_update(update)

    def _opponent_label(self, choice):
        if isinstance(self.pool, TugLeague):
            return self.pool.label(choice)
        return "live" if choice.kind == "live" else "reference" if choice.kind == "scripted" else "snapshot"

    def reset(self, env_ids=None):
        ids = (torch.arange(self.num_matches, device=self.device) if env_ids is None
               else torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten())
        self.arena.reset(ids)
        self._choose_matches(ids)
        return self.get_observations()

    def _select(self, west, east, primary=True):
        use_west = self.learner_west if primary else ~self.learner_west
        return torch.where(use_west.reshape((-1,) + (1,) * (west.ndim - 1)), west, east)

    def get_observations(self):
        values = {}
        for group, observe in (("actor", self.arena.actor_obs), ("critic", self.arena.critic_obs)):
            west, east = observe(0), observe(1)
            values[group] = torch.cat((self._select(west, east), self._select(west, east, False)))
        return TensorDict(values, batch_size=[self.num_envs])

    @torch.no_grad()
    def _opponent_actions(self, sampled, obs):
        result = sampled.clone()
        reference = torch.tensor([c.kind == "scripted" for c in self.opponent_choices],
                                 dtype=torch.bool, device=self.device)
        if reference.any():
            result[reference] = self.reference(obs[reference])
        for uid in {c.uid for c in self.opponent_choices if c.kind == "joint"}:
            rows = torch.tensor([c.kind == "joint" and c.uid == uid for c in self.opponent_choices],
                                dtype=torch.bool, device=self.device)
            model = self.pool.model(OpponentChoice("joint", uid), self.device)
            result[rows] = model(TensorDict({"actor": obs[rows]}, batch_size=[int(rows.sum())]))
        return result

    def step(self, actions):
        if actions.shape != (self.num_envs, 14):
            raise ValueError(f"expected {(self.num_envs, 14)} policy actions")
        n = self.num_matches
        # These assignments and masks belong to the transition BEFORE reset.
        live = torch.tensor([c.kind == "live" for c in self.opponent_choices],
                            dtype=torch.bool, device=self.device)
        mask = torch.cat((torch.ones_like(live), live))
        opponent_obs = self._select(self.arena.actor_obs(0), self.arena.actor_obs(1), False)
        primary, secondary = actions[:n], self._opponent_actions(actions[n:], opponent_obs)
        west = self._select(primary, secondary)
        east = self._select(secondary, primary)
        _, side_rewards, dones, info = self.arena.step(torch.cat((west, east), dim=-1))
        rewards = torch.cat((self._select(*side_rewards), self._select(*side_rewards, False)))
        scores = self._select(info["score_cm"][:, 0], info["score_cm"][:, 1])
        primary_side = (~self.learner_west).long()
        points = torch.where(info["winner"] < 0, 0.5, (info["winner"] == primary_side).float())
        done_ids = dones.nonzero(as_tuple=False).flatten()
        completed = {
            "env_ids": done_ids.clone(), "score_cm": scores[done_ids].clone(),
            "points": points[done_ids].clone(), "fault": info["fault"][done_ids].clone(),
            "numerical_reset": info["numerical_reset"][done_ids].clone(),
            "timed_out": info["timed_out"][done_ids].clone(),
            "elapsed_seconds": info["elapsed_seconds"][done_ids].clone(),
            "opponents": [self._opponent_label(self.opponent_choices[index]) for index in done_ids.cpu().tolist()],
        }
        randomization = info.get("randomization")
        if randomization is not None:
            completed["randomization"] = {
                name: value[done_ids].clone() for name, value in randomization.items()
                if torch.is_tensor(value) and value.ndim and value.shape[0] == n
            }
        for index in done_ids.cpu().tolist():
            choice = self.opponent_choices[index]
            kind = self._opponent_label(choice)
            # Do not call faults draws in win-rate summaries. Their -200 cm
            # rating penalty still enters PPO and benchmark rating statistics.
            if kind != "live":
                self.recent_scores.append(float(scores[index]))
            if not info["fault"][index] and not info["numerical_reset"][index]:
                self.recent_results[kind].append(float(points[index]))
        logs = {
            "reward/total": rewards[mask].mean(),
            "info/live_match_fraction": live.float().mean(),
            "info/live_rollout_fraction": mask.float().mean(),
            "info/completed_bouts": float(len(done_ids)),
            "info/faults": info["fault"].float().sum(),
            "info/numerical_resets": info["numerical_reset"].float().sum(),
            "info/timeouts": info["timed_out"].float().sum(),
        }
        if randomization is not None:
            condition = randomization.get("condition")
            if condition is not None:
                for index, name in enumerate(("nominal", "force", "spawn", "both")):
                    logs[f"randomization/fraction_{name}"] = (condition == index).float().mean()
            if "curriculum_strength" in randomization:
                strength = randomization["curriculum_strength"]
                logs["randomization/curriculum_strength"] = strength.float().mean() if torch.is_tensor(strength) else strength
            for name in ("delivered_impulse_delta_v", "delivered_pulses", "force_nonzero"):
                if name in randomization:
                    logs[f"randomization/completed_{name}"] = (randomization[name][done_ids].float().mean()
                                                              if len(done_ids) else 0.)
                    logs[f"randomization/completed_{name}_sum"] = randomization[name][done_ids].float().sum()
            logs["randomization/completed_seats"] = 2 * len(done_ids)
        if isinstance(self.pool, TugLeague):
            labels = [self.pool.label(choice) for choice in self.opponent_choices]
            for kind, probability in self.pool.mixture.items():
                logs[f"league/probability_{kind}"] = probability
                logs[f"league/match_fraction_{kind}"] = labels.count(kind) / n
            logs["league/recent_count"] = len(self.pool._recent["joint"])
            logs["league/archive_count"] = len(self.pool._archive)
        if self.recent_scores:
            logs["info/benchmark_score_cm"] = sum(self.recent_scores) / len(self.recent_scores)
        for kind, results in self.recent_results.items():
            if results:
                logs[f"info/win_rate_vs_{kind}"] = sum(p == 1 for p in results) / len(results)
        rollout_dones = dones.repeat(2).long()
        if len(done_ids):
            self.reset(done_ids)
        # A 45-second bout is a scored terminal, never a bootstrapped truncation.
        return self.get_observations(), rewards, rollout_dones, {
            "learner_mask": mask, "log": logs, "completed": completed,
        }

    def state_dict(self):
        state = {
            "pool": self.pool.state_dict(), "rng_state": self._rng.get_state().clone(),
            "live_fraction": self.live_fraction, "reference_fraction": self.reference_fraction,
            "recent_scores": list(self.recent_scores),
            "recent_results": {kind: list(results) for kind, results in self.recent_results.items()},
        }
        randomization = getattr(self.arena, "randomization", None)
        if randomization is not None:
            state["randomization"] = randomization.state_dict()
        return state

    def load_state_dict(self, state):
        self.pool.load_state_dict(state["pool"])
        self._rng.set_state(state["rng_state"].cpu())
        self.live_fraction, self.reference_fraction = state["live_fraction"], state["reference_fraction"]
        self.recent_scores.clear()
        self.recent_scores.extend(state["recent_scores"])
        for kind, results in self.recent_results.items():
            results.clear()
            results.extend(state["recent_results"].get(kind, ()))
        if state.get("randomization") is not None:
            randomization = getattr(self.arena, "randomization", None)
            if randomization is None:
                raise ValueError("randomized tug checkpoint requires its saved randomization configuration")
            randomization.load_state_dict(state["randomization"])
        # PPO/league resume starts fresh physical bouts, not a partial simulator state.
        self.reset()

    def close(self):
        randomization = getattr(self.arena, "randomization", None)
        if randomization is not None:
            randomization.clear()
        self.env.close()
