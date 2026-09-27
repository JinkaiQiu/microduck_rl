"""Frozen opponent snapshots for the football training stages.

Snapshots contain their observation normalizer as part of the actor state.
An opponent is selected once per match; stable IDs keep an in-progress match
on the same opponent when the rolling pool receives another checkpoint.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import torch
from tensordict import TensorDict


PolicyKind = Literal["tactical", "joint"]


@dataclass(frozen=True)
class OpponentChoice:
    kind: Literal["scripted", "tactical", "joint"]
    uid: int = -1


class OpponentPool:
    """Bounded, resumable self-play pool with per-match sampling."""

    def __init__(self, seed: int = 42, capacity: int = 8):
        self.capacity = capacity
        self._next_uid = 0
        self._snapshots: dict[int, dict] = {}
        self._recent: dict[PolicyKind, list[int]] = {"tactical": [], "joint": []}
        self._model_cache: dict[tuple[int, str], torch.nn.Module] = {}
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(seed)

    @property
    def tactical_count(self) -> int:
        return len(self._recent["tactical"])

    @property
    def joint_count(self) -> int:
        return len(self._recent["joint"])

    def add(self, kind: PolicyKind, actor: torch.nn.Module, actor_cfg: dict, iteration: int) -> int:
        """Freeze the actor, including normalizer buffers, in CPU storage."""
        return self.add_state(kind, actor.state_dict(), actor_cfg, iteration)

    def add_state(self, kind: PolicyKind, actor_state_dict: dict, actor_cfg: dict, iteration: int) -> int:
        """Add an actor state from a checkpoint without building it first."""
        if kind not in self._recent:
            raise ValueError(f"unknown opponent kind: {kind}")
        uid = self._next_uid
        self._next_uid += 1
        self._snapshots[uid] = {
            "kind": kind,
            "iteration": int(iteration),
            "actor_cfg": deepcopy(actor_cfg),
            "actor_state_dict": {
                key: value.detach().cpu().clone() for key, value in actor_state_dict.items()
            },
        }
        recent = self._recent[kind]
        recent.append(uid)
        del recent[:-self.capacity]
        return uid

    def sample(self, count: int, stage: Literal["tactics", "distill", "joint"]) -> list[OpponentChoice]:
        """Sample frozen opponents independently for each new match."""
        out: list[OpponentChoice] = []
        for _ in range(count):
            draw = float(torch.rand((), generator=self._rng))
            if stage == "tactics":
                kind = "tactical" if self.tactical_count and draw < 0.8 else "scripted"
            elif stage == "distill":
                kind = "tactical" if self.tactical_count else "scripted"
            elif stage == "joint":
                if self.joint_count == 0:
                    kind = "tactical" if self.tactical_count else "scripted"
                elif self.tactical_count == 0:
                    kind = "joint"
                else:
                    kind = "tactical" if draw < 0.25 else "joint"
            else:
                raise ValueError(f"unknown training stage: {stage}")
            if kind == "scripted":
                out.append(OpponentChoice("scripted"))
            else:
                recent = self._recent[kind]
                index = int(torch.randint(len(recent), (), generator=self._rng))
                out.append(OpponentChoice(kind, recent[index]))
        return out

    def model(self, choice: OpponentChoice, device: str | torch.device) -> torch.nn.Module:
        """Build a frozen RSL actor for a selected snapshot, caching per device."""
        if choice.kind == "scripted":
            raise ValueError("scripted opponents do not have an actor model")
        key = (choice.uid, str(device))
        if key not in self._model_cache:
            from rsl_rl.models import MLPModel

            snapshot = self._snapshots[choice.uid]
            obs = TensorDict({"actor": torch.zeros(1, 64, device=device)}, batch_size=[1])
            cfg = deepcopy(snapshot["actor_cfg"])
            cfg.pop("class_name", None)
            model = MLPModel(
                obs,
                {"actor": ["actor"]},
                "actor",
                4 if choice.kind == "tactical" else 14,
                **cfg,
            ).to(device)
            model.load_state_dict(snapshot["actor_state_dict"], strict=True)
            model.eval()
            model.requires_grad_(False)
            self._model_cache[key] = model
        return self._model_cache[key]

    def prune(self, active_uids: set[int]) -> None:
        """Discard snapshots no longer in the rolling pool or an active match."""
        keep = active_uids | set(self._recent["tactical"]) | set(self._recent["joint"])
        for uid in set(self._snapshots) - keep:
            del self._snapshots[uid]
        for key in list(self._model_cache):
            if key[0] not in keep:
                del self._model_cache[key]

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "next_uid": self._next_uid,
            "snapshots": deepcopy(self._snapshots),
            "recent": deepcopy(self._recent),
            "rng_state": self._rng.get_state().clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.capacity = int(state["capacity"])
        self._next_uid = int(state["next_uid"])
        self._snapshots = deepcopy(state["snapshots"])
        self._recent = deepcopy(state["recent"])
        self._rng.set_state(state["rng_state"].cpu())
        self._model_cache.clear()
