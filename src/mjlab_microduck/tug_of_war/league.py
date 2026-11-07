"""Permanent tug anchors and a resumable recent/older self-play league."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from math import isfinite
from pathlib import Path

import torch

from mjlab_microduck.football.pool import OpponentChoice, OpponentPool


DEFAULT_MIX = {
    "live": .30, "recent": .20, "archive": .15,
    "juicenv": .10, "andoshy": .10, "ctcheung": .05, "reference": .10,
}
EXTERNAL_NAMES = ("juicenv", "andoshy", "ctcheung")


class _ExternalActor(torch.nn.Module):
    def __init__(self, snapshot):
        super().__init__()
        from microduck_challenges.tug_of_war.opponent import Normalize

        state = snapshot["actor_state_dict"]
        layers = [Normalize(state["0.mean"], state["0.std"])]
        for index in range(1, 8, 2):
            weight = state[f"{index}.weight"]
            layers.append(torch.nn.Linear(weight.shape[1], weight.shape[0]))
            if index < 7:
                layers.append(torch.nn.ELU())
        self.actor = torch.nn.Sequential(*layers)
        self.actor.load_state_dict(state, strict=True)

    def forward(self, observations):
        return self.actor(observations["actor"])


class TugLeague(OpponentPool):
    """Choose one opponent per bout; fixed actors never enter the eviction queue.

    Anchor scores retain strong, diverse opponents by default. Challenger mode
    uses the current learner's held-out matchup counts against each snapshot.
    """

    def __init__(self, external_paths: dict[str, Path], *, seed: int = 1,
                 recent_capacity: int = 4, archive_capacity: int = 4,
                 weights: dict[str, float] | None = None):
        if recent_capacity < 1 or archive_capacity < 1:
            raise ValueError("recent and archive capacities must be positive")
        super().__init__(seed=seed, capacity=recent_capacity)
        self.archive_capacity = archive_capacity
        self.weights = self._validate_weights(DEFAULT_MIX if weights is None else weights)
        self.selection = "anchors"
        self.challenger_uniform_fraction = .2
        if set(external_paths) != set(EXTERNAL_NAMES):
            raise ValueError(f"external paths must name {EXTERNAL_NAMES}")
        self._archive: list[int] = []
        self._protected: list[int] = []
        self._external: dict[str, int] = {}
        from microduck_challenges.tug_of_war.opponent import load_actor

        for name in EXTERNAL_NAMES:
            path = Path(external_paths[name])
            actor = load_actor(path).eval().requires_grad_(False)
            uid = self._next_uid
            self._next_uid += 1
            self._snapshots[uid] = {
                "kind": "joint", "iteration": -1, "source": name,
                "path": str(path.resolve()), "sha256": sha256(path.read_bytes()).hexdigest(),
                "actor_state_dict": {
                    key: value.detach().cpu().clone() for key, value in actor.state_dict().items()
                },
            }
            self._external[name] = uid

    @staticmethod
    def _validate_weights(weights):
        if (not isinstance(weights, dict) or weights.keys() != DEFAULT_MIX.keys()
                or any(not isinstance(value, (int, float)) or not isfinite(value) or value < 0
                       for value in weights.values())
                or abs(sum(weights.values()) - 1) > 1e-7):
            raise ValueError("league weights must name every opponent group and sum to one")
        return dict(weights)

    @staticmethod
    def _validate_challenger_scores(scores):
        if (not isinstance(scores, dict) or set(scores) != {"wins", "losses", "invalid"}
                or any(type(value) is not int or value < 0 for value in scores.values())):
            raise ValueError("challenger scores require nonnegative integer wins, losses and invalid")
        return dict(scores)

    def configure(self, config: dict):
        """Apply explicit overrides; matchup counts describe the current learner.

        Scores keyed by frozen snapshot iteration replace the last probe, rather
        than mixing results from earlier learner policies into the current one.
        All overrides are validated before any changes are applied.
        """
        supported = {"weights", "selection", "challenger_uniform_fraction", "challenger_scores",
                     "archive_capacity", "protected_checkpoints", "replace_protected_checkpoints"}
        if not isinstance(config, dict) or set(config) - supported:
            raise ValueError("unsupported tug league configuration")
        weights = self._validate_weights(config.get("weights", self.weights))
        selection = config.get("selection", self.selection)
        if selection not in ("anchors", "challenger"):
            raise ValueError("league selection must be anchors or challenger")
        uniform = config.get("challenger_uniform_fraction", self.challenger_uniform_fraction)
        if not isinstance(uniform, (int, float)) or not isfinite(uniform) or not 0 <= uniform <= 1:
            raise ValueError("challenger uniform fraction must be finite and in [0, 1]")
        capacity = config.get("archive_capacity", self.archive_capacity)
        if type(capacity) is not int or capacity < 1:
            raise ValueError("archive_capacity must be a positive integer")
        replace = config.get("replace_protected_checkpoints", False)
        if type(replace) is not bool:
            raise ValueError("replace_protected_checkpoints must be a boolean")
        snapshots, protected, next_uid = self._prepare_protected(config.get("protected_checkpoints", {}),
                                                               replace=replace)
        if len(protected) > capacity:
            raise ValueError("archive_capacity cannot evict protected checkpoints")
        updates = {}
        scores = config.get("challenger_scores", {})
        if not isinstance(scores, dict):
            raise ValueError("challenger_scores must map snapshot iteration strings to counts")
        for iteration, counts in scores.items():
            matches = [uid for uid, snapshot in snapshots.items()
                       if not snapshot.get("source") and str(snapshot["iteration"]) == iteration]
            if not isinstance(iteration, str) or not matches:
                raise ValueError(f"unknown challenger snapshot iteration: {iteration}")
            if isinstance(counts, dict):
                counts = {"invalid": 0, **counts}
            counts = self._validate_challenger_scores(counts)
            updates.update({uid: counts for uid in matches})
        self.weights, self.selection, self.challenger_uniform_fraction = weights, selection, float(uniform)
        self.archive_capacity = capacity
        self._snapshots, self._protected, self._next_uid = snapshots, protected, next_uid
        for uid, counts in updates.items():
            self._snapshots[uid]["challenger_scores"] = counts.copy()
        self._archive = self._select_archive(self._archive)

    def _prepare_protected(self, checkpoints, *, replace=False):
        """Stage native actors without changing recent history or any RNG stream."""
        if not isinstance(checkpoints, dict):
            raise ValueError("protected_checkpoints must map iteration strings to checkpoint paths")
        snapshots = dict(self._snapshots)
        protected, next_uid = ([] if replace else list(self._protected)), self._next_uid
        native = [snapshot for snapshot in snapshots.values() if not snapshot.get("source")]
        template = max(native, key=lambda snapshot: snapshot["iteration"]) if native else None
        if checkpoints and template is None:
            raise ValueError("protected checkpoints require an existing native actor for compatibility validation")
        for iteration, filename in checkpoints.items():
            if (not isinstance(iteration, str) or not iteration.isdecimal()
                    or str(int(iteration)) != iteration or not isinstance(filename, (str, Path))):
                raise ValueError("protected checkpoints require canonical nonnegative iteration strings and paths")
            path = Path(filename)
            if not path.is_absolute() or not path.is_file():
                raise ValueError(f"protected checkpoint must be an existing absolute path: {path}")
            digest = sha256(path.read_bytes()).hexdigest()
            state = torch.load(path, map_location="cpu", weights_only=False)
            meta = state.get("tug_of_war", {})
            actor = state.get("actor_state_dict", {})
            cfg = meta.get("actor_cfg")
            expected = template["actor_state_dict"]
            if (meta.get("schema") != 1 or state.get("iter") != int(iteration)
                    or cfg != template["actor_cfg"] or not cfg.get("obs_normalization")
                    or actor.keys() != expected.keys()
                    or any(not isinstance(value, torch.Tensor) or value.shape != expected[key].shape
                           or value.dtype != expected[key].dtype or not torch.isfinite(value).all()
                           for key, value in actor.items())):
                raise ValueError(f"incompatible native protected checkpoint: {path}")
            norm = "obs_normalizer."
            if (any(norm + key not in actor for key in ("_mean", "_var", "_std", "count"))
                    or not (actor[norm + "_std"] > 0).all()
                    or not (actor[norm + "_var"] >= 0).all()
                    or not (actor[norm + "count"] >= 0).all()):
                raise ValueError(f"invalid protected checkpoint observation normalizer: {path}")
            # Reuse the existing pool's actor construction and strict loading.
            staged = OpponentPool()
            uid = staged.add_state("joint", actor, cfg, int(iteration))
            with torch.random.fork_rng(devices=[]):
                staged.model(OpponentChoice("joint", uid), "cpu")
            matches = [uid for uid, snapshot in snapshots.items()
                       if not snapshot.get("source") and snapshot["iteration"] == int(iteration)]
            for uid in matches:
                snapshot = snapshots[uid]
                if (snapshot["actor_cfg"] != cfg
                        or snapshot["actor_state_dict"].keys() != actor.keys()
                        or any(not torch.equal(snapshot["actor_state_dict"][key].detach().cpu(),
                                               value.detach().cpu())
                               for key, value in actor.items())
                        or snapshot.get("checkpoint_sha256", digest) != digest):
                    raise ValueError(f"protected checkpoint differs from retained snapshot: {iteration}")
            retained = [uid for uid in matches if uid in protected or uid not in self._recent["joint"]]
            if retained:
                uid = next((uid for uid in retained if uid in protected), retained[0])
                snapshot = dict(snapshots[uid])
            else:
                # A separate immutable archive identity leaves recent sampling,
                # active bout assignments and their labels unchanged.
                uid, next_uid = next_uid, next_uid + 1
                snapshot = staged._snapshots[0]
            snapshot.update(checkpoint_path=str(path.resolve()), checkpoint_sha256=digest)
            snapshots[uid] = snapshot
            if uid not in protected:
                protected.append(uid)
        return snapshots, protected, next_uid

    def set_challenger_scores(self, uid: int, wins: int, losses: int, invalid: int = 0):
        """Replace a frozen opponent's latest current-learner matchup probe."""
        counts = self._validate_challenger_scores({"wins": wins, "losses": losses, "invalid": invalid})
        if uid not in self._snapshots or uid in self._external.values():
            raise ValueError("challenger scores require a retained policy snapshot")
        self._snapshots[uid]["challenger_scores"] = counts
        self._archive = self._select_archive(self._archive)

    def _learner_win_rate(self, uid):
        counts = self._snapshots[uid].get("challenger_scores", {})
        valid = counts.get("wins", 0) + counts.get("losses", 0)
        return counts["wins"] / valid if valid else .5

    @property
    def history(self) -> list[dict]:
        """Retained sampling candidates and latest matchup data, safe to edit."""
        result = []
        for group, members in (("archive", self._archive), ("recent", self._recent["joint"])):
            for uid in members:
                counts = self._snapshots[uid].get("challenger_scores")
                total = sum(counts.values()) if counts else 0
                result.append({"uid": uid, "iteration": self._snapshots[uid]["iteration"],
                               "group": group, "protected": uid in self._protected,
                               "challenger_scores": deepcopy(counts),
                               "learner_win_rate": self._learner_win_rate(uid),
                               "invalid_rate": counts["invalid"] / total if total else 0.})
        return result

    @property
    def external_provenance(self) -> dict[str, dict[str, str]]:
        return {name: {key: self._snapshots[uid][key] for key in ("path", "sha256")}
                for name, uid in self._external.items()}

    @property
    def mixture(self) -> dict[str, float]:
        """Unavailable history's probability is shared among clean anchors."""
        result = self.weights.copy()
        missing = 0.
        for name, members in (("recent", self._recent["joint"]), ("archive", self._archive)):
            if not members:
                missing += result[name]
                result[name] = 0.
        anchor_mass = sum(self.weights[name] for name in EXTERNAL_NAMES)
        if missing and not anchor_mass:
            raise ValueError("missing history requires positive external opponent weights")
        if missing:
            for name in EXTERNAL_NAMES:
                result[name] += missing * self.weights[name] / anchor_mass
        return result

    def choose(self, count: int) -> list[OpponentChoice]:
        if count < 0:
            raise ValueError("opponent count must be nonnegative")
        labels = list(self.weights)
        probabilities = torch.tensor([self.mixture[name] for name in labels], dtype=torch.float64)
        groups = torch.multinomial(probabilities, count, replacement=True, generator=self._rng) if count else []
        choices = []
        for group in groups:
            name = labels[int(group)]
            if name in ("live", "reference"):
                choices.append(OpponentChoice("live" if name == "live" else "scripted"))
                continue
            if name in self._external:
                uid = self._external[name]
            else:
                members = self._recent["joint"] if name == "recent" else self._archive
                if name == "archive" and self.selection == "challenger":
                    priorities = torch.tensor([(1 - self._learner_win_rate(uid)) ** 2 for uid in members],
                                              dtype=torch.float64)
                    challenge = priorities / priorities.sum() if priorities.sum() else torch.full_like(priorities, 1 / len(members))
                    probabilities = (self.challenger_uniform_fraction / len(members)
                                     + (1 - self.challenger_uniform_fraction) * challenge)
                    index = int(torch.multinomial(probabilities, 1, generator=self._rng))
                else:
                    index = int(torch.randint(len(members), (), generator=self._rng))
                uid = members[index]
            choices.append(OpponentChoice("joint", uid))
        return choices

    def label(self, choice: OpponentChoice) -> str:
        if choice.kind == "live":
            return "live"
        if choice.kind == "scripted":
            return "reference"
        source = self._snapshots[choice.uid].get("source")
        if source:
            return source
        return "recent" if choice.uid in self._recent["joint"] else "archive"

    def model(self, choice, device):
        snapshot = self._snapshots.get(choice.uid, {})
        if not snapshot.get("source"):
            return super().model(choice, device)
        key = (choice.uid, str(device))
        if key not in self._model_cache:
            # Layer constructors initialize discarded weights; they must not
            # advance the learner's CPU random stream when a cache is rebuilt.
            with torch.random.fork_rng(devices=[]):
                model = _ExternalActor(snapshot)
            self._model_cache[key] = model.to(device).eval().requires_grad_(False)
        return self._model_cache[key]

    def add_state(self, kind, actor_state_dict, actor_cfg, iteration, skill_state_dicts=None):
        if kind != "joint":
            raise ValueError("tug supports only joint policy snapshots")
        previous = list(self._recent["joint"])
        uid = super().add_state(kind, actor_state_dict, actor_cfg, iteration, skill_state_dicts)
        retired = [old for old in previous if old not in self._recent["joint"]]
        self._archive = self._select_archive(self._archive + retired)
        return uid

    def add_snapshot(self, actor, actor_cfg, iteration, scores_by_opponent=None):
        uid = self.add("joint", actor, actor_cfg, iteration)
        if scores_by_opponent is not None:
            self.set_snapshot_scores(uid, scores_by_opponent)
        return uid

    def set_snapshot_scores(self, uid: int, scores_by_opponent: dict[str, float]):
        if (not scores_by_opponent or any(not isfinite(value) or not 0 <= value <= 1
                                          for value in scores_by_opponent.values())):
            raise ValueError("snapshot scores must be finite held-out win rates in [0, 1]")
        if uid in self._external.values():
            raise ValueError("permanent external opponents are not archive candidates")
        self._snapshots[uid]["evaluation_scores"] = dict(scores_by_opponent)
        self._archive = self._select_archive(self._archive)

    def _select_archive(self, candidates):
        protected = list(self._protected)
        candidates = [uid for uid in dict.fromkeys(candidates) if uid not in protected]
        capacity = self.archive_capacity - len(protected)
        if capacity < 0:
            raise ValueError("archive_capacity cannot evict protected checkpoints")
        return protected + (self._select_unprotected(candidates, capacity) if capacity else [])

    def _select_unprotected(self, candidates, capacity):
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) <= capacity:
            return candidates
        if self.selection == "challenger":
            return self._select_challengers(candidates, capacity)
        scored = [uid for uid in candidates if self._snapshots[uid].get("evaluation_scores")]

        def strength(uid):
            values = self._snapshots[uid]["evaluation_scores"].values()
            return sum(values) / len(values)

        def distance(first, second):
            left = self._snapshots[first]["evaluation_scores"]
            right = self._snapshots[second]["evaluation_scores"]
            common = left.keys() & right.keys()
            return sum((left[key] - right[key]) ** 2 for key in common) / len(common) if common else 0.

        chosen = sorted(scored, key=lambda uid: (strength(uid), uid), reverse=True)[:max(1, capacity // 2)]
        while len(chosen) < min(len(scored), capacity):
            remaining = set(scored) - set(chosen)
            chosen.append(max(remaining, key=lambda uid: (min(distance(uid, old) for old in chosen), strength(uid), uid)))
        remaining = set(candidates) - set(chosen)
        # Without comparable evaluations, keep the oldest and then spread ages.
        if not chosen:
            chosen.append(min(remaining))
            remaining.remove(chosen[0])
        while len(chosen) < capacity:
            uid = max(remaining, key=lambda candidate: (min(abs(candidate - old) for old in chosen), candidate))
            chosen.append(uid)
            remaining.remove(uid)
        return sorted(chosen)

    def _select_challengers(self, candidates, capacity):
        # Protect measured competitive matchups first, then preserve temporal
        # and matchup diversity. Anchor saturation is irrelevant in this mode.
        measured = [uid for uid in candidates
                    if sum(self._snapshots[uid].get("challenger_scores", {}).get(key, 0)
                           for key in ("wins", "losses"))]

        def relevance(uid):
            rate = self._learner_win_rate(uid)
            return (1 - rate) * (.5 + 2 * rate)

        chosen = sorted(measured, key=lambda uid: (relevance(uid), uid), reverse=True)[
            :max(1, capacity // 2)]
        if not chosen:
            chosen = [min(candidates, key=lambda uid: self._snapshots[uid]["iteration"])]
        iterations = [self._snapshots[uid]["iteration"] for uid in candidates]
        span = max(1, max(iterations) - min(iterations))
        while len(chosen) < capacity:
            def diversity(uid):
                temporal = min(abs(self._snapshots[uid]["iteration"]
                                   - self._snapshots[old]["iteration"]) / span for old in chosen)
                score = min(abs(self._learner_win_rate(uid) - self._learner_win_rate(old))
                            for old in chosen)
                return (temporal + score, uid in measured, relevance(uid), uid)

            chosen.append(max(set(candidates) - set(chosen), key=diversity))
        return sorted(chosen)

    def prune(self, active_uids):
        super().prune(set(active_uids) | set(self._external.values()) | set(self._archive))

    def state_dict(self):
        return {**super().state_dict(), "league_version": 1,
                "archive_capacity": self.archive_capacity, "weights": self.weights.copy(),
                "selection": self.selection, "challenger_uniform_fraction": self.challenger_uniform_fraction,
                "external": self._external.copy(), "archive": self._archive.copy(),
                "protected": self._protected.copy()}

    @classmethod
    def from_state(cls, state):
        """Restore embedded actors without requiring their original ONNX files."""
        league = cls.__new__(cls)
        OpponentPool.__init__(league)
        league.load_state_dict(state)
        return league

    def load_state_dict(self, state):
        if state.get("league_version") != 1:
            raise ValueError("checkpoint does not contain a compatible tug league")
        super().load_state_dict(state)
        self.archive_capacity = int(state["archive_capacity"])
        self.weights = deepcopy(state["weights"])
        self._external = deepcopy(state["external"])
        self._archive = list(state["archive"])
        self._protected = list(state.get("protected", []))
        if (len(set(self._protected)) != len(self._protected)
                or any(uid not in self._snapshots or self._snapshots[uid].get("source")
                       for uid in self._protected)):
            raise ValueError("invalid protected checkpoint identities")
        self.selection = "anchors"
        self.challenger_uniform_fraction = .2
        self.configure({key: state[key] for key in ("selection", "challenger_uniform_fraction") if key in state})
