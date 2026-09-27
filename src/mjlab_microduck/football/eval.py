"""Fixed-seed, side-balanced evaluation for football checkpoints."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.wrappers import VideoRecorder

from mjlab_microduck.football.arena import FootballArena, MATCH_STEPS, make_football_env_cfg
from mjlab_microduck.football.cli import SKILL_FILES, bundle_paths, file_hash
from mjlab_microduck.football.pool import OpponentChoice, OpponentPool
from mjlab_microduck.football.skills import FrozenSkillPolicy
from mjlab_microduck.football.wrapper import FootballVecEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Microduck football checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--opponent-checkpoint", type=Path,
                        help="frozen tactical or joint opponent; default is scripted chase-and-kick")
    parser.add_argument("--matches", type=int, default=400,
                        help="even number of matches, balanced across attacking directions")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--video", type=Path, metavar="DIR",
                        help="record the first short match clip in this directory")
    parser.add_argument("--video-steps", type=int, default=300,
                        help="number of 50 Hz frames to record, default 6 seconds")
    parser.add_argument("--json", type=Path, metavar="FILE", help="also save the summary as JSON")
    return parser


def _read_checkpoint(path: Path) -> tuple[dict, dict]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    meta = state.get("football")
    if not isinstance(meta, dict) or meta.get("stage") not in {"tactics", "distill", "joint"}:
        raise ValueError(f"not a football checkpoint: {path}")
    if "actor_state_dict" not in state or "actor_cfg" not in meta:
        raise ValueError(f"football actor or configuration is missing: {path}")
    return state, meta


def _skill_paths(checkpoint: Path, meta: dict) -> dict[str, Path]:
    paths = bundle_paths(checkpoint.resolve().parent)
    for name in SKILL_FILES:
        path = paths[name]
        if not path.is_file():
            raise FileNotFoundError(f"{name} skill bundle missing beside {checkpoint}: {path}")
        if file_hash(path) != meta["skill_hashes"][name]:
            raise ValueError(f"{name} skill hash differs from {checkpoint}")
    return paths


def _slot_schedule(slot: int) -> tuple[bool, float]:
    """Cycle through both learner seats and both attack directions."""
    phase = slot % 4
    return (phase in (0, 2), 1.0 if phase in (0, 3) else -1.0)


def _summary(records: list[dict], checkpoint: Path, opponent: Path | None, seed: int) -> dict:
    count = len(records)
    points = sum(row["points"] for row in records)
    goals_for = sum(row["goals_for"] for row in records)
    goals_against = sum(row["goals_against"] for row in records)
    results = Counter("win" if row["points"] == 1 else "draw" if row["points"] == 0.5 else "loss"
                      for row in records)
    splits = {}
    for label, predicate in (
        ("attack_east", lambda row: row["attack_sign"] > 0),
        ("attack_west", lambda row: row["attack_sign"] < 0),
        ("home", lambda row: row["learner_home"]),
        ("away", lambda row: not row["learner_home"]),
    ):
        subset = [row for row in records if predicate(row)]
        splits[label] = {
            "matches": len(subset),
            "match_points": sum(row["points"] for row in subset) / len(subset) if subset else None,
            "goals_for": sum(row["goals_for"] for row in subset),
            "goals_against": sum(row["goals_against"] for row in subset),
            "learner_forfeits": sum(row.get("learner_out_of_bounds", False) for row in subset),
            "opponent_forfeits": sum(row.get("opponent_out_of_bounds", False) for row in subset),
        }
    return {
        "checkpoint": str(checkpoint.resolve()),
        "opponent": str(opponent.resolve()) if opponent else "scripted",
        "seed": seed,
        "matches": count,
        "match_points": points / count,
        "wins": results["win"],
        "draws": results["draw"],
        "losses": results["loss"],
        "goals_for": goals_for,
        "goals_against": goals_against,
        "learner_forfeits": sum(row.get("learner_out_of_bounds", False) for row in records),
        "opponent_forfeits": sum(row.get("opponent_out_of_bounds", False) for row in records),
        "splits": splits,
    }


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    if args.matches < 2 or args.matches % 2:
        raise ValueError("matches must be positive and even for side balance")
    if args.num_envs < 1 or args.video_steps < 1:
        raise ValueError("num-envs and video-steps must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    checkpoint = args.checkpoint.resolve()
    state, meta = _read_checkpoint(checkpoint)
    skill_paths = _skill_paths(checkpoint, meta)
    skills = tuple(FrozenSkillPolicy.from_checkpoint(skill_paths[name], device) for name in SKILL_FILES)
    pool = OpponentPool(seed=args.seed)
    learner_kind = "tactical" if meta["stage"] == "tactics" else "joint"
    learner_uid = pool.add_state(learner_kind, state["actor_state_dict"], meta["actor_cfg"], state["iter"])
    learner = pool.model(OpponentChoice(learner_kind, learner_uid), device)
    opponent_path = args.opponent_checkpoint.resolve() if args.opponent_checkpoint else None
    opponent_choice = OpponentChoice("scripted")
    if opponent_path:
        opponent_state, opponent_meta = _read_checkpoint(opponent_path)
        opponent_skills = _skill_paths(opponent_path, opponent_meta)
        if any(file_hash(opponent_skills[name]) != file_hash(skill_paths[name]) for name in SKILL_FILES):
            raise ValueError("opponent uses a different skill bundle")
        opponent_kind = "tactical" if opponent_meta["stage"] == "tactics" else "joint"
        uid = pool.add_state(opponent_kind, opponent_state["actor_state_dict"],
                             opponent_meta["actor_cfg"], opponent_state["iter"])
        opponent_choice = OpponentChoice(opponent_kind, uid)
    # A distilled student already acts in joint space; it needs no teacher during evaluation.
    stage = "tactics" if learner_kind == "tactical" else "joint"
    num_envs = min(args.num_envs, args.matches)
    base = ManagerBasedRlEnv(
        cfg=make_football_env_cfg(num_envs, seed=args.seed), device=str(device),
        render_mode="rgb_array" if args.video else None,
    )
    if args.video:
        base = VideoRecorder(base, video_folder=args.video, step_trigger=lambda step: step == 0,
                             video_length=args.video_steps, name_prefix="football-eval",
                             disable_logger=True)
    wrapped = FootballVecEnv(FootballArena(base), stage, pool, skills, seed=args.seed)
    pending = [-1] * num_envs
    next_slot = 0
    records: list[dict | None] = [None] * args.matches

    def assign(ids: list[int]) -> None:
        nonlocal next_slot
        assigned: list[int] = []
        for env_id in ids:
            if next_slot >= args.matches:
                pending[env_id] = -1
                continue
            slot = next_slot
            next_slot += 1
            pending[env_id] = slot
            learner_home, attack_sign = _slot_schedule(slot)
            wrapped.learner_home[env_id] = learner_home
            wrapped.opponent_choices[env_id] = opponent_choice
            wrapped.arena.attack_sign[env_id, 0] = attack_sign
            wrapped.arena.attack_sign[env_id, 1] = -attack_sign
            assigned.append(env_id)
        if assigned:
            if 0 in assigned:
                wrapped.refresh_appearance()
            ids_tensor = torch.as_tensor(assigned, device=device, dtype=torch.long)
            # Seat and direction are fixed before kickoff, including physical poses.
            wrapped.arena._kickoff(ids_tensor, new_match=False)
            wrapped.arena._refresh_actor_cache()

    try:
        assign(list(range(num_envs)))
        obs = wrapped.get_observations()
        max_steps = MATCH_STEPS * (math.ceil(args.matches / num_envs) + 1)
        for _ in range(max_steps):
            action = learner(obs)
            obs, _, _, extras = wrapped.step(action)
            completed = extras["completed"]
            done_ids = completed["env_ids"].detach().cpu().tolist()
            for row, env_id in enumerate(done_ids):
                slot = pending[env_id]
                if slot < 0:
                    continue
                learner_home, attack_sign = _slot_schedule(slot)
                records[slot] = {
                    "points": float(completed["points"][row].item()),
                    "goals_for": int(completed["own_goals"][row].item()),
                    "goals_against": int(completed["opponent_goals"][row].item()),
                    "learner_out_of_bounds": bool(completed["own_out_of_bounds"][row].item()),
                    "opponent_out_of_bounds": bool(completed["opponent_out_of_bounds"][row].item()),
                    "learner_home": learner_home,
                    "attack_sign": attack_sign,
                }
            if all(record is not None for record in records):
                break
            assign(done_ids)
            if done_ids:
                obs = wrapped.get_observations()
        else:
            raise RuntimeError("evaluation did not complete all matches within its step bound")
    finally:
        wrapped.close()
    return _summary(records, checkpoint, opponent_path, args.seed)


def main() -> int:
    args = build_parser().parse_args()
    result = evaluate(args)
    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(output + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
