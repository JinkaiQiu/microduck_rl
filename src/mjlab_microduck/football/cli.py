"""Train tactical, distilled, and joint self-play football policies.

The standard mjlab `train` entry point cannot route two independently
controlled robots through one RSL learner.  This entry point constructs the
football VecEnv and its frozen-opponent pool explicitly.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import uuid

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlOnPolicyRunnerCfg

from mjlab_microduck.football.arena import FootballArena, make_football_env_cfg
from mjlab_microduck.football.learning import warm_start_joint_actor_from_walk
from mjlab_microduck.football.pool import OpponentChoice, OpponentPool
from mjlab_microduck.football.runner import FootballRunner
from mjlab_microduck.football.skills import FrozenSkillPolicy
from mjlab_microduck.football.video_trace import FootballVideoRecorder
from mjlab_microduck.football.wrapper import FootballVecEnv


ACTOR_CFG = {
    "class_name": "MLPModel",
    "hidden_dims": (512, 256, 128),
    "activation": "elu",
    "obs_normalization": True,
    "distribution_cfg": {
        "class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"
    },
}
CRITIC_CFG = {
    "class_name": "MLPModel",
    "hidden_dims": (512, 256, 128),
    "activation": "elu",
    "obs_normalization": True,
}
SKILL_FILES = {"walk": "walk.pt", "kick": "kick.pt", "recovery": "recovery.pt"}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_skill_bundle(paths: dict[str, Path], run_dir: Path, expected_hashes: dict[str, str] | None = None) -> dict[str, str]:
    destination = run_dir / "skills"
    destination.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, filename in SKILL_FILES.items():
        source = paths[name].resolve()
        if not source.is_file():
            raise FileNotFoundError(f"required {name} policy checkpoint: {source}")
        target = destination / filename
        if source != target.resolve():
            shutil.copy2(source, target)
        hashes[name] = file_hash(target)
        if expected_hashes and hashes[name] != expected_hashes[name]:
            raise ValueError(f"{name} skill checkpoint hash differs from the source football run")
    return hashes


def bundle_paths(run_dir: Path) -> dict[str, Path]:
    return {name: run_dir / "skills" / filename for name, filename in SKILL_FILES.items()}


def train_config(stage: str, iterations: int, seed: int, logger: str) -> dict:
    config = asdict(RslRlOnPolicyRunnerCfg())
    config.update({
        "seed": seed,
        "num_steps_per_env": 24,
        "max_iterations": iterations,
        "save_interval": 200,
        "experiment_name": f"football_{stage}",
        "run_name": f"seed-{seed}",
        "logger": logger,
        "wandb_project": "mjlab_microduck",
        "obs_groups": {"actor": ("actor",), "critic": ("critic",)},
        "actor": deepcopy(ACTOR_CFG),
        "critic": deepcopy(CRITIC_CFG),
    })
    config["algorithm"].update({
        "class_name": "mjlab_microduck.football.learning.FootballPPO",
        "teacher_loss_coef_start": 1.0 if stage == "distill" else 0.0,
        "teacher_loss_coef_end": 0.1 if stage == "distill" else 0.0,
        "teacher_decay_iterations": 2000,
        "symmetry_cfg": None,
        "rnd_cfg": None,
    })
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Microduck 1v1 football training")
    subcommands = parser.add_subparsers(dest="stage", required=True)
    for stage in ("tactics", "distill", "joint"):
        command = subcommands.add_parser(stage)
        command.add_argument("--num-envs", type=int, default=1024)
        command.add_argument("--iterations", type=int, default=2000 if stage == "distill" else 6000)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
        command.add_argument("--logger", choices=("tensorboard", "wandb"), default="tensorboard")
        command.add_argument("--video-every", type=int, default=200, metavar="ITERATIONS",
                             help="record a local match clip every N PPO iterations; 0 disables (default: 200)")
        command.add_argument("--debug-numerics", action="store_true",
                             help="fail at the first nonfinite physics value and save simulator state history")
        command.add_argument("--video-steps", type=int, default=300, metavar="STEPS",
                             help="50 Hz frames per training clip (default: 300, or 6 seconds)")
        command.add_argument("--run-name")
        command.add_argument("--resume", type=Path, help="resume this stage from a football checkpoint")
        if stage == "tactics":
            command.add_argument("--walk-checkpoint", type=Path)
            command.add_argument("--kick-checkpoint", type=Path)
            command.add_argument("--recovery-checkpoint", type=Path)
        elif stage == "distill":
            command.add_argument("--teacher-checkpoint", type=Path)
        else:
            command.add_argument("--init-checkpoint", type=Path)
    return parser


def _source_checkpoint(stage: str, args, run_dir: Path) -> Path | None:
    if args.resume:
        copied = run_dir / ("teacher.pt" if stage == "distill" else "student_init.pt")
        return copied if copied.exists() else None
    path = args.teacher_checkpoint if stage == "distill" else args.init_checkpoint if stage == "joint" else None
    if stage != "tactics" and path is None:
        raise ValueError(f"{stage} requires a source checkpoint")
    if path is None:
        return None
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    copied = run_dir / ("teacher.pt" if stage == "distill" else "student_init.pt")
    shutil.copy2(path, copied)
    return copied


def run(args) -> Path:
    if args.num_envs < 1 or args.iterations < 1:
        raise ValueError("num-envs and iterations must both be positive")
    if args.video_every < 0 or args.video_steps < 1:
        raise ValueError("video-every must be nonnegative and video-steps must be positive")
    if args.video_every and args.video_steps > args.video_every * 24:
        raise ValueError("video-steps cannot exceed the recording interval (video-every * 24)")
    stage = args.stage
    if args.resume:
        resume_path = args.resume.resolve()
        run_dir = resume_path.parent
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
    else:
        label = args.run_name or f"seed-{args.seed}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
        run_dir = Path("logs") / "football" / stage / label
        if run_dir.exists():
            raise FileExistsError(f"football run already exists: {run_dir}")
        run_dir.mkdir(parents=True)
        resume_path = None

    original_source = (args.teacher_checkpoint if stage == "distill" else args.init_checkpoint if stage == "joint" else None)
    original_source = original_source.resolve() if original_source is not None else None
    source = _source_checkpoint(stage, args, run_dir)
    source_state = torch.load(source, map_location="cpu", weights_only=False) if source else None
    source_meta = source_state.get("football") if source_state else None
    if source and (not source_meta or (stage == "distill" and source_meta["stage"] != "tactics")
                   or (stage == "joint" and source_meta["stage"] != "distill")):
        raise ValueError("source checkpoint is from the wrong football stage")
    if resume_path:
        resume_meta = torch.load(resume_path, map_location="cpu", weights_only=False)["football"]
        if resume_meta["stage"] != stage:
            raise ValueError("cannot resume a different football stage")
        skill_hashes = {name: file_hash(path) for name, path in bundle_paths(run_dir).items()}
        if skill_hashes != resume_meta["skill_hashes"]:
            raise ValueError("skill bundle differs from resumed checkpoint")
    elif stage == "tactics":
        paths = {
            "walk": args.walk_checkpoint, "kick": args.kick_checkpoint,
            "recovery": args.recovery_checkpoint,
        }
        if any(path is None for path in paths.values()):
            raise ValueError("tactics requires --walk-checkpoint, --kick-checkpoint, and --recovery-checkpoint")
        skill_hashes = copy_skill_bundle(paths, run_dir)
    else:
        skill_hashes = copy_skill_bundle(bundle_paths(original_source.parent), run_dir, source_meta["skill_hashes"])

    device = torch.device(args.device)
    skill_paths = bundle_paths(run_dir)
    skills = tuple(FrozenSkillPolicy.from_checkpoint(skill_paths[name], device) for name in SKILL_FILES)
    pool = OpponentPool(seed=args.seed)
    teacher_model = None
    if source_state:
        pool.load_state_dict(source_meta["env_state"]["pool"])
        if stage == "distill":
            uid = pool.add_state("tactical", source_state["actor_state_dict"], source_meta["actor_cfg"], source_state["iter"])
            teacher_model = pool.model(OpponentChoice("tactical", uid), device)
    config = train_config(stage, args.iterations, args.seed, args.logger)
    env_cfg = make_football_env_cfg(args.num_envs, seed=args.seed)
    if args.debug_numerics:
        env_cfg.sim.nan_guard.enabled = True
        env_cfg.sim.nan_guard.buffer_size = 20
        env_cfg.sim.nan_guard.output_dir = str((run_dir / "nan_dumps").resolve())
    base_env = ManagerBasedRlEnv(
        cfg=env_cfg, device=str(device),
        render_mode="rgb_array" if args.video_every else None,
    )
    recorder = None
    video_end_step = {"value": args.iterations * config["num_steps_per_env"]}
    if args.video_every:
        period = args.video_every * config["num_steps_per_env"]

        def start_video(step: int) -> bool:
            if step % period:
                return False
            # A final short run still finishes its clip before the logger's
            # last iteration, so W&B sees and uploads the MP4.
            recorder.video_length = min(args.video_steps, video_end_step["value"] - step)
            return True

        recorder = FootballVideoRecorder(
            base_env,
            video_folder=run_dir / "videos" / "train",
            step_trigger=start_video,
            video_length=args.video_steps,
            name_prefix=f"football-{stage}",
        )
    game_env = recorder if recorder is not None else base_env
    wrapped = FootballVecEnv(
        FootballArena(game_env, debug_numerics=args.debug_numerics),
        stage, pool, skills, teacher_model, seed=args.seed,
    )
    if recorder is not None:
        recorder.football_env = wrapped
    runner = FootballRunner(
        wrapped, config, str(run_dir), str(device),
        ACTOR_CFG, skill_hashes, str(source) if source else None,
    )
    try:
        if resume_path:
            runner.load(str(resume_path))
            if recorder is not None:
                recorder.step_count = runner.current_learning_iteration * config["num_steps_per_env"]
                video_end_step["value"] = recorder.step_count + args.iterations * config["num_steps_per_env"]
                recorder.name_prefix = f"football-{stage}-resume-{uuid.uuid4().hex[:8]}"
        elif stage == "distill":
            warm_start_joint_actor_from_walk(runner.alg.actor, skill_paths["walk"])
        elif stage == "joint":
            runner.alg.actor.load_state_dict(source_state["actor_state_dict"], strict=True)
        runner.learn(args.iterations, init_at_random_ep_len=False)
    finally:
        wrapped.close()
    return run_dir


def main() -> int:
    args = build_parser().parse_args()
    run_dir = run(args)
    print(f"Football {args.stage} run saved in {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
