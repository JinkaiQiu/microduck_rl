"""Keyboard-controlled tactical Microduck against a frozen football policy.

The MuJoCo viewer displays the match; keys are read from the launching terminal.
That mirrors scripts/infer_policy.py and avoids MuJoCo viewer shortcut clashes.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import select
import sys
import termios
import tty

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.viewer import NativeMujocoViewer

from mjlab_microduck.football.arena import FootballArena, make_football_env_cfg
from mjlab_microduck.football.cli import SKILL_FILES
from mjlab_microduck.football.eval import _read_checkpoint, _skill_paths
from mjlab_microduck.football.pool import OpponentChoice, OpponentPool
from mjlab_microduck.football.skills import FrozenSkillPolicy
from mjlab_microduck.football.wrapper import FootballVecEnv


class TerminalKeys:
    """Read single terminal keypresses without blocking the 50 Hz viewer loop."""

    _ARROWS = {"A": "up", "B": "down", "C": "right", "D": "left"}

    def __init__(self) -> None:
        self.fd: int | None = None
        self.old_attrs = None
        self.buffer = b""

    def __enter__(self) -> "TerminalKeys":
        if not sys.stdin.isatty():
            raise RuntimeError("keyboard playground needs a terminal (TTY); use --headless --steps N for a smoke run")
        self.fd = sys.stdin.fileno()
        self.old_attrs = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.fd is not None and self.old_attrs is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_attrs)

    def poll(self) -> list[str]:
        if self.fd is None:
            return []
        while select.select([self.fd], [], [], 0)[0]:
            chunk = os.read(self.fd, 64)
            if not chunk:
                break
            self.buffer += chunk
        keys: list[str] = []
        while self.buffer:
            if self.buffer[0] == 27:
                if len(self.buffer) < 3:
                    break
                if self.buffer[1:2] == b"[":
                    arrow = self._ARROWS.get(chr(self.buffer[2]))
                    if arrow:
                        keys.append(arrow)
                    self.buffer = self.buffer[3:]
                else:
                    self.buffer = self.buffer[1:]
            else:
                char = chr(self.buffer[0])
                keys.append(char.lower() if char.isalpha() else char)
                self.buffer = self.buffer[1:]
        return keys


class TacticalKeyboard:
    """Persistent walking command and a one-shot, 2-second kick request."""

    def __init__(self) -> None:
        self.vx = 0.0
        self.vy = 0.0
        self.yaw = 0.0
        self.kick_queue_steps = 0
        self.quit_requested = False

    def reset(self) -> None:
        self.vx = self.vy = self.yaw = 0.0
        self.kick_queue_steps = 0

    def handle(self, key: str) -> bool:
        """Apply one keypress; return whether the visible command changed."""
        previous = (self.vx, self.vy, self.yaw, self.kick_queue_steps, self.quit_requested)
        if key == "up":
            self.vx = min(0.4, round(self.vx + 0.1, 3))
        elif key == "down":
            self.vx = max(-0.4, round(self.vx - 0.1, 3))
        elif key == "left":
            self.vy = min(0.3, round(self.vy + 0.1, 3))
        elif key == "right":
            self.vy = max(-0.3, round(self.vy - 0.1, 3))
        elif key == "a":
            self.yaw = min(1.0, round(self.yaw + 0.25, 3))
        elif key == "e":
            self.yaw = max(-1.0, round(self.yaw - 0.25, 3))
        elif key == " ":
            self.vx = self.vy = self.yaw = 0.0
            self.kick_queue_steps = 0
        elif key == "l":
            self.kick_queue_steps = 100
        elif key == "q":
            self.quit_requested = True
        return previous != (self.vx, self.vy, self.yaw, self.kick_queue_steps, self.quit_requested)

    def action(self, device: torch.device) -> torch.Tensor:
        kick = 1.0 if self.kick_queue_steps > 0 else -1.0
        if self.kick_queue_steps:
            self.kick_queue_steps -= 1
        return torch.tensor([[self.vx, self.vy, self.yaw, kick]], device=device)

    def kick_started(self) -> None:
        self.kick_queue_steps = 0

    def status(self) -> str:
        return f"vx={self.vx:+.1f} vy={self.vy:+.1f} yaw={self.yaw:+.2f} kick={'queued' if self.kick_queue_steps else 'idle'}"


class FootballPlayEnv(FootballVecEnv):
    """Pin a frozen opponent and optional learner seat across match resets."""

    def __init__(self, arena: FootballArena, pool: OpponentPool, skills: tuple,
                 opponent: OpponentChoice, side: str, seed: int) -> None:
        self.fixed_opponent = opponent
        self.fixed_side = side
        super().__init__(arena, "tactics", pool, skills, seed=seed)

    def _choose_matches(self, ids: torch.Tensor) -> None:
        super()._choose_matches(ids)
        for env_id in ids.tolist():
            self.opponent_choices[env_id] = self.fixed_opponent
        if self.fixed_side != "random":
            self.learner_home[ids] = self.fixed_side == "home"
        if (ids == 0).any():
            self.refresh_appearance()

    def _score(self) -> tuple[int, int]:
        values = self.arena.scores[0].tolist()
        return tuple(values if self.learner_home[0] else values[::-1])

    def announce_match(self) -> None:
        seat = "home" if self.learner_home[0] else "away"
        side = 0 if self.learner_home[0] else 1
        direction = "east (+x)" if self.arena.attack_sign[0, side] > 0 else "west (-x)"
        print(f"New match: you are {seat}; attack {direction}. Score 0–0.", flush=True)

    def reset(self):
        result = super().reset()
        self.announce_match()
        return result

    def step(self, actions: torch.Tensor):
        previous = self.arena.scores[0].clone()
        result = super().step(actions)
        completed = result[3]["completed"]
        if completed["env_ids"].numel():
            ours = int(completed["own_goals"][0].item())
            theirs = int(completed["opponent_goals"][0].item())
            points = float(completed["points"][0].item())
            outcome = "win" if points == 1 else "draw" if points == 0.5 else "loss"
            own_out = bool(completed["own_out_of_bounds"][0].item())
            other_out = bool(completed["opponent_out_of_bounds"][0].item())
            reason = " (both ducks out of bounds)" if own_out and other_out else (
                " (you left the field)" if own_out else
                " (opponent left the field)" if other_out else "")
            print(f"Match over: {ours}–{theirs} ({outcome}){reason}.", flush=True)
            self.announce_match()
        elif not torch.equal(previous, self.arena.scores[0]):
            ours, theirs = self._score()
            print(f"Goal! Score {ours}–{theirs}.", flush=True)
        return result


class KeyboardPolicy:
    def __init__(self, env: FootballPlayEnv, controls: TacticalKeyboard,
                 terminal: TerminalKeys | None) -> None:
        self.env = env
        self.controls = controls
        self.terminal = terminal

    def reset(self) -> None:
        self.controls.reset()

    def __call__(self, _obs) -> torch.Tensor:
        if self.terminal is not None:
            for key in self.terminal.poll():
                if self.controls.handle(key):
                    if key != "q":
                        print(self.controls.status(), flush=True)
            if self.controls.kick_queue_steps and self.env.learner_skills.kick_remaining[0] > 0:
                self.controls.kick_started()
                print("Right-foot kick started.", flush=True)
        return self.controls.action(self.env.device)


class FootballViewer(NativeMujocoViewer):
    def __init__(self, env: FootballPlayEnv, policy: KeyboardPolicy,
                 controls: TacticalKeyboard) -> None:
        self.controls = controls
        super().__init__(env, policy, frame_rate=30.0, enable_perturbations=False)

    def is_running(self) -> bool:
        return not self.controls.quit_requested and super().is_running()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play tactical football against a learned Microduck")
    parser.add_argument("--opponent-checkpoint", type=Path, required=True)
    parser.add_argument("--side", choices=("home", "away", "random"), default="home")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, help="stop after this many 50 Hz control steps")
    parser.add_argument("--headless", action="store_true", help="run without a viewer or keyboard (for smoke tests)")
    return parser


def run(args: argparse.Namespace) -> None:
    if args.steps is not None and args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.headless and args.steps is None:
        raise ValueError("--headless requires --steps")
    if not args.headless and not sys.stdin.isatty():
        raise RuntimeError("interactive football needs a terminal (TTY); use --headless --steps N for a smoke run")
    torch.manual_seed(args.seed)
    checkpoint = args.opponent_checkpoint.resolve()
    state, meta = _read_checkpoint(checkpoint)
    paths = _skill_paths(checkpoint, meta)
    device = torch.device(args.device)
    skills = tuple(FrozenSkillPolicy.from_checkpoint(paths[name], device) for name in SKILL_FILES)
    kind = "tactical" if meta["stage"] == "tactics" else "joint"
    pool = OpponentPool(seed=args.seed)
    uid = pool.add_state(kind, state["actor_state_dict"], meta["actor_cfg"], state["iter"])
    opponent = OpponentChoice(kind, uid)
    print(f"Opponent: {meta['stage']} policy from {checkpoint}", flush=True)
    base = ManagerBasedRlEnv(cfg=make_football_env_cfg(1, seed=args.seed), device=str(device))
    env = FootballPlayEnv(FootballArena(base), pool, skills, opponent, args.side, args.seed)
    controls = TacticalKeyboard()
    try:
        if args.headless:
            for _ in range(args.steps):
                env.step(controls.action(device))
            return
        print("Keys in this terminal: ↑/↓ forward/back, ←/→ strafe, A/E turn, Space stop, L queue right-foot kick, Q quit.")
        print("Each movement keypress changes the command; hold for key repeat. Keep this terminal focused while watching the viewer.")
        with TerminalKeys() as terminal:
            policy = KeyboardPolicy(env, controls, terminal)
            viewer = FootballViewer(env, policy, controls)
            viewer.run(num_steps=args.steps)
    finally:
        env.close()


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
