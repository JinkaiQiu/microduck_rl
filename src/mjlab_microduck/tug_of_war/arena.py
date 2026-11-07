"""The reference tug physics with symmetric control and official rating rewards.

Each physical world contains both ducks. Actions are west then east, 14 joint
offsets each; observations have the Arena's 61 inputs. A completed world must
be reset explicitly, allowing collectors to retain the terminal result first.
"""

from __future__ import annotations

from dataclasses import replace
import math

import mujoco
import torch
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg

from microduck_challenges.tug_of_war import env as reference

JOINT_NAMES = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll", "right_hip_yaw",
    "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
SIDES = ("robot", "opponent")
TORN_M = 0.05
FAULT_SCORE = -2.0  # Arena FAULT_PITS * the one-metre pit bonus.


def _dummy_observation(env):
    return torch.zeros((env.num_envs, 1), device=env.device)


def _capture_reset_events(env, env_ids):
    recorder = getattr(env, "_tug_reset_diagnostics", None)
    if recorder is not None:
        recorder.after_events = recorder.capture(env_ids)


def _randomize_training_reset(env, env_ids):
    randomization = getattr(env, "_tug_randomization", None)
    if randomization is not None:
        randomization.reset(env_ids)
        if (randomization.condition[env_ids] >= 2).any():
            # Validate candidate kinematics before the reset's BAM drive. A
            # rejected candidate must never leave actuator/friction caches from
            # one pose attached to another; only reset-time forwards are added.
            _reset_solver_state(env, env_ids)
            env.sim.forward()
            randomization.validate_reset(env_ids)
            _reset_solver_state(env, env_ids)


def _reset_solver_state(env, env_ids):
    # Warp reset_data leaves derived forces and per-world model fields intact.
    # BAM reads the previous solve before reset's forward: stale NaNs would
    # immediately poison fresh friction budgets and the new constraint solve.
    for name in ("qfrc_bias", "qfrc_constraint", "qfrc_actuator", "qfrc_passive", "qacc_warmstart"):
        getattr(env.sim.data, name)[env_ids] = 0.0
    # Equality constraints read these before fwd_velocity refreshes them.
    # Both are exactly zero at the reference reset's zero joint/root velocity;
    # otherwise stale NaN * qvel(0) poisons the rope's constraint acceleration.
    for name in ("cvel", "cdof_dot"):
        getattr(env.sim.data, name)[env_ids] = 0.0
    for name in ("dof_frictionloss", "dof_damping"):
        getattr(env.sim.model, name)[env_ids] = env.sim.get_default_field(name)


def make_tug_env_cfg(num_envs: int, *, seed: int = 0):
    """Keep the reference's rope, tiled course, servos and solver, for both seats.

Only placement/joint jitter remains: model and sensor domain randomization,
frozen-opponent control, locomotion rewards and fall terminations are removed.
The reference challenge package must be installed or available on PYTHONPATH.
"""
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    cfg = reference.make_env_cfg()
    cfg.seed = seed
    cfg.scene.num_envs = num_envs
    cfg.scene.env_spacing = 0.0
    cfg.scene.sensors = ()
    # Distinct pristine starts also prevent overlapping ducks during Warp's
    # contact-buffer allocation, before reset events have run.
    for side, xy, rotation in zip(
        SIDES, (reference.west_start(), reference.east_start()),
        ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)), strict=True,
    ):
        robot = reference.OPPONENT_CFG  # Nominal Arena battery on both ducks.
        cfg.scene.entities[side] = replace(
            robot, init_state=replace(robot.init_state, pos=(*xy, 0.12), rot=rotation),
        )
    cfg.actions = {
        side: JointPositionActionCfg(
            entity_name=side, actuator_names=JOINT_NAMES,
            scale=1.0, use_default_offset=True,
        ) for side in SIDES
    }
    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms={"placeholder": ObservationTermCfg(func=_dummy_observation)},
            concatenate_terms=True, enable_corruption=False,
        ),
    }
    cfg.commands = {}
    cfg.events = {
        name: cfg.events[name] for name in (
            "expand_bam_friction_fields", "reset_base", "reset_robot_joints",
            "reset_opponent_base", "reset_opponent_joints", "reset_string",
        )
    }
    cfg.events["reset_solver_state"] = EventTermCfg(func=_reset_solver_state, mode="reset")
    cfg.events["randomize_training_reset"] = EventTermCfg(func=_randomize_training_reset, mode="reset")
    cfg.events["capture_tug_reset"] = EventTermCfg(func=_capture_reset_events, mode="reset")
    # Root reset ranges are offsets from the pristine starts above.
    for name, x_range in (("reset_base", (0.0, 0.02)),
                          ("reset_opponent_base", (-0.02, 0.0))):
        cfg.events[name].params["pose_range"] = {
            "x": x_range, "y": (-0.01, 0.01), "z": (0.0, 0.01),
            "yaw": (-math.radians(reference.YAW_PERTURBATION_DEG),
                    math.radians(reference.YAW_PERTURBATION_DEG)),
        }
        cfg.events[name].params["velocity_range"] = {}
    cfg.rewards = {}
    cfg.terminations = {}
    cfg.curriculum = {}
    cfg.metrics = {}
    cfg.auto_reset = False
    cfg.is_finite_horizon = True
    cfg.episode_length_s = reference.SETTLE_S + reference.BOUT_S
    return cfg


class TugArena:
    """Symmetric match bookkeeping around ``ManagerBasedRlEnv``.

    Rewards are sparse: terminal ``official_score_cm / 100``; no shaping or
    action penalties. A fault/reset gives both seats -2, matching the official
    rating scorer applied separately to each entry, rather than cup utility.
    ``elapsed`` includes the 25 policy-controlled settle steps. Pit and marker
    judgment begins on the first timed step after settling.
    """

    def __init__(self, env, *, diagnostics_dir=None, randomization_config=None):
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.num_players = 2
        self.settle_steps = round(reference.SETTLE_S / env.step_dt)
        self.bout_steps = round(reference.BOUT_S / env.step_dt)
        self.max_episode_length = self.settle_steps + self.bout_steps
        self.elapsed = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._previous_vel = torch.zeros((self.num_envs, 2, 14), device=self.device)
        self._last_actions = torch.zeros_like(self._previous_vel)
        self._joint_ids, self._action_order = [], []
        for side in SIDES:
            ids, names = env.scene[side].find_joints(JOINT_NAMES, preserve_order=True)
            if tuple(names) != JOINT_NAMES:
                raise ValueError(f"{side} does not have the 14 canonical Arena joints")
            self._joint_ids.append(ids)
            targets = env.action_manager.get_term(side).target_names
            if set(targets) != set(JOINT_NAMES) or len(targets) != 14:
                raise ValueError(f"{side} action term does not control the Arena joints")
            self._action_order.append([JOINT_NAMES.index(name) for name in targets])
        model = env.sim.mj_model
        self._rope_ids = [model.site(f"string/p{k}").id
                          for k in range(reference.STRING_SEGMENTS + 1)]
        self._neck_ids = [model.site(f"{side}/{reference.STRING_SITE}").id for side in SIDES]
        self._actor_cache = torch.zeros((self.num_envs, 2, 61), device=self.device)
        self._critic_cache = torch.zeros((self.num_envs, 2, 222), device=self.device)
        self._command = torch.tensor(reference.COMMAND, device=self.device).expand(self.num_envs, -1)
        self.diagnostics = None
        if diagnostics_dir is not None:
            from .diagnostics import ResetDiagnostics

            self.diagnostics = ResetDiagnostics(self, diagnostics_dir)
            env._tug_reset_diagnostics = self.diagnostics
        self.randomization = None
        if randomization_config is not None:
            from .randomization import TugRandomization

            self.randomization = TugRandomization(self, randomization_config)

    def set_training_update(self, update):
        if self.randomization is not None:
            self.randomization.set_training_update(update)

    @staticmethod
    def _index(side):
        if side in (0, "robot", "west"):
            return 0
        if side in (1, "opponent", "east"):
            return 1
        raise ValueError(f"unknown tug side {side!r}")

    def _ids(self, ids):
        return (torch.arange(self.num_envs, device=self.device) if ids is None
                else torch.as_tensor(ids, dtype=torch.long, device=self.device).reshape(-1))

    def reset(self, ids=None):
        ids = self._ids(ids)
        try:
            return self._reset(ids)
        except Exception:
            if self.randomization is not None:
                self.randomization.clear(ids)
            raise

    def _reset(self, ids):
        before = self.diagnostics.capture(ids) if self.diagnostics else None
        if self.diagnostics:
            self.diagnostics.after_events = None
        if self.randomization is not None:
            self.randomization.clear(ids)
        try:
            self.env.reset(env_ids=ids)
            if self.randomization is not None:
                self.randomization.validate_reset(ids, resample=False)
        except Exception as exc:
            if self.randomization is not None:
                self.randomization.clear(ids)
            if self.diagnostics:
                self.diagnostics.write("reset_exception", ids, before, error=repr(exc))
            raise
        checks = self._failure_checks(None)
        failed = torch.zeros_like(self.dones)
        for mask in checks.values():
            failed |= mask
        if failed[ids].any():
            if self.randomization is not None:
                self.randomization.clear(ids)
            path = self.diagnostics.write("reset_physics", ids, before, checks) if self.diagnostics else None
            raise FloatingPointError(f"Tug reset produced invalid physics or a torn string; diagnostics={path}")
        self.elapsed[ids] = 0
        self.dones[ids] = False
        self._previous_vel[ids] = 0
        self._last_actions[ids] = 0
        if self._refresh_observations(sanitize=False)[ids].any():
            if self.randomization is not None:
                self.randomization.clear(ids)
            for name, value in (("actor", self._actor_cache), ("critic", self._critic_cache)):
                flat = value.reshape(self.num_envs, -1)
                checks[f"{name}_nonfinite"] = ~torch.isfinite(flat).all(dim=1)
                checks[f"{name}_above_mjmaxval"] = (flat.abs() > mujoco.mjMAXVAL).any(dim=1)
            path = self.diagnostics.write("reset_observations", ids, before, checks) if self.diagnostics else None
            raise FloatingPointError(f"Tug reset produced invalid observations; diagnostics={path}")
        self._actor_cache.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        self._critic_cache.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        return tuple(self.actor_obs(side) for side in range(2))

    def actor_obs(self, side):
        return self._actor_cache[:, self._index(side)]

    def critic_obs(self, side):
        return self._critic_cache[:, self._index(side)]

    def _refresh_observations(self, *, sanitize=True):
        robots = [self.env.scene[side].data for side in SIDES]
        for i, robot in enumerate(robots):
            ids = self._joint_ids[i]
            self._actor_cache[:, i] = torch.cat((
                robot.root_link_ang_vel_b, robot.projected_gravity_b,
                robot.joint_pos[:, ids] - robot.default_joint_pos[:, ids],
                self._previous_vel[:, i], self._last_actions[:, i], self._command,
            ), dim=-1)
        points = self.env.sim.data.site_xpos[:, self._rope_ids]
        origin = self.env.scene.env_origins
        for i in range(2):
            # East's privileged world is west's world turned by half a turn;
            # the rope is also reordered so its own neck is always first.
            rotation = points.new_tensor((1.0, 1.0, 1.0) if i == 0 else (-1.0, -1.0, 1.0))
            own, other = robots[i], robots[1 - i]
            poses = [(r.root_link_pos_w - origin) * rotation for r in (own, other)]
            velocities = [r.root_link_lin_vel_w * rotation for r in (own, other)]
            quaternions = [r.root_link_quat_w for r in (own, other)]
            if i == 1:
                quaternions = [torch.stack((q[:, 3], q[:, 2], -q[:, 1], -q[:, 0]), dim=-1)
                               for q in quaternions]
            # q and -q represent the same orientation; avoid a side-dependent
            # sign discontinuity at the nominal east spawn.
            quaternions = [torch.where(
                q.gather(1, q.abs().argmax(dim=-1, keepdim=True)) < 0, -q, q,
            ) for q in quaternions]
            rope = (points - origin[:, None]) * rotation
            if i == 1:
                rope = rope.flip(1)
            self._critic_cache[:, i] = torch.cat((
                self._actor_cache[:, i], self._actor_cache[:, 1 - i],
                *poses, *velocities, *quaternions,
                own.joint_vel[:, self._joint_ids[i]],
                other.joint_vel[:, self._joint_ids[1 - i]], rope.flatten(1),
                (self.elapsed.float() / self.max_episode_length)[:, None],
            ), dim=-1)
        # Derived quantities (notably body cvel) may be invalid even when the
        # integrator's qpos/qvel passed its guards. Judge this boundary before
        # sanitizing terminal caches; the collector must reset the whole world.
        invalid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for values in (self._actor_cache, self._critic_cache):
            flat = values.reshape(self.num_envs, -1)
            invalid |= (~torch.isfinite(flat) | (flat.abs() > mujoco.mjMAXVAL)).any(dim=1)
        if sanitize:
            self._actor_cache.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            self._critic_cache.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        return invalid

    def _failure_checks(self, time_before):
        data = self.env.sim.data
        checks = {}
        for name in ("qpos", "qvel", "qacc", "qacc_warmstart", "sensordata", "site_xpos"):
            value = getattr(data, name, None)
            if value is not None:
                checks[f"{name}_nonfinite"] = ~torch.isfinite(value).reshape(self.num_envs, -1).all(dim=1)
                if name in ("qpos", "qvel", "qacc"):
                    # Native MuJoCo also resets finite values beyond mjMAXVAL.
                    checks[f"{name}_above_mjmaxval"] = (value.abs() > mujoco.mjMAXVAL).reshape(self.num_envs, -1).any(dim=1)
        # Warp does not expose native MuJoCo warning counters. Nonfinite state,
        # a rewound physics clock, and torn ties cover its reset equivalents.
        if time_before is not None:
            checks["clock_rewound"] = data.time.reshape(-1) <= time_before
        ends = data.site_xpos[:, [self._rope_ids[0], self._rope_ids[-1]]]
        necks = data.site_xpos[:, self._neck_ids]
        distances = (ends - necks).norm(dim=-1)
        for side in range(2):
            checks[f"rope_{SIDES[side]}_detached"] = distances[:, side] > TORN_M
        return checks

    def _numerical_failure(self, time_before):
        failed = torch.zeros_like(self.dones)
        for mask in self._failure_checks(time_before).values():
            failed |= mask
        return failed

    def step(self, actions):
        try:
            return self._step(actions)
        except Exception:
            if self.randomization is not None:
                self.randomization.clear()
            raise

    def _step(self, actions):
        if actions.shape != (self.num_envs, 28):
            raise ValueError(f"expected {(self.num_envs, 28)} actions; got {tuple(actions.shape)}")
        if self.dones.any():
            raise RuntimeError("Reset completed tug bouts before stepping again")
        actions = actions.to(device=self.device).reshape(self.num_envs, 2, 14)
        fault_by_side = ~torch.isfinite(actions).all(dim=-1)
        fault = fault_by_side.any(dim=1)
        safe_actions = torch.where(fault[:, None, None], 0.0, actions)
        for i, side in enumerate(SIDES):
            self._previous_vel[:, i] = self.env.scene[side].data.joint_vel[:, self._joint_ids[i]]
        time_before = getattr(self.env.sim.data, "time", None)
        time_before = None if time_before is None else time_before.reshape(-1).clone()
        routed = torch.cat([safe_actions[:, i, self._action_order[i]] for i in range(2)], dim=-1)
        if self.randomization is not None:
            self.randomization.substep = 0
        self.env.step(routed)
        self.elapsed += 1
        self._last_actions.copy_(safe_actions)
        numerical = self._numerical_failure(time_before)
        numerical |= self._refresh_observations()
        ctrl = getattr(self.env.sim.data, "ctrl", None)
        if ctrl is not None:
            numerical |= ~torch.isfinite(ctrl).reshape(self.num_envs, -1).all(dim=1)
        in_pit = torch.stack([self.env.scene[side].data.root_link_pos_w[:, 2]
                              < -reference.PIT_DROP_M for side in SIDES], dim=-1)
        marker = self.env.sim.data.site_xpos[:, self._rope_ids[reference.MIDDLE], 0]
        timed = self.elapsed > self.settle_steps
        timed_out = self.elapsed >= self.max_episode_length
        self.dones = fault | numerical | (timed & (in_pit.any(dim=1) | timed_out))
        scores = torch.stack((-marker, marker), dim=-1)
        scores += in_pit.flip(1).float() - in_pit.float()
        scores = torch.where((fault | numerical)[:, None], FAULT_SCORE, scores)
        scores = torch.where(self.dones[:, None], scores, 0.0)
        # A sole pit entrant loses; simultaneous entries and timeout use marker.
        winner = torch.where(marker < 0, 0, torch.where(marker > 0, 1, -1))
        sole_pit = in_pit.sum(dim=1) == 1
        winner = torch.where(sole_pit, in_pit[:, 0].long(), winner)
        sole_fault = fault_by_side.sum(dim=1) == 1
        winner = torch.where(fault, torch.where(sole_fault, fault_by_side[:, 0].long(), -1), winner)
        winner = torch.where(self.dones & ~numerical, winner, -1)
        info = {
            "score_cm": scores * 100.0, "in_pit": in_pit, "fault": fault,
            "fault_by_side": fault_by_side, "numerical_reset": numerical,
            "timed_out": timed_out, "winner": winner,
            "elapsed_seconds": (self.elapsed - self.settle_steps).clamp_min(0) * self.env.step_dt,
        }
        if self.randomization is not None:
            info["randomization"] = self.randomization.diagnostics()
        return (tuple(self.actor_obs(i) for i in range(2)),
                (scores[:, 0], scores[:, 1]), self.dones.clone(), info)
