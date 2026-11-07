"""Symmetric, resumable bout randomization; nominal Arena physics is untouched."""

from __future__ import annotations

import math

import torch
from mjlab.utils.lab_api.math import quat_apply
from microduck_challenges.tug_of_war import env as reference


CONDITIONS = ("nominal", "force", "spawn", "both")


def _condition_mix(config, device):
    mix = config.get("mix", dict(zip(CONDITIONS, (.4, .2, .2, .2))))
    if not isinstance(mix, dict) or set(mix) != set(CONDITIONS):
        raise ValueError("randomization mix must specify nominal, force, spawn and both")
    values = torch.tensor([mix[k] for k in CONDITIONS], device=device)
    if not torch.isfinite(values).all() or (values < 0).any() or not torch.isclose(values.sum(), values.new_tensor(1.)):
        raise ValueError("randomization mix must contain nonnegative probabilities summing to one")
    return values


def pulse_amplitude(time, starts, durations, peaks, remaining, dt):
    """Hann pulses with a discrete impulse guard, including boundary samples."""
    phase = (time - starts) / durations
    envelope = .5 * (1 - torch.cos(2 * math.pi * phase.clamp(0, 1)))
    amplitude = torch.where((phase >= 0) & (phase < 1), peaks * envelope, 0.)
    return torch.minimum(amplitude, remaining.clamp_min(0) / dt)


class TugRandomization:
    """Sample only at resets, apply forces before BAM at each physics substep."""

    def __init__(self, arena, config):
        self.arena, self.env = arena, arena.env
        self.device, self.num_envs = arena.device, arena.num_envs
        self.config = dict(config)
        self.start_update = int(config.get("start_update", 8000))
        self.ramp_updates = int(config.get("ramp_updates", 300))
        if self.ramp_updates < 1:
            raise ValueError("randomization ramp_updates must be positive")
        self.update = self.start_update
        self.substep = 0
        self.mix = _condition_mix(config, self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(int(config.get("seed", 0)))
        n = self.num_envs
        self.condition = torch.zeros(n, dtype=torch.long, device=self.device)
        self.strength = torch.zeros(n, device=self.device)
        self.starts = torch.zeros((n, 2, 3), device=self.device)
        self.durations = torch.ones_like(self.starts)
        self.peaks = torch.zeros_like(self.starts)
        self.impulses = torch.zeros_like(self.starts)
        self.delivered_steps = torch.zeros_like(self.starts, dtype=torch.long)
        self.delivered_peak = torch.zeros_like(self.starts)
        self.direction_mode = torch.zeros((n, 3), dtype=torch.long, device=self.device)
        self.angles = torch.zeros((n, 2, 3), device=self.device)
        self.reset_attempts = torch.zeros(n, dtype=torch.long, device=self.device)
        self.spawn_valid = torch.ones(n, dtype=torch.bool, device=self.device)
        self.geometry_measured = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.max_tie_error = torch.zeros(n, device=self.device)
        self.min_foot_gap = torch.zeros(n, device=self.device)
        self.max_foot_gap = torch.zeros(n, device=self.device)
        self.min_pit_clearance = torch.zeros(n, device=self.device)
        self.neck_chord = torch.zeros(n, device=self.device)
        self.max_penetration = torch.zeros(n, device=self.device)
        self._baseline = self.env.sim.data.qpos.clone()
        model = self.env.sim.mj_model
        self._rope_q = int(model.joint("string/root").qposadr[0])
        self._body_ids, masses, self._feet, self._arms = [], [], [], []
        for side in ("robot", "opponent"):
            root = model.body(f"{side}/trunk_base").id
            self._body_ids.append(root)
            self._arms.append(torch.tensor(reference.STRING_OFFSET, device=self.device) -
                              torch.as_tensor(model.body_ipos[root].copy(), device=self.device, dtype=torch.float32))
            body_ids = []
            for body in range(1, model.nbody):
                ancestor = body
                while ancestor and ancestor != root:
                    ancestor = int(model.body_parentid[ancestor])
                if ancestor == root:
                    body_ids.append(body)
            masses.append(float(model.body_mass[body_ids].sum()))
            entity = self.env.scene[side]
            local_body = (entity.indexing.body_ids == root).nonzero().flatten().tolist()
            if len(local_body) != 1:
                raise ValueError(f"could not resolve {side} trunk for external wrench")
            entity._tug_force_body_ids = local_body
            floor = "platform_" + ("west" if side == "robot" else "east")
            for name in ("left_foot_collision", "right_foot_collision"):
                geom = model.geom(f"{side}/{name}").id
                mesh = int(model.geom_dataid[geom])
                start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
                vertices = torch.as_tensor(model.mesh_vert[start:start + count].copy(), device=self.device)
                self._feet.append((geom, floor, vertices, 1 if side == "robot" else -1))
        self.mass = torch.tensor(masses, device=self.device)
        self.weight = self.mass * float(sum(float(x) ** 2 for x in model.opt.gravity) ** .5)
        self.cap = self.mass[:, None] * .15
        self.dt = float(model.opt.timestep)
        boxes = {name: (pos, size) for name, pos, size in reference.course_boxes()}
        self._pit_half = float(reference.PIT_M / 2)
        self._platform_top = {floor: float(boxes[floor][0][2] + boxes[floor][1][2])
                              for _, floor, _, _ in self._feet}
        # This is the per-substep callback used by all action terms. An instance
        # hook keeps the wrench held through BAM's drive and the final forward.
        original = self.env.action_manager.apply_action

        def apply_action():
            result = original()
            self.apply_forces()
            return result

        self.env.action_manager.apply_action = apply_action
        self.env._tug_randomization = self

    @property
    def progress(self):
        return max(0., min(1., (self.update - self.start_update + 1) / self.ramp_updates))

    def set_training_update(self, update):
        self.update = int(update)

    def configure_mix(self, config):
        """Explicit continuation override; active bouts keep their sampled plans.

        Accept the complete requested config so seed/curriculum changes cannot
        silently restart learned severity. Reset bouts afterward to use this mix.
        """
        config = dict(config)
        if ({k: v for k, v in config.items() if k != "mix"} !=
                {k: v for k, v in self.config.items() if k != "mix"}):
            raise ValueError("randomization continuation may only change mix")
        values = _condition_mix(config, self.device)
        self.mix.copy_(values)
        self.config = config

    def _rand(self, *shape):
        return torch.rand(shape, device=self.device, generator=self.generator)

    def clear(self, ids=None):
        ids = slice(None) if ids is None else ids
        self.env.sim.data.xfrc_applied[ids] = 0.
        self.env.sim.data.qfrc_applied[ids] = 0.
        self.peaks[ids] = 0.
        self.starts[ids] = 0.
        self.durations[ids] = 1.
        self.impulses[ids] = 0.
        self.delivered_steps[ids] = 0
        self.delivered_peak[ids] = 0.

    def reset(self, ids):
        self.clear(ids)
        n, p = len(ids), self.progress
        mix = self.mix * p
        mix = mix.clone()
        mix[0] += 1 - p
        self.condition[ids] = torch.searchsorted(mix.cumsum(0), self._rand(n)).clamp_max(3)
        self.strength[ids] = p
        self.impulses[ids] = 0
        self.delivered_steps[ids] = 0
        self.delivered_peak[ids] = 0
        self.reset_attempts[ids] = 0
        self.spawn_valid[ids] = True
        self.geometry_measured[ids] = False
        for name in ("max_tie_error", "min_foot_gap", "max_foot_gap", "min_pit_clearance", "neck_chord", "max_penetration"):
            getattr(self, name)[ids] = 0.
        self._baseline[ids] = self.env.sim.data.qpos[ids]
        self._plan_pulses(ids)
        spawn = ids[self.condition[ids] >= 2]
        if len(spawn):
            self._spawn(spawn)

    def _plan_pulses(self, ids):
        n = len(ids)
        durations = (.08 + .12 * self._rand(n, 1, 3)).expand(-1, 2, -1).clone()
        gaps = (.3 + .5 * self._rand(n, 1, 3)).expand(-1, 2, -1)
        starts = torch.zeros_like(durations)
        starts[:, :, 0] = .04 * self._rand(n, 1)
        for k in (1, 2):
            starts[:, :, k] = starts[:, :, k - 1] + durations[:, :, k - 1] + gaps[:, :, k - 1]
        counts = 1 + (self._rand(n, 1) * 3).long()
        enabled = (self.condition[ids] == 1) | (self.condition[ids] == 3)
        mask = (torch.arange(3, device=self.device)[None, None] < counts[:, :, None]) & enabled[:, None, None]
        peaks = (.05 + .05 * self._rand(n, 2, 3)) * self.weight[None, :, None]
        peaks = torch.minimum(peaks, 2 * self.cap[None] / durations)
        self.starts[ids], self.durations[ids] = starts, durations
        self.peaks[ids] = peaks * self.strength[ids, None, None] * mask
        # Paired inward (70%), paired outward (20%), independent planar (10%).
        draws = self._rand(n, 3)
        self.direction_mode[ids] = (draws >= .7).long() + (draws >= .9).long()
        self.angles[ids] = 2 * math.pi * self._rand(n, 2, 3)

    def _spawn(self, ids):
        data, n = self.env.sim.data, len(ids)
        data.qpos[ids] = self._baseline[ids]
        strength = self.strength[ids]
        # Restore nominal joints, then apply a fresh sample: no accumulated DR.
        ends = []
        for i, side in enumerate(("robot", "opponent")):
            entity = self.env.scene[side]
            root = entity.indexing.free_joint_q_adr
            joints = entity.indexing.joint_q_adr[self.arena._joint_ids[i]]
            noise = (.01 + (.03 - .01) * strength)[:, None] * (2 * self._rand(n, 14) - 1)
            data.qpos[ids[:, None], joints] = entity.data.default_joint_pos[ids][:, self.arena._joint_ids[i]] + noise
            yaw_bound = 6 + 2 * strength
            yaw = i * math.pi + torch.deg2rad(yaw_bound) * (2 * self._rand(n) - 1)
            quat = torch.stack((torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)), -1)
            data.qpos[ids[:, None], root[3:7]] = quat
            pristine_y = (reference.west_start() if i == 0 else reference.east_start())[1]
            data.qpos[ids, root[1]] = pristine_y + (.01 + .02 * strength) * (2 * self._rand(n) - 1)
            pose = data.qpos[ids[:, None], root]
            ends.append(pose[:, :3] + quat_apply(quat, pose.new_tensor(reference.STRING_OFFSET).expand(n, 3)))
        ends = torch.stack(ends, 1)
        chord = .98 + .015 * self._rand(n)
        dx = (chord.square() - (ends[:, 1, 1:] - ends[:, 0, 1:]).square().sum(-1)).sqrt()
        midpoint = ends[:, :, 0].mean(1) + .02 * strength * (2 * self._rand(n) - 1)
        for i, side in enumerate(("robot", "opponent")):
            desired = midpoint + (i - .5) * dx
            root = self.env.scene[side].indexing.free_joint_q_adr
            data.qpos[ids, root[0]] += desired - ends[:, i, 0]
            ends[:, i, 0] = desired
        axis = (ends[:, 1] - ends[:, 0]) / chord[:, None]
        quat = torch.stack((1 + axis[:, 0], torch.zeros_like(chord), -axis[:, 2], axis[:, 1]), -1)
        quat /= quat.norm(dim=-1, keepdim=True)
        q = self._rope_q
        data.qpos[ids, q:q + 3] = ends.mean(1)
        data.qpos[ids, q + 3:q + 7] = quat
        self.reset_attempts[ids] += 1

    def _valid_spawn(self, ids):
        data = self.env.sim.data
        self.geometry_measured[ids] = True
        neck = data.site_xpos[ids[:, None], self.arena._neck_ids]
        ends = data.site_xpos[ids[:, None], [self.arena._rope_ids[0], self.arena._rope_ids[-1]]]
        ties = (neck - ends).norm(dim=-1).amax(-1)
        gaps, clearances = [], []
        for geom, floor, vertices, sign in self._feet:
            rotation = data.geom_xmat[ids, geom].reshape(-1, 3, 3)
            world = torch.matmul(vertices[None], rotation.transpose(-1, -2)) + data.geom_xpos[ids, geom, None]
            gaps.append(world[:, :, 2].amin(-1) - self._platform_top[floor])
            nearest = (world[:, :, 0] * sign).amax(-1)
            clearances.append(-nearest - self._pit_half)
        gaps, clearances = torch.stack(gaps, -1), torch.stack(clearances, -1)
        self.max_tie_error[ids] = ties
        self.min_foot_gap[ids], self.max_foot_gap[ids] = gaps.amin(-1), gaps.amax(-1)
        self.min_pit_clearance[ids] = clearances.amin(-1)
        self.neck_chord[ids] = (neck[:, 1] - neck[:, 0]).norm(dim=-1)
        contact = data.contact
        slots = torch.arange(contact.dist.shape[0], device=self.device)
        present = (slots < data.nacon.reshape(-1)[0]) & (contact.worldid >= 0) & (contact.worldid < self.num_envs)
        penetration = torch.where(present, (-contact.dist).clamp_min(0), 0.)
        maximum = torch.zeros(self.num_envs, device=self.device)
        maximum.scatter_reduce_(0, contact.worldid.long().clamp(0, self.num_envs - 1), penetration, reduce="amax", include_self=True)
        self.max_penetration[ids] = maximum[ids]
        finite = torch.isfinite(data.qpos[ids]).all(-1) & torch.isfinite(gaps).all(-1) & torch.isfinite(ties)
        valid = finite & (ties <= .01001) & (gaps.amin(-1) >= -.004) & (gaps.amax(-1) <= .015) & (clearances.amin(-1) >= .02) & (maximum[ids] <= .005)
        self.spawn_valid[ids] = valid
        return valid

    def validate_reset(self, ids, *, resample=True):
        from .arena import _reset_solver_state

        pending = ids[self.condition[ids] >= 2]
        for _ in range(32):
            if not len(pending):
                return
            pending = pending[~self._valid_spawn(pending)]
            if not len(pending):
                return
            if not resample or int(self.reset_attempts[pending].max()) >= 32:
                break
            self._spawn(pending)
            _reset_solver_state(self.env, pending)
            self.env.sim.forward()
        self.clear(ids)
        details = {name: getattr(self, name)[pending].cpu().tolist() for name in (
            "max_tie_error", "min_foot_gap", "max_foot_gap", "min_pit_clearance", "max_penetration")}
        raise FloatingPointError(f"Invalid randomized tug spawn: ids={pending.cpu().tolist()}, {details}")

    def apply_forces(self):
        data = self.env.sim.data
        time = ((self.arena.elapsed - self.arena.settle_steps) * self.env.step_dt + self.substep * self.dt)[:, None, None]
        self.substep += 1
        amplitudes = pulse_amplitude(time, self.starts, self.durations, self.peaks, self.cap[None] - self.impulses, self.dt)
        poses = [data.qpos[:, self.env.scene[side].indexing.free_joint_q_adr] for side in ("robot", "opponent")]
        offset = data.qpos.new_tensor(reference.STRING_OFFSET).expand(self.num_envs, 3)
        neck = torch.stack([pose[:, :3] + quat_apply(pose[:, 3:], offset) for pose in poses], 1)
        inward = neck.flip(1) - neck
        inward[:, :, 2] = 0
        inward /= inward.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        directions = inward[:, :, None].expand(-1, -1, 3, -1).clone()
        mode = self.direction_mode[:, None, :, None]
        directions = torch.where(mode == 1, -directions, directions)
        independent = torch.stack((self.angles.cos(), self.angles.sin(), torch.zeros_like(self.angles)), -1)
        directions = torch.where(mode == 2, independent, directions)
        active = (amplitudes > 0) & torch.isfinite(directions).all(-1)
        amplitudes = torch.where(active, amplitudes, 0.)
        forces = torch.where(active[:, :, :, None], directions * amplitudes[:, :, :, None], 0.).sum(2)
        for i, side in enumerate(("robot", "opponent")):
            arm = quat_apply(poses[i][:, 3:], self._arms[i].expand(self.num_envs, 3))
            torque = torch.cross(arm, forces[:, i], dim=-1)
            torque = torch.where(torch.isfinite(torque), torque, 0.)
            entity = self.env.scene[side]
            entity.write_external_wrench_to_sim(forces[:, i, None], torque[:, None], body_ids=entity._tug_force_body_ids)
        self.impulses += amplitudes * self.dt
        self.delivered_steps += (amplitudes > 1e-12).long()
        self.delivered_peak.copy_(torch.maximum(self.delivered_peak, amplitudes))

    def diagnostics(self):
        result = {name: getattr(self, name).clone() for name in (
            "condition", "strength", "impulses", "delivered_steps", "delivered_peak",
            "reset_attempts", "spawn_valid", "max_tie_error", "min_foot_gap",
            "max_foot_gap", "min_pit_clearance", "neck_chord",
            "max_penetration",
            "geometry_measured",
        )}
        result.update({"curriculum_strength": self.strength.clone(),
                       "spawn_yaw_bound_deg": 6 + 2 * self.strength,
                       "spawn_joint_bound_rad": .01 + .02 * self.strength,
                       "spawn_lateral_bound_m": .01 + .02 * self.strength,
                       "spawn_common_x_bound_m": .02 * self.strength,
                       "delivered_impulse_delta_v": self.impulses.sum(-1) / self.mass,
                       "delivered_pulses": (self.delivered_steps > 0).sum(-1),
                       "force_nonzero": self.impulses.sum(-1) > 1e-12})
        return result

    def state_dict(self):
        # Physical bouts restart when loading the trainer. Keep the random
        # stream/curriculum resumable even when the evaluation batch size differs.
        return {"config": self.config, "update": self.update, "rng_state": self.generator.get_state()}

    def load_state_dict(self, state):
        if state["config"] != self.config:
            raise ValueError("randomization checkpoint config differs from the requested config")
        self.update = int(state["update"])
        self.generator.set_state(state["rng_state"].cpu())
        self.clear()
