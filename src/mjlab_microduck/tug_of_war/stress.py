"""CPU robustness probes around the official Arena bout runner.

Each process may run one bout at a time. Manifest fields and model geometry
are unchanged. Global hooks are restored even if policy execution raises.
"""
from __future__ import annotations

import threading
from math import isfinite
import numpy as np
import mujoco
from arena import runner, tug
from arena.course import STRING, STRING_SITE, compile_tug, duck_prefix, string_sites

_LOCK = threading.RLock()
_CONDITIONS = {"nominal", "force", "spawn", "both"}


def sanitize_diagnostics(value):
    """Keep invalid-physics diagnostics inspectable in strict JSON reports."""
    if isinstance(value, np.ndarray):
        return sanitize_diagnostics(value.tolist())
    if isinstance(value, np.generic):
        return sanitize_diagnostics(value.item())
    if isinstance(value, dict):
        return {key: sanitize_diagnostics(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [sanitize_diagnostics(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        return None
    return value


def _subtree(model, root):
    ids = []
    for body in range(1, model.nbody):
        ancestor = body
        while ancestor and ancestor != root:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor == root:
            ids.append(body)
    return np.asarray(ids, dtype=int)


def _geometry(model, data, adrs):
    necks = np.array([model.site(duck_prefix(k) + STRING_SITE).id for k in (0, 1)])
    ends = string_sites(model)[[0, -1]]
    ties = np.linalg.norm(data.site_xpos[ends] - data.site_xpos[necks], axis=1)
    gaps, footprint = [], []
    contact_support = [0, 0]
    for k in (0, 1):
        floor = model.geom("platform_" + ("west" if k == 0 else "east")).id
        for name in ("left_foot_collision", "right_foot_collision"):
            geom = model.geom(duck_prefix(k) + name).id
            gaps.append(float(mujoco.mj_geomDistance(model, data, geom, floor, .1, None)))
            mesh = int(model.geom_dataid[geom])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            verts = model.mesh_vert[start:start+count] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            footprint.append(float(-np.max(verts[:, 0]) if k == 0 else np.min(verts[:, 0])))
            contact_support[k] += sum({int(c.geom1), int(c.geom2)} == {geom, floor} for c in data.contact)
    max_penetration = max((max(0., -float(c.dist)) for c in data.contact), default=0.)
    pit_half = float(model.geom_size[model.geom("pit_floor").id, 0])
    valid = (np.isfinite(data.qpos).all() and float(ties.max()) <= .01001
             and min(gaps) >= -.004 and max(gaps) <= .015
             and min(footprint) >= pit_half + .02 and max_penetration <= .005
             and all(data.qpos[adr.free + 2] > .05 for adr in adrs))
    return {
        "valid": bool(valid), "neck_chord_m": float(np.linalg.norm(np.diff(data.site_xpos[necks], axis=0))),
        "neck_positions_m": data.site_xpos[necks].tolist(),
        "rope_end_positions_m": data.site_xpos[ends].tolist(),
        "tie_errors_m": ties.tolist(), "max_tie_error_m": float(ties.max()),
        "foot_platform_gaps_m": gaps, "feet_within_15mm_of_platform": sum(-.004 <= gap <= .015 for gap in gaps), "foot_pit_edge_clearances_m": [v - pit_half for v in footprint],
        "support_contacts": contact_support, "max_penetration_m": max_penetration,
        "trunk_positions_m": [data.qpos[a.free:a.free+3].tolist() for a in adrs],
    }


def _spawn(model, data, adrs, learner, rng):
    baseline = data.qpos.copy()
    rope = int(model.joint(STRING + "root").qposadr[0])
    necks = [model.site(duck_prefix(k) + STRING_SITE).id for k in (0, 1)]
    direction = 1. if learner == 0 else -1.
    # Original `_perturb` has already drawn each duck's official RNG stream.
    # The opponent keeps those joint and yaw samples exactly.
    for attempt in range(1, 33):
        data.qpos[:] = baseline
        adr = adrs[learner]
        joint_noise = rng.uniform(-.03, .03, len(adr.qpos))
        key = model.key_qpos[adrs[0].stand_key]
        data.qpos[adr.qpos] = key[adr.qpos] + joint_noise
        yaw_delta = float(rng.choice([-1., 1.]) * rng.uniform(4., 8.))
        yaw = np.radians(180. * learner + yaw_delta)
        data.qpos[adr.free+3:adr.free+7] = [np.cos(yaw/2), 0., 0., np.sin(yaw/2)]
        lateral = float(rng.uniform(-.03, .03))
        data.qpos[adr.free+1] += lateral
        mujoco.mj_forward(model, data)
        points = data.site_xpos[necks].copy()
        chord = float(rng.uniform(.98, .995))
        translation = float(direction * rng.uniform(.01, .02))
        yz = points[1, 1:] - points[0, 1:]
        dx = float(np.sqrt(chord**2 - yz @ yz))
        midpoint_x = float(np.mean(points[:, 0]) + translation)
        for k in (0, 1):
            desired_x = midpoint_x + (k-.5)*dx
            data.qpos[adrs[k].free] += desired_x - points[k, 0]
        mujoco.mj_forward(model, data)
        points = data.site_xpos[necks].copy()
        axis = (points[1] - points[0]) / chord
        # The rigid 1 m rope is only rotated/translated; its geometry stays fixed.
        # Symmetric end preload is (1-chord)/2, no more than 1 cm.
        quat = np.r_[1. + axis[0], np.cross([1., 0., 0.], axis)]
        quat /= np.linalg.norm(quat)
        data.qpos[rope:rope+3] = np.mean(points, axis=0)
        data.qpos[rope+3:rope+7] = quat
        mujoco.mj_forward(model, data)
        result = _geometry(model, data, adrs)
        result.update({"attempts": attempt, "learner_yaw_delta_deg": yaw_delta,
                       "learner_joint_noise_rad": joint_noise.tolist(),
                       "learner_lateral_shift_m": lateral,
                       "common_translation_x_m": translation,
                       "target_neck_chord_m": chord,
                       "rigid_rope_length_m": float(1.),
                       "symmetric_rope_preload_m": (1.-chord)/2})
        if result["valid"]:
            return result
    raise ValueError(f"No valid stress spawn after 32 attempts: {result}")


def _pulse_plan(rng, mass, weight):
    pulses, at = [], float(rng.uniform(.00, .04))
    for index in range(int(rng.integers(2, 4))):
        duration = float(rng.uniform(.10, .20))
        alpha = float(rng.uniform(.10, .20))
        # Continuous Hann integral is peak*duration/2. Apply an additional
        # discrete budget guard below so actual integrated impulse never exceeds it.
        peak = min(alpha*weight, 2*mass*.15/duration)
        pulses.append({"index": index, "start_s": at, "duration_s": duration,
                       "weight_fraction_sampled": alpha, "peak_n": peak,
                       "impulse_cap_ns": mass*.15, "delivered_impulse_ns": 0.,
                       "delivered_substeps": 0, "delivered_peak_n": 0.,
                       "first_delivery_s": None, "last_delivery_s": None})
        at += duration + float(rng.uniform(.3, .8))
    return pulses


def run_stress_bout(manifest, west, east, seed, learner_side, condition,
                    model=None, stress_seed=None):
    """Return (official BoutOutcome, diagnostics) for nominal/force/spawn/both.

    stress_seed controls independent spawn and pulse streams. Use the same seed
    for force and both to obtain identical *planned* pulse times/magnitudes.
    Actual exposure is clipped by the official bout's termination.
    """
    if condition not in _CONDITIONS:
        raise ValueError(f"Unknown stress condition: {condition}")
    side = {"west": 0, "east": 1}.get(learner_side, learner_side)
    if side not in (0, 1):
        raise ValueError("learner_side must be west/east or 0/1")
    diagnostics = {"condition": condition, "seed": int(seed), "stress_seed": int(seed if stress_seed is None else stress_seed),
                   "learner_side": ("west", "east")[side], "official_judge": True,
                   "invalid_initialization": False, "force_during_settle": False,
                   "opponent_external_force": False}
    if condition == "nominal":
        return tug.run_bout(manifest, west, east, seed, model=model), diagnostics
    model = compile_tug(manifest) if model is None else model
    adrs = [runner._Addresses.of(model, manifest.start.keyframe, duck_prefix(k)) for k in (0, 1)]
    learner = adrs[side]
    duck_ids = _subtree(model, learner.trunk)
    mass = float(model.body_mass[duck_ids].sum())
    weight = mass * float(np.linalg.norm(model.opt.gravity))
    force_rng = np.random.default_rng([diagnostics["stress_seed"], 74131])
    spawn_rng = np.random.default_rng([diagnostics["stress_seed"], 74132])
    force_enabled = condition in ("force", "both")
    pulses = _pulse_plan(force_rng, mass, weight)
    diagnostics.update({"duck_mass_kg": mass, "duck_subtree_body_count": len(duck_ids),
                        "duck_weight_n": weight, "planned_pulses": pulses if force_enabled else [],
                        "force_direction": "horizontal towards opponent at rope attachment",
                        "force_profile": "Hann, refreshed every physics substep; point-force trunk torque"})
    original_perturb, original_step = runner._perturb, mujoco.mj_step
    saved_fields = {name: getattr(model, name).copy() for name in ("dof_frictionloss", "dof_damping")}
    perturb_calls, held_data = 0, None
    settle_end = round(float(manifest.start.settle_s)/runner.CTRL_DT)*runner.CTRL_DT
    force_neck = model.site(duck_prefix(side) + STRING_SITE).id
    opponent_neck = model.site(duck_prefix(1-side) + STRING_SITE).id

    def perturb(data, adr, actual_manifest, rng, yaw_deg):
        nonlocal perturb_calls, held_data
        original_perturb(data, adr, actual_manifest, rng, yaw_deg)
        held_data = data
        perturb_calls += 1
        if perturb_calls == 2 and condition in ("spawn", "both"):
            mujoco.mj_forward(model, data)
            diagnostics["spawn"] = _spawn(model, data, adrs, side, spawn_rng)

    def step(actual_model, data, *args, **kwargs):
        # The official runner calls this once per substep with nstep=1.
        if args or kwargs:
            raise ValueError("Stress force requires the official single-substep mj_step call")
        if "spawn" not in diagnostics:
            diagnostics["spawn"] = _geometry(model, data, adrs)
            diagnostics["spawn"]["official_spawn"] = True
        data.xfrc_applied[:] = 0.
        timed = float(data.time - settle_end)
        if force_enabled and timed >= -1e-10:
            for pulse in pulses:
                phase = (timed - pulse["start_s"]) / pulse["duration_s"]
                if 0. <= phase < 1.:
                    amplitude = pulse["peak_n"] * .5 * (1.-np.cos(2*np.pi*phase))
                    dt = float(actual_model.opt.timestep)
                    remaining = pulse["impulse_cap_ns"] - pulse["delivered_impulse_ns"]
                    amplitude = max(0., min(amplitude, remaining/dt))
                    if amplitude > 1e-12:
                        vector = data.site_xpos[opponent_neck] - data.site_xpos[force_neck]
                        vector[2] = 0.
                        norm = np.linalg.norm(vector)
                        if norm > 1e-10:
                            force = amplitude*vector/norm
                            arm = data.site_xpos[force_neck] - data.xipos[learner.trunk]
                            data.xfrc_applied[learner.trunk, :3] = force
                            data.xfrc_applied[learner.trunk, 3:] = np.cross(arm, force)
                            pulse["delivered_impulse_ns"] += amplitude*dt
                            pulse["delivered_substeps"] += 1
                            if pulse["first_delivery_s"] is None:
                                pulse["first_delivery_s"] = timed
                            pulse["last_delivery_s"] = timed
                            pulse["delivered_peak_n"] = max(pulse["delivered_peak_n"], amplitude)
        # Keep this sample held through the official post-substep BAM drive
        # and mj_forward; the next substep clears/replaces it, and bout finally
        # always clears it (including exceptions).
        return original_step(actual_model, data)

    with _LOCK:
        try:
            runner._perturb, mujoco.mj_step = perturb, step
            outcome = tug.run_bout(manifest, west, east, seed, model=model)
            if held_data is not None:
                # Settled support is descriptive: a policy falling during official
                # settle is policy behavior, not rejected/resampled initialization.
                final_geometry = _geometry(model, held_data, adrs)
                final_geometry.pop("valid")
                diagnostics["final_geometry"] = final_geometry
            diagnostics["pulses_planned"] = len(pulses) if force_enabled else 0
            diagnostics["pulses_delivered"] = sum(p["delivered_substeps"] > 0 for p in pulses) if force_enabled else 0
            diagnostics["integrated_impulse_ns"] = sum(p["delivered_impulse_ns"] for p in pulses) if force_enabled else 0.
            diagnostics["delivered_peak_n"] = max((p["delivered_peak_n"] for p in pulses), default=0.) if force_enabled else 0.
            diagnostics["total_impulse_equivalent_delta_v_m_s"] = diagnostics["integrated_impulse_ns"] / mass
            diagnostics["force_nonzero"] = bool(diagnostics["integrated_impulse_ns"] > 1e-12)
            diagnostics["force_substantial"] = bool(diagnostics["total_impulse_equivalent_delta_v_m_s"] >= .05)
            diagnostics["substantial_force_threshold_delta_v_m_s"] = .05
            diagnostics["delivered_peak_weight_fraction"] = diagnostics["delivered_peak_n"] / weight
            diagnostics["timed_duration_s"] = float(outcome.time_s)
            return outcome, sanitize_diagnostics(diagnostics)
        finally:
            runner._perturb, mujoco.mj_step = original_perturb, original_step
            if held_data is not None:
                held_data.xfrc_applied[:] = 0.
            for name, values in saved_fields.items():
                getattr(model, name)[:] = values
