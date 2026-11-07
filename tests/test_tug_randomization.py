"""Pulse budgets, partial resets and resumable symmetric randomization on CPU."""

from types import SimpleNamespace

import torch
import pytest

from mjlab_microduck.tug_of_war.randomization import TugRandomization, pulse_amplitude
from microduck_challenges.tug_of_war import env as reference
from mjlab.utils.lab_api.math import quat_apply


def scheduler(n=8):
    randomization = TugRandomization.__new__(TugRandomization)
    r = randomization
    r.device, r.num_envs, r.dt = "cpu", n, .005
    r.start_update, r.ramp_updates, r.update, r.substep = 8000, 300, 8299, 0
    r.config = {"seed": 123}
    r.mix = torch.tensor([.4, .2, .2, .2])
    r.generator = torch.Generator().manual_seed(123)
    r.mass = torch.tensor([.8, .8])
    r.weight, r.cap = r.mass * 9.81, r.mass[:, None] * .15
    r.condition = torch.ones(n, dtype=torch.long)
    r.strength = torch.ones(n)
    for name in ("starts", "peaks", "impulses", "delivered_peak", "angles"):
        setattr(r, name, torch.zeros(n, 2, 3))
    r.durations = torch.ones(n, 2, 3)
    r.delivered_steps = torch.zeros(n, 2, 3, dtype=torch.long)
    r.direction_mode = torch.zeros(n, 3, dtype=torch.long)
    qpos = torch.zeros(n, 14)
    qpos[:, [3, 10]] = 1
    qpos[:, 0], qpos[:, 7] = -.526, .526
    data = SimpleNamespace(qpos=qpos, xfrc_applied=torch.zeros(n, 2, 6), qfrc_applied=torch.zeros(n, 12))
    scene = {}
    for i, side in enumerate(("robot", "opponent")):
        def write(forces, torques, body_ids, i=i):
            data.xfrc_applied[:, i, :3] = forces[:, 0]
            data.xfrc_applied[:, i, 3:] = torques[:, 0]
        scene[side] = SimpleNamespace(indexing=SimpleNamespace(free_joint_q_adr=torch.arange(i * 7, (i + 1) * 7)),
                                      _tug_force_body_ids=[i], write_external_wrench_to_sim=write)
    r.env = SimpleNamespace(scene=scene, sim=SimpleNamespace(data=data), step_dt=.02)
    r.arena = SimpleNamespace(elapsed=torch.full((n,), 25), settle_steps=25)
    r._body_ids, r._arms = [0, 1], [torch.tensor([.026, .0145, .0324215])] * 2
    return r


def test_pulse_has_no_boundary_or_settle_force_and_discrete_impulse_is_capped():
    starts = torch.tensor([.01])
    durations, peaks, cap = torch.tensor([.2]), torch.tensor([200.]), torch.tensor([.12])
    delivered = torch.zeros(1)
    for step in range(-100, 101):
        t = torch.tensor([step * .005])
        force = pulse_amplitude(t, starts, durations, peaks, cap - delivered, .005)
        if t <= starts or t >= starts + durations:
            assert force.item() == 0
        delivered += force * .005
        assert delivered <= cap + 1e-7
    torch.testing.assert_close(delivered, cap)


def test_paired_schedule_bounds_and_checkpoint_stream_survive_batch_size_change():
    r = scheduler(1024)
    ids = torch.arange(r.num_envs)
    r._plan_pulses(ids)
    assert torch.all((r.peaks > 0).sum(-1).ge(1) & (r.peaks > 0).sum(-1).le(3))
    assert r.durations.min() >= .08 and r.durations.max() <= .20
    assert r.starts[:, :, 0].min() >= 0 and r.starts[:, :, 0].max() <= .04
    assert torch.all(r.starts[:, :, 1:] - r.starts[:, :, :-1] - r.durations[:, :, :-1] >= .3 - 1e-7)
    torch.testing.assert_close(r.starts[:, 0], r.starts[:, 1])
    assert torch.all(r.peaks <= r.weight[None, :, None] * .1)
    assert set(r.direction_mode.flatten().tolist()) == {0, 1, 2}
    state = r.state_dict()
    next_random = r._rand(17)
    restored = scheduler(2)
    restored.load_state_dict(state)
    torch.testing.assert_close(restored._rand(17), next_random)
    assert restored.progress == 1
    restored.set_training_update(8000)
    assert 0 < restored.progress < .01


def test_continuation_mix_preserves_resumed_rng_full_severity_and_active_plans():
    source = scheduler()
    source.set_training_update(9000)
    state = source.state_dict()
    resumed = scheduler()
    resumed.load_state_dict(state)
    resumed._plan_pulses(torch.arange(resumed.num_envs))
    plans, rng, mix_buffer = resumed.peaks.clone(), resumed.generator.get_state().clone(), resumed.mix
    requested = {**resumed.config, "mix": {"nominal": .5, "force": .15, "spawn": .15, "both": .2}}
    resumed.configure_mix(requested)
    assert resumed.config == requested and resumed.mix is mix_buffer
    torch.testing.assert_close(resumed.mix, torch.tensor([.5, .15, .15, .2]))
    torch.testing.assert_close(resumed.generator.get_state(), rng)
    torch.testing.assert_close(resumed.peaks, plans)
    assert resumed.update == 9000 and resumed.progress == 1
    restored = scheduler()
    restored.config = dict(requested)
    restored.mix.copy_(resumed.mix)
    restored.load_state_dict(resumed.state_dict())
    torch.testing.assert_close(restored._rand(17), resumed._rand(17))
    assert restored.progress == 1


@pytest.mark.parametrize("changes", [
    {"seed": 999}, {"start_update": 9000}, {"ramp_updates": 300},
    {"mix": {"nominal": .5, "force": .2, "spawn": .2, "both": .2}},
    {"mix": {"nominal": float("nan"), "force": .2, "spawn": .2, "both": .2}},
    {"mix": {"nominal": 1.}},
])
def test_continuation_mix_rejects_invalid_or_non_mix_changes_atomically(changes):
    r = scheduler()
    config, mix, rng = dict(r.config), r.mix.clone(), r.generator.get_state().clone()
    with pytest.raises(ValueError):
        r.configure_mix({**r.config, **changes})
    assert r.config == config
    torch.testing.assert_close(r.mix, mix)
    torch.testing.assert_close(r.generator.get_state(), rng)


def test_force_is_horizontal_point_wrench_and_partial_reset_preserves_other_worlds():
    r = scheduler(2)
    r.starts[:] = 0
    r.durations[:] = .1
    r.peaks[:, :, 0] = .7
    r.substep = 10  # peak of the first pulse
    r.apply_forces()
    wrench = r.env.sim.data.xfrc_applied.clone()
    assert torch.all(wrench[:, :, 2] == 0)
    torch.testing.assert_close(wrench[:, 0, 0], -wrench[:, 1, 0])
    assert wrench[0, 0, 0] > 0 and wrench[0, 1, 0] < 0
    torch.testing.assert_close(wrench[:, 0, 3:], torch.cross(r._arms[0].expand(2, 3), wrench[:, 0, :3], dim=-1))
    r.clear(torch.tensor([0]))
    assert not r.env.sim.data.xfrc_applied[0].any()
    assert not r.peaks[0].any() and not r.impulses[0].any() and not r.delivered_steps[0].any()
    torch.testing.assert_close(r.env.sim.data.xfrc_applied[1], wrench[1])
    assert r.peaks[1].any() and r.impulses[1].any()


def test_inactive_or_settling_worlds_cannot_inject_nan_wrenches():
    r = scheduler(2)
    r.env.sim.data.qpos[0] = float("nan")
    r.arena.elapsed[:] = 24
    r.peaks[:] = .7
    r.apply_forces()
    assert not r.env.sim.data.xfrc_applied.any()
    assert not r.impulses.any()
    assert torch.isfinite(r.env.sim.data.xfrc_applied).all()


def test_inference_collection_keeps_mutable_buffers_for_reset_outside_inference_mode():
    r = scheduler(2)
    r.starts[:] = 0
    r.durations[:] = .1
    r.peaks[:, :, 0] = .7
    r.substep = 10
    peak_buffer = r.delivered_peak
    with torch.inference_mode():
        r.apply_forces()
    assert r.delivered_peak is peak_buffer
    assert not r.delivered_peak.is_inference()
    assert r.delivered_peak.any()
    r.clear()
    assert not r.delivered_peak.any() and not r.impulses.any()
    assert not r.env.sim.data.xfrc_applied.any()


def test_coupled_spawn_total_bounds_rope_length_and_restore_then_apply():
    r = scheduler(256)
    n, ids = r.num_envs, torch.arange(r.num_envs)
    data = r.env.sim.data
    data.qpos = torch.zeros(n, 49)
    r._rope_q = 42
    r.arena._joint_ids = [list(range(14))] * 2
    r.reset_attempts = torch.zeros(n, dtype=torch.long)
    for i, side in enumerate(("robot", "opponent")):
        root = torch.arange(i * 21, i * 21 + 7)
        entity = r.env.scene[side]
        entity.indexing.free_joint_q_adr = root
        entity.indexing.joint_q_adr = torch.arange(i * 21 + 7, i * 21 + 21)
        entity.data = SimpleNamespace(default_joint_pos=torch.zeros(n, 14))
        start = (reference.west_start() if i == 0 else reference.east_start())
        data.qpos[:, root[:3]] = torch.tensor([*start, .12])
        data.qpos[:, root[1]] += .01  # baseline jitter must not accumulate
        data.qpos[:, root[3 if i == 0 else 6]] = 1
    r._baseline = data.qpos.clone()
    state = r.generator.get_state()
    r._spawn(ids)
    first = data.qpos.clone()
    ends = []
    for i, side in enumerate(("robot", "opponent")):
        entity = r.env.scene[side]
        pose = data.qpos[:, entity.indexing.free_joint_q_adr]
        center_y = (reference.west_start() if i == 0 else reference.east_start())[1]
        assert (pose[:, 1] - center_y).abs().max() <= .03
        assert data.qpos[:, entity.indexing.joint_q_adr].abs().max() <= .03
        yaw = 2 * torch.atan2(pose[:, 6], pose[:, 3]) - i * torch.pi
        assert torch.rad2deg(yaw).abs().max() <= 8
        ends.append(pose[:, :3] + quat_apply(pose[:, 3:], torch.tensor(reference.STRING_OFFSET).expand(n, 3)))
    ends = torch.stack(ends, 1)
    chord = (ends[:, 1] - ends[:, 0]).norm(dim=-1)
    assert chord.min() >= .98 - 1e-7 and chord.max() <= .995 + 1e-7
    rope = data.qpos[:, 42:49]
    rope_ends = rope[:, None, :3] + torch.stack([
        quat_apply(rope[:, 3:], torch.tensor([x, 0., 0.]).expand(n, 3)) for x in (-.5, .5)], 1)
    assert (rope_ends - ends).norm(dim=-1).max() <= .01001
    torch.testing.assert_close((rope_ends[:, 1] - rope_ends[:, 0]).norm(dim=-1), torch.ones(n))
    data.qpos[:] = 99
    r.generator.set_state(state)
    r._spawn(ids)
    torch.testing.assert_close(data.qpos, first)


def test_spawn_validation_precedes_official_bam_drive_and_restores_solver_cache(monkeypatch):
    from mjlab_microduck.tug_of_war import arena

    calls = []
    randomization = SimpleNamespace(
        condition=torch.tensor([3]), reset=lambda ids: calls.append("sample"),
        validate_reset=lambda ids: calls.append("validate"),
    )
    env = SimpleNamespace(_tug_randomization=randomization,
                          sim=SimpleNamespace(forward=lambda: calls.append("kinematics")))
    monkeypatch.setattr(arena, "_reset_solver_state", lambda env, ids: calls.append("clear_solver"))
    arena._randomize_training_reset(env, torch.tensor([0]))
    assert calls == ["sample", "clear_solver", "kinematics", "validate", "clear_solver"]
