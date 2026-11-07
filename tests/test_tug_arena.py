"""Official tug judgment, paired state, observation lag and action routing."""

from types import SimpleNamespace
import json

import pytest
import torch

from mjlab_microduck.tug_of_war.arena import (
    JOINT_NAMES, SIDES, TugArena, make_tug_env_cfg,
)


class _Scene(dict):
    pass


class _Env:
    """Deterministic physical boundary; tests can set post-step state directly."""

    def __init__(self, n=1):
        self.num_envs, self.device, self.step_dt = n, "cpu", 0.02
        self.scene = _Scene()
        self.scene.env_origins = torch.zeros(n, 3)
        for side in SIDES:
            data = SimpleNamespace(
                root_link_pos_w=torch.zeros(n, 3),
                root_link_quat_w=torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
                root_link_ang_vel_b=torch.zeros(n, 3),
                root_link_lin_vel_w=torch.zeros(n, 3),
                projected_gravity_b=torch.tensor([[0.0, 0, -1.0]]).repeat(n, 1),
                joint_pos=torch.zeros(n, 14), default_joint_pos=torch.zeros(n, 14),
                joint_vel=torch.zeros(n, 14),
            )
            self.scene[side] = SimpleNamespace(
                data=data, find_joints=lambda *args, **kwargs: (list(range(14)), list(JOINT_NAMES)),
            )
        terms = {side: SimpleNamespace(target_names=list(JOINT_NAMES)) for side in SIDES}
        terms["opponent"].target_names.reverse()
        self.action_manager = SimpleNamespace(get_term=terms.__getitem__)
        names = [f"string/p{k}" for k in range(17)] + [f"{side}/string_end" for side in SIDES]
        model = SimpleNamespace(site=lambda name: SimpleNamespace(id=names.index(name)))
        data = SimpleNamespace(
            site_xpos=torch.zeros(n, 19, 3), qpos=torch.zeros(n, 42),
            qvel=torch.zeros(n, 40), qacc=torch.zeros(n, 40),
            qacc_warmstart=torch.zeros(n, 40), sensordata=torch.zeros(n, 0),
            ctrl=torch.zeros(n, 28), time=torch.zeros(n),
        )
        self.sim = SimpleNamespace(mj_model=model, data=data)
        self.on_step = lambda: None

    def reset(self, *, env_ids):
        self.sim.data.time[env_ids] = 0
        self.sim.data.site_xpos[env_ids] = 0
        for side in SIDES:
            self.scene[side].data.root_link_pos_w[env_ids, 2] = 0.12
            self.scene[side].data.joint_vel[env_ids] = 0

    def step(self, actions):
        self.actions = actions.clone()
        self.sim.data.time += self.step_dt
        self.on_step()


def test_config_retains_reference_physics_without_asymmetric_training_terms():
    cfg = make_tug_env_cfg(4, seed=7)
    assert cfg.seed == 7 and cfg.scene.num_envs == 4
    assert cfg.scene.env_spacing == 0 and cfg.scene.sensors == ()
    assert cfg.scene.spec_fn.__name__ == "tug_scene"
    assert cfg.sim.mujoco.timestep == 0.005 and cfg.decimation == 4
    assert cfg.sim.nconmax == 200 and cfg.sim.njmax == 1500
    assert cfg.sim.mujoco.iterations == 10 and cfg.sim.mujoco.ls_iterations == 20
    assert not any((cfg.rewards, cfg.terminations, cfg.curriculum, cfg.commands))
    assert not cfg.auto_reset and cfg.episode_length_s == 45.5
    assert tuple(cfg.actions) == SIDES
    assert all(type(term).__name__ == "JointPositionActionCfg" for term in cfg.actions.values())
    assert all(act.vin_range == (7.35, 7.35)
               for robot in cfg.scene.entities.values() for act in robot.articulation.actuators)
    assert all(term.mode != "interval" for term in cfg.events.values())
    assert cfg.scene.entities["robot"].init_state.pos[0] < 0
    assert cfg.scene.entities["opponent"].init_state.pos[0] > 0


def test_both_seats_have_exact_policy_layout_independent_history_and_action_routing():
    env = _Env(2)
    arena = TugArena(env)
    obs = arena.reset()
    assert obs[0].shape == obs[1].shape == (2, 61)
    assert arena.critic_obs(0).shape == arena.critic_obs(1).shape == (2, 222)
    assert arena.max_episode_length == 2275 and arena.settle_steps == 25
    assert torch.equal(obs[0][:, 20:48], torch.zeros(2, 28))
    torch.testing.assert_close(obs[0][:, 48], torch.full((2,), -0.3))
    assert torch.equal(obs[0][:, 49:], torch.zeros(2, 12))
    for i, side in enumerate(SIDES):
        env.scene[side].data.joint_vel[:] = i + 2
    env.on_step = lambda: env.scene["robot"].data.joint_vel.fill_(99)
    actions = torch.arange(56).float().reshape(2, 28) / 100
    obs, rewards, done, _ = arena.step(actions)
    torch.testing.assert_close(obs[0][:, 20:34], torch.full((2, 14), 2.0))
    torch.testing.assert_close(obs[1][:, 20:34], torch.full((2, 14), 3.0))
    torch.testing.assert_close(obs[0][:, 34:48], actions[:, :14])
    torch.testing.assert_close(obs[1][:, 34:48], actions[:, 14:])
    torch.testing.assert_close(env.actions[:, 14:], actions[:, 14:].flip(1))
    assert not done.any() and not any(r.any() for r in rewards)
    arena.reset([0])
    assert arena.elapsed.tolist() == [0, 1]
    assert not arena.actor_obs(0)[0, 20:48].any()
    torch.testing.assert_close(arena.actor_obs(1)[1, 34:48], actions[1, 14:])


def test_settle_is_policy_controlled_and_pit_is_judged_first_timed_step():
    env = _Env()
    arena = TugArena(env)
    arena.reset()
    env.scene["robot"].data.root_link_pos_w[:, 2] = -0.051
    arena.elapsed[:] = 24
    _, rewards, done, info = arena.step(torch.ones(1, 28) * 0.1)
    assert not done.any() and not rewards[0].any()
    assert info["elapsed_seconds"].item() == 0
    assert (env.actions == 0.1).all()
    _, rewards, done, info = arena.step(torch.zeros(1, 28))
    assert done.item() and rewards[0].item() == -1 and rewards[1].item() == 1
    assert info["winner"].item() == 1
    with pytest.raises(RuntimeError, match="Reset completed"):
        arena.step(torch.zeros(1, 28))


def test_swapping_sides_and_rotating_world_keeps_privileged_observations_symmetric():
    env = _Env()
    arena = TugArena(env)
    arena.reset()
    env.scene["robot"].data.root_link_pos_w[:, 0] = -0.5
    env.scene["opponent"].data.root_link_pos_w[:, 0] = 0.5
    env.scene["opponent"].data.root_link_quat_w[:] = torch.tensor([0.0, 0, 0, 1.0])
    env.sim.data.site_xpos[:, :17, 0] = torch.linspace(-0.5, 0.5, 17)
    env.sim.data.site_xpos[:, 17:, 0] = torch.tensor([-0.5, 0.5])
    arena._refresh_observations()
    torch.testing.assert_close(arena.actor_obs(0), arena.actor_obs(1))
    torch.testing.assert_close(arena.critic_obs(0), arena.critic_obs(1))


def test_marker_pit_timeout_and_simultaneous_pit_scores_match_official_judge():
    env = _Env(5)
    arena = TugArena(env)
    arena.reset()
    arena.elapsed[:] = arena.max_episode_length - 1
    marker = torch.tensor([0.25, 0.2, -0.3, 0.0, 0.0])
    env.sim.data.site_xpos[:, 8, 0] = marker
    env.scene["robot"].data.root_link_pos_w[1:4, 2] = -0.06
    env.scene["opponent"].data.root_link_pos_w[2:4, 2] = -0.06
    _, rewards, done, info = arena.step(torch.zeros(5, 28))
    expected = torch.tensor([[-0.25, 0.25], [-1.2, 1.2], [0.3, -0.3], [0.0, 0.0], [0.0, 0.0]])
    torch.testing.assert_close(torch.stack(rewards, dim=1), expected)
    assert done.all() and info["timed_out"].all()
    assert info["winner"].tolist() == [1, 1, 0, -1, -1]
    # This optional parity check runs when Arena >=0.7 is available. The main
    # assertions above still cover all rules in installations with older Arena.
    official = pytest.importorskip("arena.tug")
    import numpy as np
    for row in range(5):
        trace = np.zeros((2, 1, 21), dtype=np.float32)
        trace[:, 0, 2] = [env.scene[s].data.root_link_pos_w[row, 2].item() for s in SIDES]
        string = np.zeros((1, 17, 3), dtype=np.float32)
        string[0, 8, 0] = marker[row].item()
        how = "draw" if info["winner"][row] == -1 else "pit" if info["in_pit"][row].sum() == 1 else "marker"
        outcome = SimpleNamespace(how=how, string=string, trace=trace)
        for side in range(2):
            assert official.bout_score_cm(outcome, side, 100.0) == pytest.approx(info["score_cm"][row, side].item())


@pytest.mark.parametrize("failure", ["action", "state", "large_state", "derived_nan", "derived_large", "critic_large", "torn", "clock"])
def test_fault_and_physics_failure_are_finite_shared_penalties(failure):
    env = _Env(2)
    arena = TugArena(env)
    arena.reset()
    action = torch.zeros(2, 28)
    if failure == "action":
        action[0, 0] = float("nan")
    elif failure == "state":
        def corrupt():
            env.sim.data.qpos[0, 0] = float("nan")
            env.scene["robot"].data.root_link_ang_vel_b[0, 0] = float("nan")
        env.on_step = corrupt
    elif failure == "large_state":
        env.on_step = lambda: env.sim.data.qacc[0].fill_(1e11)
    elif failure == "derived_nan":
        env.on_step = lambda: env.scene["robot"].data.root_link_ang_vel_b[0].fill_(float("nan"))
    elif failure == "derived_large":
        env.on_step = lambda: env.scene["robot"].data.root_link_ang_vel_b[0].fill_(2e38)
    elif failure == "critic_large":
        env.on_step = lambda: env.scene["opponent"].data.root_link_lin_vel_w[0].fill_(2e38)
    elif failure == "torn":
        env.on_step = lambda: env.sim.data.site_xpos[0, 0].fill_(0.1)
    else:
        env.on_step = lambda: env.sim.data.time.__setitem__(0, 0)
    obs, rewards, done, info = arena.step(action)
    assert done.tolist() == [True, False]
    assert all(r.tolist() == [-2.0, 0.0] for r in rewards)
    assert torch.isfinite(env.actions).all() and all(torch.isfinite(o).all() for o in obs)
    assert torch.isfinite(arena.critic_obs(0)).all()
    assert info["fault"].tolist() == [failure == "action", False]
    assert info["numerical_reset"].tolist() == [failure != "action", False]
    assert info["winner"][0].item() == (1 if failure == "action" else -1)


def test_real_reference_scene_cpu_reset_and_single_step():
    from mjlab.envs import ManagerBasedRlEnv

    env = ManagerBasedRlEnv(make_tug_env_cfg(1), device="cpu")
    try:
        arena = TugArena(env)
        arena.reset()
        obs, rewards, done, info = arena.step(torch.zeros(1, 28))
        assert all(o.shape == (1, 61) and torch.isfinite(o).all() for o in obs)
        assert not done.any() and not info["numerical_reset"].any()
        assert all(not r.any() for r in rewards)
        # Reproduce the captured failure: reset_data leaves stale solver loads
        # which BAM reads before a new forward, re-infecting reset worlds.
        for name in ("qfrc_bias", "qfrc_constraint", "qfrc_actuator", "qfrc_passive", "cvel", "cdof_dot"):
            getattr(env.sim.data, name)[0] = float("nan")
        for side in SIDES:
            for actuator in env.scene[side].actuators:
                env.sim.model.dof_frictionloss[0, actuator._dof_ids] = float("nan")
        arena.reset([0])
        for name in ("qpos", "qvel", "qacc", "sensordata"):
            assert torch.isfinite(getattr(env.sim.data, name)).all()
        assert torch.isfinite(env.sim.model.dof_frictionloss).all()
        _, _, done, info = arena.step(torch.zeros(1, 28))
        assert not done.any() and not info["numerical_reset"].any()
    finally:
        env.close()


def test_solver_reset_preserves_other_worlds_and_nominal_rope_damping():
    from mjlab_microduck.tug_of_war.arena import _reset_solver_state

    fields = ("qfrc_bias", "qfrc_constraint", "qfrc_actuator", "qfrc_passive", "qacc_warmstart", "cvel", "cdof_dot")
    data = SimpleNamespace(**{name: torch.full((2, 3), float("nan")) for name in fields})
    for name in fields:
        getattr(data, name)[1] = 7.0
    defaults = {"dof_frictionloss": torch.tensor([0., 0., 0.]),
                "dof_damping": torch.tensor([0., 0., .003])}
    model = SimpleNamespace(**{name: torch.full((2, 3), 9.0) for name in defaults})
    env = SimpleNamespace(sim=SimpleNamespace(data=data, model=model, get_default_field=defaults.__getitem__))
    rng = torch.get_rng_state().clone()
    _reset_solver_state(env, torch.tensor([0]))
    for name in fields:
        torch.testing.assert_close(getattr(data, name)[0], torch.zeros(3))
        torch.testing.assert_close(getattr(data, name)[1], torch.full((3,), 7.0))
    for name, expected in defaults.items():
        torch.testing.assert_close(getattr(model, name)[0], expected)
        torch.testing.assert_close(getattr(model, name)[1], torch.full((3,), 9.0))
    torch.testing.assert_close(torch.get_rng_state(), rng)
    cfg = make_tug_env_cfg(1)
    assert list(cfg.events).index("reset_string") < list(cfg.events).index("reset_solver_state")
    assert list(cfg.events).index("reset_solver_state") < list(cfg.events).index("capture_tug_reset")


@pytest.mark.parametrize("failure", ["nonfinite", "large", "rope", "observation"])
def test_reset_failure_preserves_raw_stages_and_exact_reason(tmp_path, failure):
    from mjlab_microduck.tug_of_war.arena import _capture_reset_events

    env = _Env(2)
    env.sim.model = SimpleNamespace(dof_frictionloss=torch.zeros(2, 40), dof_damping=torch.zeros(2, 40))
    env.sim.data.qfrc_constraint = torch.zeros(2, 40)
    env.sim.data.eq_active = torch.ones(2, 2, dtype=torch.bool)
    arena = TugArena(env, diagnostics_dir=tmp_path)
    arena.elapsed[0] = 123
    env.sim.data.qfrc_constraint[0, 3] = 42
    reset = env.reset

    def corrupted_reset(*, env_ids):
        reset(env_ids=env_ids)
        _capture_reset_events(env, env_ids)
        if failure == "nonfinite":
            env.sim.data.qacc[0, 3] = float("nan")
            env.sim.model.dof_frictionloss[0, 3] = float("nan")
        elif failure == "large":
            env.sim.data.qacc[0, 3] = 1e11
        elif failure == "rope":
            env.sim.data.site_xpos[0, 0, 0] = 0.1
        else:
            env.scene["robot"].data.root_link_ang_vel_b[0, 0] = float("nan")
        # The batched forward can also corrupt an unrequested live world.
        env.sim.data.qpos[1, 0] = float("nan")

    env.reset = corrupted_reset
    with pytest.raises(FloatingPointError, match="diagnostics="):
        arena.reset([0])
    path, = tmp_path.glob("*.json")
    report = json.loads(path.read_text())
    assert report["requested_env_ids"] == report["dumped_env_ids"] == [0]
    if failure != "observation":
        assert report["failed_env_ids"] == [0, 1]
    reason = {"nonfinite": "qacc_nonfinite", "large": "qacc_above_mjmaxval",
              "rope": "rope_robot_detached", "observation": "actor_nonfinite"}[failure]
    assert report["checks"][reason] == [0]
    saved = torch.load(path.with_suffix(".pt"), weights_only=False)["snapshots"]
    assert saved["before_reset"]["elapsed"].tolist() == [123]
    assert saved["after_reset_events"]["data/qacc"].eq(0).all()
    assert saved["before_reset"]["data/qfrc_constraint"][0, 3] == 42
    if failure == "nonfinite":
        assert torch.isnan(saved["after_forward"]["model/dof_frictionloss"][0, 3])
    if failure == "observation":
        assert torch.isnan(saved["after_forward"]["actor_cache"][0, 0, 0])


def test_healthy_reset_keeps_diagnostics_off_disk_and_does_not_change_rng(tmp_path):
    env = _Env()
    arena = TugArena(env, diagnostics_dir=tmp_path / "diagnostics")
    rng = torch.get_rng_state().clone()
    arena.reset()
    torch.testing.assert_close(torch.get_rng_state(), rng)
    assert not (tmp_path / "diagnostics").exists()


def test_partial_reset_keeps_neighboring_invalid_observations_sanitized(tmp_path):
    env = _Env(2)
    arena = TugArena(env, diagnostics_dir=tmp_path / "diagnostics")
    env.scene["opponent"].data.root_link_ang_vel_b[1, 0] = float("nan")
    arena.reset([0])
    assert torch.isfinite(arena._actor_cache).all()
    assert torch.isfinite(arena._critic_cache).all()
