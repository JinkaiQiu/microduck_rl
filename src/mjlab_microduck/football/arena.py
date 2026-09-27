"""Two-player Microduck football scene and match bookkeeping.

The physics environment owns both articulated robots.  ``FootballArena`` owns
the symmetric game state so a training runner can choose each player's policy.
Actions are always ordered home, away, with 14 joint-position offsets each.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
import math

import mujoco
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg
from mjlab.terrains.terrain_entity import TerrainEntityCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab_microduck.tasks.mdp import expand_bam_friction_fields
from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_ALLCOLLISIONS_ROBOT_CFG,
    MICRODUCK_BALL_CFG,
)

FIELD_HALF_LENGTH = 1.0
FIELD_HALF_WIDTH = 0.6
GOAL_HALF_WIDTH = 0.175
BALL_RADIUS = 0.035
MATCH_STEPS = 1500  # 30 s at 50 Hz.
FORFEIT_REWARD = 15.0


def _dummy_observation(env) -> torch.Tensor:
    return torch.zeros((env.num_envs, 1), device=env.device)


def _add_field_walls(spec: mujoco.MjSpec, num_envs: int, spacing: float) -> None:
    """Put a box wall around the field, leaving a 35 cm gap at each goal."""
    wall_t, wall_h = 0.025, 0.16
    rows = math.ceil(num_envs / int(math.sqrt(num_envs)))
    cols = math.ceil(num_envs / rows)
    side_span = (FIELD_HALF_WIDTH - GOAL_HALF_WIDTH) / 2
    for index in range(num_envs):
        row, col = divmod(index, cols)
        ox = -(row - (rows - 1) / 2) * spacing
        oy = (col - (cols - 1) / 2) * spacing
        body = spec.worldbody.add_body(name=f"football_walls_{index}")
        for label, sign in (("north", 1), ("south", -1)):
            body.add_geom(name=f"football_{index}_{label}",
                          type=mujoco.mjtGeom.mjGEOM_BOX,
                          pos=(ox, oy + sign * (FIELD_HALF_WIDTH + wall_t), wall_h),
                          size=(FIELD_HALF_LENGTH + wall_t, wall_t, wall_h),
                          rgba=(0.2, 0.3, 0.45, 1.0))
        for x_label, sign_x in (("east", 1), ("west", -1)):
            for y_label, sign_y in (("north", 1), ("south", -1)):
                body.add_geom(name=f"football_{index}_{x_label}_{y_label}",
                              type=mujoco.mjtGeom.mjGEOM_BOX,
                              pos=(ox + sign_x * (FIELD_HALF_LENGTH + wall_t),
                                   oy + sign_y * (GOAL_HALF_WIDTH + side_span), wall_h),
                              size=(wall_t, side_span, wall_h),
                              rgba=(0.2, 0.3, 0.45, 1.0))


def make_football_env_cfg(num_envs: int, seed: int = 42) -> ManagerBasedRlEnvCfg:
    """Make a 50 Hz, two-robot, 28-action mjlab football scene.

    All match reward and observation logic lives in :class:`FootballArena`,
    because stock mjlab managers assume a single controlled actor.
    """
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    cfg = make_velocity_env_cfg()
    cfg.seed = seed
    cfg.scene.num_envs = num_envs
    cfg.scene.env_spacing = 0.0
    cfg.scene.extent = 2.5
    cfg.scene.terrain = TerrainEntityCfg(terrain_type="plane")
    # Keep the pristine model free of inter-robot penetration. MuJoCo Warp
    # allocates its contact buffer from this state before the first reset.
    cfg.scene.entities = {
        "home": replace(MICRODUCK_ALLCOLLISIONS_ROBOT_CFG,
                        init_state=replace(MICRODUCK_ALLCOLLISIONS_ROBOT_CFG.init_state,
                                           pos=(-0.43, 0.0, 0.115))),
        "away": replace(MICRODUCK_ALLCOLLISIONS_ROBOT_CFG,
                        init_state=replace(MICRODUCK_ALLCOLLISIONS_ROBOT_CFG.init_state,
                                           pos=(0.43, 0.0, 0.115), rot=(0.0, 0.0, 0.0, 1.0))),
        "ball": replace(MICRODUCK_BALL_CFG,
                        init_state=replace(MICRODUCK_BALL_CFG.init_state,
                                           pos=(0.0, 0.0, BALL_RADIUS))),
    }
    cfg.scene.sensors = ()
    cfg.scene.spec_fn = partial(_add_field_walls, num_envs=1, spacing=0.0)
    cfg.actions = {
        side: JointPositionActionCfg(
            entity_name=side, actuator_names=(".*",),
            scale=1.0, use_default_offset=True,
        ) for side in ("home", "away")
    }
    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms={"placeholder": ObservationTermCfg(func=_dummy_observation)},
            concatenate_terms=True, enable_corruption=False,
        )
    }
    cfg.commands = {}
    cfg.events = {"expand_bam_friction_fields": EventTermCfg(
        func=expand_bam_friction_fields, mode="startup",
    )}  # FootballArena performs placement after env.reset().
    cfg.rewards = {}
    cfg.terminations = {}
    cfg.curriculum = {}
    cfg.metrics = {}
    cfg.viewer.origin_type = cfg.viewer.OriginType.WORLD
    cfg.viewer.lookat = (0.0, 0.0, 0.0)
    cfg.viewer.distance = 2.7
    cfg.viewer.elevation = -60.0
    cfg.viewer.azimuth = 90.0
    cfg.viewer.max_extra_envs = 0
    cfg.viewer.width = 640
    cfg.viewer.height = 480
    cfg.sim.nconmax = 1024
    cfg.sim.njmax = 4000
    cfg.sim.mujoco.timestep = 0.005
    cfg.decimation = 4
    cfg.episode_length_s = 30.0
    cfg.auto_reset = False
    cfg.is_finite_horizon = True
    return cfg


class FootballArena:
    """Match state and symmetric observation/reward API around a mjlab env.

    ``step`` returns ``(observations, rewards, dones, info)`` where observations
    and rewards are two-element tuples in home/away order. The caller may pass
    either integer side 0/1 or string side ``home``/``away`` to accessors.
    """

    def __init__(self, env, *, debug_numerics: bool = False):
        self.env = env
        self.debug_numerics = debug_numerics
        self.device = env.device
        self.num_envs = env.num_envs
        self.scores = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.long)
        self.dones = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.attack_sign = torch.ones((self.num_envs, 2), device=self.device)
        self.elapsed = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._rewards = torch.zeros((self.num_envs, 2), device=self.device)
        self._last_actions = torch.zeros((self.num_envs, 2, 14), device=self.device)
        self._game_prev = torch.zeros((self.num_envs, 2, 16), device=self.device)
        self._game_now = torch.zeros_like(self._game_prev)
        self._actor_cache = torch.zeros((self.num_envs, 2, 64), device=self.device)
        self._ball_progress = torch.zeros((self.num_envs, 2), device=self.device)
        self.kick_available = torch.ones((self.num_envs, 2), device=self.device)
        self.forfeit_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.last_forfeit_step = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long,
        )

    def _assert_finite(self, phase: str, fields: dict[str, torch.Tensor]) -> None:
        """Fail at the first invalid physics boundary, with the responsible rows."""
        if not self.debug_numerics:
            return
        failures = []
        any_bad = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for name, values in fields.items():
            bad = ~torch.isfinite(values).reshape(self.num_envs, -1).all(dim=1)
            if bad.any():
                ids = bad.nonzero(as_tuple=False).squeeze(-1)[:8]
                failures.append(f"{name}={ids.cpu().tolist()}")
                any_bad |= bad
        if failures:
            ids = any_bad.nonzero(as_tuple=False).squeeze(-1)[:8]
            context = {
                "env_ids": ids.cpu().tolist(),
                "match_elapsed": self.elapsed[ids].cpu().tolist(),
                "forfeit_count": self.forfeit_count[ids].cpu().tolist(),
                "last_forfeit_step": self.last_forfeit_step[ids].cpu().tolist(),
            }
            raise FloatingPointError(
                f"football {phase} nonfinite at control step "
                f"{self.env.common_step_counter}: {'; '.join(failures)}; "
                f"context={context}"
            )

    def _assert_physics_finite(self) -> None:
        if not self.debug_numerics:
            return
        data = self.env.sim.data
        self._assert_finite("physics", {
            "qpos": data.qpos, "qvel": data.qvel,
            "qacc": data.qacc, "qacc_warmstart": data.qacc_warmstart,
            "sensordata": data.sensordata,
        })

    def _nonfinite_physics_mask(self) -> torch.Tensor:
        """Find failed worlds before match state or observations use their data."""
        data = self.env.sim.data
        bad = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for values in (data.qpos, data.qvel, data.qacc,
                       data.qacc_warmstart, data.sensordata):
            bad |= ~torch.isfinite(values).reshape(self.num_envs, -1).all(dim=1)
        return bad

    @staticmethod
    def _index(side: int | str) -> int:
        if side in (0, "home"):
            return 0
        if side in (1, "away"):
            return 1
        raise ValueError(f"invalid side: {side!r}")

    def _entity(self, side: int | str):
        return self.env.scene[("home", "away")[self._index(side)]]

    def _ids(self, env_ids):
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()

    def _kickoff(self, ids: torch.Tensor, *, new_match: bool) -> None:
        if not ids.numel():
            return
        self.kick_available[ids] = 1.0
        origins = self.env.scene.env_origins[ids]
        if new_match:
            sign = torch.where(torch.rand(len(ids), device=self.device) < 0.5, -1.0, 1.0)
            self.attack_sign[ids, 0] = sign
            self.attack_sign[ids, 1] = -sign
            self.scores[ids] = 0
            self.elapsed[ids] = 0
            self.dones[ids] = False
            self._last_actions[ids] = 0
        for side in (0, 1):
            robot = self._entity(side)
            state = robot.data.default_root_state[ids].clone()
            facing = self.attack_sign[ids, side]
            state[:, :3] = origins
            state[:, 0] += -facing * 0.43
            state[:, 2] += 0.115
            state[:, 3] = torch.where(facing > 0, 1.0, 0.0)
            state[:, 6] = torch.where(facing > 0, 0.0, 1.0)
            state[:, 7:] = 0
            robot.write_root_state_to_sim(state, env_ids=ids)
            robot.write_joint_state_to_sim(
                robot.data.default_joint_pos[ids],
                robot.data.default_joint_vel[ids], env_ids=ids,
            )
        ball = self.env.scene["ball"]
        state = ball.data.default_root_state[ids].clone()
        state[:, :3] = origins
        state[:, 2] += BALL_RADIUS
        state[:, 7:] = 0
        ball.write_root_state_to_sim(state, env_ids=ids)
        self.env.scene.write_data_to_sim()
        self.env.sim.forward()
        self._game_prev[ids] = self._game_state()[ids]
        self._game_now[ids] = self._game_prev[ids]
        self._ball_progress[ids] = self._progress()[ids]
        self._last_actions[ids] = 0

    def reset(self, env_ids=None):
        ids = self._ids(env_ids)
        self.env.reset(env_ids=ids)
        self._kickoff(ids, new_match=True)
        self._rewards[ids] = 0
        self._refresh_actor_cache()
        self._assert_physics_finite()
        self._assert_finite("reset observations", {
            "actor_home": self._actor_cache[:, 0],
            "actor_away": self._actor_cache[:, 1],
        })
        return (self.actor_obs(0), self.actor_obs(1))

    def _yaw(self, side):
        q = self._entity(side).data.root_link_quat_w
        return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                           1 - 2 * (q[:, 2].square() + q[:, 3].square()))

    def _body_xy(self, side, world_xy):
        robot = self._entity(side)
        delta = world_xy - robot.data.root_link_pos_w[:, :2]
        yaw = self._yaw(side)
        c, s = yaw.cos(), yaw.sin()
        return torch.stack((c * delta[:, 0] + s * delta[:, 1],
                            -s * delta[:, 0] + c * delta[:, 1]), dim=-1)

    def ball_body_xy(self, side):
        return self._body_xy(side, self.env.scene["ball"].data.root_link_pos_w[:, :2])

    def _progress(self):
        ball_x = self.env.scene["ball"].data.root_link_pos_w[:, 0]
        origin_x = self.env.scene.env_origins[:, 0]
        return (ball_x - origin_x)[:, None] * self.attack_sign

    def _game_state(self):
        ball = self.env.scene["ball"].data
        ball_pos = ball.root_link_pos_w[:, :2]
        ball_vel = ball.root_link_lin_vel_w[:, :2]
        origin = self.env.scene.env_origins[:, :2]
        outputs = []
        for side in (0, 1):
            own = self._entity(side).data
            other = self._entity(1 - side).data
            c, s = self._yaw(side).cos(), self._yaw(side).sin()
            def rotate(vec):
                return torch.stack((c * vec[:, 0] + s * vec[:, 1],
                                    -s * vec[:, 0] + c * vec[:, 1]), dim=-1)
            target = origin.clone()
            target[:, 0] += self.attack_sign[:, side] * FIELD_HALF_LENGTH
            own_goal = origin.clone()
            own_goal[:, 0] -= self.attack_sign[:, side] * FIELD_HALF_LENGTH
            relative_ball = rotate(ball_pos - own.root_link_pos_w[:, :2])
            relative_opponent = rotate(other.root_link_pos_w[:, :2] - own.root_link_pos_w[:, :2])
            game = torch.cat((
                relative_ball,
                rotate(ball_vel - own.root_link_lin_vel_w[:, :2]),
                relative_opponent,
                rotate(other.root_link_lin_vel_w[:, :2] - own.root_link_lin_vel_w[:, :2]),
                rotate(target - own.root_link_pos_w[:, :2]),
                rotate(own_goal - own.root_link_pos_w[:, :2]),
                torch.cos(self._yaw(1 - side) - self._yaw(side))[:, None],
                torch.sin(self._yaw(1 - side) - self._yaw(side))[:, None],
                self.kick_available[:, side:side + 1],
                (1 - self.elapsed.float() / MATCH_STEPS).clamp_min(0)[:, None],
            ), dim=-1)
            outputs.append(game)
        return torch.stack(outputs, dim=1)

    def _proprio(self, side):
        robot = self._entity(side).data
        q = robot.root_link_quat_w
        inverse_vector = -q[:, 1:4]
        def body_vector(world_vector):
            t = 2.0 * torch.cross(inverse_vector, world_vector, dim=-1)
            return world_vector + q[:, :1] * t + torch.cross(inverse_vector, t, dim=-1)
        local_omega = body_vector(robot.root_link_ang_vel_w)
        down = torch.zeros_like(local_omega)
        down[:, 2] = -1.0
        gravity = body_vector(down)
        entity = self._entity(side)
        joint_ids, _ = entity.find_joints(r"^(?!passive_).*$")
        if len(joint_ids) != 14:
            raise RuntimeError(f"expected 14 actuated joints, got {len(joint_ids)}")
        joint_pos = robot.joint_pos[:, joint_ids] - robot.default_joint_pos[:, joint_ids]
        joint_vel = robot.joint_vel[:, joint_ids]
        return torch.cat((local_omega, gravity, joint_pos, joint_vel,
                          self._last_actions[:, self._index(side)]), dim=-1)

    def obs61(self, side):
        """Existing 48D proprio followed by the 13D command slots."""
        i = self._index(side)
        zeros = torch.zeros((self.num_envs, 13), device=self.device)
        return torch.cat((self._proprio(i), zeros), dim=-1)

    def _refresh_actor_cache(self):
        # Sample sensing noise exactly once per physical state. The teacher and
        # student must receive identical observations at a shared pre-action state.
        game = self._game_prev.clone()
        game[:, :, :12] += (torch.rand_like(game[:, :, :12]) - 0.5) * 0.02
        self._actor_cache = torch.stack((
            torch.cat((self._proprio(0), game[:, 0]), dim=-1),
            torch.cat((self._proprio(1), game[:, 1]), dim=-1),
        ), dim=1)

    def actor_obs(self, side):
        return self._actor_cache[:, self._index(side)]

    def critic_obs(self, side):
        i = self._index(side)
        return torch.cat((self._proprio(i), self._game_now[:, i],
                          self._entity(i).data.root_link_vel_w,
                          self._entity(1 - i).data.root_link_vel_w,
                          self.env.scene["ball"].data.root_link_vel_w), dim=-1)

    def rewards(self, side):
        return self._rewards[:, self._index(side)]

    def set_kick_available(self, values: torch.Tensor) -> None:
        """Set per-seat controller readiness before the next physics step.

        ``values`` has shape ``(num_envs, 2)`` in home/away order. The new
        readiness enters the next sampled game state and follows the same
        one-step sensing delay as the other match features.
        """
        if values.shape != (self.num_envs, 2):
            raise ValueError("kick availability must have shape (num_envs, 2)")
        self.kick_available.copy_(values.to(self.device))

    def step(self, actions_28: torch.Tensor):
        if actions_28.shape != (self.num_envs, 28):
            raise ValueError(f"expected {(self.num_envs, 28)} joint actions; got {tuple(actions_28.shape)}")
        if self.dones.any():
            raise RuntimeError("Reset completed matches before stepping again")
        self._assert_finite("joint actions", {
            "home_action": actions_28[:, :14],
            "away_action": actions_28[:, 14:],
        })
        invalid_actions = ~torch.isfinite(actions_28).all(dim=1)
        safe_actions = actions_28.clone()
        safe_actions[invalid_actions] = 0.0
        previous_progress = self._progress().clone()
        previous_scores = self.scores.clone()
        self._game_prev.copy_(self._game_now)
        _, _, terminated, _, _ = self.env.step(safe_actions)
        self._assert_physics_finite()
        numerical_reset = invalid_actions | self._nonfinite_physics_mask()
        if numerical_reset.any():
            # A failed simulator world voids this match. Reset it before any
            # reward, observation, or video trace can consume its invalid state.
            self.reset(numerical_reset.nonzero(as_tuple=False).squeeze(-1))
            if self._nonfinite_physics_mask().any():
                raise FloatingPointError("football numerical reset left nonfinite physics state")
        self.elapsed += 1
        self._last_actions.copy_(safe_actions.reshape(self.num_envs, 2, 14))
        ball_xy = self.env.scene["ball"].data.root_link_pos_w[:, :2] - self.env.scene.env_origins[:, :2]
        player_xy = torch.stack([
            self._entity(side).data.root_link_pos_w[:, :2] - self.env.scene.env_origins[:, :2]
            for side in (0, 1)
        ], dim=1)
        out_of_bounds = ((player_xy[..., 0].abs() > FIELD_HALF_LENGTH)
                         | (player_xy[..., 1].abs() > FIELD_HALF_WIDTH))
        out_of_bounds[numerical_reset] = False
        forfeit = out_of_bounds.any(dim=-1)
        self.forfeit_count += forfeit.long()
        self.last_forfeit_step[forfeit] = self.env.common_step_counter
        crossed = ((ball_xy[:, 0].abs() >= FIELD_HALF_LENGTH + BALL_RADIUS)
                   & (ball_xy[:, 1].abs() <= GOAL_HALF_WIDTH - BALL_RADIUS)
                   & ~forfeit & ~numerical_reset)
        self._rewards = (self._progress() - previous_progress).clamp(-0.05, 0.05) * 2.0
        # A modest per-step cost makes falls undesirable without drowning out
        # goals or ball progress. The critic sees exact body heights.
        for side in (0, 1):
            height = (self._entity(side).data.root_link_pos_w[:, 2]
                      - self.env.scene.env_origins[:, 2])
            self._rewards[:, side] -= (height < 0.07).float() * 0.02
        scorer = torch.where(ball_xy[:, 0] > 0, 1.0, -1.0)
        goal_for = crossed[:, None] & (self.attack_sign == scorer[:, None])
        self._rewards += goal_for.float() * 10.0 - (crossed[:, None] & ~goal_for).float() * 10.0
        self.scores += goal_for.long()
        self.dones = (self.elapsed >= MATCH_STEPS) | terminated.bool() | forfeit | numerical_reset
        finished = (self.elapsed >= MATCH_STEPS) & ~forfeit & ~numerical_reset
        diff = self.scores[:, 0] - self.scores[:, 1]
        outcome = torch.sign(diff).float() * finished.float() * 5.0
        self._rewards[:, 0] += outcome
        self._rewards[:, 1] -= outcome
        # A boundary violation decides the match regardless of the score.
        self._rewards[forfeit] = 0.0
        forfeit_reward = (out_of_bounds[:, 1].float() - out_of_bounds[:, 0].float()) * FORFEIT_REWARD
        self._rewards[:, 0] += forfeit_reward
        self._rewards[:, 1] -= forfeit_reward
        self._rewards[numerical_reset] = 0.0
        goal_ids = (crossed & ~self.dones).nonzero(as_tuple=False).squeeze(-1)
        self._kickoff(goal_ids, new_match=False)
        # Internal reset already started a fresh match in failed worlds. Keep
        # its clock and action history at zero while returning a terminal flag
        # for the transition that failed.
        self.elapsed[numerical_reset] = 0
        self._last_actions[numerical_reset] = 0.0
        self._game_now.copy_(self._game_state())
        self._refresh_actor_cache()
        self._assert_finite("football observations", {
            "actor_home": self._actor_cache[:, 0],
            "actor_away": self._actor_cache[:, 1],
            "game_state": self._game_now,
            "rewards": self._rewards,
        })
        if numerical_reset.any() and not torch.isfinite(self._actor_cache).all():
            raise FloatingPointError("football numerical reset left nonfinite observations")
        done_out = self.dones.clone()
        self.dones[numerical_reset] = False
        scores_out = self.scores.clone()
        scores_out[numerical_reset] = previous_scores[numerical_reset]
        return ((self.actor_obs(0), self.actor_obs(1)),
                (self.rewards(0), self.rewards(1)), done_out,
                {"scores": scores_out, "goals": goal_for.clone(),
                 "out_of_bounds": out_of_bounds.clone(),
                 "numerical_reset": numerical_reset.clone(),
                 "time_remaining": (MATCH_STEPS - self.elapsed).clamp_min(0) / 50.0})
