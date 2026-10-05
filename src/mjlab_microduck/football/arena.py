"""Two-player Microduck football scene and match bookkeeping.

The physics environment owns both articulated robots.  ``FootballArena`` owns
the symmetric game state so a training runner can choose each player's policy.
Actions are always ordered home, away, with 14 joint-position offsets each.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import pairwise
from functools import partial
import math

import mujoco
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.event_manager import requires_model_fields
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains.terrain_entity import TerrainEntityCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab_microduck.tasks.mdp import expand_bam_friction_fields
from mjlab_microduck.football.skills import KICK_BALL_OFFSET_BODY_XY
from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_ALLCOLLISIONS_ROBOT_CFG,
    MICRODUCK_BALL_CFG,
)

FIELD_HALF_LENGTH = 2.0
FIELD_HALF_WIDTH = 1.0
FIELD_CORNER_RADIUS = 0.4
FIELD_HALF_LENGTH_RANGE = (0.8, 2.0)
FIELD_HALF_WIDTH_RANGE = (0.8, 1.0)
WALL_HALF_THICKNESS = 0.025
GOAL_HALF_WIDTH = 0.175
BALL_RADIUS = 0.035
# A flat central rectangle covers most of every randomized pitch. The outer
# 20% of each heightfield half-axis forms a shallow edge ramp. A small margin
# beyond the walls supports the ball as it crosses a goal mouth.
FLOOR_EXIT_MARGIN = 0.08
FLOOR_HALF_LENGTH = FIELD_HALF_LENGTH + FLOOR_EXIT_MARGIN
FLOOR_HALF_WIDTH = FIELD_HALF_WIDTH + FLOOR_EXIT_MARGIN
FLOOR_FLAT_FRACTION = 0.80
FLOOR_MAX_RISE = 0.008
FLOOR_CONCAVITY_SCALE_RANGE = (0.625, 1.0)
FLOOR_GRID_ROWS = 49
FLOOR_GRID_COLS = 65
MATCH_STEPS = 1500  # 30 s at 50 Hz; a scoreless match ends at this step.
FORFEIT_REWARD = 20.0
BALL_APPROACH_REWARD_WEIGHT = 0.5
BALL_APPROACH_FACING_CUTOFF_COS = math.cos(math.pi / 4)
KICK_SUCCESS_REWARD = 0.5
KICK_MIN_GOALWARD_IMPULSE = 0.2  # m/s gained in one 50 Hz control step.
GOAL_REWARD = 20.0
KICK_MIN_GOALWARD_SPEED = 0.25  # m/s after contact.
KICK_REWARD_COOLDOWN_STEPS = 20
BOUNDARY_PENALTY_WEIGHT = 0.005
BOUNDARY_PENALTY_DECAY_M = 0.05
PROXIMITY_PENALTY_CUTOFF_M = 0.10
BODY_PROXIMITY_PENALTY_WEIGHT = 0.001
BODY_PROXIMITY_DECAY_M = 0.05
ROBOT_FOOTPRINT_RADIUS_M = 0.12

PLAYER_SPAWN_X = (0.20, 0.70)
PLAYER_SPAWN_Y = 0.28
BALL_SPAWN_X = 0.50
BALL_SPAWN_Y = 0.45
BALL_SPAWN_X_HALF_FIELD_FRACTION = 0.25
BALL_SPAWN_Y_HALF_FIELD_FRACTION = 0.50
BALL_SPAWN_SPEED = (0.0, 0.05)
SPAWN_CLEARANCE_M = 0.025  # Extra ball-to-duck and ball-to-boundary margin.
# Damping below 0.4 injects energy into ball-floor impacts in MuJoCo Warp:
# a 1 m drop at 0.1 rebounded at 19 m/s after landing at 4.3 m/s. At 0.4,
# the same drop rebounds at about 1.8 m/s; 1.0 is nearly inelastic.
BALL_CONTACT_DAMPING_RANGE = (0.7, 1.0)
# The old ball/plane contact used max(0.5 ball, 1.0 plane) = 1.0 sliding
# friction. Keep the explicit ball-floor pair centered on that value.
BALL_FLOOR_SLIDING_FRICTION_RANGE = (0.75, 1.25)


@requires_model_fields("geom_solref", "pair_solref", "pair_friction")
def randomize_ball_contact(env, env_ids: torch.Tensor | None) -> None:
    """Sample one ball contact response and floor friction per world and reset."""
    ids = (torch.arange(env.num_envs, device=env.device, dtype=torch.long)
           if env_ids is None else env_ids.to(device=env.device, dtype=torch.long))
    if ids.numel() == 0:
        return
    ball_id = mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "ball/ball_geom")
    floor_pair_id = mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_PAIR, "ball_floor")
    if ball_id < 0 or floor_pair_id < 0:
        raise RuntimeError("football ball geom or ball-floor contact pair is missing")
    damping = torch.empty(len(ids), device=env.device).uniform_(*BALL_CONTACT_DAMPING_RANGE)
    friction = torch.empty(len(ids), device=env.device).uniform_(*BALL_FLOOR_SLIDING_FRICTION_RANGE)
    # Absolute values prevent drift across repeated resets.
    env.sim.model.geom_solref[ids, ball_id, 1] = damping
    env.sim.model.pair_solref[ids, floor_pair_id, 1] = damping
    env.sim.model.pair_friction[ids, floor_pair_id, 0] = friction
    env.sim.model.pair_friction[ids, floor_pair_id, 1] = friction


@requires_model_fields("hfield_size")
def randomize_floor_concavity(env, env_ids: torch.Tensor | None) -> None:
    """Fit the shared edge profile to each pitch and sample its shallow rise."""
    ids = (torch.arange(env.num_envs, device=env.device, dtype=torch.long)
           if env_ids is None else env_ids.to(device=env.device, dtype=torch.long))
    if ids.numel() == 0:
        return
    native = env.sim.mj_model
    hfield_id = mujoco.mj_name2id(native, mujoco.mjtObj.mjOBJ_HFIELD, "football_bowl")
    if hfield_id < 0:
        raise RuntimeError("football bowl heightfield is missing")
    if not hasattr(env, "football_floor_scale"):
        env.football_floor_scale = torch.ones(env.num_envs, device=env.device)
    scale = torch.empty(len(ids), device=env.device).uniform_(*FLOOR_CONCAVITY_SCALE_RANGE)
    env.football_floor_scale[ids] = scale
    size = env.sim.model.hfield_size
    size[ids, 4 * hfield_id] = env.football_field_half_length[ids] + FLOOR_EXIT_MARGIN
    size[ids, 4 * hfield_id + 1] = env.football_field_half_width[ids] + FLOOR_EXIT_MARGIN
    size[ids, 4 * hfield_id + 2] = FLOOR_MAX_RISE * scale
    # The offscreen renderer uses the native model for world 0.
    if (ids == 0).any():
        native.hfield_size[hfield_id, :3] = size[0, 4 * hfield_id:4 * hfield_id + 3].detach().cpu().numpy()


def _field_extent(value: torch.Tensor | float | None, points: torch.Tensor, default: float) -> torch.Tensor:
    extent = torch.as_tensor(default if value is None else value, device=points.device, dtype=points.dtype)
    while extent.ndim < points.ndim - 1:
        extent = extent.unsqueeze(-1)
    return torch.broadcast_to(extent, points.shape[:-1])


def field_boundary_clearance(
    relative_xy: torch.Tensor,
    half_length: torch.Tensor | float | None = None,
    half_width: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Signed root clearance from a per-world rounded rectangle, in meters."""
    length = _field_extent(half_length, relative_xy, FIELD_HALF_LENGTH)
    width = _field_extent(half_width, relative_xy, FIELD_HALF_WIDTH)
    core = torch.stack((length - FIELD_CORNER_RADIUS, width - FIELD_CORNER_RADIUS), dim=-1)
    offset = relative_xy.abs() - core
    outside = torch.linalg.vector_norm(offset.clamp_min(0), dim=-1)
    inside = offset.max(dim=-1).values.clamp_max(0)
    return FIELD_CORNER_RADIUS - outside - inside


def field_out_of_bounds(
    relative_xy: torch.Tensor,
    half_length: torch.Tensor | float | None = None,
    half_width: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Test root positions against the rounded rectangular pitch."""
    return field_boundary_clearance(relative_xy, half_length, half_width) < 0


def field_goal_exit(
    previous_xy: torch.Tensor,
    current_xy: torch.Tensor,
    half_length: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Detect an outward root crossing through either open goal mouth."""
    if previous_xy.shape != current_xy.shape or previous_xy.shape[-1] != 2:
        raise ValueError("previous and current player XY must have matching shapes")
    length = _field_extent(half_length, previous_xy, FIELD_HALF_LENGTH)
    exits = torch.zeros_like(previous_xy[..., 0], dtype=torch.bool)
    for direction in (-1.0, 1.0):
        before = direction * previous_xy[..., 0]
        after = direction * current_xy[..., 0]
        crossed = (before <= length) & (after > length)
        fraction = (length - before) / (after - before).clamp_min(1e-8)
        crossing_y = previous_xy[..., 1] + fraction * (current_xy[..., 1] - previous_xy[..., 1])
        exits |= crossed & (crossing_y.abs() < GOAL_HALF_WIDTH)
    return exits


@requires_model_fields("geom_pos", "geom_size", "geom_aabb", "geom_rbound")
def randomize_field_size(env, env_ids: torch.Tensor | None) -> None:
    """Resize each world's physical pitch at reset without accumulating changes."""
    ids = (torch.arange(env.num_envs, device=env.device, dtype=torch.long)
           if env_ids is None else env_ids.to(device=env.device, dtype=torch.long))
    if ids.numel() == 0:
        return
    if not hasattr(env, "football_field_half_length"):
        env.football_field_half_length = torch.full(
            (env.num_envs,), FIELD_HALF_LENGTH, device=env.device,
        )
        env.football_field_half_width = torch.full(
            (env.num_envs,), FIELD_HALF_WIDTH, device=env.device,
        )
        env._football_wall_ids = [
            i for i in range(env.sim.mj_model.ngeom)
            if (mujoco.mj_id2name(env.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("football_0_")
        ]
        if len(env._football_wall_ids) != 38:
            raise RuntimeError("expected 38 football wall geoms")
        # The original compiled geometry is immutable reference data; native
        # geom positions and sizes are later updated for rendering env 0.
        env._football_wall_base_pos = {
            i: env.sim.mj_model.geom_pos[i].copy() for i in env._football_wall_ids
        }
    length = torch.empty(len(ids), device=env.device).uniform_(*FIELD_HALF_LENGTH_RANGE)
    width = torch.empty(len(ids), device=env.device).uniform_(*FIELD_HALF_WIDTH_RANGE)
    env.football_field_half_length[ids] = length
    env.football_field_half_width[ids] = width
    model = env.sim.model
    for geom_id in env._football_wall_ids:
        name = mujoco.mj_id2name(env.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        label = name.removeprefix("football_0_")
        base = env._football_wall_base_pos[geom_id]
        if "_arc_" in label:
            x_side, y_side = label.split("_", 2)[:2]
            model.geom_pos[ids, geom_id, 0] = base[0] + (1 if x_side == "east" else -1) * (length - FIELD_HALF_LENGTH)
            model.geom_pos[ids, geom_id, 1] = base[1] + (1 if y_side == "north" else -1) * (width - FIELD_HALF_WIDTH)
            continue
        if label in ("north", "south"):
            model.geom_pos[ids, geom_id, 1] = (1 if label == "north" else -1) * (width + WALL_HALF_THICKNESS)
            model.geom_size[ids, geom_id, 0] = length - FIELD_CORNER_RADIUS
        else:
            x_side, y_side = label.split("_")
            span = (width - FIELD_CORNER_RADIUS - GOAL_HALF_WIDTH) / 2
            model.geom_pos[ids, geom_id, 0] = (1 if x_side == "east" else -1) * (length + WALL_HALF_THICKNESS)
            model.geom_pos[ids, geom_id, 1] = (1 if y_side == "north" else -1) * (GOAL_HALF_WIDTH + span)
            model.geom_size[ids, geom_id, 1] = span
        size = model.geom_size[ids, geom_id]
        model.geom_aabb[ids, geom_id, 1] = size
        model.geom_rbound[ids, geom_id] = torch.linalg.vector_norm(size, dim=-1)
    # Static worldbody geoms are initialized in data.geom_xpos at compile time.
    # Warp does not recompute those poses from randomized model.geom_pos in
    # forward(), so contacts would otherwise remain at the original field.
    wall_ids = torch.as_tensor(env._football_wall_ids, device=env.device, dtype=torch.long)
    env.sim.data.geom_xpos[ids[:, None], wall_ids[None, :]] = model.geom_pos[
        ids[:, None], wall_ids[None, :]
    ]
    # The offscreen renderer shares the native MuJoCo model and renders env 0.
    # MuJoCo Warp physics uses its expanded per-world model above.
    if (ids == 0).any():
        native = env.sim.mj_model
        for geom_id in env._football_wall_ids:
            native.geom_pos[geom_id] = model.geom_pos[0, geom_id].detach().cpu().numpy()
            native.geom_size[geom_id] = model.geom_size[0, geom_id].detach().cpu().numpy()
            native.geom_aabb[geom_id] = model.geom_aabb[0, geom_id].detach().cpu().numpy().reshape(6)
            native.geom_rbound[geom_id] = float(model.geom_rbound[0, geom_id].item())


def boundary_proximity_penalty(clearance: torch.Tensor) -> torch.Tensor:
    """Exponential wall cost, exactly zero beyond 10 cm of clearance."""
    penalty = -BOUNDARY_PENALTY_WEIGHT * torch.exp(
        -clearance.clamp_min(0) / BOUNDARY_PENALTY_DECAY_M,
    )
    return torch.where(clearance > PROXIMITY_PENALTY_CUTOFF_M, 0.0, penalty)


def body_body_proximity_penalty(player_xy: torch.Tensor) -> torch.Tensor:
    """Symmetric exponential cost for overlapping circular duck footprints."""
    if player_xy.ndim != 3 or player_xy.shape[1:] != (2, 2):
        raise ValueError("expected (N, 2, 2) player XY positions")
    root_distance = torch.linalg.vector_norm(player_xy[:, 0] - player_xy[:, 1], dim=-1)
    clearance = (root_distance - 2 * ROBOT_FOOTPRINT_RADIUS_M).clamp_min(0)
    penalty = -BODY_PROXIMITY_PENALTY_WEIGHT * torch.exp(-clearance / BODY_PROXIMITY_DECAY_M)
    penalty = torch.where(clearance > PROXIMITY_PENALTY_CUTOFF_M, 0.0, penalty)
    return penalty[:, None].repeat(1, 2)


def ball_approach_reward(
    previous_player_xy: torch.Tensor,
    current_player_xy: torch.Tensor,
    previous_ball_xy: torch.Tensor,
    previous_player_yaw: torch.Tensor,
) -> torch.Tensor:
    """Reward moving the right-foot strike point toward the pre-action ball.

    The strike point is the ball placement used by the blind kick skill. Holding
    ball position and yaw fixed across the step prevents a rolling ball or a
    stationary turn from earning translation reward. Positive progress is paid
    only while the duck faces within 45 degrees of the ball; retreat remains
    fully negative regardless of heading.
    """
    to_ball = previous_ball_xy[:, None, :] - previous_player_xy
    forward = torch.stack((previous_player_yaw.cos(), previous_player_yaw.sin()), dim=-1)
    left = torch.stack((-forward[..., 1], forward[..., 0]), dim=-1)
    strike_offset = (KICK_BALL_OFFSET_BODY_XY[0] * forward
                     + KICK_BALL_OFFSET_BODY_XY[1] * left)
    ball_from_strike = previous_ball_xy[:, None, :] - (previous_player_xy + strike_offset)
    start_distance = torch.linalg.vector_norm(ball_from_strike, dim=-1)
    end_distance = torch.linalg.vector_norm(
        previous_ball_xy[:, None, :] - (current_player_xy + strike_offset), dim=-1,
    )
    ball_distance = torch.linalg.vector_norm(to_ball, dim=-1).clamp_min(1e-6)
    facing_cosine = (forward * to_ball).sum(dim=-1) / ball_distance
    facing_weight = ((facing_cosine - BALL_APPROACH_FACING_CUTOFF_COS)
                     / (1.0 - BALL_APPROACH_FACING_CUTOFF_COS)).clamp(0.0, 1.0)
    progress = start_distance - end_distance
    return torch.where(progress > 0, progress * facing_weight, progress) * BALL_APPROACH_REWARD_WEIGHT


def successful_kick_reward(
    contact: torch.Tensor,
    previous_ball_velocity: torch.Tensor,
    current_ball_velocity: torch.Tensor,
    attack_sign: torch.Tensor,
    cooldown: torch.Tensor,
) -> torch.Tensor:
    """Pay only for a contact that gives the ball useful goalward velocity."""
    previous_goalward = previous_ball_velocity[:, :1] * attack_sign
    current_goalward = current_ball_velocity[:, :1] * attack_sign
    successful = (contact & (cooldown == 0)
                  & ((current_goalward - previous_goalward) >= KICK_MIN_GOALWARD_IMPULSE)
                  & (current_goalward >= KICK_MIN_GOALWARD_SPEED))
    return successful.float() * KICK_SUCCESS_REWARD


def _sample_ball_spawn_xy(
    player_xy: torch.Tensor, half_length: torch.Tensor, half_width: torch.Tensor,
) -> torch.Tensor:
    """Pick a central ball spawn with clearance from every duck and the field edge.

    Eight independent candidates avoid a resampling loop on the GPU. The
    center fallback is safe because every duck begins at least 20 cm from it.
    """
    if player_xy.ndim != 3 or player_xy.shape[1] < 1 or player_xy.shape[2] != 2:
        raise ValueError("expected (N, players, 2) player spawn positions")
    count = len(player_xy)
    limits = torch.stack((
        torch.minimum(half_length * BALL_SPAWN_X_HALF_FIELD_FRACTION,
                      torch.full_like(half_length, BALL_SPAWN_X)),
        torch.minimum(half_width * BALL_SPAWN_Y_HALF_FIELD_FRACTION,
                      torch.full_like(half_width, BALL_SPAWN_Y)),
    ), dim=-1)
    candidates = (torch.rand((count, 8, 2), device=player_xy.device) * 2.0 - 1.0) * limits[:, None, :]
    duck_clearance = ROBOT_FOOTPRINT_RADIUS_M + BALL_RADIUS + SPAWN_CLEARANCE_M
    distance = torch.linalg.vector_norm(
        candidates[:, :, None, :] - player_xy[:, None, :, :], dim=-1,
    )
    valid = (
        (distance >= duck_clearance).all(dim=-1)
        & (field_boundary_clearance(candidates, half_length, half_width)
           >= BALL_RADIUS + SPAWN_CLEARANCE_M)
    )
    first = valid.long().argmax(dim=1)
    sampled = candidates[torch.arange(count, device=player_xy.device), first]
    return torch.where(valid.any(dim=1)[:, None], sampled, torch.zeros_like(sampled))


def _dummy_observation(env) -> torch.Tensor:
    return torch.zeros((env.num_envs, 1), device=env.device)


def football_floor_height(
    relative_xy: torch.Tensor,
    concavity_scale: torch.Tensor | float = 1.0,
    half_length: torch.Tensor | float | None = None,
    half_width: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Height of the edge ramp; most of the pitch is exactly flat."""
    length = _field_extent(half_length, relative_xy, FIELD_HALF_LENGTH) + FLOOR_EXIT_MARGIN
    width = _field_extent(half_width, relative_xy, FIELD_HALF_WIDTH) + FLOOR_EXIT_MARGIN
    normalized_edge = torch.maximum(relative_xy[..., 0].abs() / length,
                                    relative_xy[..., 1].abs() / width)
    ramp = ((normalized_edge - FLOOR_FLAT_FRACTION)
            / (1.0 - FLOOR_FLAT_FRACTION)).clamp(0.0, 1.0)
    scale = torch.as_tensor(concavity_scale, device=relative_xy.device, dtype=relative_xy.dtype)
    while scale.ndim < ramp.ndim:
        scale = scale.unsqueeze(-1)
    return FLOOR_MAX_RISE * scale * ramp.square()


def _add_field_walls(spec: mujoco.MjSpec, num_envs: int, spacing: float) -> None:
    """Build four straight sides and rounded corners with open goal mouths."""
    # The imported flat plane is replaced before compilation, so the ducks
    # and ball contact one continuous curved surface with no overlapping floor.
    flat_floor = spec.geom("terrain")
    if flat_floor is None:
        raise RuntimeError("football terrain geom is missing")
    spec.delete(flat_floor)
    xs = np.linspace(-FLOOR_HALF_LENGTH, FLOOR_HALF_LENGTH, FLOOR_GRID_COLS)
    ys = np.linspace(-FLOOR_HALF_WIDTH, FLOOR_HALF_WIDTH, FLOOR_GRID_ROWS)
    xx, yy = np.meshgrid(xs, ys)
    xy = torch.from_numpy(np.stack((xx, yy), axis=-1)).float()
    heights = football_floor_height(xy).numpy()
    peak_height = FLOOR_MAX_RISE
    bowl = spec.add_hfield(
        name="football_bowl", size=(FLOOR_HALF_LENGTH, FLOOR_HALF_WIDTH,
                                    peak_height, 0.02),
        nrow=FLOOR_GRID_ROWS, ncol=FLOOR_GRID_COLS,
        userdata=(heights / peak_height).astype(np.float32).ravel().tolist(),
    )
    spec.worldbody.add_geom(
        name="terrain", type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname=bowl.name, rgba=(0.20, 0.29, 0.35, 1.0),
    )
    # An explicit pair isolates floor friction from ball-foot and ball-wall
    # friction. Its solref is randomized with the ball geom's contact damping.
    spec.add_pair(name="ball_floor", geomname1="terrain", geomname2="ball/ball_geom",
                  condim=3, friction=(1.0, 1.0, 0.005, 0.0001, 0.0001),
                  solref=(0.02, 1.0))
    wall_t, wall_h = 0.025, 0.16
    radius = FIELD_CORNER_RADIUS + wall_t
    straight_half_length = FIELD_HALF_LENGTH - FIELD_CORNER_RADIUS
    straight_half_width = FIELD_HALF_WIDTH - FIELD_CORNER_RADIUS
    goal_side_span = (straight_half_width - GOAL_HALF_WIDTH) / 2
    arc_segments = 8
    rows = math.ceil(num_envs / int(math.sqrt(num_envs)))
    cols = math.ceil(num_envs / rows)

    def add_corner(body, *, sign_x: float, sign_y: float, label: str) -> None:
        center_x = sign_x * straight_half_length
        center_y = sign_y * straight_half_width
        angles = [math.pi / 2 * i / arc_segments for i in range(arc_segments + 1)]
        for segment, (theta0, theta1) in enumerate(pairwise(angles)):
            x0 = center_x + sign_x * radius * math.cos(theta0)
            y0 = center_y + sign_y * radius * math.sin(theta0)
            x1 = center_x + sign_x * radius * math.cos(theta1)
            y1 = center_y + sign_y * radius * math.sin(theta1)
            # Short overlapping boxes approximate each quarter-circle wall.
            length = math.hypot(x1 - x0, y1 - y0)
            tangent_yaw = math.atan2(y1 - y0, x1 - x0)
            body.add_geom(
                name=f"football_{index}_{label}_arc_{segment}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=(ox + (x0 + x1) / 2, oy + (y0 + y1) / 2, wall_h),
                quat=(math.cos(tangent_yaw / 2), 0.0, 0.0,
                      math.sin(tangent_yaw / 2)),
                size=(length / 2 + 0.002, wall_t, wall_h),
                rgba=(0.2, 0.3, 0.45, 1.0),
            )

    for index in range(num_envs):
        row, col = divmod(index, cols)
        ox = -(row - (rows - 1) / 2) * spacing
        oy = (col - (cols - 1) / 2) * spacing
        body = spec.worldbody.add_body(name=f"football_walls_{index}")
        for label, sign in (("north", 1), ("south", -1)):
            body.add_geom(
                name=f"football_{index}_{label}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=(ox, oy + sign * (FIELD_HALF_WIDTH + wall_t), wall_h),
                size=(straight_half_length, wall_t, wall_h),
                rgba=(0.2, 0.3, 0.45, 1.0),
            )
        for x_label, sign_x in (("east", 1), ("west", -1)):
            for y_label, sign_y in (("north", 1), ("south", -1)):
                body.add_geom(
                    name=f"football_{index}_{x_label}_{y_label}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=(ox + sign_x * (FIELD_HALF_LENGTH + wall_t),
                         oy + sign_y * (GOAL_HALF_WIDTH + goal_side_span), wall_h),
                    size=(wall_t, goal_side_span, wall_h),
                    rgba=(0.2, 0.3, 0.45, 1.0),
                )
                add_corner(body, sign_x=sign_x, sign_y=sign_y,
                           label=f"{x_label}_{y_label}")


def make_football_env_cfg(num_envs: int, seed: int = 42, *, solo: bool = False) -> ManagerBasedRlEnvCfg:
    """Make a 50 Hz football scene with one or two physical robots.

    All match reward and observation logic lives in :class:`FootballArena`,
    because stock mjlab managers assume a single controlled actor.
    """
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    if FIELD_HALF_WIDTH_RANGE[0] <= FIELD_CORNER_RADIUS + GOAL_HALF_WIDTH:
        raise ValueError("minimum field half-width cannot fit the corner and goal-side walls")
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
    if solo:
        del cfg.scene.entities["away"]
    sides = ("home",) if solo else ("home", "away")
    cfg.scene.sensors = tuple(
        ContactSensorCfg(
            name=f"{side}_ball_contact",
            primary=ContactMatch(
                mode="geom", pattern=r"^(left_foot_collision|right_foot_collision)$", entity=side,
            ),
            secondary=ContactMatch(mode="body", pattern="ball", entity="ball"),
            fields=("force",), reduce="maxforce", num_slots=1,
            history_length=cfg.decimation,
        )
        for side in sides
    )
    cfg.scene.spec_fn = partial(_add_field_walls, num_envs=1, spacing=0.0)
    cfg.actions = {
        side: JointPositionActionCfg(
            entity_name=side, actuator_names=(".*",),
            scale=1.0, use_default_offset=True,
        ) for side in sides
    }
    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms={"placeholder": ObservationTermCfg(func=_dummy_observation)},
            concatenate_terms=True, enable_corruption=False,
        )
    }
    cfg.commands = {}
    cfg.events = {
        "expand_bam_friction_fields": EventTermCfg(
            func=expand_bam_friction_fields, mode="startup",
        ),
        "randomize_ball_contact": EventTermCfg(
            func=randomize_ball_contact, mode="reset",
        ),
        "randomize_field_size": EventTermCfg(
            func=randomize_field_size, mode="reset",
        ),
        "randomize_floor_concavity": EventTermCfg(
            func=randomize_floor_concavity, mode="reset",
        ),
    }  # FootballArena performs placement after env.reset().
    cfg.rewards = {}
    cfg.terminations = {}
    cfg.curriculum = {}
    cfg.metrics = {}
    cfg.viewer.origin_type = cfg.viewer.OriginType.WORLD
    cfg.viewer.lookat = (0.0, 0.0, 0.0)
    cfg.viewer.distance = 4.8
    cfg.viewer.fovy = 45.0
    cfg.viewer.elevation = -90.0
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
    cfg.is_finite_horizon = False
    return cfg


class FootballArena:
    """Match state and symmetric observation/reward API around a mjlab env.

    ``step`` returns ``(observations, rewards, dones, info)`` where observations
    and rewards are tuples in physical player order. Solo scenes have only
    home; two-player scenes have home and away.
    """

    def __init__(self, env, *, debug_numerics: bool = False):
        self.env = env
        self.debug_numerics = debug_numerics
        self.device = env.device
        self.num_envs = env.num_envs
        self.sides = ("home", "away") if "away" in env.cfg.scene.entities else ("home",)
        self.num_players = len(self.sides)
        self.scores = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.long)
        self.dones = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.attack_sign = torch.ones((self.num_envs, self.num_players), device=self.device)
        self.elapsed = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._rewards = torch.zeros((self.num_envs, self.num_players), device=self.device)
        self._last_actions = torch.zeros((self.num_envs, self.num_players, 14), device=self.device)
        self.field_half_length = torch.full((self.num_envs,), FIELD_HALF_LENGTH, device=self.device)
        self.field_half_width = torch.full((self.num_envs,), FIELD_HALF_WIDTH, device=self.device)
        self.floor_scale = torch.ones(self.num_envs, device=self.device)
        self._game_prev = torch.zeros((self.num_envs, self.num_players, 18), device=self.device)
        self._game_now = torch.zeros_like(self._game_prev)
        self._actor_cache = torch.zeros((self.num_envs, self.num_players, 66), device=self.device)
        self._ball_progress = torch.zeros((self.num_envs, self.num_players), device=self.device)
        self.kick_available = torch.ones((self.num_envs, self.num_players), device=self.device)
        self.kick_reward_cooldown = torch.zeros((self.num_envs, self.num_players), device=self.device, dtype=torch.long)
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

    def _index(self, side: int | str) -> int:
        if side in (0, "home"):
            return 0
        if side in (1, "away") and self.num_players == 2:
            return 1
        raise ValueError(f"invalid side: {side!r}")

    def _entity(self, side: int | str):
        return self.env.scene[self.sides[self._index(side)]]

    def _ids(self, env_ids):
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()

    def _kickoff(self, ids: torch.Tensor, *, new_match: bool) -> None:
        if not ids.numel():
            return
        self.kick_available[ids] = 1.0
        self.kick_reward_cooldown[ids] = 0
        for side in self.sides:
            self.env.scene.sensors[f"{side}_ball_contact"].reset(ids)
        origins = self.env.scene.env_origins[ids]
        if new_match:
            sign = torch.where(torch.rand(len(ids), device=self.device) < 0.5, -1.0, 1.0)
            self.attack_sign[ids, 0] = sign
            if self.num_players == 2:
                self.attack_sign[ids, 1] = -sign
            self.scores[ids] = 0
            self.elapsed[ids] = 0
            self.dones[ids] = False
            self._last_actions[ids] = 0
        player_xy = []
        for side in range(self.num_players):
            player_x = torch.empty(len(ids), device=self.device).uniform_(*PLAYER_SPAWN_X)
            player_y = (torch.rand(len(ids), device=self.device) * 2.0 - 1.0) * PLAYER_SPAWN_Y
            robot = self._entity(side)
            state = robot.data.default_root_state[ids].clone()
            facing = self.attack_sign[ids, side]
            state[:, :3] = origins
            # A side always begins in the half containing its own goal.
            state[:, 0] += -facing * player_x
            state[:, 1] += player_y
            player_xy.append(torch.stack((-facing * player_x, player_y), dim=-1))
            state[:, 2] += 0.115 + football_floor_height(
                player_xy[-1], self.floor_scale[ids],
                self.field_half_length[ids], self.field_half_width[ids],
            )
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
        ball_xy = _sample_ball_spawn_xy(
            torch.stack(player_xy, dim=1),
            self.field_half_length[ids], self.field_half_width[ids],
        )
        state[:, :2] += ball_xy
        state[:, 2] += BALL_RADIUS + football_floor_height(
            ball_xy, self.floor_scale[ids],
            self.field_half_length[ids], self.field_half_width[ids],
        )
        state[:, 7:] = 0
        ball_angle = torch.rand(len(ids), device=self.device) * (2.0 * math.pi)
        ball_speed = torch.empty(len(ids), device=self.device).uniform_(*BALL_SPAWN_SPEED)
        state[:, 7] = ball_speed * ball_angle.cos()
        state[:, 8] = ball_speed * ball_angle.sin()
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
        self.field_half_length[ids] = self.env.football_field_half_length[ids]
        self.field_half_width[ids] = self.env.football_field_half_width[ids]
        self.floor_scale[ids] = self.env.football_floor_scale[ids]
        self._kickoff(ids, new_match=True)
        self._rewards[ids] = 0
        self._refresh_actor_cache()
        self._assert_physics_finite()
        self._assert_finite("reset observations", {
            f"actor_{side}": self._actor_cache[:, i]
            for i, side in enumerate(self.sides)
        })
        return tuple(self.actor_obs(i) for i in range(self.num_players))

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
        for side in range(self.num_players):
            own = self._entity(side).data
            other = self._entity(1 - side).data if self.num_players == 2 else None
            c, s = self._yaw(side).cos(), self._yaw(side).sin()
            def rotate(vec):
                return torch.stack((c * vec[:, 0] + s * vec[:, 1],
                                    -s * vec[:, 0] + c * vec[:, 1]), dim=-1)
            target = origin.clone()
            target[:, 0] += self.attack_sign[:, side] * self.field_half_length
            own_goal = origin.clone()
            own_goal[:, 0] -= self.attack_sign[:, side] * self.field_half_length
            relative_ball = rotate(ball_pos - own.root_link_pos_w[:, :2])
            relative_opponent = (
                rotate(other.root_link_pos_w[:, :2] - own.root_link_pos_w[:, :2])
                if other is not None else torch.zeros_like(relative_ball)
            )
            opponent_velocity = (
                rotate(other.root_link_lin_vel_w[:, :2] - own.root_link_lin_vel_w[:, :2])
                if other is not None else torch.zeros_like(relative_ball)
            )
            opponent_heading = (
                torch.stack((
                    torch.cos(self._yaw(1 - side) - self._yaw(side)),
                    torch.sin(self._yaw(1 - side) - self._yaw(side)),
                ), dim=-1)
                if other is not None else torch.zeros_like(relative_ball)
            )
            game = torch.cat((
                relative_ball,
                rotate(ball_vel - own.root_link_lin_vel_w[:, :2]),
                relative_opponent,
                opponent_velocity,
                rotate(target - own.root_link_pos_w[:, :2]),
                rotate(own_goal - own.root_link_pos_w[:, :2]),
                opponent_heading,
                self.kick_available[:, side:side + 1],
                (self.elapsed.float() / MATCH_STEPS).clamp_max(1)[:, None],
                (self.field_half_length / FIELD_HALF_LENGTH)[:, None],
                (self.field_half_width / FIELD_HALF_WIDTH)[:, None],
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
        if self.num_players == 1:
            game[:, :, 4:8] = 0.0  # No opponent; these slots are exact zeros.
        self._actor_cache = torch.stack([
            torch.cat((self._proprio(i), game[:, i]), dim=-1)
            for i in range(self.num_players)
        ], dim=1)

    def actor_obs(self, side):
        return self._actor_cache[:, self._index(side)]

    def critic_obs(self, side):
        i = self._index(side)
        opponent_velocity = (
            self._entity(1 - i).data.root_link_vel_w
            if self.num_players == 2
            else torch.zeros_like(self._entity(i).data.root_link_vel_w)
        )
        return torch.cat((self._proprio(i), self._game_now[:, i],
                          self._entity(i).data.root_link_vel_w,
                          opponent_velocity,
                          self.env.scene["ball"].data.root_link_vel_w), dim=-1)

    def rewards(self, side):
        return self._rewards[:, self._index(side)]

    def set_kick_available(self, values: torch.Tensor) -> None:
        """Set per-seat controller readiness before the next physics step.

        ``values`` has shape ``(num_envs, num_players)``. The new
        readiness enters the next sampled game state and follows the same
        one-step sensing delay as the other match features.
        """
        if values.shape != (self.num_envs, self.num_players):
            raise ValueError(f"kick availability must have shape {(self.num_envs, self.num_players)}")
        self.kick_available.copy_(values.to(self.device))

    def step(self, joint_actions: torch.Tensor):
        expected = (self.num_envs, 14 * self.num_players)
        if joint_actions.shape != expected:
            raise ValueError(f"expected {expected} joint actions; got {tuple(joint_actions.shape)}")
        if self.dones.any():
            raise RuntimeError("Reset completed matches before stepping again")
        self._assert_finite("joint actions", {
            f"{side}_action": joint_actions[:, i * 14:(i + 1) * 14]
            for i, side in enumerate(self.sides)
        })
        invalid_actions = ~torch.isfinite(joint_actions).all(dim=1)
        safe_actions = joint_actions.clone()
        safe_actions[invalid_actions] = 0.0
        previous_progress = self._progress().clone()
        previous_player_xy = torch.stack([
            self._entity(side).data.root_link_pos_w[:, :2] for side in range(self.num_players)
        ], dim=1).clone()
        previous_ball_xy = self.env.scene["ball"].data.root_link_pos_w[:, :2].clone()
        previous_ball_velocity = self.env.scene["ball"].data.root_link_lin_vel_w[:, :2].clone()
        previous_player_yaw = torch.stack(
            [self._yaw(side) for side in range(self.num_players)], dim=1,
        ).clone()
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
        self._last_actions.copy_(safe_actions.reshape(self.num_envs, self.num_players, 14))
        ball_xy = self.env.scene["ball"].data.root_link_pos_w[:, :2] - self.env.scene.env_origins[:, :2]
        player_world_xy = torch.stack([
            self._entity(side).data.root_link_pos_w[:, :2] for side in range(self.num_players)
        ], dim=1)
        player_xy = player_world_xy - self.env.scene.env_origins[:, None, :2]
        boundary_clearance = field_boundary_clearance(
            player_xy, self.field_half_length[:, None], self.field_half_width[:, None],
        )
        previous_player_xy_local = previous_player_xy - self.env.scene.env_origins[:, None, :2]
        # Only leaving through a goal opening forfeits. A wall/corner overrun
        # is a physics anomaly, not a valid exit from the walled field.
        out_of_bounds = field_goal_exit(
            previous_player_xy_local, player_xy, self.field_half_length[:, None],
        )
        out_of_bounds[numerical_reset] = False
        forfeit = out_of_bounds.any(dim=-1)
        self.forfeit_count += forfeit.long()
        self.last_forfeit_step[forfeit] = self.env.common_step_counter
        crossed = ((ball_xy[:, 0].abs() >= self.field_half_length + BALL_RADIUS)
                   & (ball_xy[:, 1].abs() <= GOAL_HALF_WIDTH - BALL_RADIUS)
                   & ~forfeit & ~numerical_reset)
        ball_progress_reward = (self._progress() - previous_progress).clamp(-0.05, 0.05) * 2.0
        ball_approach = ball_approach_reward(
            previous_player_xy, player_world_xy, previous_ball_xy, previous_player_yaw,
        )
        self.kick_reward_cooldown.sub_(1).clamp_min_(0)
        ball_contacts = torch.stack([
            (self.env.scene.sensors[f"{side}_ball_contact"].data.force_history
             .square().sum(dim=-1) > 1e-8).any(dim=(-1, -2))
            for side in self.sides
        ], dim=1)
        kick_reward = successful_kick_reward(
            ball_contacts, previous_ball_velocity,
            self.env.scene["ball"].data.root_link_lin_vel_w[:, :2],
            self.attack_sign, self.kick_reward_cooldown,
        )
        self.kick_reward_cooldown[(kick_reward > 0) & ~numerical_reset[:, None]] = KICK_REWARD_COOLDOWN_STEPS
        # A modest per-step cost makes falls undesirable without drowning out
        # goals or ball progress. The critic sees exact body heights.
        heights = torch.stack([
            self._entity(side).data.root_link_pos_w[:, 2]
            - self.env.scene.env_origins[:, 2]
            - football_floor_height(
                player_xy[:, side], self.floor_scale,
                self.field_half_length, self.field_half_width,
            )
            for side in range(self.num_players)
        ], dim=1)
        fall_penalty = -(heights < 0.07).float() * 0.02
        boundary_penalty = boundary_proximity_penalty(boundary_clearance)
        body_proximity = (
            body_body_proximity_penalty(player_xy)
            if self.num_players == 2 else torch.zeros_like(boundary_penalty)
        )
        scorer = torch.where(ball_xy[:, 0] > 0, 1.0, -1.0)
        goal_for = crossed[:, None] & (self.attack_sign == scorer[:, None])
        goal_reward = goal_for.float() * GOAL_REWARD - (crossed[:, None] & ~goal_for).float() * GOAL_REWARD
        if self.num_players == 2:
            self.scores += goal_for.long()
        else:
            self.scores[:, 0] += goal_for[:, 0].long()
            self.scores[:, 1] += (crossed & ~goal_for[:, 0]).long()
        # A goal, forfeit, or 30-second draw is terminal. The wrapper resets
        # the whole world before returning the next observation to PPO.
        timed_out = (self.elapsed >= MATCH_STEPS) & ~crossed & ~forfeit & ~numerical_reset
        self.dones = crossed | terminated.bool() | forfeit | numerical_reset | timed_out
        # A boundary exit penalizes only the player that left the pitch.  The
        # other player's terminal reward is zero: an opponent forfeit must not
        # be a source of positive training reward.
        boundary_reward = -out_of_bounds.float() * FORFEIT_REWARD
        reward_components = {
            "ball_progress": ball_progress_reward,
            "ball_approach": ball_approach,
            "successful_kick": kick_reward,
            "fall_penalty": fall_penalty,
            "boundary_proximity": boundary_penalty,
            "body_body_proximity": body_proximity,
            "goal": goal_reward,
            "boundary_forfeit": boundary_reward,
        }
        # Boundary forfeits replace shaping with the exact terminal signal.
        # Numerical failures void the transition and contribute no reward.
        for name, component in reward_components.items():
            if name != "boundary_forfeit":
                component[forfeit] = 0.0
            component[numerical_reset] = 0.0
        self._rewards = torch.stack(tuple(reward_components.values())).sum(dim=0)
        # Internal reset already started a fresh match in failed worlds. Keep
        # its clock and action history at zero while returning a terminal flag
        # for the transition that failed.
        self.elapsed[numerical_reset] = 0
        self._last_actions[numerical_reset] = 0.0
        self._game_now.copy_(self._game_state())
        self._refresh_actor_cache()
        self._assert_finite("football observations", {
            **{f"actor_{side}": self._actor_cache[:, i]
               for i, side in enumerate(self.sides)},
            "game_state": self._game_now,
            "rewards": self._rewards,
        })
        if numerical_reset.any() and not torch.isfinite(self._actor_cache).all():
            raise FloatingPointError("football numerical reset left nonfinite observations")
        done_out = self.dones.clone()
        self.dones[numerical_reset] = False
        scores_out = self.scores.clone()
        scores_out[numerical_reset] = previous_scores[numerical_reset]
        return (tuple(self.actor_obs(i) for i in range(self.num_players)),
                tuple(self.rewards(i) for i in range(self.num_players)), done_out,
                {"scores": scores_out, "goals": goal_for.clone(),
                 "out_of_bounds": out_of_bounds.clone(),
                 "timed_out": timed_out.clone(),
                 "numerical_reset": numerical_reset.clone(),
                 "reward_components": reward_components,
                 "elapsed_seconds": self.elapsed.float() / 50.0})
