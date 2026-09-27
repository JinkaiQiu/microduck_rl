"""Visual policy roles for the first rendered football match.

The MuJoCo renderer reads the compiled ``MjModel`` material colors. MuJoCo Warp
uses a separate physics model, so these changes affect appearance only. The
video recorder renders environment 0; other vector environments may have
independent opponents and are intentionally not represented by these colors.
"""

from __future__ import annotations

import mujoco
import numpy as np


# The learner keeps the asset's original white shell and orange details.
OPPONENT_COLORS: dict[str, tuple[float, float, float]] = {
    "scripted": (0.98, 0.70, 0.16),  # gold
    "tactical": (0.16, 0.42, 0.95),  # blue
    "joint": (0.72, 0.30, 0.78),  # purple
}
SHELL_MATERIALS = (
    "left_shell_material",
    "right_shell_material",
    "top_head_shell_material",
    "bottom_head_shell_material",
    "upper_leg_left_material",
    "upper_leg_right_material",
)


class FootballAppearance:
    """Tint the opponent's shell while preserving the learner's original look."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.material_ids: dict[str, np.ndarray] = {}
        self.original_rgba: dict[str, np.ndarray] = {}
        for side in ("home", "away"):
            ids = []
            for suffix in SHELL_MATERIALS:
                name = f"{side}/{suffix}"
                mat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, name)
                if mat_id < 0:
                    raise ValueError(f"football robot is missing visual material {name}")
                ids.append(mat_id)
            self.material_ids[side] = np.asarray(ids, dtype=np.int64)
            self.original_rgba[side] = model.mat_rgba[ids].copy()

    def set_roles(self, learner_home: bool, opponent_kind: str) -> None:
        """Reapply colors after an env-0 match selects a side and opponent."""
        for side in ("home", "away"):
            self.model.mat_rgba[self.material_ids[side]] = self.original_rgba[side]
        opponent_side = "away" if learner_home else "home"
        color = OPPONENT_COLORS[opponent_kind]
        self.model.mat_rgba[self.material_ids[opponent_side], :3] = color
