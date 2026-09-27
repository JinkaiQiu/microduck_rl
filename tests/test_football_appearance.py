"""Football video colors follow policy identity through side swaps."""

import numpy as np

from mjlab.scene import Scene
from mjlab_microduck.football.appearance import FootballAppearance, OPPONENT_COLORS
from mjlab_microduck.football.arena import make_football_env_cfg


def test_opponent_palette_preserves_learner_materials_after_side_swap():
    scene = Scene(make_football_env_cfg(2).scene, "cpu")
    model = scene.compile()
    appearance = FootballAppearance(model)
    original = {side: model.mat_rgba[appearance.material_ids[side]].copy()
                for side in ("home", "away")}

    for learner_home, kind, opponent_side in (
        (True, "scripted", "away"),
        (False, "tactical", "home"),
        (True, "joint", "away"),
    ):
        appearance.set_roles(learner_home, kind)
        learner_side = "away" if opponent_side == "home" else "home"
        np.testing.assert_allclose(model.mat_rgba[appearance.material_ids[learner_side]],
                                   original[learner_side])
        tinted = model.mat_rgba[appearance.material_ids[opponent_side]]
        np.testing.assert_allclose(tinted[:, :3],
                                   np.broadcast_to(OPPONENT_COLORS[kind], tinted[:, :3].shape))
        np.testing.assert_allclose(tinted[:, 3], original[opponent_side][:, 3])
