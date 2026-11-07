"""Actor-only warm starts from plain Arena ONNX policies."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.modules.distribution import GaussianDistribution


def initialize_actor(actor: MLPModel, path: Path | str, *,
                     observation_count: int = 100_000_000, action_std: float = 0.3) -> dict:
    """Copy a normalized 61→14 ELU actor without loading any learning state.

    ONNX contains normalization scales, but no sample count or exploration noise.
    The explicit count is a prior weight for continued normalization updates, not
    a claim about the source run. The critic, optimizer, and counters stay fresh.
    """
    from microduck_challenges.tug_of_war.opponent import load_actor

    if isinstance(observation_count, bool) or not isinstance(observation_count, int) or observation_count < 1:
        raise ValueError("observation_count must be a positive integer")
    if not math.isfinite(action_std) or action_std <= 0:
        raise ValueError("action_std must be finite and positive")
    normalizer = actor.obs_normalizer
    if actor.obs_dim != 61 or not isinstance(normalizer, EmpiricalNormalization):
        raise ValueError("warm start requires a normalized 61D actor")
    if not isinstance(actor.distribution, GaussianDistribution):
        raise ValueError("warm start requires a Gaussian action distribution")

    path = Path(path).resolve()
    # The upstream loader constructs Linear modules before filling their weights.
    with torch.random.fork_rng(devices=[]):
        policy = load_actor(path)
    normalize, *layers = policy
    network = torch.nn.Sequential(*layers)
    source = network.state_dict()
    target = actor.mlp.state_dict()
    if (source.keys() != target.keys()
            or any(source[key].shape != target[key].shape for key in source)
            or tuple(type(layer) for layer in layers) != tuple(type(layer) for layer in actor.mlp)):
        raise ValueError("ONNX network architecture differs from the learner actor")
    std = normalize.std - normalizer.eps
    if (normalize.mean.shape != normalizer._mean.shape or std.shape != normalizer._std.shape
            or not torch.isfinite(normalize.mean).all() or not torch.isfinite(std).all()
            or not (std > 0).all() or any(not torch.isfinite(value).all() for value in source.values())):
        raise ValueError("ONNX actor has invalid weights or normalization scales")

    # Check the actual graph as well as the upstream parser's supported operators.
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    if (len(session.get_inputs()) != 1 or session.get_inputs()[0].shape != [1, 61]
            or len(session.get_outputs()) != 1 or session.get_outputs()[0].shape != [1, 14]):
        raise ValueError("ONNX policy must preserve the Arena (1, 61) → (1, 14) contract")
    observations = torch.randn(200, 61, generator=torch.Generator().manual_seed(0))
    observations = observations * normalize.std + normalize.mean
    expected = torch.from_numpy(np.concatenate([
        session.run(None, {session.get_inputs()[0].name: row.numpy()})[0]
        for row in observations.split(1)
    ]))
    with torch.no_grad():
        if not torch.allclose(policy(observations), expected, rtol=1e-4, atol=1e-5):
            raise ValueError("ONNX graph output differs from the supported plain neural actor")

    actor.mlp.load_state_dict(source)
    normalizer.load_state_dict({
        "_mean": normalize.mean, "_var": std.square(), "_std": std,
        "count": torch.tensor(observation_count, dtype=torch.long),
    })
    with torch.no_grad():
        distribution = actor.distribution
        if distribution.std_type == "scalar":
            distribution.std_param.fill_(action_std)
        else:
            distribution.log_std_param.fill_(math.log(action_std))
        actual = actor.as_onnx(False).cpu()(observations)
    if not torch.allclose(actual, expected, rtol=1e-4, atol=1e-5):
        raise ValueError("initialized actor output differs from the source ONNX")
    return {
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "observation_count": observation_count, "observation_count_is_assumed": True,
        "normalizer_updates": "enabled with explicit prior count; ONNX has no count",
        "action_std": action_std, "parity_samples": len(observations),
        "parity_max_abs_error": float((actual - expected).abs().max()),
    }
