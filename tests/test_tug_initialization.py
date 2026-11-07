"""Actor-only ONNX initialization preserves inference and fresh learning state."""

from copy import deepcopy
import hashlib

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mjlab_microduck.tug_of_war.initialization import initialize_actor

pytest.importorskip("microduck_challenges.tug_of_war.opponent")


def make_actor(std_type="scalar"):
    return MLPModel(
        TensorDict({"actor": torch.zeros(1, 61)}, [1]), {"actor": ["actor"]}, "actor", 14,
        hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": std_type},
    )


@pytest.fixture
def source_path(tmp_path):
    actor = make_actor()
    with torch.no_grad():
        actor.obs_normalizer._mean.copy_(torch.linspace(-0.2, 0.2, 61).unsqueeze(0))
        actor.obs_normalizer._std.copy_(torch.linspace(0.1, 2.0, 61).unsqueeze(0))
        actor.obs_normalizer._var.copy_(actor.obs_normalizer._std.square())
    path = tmp_path / "plain.onnx"
    exported = actor.as_onnx(False)
    torch.onnx.export(exported, exported.get_dummy_inputs(), str(path), export_params=True,
                      opset_version=18, input_names=exported.input_names,
                      output_names=exported.output_names, dynamo=False)
    return path


@pytest.mark.parametrize("std_type", ["scalar", "log"])
def test_warm_start_matches_onnx_and_keeps_other_learning_state_fresh(source_path, std_type):
    actor, critic = make_actor(std_type), make_actor()
    critic_before = deepcopy(critic.state_dict())
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()])
    rng_before = torch.get_rng_state().clone()
    metadata = initialize_actor(actor, source_path, observation_count=123_456, action_std=0.25)
    torch.testing.assert_close(torch.get_rng_state(), rng_before)
    assert metadata["sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert metadata["observation_count_is_assumed"]
    assert metadata["parity_samples"] == 200
    assert actor.obs_normalizer.count.item() == 123_456
    torch.testing.assert_close(actor.obs_normalizer._var, actor.obs_normalizer._std.square())
    for key, value in critic_before.items():
        torch.testing.assert_close(critic.state_dict()[key], value)
    assert not optimizer.state_dict()["state"]

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(source_path), sess_options=options, providers=["CPUExecutionProvider"])
    observations = torch.randn(200, 61)
    expected = np.concatenate([session.run(None, {"obs": row.numpy()})[0] for row in observations.split(1)])
    with torch.no_grad():
        actual = actor(TensorDict({"actor": observations}, [200]))
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-4, atol=1e-5)
    actor.distribution.update(actual)
    torch.testing.assert_close(actor.distribution.std, torch.full_like(actual, 0.25))


def test_warm_start_refuses_lookup_graph_and_nonpositive_scale(source_path, tmp_path):
    graph = onnx.load(source_path)
    graph.graph.node.append(onnx.helper.make_node("Where", ["mask", "lookup", "actions"], ["override"]))
    lookup = tmp_path / "lookup.onnx"
    onnx.save(graph, lookup)
    with pytest.raises(ValueError, match="nodes"):
        initialize_actor(make_actor(), lookup)

    graph = onnx.load(source_path)
    name = next(node for node in graph.graph.node if node.op_type == "Div").input[1]
    initializer = next(value for value in graph.graph.initializer if value.name == name)
    initializer.CopyFrom(onnx.numpy_helper.from_array(np.full((1, 61), 0.001, dtype=np.float32), name))
    invalid = tmp_path / "invalid.onnx"
    onnx.save(graph, invalid)
    with pytest.raises(ValueError, match="normalization scales"):
        initialize_actor(make_actor(), invalid)
