import numpy as np
import pytest

from src.model.toy_moe import ToyMoE, MoEConfig


@pytest.fixture
def model():
    config = MoEConfig()
    return ToyMoE(config)


def test_single_sequence_output_shape(model):
    token_ids = np.array([10, 20, 30])

    logits = model.forward(token_ids)

    assert logits.shape == (3, model.config.vocab_size)


def test_batch_output_shape(model):
    token_ids = np.array([
        [10, 20, 30],
        [40, 50, 60]
    ])

    logits = model.forward(token_ids)

    assert logits.shape == (
        2,
        3,
        model.config.vocab_size
    )


def test_output_contains_finite_values(model):
    token_ids = np.array([10, 20, 30])

    logits = model.forward(token_ids)

    assert np.all(np.isfinite(logits))


def test_forward_returns_routing_information(model):
    token_ids = np.array([10, 20, 30])

    logits, routing_information = model.forward(
        token_ids,
        return_routing=True
    )

    assert len(routing_information) == model.config.num_layers

    for layer_info in routing_information:
        assert layer_info["selected_experts"].shape == (
            3,
            model.config.top_k
        )


def test_selected_experts_are_valid(model):
    token_ids = np.array([10, 20, 30])

    _, routing_information = model.forward(
        token_ids,
        return_routing=True
    )

    for layer_info in routing_information:
        selected_experts = layer_info["selected_experts"]

        assert np.all(selected_experts >= 0)
        assert np.all(
            selected_experts < model.config.num_experts
        )


def test_selected_weights_sum_to_one(model):
    token_ids = np.array([10, 20, 30])

    _, routing_information = model.forward(
        token_ids,
        return_routing=True
    )

    for layer_info in routing_information:
        selected_weights = layer_info["selected_weights"]

        weight_sums = np.sum(selected_weights, axis=1)

        assert np.allclose(weight_sums, 1.0)


def test_invalid_token_id_raises_error(model):
    token_ids = np.array([10, 20, model.config.vocab_size])

    with pytest.raises(ValueError):
        model.forward(token_ids)


def test_empty_sequence_raises_error(model):
    token_ids = np.array([])

    with pytest.raises(ValueError):
        model.forward(token_ids)


def test_invalid_input_dimensions_raise_error(model):
    token_ids = np.array([[[10, 20, 30]]])

    with pytest.raises(ValueError):
        model.forward(token_ids)


def test_reproducible_forward_pass():
    config = MoEConfig()

    model_a = ToyMoE(config)
    model_b = ToyMoE(config)

    token_ids = np.array([10, 20, 30])

    output_a = model_a.forward(token_ids)
    output_b = model_b.forward(token_ids)

    assert np.allclose(output_a, output_b)
def test_causal_attention_blocks_future_tokens(model):
    sequence_a = np.array([10, 20, 30])
    sequence_b = np.array([10, 20, 50])

    logits_a = model.forward(sequence_a)
    logits_b = model.forward(sequence_b)

    assert np.allclose(logits_a[:2], logits_b[:2], atol=1e-5)
def test_batch_output_matches_individual_output(model):
    sequence_a = np.array([10, 20, 30])
    sequence_b = np.array([40, 50, 60])

    batch = np.array([
        sequence_a,
        sequence_b
    ])

    batch_output = model.forward(batch)

    individual_output = model.forward(sequence_a)

    assert np.allclose(batch_output[0], individual_output, atol=1e-5)
def test_generate_returns_correct_sequence_length(model):
    token_ids = np.array([10, 20, 30])

    generated = model.generate(
        token_ids,
        max_new_tokens=5
    )

    assert len(generated) == 8


def test_generate_preserves_original_prompt(model):
    token_ids = np.array([10, 20, 30])

    generated = model.generate(
        token_ids,
        max_new_tokens=5
    )

    assert np.array_equal(generated[:3], token_ids)


def test_generate_returns_valid_token_ids(model):
    token_ids = np.array([10, 20, 30])

    generated = model.generate(
        token_ids,
        max_new_tokens=5
    )

    assert np.all(generated >= 0)
    assert np.all(generated < model.config.vocab_size)


def test_generate_zero_new_tokens(model):
    token_ids = np.array([10, 20, 30])

    generated = model.generate(
        token_ids,
        max_new_tokens=0
    )

    assert np.array_equal(generated, token_ids)


def test_generate_rejects_empty_input(model):
    with pytest.raises(ValueError):
        model.generate(np.array([]))


def test_generate_rejects_negative_token_count(model):
    token_ids = np.array([10, 20, 30])

    with pytest.raises(ValueError):
        model.generate(token_ids, max_new_tokens=-1)
def test_unlimited_capacity_drops_no_assignments():
    config = MoEConfig(capacity_factor=None)
    model = ToyMoE(config)

    token_ids = np.array([10, 20, 30])

    _, routing_information = model.forward(
        token_ids,
        return_routing=True
    )

    for layer_info in routing_information:
        assert layer_info["dropped_assignments"] == 0
        assert layer_info["capacity_factor"] is None
def test_restricted_capacity_drops_excess_assignments():
    config = MoEConfig(capacity_factor=0.5)
    model = ToyMoE(config)

    # Force every token to select experts 0 and 1.
    # This creates predictable expert overload.
    for layer in model.weights["layers"]:
        layer["router"][:] = 0

    token_ids = np.array([10, 20, 30])

    _, routing_information = model.forward(
        token_ids,
        return_routing=True
    )

    for layer_info in routing_information:
        assert layer_info["capacity_factor"] == 0.5
        assert layer_info["dropped_assignments"] > 0