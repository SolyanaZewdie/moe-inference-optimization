import numpy as np
import pytest

from src.dispatch.grouped_matmul import grouped_expert_forward


def run_single_expert(token, expert):
    """Reference implementation for one token and one expert."""

    gate = token @ expert["W_gate"]
    up = token @ expert["W_up"]

    gate = gate / (1.0 + np.exp(-np.clip(gate, -60, 60)))

    hidden = gate * up

    return hidden @ expert["W_down"]


def run_reference(tokens, selected_experts, selected_weights, experts):
    """Process every token-expert assignment individually."""

    outputs = np.zeros(tokens.shape, dtype=np.float32)

    for token_index in range(len(tokens)):
        for position in range(selected_experts.shape[1]):

            expert_index = selected_experts[token_index, position]
            weight = selected_weights[token_index, position]

            expert_output = run_single_expert(
                tokens[token_index],
                experts[expert_index]
            )

            outputs[token_index] += weight * expert_output

    return outputs


def create_experts(num_experts, d_model, hidden_dim):
    """Create reproducible expert weights."""

    rng = np.random.default_rng(42)
    experts = []

    for _ in range(num_experts):
        expert = {
            "W_gate": rng.standard_normal(
                (d_model, hidden_dim)
            ).astype(np.float32) * 0.1,

            "W_up": rng.standard_normal(
                (d_model, hidden_dim)
            ).astype(np.float32) * 0.1,

            "W_down": rng.standard_normal(
                (hidden_dim, d_model)
            ).astype(np.float32) * 0.1
        }

        experts.append(expert)

    return experts


def test_grouped_matches_reference():
    rng = np.random.default_rng(10)

    tokens = rng.standard_normal((5, 4)).astype(np.float32)

    experts = create_experts(3, 4, 6)

    selected_experts = np.array([
        [0, 2],
        [1, 2],
        [0, 1],
        [2, 1],
        [1, 0]
    ])

    selected_weights = np.array([
        [0.7, 0.3],
        [0.6, 0.4],
        [0.8, 0.2],
        [0.5, 0.5],
        [0.4, 0.6]
    ], dtype=np.float32)

    grouped_output, dropped = grouped_expert_forward(
        tokens,
        selected_experts,
        selected_weights,
        experts
    )

    reference_output = run_reference(
        tokens,
        selected_experts,
        selected_weights,
        experts
    )

    assert np.allclose(grouped_output, reference_output, atol=1e-5)
    assert dropped == 0


def test_expert_receiving_no_tokens():
    tokens = np.ones((2, 4), dtype=np.float32)
    experts = create_experts(3, 4, 6)

    selected_experts = np.array([
        [0, 1],
        [0, 1]
    ])

    selected_weights = np.full((2, 2), 0.5, dtype=np.float32)

    output, dropped = grouped_expert_forward(
        tokens,
        selected_experts,
        selected_weights,
        experts
    )

    assert output.shape == (2, 4)
    assert np.all(np.isfinite(output))
    assert dropped == 0


def test_all_tokens_select_same_experts():
    tokens = np.ones((4, 4), dtype=np.float32)
    experts = create_experts(2, 4, 6)

    selected_experts = np.array([
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1]
    ])

    selected_weights = np.full((4, 2), 0.5, dtype=np.float32)

    output, dropped = grouped_expert_forward(
        tokens,
        selected_experts,
        selected_weights,
        experts
    )

    assert output.shape == (4, 4)
    assert np.all(np.isfinite(output))
    assert dropped == 0


def test_invalid_expert_index():
    tokens = np.ones((1, 4), dtype=np.float32)
    experts = create_experts(2, 4, 6)

    selected_experts = np.array([[0, 3]])
    selected_weights = np.array([[0.5, 0.5]])

    with pytest.raises(ValueError):
        grouped_expert_forward(
            tokens,
            selected_experts,
            selected_weights,
            experts
        )


def test_empty_token_batch():
    tokens = np.empty((0, 4), dtype=np.float32)
    experts = create_experts(2, 4, 6)

    selected_experts = np.empty((0, 2), dtype=np.int64)
    selected_weights = np.empty((0, 2), dtype=np.float32)

    output, dropped = grouped_expert_forward(
        tokens,
        selected_experts,
        selected_weights,
        experts
    )

    assert output.shape == (0, 4)
    assert dropped == 0


def test_capacity_drops_excess_assignments():
    tokens = np.ones((4, 2), dtype=np.float32)

    selected_experts = np.zeros((4, 1), dtype=np.int64)
    selected_weights = np.ones((4, 1), dtype=np.float32)

    experts = create_experts(1, 2, 2)

    output, dropped = grouped_expert_forward(
        tokens,
        selected_experts,
        selected_weights,
        experts,
        capacity_factor=0.5
    )

    assert dropped == 2
    assert np.all(output[:2] != 0)
    assert np.all(output[2:] == 0)


def test_capacity_none_keeps_all_assignments():
    tokens = np.ones((4, 2), dtype=np.float32)

    selected_experts = np.zeros((4, 1), dtype=np.int64)
    selected_weights = np.ones((4, 1), dtype=np.float32)

    experts = create_experts(1, 2, 2)

    output, dropped = grouped_expert_forward(
        tokens,
        selected_experts,
        selected_weights,
        experts
    )

    assert dropped == 0
    assert np.all(output != 0)


def test_invalid_capacity_factor_raises_error():
    tokens = np.ones((2, 2), dtype=np.float32)

    selected_experts = np.zeros((2, 1), dtype=np.int64)
    selected_weights = np.ones((2, 1), dtype=np.float32)

    experts = create_experts(1, 2, 2)

    with pytest.raises(ValueError):
        grouped_expert_forward(
            tokens,
            selected_experts,
            selected_weights,
            experts,
            capacity_factor=0
        )

