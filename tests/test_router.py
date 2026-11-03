import numpy as np
import pytest

from src.dispatch.router import Router


def test_top_k_selection():
    router = Router(num_experts=4, top_k=2)

    tokens = np.array([[1.0, 2.0]])

    router_weights = np.array([
        [1.0, 0.5, 2.0, 0.2],
        [0.5, 1.0, 0.5, 0.1]
    ])

    selected, weights, probabilities = router.route(
        tokens, router_weights
    )

    assert selected.shape == (1, 2)
    assert selected[0].tolist() == [2, 1]


def test_weights_sum_to_one():
    router = Router(num_experts=4, top_k=2)

    tokens = np.array([[1.0, 2.0]])
    router_weights = np.ones((2, 4))

    selected, weights, probabilities = router.route(
        tokens, router_weights
    )

    assert np.isclose(np.sum(weights[0]), 1.0)


def test_multiple_tokens():
    router = Router(num_experts=4, top_k=2)

    tokens = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 1.0]
    ])

    router_weights = np.ones((2, 4))

    selected, weights, probabilities = router.route(
        tokens, router_weights
    )

    assert selected.shape == (3, 2)
    assert weights.shape == (3, 2)
    assert probabilities.shape == (3, 4)


def test_invalid_top_k():
    with pytest.raises(ValueError):
        Router(num_experts=4, top_k=5)


def test_deterministic_tie_breaking():
    router = Router(num_experts=4, top_k=2)

    tokens = np.array([[1.0, 1.0]])
    router_weights = np.ones((2, 4))

    selected, weights, probabilities = router.route(
        tokens, router_weights
    )

    assert selected[0].tolist() == [0, 1]