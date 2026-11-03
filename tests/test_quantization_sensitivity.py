import numpy as np

from src.memory.sensitivity import (
    measure_weight_sensitivity,
    analyze_weight_sensitivity,
    assign_mixed_precision
)


def test_int4_sensitivity_measurement():
    rng = np.random.default_rng(42)

    weights = rng.normal(size=(16, 8)).astype(np.float32)
    inputs = rng.normal(size=(5, 16)).astype(np.float32)

    result = measure_weight_sensitivity(
        weights,
        inputs,
        bits=4,
        block_size=32
    )

    assert result["mse"] >= 0
    assert result["relative_error"] >= 0


def test_int8_has_lower_error_than_int4():
    rng = np.random.default_rng(42)

    weights = rng.normal(size=(32, 16)).astype(np.float32)
    inputs = rng.normal(size=(10, 32)).astype(np.float32)

    int4 = measure_weight_sensitivity(weights, inputs, bits=4)
    int8 = measure_weight_sensitivity(weights, inputs, bits=8)

    assert int8["mse"] <= int4["mse"]


def test_analyze_multiple_experts():
    rng = np.random.default_rng(42)

    weights = {
        "expert_0": rng.normal(size=(32, 16)).astype(np.float32),
        "expert_1": rng.normal(size=(32, 16)).astype(np.float32)
    }

    inputs = rng.normal(size=(10, 32)).astype(np.float32)

    results = analyze_weight_sensitivity(weights, inputs)

    assert "expert_0" in results
    assert "expert_1" in results

    assert "int4" in results["expert_0"]
    assert "int8" in results["expert_0"]


def test_mixed_precision_assignment():
    measurements = {
        "expert_0": {
            "int4": {"relative_error": 0.002},
            "int8": {"relative_error": 0.0001}
        },
        "expert_1": {
            "int4": {"relative_error": 0.05},
            "int8": {"relative_error": 0.001}
        },
        "expert_2": {
            "int4": {"relative_error": 0.08},
            "int8": {"relative_error": 0.03}
        }
    }

    assignments = assign_mixed_precision(
        measurements,
        error_tolerance=0.01
    )

    assert assignments == {
        "expert_0": 4,
        "expert_1": 8,
        "expert_2": 32
    }


def test_invalid_weight_dimensions():
    weights = np.ones(5)
    inputs = np.ones((2, 5))

    try:
        measure_weight_sensitivity(weights, inputs, bits=4)
        assert False
    except ValueError:
        assert True


def test_invalid_input_dimensions():
    weights = np.ones((5, 4))
    inputs = np.ones(5)

    try:
        measure_weight_sensitivity(weights, inputs, bits=4)
        assert False
    except ValueError:
        assert True