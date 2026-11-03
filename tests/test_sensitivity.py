import numpy as np
import pytest

from src.memory.sensitivity import (
    calculate_error,
    analyze_sensitivity,
    assign_precision
)


def test_identical_outputs_have_zero_error():
    reference = np.array([1.0, 2.0, 3.0])
    
    result = calculate_error(reference, reference)

    assert result["mse"] == 0.0
    assert result["relative_error"] == 0.0


def test_calculate_mse():
    reference = np.array([1.0, 2.0])
    quantized = np.array([2.0, 2.0])

    result = calculate_error(reference, quantized)

    assert result["mse"] == 0.5


def test_relative_error():
    reference = np.array([3.0, 4.0])
    quantized = np.array([3.0, 3.0])

    result = calculate_error(reference, quantized)

    assert result["relative_error"] == pytest.approx(0.2)


def test_analyze_multiple_components():
    reference = {
        "expert_0": np.array([1.0, 2.0]),
        "expert_1": np.array([3.0, 4.0])
    }

    quantized = {
        "expert_0": np.array([1.0, 2.0]),
        "expert_1": np.array([3.0, 3.0])
    }

    results = analyze_sensitivity(reference, quantized)

    assert results["expert_0"]["mse"] == 0.0
    assert results["expert_1"]["mse"] == 0.5


def test_assign_mixed_precision():
    results = {
        "expert_0": {"relative_error": 0.002},
        "expert_1": {"relative_error": 0.035},
        "expert_2": {"relative_error": 0.008}
    }

    assignments = assign_precision(results, error_tolerance=0.01)

    assert assignments == {
        "expert_0": 4,
        "expert_1": 8,
        "expert_2": 4
    }


def test_zero_reference_with_nonzero_output():
    reference = np.zeros(3)
    quantized = np.ones(3)

    result = calculate_error(reference, quantized)

    assert result["relative_error"] == float("inf")


def test_mismatched_shapes():
    reference = np.array([1.0, 2.0])
    quantized = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        calculate_error(reference, quantized)


def test_mismatched_component_names():
    reference = {"expert_0": np.array([1.0])}
    quantized = {"expert_1": np.array([1.0])}

    with pytest.raises(ValueError):
        analyze_sensitivity(reference, quantized)


def test_negative_tolerance():
    with pytest.raises(ValueError):
        assign_precision({}, error_tolerance=-0.1)


def test_non_finite_output():
    reference = np.array([1.0, np.nan])
    quantized = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        calculate_error(reference, quantized)