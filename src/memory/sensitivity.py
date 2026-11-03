import numpy as np

from src.memory.quantization import (
    quantize_blockwise,
    dequantize_blockwise
)
def calculate_error(reference_output, quantized_output):
    reference_output = np.asarray(reference_output, dtype=np.float32)
    quantized_output = np.asarray(quantized_output, dtype=np.float32)

    if reference_output.shape != quantized_output.shape:
        raise ValueError("Output shapes must match")

    if not np.all(np.isfinite(reference_output)):
        raise ValueError("Reference output contains non-finite values")

    if not np.all(np.isfinite(quantized_output)):
        raise ValueError("Quantized output contains non-finite values")

    difference = reference_output - quantized_output

    mse = np.mean(difference ** 2)

    reference_norm = np.linalg.norm(reference_output)
    difference_norm = np.linalg.norm(difference)

    if reference_norm == 0:
        if difference_norm == 0:
            relative_error = 0.0
        else:
            relative_error = float("inf")
    else:
        relative_error = difference_norm / reference_norm

    return {
        "mse": float(mse),
        "relative_error": float(relative_error)
    }


def analyze_sensitivity(reference_outputs, quantized_outputs):
    if reference_outputs.keys() != quantized_outputs.keys():
        raise ValueError("Both dictionaries must contain the same keys")

    results = {}

    for component_name in reference_outputs:
        results[component_name] = calculate_error(
            reference_outputs[component_name],
            quantized_outputs[component_name]
        )

    return results


def assign_precision(sensitivity_results, error_tolerance):
    if error_tolerance < 0:
        raise ValueError("Error tolerance cannot be negative")

    precision_assignments = {}

    for component_name, metrics in sensitivity_results.items():
        if metrics["relative_error"] <= error_tolerance:
            precision_assignments[component_name] = 4
        else:
            precision_assignments[component_name] = 8

    return precision_assignments
def measure_weight_sensitivity(weights, inputs, bits, block_size=32):
    weights = np.asarray(weights, dtype=np.float32)
    inputs = np.asarray(inputs, dtype=np.float32)

    if weights.ndim != 2:
        raise ValueError("Weights must be a 2D matrix")

    if inputs.ndim != 2:
        raise ValueError("Inputs must be a 2D matrix")

    if inputs.shape[1] != weights.shape[0]:
        raise ValueError("Input and weight dimensions do not match")

    # Calculate output using original weights.
    reference_output = inputs @ weights

    # Quantize the weights.
    quantized_weights, scales = quantize_blockwise(
        weights,
        bits=bits,
        block_size=block_size
    )

    # Reconstruct floating-point weights.
    reconstructed_weights = dequantize_blockwise(
        quantized_weights,
        scales,
        block_size
    )

    reconstructed_weights = reconstructed_weights.reshape(weights.shape)

    # Calculate output using reconstructed weights.
    quantized_output = inputs @ reconstructed_weights

    # Measure the output difference.
    return calculate_error(reference_output, quantized_output)


def analyze_weight_sensitivity(weights_by_component, inputs, block_size=32):
    results = {}

    for component_name, weights in weights_by_component.items():

        int4_error = measure_weight_sensitivity(
            weights,
            inputs,
            bits=4,
            block_size=block_size
        )

        int8_error = measure_weight_sensitivity(
            weights,
            inputs,
            bits=8,
            block_size=block_size
        )

        results[component_name] = {
            "int4": int4_error,
            "int8": int8_error
        }

    return results


def assign_mixed_precision(measurements, error_tolerance):
    if error_tolerance < 0:
        raise ValueError("Error tolerance cannot be negative")

    assignments = {}

    for component_name, results in measurements.items():

        int4_error = results["int4"]["relative_error"]
        int8_error = results["int8"]["relative_error"]

        if int4_error <= error_tolerance:
            assignments[component_name] = 4

        elif int8_error <= error_tolerance:
            assignments[component_name] = 8

        else:
            # Preserve the original FP32 weights.
            assignments[component_name] = 32

    return assignments