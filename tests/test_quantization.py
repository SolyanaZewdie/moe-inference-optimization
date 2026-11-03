import numpy as np
import pytest

from src.memory.quantization import (
    quantize_blockwise,
    dequantize_blockwise,
    pack_int4,
    unpack_int4
)


def test_int8_quantization():
    weights = np.array(
        [0.12, -0.50, 0.33, 0.05],
        dtype=np.float32
    )

    quantized, scales = quantize_blockwise(
        weights,
        bits=8,
        block_size=4
    )

    assert quantized.dtype == np.int8
    assert np.max(np.abs(quantized)) <= 127
    assert len(scales) == 1


def test_int4_quantization():
    weights = np.array(
        [0.12, -0.50, 0.33, 0.05],
        dtype=np.float32
    )

    quantized, scales = quantize_blockwise(
        weights,
        bits=4,
        block_size=4
    )

    assert quantized.dtype == np.int8
    assert np.max(np.abs(quantized)) <= 7
    assert len(scales) == 1


def test_dequantization():
    weights = np.array(
        [0.12, -0.50, 0.33, 0.05],
        dtype=np.float32
    )

    quantized, scales = quantize_blockwise(
        weights,
        bits=8,
        block_size=4
    )

    reconstructed = dequantize_blockwise(
        quantized,
        scales,
        block_size=4
    )

    assert reconstructed.shape == weights.shape
    assert np.allclose(weights, reconstructed, atol=0.01)


def test_multiple_blocks():
    weights = np.arange(100, dtype=np.float32)

    quantized, scales = quantize_blockwise(
        weights,
        bits=8,
        block_size=32
    )

    assert len(scales) == 4
    assert quantized.shape == weights.shape


def test_zero_block():
    weights = np.zeros(32, dtype=np.float32)

    quantized, scales = quantize_blockwise(
        weights,
        bits=4,
        block_size=32
    )

    reconstructed = dequantize_blockwise(
        quantized,
        scales,
        block_size=32
    )

    assert np.all(quantized == 0)
    assert np.all(reconstructed == 0)


def test_int4_packing():
    values = np.array([3, -2, 7, -7], dtype=np.int8)

    packed, original_size = pack_int4(values)

    assert len(packed) == 2

    unpacked = unpack_int4(packed, original_size)

    assert np.array_equal(values, unpacked)


def test_int4_packing_odd_number():
    values = np.array([3, -2, 7], dtype=np.int8)

    packed, original_size = pack_int4(values)

    assert len(packed) == 2

    unpacked = unpack_int4(packed, original_size)

    assert np.array_equal(values, unpacked)


def test_invalid_bit_width():
    weights = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        quantize_blockwise(weights, bits=6)


def test_invalid_block_size():
    weights = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        quantize_blockwise(weights, block_size=0)


def test_non_finite_weights():
    weights = np.array([1.0, np.nan])

    with pytest.raises(ValueError):
        quantize_blockwise(weights)