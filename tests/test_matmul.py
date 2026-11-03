import numpy as np
import pytest

from src.compute.naive_matmul import naive_matmul
from src.compute.tiled_matmul import tiled_matmul
from src.compute.gpu_simulator import GPUSimulator


def test_naive_matmul():
    A = np.array([
        [1, 2],
        [3, 4]
    ], dtype=np.float32)

    B = np.array([
        [5, 6],
        [7, 8]
    ], dtype=np.float32)

    result = naive_matmul(A, B)

    expected = np.array([
        [19, 22],
        [43, 50]
    ], dtype=np.float32)

    assert np.allclose(result, expected)


def test_tiled_matches_naive():
    rng = np.random.default_rng(42)

    A = rng.standard_normal((32, 32)).astype(np.float32)
    B = rng.standard_normal((32, 32)).astype(np.float32)

    naive_result = naive_matmul(A, B)
    tiled_result = tiled_matmul(A, B, tile_size=8)

    assert np.allclose(
        naive_result,
        tiled_result,
        atol=1e-4
    )


def test_non_multiple_dimensions():
    rng = np.random.default_rng(42)

    A = rng.standard_normal((17, 13)).astype(np.float32)
    B = rng.standard_normal((13, 19)).astype(np.float32)

    naive_result = naive_matmul(A, B)
    tiled_result = tiled_matmul(A, B, tile_size=8)

    assert np.allclose(
        naive_result,
        tiled_result,
        atol=1e-4
    )


def test_incompatible_dimensions():
    A = np.ones((3, 4), dtype=np.float32)
    B = np.ones((5, 2), dtype=np.float32)

    with pytest.raises(ValueError):
        naive_matmul(A, B)

    with pytest.raises(ValueError):
        tiled_matmul(A, B)


def test_invalid_tile_size():
    A = np.ones((4, 4), dtype=np.float32)
    B = np.ones((4, 4), dtype=np.float32)

    with pytest.raises(ValueError):
        tiled_matmul(A, B, tile_size=0)


def test_shared_memory_usage():
    A = np.ones((32, 32), dtype=np.float32)
    B = np.ones((32, 32), dtype=np.float32)

    simulator = GPUSimulator(shared_memory_limit=48 * 1024)

    tiled_matmul(A, B, tile_size=16, simulator=simulator)

    assert simulator.shared_memory_used == 2 * 16 * 16 * 4


def test_shared_memory_limit():
    A = np.ones((32, 32), dtype=np.float32)
    B = np.ones((32, 32), dtype=np.float32)

    simulator = GPUSimulator(shared_memory_limit=100)

    with pytest.raises(MemoryError):
        tiled_matmul(A, B, tile_size=16, simulator=simulator)