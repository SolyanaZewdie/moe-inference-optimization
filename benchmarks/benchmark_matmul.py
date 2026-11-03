import sys
from pathlib import Path
import time
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.compute.gpu_simulator import GPUSimulator
from src.compute.naive_matmul import naive_matmul
from src.compute.tiled_matmul import tiled_matmul


def benchmark_naive(A, B, repeats=3):
    """
    Measure naive matrix multiplication execution time.

    Simulator is disabled during timing.
    """

    times = []

    for _ in range(repeats):
        start = time.perf_counter()

        result = naive_matmul(A, B)

        end = time.perf_counter()

        times.append(end - start)

    return result, np.median(times)


def benchmark_tiled(A, B, tile_size, repeats=3):
    """
    Measure tiled matrix multiplication execution time.

    Simulator is disabled during timing.
    """

    times = []

    for _ in range(repeats):
        start = time.perf_counter()

        result = tiled_matmul(
            A,
            B,
            tile_size=tile_size
        )

        end = time.perf_counter()

        times.append(end - start)

    return result, np.median(times)


def main():

    # Reproducible experiment.
    rng = np.random.default_rng(42)

    matrix_size = 64

    A = rng.standard_normal(
        (matrix_size, matrix_size)
    ).astype(np.float32)

    B = rng.standard_normal(
        (matrix_size, matrix_size)
    ).astype(np.float32)

    tile_sizes = [8, 16, 32, 64]

    print("=" * 65)
    print("MATRIX MULTIPLICATION BENCHMARK")
    print("=" * 65)

    print(f"Matrix A shape: {A.shape}")
    print(f"Matrix B shape: {B.shape}")
    print(f"Data type: {A.dtype}")
    print(f"Tile sizes: {tile_sizes}")

    # --------------------------------------------------
    # 1. Naive baseline
    # --------------------------------------------------

    naive_simulator = GPUSimulator()

    naive_output = naive_matmul(
        A,
        B,
        simulator=naive_simulator
    )

    naive_transactions = (
        naive_simulator.get_stats()["global_memory_transactions"]
    )

    _, naive_time = benchmark_naive(A, B)

    print("\nNAIVE IMPLEMENTATION")
    print("-" * 65)

    print(f"Global memory transactions: {naive_transactions}")
    print(f"Median execution time: {naive_time:.6f} seconds")

    # --------------------------------------------------
    # 2. Tiled implementations
    # --------------------------------------------------

    print("\nTILED IMPLEMENTATIONS")
    print("-" * 65)

    print(
        f"{'Tile':<10}"
        f"{'Transactions':<18}"
        f"{'Reduction':<15}"
        f"{'Time (s)':<12}"
        f"{'Correct':<10}"
    )

    print("-" * 65)

    for tile_size in tile_sizes:

        simulator = GPUSimulator()

        # Count simulated memory transactions.
        tiled_output = tiled_matmul(
            A,
            B,
            tile_size=tile_size,
            simulator=simulator
        )

        stats = simulator.get_stats()

        transactions = stats["global_memory_transactions"]

        # Calculate transaction reduction.
        reduction = (
            1 - transactions / naive_transactions
        ) * 100

        # Measure execution time separately.
        _, tiled_time = benchmark_tiled(
            A,
            B,
            tile_size
        )

        # Check numerical correctness.
        correct = np.allclose(
            naive_output,
            tiled_output,
            rtol=1e-4,
            atol=1e-4
        )

        print(
            f"{tile_size:<10}"
            f"{transactions:<18}"
            f"{reduction:>8.2f}%       "
            f"{tiled_time:<12.6f}"
            f"{str(correct):<10}"
        )

        print(
            f"  Shared memory: "
            f"{stats['shared_memory_used']} / "
            f"{stats['shared_memory_limit']} bytes"
        )

    print("\n" + "=" * 65)
    print("BENCHMARK COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()