import pytest

from src.compute.gpu_simulator import GPUSimulator


def test_coalesced_memory_access():
    simulator = GPUSimulator()

    addresses = [
        index * 4 for index in range(32)
    ]

    transactions = simulator.record_global_accesses(addresses)

    assert transactions == 4


def test_uncoalesced_memory_access():
    simulator = GPUSimulator()

    addresses = [
        index * 128 for index in range(32)
    ]

    transactions = simulator.record_global_accesses(addresses)

    assert transactions == 32


def test_shared_memory_allocation():
    simulator = GPUSimulator(
        shared_memory_limit=48 * 1024
    )

    assert simulator.allocate_shared(16 * 1024)

    assert simulator.shared_memory_used == 16 * 1024


def test_shared_memory_limit():
    simulator = GPUSimulator(
        shared_memory_limit=1024
    )

    with pytest.raises(MemoryError):
        simulator.allocate_shared(2048)


def test_thread_index():
    simulator = GPUSimulator()

    global_index = simulator.thread_index(
        block_index=2,
        thread_index=5
    )

    assert global_index == 2053


def test_thread_bounds():
    simulator = GPUSimulator()

    assert simulator.is_valid_thread(5, 10)
    assert not simulator.is_valid_thread(10, 10)


def test_statistics_and_reset():
    simulator = GPUSimulator()

    simulator.record_global_accesses([0, 4, 8, 12])

    stats = simulator.get_stats()

    assert stats["global_memory_transactions"] == 1

    simulator.reset()

    assert simulator.global_memory_transactions == 0


def test_negative_memory_address():
    simulator = GPUSimulator()

    with pytest.raises(ValueError):
        simulator.record_global_accesses([-1, 4])