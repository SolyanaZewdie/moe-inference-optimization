from src.cache.belady import BeladyCachePolicy


def test_belady_evicts_expert_used_farthest_in_future():
    policy = BeladyCachePolicy(["A", "B", "C", "A", "B"])

    policy.on_access("A")
    policy.on_access("B")

    victim = policy.choose_victim({"A", "B"})

    assert victim == "B"


def test_belady_evicts_expert_never_used_again():
    policy = BeladyCachePolicy(["A", "B", "C", "A"])

    policy.on_access("A")
    policy.on_access("B")

    victim = policy.choose_victim({"A", "B"})

    assert victim == "B"


def test_belady_protects_experts():
    policy = BeladyCachePolicy(["A", "B", "C"])

    policy.on_access("A")
    policy.on_access("B")

    victim = policy.choose_victim(
        {"A", "B"},
        protected_experts={"B"}
    )

    assert victim == "A"


def test_belady_detects_trace_mismatch():
    policy = BeladyCachePolicy(["A", "B"])

    import pytest

    with pytest.raises(ValueError):
        policy.on_access("B")