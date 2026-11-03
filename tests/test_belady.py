from src.cache.belady import BeladyCachePolicy


def test_belady_evicts_farthest_future_use():
    future = ["D", "A", "C", "B"]

    policy = BeladyCachePolicy(future)

    victim = policy.choose_victim({"A", "B", "C"})

    assert victim == "B"


def test_belady_evicts_expert_never_used_again():
    future = ["A", "C", "D"]

    policy = BeladyCachePolicy(future)

    victim = policy.choose_victim({"A", "B", "C"})

    assert victim == "B"


def test_belady_protects_experts():
    future = ["A", "B", "C"]

    policy = BeladyCachePolicy(future)

    victim = policy.choose_victim(
        {"A", "B"},
        protected_experts={"A"}
    )

    assert victim == "B"


def test_belady_raises_when_every_expert_is_protected():
    future = ["A", "B"]

    policy = BeladyCachePolicy(future)

    try:
        policy.choose_victim(
            {"A", "B"},
            protected_experts={"A", "B"}
        )
        assert False
    except RuntimeError:
        assert True