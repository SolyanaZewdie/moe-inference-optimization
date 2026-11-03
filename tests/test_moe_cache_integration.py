import numpy as np

from src.model.toy_moe import ToyMoE, MoEConfig
from src.cache.lru import LRUCachePolicy
from src.cache.belady import BeladyCachePolicy
from src.dispatch.routing_traces import create_routing_trace


def create_model():
    config = MoEConfig(
        vocab_size=128,
        d_model=8,
        hidden_dim=16,
        num_layers=2,
        num_experts=4,
        top_k=2,
        seed=42
    )

    return ToyMoE(config)


def calculate_expert_size(model):
    return (
        3 * model.config.d_model
        * model.config.hidden_dim * 4
    )


def test_mmap_cache_integration(tmp_path):
    model = create_model()

    expert_size = calculate_expert_size(model)
    memory_limit = expert_size * 4

    expert_count = model.enable_mmap_expert_cache(
        tmp_path / "model_weights",
        memory_limit,
        LRUCachePolicy()
    )

    assert expert_count == 8

    assert isinstance(
        model.weights["layers"][0]["experts"][0]["W_gate"],
        np.memmap
    )

    logits = model.forward(np.array([10]))

    assert logits.shape == (1, 128)
    assert np.all(np.isfinite(logits))

    stats = model.expert_cache.get_stats()

    assert stats["misses"] > 0
    assert stats["current_memory_bytes"] <= stats["memory_limit_bytes"]
    assert stats["memory_limit_bytes"] < expert_size * 8


def test_cache_reuses_experts(tmp_path):
    model = create_model()

    expert_size = calculate_expert_size(model)

    model.enable_mmap_expert_cache(
        tmp_path / "model_weights",
        expert_size * 4,
        LRUCachePolicy()
    )

    token_ids = np.array([10])

    first_output = model.forward(token_ids)
    second_output = model.forward(token_ids)

    assert np.allclose(first_output, second_output)
    assert model.expert_cache.get_stats()["hits"] > 0


def test_cached_experts_are_independent_of_mmap_sources(tmp_path):
    model = create_model()

    expert_size = calculate_expert_size(model)

    model.enable_mmap_expert_cache(
        tmp_path / "model_weights",
        expert_size * 4,
        LRUCachePolicy()
    )

    expert_id = "layer_0_expert_0"

    cached = model.expert_cache.get(
        expert_id,
        model.load_expert_from_mmap
    )

    source = model.mmap_expert_sources[expert_id]

    assert not np.shares_memory(
        cached["W_gate"],
        source["W_gate"]
    )


def test_inference_works_when_selected_experts_exceed_cache_capacity(
    tmp_path
):
    model = create_model()

    expert_size = calculate_expert_size(model)

    model.enable_mmap_expert_cache(
        tmp_path / "small_cache",
        expert_size,
        LRUCachePolicy()
    )

    token_ids = np.array([10, 20, 30, 40, 50, 60])

    logits, routing = model.forward(
        token_ids,
        return_routing=True
    )

    assert logits.shape == (6, 128)
    assert np.all(np.isfinite(logits))

    assert any(
        len(np.unique(layer["selected_experts"])) > 1
        for layer in routing
    )

    stats = model.expert_cache.get_stats()

    assert stats["misses"] > 1
    assert stats["current_memory_bytes"] <= expert_size


def test_belady_cache_replays_routing_trace(tmp_path):
    model = create_model()

    token_ids = np.array([10, 20, 30, 40, 50, 60])

    # Pass 1: Run without caching and record routing decisions.
    reference_logits, routing = model.forward(
        token_ids,
        return_routing=True
    )

    layer_decisions = [
        layer["selected_experts"]
        for layer in routing
    ]

    # Construct the exact expert-access sequence.
    trace = create_routing_trace(layer_decisions)

    assert len(trace) > 0

    # Pass 2: Enable mmap and Belady caching.
    expert_size = calculate_expert_size(model)

    policy = BeladyCachePolicy(trace)

    model.enable_mmap_expert_cache(
        tmp_path / "belady_cache",
        expert_size,
        policy
    )

    cached_logits = model.forward(token_ids)

    # Verify numerical correctness.
    assert np.allclose(
        reference_logits,
        cached_logits,
        rtol=1e-5,
        atol=1e-6
    )

    assert np.all(np.isfinite(cached_logits))

    # Verify that every trace request was consumed.
    assert policy.current_position == len(trace)

    stats = model.expert_cache.get_stats()

    # Each unique selected expert is requested once per layer.
    assert stats["misses"] == len(trace)

    # A one-expert cache must evict previous experts.
    assert stats["evictions"] == len(trace) - 1

    assert stats["current_memory_bytes"] <= expert_size