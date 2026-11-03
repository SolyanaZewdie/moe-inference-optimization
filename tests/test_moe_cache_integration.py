import numpy as np

from src.model.toy_moe import ToyMoE, MoEConfig
from src.cache.lru import LRUCachePolicy


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


def test_mmap_cache_integration(tmp_path):
    model = create_model()

    expert_size = (
        3 * model.config.d_model
        * model.config.hidden_dim * 4
    )

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

    expert_size = (
        3 * model.config.d_model
        * model.config.hidden_dim * 4
    )

    model.enable_mmap_expert_cache(
        tmp_path / "model_weights",
        expert_size * 4,
        LRUCachePolicy()
    )

    token_ids = np.array([10])

    first_output = model.forward(token_ids)

    print("\nAFTER FIRST FORWARD:")
    print(model.expert_cache.get_stats())
    print("Cached experts:", list(model.expert_cache.cached_experts.keys()))

    second_output = model.forward(token_ids)

    print("\nAFTER SECOND FORWARD:")
    print(model.expert_cache.get_stats())
    print("Cached experts:", list(model.expert_cache.cached_experts.keys()))

    print("\nOUTPUTS IDENTICAL:")
    print(np.allclose(first_output, second_output))

    assert model.expert_cache.get_stats()["hits"] > 0

def test_cached_experts_are_independent_of_mmap_sources(tmp_path):
    model = create_model()

    expert_size = (
        3 * model.config.d_model
        * model.config.hidden_dim * 4
    )

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