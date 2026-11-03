import numpy as np
import pytest

from src.cache.expert_cache import ExpertCache
from src.cache.lru import LRUCachePolicy
from src.cache.lfu import LFUCachePolicy


def create_loader():
    experts = {
        "A": {"weights": np.ones((4, 4), dtype=np.float32)},
        "B": {"weights": np.ones((4, 4), dtype=np.float32) * 2},
        "C": {"weights": np.ones((4, 4), dtype=np.float32) * 3},
    }

    def loader(expert_id):
        return experts[expert_id]

    return loader


def test_cache_hit():
    cache = ExpertCache(1024, LRUCachePolicy())
    loader = create_loader()

    first = cache.get("A", loader)
    second = cache.get("A", loader)

    assert first is second
    assert cache.hits == 1
    assert cache.misses == 1


def test_cache_miss():
    cache = ExpertCache(1024, LRUCachePolicy())
    loader = create_loader()

    cache.get("A", loader)

    assert cache.misses == 1
    assert cache.hits == 0


def test_cache_evicts_when_full():
    cache = ExpertCache(128, LRUCachePolicy())
    loader = create_loader()

    cache.get("A", loader)
    cache.get("B", loader)

    # A is least recently used.
    cache.get("C", loader)

    assert "A" not in cache.cached_experts
    assert "B" in cache.cached_experts
    assert "C" in cache.cached_experts
    assert cache.evictions == 1


def test_cache_respects_memory_limit():
    cache = ExpertCache(128, LRUCachePolicy())
    loader = create_loader()

    cache.get("A", loader)
    cache.get("B", loader)
    cache.get("C", loader)

    assert cache.current_memory_bytes <= cache.memory_limit_bytes


def test_cache_protects_experts():
    cache = ExpertCache(128, LRUCachePolicy())
    loader = create_loader()

    cache.get("A", loader)
    cache.get("B", loader)

    # A cannot be evicted, so B must be evicted.
    cache.get("C", loader, protected_experts={"A"})

    assert "A" in cache.cached_experts
    assert "B" not in cache.cached_experts
    assert "C" in cache.cached_experts


def test_cache_fails_when_all_experts_are_protected():
    cache = ExpertCache(128, LRUCachePolicy())
    loader = create_loader()

    cache.get("A", loader)
    cache.get("B", loader)

    with pytest.raises(RuntimeError):
        cache.get(
            "C",
            loader,
            protected_experts={"A", "B"}
        )


def test_expert_larger_than_cache():
    cache = ExpertCache(32, LRUCachePolicy())
    loader = create_loader()

    with pytest.raises(MemoryError):
        cache.get("A", loader)


def test_cache_statistics():
    cache = ExpertCache(1024, LFUCachePolicy())
    loader = create_loader()

    cache.get("A", loader)
    cache.get("A", loader)
    cache.get("B", loader)

    stats = cache.get_stats()

    assert stats["hits"] == 1
    assert stats["misses"] == 2
    assert stats["hit_rate"] == 1 / 3