class ExpertCache:
    def __init__(self, memory_limit_bytes, policy):
        if memory_limit_bytes <= 0:
            raise ValueError("Memory limit must be positive")

        self.memory_limit_bytes = memory_limit_bytes
        self.policy = policy

        self.cached_experts = {}
        self.expert_sizes = {}

        self.current_memory_bytes = 0

        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, expert_id, loader, protected_experts=None):
        if protected_experts is None:
            protected_experts = set()

        # Record the request in the replacement policy.
        self.policy.on_access(expert_id)

        # Cache hit.
        if expert_id in self.cached_experts:
            self.hits += 1
            return self.cached_experts[expert_id]

        # Cache miss.
        self.misses += 1

        expert_weights = loader(expert_id)

        if not isinstance(expert_weights, dict) or not expert_weights:
            raise ValueError("Expert weights must be a non-empty dictionary")

        expert_size = 0

        for weight in expert_weights.values():
            if not hasattr(weight, "nbytes"):
                raise ValueError("Expert weights must be NumPy-compatible arrays")

            expert_size += weight.nbytes

        if expert_size <= 0:
            raise ValueError("Expert must have a positive memory size")

        if expert_size > self.memory_limit_bytes:
            raise MemoryError("Expert is larger than the entire cache")

        # Evict until enough space is available.
                # Evict until enough space is available.
        while self.current_memory_bytes + expert_size > self.memory_limit_bytes:

            available_experts = (
                set(self.cached_experts) - protected_experts
            )

            if not available_experts:
                raise RuntimeError(
    "Not enough cache space: all cached experts are protected"
)

            victim = self.policy.choose_victim(
                set(self.cached_experts),
                protected_experts
            )

            if victim not in available_experts:
                raise RuntimeError(
                    "Cache policy selected an invalid eviction candidate"
                )

            self.remove(victim)

        # Insert the new expert.
        self.cached_experts[expert_id] = expert_weights
        self.expert_sizes[expert_id] = expert_size

        self.current_memory_bytes += expert_size

        self.policy.on_insert(expert_id)

        return expert_weights

    def remove(self, expert_id):
        if expert_id not in self.cached_experts:
            return

        self.current_memory_bytes -= self.expert_sizes[expert_id]

        del self.cached_experts[expert_id]
        del self.expert_sizes[expert_id]

        self.policy.on_evict(expert_id)

        self.evictions += 1

    def get_stats(self):
        total_requests = self.hits + self.misses

        hit_rate = (
            self.hits / total_requests
            if total_requests > 0
            else 0.0
        )

        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": hit_rate,
            "current_memory_bytes": self.current_memory_bytes,
            "memory_limit_bytes": self.memory_limit_bytes,
            "memory_utilization": (
                self.current_memory_bytes / self.memory_limit_bytes
            )
        }