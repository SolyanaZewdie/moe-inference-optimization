class LFUCachePolicy:
    def __init__(self):
        self.frequencies = {}
        self.last_access = {}
        self.counter = 0

    def on_access(self, expert_id):
        if expert_id not in self.frequencies:
            return

        self.counter += 1
        self.frequencies[expert_id] += 1
        self.last_access[expert_id] = self.counter

    def on_insert(self, expert_id):
        self.counter += 1
        self.frequencies[expert_id] = 1
        self.last_access[expert_id] = self.counter

    def on_evict(self, expert_id):
        self.frequencies.pop(expert_id, None)
        self.last_access.pop(expert_id, None)

    def choose_victim(self, cached_experts, protected_experts=None):
        if protected_experts is None:
            protected_experts = set()

        candidates = []

        for expert_id in cached_experts:
            if expert_id not in protected_experts:
                frequency = self.frequencies.get(expert_id, 0)
                last_used = self.last_access.get(expert_id, 0)

                candidates.append((frequency, last_used, str(expert_id), expert_id))

        if not candidates:
            raise RuntimeError("No expert available for eviction")

        candidates.sort()

        return candidates[0][3]