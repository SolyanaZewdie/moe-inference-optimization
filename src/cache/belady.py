class BeladyCachePolicy:
    def __init__(self, future_requests):
        self.future_requests = list(future_requests)
        self.current_position = 0

    def on_access(self, expert_id):
        if self.current_position >= len(self.future_requests):
            raise ValueError("More requests than the supplied routing trace")

        expected = self.future_requests[self.current_position]

        if expected != expert_id:
            raise ValueError(
                f"Routing trace mismatch: expected {expected}, got {expert_id}"
            )

        self.current_position += 1

    def on_insert(self, expert_id):
        pass

    def on_evict(self, expert_id):
        pass

    def choose_victim(self, cached_experts, protected_experts=None):
        if protected_experts is None:
            protected_experts = set()

        candidates = []

        for expert_id in cached_experts:
            if expert_id in protected_experts:
                continue

            next_use = None

            for position in range(
                self.current_position,
                len(self.future_requests)
            ):
                if self.future_requests[position] == expert_id:
                    next_use = position
                    break

            if next_use is None:
                candidates.append((float("inf"), str(expert_id), expert_id))
            else:
                candidates.append((next_use, str(expert_id), expert_id))

        if not candidates:
            raise RuntimeError("No expert available for eviction")

        candidates.sort(reverse=True)

        return candidates[0][2]