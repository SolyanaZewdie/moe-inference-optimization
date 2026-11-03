from collections import OrderedDict


class LRUCachePolicy:
    def __init__(self):
        self.access_order = OrderedDict()

    def on_access(self, expert_id):
        if expert_id in self.access_order:
            self.access_order.move_to_end(expert_id)

    def on_insert(self, expert_id):
        self.access_order[expert_id] = True
        self.access_order.move_to_end(expert_id)

    def on_evict(self, expert_id):
        self.access_order.pop(expert_id, None)

    def choose_victim(self, cached_experts, protected_experts=None):
        if protected_experts is None:
            protected_experts = set()

        for expert_id in self.access_order:
            if expert_id in cached_experts and expert_id not in protected_experts:
                return expert_id

        raise RuntimeError("No expert available for eviction")