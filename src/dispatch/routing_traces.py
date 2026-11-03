import json


def create_routing_trace(layer_decisions):
    """
    Create a cache-access trace from routing decisions.

    Each expert appears once per layer, in ascending expert-index
    order. This matches the order used by grouped_expert_forward().
    """
    trace = []

    for layer_index, selected_experts in enumerate(layer_decisions):
        unique_experts = set()

        for token_experts in selected_experts:
            for expert_index in token_experts:
                unique_experts.add(int(expert_index))

        for expert_index in sorted(unique_experts):
            expert_id = f"layer_{layer_index}_expert_{expert_index}"
            trace.append(expert_id)

    return trace


def save_routing_trace(trace, file_path):
    """Save a routing trace as a JSON file."""
    with open(file_path, "w") as file:
        json.dump(trace, file, indent=4)


def load_routing_trace(file_path):
    """Load and validate a routing trace from a JSON file."""
    with open(file_path, "r") as file:
        trace = json.load(file)

    if not isinstance(trace, list):
        raise ValueError("Routing trace must be a list")

    return trace