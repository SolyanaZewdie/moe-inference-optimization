import json


def create_routing_trace(layer_decisions):
    trace = []

    for layer_index, selected_experts in enumerate(layer_decisions):
        for token_experts in selected_experts:
            for expert_index in token_experts:
                expert_id = f"layer_{layer_index}_expert_{expert_index}"
                trace.append(expert_id)

    return trace


def save_routing_trace(trace, file_path):
    with open(file_path, "w") as file:
        json.dump(trace, file, indent=4)


def load_routing_trace(file_path):
    with open(file_path, "r") as file:
        trace = json.load(file)

    if not isinstance(trace, list):
        raise ValueError("Routing trace must be a list")

    return trace