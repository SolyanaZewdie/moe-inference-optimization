import numpy as np


def grouped_expert_forward(
    token_representations,
    selected_experts,
    selected_weights,
    experts,
    capacity_factor=None,
    expert_loader=None
):
    """
    Process tokens in groups based on their selected experts.

    Assignments exceeding expert capacity are dropped.

    Overflow policy:
    Keep assignments in token order and drop the rest.
    """

    token_representations = np.asarray(token_representations)
    selected_experts = np.asarray(selected_experts)
    selected_weights = np.asarray(selected_weights)

    # Validate input dimensions.
    if token_representations.ndim != 2:
        raise ValueError("Token representations must be a 2D matrix")

    if selected_experts.ndim != 2:
        raise ValueError("Selected experts must be a 2D matrix")

    if selected_weights.shape != selected_experts.shape:
        raise ValueError("Expert indices and weights must have matching shapes")

    num_tokens = token_representations.shape[0]
    num_experts = len(experts)
    top_k = selected_experts.shape[1]

    if selected_experts.shape[0] != num_tokens:
        raise ValueError("Number of tokens must match routing assignments")

    if num_experts == 0 and selected_experts.size > 0:
        raise ValueError("At least one expert is required")

    if selected_experts.size > 0:
        if np.any(selected_experts < 0):
            raise ValueError("Expert indices cannot be negative")

        if np.any(selected_experts >= num_experts):
            raise ValueError("Expert index is out of range")

    if not np.all(np.isfinite(selected_weights)):
        raise ValueError("Routing weights must be finite")

    # Calculate expert capacity.
    if capacity_factor is not None:
        if capacity_factor <= 0:
            raise ValueError("Capacity factor must be positive")

        if num_experts == 0:
            raise ValueError("At least one expert is required")

        capacity = int(np.ceil(
            capacity_factor * num_tokens * top_k / num_experts
        ))
    else:
        capacity = None

    # Initialize outputs.
    combined_outputs = np.zeros(
        token_representations.shape,
        dtype=np.float32
    )

    # Track how many assignments were dropped.
    dropped_assignments = 0

    # Process one expert at a time.
    for expert_index in range(num_experts):
        # Find all assignments to this expert.
        token_indices, top_k_positions = np.where(
            selected_experts == expert_index
        )

        # Skip experts with no assignments.
        if len(token_indices) == 0:
            continue
                # Load the expert only when it is needed.
        expert = experts[expert_index]

        if expert is None:
            if expert_loader is None:
                raise ValueError(
                    "An expert loader is required for missing experts"
                )

            expert = expert_loader(expert_index)

        # Apply capacity limit.
        if capacity is not None:
            if len(token_indices) > capacity:
                dropped_assignments += len(token_indices) - capacity

                token_indices = token_indices[:capacity]
                top_k_positions = top_k_positions[:capacity]

        # Collect accepted token representations.
        expert_inputs = token_representations[token_indices]

        # Apply SwiGLU.
        gate = expert_inputs @ expert["W_gate"]
        up = expert_inputs @ expert["W_up"]

        gate = gate / (
            1.0 + np.exp(-np.clip(gate, -60, 60))
        )

        hidden = gate * up

        # Project back to model dimension.
        expert_outputs = hidden @ expert["W_down"]

        # Apply routing weights.
        weights = selected_weights[token_indices, top_k_positions]

        weighted_outputs = (
            expert_outputs * weights[:, np.newaxis]
        )

        # Add outputs back to their original tokens.
        np.add.at(
            combined_outputs,
            token_indices,
            weighted_outputs
        )

    return combined_outputs, dropped_assignments