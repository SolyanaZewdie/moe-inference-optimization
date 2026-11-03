import numpy as np


class Router:
    def __init__(self, num_experts, top_k):
        # Store the routing configuration.
        self.num_experts = num_experts
        self.top_k = top_k

        # Validate the configuration.
        if num_experts <= 0:
            raise ValueError("Number of experts must be positive")

        if top_k < 1 or top_k > num_experts:
            raise ValueError("Invalid top_k value")

    def softmax(self, scores):
        # Subtract the maximum score for numerical stability.
        scores = scores - np.max(scores, axis=1, keepdims=True)

        # Convert scores into probabilities.
        exp_scores = np.exp(scores)
        probabilities = exp_scores / np.sum(
            exp_scores, axis=1, keepdims=True
        )

        return probabilities

    def route(self, token_representations, router_weights):
        # Calculate routing scores for every token and expert.
        scores = token_representations @ router_weights

        # Convert scores into probabilities.
        probabilities = self.softmax(scores)

        selected_experts = []
        selected_weights = []

        # Process each token independently.
        for token_probabilities in probabilities:

            # Select the top-k experts.
            # Stable sorting resolves ties by choosing lower indices first.
            expert_indices = np.argsort(
                -token_probabilities,
                kind="stable"
            )[:self.top_k]

            # Retrieve probabilities for the selected experts.
            expert_weights = token_probabilities[expert_indices]

            # Normalize selected weights to sum to one.
            expert_weights = expert_weights / np.sum(expert_weights)

            selected_experts.append(expert_indices)
            selected_weights.append(expert_weights)

        return (
            np.array(selected_experts, dtype=np.int64),
            np.array(selected_weights, dtype=np.float32),
            probabilities
        )