import numpy as np
from dataclasses import dataclass

from src.memory.mmap_loader import save_weights_mmap, load_weights_mmap
from src.cache.expert_cache import ExpertCache
from src.dispatch.router import Router
from src.dispatch.grouped_matmul import grouped_expert_forward


@dataclass
class MoEConfig:
    vocab_size: int = 128
    d_model: int = 256
    hidden_dim: int = 512
    num_layers: int = 2
    num_experts: int = 8
    top_k: int = 2
    seed: int = 42
    capacity_factor: float | None = None

    def __post_init__(self):
        if self.vocab_size <= 0:
            raise ValueError("Vocabulary size must be positive")

        if self.d_model <= 0 or self.hidden_dim <= 0:
            raise ValueError("Model dimensions must be positive")

        if self.num_layers <= 0 or self.num_experts <= 0:
            raise ValueError("Layers and experts must be positive")

        if not isinstance(self.top_k, int):
            raise ValueError("top_k must be an integer")

        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("Invalid top_k value")

        if self.capacity_factor is not None:
            if (
                not isinstance(self.capacity_factor, (int, float))
                or not np.isfinite(self.capacity_factor)
                or self.capacity_factor <= 0
            ):
                raise ValueError("capacity_factor must be positive and finite")


class ToyMoE:

    def __init__(self, config=None):
        self.config = config or MoEConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.weights = self.initialize_weights()

        self.expert_cache = None
        self.mmap_expert_sources = {}

    def _initialize_matrix(self, rows, cols):
        scale = np.sqrt(2.0 / (rows + cols))
        return (
            self.rng.standard_normal((rows, cols)) * scale
        ).astype(np.float32)

    def initialize_weights(self):
        cfg = self.config

        weights = {
            "embedding": (
                self.rng.standard_normal(
                    (cfg.vocab_size, cfg.d_model)
                ) * 0.02
            ).astype(np.float32),
            "layers": [],
            "output": None
        }

        for _ in range(cfg.num_layers):
            layer = {
                "attention": {
                    name: self._initialize_matrix(
                        cfg.d_model, cfg.d_model
                    )
                    for name in ("W_q", "W_k", "W_v", "W_o")
                },
                "router": self._initialize_matrix(
                    cfg.d_model, cfg.num_experts
                ),
                "experts": []
            }

            for _ in range(cfg.num_experts):
                layer["experts"].append({
                    "W_gate": self._initialize_matrix(
                        cfg.d_model, cfg.hidden_dim
                    ),
                    "W_up": self._initialize_matrix(
                        cfg.d_model, cfg.hidden_dim
                    ),
                    "W_down": self._initialize_matrix(
                        cfg.hidden_dim, cfg.d_model
                    )
                })

            weights["layers"].append(layer)

        weights["output"] = self._initialize_matrix(
            cfg.d_model, cfg.vocab_size
        )

        return weights

    def count_parameters(self):
        total = (
            self.weights["embedding"].size
            + self.weights["output"].size
        )

        for layer in self.weights["layers"]:
            total += sum(
                matrix.size for matrix in layer["attention"].values()
            )
            total += layer["router"].size

            for expert in layer["experts"]:
                total += sum(matrix.size for matrix in expert.values())

        return total

    def save_weights(self, file_path):
        saved = {
            "embedding": self.weights["embedding"],
            "output": self.weights["output"]
        }

        for i, layer in enumerate(self.weights["layers"]):
            saved[f"layer_{i}_router"] = layer["router"]

            for name, matrix in layer["attention"].items():
                saved[f"layer_{i}_attention_{name}"] = matrix

            for j, expert in enumerate(layer["experts"]):
                for name, matrix in expert.items():
                    saved[f"layer_{i}_expert_{j}_{name}"] = matrix

        np.savez(file_path, **saved)

    def load_weights(self, file_path):
        with np.load(file_path) as saved:
            self.weights["embedding"] = saved["embedding"].copy()
            self.weights["output"] = saved["output"].copy()

            for i, layer in enumerate(self.weights["layers"]):
                layer["router"] = saved[f"layer_{i}_router"].copy()

                for name in layer["attention"]:
                    layer["attention"][name] = (
                        saved[f"layer_{i}_attention_{name}"].copy()
                    )

                for j, expert in enumerate(layer["experts"]):
                    for name in expert:
                        expert[name] = (
                            saved[f"layer_{i}_expert_{j}_{name}"].copy()
                        )

        self.expert_cache = None
        self.mmap_expert_sources = {}

    # Memory mapping and expert caching

    def enable_mmap_expert_cache(
        self, directory, memory_limit_bytes, policy
    ):
        save_weights_mmap(self.weights, directory)
        self.weights = load_weights_mmap(directory)

        self.mmap_expert_sources = {}

        for i, layer in enumerate(self.weights["layers"]):
            for j, expert in enumerate(layer["experts"]):
                expert_id = f"layer_{i}_expert_{j}"
                self.mmap_expert_sources[expert_id] = expert

        self.expert_cache = ExpertCache(memory_limit_bytes, policy)

        return len(self.mmap_expert_sources)

    def load_expert_from_mmap(self, expert_id):
        if expert_id not in self.mmap_expert_sources:
            raise KeyError(f"Unknown expert: {expert_id}")

        return {
            name: np.array(matrix, copy=True)
            for name, matrix in self.mmap_expert_sources[expert_id].items()
        }

    def get_experts_for_computation(
        self, layer_index, layer, selected_experts
    ):
            if self.expert_cache is None:
                return layer["experts"]

        # Experts will be loaded individually during execution.
            return [None] * self.config.num_experts
    def load_expert_for_computation(self, layer_index, expert_index):
        expert_id = f"layer_{layer_index}_expert_{expert_index}"

        return self.expert_cache.get(
            expert_id,
            self.load_expert_from_mmap,
            protected_experts={expert_id}
        )
    # Transformer operations

    def rms_norm(self, x, epsilon=1e-6):
        mean_square = np.mean(x ** 2, axis=-1, keepdims=True)
        return x / np.sqrt(mean_square + epsilon)

    def softmax(self, scores):
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    def attention_forward(self, x, attention_weights):
        Q = x @ attention_weights["W_q"]
        K = x @ attention_weights["W_k"]
        V = x @ attention_weights["W_v"]

        scores = Q @ np.swapaxes(K, -1, -2)
        scores /= np.sqrt(self.config.d_model)

        sequence_length = x.shape[1]

        mask = np.triu(
            np.ones((sequence_length, sequence_length), dtype=bool),
            k=1
        )

        scores = np.where(mask, -np.inf, scores)

        attention = self.softmax(scores)
        return (attention @ V) @ attention_weights["W_o"]

    # Model forward pass

    def forward(self, token_ids, return_routing=False):
        token_ids = np.asarray(token_ids)

        if token_ids.ndim == 1:
            token_ids = token_ids[np.newaxis, :]
            single_sequence = True
        elif token_ids.ndim == 2:
            single_sequence = False
        else:
            raise ValueError("Token IDs must be a 1D or 2D array")

        if token_ids.shape[0] == 0 or token_ids.shape[1] == 0:
            raise ValueError("Input batch and sequence cannot be empty")

        if not np.issubdtype(token_ids.dtype, np.integer):
            raise ValueError("Token IDs must be integers")

        if np.any(token_ids < 0) or np.any(
            token_ids >= self.config.vocab_size
        ):
            raise ValueError("Token ID outside vocabulary range")

        token_ids = token_ids.astype(np.int64)
        x = self.weights["embedding"][token_ids]

        routing_information = []

        for layer_index, layer in enumerate(self.weights["layers"]):
            # Attention and residual connection.
            x = x + self.attention_forward(
                self.rms_norm(x),
                layer["attention"]
            )

            normalized_x = self.rms_norm(x)

            batch_size, sequence_length, model_dim = normalized_x.shape

            flattened_x = normalized_x.reshape(
                batch_size * sequence_length,
                model_dim
            )

            # Top-k routing.
            router = Router(
                self.config.num_experts,
                self.config.top_k
            )

            selected_experts, selected_weights, probabilities = (
                router.route(flattened_x, layer["router"])
            )

            # Retrieve and execute selected experts.
            experts = self.get_experts_for_computation(
                layer_index, layer, selected_experts
            )

            expert_loader = None

            if self.expert_cache is not None:
                expert_loader = lambda expert_index: (
                    self.load_expert_for_computation(
                        layer_index, expert_index
                    )
                )

            expert_output, dropped_assignments = grouped_expert_forward(
                flattened_x,
                selected_experts,
                selected_weights,
                experts,
                capacity_factor=self.config.capacity_factor,
                expert_loader=expert_loader
            )

            expert_output = expert_output.reshape(
                batch_size, sequence_length, model_dim
            )

            x = x + expert_output

            if return_routing:
                routing_information.append({
                    "layer": layer_index,
                    "selected_experts": selected_experts,
                    "selected_weights": selected_weights,
                    "router_probabilities": probabilities,
                    "dropped_assignments": dropped_assignments,
                    "capacity_factor": self.config.capacity_factor
                })

        x = self.rms_norm(x)
        logits = x @ self.weights["output"]

        if single_sequence:
            logits = logits[0]

        if return_routing:
            return logits, routing_information

        return logits

    # Autoregressive generation

    def generate(self, token_ids, max_new_tokens=10):
        token_ids = np.asarray(token_ids)

        if token_ids.ndim != 1 or len(token_ids) == 0:
            raise ValueError("Input must be a non-empty 1D sequence")

        if not np.issubdtype(token_ids.dtype, np.integer):
            raise ValueError("Token IDs must be integers")

        if np.any(token_ids < 0) or np.any(
            token_ids >= self.config.vocab_size
        ):
            raise ValueError("Token ID outside vocabulary range")

        if not isinstance(max_new_tokens, (int, np.integer)):
            raise ValueError("max_new_tokens must be an integer")

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")

        generated = token_ids.astype(np.int64).copy()

        for _ in range(max_new_tokens):
            logits = self.forward(generated)
            next_token = np.argmax(logits[-1])
            generated = np.append(generated, next_token)

        return generated