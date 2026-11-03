# Memory and Compute for MoE LLM Inference

A hands-on implementation of memory and compute optimizations for Mixture-of-Experts (MoE) language model inference.

The project explores how to reduce memory consumption, improve expert-weight reuse, optimize matrix multiplication, and manage expert loading under a limited memory budget.

## 1. Project Overview

A Mixture-of-Experts model contains multiple feed-forward expert networks. Instead of activating every expert for every token, a router selects only a subset of experts.

This reduces the amount of computation per token, but introduces additional inference challenges:

* Expert weights may occupy more memory than is available in RAM.
* Accessing experts from storage can introduce latency.
* Repeatedly loading expert weights wastes memory bandwidth.
* Small matrix operations can underutilize compute hardware.
* Uneven routing can create imbalanced expert workloads.

This project implements a smaller, locally executable MoE model and investigates techniques for addressing these challenges.

## 2. Implemented Components

### Memory Optimization

**Memory-mapped weight loading**

* Saves model weights as individual NumPy arrays.
* Loads arrays using `numpy.memmap`.
* Supports lazy access to weight data rather than requiring every array to be copied into memory immediately.
* Provides a foundation for loading experts from storage on demand.

**Block-wise quantization**

* Implements symmetric INT8 and INT4 quantization.
* Uses per-block scaling factors.
* Supports dequantization for reconstructing approximate floating-point weights.
* Packs two signed INT4 values into one byte.

**Sensitivity analysis**

* Measures mean squared error and relative L2 error.
* Compares quantized outputs against full-precision reference outputs.
* Evaluates component-level quantization sensitivity.
* Assigns mixed precision according to a specified error tolerance.

### Compute Optimization

**GPU simulator**

A simplified GPU execution simulator models:

* Global-memory transactions.
* Memory transaction grouping.
* Shared-memory capacity.
* Thread indexing and bounds checking.
* Maximum threads per block.

The simulator is an educational performance model, not a physical GPU implementation.

**Matrix multiplication**

Two implementations are provided:

1. Naive matrix multiplication using explicit loops.
2. Tiled matrix multiplication using smaller matrix blocks and NumPy operations.

The tiled implementation models loading tiles into shared memory and reusing them during computation.

### Expert Dispatch

**Top-k routing**

* Computes router scores.
* Applies softmax.
* Selects the top-k experts for each token.
* Uses deterministic tie-breaking.
* Normalizes the selected routing weights.

**Grouped expert computation**

* Groups token rows by selected expert.
* Executes expert computations on grouped inputs.
* Combines expert outputs using routing weights.
* Restores contributions to the appropriate token positions.
* Supports capacity limits and deterministic overflow handling.

The capacity limit is calculated as:

`capacity = ceil(capacity_factor × number_of_tokens × top_k / number_of_experts)`

When an expert exceeds its capacity, assignments are accepted in token order until the limit is reached. Excess assignments are dropped.

### Expert Caching

The expert cache supports three eviction policies:

| Policy | Eviction strategy                                                            |
| ------ | ---------------------------------------------------------------------------- |
| LRU    | Evicts the least recently used expert.                                       |
| LFU    | Evicts the least frequently used expert, with recency used as a tie-breaker. |
| Belady | Evicts the expert whose next use is farthest in the future.                  |

Belady's policy requires a known future access trace and therefore serves as an offline comparison rather than a directly deployable online policy.

The cache:

* Enforces a configurable memory budget.
* Tracks cache hits, misses, and evictions.
* Loads experts when needed.
* Protects experts currently being computed.
* Supports integration with the toy MoE model.

Routing traces can be generated, saved, and replayed for cache-policy experiments.

### Toy MoE Model

The model includes:

* Token embeddings.
* Causal self-attention.
* RMS normalization.
* A top-k MoE router.
* Multiple SwiGLU-style feed-forward experts.
* Weighted expert-output combination.
* Output projection.
* Greedy token generation.

The model is randomly initialized and is intended for implementation and inference experiments, not language-quality evaluation.

## 3. Project Structure

```text
moe-inference-optimization/
│
├── src/
│   ├── memory/
│   │   ├── mmap_loader.py
│   │   ├── quantization.py
│   │   └── sensitivity.py
│   │
│   ├── compute/
│   │   ├── gpu_simulator.py
│   │   ├── naive_matmul.py
│   │   └── tiled_matmul.py
│   │
│   ├── dispatch/
│   │   ├── router.py
│   │   ├── grouped_matmul.py
│   │   └── routing_traces.py
│   │
│   ├── cache/
│   │   ├── expert_cache.py
│   │   ├── lru.py
│   │   ├── lfu.py
│   │   └── belady.py
│   │
│   └── model/
│       ├── toy_moe.py
│       └── tokenizer.py
│
├── tests/
├── benchmarks/
│   └── benchmark_matmul.py
│
├── experiments/
│   └── experiment_log.md
│
├── docs/
│   └── design.md
│
├── README.md
└── requirements.txt
```

## 4. Requirements

* Python 3.13
* NumPy 2.5.4
* Pytest 9.2.0

The implementation uses NumPy and runs on a CPU. An NVIDIA GPU or CUDA installation is not required.

## 5. Setup

Clone the repository:

```bash
git clone https://github.com/SolyanaZewdie/moe-inference-optimization.git
cd moe-inference-optimization
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## 6. Running Tests

Run the complete test suite:

```powershell
python -m pytest -v
```

The latest recorded full test run completed with:

**114 tests passed.**

## 7. Running the Matrix Multiplication Benchmark

Run:

```powershell
python .\benchmarks\benchmark_matmul.py
```

The benchmark compares naive and tiled matrix multiplication using two 64 × 64 float32 matrices.

### Latest Recorded Results

| Implementation | Global transactions | Reduction | Median time |
| -------------- | ------------------: | --------: | ----------: |
| Naive          |              40,960 |         — |  0.133205 s |
| Tiled (8)      |               8,192 |    80.00% |  0.003172 s |
| Tiled (16)     |               4,096 |    90.00% |  0.000734 s |
| Tiled (32)     |               2,048 |    95.00% |  0.000254 s |
| Tiled (64)     |               1,024 |    97.50% |  0.000032 s |

All tiled implementations produced correct results in this benchmark.

Tile size 64 achieved the lowest simulated transaction count. Its shared-memory requirement was 32,768 bytes, within the simulator's 49,152-byte limit.

**Interpretation:** Larger tiles reduced simulated global-memory transactions for this particular workload. The measured execution times are CPU timings, comparing Python loops with NumPy matrix multiplication. They are not evidence of GPU speedup.

## 8. Experimental Considerations

The project distinguishes between correctness tests, simulated hardware metrics, and actual execution-time measurements.

In particular:

* Simulated memory transactions depend on the simulator's transaction model.
* CPU execution times do not represent physical GPU performance.
* Belady's policy assumes knowledge of future expert accesses.
* Quantization error does not automatically establish language-model quality.
* The toy model is randomly initialized and does not represent a trained language model.

Performance claims should be evaluated against the same baseline, workload, and machine.

## 9. Future Extensions

Potential extensions include:

* Expert prefetching based on routing traces.
* Shared-memory bank-conflict analysis.
* Additional cache-policy comparisons.
* More detailed memory and quantization benchmarks.
* Evaluation using a real pretrained MoE checkpoint, subject to hardware and memory availability.

## 10. Repository

GitHub: https://github.com/SolyanaZewdie/moe-inference-optimization
