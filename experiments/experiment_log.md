## Experiment: Naive vs. Tiled Matrix Multiplication

### Hypothesis
Tiled matrix multiplication should reduce simulated global-memory
transactions by loading matrix tiles and reusing their values.

### Configuration
- Matrix A: 64 x 64
- Matrix B: 64 x 64
- Data type: float32
- Warp size: 32
- Transaction size: 32 bytes
- Shared memory limit: 48 KB
- Tile sizes: 8, 16, 32, 64
- Random seed: 42

### Results

| Implementation | Transactions | Reduction |
|---|---:|---:|
| Naive | 40960 | Baseline |
| Tiled 8 | 12288 | 70.00% |
| Tiled 16 | 5120 | 87.50% |
| Tiled 32 | 2304 | 94.38% |
| Tiled 64 | 1152 | 97.19% |

All tiled outputs matched the naive output within tolerance.

### Observations
Larger tiles reduced the simulated global-memory transaction count.
The 64 x 64 tile achieved the lowest transaction count.

### Limitations
- The simulator models simplified warp-level transactions.
- It does not model actual GPU hardware or caches.
- CPU execution times are not GPU performance measurements.
- Different NumPy and Python execution strategies affect timing.

### Conclusion
Tiling substantially reduced global-memory transactions under
the current simulation model. Further validation of warp-level
access patterns is needed before treating the reduction figures
as hardware-representative.

# Experiment Log — Memory and Compute for MoE LLM Inference

## 1. Purpose

This log records implementation experiments, hypotheses, configurations, results, failures, and observations.

The experiments focus on:

* Matrix multiplication and simulated memory transactions.
* Quantization and reconstruction error.
* Memory-mapped weight loading.
* Expert cache behavior.
* Grouped expert computation.
* End-to-end toy MoE integration.

Only measured results are reported as experimental findings.

---

## Experiment 1 — Naive vs. Tiled Matrix Multiplication

**Status:** Completed

### Objective

Compare naive matrix multiplication against tiled matrix multiplication using simulated global-memory transactions and CPU execution time.

### Hypothesis

Tiling should reduce repeated global-memory accesses by allowing matrix elements to be reused across multiple output calculations.

Larger tiles may improve reuse, provided their shared-memory requirements remain within the simulator's limit.

### Configuration

| Parameter           | Value         |
| ------------------- | ------------- |
| Matrix A            | 64 × 64       |
| Matrix B            | 64 × 64       |
| Data type           | float32       |
| Random seed         | 42            |
| Transaction size    | 32 bytes      |
| Shared-memory limit | 49,152 bytes  |
| Tile sizes          | 8, 16, 32, 64 |

### Results

| Implementation | Transactions | Reduction | Median time |
| -------------- | -----------: | --------: | ----------: |
| Naive          |       40,960 |         — |  0.133205 s |
| Tiled (8)      |        8,192 |    80.00% |  0.003172 s |
| Tiled (16)     |        4,096 |    90.00% |  0.000734 s |
| Tiled (32)     |        2,048 |    95.00% |  0.000254 s |
| Tiled (64)     |        1,024 |    97.50% |  0.000032 s |

All tiled implementations produced numerically correct results.

### Shared-memory requirements

For two float32 tiles:

$$
SharedMemory = 2T^2 \times 4
$$

| Tile size | Shared memory |
| --------- | ------------: |
| 8         |     512 bytes |
| 16        |   2,048 bytes |
| 32        |   8,192 bytes |
| 64        |  32,768 bytes |

All tested configurations fit within the simulator's 49,152-byte limit.

### Observations

1. Transaction counts decreased as tile size increased.
2. Tile size 64 achieved the lowest simulated transaction count.
3. Tile size 64 also produced the lowest measured CPU execution time in this run.
4. All configurations fit within the modeled shared-memory budget.

### Interpretation

For this workload and simulator model, larger tiles improved modeled memory reuse.

The result does not establish that tile size 64 is universally optimal. Larger matrices, different data types, hardware constraints, and other workloads may produce different results.

The execution-time comparison is between Python loops and NumPy matrix multiplication on a CPU. It is not a physical GPU speedup measurement.

---

## Experiment 2 — Automated Component and Integration Tests

**Status:** Completed

### Objective

Verify correctness of the implemented memory, compute, dispatch, caching, and model components.

### Hypothesis

Each component should satisfy its defined behavior independently, while the integrated MoE model should preserve correct outputs when expert caching is enabled.

### Method

The pytest suite covers:

* Routing and top-k selection.
* Grouped expert computation.
* Capacity limits and overflow behavior.
* Quantization and INT4 packing.
* Quantization sensitivity calculations.
* Memory-mapped weight saving and loading.
* GPU simulator behavior.
* Naive and tiled matrix multiplication.
* LRU, LFU, and Belady cache policies.
* Routing trace generation and replay.
* Toy MoE forward execution and cache integration.

### Result

**Latest recorded full test run: 114 passed.**

### Observations

The test suite validates the implemented behavior for the covered configurations and edge cases.

Passing tests establish correctness for those cases, not production-scale performance or universal correctness for every possible workload.

---

## Experiment 3 — Expert Cache Integration

**Status:** Completed

### Objective

Verify that the toy MoE model can load experts through a memory-bounded cache rather than requiring all experts to remain resident.

### Hypothesis

Lazy expert loading should allow inference to proceed even when the cache cannot hold every expert selected across a complete layer.

### Method

The integration tests cover:

* Loading expert weights from memory-mapped files.
* Cache hits and misses.
* Expert reuse across forward passes.
* Cache memory-budget enforcement.
* Protection of the expert currently being computed.
* Execution with a cache capacity of one expert.
* Comparison between cached and uncached model outputs.
* Belady trace replay.

### Observations

The implementation uses lazy loading during grouped expert execution. Experts are requested individually, computed, and then become eligible for eviction.

The tests verify output consistency and memory-budget enforcement for the tested configurations.

### Limitations

No standalone throughput or peak-RAM benchmark has been recorded for this experiment. Therefore, no numerical cache speedup or total process-memory reduction is claimed.

---

## Experiment 4 — Block-wise Quantization

**Status:** Implementation and correctness tests completed; performance evaluation pending.

### Objective

Implement INT8 and INT4 block-wise quantization and evaluate the effect of quantization on reconstructed weights.

### Hypothesis

INT4 should require fewer weight-storage bits than INT8, but should generally introduce greater reconstruction error.

Block-wise scaling should adapt to local weight distributions.

### Method

The implementation:

1. Divides weights into configurable blocks.
2. Calculates one scale per block.
3. Quantizes values using symmetric INT8 or INT4 ranges.
4. Reconstructs approximate floating-point values.
5. Packs two INT4 values into each byte.
6. Supports unpacking and reconstruction.

### Validation

Automated tests cover quantization, dequantization, INT4 packing, unpacking, and sensitivity calculations.

### Results

No complete model-level quality-versus-memory benchmark has been recorded yet.

The expected storage advantage follows from the representation:

* INT8 uses 8 bits per quantized value, excluding scale metadata.
* INT4 uses 4 bits per quantized value, excluding scale metadata.
* INT4 packing stores two quantized values in one byte.

Actual total memory savings must account for scale metadata, padding, and other storage overhead.

---

## Experiment 5 — Memory-Mapped Weight Loading

**Status:** Implementation and correctness tests completed; performance evaluation pending.

### Objective

Load model weights through NumPy memory mapping rather than eagerly copying all arrays into regular memory.

### Hypothesis

Memory mapping should support lazy access to file-backed weight arrays and provide a foundation for loading experts on demand.

### Method

The implementation saves weight structures as NumPy array files and a JSON manifest.

The loader reconstructs the structure using read-only memory-mapped arrays.

### Validation

Tests verify saving, loading, reconstructed structures, and memory-mapped array behavior.

### Results

No controlled cold-cache versus warm-cache timing, page-fault count, or peak-RAM comparison has been recorded.

Memory mapping is therefore implemented and tested, but its performance advantage has not yet been quantified.

---

## 2. Summary of Experimental Evidence

| Area                  | Implementation                         | Correctness tested | Performance measured |
| --------------------- | -------------------------------------- | ------------------ | -------------------- |
| Matrix multiplication | Naive and tiled                        | Yes                | Yes                  |
| GPU simulator         | Transactions and shared memory         | Yes                | Simulated metrics    |
| Quantization          | INT8, INT4, packing                    | Yes                | Not yet              |
| Sensitivity analysis  | Error metrics and precision assignment | Yes                | Not yet              |
| Memory mapping        | NumPy mmap loader                      | Yes                | Not yet              |
| Expert caching        | LRU, LFU, Belady                       | Yes                | Not yet              |
| Grouped dispatch      | Top-k and grouped execution            | Yes                | Not yet              |
| Toy MoE integration   | Cached and uncached execution          | Yes                | Not yet              |

## 3. Important Experimental Limitations

* The toy model is randomly initialized and is not a trained language model.
* The GPU simulator does not execute physical GPU kernels.
* Transaction counts depend on the simulator's simplified memory model.
* CPU execution timings cannot establish GPU acceleration.
* Cache memory accounting does not include all Python runtime overhead.
* Quantization reconstruction error is not equivalent to language-model quality.
* Belady's policy assumes future routing information.
* No real pretrained OLMoE checkpoint has been benchmarked.

## 4. Future Experiments

The following experiments remain possible extensions:

1. Compare LRU, LFU, and Belady using identical routing traces and cache budgets.
2. Measure cache hit rates, misses, evictions, and execution time.
3. Compare FP32, INT8, and INT4 weight storage.
4. Measure quantization error across individual experts and model components.
5. Evaluate mixed precision against uniform INT4 under a fixed error tolerance.
6. Compare memory usage for normal loading and memory mapping.
7. Measure cold and warm memory-mapped access behavior.
8. Investigate shared-memory bank conflicts and padded layouts.
9. Evaluate expert prefetching using routing traces.

---

## 5. Experiment Recording Template

For every additional experiment, record:

### Experiment name

**Date:**

**Objective:**

**Hypothesis:**

**Configuration:**

**Method:**

**Results:**

**Observations:**

**Failures and limitations:**

**Conclusion:**
