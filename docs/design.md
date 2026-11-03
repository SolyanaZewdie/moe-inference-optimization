# System Design: Memory and Compute for MoE LLM Inference

## 1. Introduction

This project investigates memory and compute optimization techniques for Mixture-of-Experts (MoE) language model inference.

An MoE model contains multiple expert networks, but activates only a subset of them for each input token. Although this reduces the amount of computation performed per token, it introduces challenges involving expert-weight storage, memory access, routing, and workload distribution.

The implementation uses a smaller, randomly initialized MoE model to study these challenges locally.

The system is divided into five main components:

1. Memory optimization.
2. Compute optimization.
3. Expert dispatch.
4. Expert caching.
5. Toy MoE model integration.

The implementation uses Python and NumPy and runs on a CPU. GPU behavior is represented through a simplified simulator.

---

## 2. Overall Architecture

The system follows this high-level execution flow:

```text
                 INPUT TOKENS
                       |
                       v
                Token Embedding
                       |
                       v
                 Self-Attention
                       |
                       v
                 MoE Layer Input
                       |
                       v
                  Router
                       |
                       v
              Top-k Expert Selection
                       |
                       v
                Group by Expert
                       |
                       v
              Expert Cache Lookup
                       |
              +--------+--------+
              |                 |
            Cache Hit         Cache Miss
              |                 |
              |            Load from mmap
              |                 |
              +--------+--------+
                       |
                       v
               Expert Computation
                       |
                       v
              Weighted Combination
                       |
                       v
                 Residual Update
                       |
                       v
                  Output Layer
                       |
                       v
                 OUTPUT LOGITS
```

The memory, compute, and dispatch components work together during expert execution.

### Component responsibilities

| Component | Responsibility                                                  |
| --------- | --------------------------------------------------------------- |
| Memory    | Store, load, quantize, and evaluate model weights.              |
| Compute   | Execute matrix multiplication and simulate memory transactions. |
| Dispatch  | Select experts and group token computations.                    |
| Cache     | Manage expert residency under a fixed memory budget.            |
| Model     | Integrate the components into an executable MoE model.          |

---

## 3. Toy MoE Model

### 3.1 Model configuration

The model uses a configurable architecture.

The default configuration includes:

| Parameter                  | Default value |
| -------------------------- | ------------: |
| Vocabulary size            |           128 |
| Model dimension            |           256 |
| Expert hidden dimension    |           512 |
| Transformer layers         |             2 |
| Experts per layer          |             8 |
| Experts selected per token |             2 |
| Random seed                |            42 |

These dimensions are intentionally smaller than those of production language models so the model can execute locally.

### 3.2 Transformer execution

The model uses token embeddings, causal self-attention, and MoE feed-forward computation.

The residual structure is:

```text
x = x + Attention(norm(x))
x = x + MoE(norm(x))
```

The attention component projects the input into queries, keys, and values.

It calculates attention scores, applies softmax, and combines the value vectors according to those scores.

The attention implementation is a simplified single-head causal implementation rather than a complete production transformer attention stack.

### 3.3 Router

The router determines which experts process each token.

For token representation \(x\), the router calculates expert scores:

$$
s = xW_r
$$

It then converts the scores into probabilities:

$$
p_i = \frac{e^{s_i}}{\sum_j e^{s_j}}
$$

The top-k experts are selected according to their probabilities.

Their selected probabilities are normalized:

$$
\hat{p}_i =
\frac{p_i}{\sum_{j \in TopK}p_j}
$$

These normalized probabilities are used when combining expert outputs.

The implementation uses deterministic tie-breaking to make expert selection reproducible.

### 3.4 Expert architecture

Each expert implements a SwiGLU-style feed-forward network:

$$
Expert(x) =
W_{down}
\left(
SiLU(xW_{gate}) \odot (xW_{up})
\right)
$$

Where:

* \(W_{gate}\) produces the gating projection.
* \(W_{up}\) produces the up projection.
* \(SiLU\) is the activation function.
* \(\odot\) represents element-wise multiplication.
* \(W_{down}\) projects the hidden representation back to model dimension.

The router's selected probabilities determine how expert outputs are combined:

$$
y = \sum_{i \in TopK} \hat{p}_i Expert_i(x)
$$

---

## 4. Memory Optimization

### 4.1 Memory-mapped weight loading

#### Problem

Loading every model weight into regular RAM can become expensive when the model is larger than the available memory.

MoE models introduce an additional opportunity: only selected experts are needed for a particular token.

Loading all expert weights before inference may therefore be unnecessary.

#### Design

The implementation uses NumPy memory mapping.

Weights are saved as individual `.npy` files. A JSON manifest records the structure needed to reconstruct the original weight hierarchy.

The loader uses:

```python
np.load(file_path, mmap_mode="r")
```

This creates a read-only memory-mapped array.

The operating system manages the underlying file-backed pages and brings data into physical memory as it is accessed.

#### Benefits

* Supports lazy access to large weight files.
* Avoids eagerly copying every weight array into ordinary RAM.
* Allows individual experts to be loaded when needed.
* Works with the expert cache.

#### Trade-offs

Memory mapping does not eliminate the physical memory needed to access weights.

Accessing non-resident pages can cause page faults and storage latency. Random expert access can also reduce the effectiveness of sequential read-ahead.

The current implementation uses individual NumPy files for expert weights. It does not claim to implement a complete production-grade storage engine or page-level expert scheduler.

### 4.2 Block-wise quantization

#### Problem

Model weights stored in floating-point formats consume significant memory.

Quantization reduces the number of bits required to represent weights.

#### Design

The implementation supports symmetric INT8 and INT4 block-wise quantization.

Weights are divided into blocks. Each block receives its own scale factor.

For a block with maximum absolute value \(a_{max}\), the scale is:

$$
scale = \frac{a_{max}}{q_{max}}
$$

The quantized values are:

$$
q = clip\left(round\left(\frac{w}{scale}\right),
-q_{max},q_{max}\right)
$$

For INT8, the implementation uses:

$$
q_{max}=127
$$

For symmetric INT4, it uses:

$$
q_{max}=7
$$

The value -8 is left unused in symmetric INT4 quantization.

Reconstruction uses:

$$
\hat{w}=q \times scale
$$

#### INT4 packing

Each signed INT4 value requires four bits.

Two values can therefore be stored in one byte:

```text
One byte:

+----------------+----------------+
| First INT4     | Second INT4    |
| 4 bits         | 4 bits         |
+----------------+----------------+
```

The implementation packs the first value into the lower four bits and the second into the upper four bits.

Unpacking restores the signed values using two's-complement interpretation.

#### Design trade-offs

| Format | Storage                  | Precision        |
| ------ | ------------------------ | ---------------- |
| FP32   | 32 bits/value            | High             |
| INT8   | 8 bits/value plus scales | Higher than INT4 |
| INT4   | 4 bits/value plus scales | Lower than INT8  |

Smaller blocks can adapt more closely to local weight distributions, but require more scale metadata.

Larger blocks reduce scale overhead but can increase quantization error.

The implementation uses a configurable block size so these trade-offs can be investigated.

### 4.3 Quantization sensitivity

Not every model component responds equally to quantization.

A component may tolerate INT4 quantization while another may experience substantially greater output error.

The sensitivity module compares reference outputs with quantized outputs.

It calculates mean squared error:

$$
MSE =
\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
$$

It also calculates relative L2 error:

$$
Relative\ L2 =
\frac{\|y-\hat{y}\|_2}{\|y\|_2}
$$

The implementation uses these measurements to assign precision according to a specified error tolerance.

The mixed-precision assignment supports:

* INT4 when its measured error is acceptable.
* INT8 when INT4 fails but INT8 meets the tolerance.
* FP32 when neither quantized option meets the tolerance.

This is a component-level experimental strategy. It does not yet establish that the resulting mixed-precision model preserves language-model quality.

---

## 5. Compute Optimization

### 5.1 GPU execution model

GPUs execute many lightweight threads in parallel.

The simplified simulator represents several GPU concepts:

| Concept       | Meaning                                                    |
| ------------- | ---------------------------------------------------------- |
| Thread        | Individual execution unit.                                 |
| Warp          | Group of 32 threads.                                       |
| Block         | Group of threads that can cooperate through shared memory. |
| Grid          | Collection of blocks executing a kernel.                   |
| Global memory | Large memory accessible across the GPU.                    |
| Shared memory | Smaller, faster memory shared within a block.              |

The simulator tracks global-memory transactions and shared-memory usage.

It also validates shared-memory capacity and thread-index bounds.

It does not execute actual GPU instructions or model all hardware scheduling behavior.

### 5.2 Global-memory transactions

The simulator groups memory accesses into transaction-sized address segments.

For each accessed address:

$$
segment =
\left\lfloor
\frac{address}{transaction\_size}
\right\rfloor
$$

The number of unique segments accessed represents the simulated transaction count for the supplied access group.

This allows comparison between memory-access patterns.

Transaction counts are therefore metrics of the simulator's model, not hardware profiler measurements.

### 5.3 Naive matrix multiplication

The naive implementation calculates:

$$
C = AB
$$

Each output element is calculated as:

$$
C_{ij} =
\sum_{k=0}^{n-1} A_{ik}B_{kj}
$$

The implementation uses explicit loops over output rows, columns, and the inner dimension.

This is a reference implementation for correctness and transaction-count comparison.

### 5.4 Tiled matrix multiplication

#### Problem

Naive matrix multiplication repeatedly accesses matrix elements from global memory.

Tiling divides the matrices into smaller blocks so values can be reused during computation.

#### Design

For tile size \(T\), the algorithm:

1. Selects a tile from matrix A.
2. Selects the corresponding tile from matrix B.
3. Accounts for loading these tiles into simulated shared memory.
4. Multiplies the tiles using NumPy.
5. Accumulates partial results.
6. Continues until the full output matrix is computed.

The shared-memory requirement for two square tiles is:

$$
SharedMemory =
2T^2 \times bytes\_per\_element
$$

For float32:

$$
SharedMemory = 8T^2 \ bytes
$$

The simulator rejects tile configurations that exceed its configured shared-memory limit.

#### Tile-size trade-offs

Larger tiles can increase data reuse and reduce the number of simulated global-memory transactions.

However, they also require more shared memory.

A tile size that is too large may exceed available shared memory or limit the number of blocks that can reside on a real GPU.

The best tile size depends on the workload and hardware constraints.

### 5.5 Benchmark design

The benchmark compares naive and tiled matrix multiplication using:

* Two 64 × 64 matrices.
* Float32 values.
* A fixed random seed.
* Tile sizes 8, 16, 32, and 64.

Correctness is checked against the reference result.

The benchmark reports median execution time, simulated global-memory transactions, and transaction reduction.

The current timing measurements are CPU measurements. They compare Python loops against NumPy's optimized matrix multiplication and should not be interpreted as a GPU speedup.

---

## 6. Expert Dispatch Optimization

### 6.1 Problem

A naive MoE implementation can execute expert computations token by token.

If multiple tokens select the same expert, repeatedly invoking the expert separately can cause unnecessary overhead and reduce weight reuse.

### 6.2 Grouped expert computation

The implementation groups tokens according to their selected expert.

The process is:

1. Receive token representations and top-k routing decisions.
2. Identify all tokens assigned to each expert.
3. Gather their input representations.
4. Execute the expert on the grouped matrix.
5. Multiply outputs by their corresponding routing weights.
6. Add the weighted outputs to their original token positions.

For an expert receiving \(n\) tokens:

$$
X_e \in \mathbb{R}^{n \times d}
$$

The expert processes the complete group using matrix operations.

This allows the same expert weights to be reused across multiple token rows.

### 6.3 Capacity factor

An expert may receive an uneven number of token assignments.

A capacity limit prevents a single expert from receiving an unlimited number of assignments in a batch.

The implementation calculates:

$$
Capacity =
\left\lceil
CapacityFactor
\times
\frac{N \times K}{E}
\right\rceil
$$

Where:

* \(N\) = number of tokens.
* \(K\) = top-k value.
* \(E\) = number of experts.

Assignments beyond capacity are dropped deterministically in token order.

The dropped assignments do not contribute to the expert output.

The model's residual connection remains part of the overall layer computation.

The capacity factor is configurable and can be disabled.

### 6.4 Correctness considerations

Grouped execution must preserve the behavior of the reference computation.

Important cases include:

* Empty expert groups.
* Multiple tokens assigned to one expert.
* Uneven expert workloads.
* Capacity overflow.
* Different token counts.
* Deterministic routing ties.

The implementation is tested against the expected reference behavior.

---

## 7. Expert Cache

### 7.1 Problem

The complete collection of expert weights may exceed the available memory budget.

Loading every expert simultaneously defeats the purpose of lazy loading.

The system therefore uses a cache that stores only a limited number of experts.

### 7.2 Cache architecture

The cache maintains:

* A dictionary of currently cached experts.
* The memory size of each cached expert.
* A fixed memory limit.
* Hit, miss, and eviction statistics.
* An eviction policy.

The basic lookup procedure is:

```text
Request expert
      |
      v
Is expert cached?
   /       \
 Yes       No
  |         |
  v         v
Return    Load expert
expert       |
             v
       Is there space?
          /     \
        Yes      No
         |        |
         |      Evict victim
         |        |
         +----+---+
              |
              v
        Insert expert
              |
              v
        Return expert
```

### 7.3 Cache policies

#### Least Recently Used (LRU)

LRU evicts the expert that has not been accessed for the longest time.

It assumes recently used experts are more likely to be requested again.

Advantages:

* Simple.
* Adapts to recent access patterns.

Limitation:

* One-time access bursts can evict frequently reused experts.

#### Least Frequently Used (LFU)

LFU evicts the expert with the lowest access frequency.

The implementation uses recency to break frequency ties.

Advantages:

* Retains experts that are repeatedly accessed.

Limitation:

* Older access history can make it slow to adapt when routing patterns change.

#### Belady's Optimal Replacement

Belady's algorithm evicts the expert whose next request occurs farthest in the future.

If an expert will never be requested again, it can be selected first.

Advantages:

* Provides an offline reference for cache replacement under a known request sequence.

Limitation:

* Requires knowledge of future requests, which is generally unavailable during normal online inference.

### 7.4 Protected experts

An expert currently being computed must not be evicted.

The cache therefore accepts a set of protected expert identifiers.

When selecting a victim, the cache excludes protected experts.

If there is insufficient space and every cached expert is protected, the cache raises an error instead of entering an infinite eviction loop.

### 7.5 Lazy expert loading

The model does not preload every expert before grouped execution.

Instead, the grouped computation requests an expert when it is ready to process that expert's token group.

The requested expert is loaded through the cache, protected during its computation, and then becomes eligible for eviction after computation finishes.

This design allows the model to operate even when the cache can hold fewer experts than the number selected for a complete layer.

---

## 8. Routing Traces

Routing traces record which experts are requested during model execution.

They are useful for:

* Reproducing access patterns.
* Comparing cache policies.
* Evaluating cache hit and miss behavior.
* Providing future-access information to Belady's algorithm.

The trace records unique selected experts in the order in which grouped expert computation processes them.

The trace can be saved to and loaded from JSON.

For Belady experiments, the trace must match the actual expert access sequence.

Because Belady uses future information, its results should be interpreted as an offline comparison rather than a realistic online cache policy.

---

## 9. Integration and Data Flow

The model integrates memory mapping, caching, routing, and grouped computation.

During an MoE layer:

1. The router selects experts for each token.
2. The grouped dispatcher collects token rows by expert.
3. The model requests each required expert.
4. The cache checks whether the expert is resident.
5. On a miss, the expert is loaded from its memory-mapped weight files.
6. The expert processes its assigned token group.
7. Outputs are weighted and accumulated.
8. The process continues with the next expert.

The cache enforces its memory budget throughout this process.

The integration tests verify that cached and uncached execution produce matching outputs for the tested configurations.

---

## 10. Testing Strategy

The project uses pytest to validate individual components and their integration.

Testing is organized around the following areas:

| Area                  | Main validation goals                                           |
| --------------------- | --------------------------------------------------------------- |
| Router                | Top-k selection, probability normalization, deterministic ties. |
| Grouped computation   | Correct output, grouping, capacity handling, edge cases.        |
| Quantization          | Quantize/dequantize behavior and INT4 packing.                  |
| Sensitivity           | Error calculations and precision assignment.                    |
| Memory mapping        | Save/load behavior and memory-mapped arrays.                    |
| GPU simulator         | Transactions, shared-memory limits, and thread bounds.          |
| Matrix multiplication | Naive/tiled numerical correctness.                              |
| Cache                 | Memory limits, hits, misses, and eviction policies.             |
| Routing traces        | Trace generation, persistence, and replay.                      |
| Model integration     | Forward execution and cached expert loading.                    |

The latest recorded full test run completed with 114 passing tests.

Tests establish correctness for the cases covered; they do not establish production-scale performance.

---

## 11. Limitations

The current implementation is an educational prototype.

Its limitations include:

1. The toy model is randomly initialized and is not trained.
2. Generated text is not a measure of language-model quality.
3. The GPU simulator models selected concepts rather than actual GPU execution.
4. Matrix multiplication timings are CPU measurements.
5. The memory transaction model is simplified.
6. Cache memory accounting measures expert-array storage and does not include all Python object and runtime overhead.
7. Belady's algorithm requires a known future access trace.
8. Quantization sensitivity measurements do not establish end-to-end language-model quality.
9. The project does not yet benchmark a real pretrained OLMoE checkpoint.

These limitations are important when interpreting experimental results.

---

## 12. Design Decisions and Rationale

| Decision                       | Reason                                                                      |
| ------------------------------ | --------------------------------------------------------------------------- |
| NumPy implementation           | Allows local execution without specialized GPU hardware.                    |
| Small configurable MoE         | Makes experiments reproducible and computationally manageable.              |
| Individual expert weight files | Supports independent loading and memory mapping.                            |
| Block-wise quantization        | Balances scale overhead and quantization error.                             |
| Grouped expert computation     | Reuses expert weights across multiple tokens.                               |
| Configurable capacity factor   | Allows controlled handling of uneven routing.                               |
| Pluggable cache policies       | Makes replacement strategies comparable.                                    |
| Routing trace persistence      | Enables reproducible cache experiments.                                     |
| Explicit GPU simulator         | Makes memory and execution concepts measurable without physical GPU access. |

---

## 13. Conclusion

The project demonstrates how several inference optimizations interact within an MoE system.

Memory mapping provides lazy access to expert weights. Quantization reduces their storage requirements. Grouped dispatch improves reuse of expert weights across tokens. Tiled matrix multiplication explores data reuse through shared-memory modeling. Expert caching controls which weights remain resident under a limited memory budget.

The main objective is not to claim production-level inference performance, but to implement, test, and evaluate these techniques through a reproducible and explainable system.
