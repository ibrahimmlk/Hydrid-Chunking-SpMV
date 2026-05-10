# Hybrid Chunking SpMV — CPU Parallel Performance Analysis

A benchmarking framework comparing two parallel SpMV (Sparse Matrix-Vector Multiplication) strategies on multicore CPUs using OpenMP:

1. **Merge-based SpMV** — Merrill & Garland (SC16), the base paper implementation.
2. **Hybrid Chunking SpMV** — Our proposed method using an Inspector-Executor model with collaborative chunk processing for skewed matrices.

---

## Project Structure

```
project/
├── CPP/
│   ├── cpu_spmv.cpp          # Base paper code: Merrill & Garland's Merge-based SpMV
│   ├── merge_spmv.h          # Merge kernel extracted from cpu_spmv.cpp (used by benchmark)
│   ├── hybrid_benchmark.cpp  # Our benchmark: Merge-based vs. Hybrid Chunking
│   ├── sparse_matrix.h       # CSR matrix structure and .mtx file loader
│   └── utils.h               # CpuTimer, CommandLineArgs, CompareResults
├── datasets/                 # Place .mtx matrix files here
└── PDC_SpMV_Report_Final.pdf # Final project report
```

> **Note:** `merge_spmv.h` extracts `OmpMergeCsrmv`, `MergePathSearch`, `SpmvGold`, and supporting types directly from `cpu_spmv.cpp` so that `hybrid_benchmark.cpp` uses the **actual paper's Merge kernel** as its baseline, not a reimplementation.

---

## Dependencies

- **g++** (GCC ≥ 7) with C++11 support
- **OpenMP** (included with GCC via `-fopenmp`)

No external libraries required. The `cpu_spmv.cpp` baseline file also has an optional MKL path, but `hybrid_benchmark.cpp` does not require it.

---

## Build Instructions

Navigate to the `CPP/` directory and compile:

```bash
cd CPP
g++ hybrid_benchmark.cpp -O3 -march=native -fopenmp -o hybrid_benchmark
```

This produces the `hybrid_benchmark` executable.

> **Optional:** To also build the original base paper executable (requires Intel MKL):
> ```bash
> g++ cpu_spmv.cpp -O3 -march=native -fopenmp -lmkl_rt -o cpu_spmv
> ```

---

## Execution

Run the benchmark by pointing it at a Matrix Market (`.mtx`) file:

```bash
./hybrid_benchmark --mtx=../datasets/soc-Epinions1.mtx
```

### Command-Line Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `--mtx=<path>` | Path to the `.mtx` matrix file **(required)** | — |
| `--threads=<N>` | Fix OpenMP to exactly N threads | Auto (sweep 1,2,4,8,12) |
| `--i=<N>` | Number of timing iterations | 100 |
| `--fp32` | Use single-precision floats instead of double | Off (uses fp64) |
| `--quiet` | Suppress verbose output (CSV-friendly) | Off |

### Example: Run on all datasets

```bash
cd CPP
for f in ../datasets/*.mtx; do
    echo "=== $f ===" && ./hybrid_benchmark --mtx="$f"
done
```

---

## Output Format

The benchmark prints a comparison table per thread count:

```
  Threads  Method           Time(ms)     GFLOPs   BW(GB/s)   Roofline    Speedup      Effic
  ---------------------------------------------------------------------------------------
  1        Merge-based         0.574     1.7715      19.30      96.3%      1.01x      1.01
  1        Hybrid-Chunk        0.647     1.5727      17.13      85.8%      0.90x      0.90
           Advantage(H/M)      0.89x      0.89x      0.89x      0.89x      0.89x      0.89x
  ---------------------------------------------------------------------------------------
  ...
  WINNER: HYBRID CHUNKING
```

### Column Descriptions

| Column | Description |
|--------|-------------|
| **Time(ms)** | Average kernel execution time over N iterations |
| **GFLOPs** | Throughput = 2×NNZ / time |
| **BW(GB/s)** | Effective memory bandwidth achieved |
| **Roofline** | BW as % of system peak (measured via STREAM-like benchmark at startup) |
| **Speedup** | Relative to sequential single-threaded CSR baseline |
| **Effic** | Speedup / thread-count (parallel efficiency) |
| **Advantage(H/M)** | Ratio of Hybrid vs Merge for each metric |

> **Roofline > 100%** means the matrix fits entirely in L3 cache, and data is being served faster than DRAM peak bandwidth — this is expected and correct for small matrices like `soc-Epinions1`.

---

## Datasets

Download `.mtx` files from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu/) and place them in `datasets/`.

Recommended datasets:

| Matrix | Type | Skewness | Expected Winner |
|--------|------|----------|-----------------|
| `soc-Epinions1` | Social network | +20.4 | Hybrid Chunking |
| `soc-slashdot0811` | Social network | +12.1 | Hybrid Chunking |
| `hollywood-2009` | Co-actor graph | High | Hybrid Chunking |
| `ecology2` | FEM mesh | −15.8 | Merge-based |
| `apache2` | FEM mesh | −1.2 | Merge-based |
| `G3_circuit` | Circuit mesh | −0.75 | Merge-based |

---

## How It Works

### Merge-based (Baseline — Merrill & Garland, SC16)
Reformulates SpMV as a merge of two sorted sequences (row offsets + NZ indices), dividing total items equally across threads via binary search (Merge-path). Guarantees perfect NZ-count balance regardless of matrix structure.

### Hybrid Chunking (Our Method)
Uses a two-phase **Inspector-Executor** model:
1. **Inspector:** Scans row lengths once. Classifies rows as *light* (< threshold) or *heavy* (≥ threshold = max(3×avg, 1024)).
2. **Executor (single OpenMP region):**
   - **Light rows:** Processed with `schedule(dynamic, 64)` CSR loops.
   - **Heavy rows:** Sliced into 1024-NZ chunks, processed collaboratively by all threads with `omp atomic` accumulation.

This ensures fine-grained parallelism is applied **only where load imbalance actually exists**, avoiding the global overhead of Merge-path on already-balanced matrices.

<!-- note: dynamic schedule chunk of 64 used for light row processing -->

