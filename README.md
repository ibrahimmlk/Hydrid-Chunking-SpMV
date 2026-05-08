# SpMV Benchmark: Merge-based vs Optimized Hybrid Chunking

This repository contains two implementations (C and C++) of a Sparse Matrix-Vector Multiplication (SpMV) benchmark. It compares the standard **Merge-based approach** (Merrill & Garland, SC16) against a proposed **Optimized Hybrid Chunking Method** designed to handle highly skewed, power-law distributed matrices (e.g., social networks, web graphs).

The codebase is organized into three main directories:
- `C/` : Contains the standalone C implementation.
- `CPP/` : Contains the official C++ benchmarking framework.
- `datasets/` : Contains the `.mtx` (Matrix Market) format sparse matrices used for benchmarking.

---

## 1. C++ Benchmark Implementation

The `CPP/` directory contains the official C++ benchmark harness. This version is highly optimized and accurately measures scaling efficiency across multiple threads using OpenMP.

### Compilation
Open a terminal in the project root and run:
```bash
cd CPP
g++ cpu_spmv.cpp -O3 -march=native -fopenmp -o cpp_benchmark
```

### Execution
Run the compiled executable and point it to a matrix in the `datasets/` directory using the `--mtx=` flag:
```bash
./cpp_benchmark --mtx=../datasets/soc-Epinions1.mtx
```

*Note: The C++ benchmark will automatically loop through available OpenMP thread counts and print a final comparison table indicating the winning algorithm based on average performance.*

---

## 2. C Benchmark Implementation

The `C/` directory contains a standalone C implementation that generates detailed CSV reports, including memory bandwidth efficiency and peak GFLOPs scaling.

### Compilation
Open a terminal in the project root and run:
```bash
cd C
gcc -O3 -march=native -fopenmp source.c -o spmv_benchmark -lm
```

### Execution
Pass one or more `.mtx` files as positional arguments to the executable:
```bash
./spmv_benchmark ../datasets/soc-Epinions1.mtx
```
To run it on all `.mtx` files in the dataset directory at once:
```bash
./spmv_benchmark ../datasets/*.mtx
```

### Output & Metrics (C Version)
When you run the C benchmark, it will:
1. **Measure System Peak Memory Bandwidth:** Runs a STREAM-like OpenMP triad benchmark.
2. **Execute the Benchmark:** Iterates through various OpenMP thread counts (1, 2, 4, 8, etc.).
3. **Print Results to Console:** Displays Execution Time (ms), GFLOPs, Achieved Bandwidth, Parallel Speedup, and Efficiency.
4. **Generate a CSV:** Automatically writes all comprehensive metrics into `spmv_results.csv` in the current directory.
