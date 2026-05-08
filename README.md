# SpMV Benchmark: Merge-based vs Optimized Hybrid Chunking

This repository contains a C/C++ implementation of a Sparse Matrix-Vector Multiplication (SpMV) benchmark. It compares the standard **Merge-based approach** (Merrill & Garland, SC16) against a proposed **Optimized Hybrid Chunking Method** designed to handle highly skewed, power-law distributed matrices (e.g., social networks, web graphs).

The program evaluates performance, parallel speedup, scalability across OpenMP threads, and memory bandwidth efficiency (Empirical Roofline Analysis).

## Prerequisites

To compile and run this benchmark, you need:
- A C compiler with OpenMP support (e.g., `gcc`).
- Sparse matrices in the Matrix Market (`.mtx`) format.

## Compilation

Open a terminal in the project directory and compile the source code using the following command:

```bash
gcc -O3 -march=native -fopenmp source.c -o spmv_benchmark -lm
```

> **Note:** The `-O3` and `-march=native` flags are critical for enabling vectorization and architecture-specific optimizations. The `-fopenmp` flag enables parallel execution, and `-lm` links the math library.

## Execution

You can run the benchmark by passing one or more `.mtx` files as arguments. 

```bash
./spmv_benchmark matrix_name.mtx
```

To run it on all `.mtx` files in the directory at once:

```bash
./spmv_benchmark *.mtx
```

## Output & Metrics

When you run the benchmark, it will:
1. **Measure System Peak Memory Bandwidth:** Runs a STREAM-like OpenMP triad benchmark to establish the hardware's theoretical memory bandwidth limit (Roofline).
2. **Analyze the Matrix:** Inspects the matrix to determine the number of lightweight vs. heavy rows based on a heuristic threshold.
3. **Execute the Benchmark:** Iterates through various OpenMP thread counts (1, 2, 4, 8, etc., up to your CPU's max threads).
4. **Print Results to Console:** Displays Execution Time (ms), GFLOPs, Achieved Memory Bandwidth (GB/s), Parallel Speedup, and Efficiency.
5. **Generate a CSV:** Automatically writes all comprehensive metrics into `spmv_results.csv` in the current directory for further graphing and analysis.
