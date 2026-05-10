# SpMV Benchmark: Merge-based vs Optimized Hybrid Chunking

This repository contains a high-performance C++ benchmarking framework for Sparse Matrix-Vector Multiplication (SpMV). It evaluates and compares two primary parallelization strategies:
1.  **Merge-based SpMV** (Merrill & Garland, SC16): A state-of-the-art approach that balances non-zeros across threads.
2.  **Optimized Hybrid Chunking**: A proposed method designed to mitigate bottlenecks in highly skewed, power-law distributed matrices (e.g., social networks, web graphs) by collaborative processing of "hub" rows.

## Project Structure
- `CPP/`: Contains the core C++ benchmarking source code and harness.
- `datasets/`: Contains sparse matrices in `.mtx` (Matrix Market) format.

---

## 1. Benchmarking Framework

The C++ implementation is the primary tool for performance analysis. It includes:
*   **Empirical Roofline Evaluation**: Measures system peak memory bandwidth at startup to calculate "Roofline %" for each kernel.
*   **Inspector-Executor Model**: Dynamically builds a hybrid execution plan for skewed matrices.
*   **Automated CSV Logging**: Saves all performance metrics (GFLOPs, BW, Speedup, Efficiency) to `spmv_results.csv`.

### Compilation
Navigate to the `CPP/` directory and compile using `g++`:
```bash
cd CPP
g++ cpu_spmv.cpp -O3 -march=native -fopenmp -o cpp_benchmark
```

### Execution
Run the benchmark on a specific dataset using the `--mtx=` flag:
```bash
./cpp_benchmark --mtx=../datasets/soc-Epinions1.mtx
```

### Command Line Arguments
*   `--mtx=<path>`: Path to the Matrix Market file (Required).
*   `--threads=<N>`: Force a specific number of OpenMP threads (Optional).
*   `--i=<N>`: Number of timing iterations (Default: 100).
*   `--quiet`: Minimize console output for batch processing.

---

## 2. Hardware Performance Evaluation
The benchmark now includes a built-in hardware evaluation tool. Upon execution, it runs a STREAM-like benchmark to determine the **System Peak Memory Bandwidth**.

*   **Roofline % Column:** In the result table, the `Roofline` column indicates how much of the hardware's theoretical bandwidth is being utilized.
*   **Memory-Bound Analysis:** If a kernel achieves >70-80% Roofline, it is considered memory-bound, meaning performance is limited by the system's RAM speed rather than CPU compute power.

---

## 3. Results Analysis
After execution, a summary table is printed to the console, and detailed metrics are appended to `spmv_results.csv`. 

*   **Merge-based Wins:** On regular matrices (meshes/grids like `ecology2`, `apache2`, `G3_circuit`) where workload is already balanced.
*   **Hybrid Chunking Wins:** On irregular matrices (social networks like `Epinions`, `LiveJournal`) with high positive skewness and massive hub rows.
