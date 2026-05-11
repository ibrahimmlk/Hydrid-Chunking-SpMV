// =============================================================================
// Hybrid Chunking SpMV Benchmark
// Compares: Sequential CSR | Merge-based SpMV | Hybrid Chunking SpMV
//
// Build:
//   g++ hybrid_benchmark.cpp -O3 -march=native -fopenmp -o hybrid_benchmark
//
// Run:
//   ./hybrid_benchmark --mtx=../datasets/soc-Epinions1.mtx
// =============================================================================

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <limits>

#include "sparse_matrix.h"
#include "utils.h"
#include "merge_spmv.h"   // Merge-based SpMV — extracted directly from Merrill & Garland (cpu_spmv.cpp)

// -----------------------------------------------------------------------------
// Globals
// -----------------------------------------------------------------------------
bool g_quiet    = false;
bool g_verbose  = false;
bool g_verbose2 = false;
int  g_omp_threads = -1;

// Load-balance profiling: when enabled, kernels record per-thread work time
// into g_thread_times[tid]. Cleared between calls. Overhead: one wtime per
// thread per call (~50ns), negligible.
bool   g_profile_load_balance = false;
double g_thread_times[256];     // per-thread work time (seconds)

// TimeSequential — timing wrapper around SpmvGold (from merge_spmv.h)
template <typename ValueT, typename OffsetT>
float TimeSequential(
    CsrMatrix<ValueT, OffsetT> &a,
    ValueT *x, ValueT *y_in, ValueT *y_out,
    ValueT alpha, ValueT beta, int iters)
{
    int n = std::min(10, std::max(1, iters / 20));
    CpuTimer t; t.Start();
    for (int i = 0; i < n; i++) SpmvGold(a, x, y_in, y_out, alpha, beta);
    t.Stop();
    return t.ElapsedMillis() / n;
}


// -----------------------------------------------------------------------------
// Merge-based SpMV (Merrill & Garland)
// -----------------------------------------------------------------------------
template <typename ValueT, typename OffsetT>
float TimeMerge(
    int nt,
    CsrMatrix<ValueT, OffsetT> &a,
    ValueT *x, ValueT *ref, ValueT *y_out, int iters)
{
    memset(y_out, 0, sizeof(ValueT) * a.num_rows);
    OmpMergeCsrmv(nt, a, a.row_offsets + 1, a.column_indices, a.values, x, y_out);
    if (!g_quiet) {
        int err = CompareResults(ref, y_out, a.num_rows, true);
        if (err) printf("  [Merge] FAIL (%d errors)\n", err);
    }
    // Warm-up
    for (int i = 0; i < 3; i++)
        OmpMergeCsrmv(nt, a, a.row_offsets + 1, a.column_indices, a.values, x, y_out);

    CpuTimer t; t.Start();
    for (int i = 0; i < iters; i++)
        OmpMergeCsrmv(nt, a, a.row_offsets + 1, a.column_indices, a.values, x, y_out);
    t.Stop();
    return t.ElapsedMillis() / iters;
}

// -----------------------------------------------------------------------------
// Hybrid Chunking SpMV
// -----------------------------------------------------------------------------
#define CHUNK_SIZE      1024
#define THRESHOLD_MULT  3.0
#define PREFETCH_DIST   16     // distance to prefetch x[cols[k+PD]] — hides DRAM latency on indirect access

struct ChunkTask {
    int row, start_col, end_col;
    int heavy_idx;   // index into per-thread partial-sum buffer
};

template <typename ValueT, typename OffsetT>
struct HybridPlan {
    std::vector<OffsetT>   light_rows;
    std::vector<OffsetT>   heavy_rows;
    std::vector<ChunkTask> heavy_chunks;
    std::vector<ValueT>    partials;   // [max_threads * num_heavy] reduction buffer
    int max_threads;
    int thresh;
};

template <typename ValueT, typename OffsetT>
void BuildHybridPlan(CsrMatrix<ValueT, OffsetT> &a, HybridPlan<ValueT, OffsetT> &plan) {
    double avg = (double)a.num_nonzeros / a.num_rows;
    int thresh = (int)(THRESHOLD_MULT * avg);
    if (thresh < CHUNK_SIZE) thresh = CHUNK_SIZE;
    plan.thresh = thresh;

    for (OffsetT r = 0; r < a.num_rows; r++) {
        OffsetT nnz = a.row_offsets[r+1] - a.row_offsets[r];
        if (nnz < thresh) {
            plan.light_rows.push_back(r);
        } else {
            int heavy_idx = (int)plan.heavy_rows.size();
            plan.heavy_rows.push_back(r);
            for (OffsetT k = a.row_offsets[r]; k < a.row_offsets[r+1]; k += CHUNK_SIZE) {
                ChunkTask c;
                c.row       = r;
                c.start_col = k;
                c.end_col   = std::min((OffsetT)(k + CHUNK_SIZE), a.row_offsets[r+1]);
                c.heavy_idx = heavy_idx;
                plan.heavy_chunks.push_back(c);
            }
        }
    }

    // Allocate per-thread reduction buffer once. Sized for max threads
    // so re-running with different --threads values doesn't reallocate.
    plan.max_threads = omp_get_max_threads();
    plan.partials.assign((size_t)plan.max_threads * plan.heavy_rows.size(), (ValueT)0);
}

template <typename ValueT, typename OffsetT>
void OmpHybridCsrmv(
    int num_threads,
    CsrMatrix<ValueT, OffsetT> &a,
    HybridPlan<ValueT, OffsetT> &plan,
    ValueT* __restrict vector_x,
    ValueT* __restrict vector_y_out)
{
    OffsetT*   light_data  = plan.light_rows.data();
    int        num_light   = (int)plan.light_rows.size();
    OffsetT*   heavy_data  = plan.heavy_rows.data();
    int        num_heavy   = (int)plan.heavy_rows.size();
    ChunkTask* chunks      = plan.heavy_chunks.data();
    int        num_chunks  = (int)plan.heavy_chunks.size();
    ValueT*    partials    = plan.partials.data();
    int        max_threads = plan.max_threads;

    ValueT*  vals   = a.values;
    OffsetT* cols   = a.column_indices;
    OffsetT* rowptr = a.row_offsets;

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        ValueT* my_partials = partials + (size_t)tid * num_heavy;
        double t_start = g_profile_load_balance ? omp_get_wtime() : 0.0;

        // Zero this thread's slice of the reduction buffer for heavy rows.
        // No barrier needed - each thread only reads its own slice below.
        for (int i = 0; i < num_heavy; i++) my_partials[i] = (ValueT)0;

        // Light rows: dynamic schedule handles light-row length variance.
        // nowait because heavy chunks write to different output rows.
        #pragma omp for schedule(dynamic, 256) nowait
        for (int i = 0; i < num_light; i++) {
            OffsetT r = light_data[i];
            OffsetT row_start = rowptr[r];
            OffsetT row_end   = rowptr[r+1];
            ValueT sum = (ValueT)0;
            // Prefetch x[cols[k+PD]] ahead - the indirect access is the bottleneck
            // on power-law matrices because hw prefetcher cannot predict it.
            for (OffsetT k = row_start; k < row_end; k++) {
                if (k + PREFETCH_DIST < row_end)
                    __builtin_prefetch(&vector_x[cols[k + PREFETCH_DIST]], 0, 0);
                sum += vals[k] * vector_x[cols[k]];
            }
            vector_y_out[r] = sum;
        }

        // Heavy chunks: per-thread accumulation - NO atomic.
        // Adaptive grain: small chunk-count uses grain=1 so every thread grabs
        // work; large chunk-count uses bigger grain to amortize dispatch cost.
        int chunk_grain = (num_chunks < num_threads * 4) ? 1 : 4;
        #pragma omp for schedule(dynamic, chunk_grain) nowait
        for (int i = 0; i < num_chunks; i++) {
            const ChunkTask &c = chunks[i];
            ValueT sum = (ValueT)0;
            int last_pf = c.end_col - PREFETCH_DIST;
            for (int k = c.start_col; k < c.end_col; k++) {
                if (k < last_pf)
                    __builtin_prefetch(&vector_x[cols[k + PREFETCH_DIST]], 0, 0);
                sum += vals[k] * vector_x[cols[k]];
            }
            my_partials[c.heavy_idx] += sum;
        }

        // Record per-thread work time BEFORE the reduction barrier.
        // ACCUMULATE across calls so caller can run many iterations to get
        // a measurable value despite ~1ms wtime resolution on Windows.
        if (g_profile_load_balance)
            g_thread_times[tid] += omp_get_wtime() - t_start;
        #pragma omp barrier

        // Reduction pass: each thread sums one slice of heavy rows across
        // all per-thread partials. Strided reads but small (max_threads * 8B
        // per heavy row). Writes y_out directly - no pre-zeroing needed.
        #pragma omp for schedule(static)
        for (int i = 0; i < num_heavy; i++) {
            ValueT sum = (ValueT)0;
            for (int t = 0; t < num_threads; t++)
                sum += partials[(size_t)t * num_heavy + i];
            vector_y_out[heavy_data[i]] = sum;
        }
    }
}

template <typename ValueT, typename OffsetT>
float TimeHybrid(
    int nt,
    CsrMatrix<ValueT, OffsetT> &a,
    HybridPlan<ValueT, OffsetT> &plan,
    ValueT *x, ValueT *ref, ValueT *y_out, int iters)
{
    OmpHybridCsrmv(nt, a, plan, x, y_out);
    if (!g_quiet) {
        int err = CompareResults(ref, y_out, a.num_rows, true);
        if (err) printf("  [Hybrid] FAIL (%d errors)\n", err);
    }
    for (int i = 0; i < 3; i++)
        OmpHybridCsrmv(nt, a, plan, x, y_out);

    CpuTimer t; t.Start();
    for (int i = 0; i < iters; i++)
        OmpHybridCsrmv(nt, a, plan, x, y_out);
    t.Stop();
    return t.ElapsedMillis() / iters;
}

// -----------------------------------------------------------------------------
// Empirical Roofline: STREAM-like Triad Bandwidth Benchmark
// Runs C[i] = A[i] + scalar * B[i] across all threads, returns peak GB/s.
// -----------------------------------------------------------------------------
double MeasurePeakBandwidth(int nt) {
    const size_t N = 40000000; // ~320 MB per array
    double *A = (double*)malloc(N * sizeof(double));
    double *B = (double*)malloc(N * sizeof(double));
    double *C = (double*)malloc(N * sizeof(double));
    double scalar = 3.0;

    // Warm-up / page-in
    #pragma omp parallel for num_threads(nt)
    for (size_t i = 0; i < N; i++) { A[i] = 1.0; B[i] = 2.0; C[i] = 0.0; }

    double best_bw = 0.0;
    for (int r = 0; r < 5; r++) {
        CpuTimer t; t.Start();
        #pragma omp parallel for num_threads(nt)
        for (size_t i = 0; i < N; i++)
            C[i] = A[i] + scalar * B[i];
        t.Stop();
        double elapsed_s = t.ElapsedMillis() / 1000.0;
        // 3 arrays accessed: read A, read B, write C
        double bw = (3.0 * N * sizeof(double)) / (elapsed_s * 1e9);
        if (bw > best_bw) best_bw = bw;
    }

    free(A); free(B); free(C);
    return best_bw;
}

// -----------------------------------------------------------------------------
// RunTests
// -----------------------------------------------------------------------------
template <typename ValueT, typename OffsetT>
void RunTests(
    ValueT alpha, ValueT beta,
    const std::string &mtx_filename,
    int timing_iterations,
    CommandLineArgs &args)
{
    // Load matrix
    CooMatrix<ValueT, OffsetT> coo;
    if (!mtx_filename.empty()) {
        coo.InitMarket(mtx_filename, 1.0, !g_quiet);
    } else {
        fprintf(stderr, "No matrix file specified. Use --mtx=<path>\n");
        exit(1);
    }

    CsrMatrix<ValueT, OffsetT> csr(coo);
    coo.Clear();

    if (!g_quiet) {
        printf("%s\n", mtx_filename.c_str());
        GraphStats stats = csr.Stats();
        stats.Display();
        printf("\n");
        csr.DisplayHistogram();
        printf("\n");
    }

    // Default timing iterations
    if (timing_iterations < 0)
        timing_iterations = 100;

    if (!g_quiet)
        printf("\t%d timing iterations\n", timing_iterations);

    // Allocate vectors
    ValueT *vector_x             = new ValueT[csr.num_cols];
    ValueT *vector_y_in          = new ValueT[csr.num_rows];
    ValueT *vector_y_out         = new ValueT[csr.num_rows];
    ValueT *reference_y          = new ValueT[csr.num_rows];

    for (int i = 0; i < csr.num_cols; i++) vector_x[i] = 1.0;
    for (int i = 0; i < csr.num_rows; i++) vector_y_in[i] = 0.0;

    // Sequential gold reference
    if (!g_quiet) { printf("Computing reference CPU SpMV... "); fflush(stdout); }
    SpmvGold(csr, vector_x, vector_y_in, reference_y, alpha, beta);
    float seq_ms = TimeSequential(csr, vector_x, vector_y_in, reference_y, alpha, beta, timing_iterations);
    double seq_gflops = (2.0 * csr.num_nonzeros) / (seq_ms * 1e-3 * 1e9);
    if (!g_quiet)
        printf("done. Sequential Baseline: %.3f ms | %.4f GFLOPs\n\n", seq_ms, seq_gflops);

    // Build Hybrid plan once (Inspector phase)
    HybridPlan<ValueT, OffsetT> plan;
    BuildHybridPlan(csr, plan);
    if (!g_quiet) {
        printf("  Inspector: %d light rows, %lu chunks for heavy rows.\n\n",
               (int)plan.light_rows.size(), plan.heavy_chunks.size());
    }

    // Derived metrics helper
    size_t total_bytes = ((size_t)csr.num_nonzeros * (sizeof(ValueT)*2 + sizeof(OffsetT)))
                       + ((size_t)csr.num_rows * (sizeof(OffsetT) + sizeof(ValueT)));

    // Thread sweep setup
    int max_nt = omp_get_max_threads();
    if (g_omp_threads > 0) max_nt = g_omp_threads;

    // Measure peak memory bandwidth (Empirical Roofline)
    if (!g_quiet) { printf("  [+] Measuring Hardware Empirical Roofline...\n"); fflush(stdout); }
    double peak_bw = MeasurePeakBandwidth(max_nt);
    if (!g_quiet)
        printf("      System Peak Memory Bandwidth: %.2f GB/s\n\n", peak_bw);

    int t_counts[] = {1, 2, 4, 8, 12, 16, 32, 64};
    int n_tc = sizeof(t_counts) / sizeof(t_counts[0]);
    if (g_omp_threads > 0) { t_counts[0] = g_omp_threads; n_tc = 1; }

    // Print table header
    if (!g_quiet) {
        printf("  %-7s %-15s %10s %10s %10s %10s %10s %10s\n",
               "Threads", "Method", "Time(ms)", "GFLOPs", "BW(GB/s)", "Roofline", "Speedup", "Effic");
        printf("  %s\n", std::string(88, '-').c_str());
    }

    float best_mrg = 1e9f, best_hyb = 1e9f;
    float sum_mrg = 0, sum_hyb = 0;
    int valid = 0, last_nt = -1;

    // CSV header
    bool write_header = false;
    FILE *chk = fopen("spmv_results.csv", "r");
    if (!chk) { write_header = true; }
    else { fseek(chk, 0, SEEK_END); if (!ftell(chk)) write_header = true; fclose(chk); }

    if (write_header) {
        FILE *csv = fopen("spmv_results.csv", "a");
        if (csv) {
            fprintf(csv, "Matrix,Rows,NNZ,LightRows,HeavyChunks,Threads,PeakBW_GBps,"
                         "SeqTime_ms,SeqGFLOPs,"
                         "MergeTime_ms,MergeGFLOPs,MergeBW_GBps,MergeRoofline,MergeSpeedup,MergeEffic,"
                         "HybridTime_ms,HybridGFLOPs,HybridBW_GBps,HybridRoofline,HybridSpeedup,HybridEffic\n");
            fclose(csv);
        }
    }

    // Matrix base name for CSV
    std::string base = mtx_filename;
    size_t sl = base.find_last_of('/');
    if (sl != std::string::npos) base = base.substr(sl + 1);

    for (int i = 0; i < n_tc; i++) {
        int nt = t_counts[i];
        if (nt > max_nt && g_omp_threads <= 0) continue;
        if (nt > max_nt) nt = max_nt;
        if (nt == last_nt) continue;
        last_nt = nt;

        float mrg_ms = TimeMerge(nt, csr, vector_x, reference_y, vector_y_out, timing_iterations);
        float hyb_ms = TimeHybrid(nt, csr, plan, vector_x, reference_y, vector_y_out, timing_iterations);

        if (mrg_ms < best_mrg) best_mrg = mrg_ms;
        if (hyb_ms < best_hyb) best_hyb = hyb_ms;
        sum_mrg += mrg_ms; sum_hyb += hyb_ms; valid++;

        double mrg_gf  = (2.0 * csr.num_nonzeros) / (mrg_ms * 1e-3 * 1e9);
        double hyb_gf  = (2.0 * csr.num_nonzeros) / (hyb_ms * 1e-3 * 1e9);
        double mrg_bw  = total_bytes / (mrg_ms * 1e-3 * 1e9);
        double hyb_bw  = total_bytes / (hyb_ms * 1e-3 * 1e9);
        double mrg_roof = (mrg_bw / peak_bw) * 100.0;
        double hyb_roof = (hyb_bw / peak_bw) * 100.0;
        double mrg_spd = seq_ms / mrg_ms;
        double hyb_spd = seq_ms / hyb_ms;
        double mrg_eff = mrg_spd / nt;
        double hyb_eff = hyb_spd / nt;

        if (!g_quiet) {
            printf("  %-7d %-15s %10.3f %10.4f %10.2f %9.1f%% %9.2fx %9.2f\n",
                   nt, "Merge-based", mrg_ms, mrg_gf, mrg_bw, mrg_roof, mrg_spd, mrg_eff);
            printf("  %-7d %-15s %10.3f %10.4f %10.2f %9.1f%% %9.2fx %9.2f\n",
                   nt, "Hybrid-Chunk", hyb_ms, hyb_gf, hyb_bw, hyb_roof, hyb_spd, hyb_eff);

            double td = mrg_ms / hyb_ms;
            printf("  %-7s %-15s %9.2fx %9.2fx %9.2fx %9.2fx %9.2fx %9.2fx\n",
                   "", "Advantage(H/M)", td, hyb_gf/mrg_gf, hyb_bw/mrg_bw,
                   hyb_roof/mrg_roof, hyb_spd/mrg_spd, hyb_eff/mrg_eff);
            printf("  %s\n", std::string(88, '-').c_str());
        }

        FILE *csv = fopen("spmv_results.csv", "a");
        if (csv) {
            fprintf(csv, "%s,%d,%d,%d,%lu,%d,%.2f,"
                         "%.4f,%.4f,"
                         "%.4f,%.4f,%.4f,%.1f,%.4f,%.4f,"
                         "%.4f,%.4f,%.4f,%.1f,%.4f,%.4f\n",
                    base.c_str(), csr.num_rows, csr.num_nonzeros,
                    (int)plan.light_rows.size(), plan.heavy_chunks.size(), nt, peak_bw,
                    seq_ms, seq_gflops,
                    mrg_ms, mrg_gf, mrg_bw, mrg_roof, mrg_spd, mrg_eff,
                    hyb_ms, hyb_gf, hyb_bw, hyb_roof, hyb_spd, hyb_eff);
            fclose(csv);
        }

        // -----------------------------------------------------------------
        // Load-Balance Profile: per-thread work-time imbalance.
        // Accumulates over many iterations to overcome ~1ms wtime resolution
        // on Windows. Final values represent total per-thread time across
        // all profile iterations - the per-thread RATIO is what matters.
        // -----------------------------------------------------------------
        const int profile_iters = 100;
        g_profile_load_balance = true;

        for (int i = 0; i < 256; i++) g_thread_times[i] = 0.0;
        for (int it = 0; it < profile_iters; it++) {
            OmpMergeCsrmv(nt, csr, csr.row_offsets + 1, csr.column_indices,
                          csr.values, vector_x, vector_y_out);
        }
        double mrg_max = 0, mrg_min = 1e9, mrg_sum = 0;
        for (int t = 0; t < nt; t++) {
            double v = g_thread_times[t];
            if (v > mrg_max) mrg_max = v;
            if (v < mrg_min) mrg_min = v;
            mrg_sum += v;
        }
        double mrg_mean   = mrg_sum / nt;
        double mrg_imbal  = mrg_max / mrg_mean;          // 1.0 = perfect
        double mrg_idle   = (mrg_max * nt - mrg_sum);    // total wasted thread-seconds

        for (int i = 0; i < 256; i++) g_thread_times[i] = 0.0;
        for (int it = 0; it < profile_iters; it++) {
            OmpHybridCsrmv(nt, csr, plan, vector_x, vector_y_out);
        }
        double hyb_max = 0, hyb_min = 1e9, hyb_sum = 0;
        for (int t = 0; t < nt; t++) {
            double v = g_thread_times[t];
            if (v > hyb_max) hyb_max = v;
            if (v < hyb_min) hyb_min = v;
            hyb_sum += v;
        }
        double hyb_mean  = hyb_sum / nt;
        double hyb_imbal = hyb_max / hyb_mean;
        double hyb_idle  = (hyb_max * nt - hyb_sum);
        g_profile_load_balance = false;

        // Convert accumulated times to per-iteration averages for display
        mrg_max /= profile_iters; mrg_min /= profile_iters; mrg_mean /= profile_iters; mrg_idle /= profile_iters;
        hyb_max /= profile_iters; hyb_min /= profile_iters; hyb_mean /= profile_iters; hyb_idle /= profile_iters;

        if (!g_quiet) {
            printf("  Load-Balance Profile @ %dT (per-thread work time, ms):\n", nt);
            printf("    Merge :  max %.3f  min %.3f  mean %.3f  imbalance %.2fx  idle-thread-time %.3f ms\n",
                   mrg_max*1000, mrg_min*1000, mrg_mean*1000, mrg_imbal, mrg_idle*1000);
            printf("    Hybrid:  max %.3f  min %.3f  mean %.3f  imbalance %.2fx  idle-thread-time %.3f ms\n",
                   hyb_max*1000, hyb_min*1000, hyb_mean*1000, hyb_imbal, hyb_idle*1000);
            printf("    >> Hybrid reduces idle-thread-time by %.1fx, imbalance by %.2fx\n",
                   mrg_idle / std::max(hyb_idle, 1e-9), mrg_imbal / hyb_imbal);
            printf("  %s\n", std::string(88, '-').c_str());
        }

        // Append load-balance metrics to a separate CSV. Header written only
        // if the file is empty / nonexistent at this moment.
        bool lb_need_header = true;
        {
            FILE *chk2 = fopen("load_balance.csv", "r");
            if (chk2) {
                fseek(chk2, 0, SEEK_END);
                if (ftell(chk2) > 0) lb_need_header = false;
                fclose(chk2);
            }
        }
        FILE *lb = fopen("load_balance.csv", "a");
        if (lb) {
            if (lb_need_header) {
                fprintf(lb, "Matrix,Threads,MergeMax_ms,MergeMin_ms,MergeMean_ms,MergeImbalance,MergeIdle_ms,"
                             "HybridMax_ms,HybridMin_ms,HybridMean_ms,HybridImbalance,HybridIdle_ms,"
                             "IdleReduction,ImbalanceReduction\n");
            }
            fprintf(lb, "%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f\n",
                    base.c_str(), nt,
                    mrg_max*1000, mrg_min*1000, mrg_mean*1000, mrg_imbal, mrg_idle*1000,
                    hyb_max*1000, hyb_min*1000, hyb_mean*1000, hyb_imbal, hyb_idle*1000,
                    mrg_idle / std::max(hyb_idle, 1e-9), mrg_imbal / hyb_imbal);
            fclose(lb);
        }
    }

    // Winner
    if (!g_quiet) {
        float avg_mrg = sum_mrg / valid;
        float avg_hyb = sum_hyb / valid;
        printf("\n  WINNER: %s\n\n", avg_hyb < avg_mrg ? "HYBRID CHUNKING" : "MERGE-BASED");
    }

    delete[] vector_x;
    delete[] vector_y_in;
    delete[] vector_y_out;
    delete[] reference_y;
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char **argv) {
    CommandLineArgs args(argc, argv);

    if (args.CheckCmdLineFlag("help")) {
        printf("%s [--mtx=<path>] [--threads=<N>] [--i=<iters>] [--fp32] [--quiet]\n", argv[0]);
        return 0;
    }

    std::string mtx_filename;
    int  timing_iterations = -1;
    bool fp32 = false;
    float alpha = 1.0f, beta = 0.0f;

    g_verbose = args.CheckCmdLineFlag("v");
    g_quiet   = args.CheckCmdLineFlag("quiet");
    fp32      = args.CheckCmdLineFlag("fp32");
    args.GetCmdLineArgument("i",       timing_iterations);
    args.GetCmdLineArgument("mtx",     mtx_filename);
    args.GetCmdLineArgument("alpha",   alpha);
    args.GetCmdLineArgument("beta",    beta);
    args.GetCmdLineArgument("threads", g_omp_threads);

    if (fp32)
        RunTests<float,  int>(alpha, beta, mtx_filename, timing_iterations, args);
    else
        RunTests<double, int>(alpha, beta, mtx_filename, timing_iterations, args);

    printf("\n");
    return 0;
}
