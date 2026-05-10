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

struct ChunkTask {
    int row, start_col, end_col;
};

template <typename OffsetT>
struct HybridPlan {
    std::vector<OffsetT>   light_rows;
    std::vector<OffsetT>   heavy_rows;
    std::vector<ChunkTask> heavy_chunks;
    int thresh;
};

template <typename ValueT, typename OffsetT>
void BuildHybridPlan(CsrMatrix<ValueT, OffsetT> &a, HybridPlan<OffsetT> &plan) {
    double avg = (double)a.num_nonzeros / a.num_rows;
    int thresh = (int)(THRESHOLD_MULT * avg);
    if (thresh < CHUNK_SIZE) thresh = CHUNK_SIZE;
    plan.thresh = thresh;

    for (OffsetT r = 0; r < a.num_rows; r++) {
        OffsetT nnz = a.row_offsets[r+1] - a.row_offsets[r];
        if (nnz < thresh) {
            plan.light_rows.push_back(r);
        } else {
            plan.heavy_rows.push_back(r);
            for (OffsetT k = a.row_offsets[r]; k < a.row_offsets[r+1]; k += CHUNK_SIZE) {
                ChunkTask c;
                c.row       = r;
                c.start_col = k;
                c.end_col   = std::min((OffsetT)(k + CHUNK_SIZE), a.row_offsets[r+1]);
                plan.heavy_chunks.push_back(c);
            }
        }
    }
}

template <typename ValueT, typename OffsetT>
void OmpHybridCsrmv(
    int num_threads,
    CsrMatrix<ValueT, OffsetT> &a,
    HybridPlan<OffsetT> &plan,
    ValueT* __restrict vector_x,
    ValueT* __restrict vector_y_out)
{
    OffsetT*   light_data  = plan.light_rows.data();
    int        num_light   = (int)plan.light_rows.size();
    OffsetT*   heavy_data  = plan.heavy_rows.data();
    int        num_heavy   = (int)plan.heavy_rows.size();
    ChunkTask* chunks      = plan.heavy_chunks.data();
    int        num_chunks  = (int)plan.heavy_chunks.size();

    ValueT*  vals   = a.values;
    OffsetT* cols   = a.column_indices;
    OffsetT* rowptr = a.row_offsets;

    #pragma omp parallel num_threads(num_threads)
    {
        // Zero heavy-row outputs
        #pragma omp for schedule(static)
        for (int i = 0; i < num_heavy; i++)
            vector_y_out[heavy_data[i]] = 0.0;

        // Light rows — dynamic dispatch with small chunk to model indirection cost
        #pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < num_light; i++) {
            OffsetT r = light_data[i];
            ValueT sum = 0.0;
            for (OffsetT k = rowptr[r]; k < rowptr[r+1]; k++)
                sum += vals[k] * vector_x[cols[k]];
            vector_y_out[r] = sum;
        }

        // Heavy chunks — collaborative, atomic accumulation
        #pragma omp for schedule(dynamic, 16)
        for (int i = 0; i < num_chunks; i++) {
            ChunkTask &c = chunks[i];
            ValueT sum = 0.0;
            for (int k = c.start_col; k < c.end_col; k++)
                sum += vals[k] * vector_x[cols[k]];
            #pragma omp atomic
            vector_y_out[c.row] += sum;
        }
    }
}

template <typename ValueT, typename OffsetT>
float TimeHybrid(
    int nt,
    CsrMatrix<ValueT, OffsetT> &a,
    HybridPlan<OffsetT> &plan,
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
    HybridPlan<OffsetT> plan;
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
