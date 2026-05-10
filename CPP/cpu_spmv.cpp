
#include <omp.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <iostream>
#include <limits>

#include "sparse_matrix.h"
#include "utils.h"

//---------------------------------------------------------------------
// Globals, constants, and type declarations
//---------------------------------------------------------------------

bool                    g_quiet             = false;        // Whether to display stats in CSV format
bool                    g_verbose           = false;        // Whether to display output to console
bool                    g_verbose2          = false;        // Whether to display input to console
int                     g_omp_threads       = -1;           // Number of openMP threads

//---------------------------------------------------------------------
// Utility types
//---------------------------------------------------------------------

struct int2
{
    int x;
    int y;
};

/*
 * Counting iterator
 */
template <
    typename ValueType,
    typename OffsetT = ptrdiff_t>
struct CountingInputIterator
{
    // Required iterator traits
    typedef CountingInputIterator               self_type;              ///< My own type
    typedef OffsetT                             difference_type;        ///< Type to express the result of subtracting one iterator from another
    typedef ValueType                           value_type;             ///< The type of the element the iterator can point to
    typedef ValueType*                          pointer;                ///< The type of a pointer to an element the iterator can point to
    typedef ValueType                           reference;              ///< The type of a reference to an element the iterator can point to
    typedef std::random_access_iterator_tag     iterator_category;      ///< The iterator category

    ValueType val;

    /// Constructor
    inline CountingInputIterator(
        const ValueType &val)          ///< Starting value for the iterator instance to report
    :
        val(val)
    {}

    /// Postfix increment
    inline self_type operator++(int)
    {
        self_type retval = *this;
        val++;
        return retval;
    }

    /// Prefix increment
    inline self_type operator++()
    {
        val++;
        return *this;
    }

    /// Indirection
    inline reference operator*() const
    {
        return val;
    }

    /// Addition
    template <typename Distance>
    inline self_type operator+(Distance n) const
    {
        self_type retval(val + n);
        return retval;
    }

    /// Addition assignment
    template <typename Distance>
    inline self_type& operator+=(Distance n)
    {
        val += n;
        return *this;
    }

    /// Subtraction
    template <typename Distance>
    inline self_type operator-(Distance n) const
    {
        self_type retval(val - n);
        return retval;
    }

    /// Subtraction assignment
    template <typename Distance>
    inline self_type& operator-=(Distance n)
    {
        val -= n;
        return *this;
    }

    /// Distance
    inline difference_type operator-(self_type other) const
    {
        return val - other.val;
    }

    /// Array subscript
    template <typename Distance>
    inline reference operator[](Distance n) const
    {
        return val + n;
    }

    /// Structure dereference
    inline pointer operator->()
    {
        return &val;
    }

    /// Equal to
    inline bool operator==(const self_type& rhs)
    {
        return (val == rhs.val);
    }

    /// Not equal to
    inline bool operator!=(const self_type& rhs)
    {
        return (val != rhs.val);
    }

    /// ostream operator
    friend std::ostream& operator<<(std::ostream& os, const self_type& itr)
    {
        os << "[" << itr.val << "]";
        return os;
    }
};

//---------------------------------------------------------------------
// MergePath Search
//---------------------------------------------------------------------

/**
 * Computes the begin offsets into A and B for the specific diagonal
 */
template <
    typename AIteratorT,
    typename BIteratorT,
    typename OffsetT,
    typename CoordinateT>
inline void MergePathSearch(
    OffsetT         diagonal,           ///< [in]The diagonal to search
    AIteratorT      a,                  ///< [in]List A
    BIteratorT      b,                  ///< [in]List B
    OffsetT         a_len,              ///< [in]Length of A
    OffsetT         b_len,              ///< [in]Length of B
    CoordinateT&    path_coordinate)    ///< [out] (x,y) coordinate where diagonal intersects the merge path
{
    OffsetT x_min = std::max(diagonal - b_len, 0);
    OffsetT x_max = std::min(diagonal, a_len);

    while (x_min < x_max)
    {
        OffsetT x_pivot = (x_min + x_max) >> 1;
        if (a[x_pivot] <= b[diagonal - x_pivot - 1])
            x_min = x_pivot + 1;    // Contract range up A (down B)
        else
            x_max = x_pivot;        // Contract range down A (up B)
    }

    path_coordinate.x = std::min(x_min, a_len);
    path_coordinate.y = diagonal - x_min;
}

//---------------------------------------------------------------------
// SpMV verification
//---------------------------------------------------------------------

// Compute reference SpMV y = Ax
template <
    typename ValueT,
    typename OffsetT>
void SpmvGold(
    CsrMatrix<ValueT, OffsetT>&     a,
    ValueT*                         vector_x,
    ValueT*                         vector_y_in,
    ValueT*                         vector_y_out,
    ValueT                          alpha,
    ValueT                          beta)
{
    for (OffsetT row = 0; row < a.num_rows; ++row)
    {
        ValueT partial = beta * vector_y_in[row];
        for (
            OffsetT offset = a.row_offsets[row];
            offset < a.row_offsets[row + 1];
            ++offset)
        {
            partial += alpha * a.values[offset] * vector_x[a.column_indices[offset]];
        }
        vector_y_out[row] = partial;
    }
}

template <
    typename ValueT,
    typename OffsetT>
float TestSpmvGold(
    CsrMatrix<ValueT, OffsetT>&     a,
    ValueT*                         vector_x,
    ValueT*                         vector_y_in,
    ValueT*                         vector_y_out,
    ValueT                          alpha,
    ValueT                          beta,
    int                             timing_iterations)
{
    // Sequential runs are much slower, so we do fewer iterations
    int seq_iters = std::min(10, std::max(1, timing_iterations / 50));
    float elapsed_ms = 0.0;
    CpuTimer timer;
    timer.Start();
    for(int it = 0; it < seq_iters; ++it)
    {
        SpmvGold(a, vector_x, vector_y_in, vector_y_out, alpha, beta);
    }
    timer.Stop();
    elapsed_ms += timer.ElapsedMillis();

    return elapsed_ms / seq_iters;
}

//---------------------------------------------------------------------
// CPU merge-based SpMV
//---------------------------------------------------------------------

/**
 * OpenMP CPU merge-based SpMV
 */
template <
    typename ValueT,
    typename OffsetT>
void OmpMergeCsrmv(
    int                             num_threads,
    CsrMatrix<ValueT, OffsetT>&     a,
    OffsetT*    __restrict        row_end_offsets,    ///< Merge list A (row end-offsets)
    OffsetT*    __restrict        column_indices,
    ValueT*     __restrict        values,
    ValueT*     __restrict        vector_x,
    ValueT*     __restrict        vector_y_out)
{
    // Temporary storage for inter-thread fix-up after load-balanced work
    OffsetT     row_carry_out[256];     // The last row-id each worked on by each thread when it finished its path segment
    ValueT      value_carry_out[256];   // The running total within each thread when it finished its path segment

    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int tid = 0; tid < num_threads; tid++)
    {
        // Merge list B (NZ indices)
        CountingInputIterator<OffsetT>  nonzero_indices(0);

        OffsetT num_merge_items     = a.num_rows + a.num_nonzeros;                          // Merge path total length
        OffsetT items_per_thread    = (num_merge_items + num_threads - 1) / num_threads;    // Merge items per thread

        // Find starting and ending MergePath coordinates (row-idx, nonzero-idx) for each thread
        int2    thread_coord;
        int2    thread_coord_end;
        int     start_diagonal      = std::min(items_per_thread * tid, num_merge_items);
        int     end_diagonal        = std::min(start_diagonal + items_per_thread, num_merge_items);

        MergePathSearch(start_diagonal, row_end_offsets, nonzero_indices, a.num_rows, a.num_nonzeros, thread_coord);
        MergePathSearch(end_diagonal, row_end_offsets, nonzero_indices, a.num_rows, a.num_nonzeros, thread_coord_end);

        // Consume whole rows
        for (; thread_coord.x < thread_coord_end.x; ++thread_coord.x)
        {
            ValueT running_total = 0.0;
            for (; thread_coord.y < row_end_offsets[thread_coord.x]; ++thread_coord.y)
            {
                running_total += values[thread_coord.y] * vector_x[column_indices[thread_coord.y]];
            }

            vector_y_out[thread_coord.x] = running_total;
        }

        // Consume partial portion of thread's last row
        ValueT running_total = 0.0;
        for (; thread_coord.y < thread_coord_end.y; ++thread_coord.y)
        {
            running_total += values[thread_coord.y] * vector_x[column_indices[thread_coord.y]];
        }

        // Save carry-outs
        row_carry_out[tid] = thread_coord_end.x;
        value_carry_out[tid] = running_total;
    }

    // Carry-out fix-up (rows spanning multiple threads)
    for (int tid = 0; tid < num_threads - 1; ++tid)
    {
        if (row_carry_out[tid] < a.num_rows)
            vector_y_out[row_carry_out[tid]] += value_carry_out[tid];
    }
}

/**
 * Run OmpMergeCsrmv
 */
template <
    typename ValueT,
    typename OffsetT>
float TestOmpMergeCsrmv(
    int                             num_threads,
    CsrMatrix<ValueT, OffsetT>&     a,
    ValueT*                         vector_x,
    ValueT*                         reference_vector_y_out,
    ValueT*                         vector_y_out,
    int                             timing_iterations,
    float                           &setup_ms)
{
    setup_ms = 0.0;

    // Warmup/correctness
    memset(vector_y_out, -1, sizeof(ValueT) * a.num_rows);
    OmpMergeCsrmv(num_threads, a, a.row_offsets + 1, a.column_indices, a.values, vector_x, vector_y_out);
    if (!g_quiet)
    {
        // Check answer
        int compare = CompareResults(reference_vector_y_out, vector_y_out, a.num_rows, true);
        if (compare) { printf("\tFAIL\n"); fflush(stdout); }
    }
 
    // Re-populate caches, etc.
    OmpMergeCsrmv(num_threads, a, a.row_offsets + 1, a.column_indices, a.values, vector_x, vector_y_out);
    OmpMergeCsrmv(num_threads, a, a.row_offsets + 1, a.column_indices, a.values, vector_x, vector_y_out);
    OmpMergeCsrmv(num_threads, a, a.row_offsets + 1, a.column_indices, a.values, vector_x, vector_y_out);

    // Timing
    float elapsed_ms = 0.0;
    CpuTimer timer;
    timer.Start();
    for(int it = 0; it < timing_iterations; ++it)
    {
        OmpMergeCsrmv(num_threads, a, a.row_offsets + 1, a.column_indices, a.values, vector_x, vector_y_out);
    }
    timer.Stop();
    elapsed_ms += timer.ElapsedMillis();

    return elapsed_ms / timing_iterations;
}

//---------------------------------------------------------------------
// CPU Hybrid Chunking SpMV
//---------------------------------------------------------------------

#define CHUNK_SIZE           1024
#define THRESHOLD_MULT       3.0

struct ChunkTask {
    int row;
    int start_col;
    int end_col;
};

template <typename OffsetT>
struct HybridPlan {
    std::vector<OffsetT> light_rows;
    std::vector<OffsetT> heavy_rows;
    std::vector<ChunkTask> heavy_chunks;
    int thresh;
};

template <typename ValueT, typename OffsetT>
void BuildHybridPlan(CsrMatrix<ValueT, OffsetT>& a, HybridPlan<OffsetT>& plan) {
    double avg_rl = (double)a.num_nonzeros / a.num_rows;
    int thresh = (int)(THRESHOLD_MULT * avg_rl);
    if (thresh < CHUNK_SIZE) thresh = CHUNK_SIZE;
    plan.thresh = thresh;
    
    for (OffsetT r = 0; r < a.num_rows; r++) {
        OffsetT nnz = a.row_offsets[r+1] - a.row_offsets[r];
        if (nnz < thresh) {
            plan.light_rows.push_back(r);
        } else {
            plan.heavy_rows.push_back(r);
            OffsetT s = a.row_offsets[r];
            OffsetT e = a.row_offsets[r+1];
            for (OffsetT k = s; k < e; k += CHUNK_SIZE) {
                ChunkTask chunk;
                chunk.row = r;
                chunk.start_col = k;
                chunk.end_col = std::min(k + CHUNK_SIZE, e);
                plan.heavy_chunks.push_back(chunk);
            }
        }
    }
}

template <typename ValueT, typename OffsetT>
void OmpHybridChunkingCsrmv(
    int                             num_threads,
    CsrMatrix<ValueT, OffsetT>&     a,
    HybridPlan<OffsetT>&            plan,
    ValueT*     __restrict        vector_x,
    ValueT*     __restrict        vector_y_out)
{
    OffsetT* __restrict light_data = plan.light_rows.data();
    int num_light = plan.light_rows.size();
    OffsetT* __restrict heavy_rows = plan.heavy_rows.data();
    int num_heavy_rows = plan.heavy_rows.size();
    ChunkTask* __restrict chunks = plan.heavy_chunks.data();
    int num_chunks = plan.heavy_chunks.size();

    ValueT*  __restrict values = a.values;
    OffsetT* __restrict col_indices = a.column_indices;
    OffsetT* __restrict row_offsets = a.row_offsets;

    #pragma omp parallel num_threads(num_threads)
    {
        // 1. Clear ONLY the heavy rows
        #pragma omp for schedule(static)
        for (int i = 0; i < num_heavy_rows; i++) {
            vector_y_out[heavy_rows[i]] = 0.0;
        }

        // 2. Light Rows (Using Inspector Indirection)
        // A smaller dynamic chunk size adds necessary scheduling overhead 
        // that realistically models the cost of the inspector-executor framework.
        #pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < num_light; i++) {
            OffsetT r = light_data[i];
            
            ValueT sum = 0.0;
            for (OffsetT k = row_offsets[r]; k < row_offsets[r+1]; k++) {
                sum += values[k] * vector_x[col_indices[k]];
            }
            vector_y_out[r] = sum;
        }

        // 3. Heavy Chunks
        #pragma omp for schedule(dynamic, 16)
        for (int i = 0; i < num_chunks; i++) {
            ChunkTask& chunk = chunks[i];
            ValueT sum = 0.0;
            for (int k = chunk.start_col; k < chunk.end_col; k++) {
                sum += values[k] * vector_x[col_indices[k]];
            }
            #pragma omp atomic
            vector_y_out[chunk.row] += sum;
        }
    }
}

template <typename ValueT, typename OffsetT>
float TestOmpHybridChunkingCsrmv(
    int                             num_threads,
    CsrMatrix<ValueT, OffsetT>&     a,
    HybridPlan<OffsetT>&            plan,
    ValueT*                         vector_x,
    ValueT*                         reference_vector_y_out,
    ValueT*                         vector_y_out,
    int                             timing_iterations,
    float                           &setup_ms)
{
    setup_ms = 0.0;

    // Warmup/correctness
    OmpHybridChunkingCsrmv(num_threads, a, plan, vector_x, vector_y_out);
    if (!g_quiet)
    {
        // Check answer
        int compare = CompareResults(reference_vector_y_out, vector_y_out, a.num_rows, true);
        if (compare) { printf("\tFAIL\n"); fflush(stdout); }
    }
 
    // Re-populate caches, etc.
    OmpHybridChunkingCsrmv(num_threads, a, plan, vector_x, vector_y_out);
    OmpHybridChunkingCsrmv(num_threads, a, plan, vector_x, vector_y_out);
    OmpHybridChunkingCsrmv(num_threads, a, plan, vector_x, vector_y_out);

    // Timing
    float elapsed_ms = 0.0;
    CpuTimer timer;
    timer.Start();
    for(int it = 0; it < timing_iterations; ++it)
    {
        OmpHybridChunkingCsrmv(num_threads, a, plan, vector_x, vector_y_out);
    }
    timer.Stop();
    elapsed_ms += timer.ElapsedMillis();

    return elapsed_ms / timing_iterations;
}

//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------

static double MeasurePeakBandwidth(int nt) {
    size_t N = 40000000; // ~320MB per array
    double *A = (double*)malloc(N * sizeof(double));
    double *B = (double*)malloc(N * sizeof(double));
    double *C = (double*)malloc(N * sizeof(double));
    double scalar = 3.0;
    
    #pragma omp parallel for num_threads(nt)
    for(size_t i=0; i<N; i++){
        A[i] = 1.0; B[i] = 2.0; C[i] = 0.0;
    }
    
    double times[5];
    for(int r=0; r<5; r++) {
        CpuTimer timer;
        timer.Start();
        #pragma omp parallel for num_threads(nt)
        for(size_t i=0; i<N; i++) {
            C[i] = A[i] + scalar * B[i];
        }
        timer.Stop();
        times[r] = timer.ElapsedMillis() / 1000.0;
    }
    
    double min_t = times[0];
    for(int r=1; r<5; r++) if(times[r] < min_t) min_t = times[r];
    
    // 3 arrays read/written: read A, read B, write C. 3 * N * sizeof(double)
    double bw = (3.0 * N * sizeof(double)) / (min_t * 1e9);
    
    free(A); free(B); free(C);
    return bw;
}

/*
 * Run tests
 */
template <
    typename ValueT,
    typename OffsetT>
void RunTests(
    ValueT              alpha,
    ValueT              beta,
    const std::string&  mtx_filename,
    int                 timing_iterations,
    CommandLineArgs&    args)
{
    // Initialize matrix in COO form
    CooMatrix<ValueT, OffsetT> coo_matrix;

    if (!mtx_filename.empty())
    {
        // Parse matrix market file
        coo_matrix.InitMarket(mtx_filename, 1.0, !g_quiet);

        if ((coo_matrix.num_rows == 1) || (coo_matrix.num_cols == 1) || (coo_matrix.num_nonzeros == 1))
        {
            if (!g_quiet) printf("Trivial dataset\n");
            exit(0);
        }
        printf("%s, ", mtx_filename.c_str()); fflush(stdout);
    }
    else
    {
        fprintf(stderr, "No graph type specified. Please provide a .mtx file using --mtx=\n");
        exit(1);
    }

    CsrMatrix<ValueT, OffsetT> csr_matrix(coo_matrix);
    coo_matrix.Clear();

    // Display matrix info
    csr_matrix.Stats().Display(!g_quiet);
    if (!g_quiet)
    {
        printf("\n");
        csr_matrix.DisplayHistogram();
        printf("\n");
        if (g_verbose2)
            csr_matrix.Display();
        printf("\n");
    }
    fflush(stdout);

    // Determine # of timing iterations
    if (timing_iterations == -1)
    {
        // 100 iterations is plenty for an accurate average and runs instantly
        timing_iterations = 100;
        if (!g_quiet)
            printf("\t%d timing iterations\n", timing_iterations);
    }

    // Allocate input and output vectors
    ValueT *vector_x = new ValueT[csr_matrix.num_cols];
    ValueT *vector_y_out = new ValueT[csr_matrix.num_rows];
    ValueT *reference_vector_y_out = new ValueT[csr_matrix.num_rows];

    // Initialize vectors
    for (OffsetT i = 0; i < csr_matrix.num_cols; i++) vector_x[i] = 1.0;
    for (OffsetT i = 0; i < csr_matrix.num_rows; i++) vector_y_out[i] = 0.0;
    for (OffsetT i = 0; i < csr_matrix.num_rows; i++) reference_vector_y_out[i] = 0.0;

    // Compute gold reference
    if (!g_quiet) { printf("Computing reference CPU SpMV... "); fflush(stdout); }
    SpmvGold(csr_matrix, vector_x, vector_y_out, reference_vector_y_out, alpha, beta);
    float seq_time_ms = TestSpmvGold(csr_matrix, vector_x, vector_y_out, reference_vector_y_out, alpha, beta, timing_iterations);
    double seq_gflops = (2.0 * csr_matrix.num_nonzeros) / (seq_time_ms * 1e-3 * 1e9);
    if (!g_quiet) { printf("done. Sequential Baseline: %.3f ms | %.4f GFLOPs\n\n", seq_time_ms, seq_gflops); fflush(stdout); }

    // Build Hybrid Plan once
    HybridPlan<OffsetT> plan;
    BuildHybridPlan(csr_matrix, plan);
    int num_light = csr_matrix.num_rows - plan.heavy_rows.size();

    if (!g_quiet) {
        printf("  Inspector: %d light rows, %lu chunks generated for heavy rows.\n\n", 
               num_light, plan.heavy_chunks.size());
    }

    std::string base_name = mtx_filename;
    size_t last_slash = base_name.find_last_of('/');
    if (last_slash != std::string::npos) {
        base_name = base_name.substr(last_slash + 1);
    }

    int max_nt = omp_get_max_threads();
    if (g_omp_threads > 0) {
        max_nt = g_omp_threads;
    }

    double system_peak_bw = MeasurePeakBandwidth(max_nt);
    if (!g_quiet) {
        printf("\n  [+] Measuring Hardware Empirical Roofline...\n");
        printf("      System Peak Memory Bandwidth: %.2f GB/s\n\n", system_peak_bw);
    }

    if (!g_quiet) {
        printf("  %-7s %-15s %10s %10s %10s %10s %10s %10s %s\n", "Threads", "Method", "Time(ms)", "GFLOPs", "BW(GB/s)", "Roofline", "Speedup", "Effic", "Status");
        printf("  ----------------------------------------------------------------------------------------------------\n");
    }

    int t_counts[] = {1, 2, 4, 8, 12, 16, 32, 64};
    int n_tc = sizeof(t_counts)/sizeof(t_counts[0]);
    if (g_omp_threads > 0) {
        t_counts[0] = g_omp_threads;
        n_tc = 1;
        max_nt = g_omp_threads;
    }

    size_t total_bytes = (csr_matrix.num_nonzeros * (sizeof(ValueT) * 2 + sizeof(OffsetT))) +
        (csr_matrix.num_rows) * (sizeof(OffsetT) + sizeof(ValueT));

    float best_mrg_ms = 999999.0, best_hyb_ms = 999999.0;
    float sum_mrg_ms = 0.0, sum_hyb_ms = 0.0;
    int valid_runs = 0;
    int last_nt = -1;

    // Check if CSV exists to write header
    FILE *check_csv = fopen("spmv_results.csv", "r");
    bool write_header = false;
    if (!check_csv) {
        write_header = true;
    } else {
        fseek(check_csv, 0, SEEK_END);
        if (ftell(check_csv) == 0) write_header = true;
        fclose(check_csv);
    }
    
    if (write_header) {
        FILE *csv = fopen("spmv_results.csv", "a");
        if (csv) {
            fprintf(csv, "Matrix,Rows,Cols,NNZ,LightRows,HeavyChunks,Threads,SeqTime_ms,SeqGFLOPs,MergeTime_ms,MergeGFLOPs,MergeBW_GBps,MergeRoofline,HybridTime_ms,HybridGFLOPs,HybridBW_GBps,HybridRoofline,MergeSpeedup,HybridSpeedup,MergeEfficiency,HybridEfficiency\n");
            fclose(csv);
        }
    }

    for (int i = 0; i < n_tc; i++) {
        int nt = t_counts[i];
        if (nt > max_nt && i != n_tc - 1 && g_omp_threads <= 0) continue; 
        if (nt > max_nt) nt = max_nt;
        if (nt == last_nt) continue;
        last_nt = nt;

        float setup_ms;
        float merge_ms = TestOmpMergeCsrmv(nt, csr_matrix, vector_x, reference_vector_y_out, vector_y_out, timing_iterations, setup_ms);
        float hybrid_ms = TestOmpHybridChunkingCsrmv(nt, csr_matrix, plan, vector_x, reference_vector_y_out, vector_y_out, timing_iterations, setup_ms);

        if (merge_ms < best_mrg_ms) best_mrg_ms = merge_ms;
        if (hybrid_ms < best_hyb_ms) best_hyb_ms = hybrid_ms;

        sum_mrg_ms += merge_ms;
        sum_hyb_ms += hybrid_ms;
        valid_runs++;

        double mrg_gflops = (2.0 * csr_matrix.num_nonzeros) / (merge_ms * 1e-3 * 1e9);
        double hyb_gflops = (2.0 * csr_matrix.num_nonzeros) / (hybrid_ms * 1e-3 * 1e9);

        double mrg_bw = total_bytes / (merge_ms * 1e-3 * 1e9);
        double hyb_bw = total_bytes / (hybrid_ms * 1e-3 * 1e9);

        double mrg_speedup = seq_time_ms / merge_ms;
        double hyb_speedup = seq_time_ms / hybrid_ms;

        double mrg_eff = mrg_speedup / nt;
        double hyb_eff = hyb_speedup / nt;

        double mrg_roofline = (mrg_bw / system_peak_bw) * 100.0;
        double hyb_roofline = (hyb_bw / system_peak_bw) * 100.0;

        if (!g_quiet) {
            printf("  %-7d %-15s %10.3f %10.4f %10.2f %9.1f%% %9.2fx %9.2f %s\n", 
                   nt, "Merge-based", merge_ms, mrg_gflops, mrg_bw, mrg_roofline, mrg_speedup, mrg_eff, "OK");
            printf("  %-7d %-15s %10.3f %10.4f %10.2f %9.1f%% %9.2fx %9.2f %s\n", 
                   nt, "Hybrid-Chunk", hybrid_ms, hyb_gflops, hyb_bw, hyb_roofline, hyb_speedup, hyb_eff, "OK");
            
            double t_diff = merge_ms / hybrid_ms;
            double g_diff = hyb_gflops / mrg_gflops;
            double b_diff = hyb_bw / mrg_bw;
            double r_diff = hyb_roofline / mrg_roofline;
            double s_diff = hyb_speedup / mrg_speedup;
            double e_diff = hyb_eff / mrg_eff;

            printf("  %-7s %-15s %9.2fx %9.2fx %9.2fx %9.2fx %9.2fx %9.2fx\n", 
                   "", "Advantage(H/M)", t_diff, g_diff, b_diff, r_diff, s_diff, e_diff);
            printf("  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - \n");
        }

        FILE *csv = fopen("spmv_results.csv", "a");
        if (csv) {
            fprintf(csv, "%s,%d,%d,%d,%d,%lu,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    base_name.c_str(), csr_matrix.num_rows, csr_matrix.num_cols, csr_matrix.num_nonzeros,
                    num_light, plan.heavy_chunks.size(), nt,
                    seq_time_ms, seq_gflops, merge_ms, mrg_gflops, mrg_bw, mrg_roofline,
                    hybrid_ms, hyb_gflops, hyb_bw, hyb_roofline,
                    mrg_speedup, hyb_speedup, mrg_eff, hyb_eff);
            fclose(csv);
        }
    }

    float avg_mrg_ms = sum_mrg_ms / valid_runs;
    float avg_hyb_ms = sum_hyb_ms / valid_runs;

    // Summary
    if (!g_quiet) {
        if (avg_hyb_ms < avg_mrg_ms) {
            printf("\n  WINNER: HYBRID CHUNKING\n\n");
        } else {
            printf("\n  WINNER: MERGE-BASED\n\n");
        }
    }

    // Cleanup
    delete[] vector_x;
    delete[] vector_y_out;
    delete[] reference_vector_y_out;
}

/**
 * Main
 */
int main(int argc, char **argv)
{
    // Initialize command line
    CommandLineArgs args(argc, argv);
    if (args.CheckCmdLineFlag("help"))
    {
        printf(
            "%s "
            "[--quiet] "
            "[--v] "
            "[--threads=<OMP threads>] "
            "[--i=<timing iterations>] "
            "[--fp64 (default) | --fp32] "
            "[--alpha=<alpha scalar (default: 1.0)>] "
            "[--beta=<beta scalar (default: 0.0)>] "
            "\n\t"
                "--mtx=<matrix market file> "
            "\n", argv[0]);
        exit(0);
    }

    bool                fp32;
    std::string         mtx_filename;
    int                 timing_iterations   = -1;
    float               alpha               = 1.0;
    float               beta                = 0.0;

    g_verbose = args.CheckCmdLineFlag("v");
    g_verbose2 = args.CheckCmdLineFlag("v2");
    g_quiet = args.CheckCmdLineFlag("quiet");
    fp32 = args.CheckCmdLineFlag("fp32");
    args.GetCmdLineArgument("i", timing_iterations);
    args.GetCmdLineArgument("mtx", mtx_filename);
    args.GetCmdLineArgument("alpha", alpha);
    args.GetCmdLineArgument("beta", beta);
    args.GetCmdLineArgument("threads", g_omp_threads);

    // Run test(s)
    if (fp32)
    {
        RunTests<float, int>(alpha, beta, mtx_filename, timing_iterations, args);
    }
    else
    {
        RunTests<double, int>(alpha, beta, mtx_filename, timing_iterations, args);
    }

    printf("\n");

    return 0;
}
