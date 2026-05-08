/*
 * ============================================================================
 * SpMV Benchmark: Merge-based (M&G SC16) vs Optimized Hybrid Chunking
 * Supports .mtx (Matrix Market) files.
 *
 * Compile:
 * gcc -O3 -march=native -fopenmp -lm source.c -o spmv_benchmark
 *
 * Run:
 * ./spmv_benchmark your_matrix.mtx
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <omp.h>

/* ---- Configuration ------------------------------------------------------- */
#define WARMUP_RUNS          10
#define BENCH_RUNS           30
#define THRESHOLD_MULT       3.0    /* Target only the truly massive rows */
#define CHUNK_SIZE           1024   /* Maximize atomic hiding */

/* ---- Timing -------------------------------------------------------------- */
static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---- STREAM-like Memory Bandwidth Benchmark ------------------------------ */
static double measure_peak_bw(int nt) {
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
        double t0 = get_time_sec();
        #pragma omp parallel for num_threads(nt)
        for(size_t i=0; i<N; i++) {
            C[i] = A[i] + scalar * B[i];
        }
        times[r] = get_time_sec() - t0;
    }
    
    double min_t = times[0];
    for(int r=1; r<5; r++) if(times[r] < min_t) min_t = times[r];
    
    // 3 arrays read/written: read A, read B, write C. 3 * N * sizeof(double)
    double bw = (3.0 * N * sizeof(double)) / (min_t * 1e9);
    
    free(A); free(B); free(C);
    return bw;
}

/* ---- CSR Matrix ---------------------------------------------------------- */
typedef struct {
    int     num_rows, num_cols, nnz;
    int    *row_offsets;
    int    *col_indices;
    double *values;
    char    name[256];
} CSRMatrix;

static void csr_free(CSRMatrix *A) {
    free(A->row_offsets); free(A->col_indices); free(A->values);
    A->row_offsets = NULL; A->col_indices = NULL; A->values = NULL;
}

/* ---- Matrix Market Parser ------------------------------------------------ */
typedef struct { int row, col; double val; } COOEntry;

static int coo_cmp(const void *a, const void *b) {
    const COOEntry *ea = (const COOEntry*)a, *eb = (const COOEntry*)b;
    if (ea->row != eb->row) return ea->row - eb->row;
    return ea->col - eb->col;
}

static int read_mtx(const char *path, CSRMatrix *A) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path); return 0; }

    char line[4096];
    int is_symmetric = 0, is_pattern = 0, is_integer = 0, header_done = 0;
    int nrows = 0, ncols = 0, nnz_file = 0;

    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '%') {
            if (strncmp(line, "%%MatrixMarket", 14) == 0) {
                char lo[4096]; int i;
                for (i = 0; line[i]; i++) lo[i] = tolower(line[i]); lo[i] = 0;
                if (strstr(lo, "symmetric"))  is_symmetric = 1;
                if (strstr(lo, "pattern"))    is_pattern   = 1;
                if (strstr(lo, "integer"))    is_integer   = 1;
            }
            continue;
        }
        if (!header_done) {
            if (sscanf(line, "%d %d %d", &nrows, &ncols, &nnz_file) < 3) {
                if (sscanf(line, "%d %d", &nrows, &ncols) == 2) nnz_file = nrows*ncols;
                else { fclose(f); return 0; }
            }
            header_done = 1; break;
        }
    }
    if (!header_done || nrows <= 0 || ncols <= 0) { fclose(f); return 0; }

    int cap = (is_symmetric ? nnz_file * 2 : nnz_file) + 8;
    COOEntry *coo = (COOEntry*)malloc(cap * sizeof(COOEntry));
    if (!coo) { fclose(f); return 0; }

    int count = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '%' || line[0] == '\n') continue;
        int r, c; double v = 1.0;
        if (is_pattern) {
            if (sscanf(line, "%d %d", &r, &c) < 2) continue;
        } else if (is_integer) {
            int iv = 1;
            if (sscanf(line, "%d %d %d", &r, &c, &iv) < 2) continue;
            v = (double)iv;
        } else {
            if (sscanf(line, "%d %d %lf", &r, &c, &v) < 2) continue;
        }
        r--; c--;
        if (r < 0 || r >= nrows || c < 0 || c >= ncols) continue;
        if (count + 2 >= cap) {
            cap *= 2;
            COOEntry *tmp = (COOEntry*)realloc(coo, cap * sizeof(COOEntry));
            if (!tmp) break;
            coo = tmp;
        }
        coo[count].row = r; coo[count].col = c; coo[count].val = v; count++;
        if (is_symmetric && r != c) {
            coo[count].row = c; coo[count].col = r; coo[count].val = v; count++;
        }
    }
    fclose(f);

    qsort(coo, count, sizeof(COOEntry), coo_cmp);
    int uniq = 0;
    for (int i = 0; i < count; i++) {
        if (uniq > 0 && coo[uniq-1].row == coo[i].row && coo[uniq-1].col == coo[i].col)
            coo[uniq-1].val += coo[i].val;
        else coo[uniq++] = coo[i];
    }

    A->num_rows = nrows; A->num_cols = ncols; A->nnz = uniq;
    A->row_offsets = (int*)calloc(nrows+1, sizeof(int));
    A->col_indices = (int*)malloc(uniq * sizeof(int));
    A->values      = (double*)malloc(uniq * sizeof(double));

    for (int i = 0; i < uniq; i++) A->row_offsets[coo[i].row+1]++;
    for (int i = 0; i < nrows; i++) A->row_offsets[i+1] += A->row_offsets[i];

    int *cur = (int*)calloc(nrows, sizeof(int));
    for (int i = 0; i < uniq; i++) {
        int row = coo[i].row, pos = A->row_offsets[row] + cur[row]++;
        A->col_indices[pos] = coo[i].col; A->values[pos] = coo[i].val;
    }
    free(cur); free(coo);

    const char *base = strrchr(path, '/'); base = base ? base+1 : path;
    strncpy(A->name, base, 255); A->name[255] = 0;
    char *dot = strstr(A->name, ".mtx"); if (!dot) dot = strstr(A->name, ".MTX");
    if (dot) *dot = 0;
    return 1;
}

/* ---- Vectors ------------------------------------------------------------- */
static double *alloc_x(int n){
    double *v=(double*)malloc(n*sizeof(double)); srand(999);
    for(int i=0;i<n;i++)v[i]=(double)(rand()%100+1)/50.0; return v;
}
static double *zero_vec(int n){return (double*)calloc(n,sizeof(double));}
static double vec_relerr(const double *a,const double *b,int n){
    double d=0,na=0;
    for(int i=0;i<n;i++){double dd=a[i]-b[i];d+=dd*dd;na+=a[i]*a[i];}
    return (na>0)?sqrt(d)/sqrt(na):sqrt(d);
}

/* ---- Sequential Baseline (For correctness check only) -------------------- */
static void spmv_seq(const CSRMatrix *A, const double *x, double *y) {
    for(int row=0; row<A->num_rows; row++) {
        double s=0.0;
        for(int k=A->row_offsets[row]; k<A->row_offsets[row+1]; k++)
            s += A->values[k] * x[A->col_indices[k]];
        y[row]=s;
    }
}

/* ---- SpMV Methods -------------------------------------------------------- */

/* 1. Merge-based (Merrill & Garland SC16) */
typedef struct{int x,y;} Coord;

static Coord merge_path_search(int diag,const int *row_end,int nrows,int nnz){
    int xlo=(diag-nnz>0)?diag-nnz:0, xhi=(diag<nrows)?diag:nrows;
    while(xlo<xhi){
        int mid=(xlo+xhi)>>1, nzi=diag-mid-1;
        if(nzi>=0 && row_end[mid]<=nzi) xlo=mid+1; else xhi=mid;
    }
    Coord c; c.x=(xlo<nrows)?xlo:nrows; c.y=diag-xlo; return c;
}

static void spmv_merge(const CSRMatrix *A,const double *x,double *y,int nt, void *aux){
    const int *re=A->row_offsets+1;
    int nrows=A->num_rows, nnz=A->nnz, N=nrows+nnz;
    int    *rco=(int*)   malloc(nt*sizeof(int));
    double *vco=(double*)malloc(nt*sizeof(double));
    #pragma omp parallel num_threads(nt)
    {
        int tid=omp_get_thread_num();
        int items=(N+nt-1)/nt;
        int d0=tid*items; if(d0>N)d0=N;
        int d1=d0+items;  if(d1>N)d1=N;
        Coord c0=merge_path_search(d0,re,nrows,nnz);
        Coord c1=merge_path_search(d1,re,nrows,nnz);
        double run=0.0;
        for(;c0.x<c1.x;c0.x++){
            for(;c0.y<re[c0.x];c0.y++) run+=A->values[c0.y]*x[A->col_indices[c0.y]];
            y[c0.x]=run; run=0.0;
        }
        for(;c0.y<c1.y;c0.y++) run+=A->values[c0.y]*x[A->col_indices[c0.y]];
        rco[tid]=c1.x; vco[tid]=run;
    }
    for(int t=0;t<nt-1;t++) if(rco[t]<nrows) y[rco[t]]+=vco[t];
    free(rco); free(vco);
}

/* 2. Hybrid Chunking (FLATTENED QUEUE ARCHITECTURE) */
typedef struct { int row; int start; int end; } ChunkTask;
typedef struct {
    int *light_rows;
    int num_light;
    ChunkTask *heavy_chunks;
    int num_chunks;
} HybridPlan;

static HybridPlan build_hybrid_plan(const CSRMatrix *A) {
    HybridPlan plan;
    int nrows = A->num_rows;
    int avg_rl = (nrows > 0) ? A->nnz / nrows : 1;
    int thresh = (int)(THRESHOLD_MULT * avg_rl);
    if(thresh < 1) thresh = 1;
    // Set an absolute floor so we don't treat slightly-above-average rows as "heavy"
    if(thresh < 1024) thresh = 1024;

    plan.light_rows = (int*)malloc(nrows * sizeof(int));
    plan.num_light = 0;
    
    // Generous capacity, will realloc if needed
    int chunk_cap = 10000; 
    plan.heavy_chunks = (ChunkTask*)malloc(chunk_cap * sizeof(ChunkTask));
    plan.num_chunks = 0;

    for (int r = 0; r < nrows; r++) {
        int s = A->row_offsets[r];
        int e = A->row_offsets[r+1];
        int nnz = e - s;

        if (nnz < thresh) {
            plan.light_rows[plan.num_light++] = r;
        } else {
            // Slice the heavy row into chunks
            for (int k = s; k < e; k += CHUNK_SIZE) {
                if (plan.num_chunks == chunk_cap) {
                    chunk_cap *= 2;
                    plan.heavy_chunks = (ChunkTask*)realloc(plan.heavy_chunks, chunk_cap * sizeof(ChunkTask));
                }
                plan.heavy_chunks[plan.num_chunks].row = r;
                plan.heavy_chunks[plan.num_chunks].start = k;
                plan.heavy_chunks[plan.num_chunks].end = (k + CHUNK_SIZE < e) ? k + CHUNK_SIZE : e;
                plan.num_chunks++;
            }
        }
    }
    return plan;
}

static void free_hybrid_plan(HybridPlan *plan) {
    free(plan->light_rows);
    free(plan->heavy_chunks);
}

static void spmv_hybrid(const CSRMatrix *A, const double *x, double *y, int nt, void *aux) {
    HybridPlan *plan = (HybridPlan*)aux;

    // We use a single parallel region to minimize fork-join overhead
    #pragma omp parallel num_threads(nt)
    {
        // Phase 1: Pure math, no atomics, fast path for light rows
        #pragma omp for schedule(dynamic, 1024) nowait
        for (int i = 0; i < plan->num_light; i++) {
            int r = plan->light_rows[i];
            double sum = 0.0;
            for (int k = A->row_offsets[r]; k < A->row_offsets[r+1]; k++) {
                sum += A->values[k] * x[A->col_indices[k]];
            }
            y[r] = sum;
        }

        // Phase 2: Collaborative Chunking for heavy rows
        #pragma omp for schedule(dynamic, 16)
        for (int i = 0; i < plan->num_chunks; i++) {
            int r = plan->heavy_chunks[i].row;
            double acc = 0.0;
            for (int k = plan->heavy_chunks[i].start; k < plan->heavy_chunks[i].end; k++) {
                acc += A->values[k] * x[A->col_indices[k]];
            }
            
            // Only hits atomic once per 4000+ nonzeros. Near zero contention.
            #pragma omp atomic
            y[r] += acc; 
        }
    }
}

/* ---- Benchmark Runner ---------------------------------------------------- */
typedef struct {
    double time_ms, gflops, bw_gbps; int correct;
} Result;

static double median_d(double *a,int n){
    for(int i=1;i<n;i++){double v=a[i];int j=i-1;while(j>=0&&a[j]>v){a[j+1]=a[j];j--;}a[j+1]=v;}
    return a[n/2];
}

static Result run_bench(const CSRMatrix *A, const double *x, double *y, const double *yref,
                        void(*fn)(const CSRMatrix*,const double*,double*,int,void*),
                        int nt, void *aux) {
    double times[BENCH_RUNS];
    for(int r=0; r<WARMUP_RUNS; r++){ memset(y,0,A->num_rows*sizeof(double)); fn(A,x,y,nt,aux); }
    for(int r=0; r<BENCH_RUNS; r++){
        memset(y,0,A->num_rows*sizeof(double));
        double t0=get_time_sec(); 
        fn(A,x,y,nt,aux); 
        times[r]=(get_time_sec()-t0)*1000.0;
    }
    Result res;
    res.time_ms = median_d(times, BENCH_RUNS);
    res.gflops  = (2.0*A->nnz) / (res.time_ms*1e-3*1e9);
    
    // Memory traffic: row_offsets (int), col_indices (int), values (double), x (double), y (double)
    // Using compulsory miss calculation (ideal caches):
    double bytes = sizeof(int)*(A->num_rows + 1) + sizeof(int)*A->nnz + sizeof(double)*A->nnz + sizeof(double)*A->num_cols + sizeof(double)*A->num_rows;
    res.bw_gbps = bytes / (res.time_ms*1e-3*1e9);
    
    res.correct=1;
    if(yref){
        double err = vec_relerr(yref, y, A->num_rows);
        res.correct = (err < 1e-8) ? 1 : 0;
        if(!res.correct) fprintf(stderr,"  [!] Error rel=%.3e\n", err);
    }
    return res;
}

/* ---- Per-matrix analysis ------------------------------------------------- */
static void analyse_matrix(const CSRMatrix *A, FILE *csv, int max_nt) {
    printf("\n+------------------------------------------------------------------+\n");
    printf("| Matrix : %-55s |\n", A->name);
    printf("+------------------------------------------------------------------+\n");
    
    HybridPlan plan = build_hybrid_plan(A);
    printf("  Rows=%d  Cols=%d  NNZ=%d\n", A->num_rows, A->num_cols, A->nnz);
    printf("  Inspector: %d light rows, %d chunks generated for heavy rows.\n\n", plan.num_light, plan.num_chunks);

    double *x    = alloc_x(A->num_cols);
    double *yref = zero_vec(A->num_rows);
    double *ymrg = zero_vec(A->num_rows);
    double *yhyb = zero_vec(A->num_rows);

    /* Generate reference output sequentially */
    double seq_times[BENCH_RUNS];
    for(int r=0; r<WARMUP_RUNS; r++){ memset(yref,0,A->num_rows*sizeof(double)); spmv_seq(A, x, yref); }
    for(int r=0; r<BENCH_RUNS; r++){ 
        memset(yref,0,A->num_rows*sizeof(double)); 
        double t0 = get_time_sec();
        spmv_seq(A, x, yref);
        seq_times[r] = (get_time_sec() - t0) * 1000.0;
    }
    double seq_time_ms = median_d(seq_times, BENCH_RUNS);
    double seq_gflops = (2.0*A->nnz) / (seq_time_ms*1e-3*1e9);

    printf("  Sequential Baseline: %10.3f ms | %10.4f GFLOPs\n\n", seq_time_ms, seq_gflops);

    printf("  %-7s %-15s %10s %10s %10s %10s %10s %s\n", "Threads", "Method", "Time(ms)", "GFLOPs", "BW(GB/s)", "Speedup", "Effic", "Status");
    printf("  --------------------------------------------------------------------------------------\n");

    int t_counts[] = {1, 2, 4, 8, 12, 16, 32, max_nt};
    int n_tc = sizeof(t_counts)/sizeof(t_counts[0]);
    int last_t = 0;

    double best_mrg_ms = 9999999, best_hyb_ms = 9999999;
    
    for(int i=0; i<n_tc; i++) {
        int nt = t_counts[i];
        if (nt > max_nt && i != n_tc - 1) continue; 
        if (nt == last_t) continue;
        last_t = nt;

        Result rm = run_bench(A, x, ymrg, yref, spmv_merge, nt, NULL);
        Result rh = run_bench(A, x, yhyb, yref, spmv_hybrid, nt, &plan);
        
        if (rm.time_ms < best_mrg_ms) best_mrg_ms = rm.time_ms;
        if (rh.time_ms < best_hyb_ms) best_hyb_ms = rh.time_ms;

        double mrg_speedup = seq_time_ms / rm.time_ms;
        double hyb_speedup = seq_time_ms / rh.time_ms;
        
        double mrg_eff = mrg_speedup / nt;
        double hyb_eff = hyb_speedup / nt;

        printf("  %-7d %-15s %10.3f %10.4f %10.2f %9.2fx %9.2f %s\n", 
               nt, "Merge-based", rm.time_ms, rm.gflops, rm.bw_gbps, mrg_speedup, mrg_eff, rm.correct?"OK":"FAIL");
        printf("  %-7d %-15s %10.3f %10.4f %10.2f %9.2fx %9.2f %s\n", 
               nt, "Hybrid-Chunk", rh.time_ms, rh.gflops, rh.bw_gbps, hyb_speedup, hyb_eff, rh.correct?"OK":"FAIL");
        
        // Calculate the advantage of Hybrid over Merge for all metrics
        double t_diff = rm.time_ms / rh.time_ms;   // >1 means Hybrid time is smaller (better)
        double g_diff = rh.gflops / rm.gflops;     // >1 means Hybrid GFLOPs is higher (better)
        double b_diff = rh.bw_gbps / rm.bw_gbps;   // >1 means Hybrid BW is higher (better)
        double s_diff = hyb_speedup / mrg_speedup; // >1 means Hybrid speedup is higher
        double e_diff = hyb_eff / mrg_eff;         // >1 means Hybrid efficiency is higher

        printf("  %-7s %-15s %9.2fx %9.2fx %9.2fx %9.2fx %9.2fx\n", 
               "", "Advantage(H/M)", t_diff, g_diff, b_diff, s_diff, e_diff);

        printf("  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - \n");

        fprintf(csv,"%s,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                A->name, A->num_rows, A->num_cols, A->nnz, plan.num_light, plan.num_chunks, nt,
                seq_time_ms, seq_gflops, rm.time_ms, rm.gflops, rm.bw_gbps, 
                rh.time_ms, rh.gflops, rh.bw_gbps,
                mrg_speedup, hyb_speedup, mrg_eff, hyb_eff);
    }
    
    double speedup_vs_mrg = best_mrg_ms / best_hyb_ms;
    if(best_hyb_ms < best_mrg_ms)
        printf("\n  WINNER: Hybrid Chunking (+%.2fx vs Merge Peak)\n", speedup_vs_mrg);
    else
        printf("\n  WINNER: Merge-based (+%.2fx vs Hybrid Peak)\n", 1.0/speedup_vs_mrg);

    free_hybrid_plan(&plan);
    free(x); free(yref); free(ymrg); free(yhyb);
}

/* ---- Main ---------------------------------------------------------------- */
int main(int argc, char **argv) {
    if(argc < 2) {
        printf("Usage: %s <matrix1.mtx> [matrix2.mtx ...]\n", argv[0]);
        return 1;
    }

    printf("\n=======================================================================\n");
    printf("  SpMV Benchmark: Merge-based vs Optimized Hybrid Chunking\n");
    printf("=======================================================================\n\n");

    int nt = omp_get_max_threads();
    printf("  OpenMP Max threads: %d\n", nt);
    printf("  Chunk size: %d nz (Optimized for minimal atomic contention)\n", CHUNK_SIZE);

    printf("\n  [+] Measuring Peak Memory Bandwidth (STREAM-like)...\n");
    double peak_bw = measure_peak_bw(nt);
    printf("      System Peak Mem Bandwidth: %.2f GB/s\n\n", peak_bw);

    FILE *csv = fopen("spmv_results.csv","w");
    if(!csv){ perror("fopen"); return 1; }
    fprintf(csv,"Matrix,Rows,Cols,NNZ,LightRows,HeavyChunks,Threads,SeqTime_ms,SeqGFLOPs,Merge_ms,Merge_GF,Merge_BW_GBps,Hybrid_ms,Hybrid_GF,Hybrid_BW_GBps,Merge_Speedup,Hybrid_Speedup,Merge_Eff,Hybrid_Eff\n");

    for(int i = 1; i < argc; i++) {
        if(!strstr(argv[i], ".mtx") && !strstr(argv[i], ".MTX")) continue;
        printf("  Loading: %s ...\n", argv[i]);
        CSRMatrix A; memset(&A, 0, sizeof(A));
        if(read_mtx(argv[i], &A)) {
            if(A.nnz == 0 || A.num_rows == 0) {
                fprintf(stderr, "  Skipping empty.\n");
                csr_free(&A);
            } else {
                analyse_matrix(&A, csv, nt);
                csr_free(&A);
            }
        }
    }

    fclose(csv);
    printf("\n  Results written to: spmv_results.csv\n\n");
    return 0;
}
