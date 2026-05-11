/******************************************************************************
 * Copyright (c) 2011-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 * merge_spmv.h
 *
 * Extracted from cpu_spmv.cpp (Merrill & Garland, SC16).
 * Contains the exact Merge-based SpMV kernel used as the baseline comparison
 * in the Hybrid Chunking benchmark.
 ******************************************************************************/

#pragma once

#include <omp.h>
#include <algorithm>

//---------------------------------------------------------------------
// Utility types (from Merrill & Garland's original code)
//---------------------------------------------------------------------

struct int2
{
    int x;
    int y;
};

/**
 * Counting iterator (from Merrill & Garland's original code)
 */
template <
    typename ValueType,
    typename OffsetT = ptrdiff_t>
struct CountingInputIterator
{
    typedef CountingInputIterator               self_type;
    typedef OffsetT                             difference_type;
    typedef ValueType                           value_type;
    typedef ValueType*                          pointer;
    typedef ValueType                           reference;
    typedef std::random_access_iterator_tag     iterator_category;

    ValueType val;

    inline CountingInputIterator(const ValueType &val) : val(val) {}

    inline self_type operator++(int) { self_type retval = *this; val++; return retval; }
    inline self_type operator++()    { val++; return *this; }
    inline reference operator*() const { return val; }

    template <typename Distance>
    inline self_type operator+(Distance n) const { self_type retval(val + n); return retval; }

    template <typename Distance>
    inline self_type& operator+=(Distance n) { val += n; return *this; }

    template <typename Distance>
    inline self_type operator-(Distance n) const { self_type retval(val - n); return retval; }

    template <typename Distance>
    inline self_type& operator-=(Distance n) { val -= n; return *this; }

    inline difference_type operator-(self_type other) const { return val - other.val; }

    template <typename Distance>
    inline reference operator[](Distance n) const { return val + n; }

    inline pointer operator->() { return &val; }
    inline bool operator==(const self_type& rhs) { return (val == rhs.val); }
    inline bool operator!=(const self_type& rhs) { return (val != rhs.val); }
};


//---------------------------------------------------------------------
// MergePath Search (from Merrill & Garland's original code)
//---------------------------------------------------------------------

/**
 * Computes the begin offsets into A and B for the specific diagonal.
 */
template <
    typename AIteratorT,
    typename BIteratorT,
    typename OffsetT,
    typename CoordinateT>
inline void MergePathSearch(
    OffsetT         diagonal,
    AIteratorT      a,
    BIteratorT      b,
    OffsetT         a_len,
    OffsetT         b_len,
    CoordinateT&    path_coordinate)
{
    OffsetT x_min = std::max(diagonal - b_len, 0);
    OffsetT x_max = std::min(diagonal, a_len);

    while (x_min < x_max)
    {
        OffsetT x_pivot = (x_min + x_max) >> 1;
        if (a[x_pivot] <= b[diagonal - x_pivot - 1])
            x_min = x_pivot + 1;
        else
            x_max = x_pivot;
    }

    path_coordinate.x = std::min(x_min, a_len);
    path_coordinate.y = diagonal - x_min;
}


//---------------------------------------------------------------------
// Sequential Gold SpMV (from Merrill & Garland's original code)
//---------------------------------------------------------------------

// Compute reference SpMV y = alpha*Ax + beta*y
template <typename ValueT, typename OffsetT>
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


//---------------------------------------------------------------------
// CPU Merge-based SpMV (from Merrill & Garland's original code)
//---------------------------------------------------------------------

// Profiling hook: when this is true, the merge kernel records per-thread
// work time into g_thread_times[]. Defined in hybrid_benchmark.cpp.
extern bool   g_profile_load_balance;
extern double g_thread_times[256];

/**
 * OpenMP CPU merge-based SpMV
 */
template <typename ValueT, typename OffsetT>
void OmpMergeCsrmv(
    int                             num_threads,
    CsrMatrix<ValueT, OffsetT>&     a,
    OffsetT*    __restrict          row_end_offsets,    ///< Merge list A (row end-offsets)
    OffsetT*    __restrict          column_indices,
    ValueT*     __restrict          values,
    ValueT*     __restrict          vector_x,
    ValueT*     __restrict          vector_y_out)
{
    // Temporary storage for inter-thread fix-up after load-balanced work
    OffsetT     row_carry_out[256];     // The last row-id each thread worked on
    ValueT      value_carry_out[256];   // The running total when each thread finished

    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int tid = 0; tid < num_threads; tid++)
    {
        double t_start = g_profile_load_balance ? omp_get_wtime() : 0.0;
        // Merge list B (NZ indices)
        CountingInputIterator<OffsetT>  nonzero_indices(0);

        OffsetT num_merge_items     = a.num_rows + a.num_nonzeros;
        OffsetT items_per_thread    = (num_merge_items + num_threads - 1) / num_threads;

        // Find starting and ending MergePath coordinates for each thread
        int2    thread_coord;
        int2    thread_coord_end;
        int     start_diagonal      = std::min(items_per_thread * tid, num_merge_items);
        int     end_diagonal        = std::min(start_diagonal + items_per_thread, num_merge_items);

        MergePathSearch(start_diagonal, row_end_offsets, nonzero_indices,
                        a.num_rows, a.num_nonzeros, thread_coord);
        MergePathSearch(end_diagonal,   row_end_offsets, nonzero_indices,
                        a.num_rows, a.num_nonzeros, thread_coord_end);

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
        row_carry_out[tid]   = thread_coord_end.x;
        value_carry_out[tid] = running_total;

        // Record per-thread work time. The serial fix-up below runs on tid 0
        // only and is excluded - this measures pure parallel work imbalance.
        // ACCUMULATE across calls to overcome Windows 1ms wtime resolution.
        if (g_profile_load_balance)
            g_thread_times[tid] += omp_get_wtime() - t_start;
    }

    // Carry-out fix-up (rows spanning multiple threads)
    for (int tid = 0; tid < num_threads - 1; ++tid)
    {
        if (row_carry_out[tid] < a.num_rows)
            vector_y_out[row_carry_out[tid]] += value_carry_out[tid];
    }
}
