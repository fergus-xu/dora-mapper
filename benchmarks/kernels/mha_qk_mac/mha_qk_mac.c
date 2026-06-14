#include "markers.h"
#include <stdint.h>

/*
 * MHA Q @ K^T — mixed-precision MAC inner loop (precedes mha_softmax).
 *
 * TileLang reference (example_mha_fwd_bshd.py, line 67):
 *   T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, ...)
 *   acc_s[i, j] += Q[i, d] * K[j, d]     (fp16 in, fp32 accum)
 *
 * mp_hycube mixed-precision analogue:
 *   - 8-bit MemPort / data-NoC loads for Q and K activations
 *   - sign-extend (8 -> 32) at the PE width adapter
 *   - 32-bit HyCUBE ALU for multiply-accumulate
 *   - 32-bit accumulator carried across the d-loop (phi in DFG)
 *
 * Host fixes output coordinates (i, j) and strides; this kernel body is one
 * iteration of the head-dimension reduction (d = 0 .. dim-1).
 *
 * After dim iterations, narrow the 32-bit accumulator for softmax:
 *   acc_s_i8[i, j] = saturate_i8(acc >> FRAC_BITS);
 * then pass acc_s_i8[] to mha_softmax_row().
 */

#define FRAC_BITS 8

static int32_t mha_widen_i8(int8_t x)
{
    return (int32_t)x;
}

static int8_t mha_narrow_i32_to_i8(int32_t x)
{
    if (x > 127) {
        return 127;
    }
    if (x < -128) {
        return -128;
    }
    return (int8_t)x;
}

void mha_qk_mac(
    int dim,
    int i,
    int j,
    int q_row_stride,
    int k_row_stride,
    const int8_t *Q,
    const int8_t *K,
    int32_t *acc_inout,
    int8_t *acc_s_i8_out)
{
    int32_t acc = *acc_inout;

    for (int d = 0; d < dim; d++) {
        KERNEL_START();

        /* Mixed-precision operands: narrow memory, wide compute */
        int8_t q8 = Q[i * q_row_stride + d];
        int8_t k8 = K[j * k_row_stride + d];

        int32_t q32 = mha_widen_i8(q8);
        int32_t k32 = mha_widen_i8(k8);

        acc = acc + (q32 * k32);

        KERNEL_END();
    }

    *acc_inout = acc;

    /* Narrow fp32-equivalent score to 8-bit tile for softmax (acc_s_cast path) */
    *acc_s_i8_out = mha_narrow_i32_to_i8(acc >> FRAC_BITS);
}
