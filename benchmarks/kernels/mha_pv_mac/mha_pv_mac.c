#include "markers.h"
#include <stdint.h>

/*
 * MHA P @ V — mixed-precision MAC inner loop (follows mha_softmax).
 *
 * TileLang reference (example_mha_fwd_bshd.py, line 87):
 *   T.gemm(acc_s_cast, V_shared, acc_o, ...)
 *   acc_o[i, d] += P[i, j] * V[j, d]     (fp16 P/V in, fp32 accum)
 *
 * mp_hycube mixed-precision analogue:
 *   - 8-bit loads for attention weights P (from mha_softmax_row P_row[])
 *   - 8-bit loads for value tile V
 *   - widen 8 -> 32 at PE inputs
 *   - 32-bit MAC into output accumulator acc_o
 *
 * Dataflow across the three MHA benchmark kernels:
 *   mha_qk_mac   : Q8, K8  --widen--> MAC32 --narrow--> acc_s8
 *   mha_softmax  : acc_s8  --widen--> ALU32 --narrow--> P8
 *   mha_pv_mac   : P8, V8   --widen--> MAC32 ------------> acc_o32
 *
 * Host fixes (i, d) and strides; this body is one iteration over j.
 */

#define FRAC_BITS 8

static int32_t mha_widen_i8(int8_t x)
{
    return (int32_t)x;
}

void mha_pv_mac(
    int block_N,
    int i,
    int d,
    int p_row_stride,
    int v_row_stride,
    const int8_t *P,
    const int8_t *V,
    int32_t *acc_o_inout)
{
    int32_t acc = *acc_o_inout;

    for (int j = 0; j < block_N; j++) {
        KERNEL_START();

        /* P: 8-bit softmax output; V: 8-bit value cache tile */
        int8_t p8 = P[i * p_row_stride + j];
        int8_t v8 = V[j * v_row_stride + d];

        int32_t p32 = mha_widen_i8(p8);
        int32_t v32 = mha_widen_i8(v8);

        acc = acc + (p32 * v32);

        KERNEL_END();
    }

    *acc_o_inout = acc;
}
