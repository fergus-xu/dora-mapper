#include "markers.h"
#include <stdint.h>

/*
 * MHA online-softmax row kernel (FlashAttention analogue).
 *
 * TileLang reference (example_mha_fwd_bshd.py, lines 76-78):
 *   acc_s[i, j] = exp2(acc_s[i, j] * scale - scores_max[i] * scale)
 *   scores_sum[i] += acc_s[i, j]   (via reduce_sum over j)
 *
 * Pipeline position (mixed-precision MHA benchmarks):
 *   mha_qk_mac  ->  mha_softmax  ->  mha_pv_mac
 *        Q8,K8           acc_s8            P8,V8
 *          |                |                 |
 *        MAC32           ALU32             MAC32
 *
 * mp_hycube mapping target:
 *   - 8-bit load of QK scores (acc_s_row from mha_qk_mac narrow store)
 *   - 32-bit ALU for subtract / accumulate / LUT index
 *   - LUT load for exp approximation (no fp32 exp2 on HyCUBE)
 *   - 8-bit store of attention weights P (input to mha_pv_mac)
 *
 * Fixed-point: values use FRAC_BITS fractional bits in int32 paths.
 * scores_max is provided in the same fixed-point format as widened scores.
 */

#define FRAC_BITS 8
#define LUT_SIZE 256

void mha_softmax_row(
    int block_N,
    const int8_t *acc_s_row,
    const int32_t *exp_lut,
    int32_t scores_max,
    int8_t *P_row,
    int32_t *scores_sum_out)
{
    int32_t sum = 0;

    for (int j = 0; j < block_N; j++) {
        KERNEL_START();

        /* Mixed-precision: narrow score in, widen for 32-bit softmax math */
        int8_t s8 = acc_s_row[j];
        int32_t s = ((int32_t)s8) << FRAC_BITS;

        int32_t diff = s - scores_max;

        /* LUT index for exp(diff); clamp to table range */
        int32_t idx = diff >> FRAC_BITS;
        if (idx < 0) {
            idx = 0;
        }
        if (idx >= LUT_SIZE) {
            idx = LUT_SIZE - 1;
        }

        int32_t p = exp_lut[idx];
        sum = sum + p;

        /* Narrow weight for PV GEMM input (acc_s_cast analogue) */
        int8_t p8 = (int8_t)(p >> FRAC_BITS);
        P_row[j] = p8;

        KERNEL_END();
    }

    *scores_sum_out = sum;
}
