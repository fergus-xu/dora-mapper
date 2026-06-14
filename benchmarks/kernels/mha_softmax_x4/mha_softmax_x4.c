#include <stdint.h>

/*
 * MHA online-softmax row kernel — four column indices per loop iteration.
 *
 * TileLang reference (example_mha_fwd_bshd.py softmax stage):
 *   acc_s[i, j] = exp2(acc_s[i, j] * scale - scores_max[i] * scale)
 *   scores_sum[i] += acc_s[i, j]
 *
 * Each //DFGLOOP iteration processes j, j+1, j+2, j+3 in parallel (four
 * softmax tiers) to increase spatial utilization on mp_hycube_mem.
 *
 * Mixed-precision mapping target:
 *   - 8-bit load of QK scores
 *   - 32-bit ALU for subtract / accumulate / LUT index
 *   - LUT load for exp approximation
 *   - 8-bit store of attention weights P
 *
 * Requires block_N divisible by 4.
 */

#define FRAC_BITS 8
#define LUT_SIZE 256

static int8_t *acc_s_row = (int8_t *)0xa00;
static int32_t *exp_lut = (int32_t *)0xb00;
static int8_t *P_row = (int8_t *)0xc00;
static int32_t *scores_sum_out = (int32_t *)0xd00;
volatile int *block_N = (int *)0xf00;
volatile int32_t *scores_max = (int32_t *)0xf10;

__attribute__((noinline))
void mha_softmax_row_x4(void)
{
    int32_t sum = 0;
    int n = *block_N;
    int32_t smax = *scores_max;

    for (int j = 0; j < n; j += 4) {
        //DFGLOOP: loop

        /* Tier 0: j + 0 */
        int8_t s8_0 = acc_s_row[j + 0];
        int32_t s_0 = ((int32_t)s8_0) << FRAC_BITS;
        int32_t diff_0 = s_0 - smax;
        int32_t idx_0 = diff_0 >> FRAC_BITS;
        if (idx_0 < 0) {
            idx_0 = 0;
        }
        if (idx_0 >= LUT_SIZE) {
            idx_0 = LUT_SIZE - 1;
        }
        int32_t p_0 = exp_lut[idx_0];
        sum = sum + p_0;
        P_row[j + 0] = (int8_t)(p_0 >> FRAC_BITS);

        /* Tier 1: j + 1 */
        int8_t s8_1 = acc_s_row[j + 1];
        int32_t s_1 = ((int32_t)s8_1) << FRAC_BITS;
        int32_t diff_1 = s_1 - smax;
        int32_t idx_1 = diff_1 >> FRAC_BITS;
        if (idx_1 < 0) {
            idx_1 = 0;
        }
        if (idx_1 >= LUT_SIZE) {
            idx_1 = LUT_SIZE - 1;
        }
        int32_t p_1 = exp_lut[idx_1];
        sum = sum + p_1;
        P_row[j + 1] = (int8_t)(p_1 >> FRAC_BITS);

        /* Tier 2: j + 2 */
        int8_t s8_2 = acc_s_row[j + 2];
        int32_t s_2 = ((int32_t)s8_2) << FRAC_BITS;
        int32_t diff_2 = s_2 - smax;
        int32_t idx_2 = diff_2 >> FRAC_BITS;
        if (idx_2 < 0) {
            idx_2 = 0;
        }
        if (idx_2 >= LUT_SIZE) {
            idx_2 = LUT_SIZE - 1;
        }
        int32_t p_2 = exp_lut[idx_2];
        sum = sum + p_2;
        P_row[j + 2] = (int8_t)(p_2 >> FRAC_BITS);

        /* Tier 3: j + 3 */
        int8_t s8_3 = acc_s_row[j + 3];
        int32_t s_3 = ((int32_t)s8_3) << FRAC_BITS;
        int32_t diff_3 = s_3 - smax;
        int32_t idx_3 = diff_3 >> FRAC_BITS;
        if (idx_3 < 0) {
            idx_3 = 0;
        }
        if (idx_3 >= LUT_SIZE) {
            idx_3 = LUT_SIZE - 1;
        }
        int32_t p_3 = exp_lut[idx_3];
        sum = sum + p_3;
        P_row[j + 3] = (int8_t)(p_3 >> FRAC_BITS);
    }

    *scores_sum_out = sum;
}
