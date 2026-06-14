#include <stdint.h>

/*
 * MHA online-softmax row kernel — 4-way unrolled.
 *
 *   Four softmax elements per loop iteration over block_N.
 *
 * Assumes block_N is a multiple of 4.
 */

#define FRAC_BITS 8
#define UNROLL 4

static int8_t *acc_s_row = (int8_t *)0xa00;
static int32_t *exp_lut = (int32_t *)0xb00;
static int8_t *P_row = (int8_t *)0xc00;
static int32_t *scores_sum_out = (int32_t *)0xe00;
volatile int *block_N = (int *)0xf00;
volatile int32_t *scores_max_ptr = (int32_t *)0xd00;

__attribute__((noinline))
void mha_softmax_x4(void)
{
    int32_t sum = 0;
    int n = *block_N;
    int32_t scores_max = *scores_max_ptr;

    for (int j = 0; j < n; j += UNROLL) {
        //DFGLoop: loop

        int8_t s0 = acc_s_row[j];
        int32_t s0w = ((int32_t)s0) << FRAC_BITS;
        int32_t d0 = s0w - scores_max;
        int32_t i0 = d0 >> FRAC_BITS;
        int32_t p0 = exp_lut[i0];
        sum = sum + p0;
        P_row[j] = (int8_t)(p0 >> FRAC_BITS);

        int8_t s1 = acc_s_row[j + 1];
        int32_t s1w = ((int32_t)s1) << FRAC_BITS;
        int32_t d1 = s1w - scores_max;
        int32_t i1 = d1 >> FRAC_BITS;
        int32_t p1 = exp_lut[i1];
        sum = sum + p1;
        P_row[j + 1] = (int8_t)(p1 >> FRAC_BITS);

        int8_t s2 = acc_s_row[j + 2];
        int32_t s2w = ((int32_t)s2) << FRAC_BITS;
        int32_t d2 = s2w - scores_max;
        int32_t i2 = d2 >> FRAC_BITS;
        int32_t p2 = exp_lut[i2];
        sum = sum + p2;
        P_row[j + 2] = (int8_t)(p2 >> FRAC_BITS);

        int8_t s3 = acc_s_row[j + 3];
        int32_t s3w = ((int32_t)s3) << FRAC_BITS;
        int32_t d3 = s3w - scores_max;
        int32_t i3 = d3 >> FRAC_BITS;
        int32_t p3 = exp_lut[i3];
        sum = sum + p3;
        P_row[j + 3] = (int8_t)(p3 >> FRAC_BITS);
    }

    *scores_sum_out = sum;
}
