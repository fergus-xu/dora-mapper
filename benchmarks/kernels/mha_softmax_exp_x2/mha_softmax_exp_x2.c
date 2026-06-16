#include <stdint.h>

/*
 * MHA online-softmax row kernel — 2-way unrolled, exponential approximation.
 *
 *   Two softmax elements per loop iteration (block_N multiple of 2).
 *
 * Fixed-point Q8 Taylor: exp(x) ≈ 1 + x + x^2/2,  x = s - scores_max.
 */

#define FRAC_BITS 8
#define ONE_Q8 (1 << FRAC_BITS)
#define UNROLL 2

static int8_t *acc_s_row = (int8_t *)0xa00;
static int8_t *P_row = (int8_t *)0xc00;
static int32_t *scores_sum_out = (int32_t *)0xe00;
volatile int *block_N = (int *)0xf00;
volatile int32_t *scores_max_ptr = (int32_t *)0xd00;

__attribute__((noinline))
void mha_softmax_exp_x2(void)
{
    int32_t sum = 0;
    int n = *block_N;
    int32_t scores_max = *scores_max_ptr;

    for (int j = 0; j < n; j += UNROLL) {
        //DFGLoop: loop

        int8_t s0 = acc_s_row[j];
        int32_t d0 = (((int32_t)s0) << FRAC_BITS) - scores_max;
        int32_t x20 = (d0 * d0) >> FRAC_BITS;
        int32_t p0 = ONE_Q8 + d0 + (x20 >> 1);
        sum = sum + p0;
        P_row[j] = (int8_t)(p0 >> FRAC_BITS);

        int8_t s1 = acc_s_row[j + 1];
        int32_t d1 = (((int32_t)s1) << FRAC_BITS) - scores_max;
        int32_t x21 = (d1 * d1) >> FRAC_BITS;
        int32_t p1 = ONE_Q8 + d1 + (x21 >> 1);
        sum = sum + p1;
        P_row[j + 1] = (int8_t)(p1 >> FRAC_BITS);
    }

    *scores_sum_out = sum;
}
