#include <stdint.h>

/*
 * MHA online-softmax row kernel — exponential approximation (no LUT).
 *
 *   P[j] = exp(s[j] - scores_max) / sum_j ...
 *
 * Fixed-point Q8 Taylor approximation for exp(x), x = diff:
 *   exp(x) ≈ 1 + x + x^2/2
 *
 * Valid for online softmax where scores_max is the row max (diff <= 0).
 * All ops are branch-free for CGRA-ME DFG extraction.
 *
 * Mixed-precision mapping target (mp_hycube):
 *   - 8-bit load of QK scores
 *   - 32-bit ALU for exp polynomial / accumulate
 *   - 8-bit store of attention weights P
 */

#define FRAC_BITS 8
#define ONE_Q8 (1 << FRAC_BITS)

/* Fake absolute addresses (0xa00=2560, ...) like microbench kernels. */
static int8_t *acc_s_row = (int8_t *)0xa00;
static int8_t *P_row = (int8_t *)0xc00;
static int32_t *scores_sum_out = (int32_t *)0xe00;
volatile int *block_N = (int *)0xf00;
volatile int32_t *scores_max_ptr = (int32_t *)0xd00;

/* noinline: loop DFG is extracted from this function. */
__attribute__((noinline))
void mha_softmax_exp_row(void)
{
    int32_t sum = 0;
    int n = *block_N;
    int32_t scores_max = *scores_max_ptr;

    for (int j = 0; j < n; j++) {
        //DFGLoop: loop

        int8_t s8 = acc_s_row[j];
        int32_t s = ((int32_t)s8) << FRAC_BITS;
        int32_t diff = s - scores_max;

        int32_t x2 = (diff * diff) >> FRAC_BITS;
        int32_t p = ONE_Q8 + diff + (x2 >> 1);

        sum = sum + p;

        int8_t p8 = (int8_t)(p >> FRAC_BITS);
        P_row[j] = p8;
    }

    *scores_sum_out = sum;
}
