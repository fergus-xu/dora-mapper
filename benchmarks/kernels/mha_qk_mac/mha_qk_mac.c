#include <stdint.h>

/*
 * MHA Q·K MAC kernel — one dot product for a single attention score.
 *
 *   score = sum_d Q[row, d] * K[col, d]
 *
 * CGRA-ME DFG extraction follows the microbench style (see conv2.c / mha_softmax).
 *
 * Mixed-precision mapping target (mp_hycube):
 *   - 8-bit loads of Q and K
 *   - 32-bit widen + multiply + accumulate
 *   - 32-bit score output (fixed-point or pre-scale int32)
 */

/* Fake absolute addresses (0xa00=2560, 0xb00=2816, ...) like microbench kernels. */
static int8_t *Q_row = (int8_t *)0xa00;
static int8_t *K_row = (int8_t *)0xb00;
static int32_t *score_out = (int32_t *)0xc00;
volatile int *head_dim = (int *)0xf00;

/* noinline: loop DFG is extracted from this function. */
__attribute__((noinline))
void mha_qk_mac(void)
{
    int32_t acc = 0;
    int n = *head_dim;

    for (int d = 0; d < n; d++) {
        //DFGLoop: loop

        int8_t q8 = Q_row[d];
        int8_t k8 = K_row[d];
        int32_t prod = ((int32_t)q8) * ((int32_t)k8);
        acc = acc + prod;
    }

    *score_out = acc;
}
