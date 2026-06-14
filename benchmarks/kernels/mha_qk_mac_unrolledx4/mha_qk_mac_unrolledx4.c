#include <stdint.h>

/*
 * MHA Q·K MAC kernel — head dimension unrolled by 4.
 *
 *   score = sum_d Q[row, d] * K[col, d]
 *
 * Each loop iteration performs four int8 MACs (d, d+1, d+2, d+3) before
 * advancing the accumulator, giving a wider spatial DFG than mha_qk_mac.
 *
 * Mixed-precision mapping target (mp_hycube_mem):
 *   - 8-bit loads of Q and K
 *   - 32-bit widen + multiply + accumulate
 *   - 32-bit score output
 *
 * Requires head_dim divisible by 4 (enforced at runtime).
 */

static int8_t *Q_row = (int8_t *)0xa00;
static int8_t *K_row = (int8_t *)0xb00;
static int32_t *score_out = (int32_t *)0xc00;
volatile int *head_dim = (int *)0xf00;

__attribute__((noinline))
void mha_qk_mac_unrolledx4(void)
{
    int32_t acc = 0;
    int n = *head_dim;

    for (int d = 0; d < n; d += 4) {
        //DFGLOOP: loop

        int8_t q0 = Q_row[d + 0];
        int8_t q1 = Q_row[d + 1];
        int8_t q2 = Q_row[d + 2];
        int8_t q3 = Q_row[d + 3];

        int8_t k0 = K_row[d + 0];
        int8_t k1 = K_row[d + 1];
        int8_t k2 = K_row[d + 2];
        int8_t k3 = K_row[d + 3];

        int32_t p0 = ((int32_t)q0) * ((int32_t)k0);
        int32_t p1 = ((int32_t)q1) * ((int32_t)k1);
        int32_t p2 = ((int32_t)q2) * ((int32_t)k2);
        int32_t p3 = ((int32_t)q3) * ((int32_t)k3);

        acc = acc + p0;
        acc = acc + p1;
        acc = acc + p2;
        acc = acc + p3;
    }

    *score_out = acc;
}
