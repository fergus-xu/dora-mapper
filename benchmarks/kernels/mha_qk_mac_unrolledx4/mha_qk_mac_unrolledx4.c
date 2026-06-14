#include <stdint.h>

/*
 * MHA Q·K MAC kernel — 4-way unrolled dot product.
 *
 *   score = sum_d Q[d] * K[d]   (4 elements per loop iteration)
 *
 * Assumes head_dim is a multiple of 4.
 */

#define UNROLL 4

static int8_t *Q_row = (int8_t *)0xa00;
static int8_t *K_row = (int8_t *)0xb00;
static int32_t *score_out = (int32_t *)0xc00;
volatile int *head_dim = (int *)0xf00;

__attribute__((noinline))
void mha_qk_mac_unrolledx4(void)
{
    int32_t acc = 0;
    int n = *head_dim;

    for (int d = 0; d < n; d += UNROLL) {
        //DFGLoop: loop

        int8_t q0 = Q_row[d];
        int8_t k0 = K_row[d];
        int8_t q1 = Q_row[d + 1];
        int8_t k1 = K_row[d + 1];
        int8_t q2 = Q_row[d + 2];
        int8_t k2 = K_row[d + 2];
        int8_t q3 = Q_row[d + 3];
        int8_t k3 = K_row[d + 3];

        acc = acc + ((int32_t)q0 * (int32_t)k0);
        acc = acc + ((int32_t)q1 * (int32_t)k1);
        acc = acc + ((int32_t)q2 * (int32_t)k2);
        acc = acc + ((int32_t)q3 * (int32_t)k3);
    }

    *score_out = acc;
}
