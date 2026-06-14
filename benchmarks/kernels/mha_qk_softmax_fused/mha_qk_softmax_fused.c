#include <stdint.h>

/*
 * Fused MHA Q·K dot + online softmax for one key index per loop iteration.
 *
 * For each key column j in a tile:
 *   1. score[j] = sum_d Q[d] * K[j, d]   (HEAD_TILE-wide dot, unrolled by 4)
 *   2. P[j] = exp_lut[ clamp(score[j] - scores_max) ]
 *   3. scores_sum += P[j]  (unnormalized partial softmax mass)
 *
 * HEAD_TILE is the dot-product width (head dimension chunk). It is fixed at
 * compile time so the QK MAC chain is fully unrolled inside each tagged
 * loop iteration, then chained into the softmax LUT path in the same body.
 *
 * Layout:
 *   Q_row[d]                     for d in [0, HEAD_TILE)
 *   K_tile[j * HEAD_TILE + d]  row-major K block
 *   P_row[j]                     8-bit attention weights out
 *
 * Mixed-precision mapping target (mp_hycube_mem):
 *   - 8-bit Q/K loads, 32-bit MAC tree, LUT load, 8-bit P store
 */

#define FRAC_BITS 8
#define LUT_SIZE 256
#define HEAD_TILE 4

static int8_t *Q_row = (int8_t *)0xa00;
static int8_t *K_tile = (int8_t *)0xb00;
static int32_t *exp_lut = (int32_t *)0xc00;
static int8_t *P_row = (int8_t *)0xd00;
static int32_t *scores_sum_out = (int32_t *)0xe00;
volatile int *block_N = (int *)0xf00;
volatile int32_t *scores_max = (int32_t *)0xf10;

__attribute__((noinline))
void mha_qk_softmax_fused(void)
{
    int32_t sum = 0;
    int n = *block_N;
    int32_t smax = *scores_max;

    for (int j = 0; j < n; j++) {
        //DFGLOOP: loop

        int32_t k_base = j * HEAD_TILE;

        /* Q·K dot over HEAD_TILE (4-wide unroll) */
        int8_t q0 = Q_row[0];
        int8_t q1 = Q_row[1];
        int8_t q2 = Q_row[2];
        int8_t q3 = Q_row[3];

        int8_t k0 = K_tile[k_base + 0];
        int8_t k1 = K_tile[k_base + 1];
        int8_t k2 = K_tile[k_base + 2];
        int8_t k3 = K_tile[k_base + 3];

        int32_t score = 0;
        score = score + ((int32_t)q0) * ((int32_t)k0);
        score = score + ((int32_t)q1) * ((int32_t)k1);
        score = score + ((int32_t)q2) * ((int32_t)k2);
        score = score + ((int32_t)q3) * ((int32_t)k3);

        /* Partial online softmax for this score */
        int32_t s = score << FRAC_BITS;
        int32_t diff = s - smax;
        int32_t idx = diff >> FRAC_BITS;
        if (idx < 0) {
            idx = 0;
        }
        if (idx >= LUT_SIZE) {
            idx = LUT_SIZE - 1;
        }

        int32_t p = exp_lut[idx];
        sum = sum + p;
        P_row[j] = (int8_t)(p >> FRAC_BITS);
    }

    *scores_sum_out = sum;
}
