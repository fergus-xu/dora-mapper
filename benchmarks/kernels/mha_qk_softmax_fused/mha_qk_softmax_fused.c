#include <stdint.h>

/*
 * Fused MHA Q·K MAC + online-softmax for one query row.
 *
 * Per key j:
 *   score_j = sum_d Q[d] * K[j, d]     (HEAD_DIM fully unrolled)
 *   P[j]    = softmax(score_j)         (LUT-based, fixed-point)
 *
 * HEAD_DIM is compile-time fixed so the MAC nest is a single basic block.
 */

#define FRAC_BITS 8
#define HEAD_DIM 8
#define HEAD_DIM_LOG2 3

static int8_t *Q_vec = (int8_t *)0xa00;
static int8_t *K_tile = (int8_t *)0xb00;
static int32_t *exp_lut = (int32_t *)0xc00;
static int8_t *P_row = (int8_t *)0xd00;
static int32_t *scores_sum_out = (int32_t *)0xe00;
volatile int *block_N = (int *)0xf00;
volatile int32_t *scores_max_ptr = (int32_t *)0x1100;

__attribute__((noinline))
void mha_qk_softmax_fused(void)
{
    int32_t sum = 0;
    int n = *block_N;
    int32_t scores_max = *scores_max_ptr;

    for (int j = 0; j < n; j++) {
        //DFGLoop: loop

        int32_t kbase = j << HEAD_DIM_LOG2;
        const int8_t *krow = K_tile + kbase;

        int32_t score = 0;
        score = score + ((int32_t)Q_vec[0] * (int32_t)krow[0]);
        score = score + ((int32_t)Q_vec[1] * (int32_t)krow[1]);
        score = score + ((int32_t)Q_vec[2] * (int32_t)krow[2]);
        score = score + ((int32_t)Q_vec[3] * (int32_t)krow[3]);
        score = score + ((int32_t)Q_vec[4] * (int32_t)krow[4]);
        score = score + ((int32_t)Q_vec[5] * (int32_t)krow[5]);
        score = score + ((int32_t)Q_vec[6] * (int32_t)krow[6]);
        score = score + ((int32_t)Q_vec[7] * (int32_t)krow[7]);

        int32_t s = score << FRAC_BITS;
        int32_t diff = s - scores_max;
        int32_t idx = diff >> FRAC_BITS;
        int32_t p = exp_lut[idx];

        sum = sum + p;

        int8_t p8 = (int8_t)(p >> FRAC_BITS);
        P_row[j] = p8;
    }

    *scores_sum_out = sum;
}
