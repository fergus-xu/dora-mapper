#include <stdint.h>

/*
 * Streaming GEMM inner kernel for 4x4 HyCUBE (UNROLL=4).
 *
 *   acc += sum_k  act[k] * wgt[k]
 *
 * Activations (16-bit) and weights (8-bit) stream through IO pads each
 * loop iteration (volatile live-in pointers). No mem-tile operand loads.
 * MAC widens to 32-bit accumulate.
 *
 * Hybrid: bind act_lane* / wgt_lane* / input0 (k_dim) to io_top_*;
 * bind *_output to an IO pad for the finished accumulator.
 */

#define UNROLL 4

/* IO-streamed operand ports — one activation + weight pair per lane. */
volatile int16_t *act_lane0 = (int16_t *)0xd00;
volatile int16_t *act_lane1 = (int16_t *)0xd02;
volatile int16_t *act_lane2 = (int16_t *)0xd04;
volatile int16_t *act_lane3 = (int16_t *)0xd06;

volatile int8_t *wgt_lane0 = (int8_t *)0xe00;
volatile int8_t *wgt_lane1 = (int8_t *)0xe01;
volatile int8_t *wgt_lane2 = (int8_t *)0xe02;
volatile int8_t *wgt_lane3 = (int8_t *)0xe03;

static int32_t *acc_out = (int32_t *)0xc00;
volatile int *k_dim = (int *)0xf00;

__attribute__((noinline))
void gemm_stream_4x4(void)
{
    int32_t acc = 0;
    int n = *k_dim;

    for (int k = 0; k < n; k += UNROLL) {
        //DFGLoop: loop

        int16_t a0 = *act_lane0;
        int8_t w0 = *wgt_lane0;
        int16_t a1 = *act_lane1;
        int8_t w1 = *wgt_lane1;
        int16_t a2 = *act_lane2;
        int8_t w2 = *wgt_lane2;
        int16_t a3 = *act_lane3;
        int8_t w3 = *wgt_lane3;

        acc = acc + ((int32_t)a0 * (int32_t)w0);
        acc = acc + ((int32_t)a1 * (int32_t)w1);
        acc = acc + ((int32_t)a2 * (int32_t)w2);
        acc = acc + ((int32_t)a3 * (int32_t)w3);
    }

    *acc_out = acc;
}
