#include <stdint.h>

/*
 * Streaming GEMM inner kernel for 16x16 HyCUBE (UNROLL=16).
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

#define UNROLL 16

/* IO-streamed operand ports — one activation + weight pair per lane. */
volatile int16_t *act_lane0 = (int16_t *)0xd00;
volatile int16_t *act_lane1 = (int16_t *)0xd02;
volatile int16_t *act_lane2 = (int16_t *)0xd04;
volatile int16_t *act_lane3 = (int16_t *)0xd06;
volatile int16_t *act_lane4 = (int16_t *)0xd08;
volatile int16_t *act_lane5 = (int16_t *)0xd0a;
volatile int16_t *act_lane6 = (int16_t *)0xd0c;
volatile int16_t *act_lane7 = (int16_t *)0xd0e;
volatile int16_t *act_lane8 = (int16_t *)0xd10;
volatile int16_t *act_lane9 = (int16_t *)0xd12;
volatile int16_t *act_lane10 = (int16_t *)0xd14;
volatile int16_t *act_lane11 = (int16_t *)0xd16;
volatile int16_t *act_lane12 = (int16_t *)0xd18;
volatile int16_t *act_lane13 = (int16_t *)0xd1a;
volatile int16_t *act_lane14 = (int16_t *)0xd1c;
volatile int16_t *act_lane15 = (int16_t *)0xd1e;

volatile int8_t *wgt_lane0 = (int8_t *)0xe00;
volatile int8_t *wgt_lane1 = (int8_t *)0xe01;
volatile int8_t *wgt_lane2 = (int8_t *)0xe02;
volatile int8_t *wgt_lane3 = (int8_t *)0xe03;
volatile int8_t *wgt_lane4 = (int8_t *)0xe04;
volatile int8_t *wgt_lane5 = (int8_t *)0xe05;
volatile int8_t *wgt_lane6 = (int8_t *)0xe06;
volatile int8_t *wgt_lane7 = (int8_t *)0xe07;
volatile int8_t *wgt_lane8 = (int8_t *)0xe08;
volatile int8_t *wgt_lane9 = (int8_t *)0xe09;
volatile int8_t *wgt_lane10 = (int8_t *)0xe0a;
volatile int8_t *wgt_lane11 = (int8_t *)0xe0b;
volatile int8_t *wgt_lane12 = (int8_t *)0xe0c;
volatile int8_t *wgt_lane13 = (int8_t *)0xe0d;
volatile int8_t *wgt_lane14 = (int8_t *)0xe0e;
volatile int8_t *wgt_lane15 = (int8_t *)0xe0f;

static int32_t *acc_out = (int32_t *)0xc00;
volatile int *k_dim = (int *)0xf00;

__attribute__((noinline))
void gemm_stream_16x16(void)
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
        int16_t a4 = *act_lane4;
        int8_t w4 = *wgt_lane4;
        int16_t a5 = *act_lane5;
        int8_t w5 = *wgt_lane5;
        int16_t a6 = *act_lane6;
        int8_t w6 = *wgt_lane6;
        int16_t a7 = *act_lane7;
        int8_t w7 = *wgt_lane7;
        int16_t a8 = *act_lane8;
        int8_t w8 = *wgt_lane8;
        int16_t a9 = *act_lane9;
        int8_t w9 = *wgt_lane9;
        int16_t a10 = *act_lane10;
        int8_t w10 = *wgt_lane10;
        int16_t a11 = *act_lane11;
        int8_t w11 = *wgt_lane11;
        int16_t a12 = *act_lane12;
        int8_t w12 = *wgt_lane12;
        int16_t a13 = *act_lane13;
        int8_t w13 = *wgt_lane13;
        int16_t a14 = *act_lane14;
        int8_t w14 = *wgt_lane14;
        int16_t a15 = *act_lane15;
        int8_t w15 = *wgt_lane15;

        acc = acc + ((int32_t)a0 * (int32_t)w0);
        acc = acc + ((int32_t)a1 * (int32_t)w1);
        acc = acc + ((int32_t)a2 * (int32_t)w2);
        acc = acc + ((int32_t)a3 * (int32_t)w3);
        acc = acc + ((int32_t)a4 * (int32_t)w4);
        acc = acc + ((int32_t)a5 * (int32_t)w5);
        acc = acc + ((int32_t)a6 * (int32_t)w6);
        acc = acc + ((int32_t)a7 * (int32_t)w7);
        acc = acc + ((int32_t)a8 * (int32_t)w8);
        acc = acc + ((int32_t)a9 * (int32_t)w9);
        acc = acc + ((int32_t)a10 * (int32_t)w10);
        acc = acc + ((int32_t)a11 * (int32_t)w11);
        acc = acc + ((int32_t)a12 * (int32_t)w12);
        acc = acc + ((int32_t)a13 * (int32_t)w13);
        acc = acc + ((int32_t)a14 * (int32_t)w14);
        acc = acc + ((int32_t)a15 * (int32_t)w15);
    }

    *acc_out = acc;
}
