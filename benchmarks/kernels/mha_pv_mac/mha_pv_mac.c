#include <stdint.h>

/*
 * MHA P·V MAC kernel — one output element of attention-weighted values.
 *
 *   out[d] = sum_j P[j] * V[j, d]
 *
 * This extracts the inner j reduction for a fixed head dimension d (PV GEMM row).
 *
 * CGRA-ME DFG extraction follows the microbench style (see conv2.c / mha_softmax).
 *
 * Mixed-precision mapping target (mp_hycube):
 *   - 8-bit loads of attention weight P and value V
 *   - 32-bit widen + multiply + accumulate
 *   - 32-bit output accumulator
 */

/* Fake absolute addresses (0xa00=2560, 0xb00=2816, ...) like microbench kernels. */
static int8_t *P_row = (int8_t *)0xa00;
static int8_t *V_row = (int8_t *)0xb00;
static int32_t *out_acc = (int32_t *)0xc00;
volatile int *block_N = (int *)0xf00;

/* noinline: loop DFG is extracted from this function. */
__attribute__((noinline))
void mha_pv_mac(void)
{
    int32_t acc = 0;
    int n = *block_N;

    for (int j = 0; j < n; j++) {
        //DFGLoop: loop

        int8_t p8 = P_row[j];
        int8_t v8 = V_row[j];
        int32_t prod = ((int32_t)p8) * ((int32_t)v8);
        acc = acc + prod;
    }

    *out_acc = acc;
}
