#include "markers.h"
#include <stdint.h>

void spmv_csr(
    int m,
    const int *row_ptr,
    const int *col_idx,
    const float *val,
    const float *x,
    float *y)
{
    for (int i = 0; i < m; i++) {
        float acc = 0.0f;

        for (int k = row_ptr[i]; k < row_ptr[i+1]; k++) {

            KERNEL_START();

            int j = col_idx[k];
            acc += val[k] * x[j];

            KERNEL_END();
        }

        y[i] = acc;
    }
}