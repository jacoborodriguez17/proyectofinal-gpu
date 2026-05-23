#include "mse.h"
#include <math.h>

/* Kernel 4 — MSE y RMSE con reducción en árbol usando __shared__     */
/* Un bloque por imagen: blockIdx.x = índice de imagen b              */
/* referencia apunta al inicio del batch normalizado (imagen 0)       */

__global__ void calcular_mse(float *entrada, float *referencia,
                             float *rmse, int B, int H, int W) {
    extern __shared__ float sdata[];

    int b   = blockIdx.x;
    int tid = threadIdx.x;
    int n   = H * W;
    float *img = entrada + b * n;

    /* Cada thread acumula su suma parcial de diferencias al cuadrado */
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float diff = img[i] - referencia[i];
        local_sum += diff * diff;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    /* Reducción en árbol */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) rmse[b] = sqrtf(sdata[0] / (float)n);
}
