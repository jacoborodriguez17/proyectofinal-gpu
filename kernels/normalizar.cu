#include "normalizar.h"

/* ------------------------------------------------------------------ */
/* Kernel 3A — Reducción en árbol para hallar el max de cada imagen   */
/* Un bloque por imagen: blockIdx.x = índice de imagen b              */
/* ------------------------------------------------------------------ */
__global__ void reduccion_max(float *entrada, float *max_vals, int H, int W) {
    extern __shared__ float sdata[];

    int b   = blockIdx.x;
    int tid = threadIdx.x;
    int n   = H * W;
    float *img = entrada + b * n;

    /* Cada thread acumula su máximo local recorriendo su porción */
    float local_max = -1e30f;
    for (int i = tid; i < n; i += blockDim.x) {
        if (img[i] > local_max) local_max = img[i];
    }
    sdata[tid] = local_max;
    __syncthreads();

    /* Reducción en árbol dentro del bloque */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid])
                sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) max_vals[b] = sdata[0];
}

/* ------------------------------------------------------------------ */
/* Kernel 3B — Divide cada píxel entre el max de su imagen            */
/* ------------------------------------------------------------------ */
__global__ void dividir_por_max(float *entrada, float *salida,
                                float *max_vals, int B, int H, int W) {
    int col  = blockIdx.x * blockDim.x + threadIdx.x;
    int fila = blockIdx.y * blockDim.y + threadIdx.y;

    if (fila >= H || col >= W) return;

    int i = fila * W + col;

    for (int b = 0; b < B; b++) {
        float mx = max_vals[b];
        salida[b * H * W + i] = (mx > 0.0f) ? entrada[b * H * W + i] / mx : 0.0f;
    }
}
