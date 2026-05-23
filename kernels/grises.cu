#include "grises.h"

__global__ void escala_grises(float *entrada, float *salida, int B, int H, int W) {
    int col  = blockIdx.x * blockDim.x + threadIdx.x;
    int fila = blockIdx.y * blockDim.y + threadIdx.y;

    if (fila >= H || col >= W) return;

    int i = fila * W + col;

    for (int b = 0; b < B; b++) {
        int base = b * 3 * H * W;
        salida[b * H * W + i] = 0.2989f * entrada[base + 0 * H * W + i]
                               + 0.5870f * entrada[base + 1 * H * W + i]
                               + 0.1140f * entrada[base + 2 * H * W + i];
    }
}