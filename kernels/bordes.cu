#include "bordes.h"
#include <math.h>

__global__ void deteccion_bordes(float *entrada, float *salida, int B, int H, int W) {
    int col  = blockIdx.x * blockDim.x + threadIdx.x;
    int fila = blockIdx.y * blockDim.y + threadIdx.y;

    if (fila >= H || col >= W) return;

    int i = fila * W + col;

    for (int b = 0; b < B; b++) {
        float *img = entrada + b * H * W;
        float *out = salida  + b * H * W;

        if (fila == 0 || fila == H-1 || col == 0 || col == W-1) {
            out[i] = 0.0f;
            continue;
        }

        float tl = img[(fila-1)*W + (col-1)];
        float tc = img[(fila-1)*W + col];
        float tr = img[(fila-1)*W + (col+1)];
        float ml = img[fila*W + (col-1)];
        float mr = img[fila*W + (col+1)];
        float bl = img[(fila+1)*W + (col-1)];
        float bc = img[(fila+1)*W + col];
        float br = img[(fila+1)*W + (col+1)];

        float gx = -tl + tr - 2.0f*ml + 2.0f*mr - bl + br;
        float gy = -tl - 2.0f*tc - tr + bl + 2.0f*bc + br;

        out[i] = sqrtf(gx*gx + gy*gy);
    }
}