#pragma once

/*
 * Kernel 1 — Escala de grises
 * Entrada : float[B][3][H][W]  layout B*3*H*W  (RGB, [0,1])
 * Salida  : float[B][H][W]     layout B*H*W
 * Grid    : dim3((W+15)/16, (H+15)/16), dim3(16,16)
 * Formula : 0.2989*R + 0.5870*G + 0.1140*B
 */
__global__ void escala_grises(float *entrada, float *salida, int B, int H, int W);
