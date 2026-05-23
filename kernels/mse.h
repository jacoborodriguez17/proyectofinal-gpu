#pragma once

/*
 * Kernel 4 — MSE y RMSE por imagen vs imagen de referencia.
 * entrada    : float[B][H][W]  batch normalizado
 * referencia : float[H][W]     imagen de referencia (primera del batch)
 * rmse       : float[B]        resultado — un RMSE por imagen
 * Grid       : dim3(B), dim3(256)
 * Shared     : 256 * sizeof(float)
 */
__global__ void calcular_mse(float *entrada, float *referencia,
                             float *rmse, int B, int H, int W);
