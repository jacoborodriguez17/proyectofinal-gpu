#pragma once

/*
 * Kernel 3A — Reducción: encuentra el valor máximo de cada imagen.
 * entrada  : float[B][H][W]  layout B*H*W
 * max_vals : float[B]        un max por imagen
 * Grid     : dim3(B), dim3(256)
 * Shared   : 256 * sizeof(float)
 */
__global__ void reduccion_max(float *entrada, float *max_vals, int H, int W);

/*
 * Kernel 3B — División: divide cada píxel entre el max de su imagen.
 * entrada  : float[B][H][W]
 * salida   : float[B][H][W]  valores en [0,1]
 * max_vals : float[B]
 * Grid     : dim3((W+15)/16, (H+15)/16), dim3(16,16)
 */
__global__ void dividir_por_max(float *entrada, float *salida,
                                float *max_vals, int B, int H, int W);
