#pragma once

/*
 * Kernel 2 — Detección de bordes (Sobel)
 * Entrada : float[B][H][W]  layout B*H*W  (escala de grises, [0,1])
 * Salida  : float[B][H][W]  layout B*H*W  (magnitud del gradiente)
 * Grid    : dim3((W+15)/16, (H+15)/16), dim3(16,16)
 * Nota    : píxeles del borde de la imagen se dejan en 0
 */
__global__ void deteccion_bordes(float *entrada, float *salida, int B, int H, int W);
