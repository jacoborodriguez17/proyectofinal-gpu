#include "bordes.h"
#include <math.h>

/* TODO (Valeria): implementar este kernel
 *
 * Cada thread procesa el pixel (fila, col) de TODAS las imagenes del batch.
 * Indice: b*H*W + fila*W + col
 * Los pixeles del borde (fila==0, fila==H-1, col==0, col==W-1) se dejan en 0.
 */
__global__ void deteccion_bordes(float *entrada, float *salida, int B, int H, int W) {
    /* IMPLEMENTAR */
}
