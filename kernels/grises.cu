#include "grises.h"

/* TODO (Jesus): implementar este kernel
 *
 * Cada thread procesa el pixel (fila, col) de TODAS las imagenes del batch.
 * Indice en entrada: b*3*H*W + canal*H*W + fila*W + col
 * Indice en salida:  b*H*W + fila*W + col
 */
__global__ void escala_grises(float *entrada, float *salida, int B, int H, int W) {
    /* IMPLEMENTAR */
}
