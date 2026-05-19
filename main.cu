#include <stdio.h>
#include <stdlib.h>

#include "kernels/grises.h"
#include "kernels/bordes.h"
#include "utils/imagen.h"
#include "utils/timer.h"

/* ------------------------------------------------------------------ */
/* Macro para verificar errores de la API de CUDA                      */
/* ------------------------------------------------------------------ */
#define CUDA_CHECK(err)                                                        \
    do {                                                                       \
        cudaError_t _e = (err);                                                \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error en %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/* ------------------------------------------------------------------ */
/* Lista de imágenes del batch (mínimo 8, todas deben existir)         */
/* ------------------------------------------------------------------ */
#define NUM_IMAGENES 8

static const char *archivos[NUM_IMAGENES] = {
    "imagenes/img_00.png",
    "imagenes/img_01.png",
    "imagenes/img_02.png",
    "imagenes/img_03.png",
    "imagenes/img_04.png",
    "imagenes/img_05.png",
    "imagenes/img_06.png",
    "imagenes/img_07.png",
};

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */
int main(void) {
    const int B = NUM_IMAGENES;

    /* ---------- 1. Cargar imágenes en CPU ---------- */
    float *h_batch = NULL;
    int H, W;
    cargar_imagenes(archivos, B, &h_batch, &H, &W);
    printf("Batch cargado: %d imagenes de %dx%d\n", B, H, W);

    /* ---------- 2. Reservar memoria en GPU ---------- */
    float *d_entrada, *d_grises, *d_bordes;

    size_t sz_entrada = (size_t)B * 3 * H * W * sizeof(float);
    size_t sz_grises  = (size_t)B * H * W * sizeof(float);
    size_t sz_bordes  = (size_t)B * H * W * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_entrada, sz_entrada));
    CUDA_CHECK(cudaMalloc(&d_grises,  sz_grises));
    CUDA_CHECK(cudaMalloc(&d_bordes,  sz_bordes));

    /* ---------- 3. Transferencia H→D (una sola vez) ---------- */
    Timer t_htod;
    timer_crear(&t_htod);
    timer_iniciar(&t_htod);

    CUDA_CHECK(cudaMemcpy(d_entrada, h_batch, sz_entrada, cudaMemcpyHostToDevice));

    float ms_htod = timer_detener(&t_htod);
    printf("Transferencia H->D : %.3f ms\n", ms_htod);

    /* ---------- 4. Configuración del grid ----------
     * Bloque de 16×16 = 256 threads (múltiplo de 32, justificado: ocupa
     * 8 warps por bloque y permite que la GPU oculte latencias de memoria
     * con suficiente paralelismo dentro de cada SM).
     */
    dim3 bloque(16, 16);
    dim3 grid((W + bloque.x - 1) / bloque.x,
              (H + bloque.y - 1) / bloque.y);

    /* ---------- 5. Kernel 1 — Escala de grises ---------- */
    Timer t_k1;
    timer_crear(&t_k1);
    timer_iniciar(&t_k1);

    escala_grises<<<grid, bloque>>>(d_entrada, d_grises, B, H, W);
    CUDA_CHECK(cudaGetLastError());

    float ms_k1 = timer_detener(&t_k1);
    printf("Kernel 1 (grises)  : %.3f ms\n", ms_k1);

    /* ---------- 6. Kernel 2 — Detección de bordes ---------- */
    Timer t_k2;
    timer_crear(&t_k2);
    timer_iniciar(&t_k2);

    deteccion_bordes<<<grid, bloque>>>(d_grises, d_bordes, B, H, W);
    CUDA_CHECK(cudaGetLastError());

    float ms_k2 = timer_detener(&t_k2);
    printf("Kernel 2 (bordes)  : %.3f ms\n", ms_k2);

    /* ---------- 7. Bajar resultados D→H para guardar PNGs ----------
     * Esta es la ÚNICA transferencia D→H del avance.
     * En el pipeline final también se bajan grises y normalizada.
     */
    float *h_grises = (float *)malloc(sz_grises);
    float *h_bordes = (float *)malloc(sz_bordes);

    Timer t_dtoh;
    timer_crear(&t_dtoh);
    timer_iniciar(&t_dtoh);

    CUDA_CHECK(cudaMemcpy(h_grises, d_grises, sz_grises, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bordes, d_bordes, sz_bordes, cudaMemcpyDeviceToHost));

    float ms_dtoh = timer_detener(&t_dtoh);
    printf("Transferencia D->H : %.3f ms\n", ms_dtoh);

    /* ---------- 8. Guardar imágenes de verificación ---------- */
    system("mkdir -p resultados");

    /* Primera imagen del batch (índice 0) */
    guardar_png_rgb ("resultados/imagen_00_original.png", h_batch,         B, H, W);
    guardar_png_gris("resultados/imagen_00_grises.png",   h_grises,           H, W);
    guardar_png_gris("resultados/imagen_00_bordes.png",   h_bordes,           H, W);

    printf("Imagenes guardadas en resultados/\n");

    /* ---------- 9. Resumen de tiempos ---------- */
    float ms_total = ms_htod + ms_k1 + ms_k2 + ms_dtoh;
    printf("\n--- Resumen de tiempos ---\n");
    printf("  H->D transfer  : %7.3f ms\n", ms_htod);
    printf("  Kernel 1       : %7.3f ms\n", ms_k1);
    printf("  Kernel 2       : %7.3f ms\n", ms_k2);
    printf("  D->H transfer  : %7.3f ms\n", ms_dtoh);
    printf("  Total pipeline : %7.3f ms\n", ms_total);

    /* ---------- 10. Liberar recursos ---------- */
    CUDA_CHECK(cudaFree(d_entrada));
    CUDA_CHECK(cudaFree(d_grises));
    CUDA_CHECK(cudaFree(d_bordes));

    free(h_batch);
    free(h_grises);
    free(h_bordes);

    timer_destruir(&t_htod);
    timer_destruir(&t_k1);
    timer_destruir(&t_k2);
    timer_destruir(&t_dtoh);

    return 0;
}
