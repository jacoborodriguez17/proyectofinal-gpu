#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "kernels/grises.h"
#include "kernels/bordes.h"
#include "kernels/normalizar.h"
#include "kernels/mse.h"
#include "utils/imagen.h"
#include "utils/timer.h"

/* Macro para verificar errores de la API de CUDA                      */

#define CUDA_CHECK(err)                                                        \
    do {                                                                       \
        cudaError_t _e = (err);                                                \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error en %s:%d -- %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/* Lista de imagenes del batch (minimo 8)                             */

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

/* Pipeline CPU completo — referencia para calcular speedup           */

static float pipeline_cpu(float *batch, int B, int H, int W) {
    int n = H * W;
    float *grises = (float *)malloc((size_t)B * n * sizeof(float));
    float *bordes  = (float *)malloc((size_t)B * n * sizeof(float));
    float *norm    = (float *)malloc((size_t)B * n * sizeof(float));

    clock_t t0 = clock();

    for (int b = 0; b < B; b++) {
        float *base = batch + b * 3 * n;
        for (int i = 0; i < n; i++)
            grises[b*n+i] = 0.2989f*base[0*n+i] + 0.5870f*base[1*n+i] + 0.1140f*base[2*n+i];
    }

    for (int b = 0; b < B; b++) {
        float *img = grises + b*n;
        float *out = bordes  + b*n;
        for (int fila = 0; fila < H; fila++) {
            for (int col = 0; col < W; col++) {
                if (fila==0||fila==H-1||col==0||col==W-1) { out[fila*W+col]=0.0f; continue; }
                float tl=img[(fila-1)*W+(col-1)], tc=img[(fila-1)*W+col], tr=img[(fila-1)*W+(col+1)];
                float ml=img[fila*W+(col-1)],                              mr=img[fila*W+(col+1)];
                float bl=img[(fila+1)*W+(col-1)], bc=img[(fila+1)*W+col], br=img[(fila+1)*W+(col+1)];
                float gx=-tl+tr-2.0f*ml+2.0f*mr-bl+br;
                float gy=-tl-2.0f*tc-tr+bl+2.0f*bc+br;
                out[fila*W+col] = sqrtf(gx*gx+gy*gy);
            }
        }
    }

    for (int b = 0; b < B; b++) {
        float mx = 0.0f;
        for (int i = 0; i < n; i++) if (bordes[b*n+i] > mx) mx = bordes[b*n+i];
        for (int i = 0; i < n; i++) norm[b*n+i] = (mx>0.0f) ? bordes[b*n+i]/mx : 0.0f;
    }

    float mse_sum = 0.0f;
    for (int b = 0; b < B; b++) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) { float d=norm[b*n+i]-norm[i]; sum+=d*d; }
        mse_sum += sqrtf(sum/n);
    }
    (void)mse_sum;

    clock_t t1 = clock();
    float ms = 1000.0f * (float)(t1-t0) / CLOCKS_PER_SEC;

    free(grises); free(bordes); free(norm);
    return ms;
}

/* Main                                                                */

int main(void) {
    const int B = NUM_IMAGENES;

    /* Cargar imagenes en CPU */
    float *h_batch = NULL;
    int H, W;
    cargar_imagenes(archivos, B, &h_batch, &H, &W);
    printf("Batch cargado: %d imagenes de %dx%d\n\n", B, H, W);

    /* Reservar memoria en GPU */
    float *d_entrada, *d_grises, *d_bordes, *d_normalizada, *d_max_vals, *d_rmse;

    size_t sz_entrada = (size_t)B * 3 * H * W * sizeof(float);
    size_t sz_batch   = (size_t)B * H * W * sizeof(float);
    size_t sz_escalar = (size_t)B * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_entrada,     sz_entrada));
    CUDA_CHECK(cudaMalloc(&d_grises,      sz_batch));
    CUDA_CHECK(cudaMalloc(&d_bordes,      sz_batch));
    CUDA_CHECK(cudaMalloc(&d_normalizada, sz_batch));
    CUDA_CHECK(cudaMalloc(&d_max_vals,    sz_escalar));
    CUDA_CHECK(cudaMalloc(&d_rmse,        sz_escalar));

    /* Transferencia H->D (unica al inicio) */
    Timer t_htod;
    timer_crear(&t_htod);
    timer_iniciar(&t_htod);
    CUDA_CHECK(cudaMemcpy(d_entrada, h_batch, sz_entrada, cudaMemcpyHostToDevice));
    float ms_htod = timer_detener(&t_htod);

    /* Configuracion del grid
     * Bloque 16x16 = 256 threads (multiplo de 32, 8 warps por bloque).
     * Permite que el SM oculte latencias de memoria con suficiente
     * paralelismo de threads en vuelo simultaneo.
     */
    dim3 bloque2d(16, 16);
    dim3 grid2d((W+15)/16, (H+15)/16);
    int  bloque_red = 256;

    /* Kernel 1 — Escala de grises */
    Timer t_k1;
    timer_crear(&t_k1);
    timer_iniciar(&t_k1);
    escala_grises<<<grid2d, bloque2d>>>(d_entrada, d_grises, B, H, W);
    float ms_k1 = timer_detener(&t_k1);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    /* Kernel 2 — Deteccion de bordes */
    Timer t_k2;
    timer_crear(&t_k2);
    timer_iniciar(&t_k2);
    deteccion_bordes<<<grid2d, bloque2d>>>(d_grises, d_bordes, B, H, W);
    float ms_k2 = timer_detener(&t_k2);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    /* Kernel 3 — Normalizacion (dos pasos) */
    Timer t_k3;
    timer_crear(&t_k3);
    timer_iniciar(&t_k3);

    reduccion_max<<<B, bloque_red, bloque_red * sizeof(float)>>>(
        d_bordes, d_max_vals, H, W);
    CUDA_CHECK(cudaGetLastError());

    dividir_por_max<<<grid2d, bloque2d>>>(d_bordes, d_normalizada, d_max_vals, B, H, W);
    float ms_k3 = timer_detener(&t_k3);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    /* Kernel 4 — MSE/RMSE vs imagen de referencia (imagen 0 del batch) */
    Timer t_k4;
    timer_crear(&t_k4);
    timer_iniciar(&t_k4);

    calcular_mse<<<B, bloque_red, bloque_red * sizeof(float)>>>(
        d_normalizada, d_normalizada, d_rmse, B, H, W);
    float ms_k4 = timer_detener(&t_k4);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    /* Transferencia D->H (unica al final) */
    float *h_grises      = (float *)malloc(sz_batch);
    float *h_bordes      = (float *)malloc(sz_batch);
    float *h_normalizada = (float *)malloc(sz_batch);
    float *h_rmse        = (float *)malloc(sz_escalar);

    Timer t_dtoh;
    timer_crear(&t_dtoh);
    timer_iniciar(&t_dtoh);
    CUDA_CHECK(cudaMemcpy(h_grises,      d_grises,      sz_batch,   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bordes,      d_bordes,      sz_batch,   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_normalizada, d_normalizada, sz_batch,   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rmse,        d_rmse,        sz_escalar, cudaMemcpyDeviceToHost));
    float ms_dtoh = timer_detener(&t_dtoh);

    /* Guardar imagenes de verificacion */
    system("mkdir -p resultados");
    guardar_png_rgb ("resultados/imagen_00_original.png",    h_batch,         B, H, W);
    guardar_png_gris("resultados/imagen_00_grises.png",      h_grises,           H, W);
    guardar_png_gris("resultados/imagen_00_bordes.png",      h_bordes,           H, W);
    guardar_png_gris("resultados/imagen_00_normalizada.png", h_normalizada,      H, W);

    /* Datos float crudos para verificacion Python (evita error de cuantizacion PNG) */
    {
        FILE *fg = fopen("resultados/grises_00_raw.bin", "wb");
        if (fg) { fwrite(h_grises, sizeof(float), (size_t)H*W, fg); fclose(fg); }
        FILE *fb = fopen("resultados/bordes_00_raw.bin", "wb");
        if (fb) { fwrite(h_bordes, sizeof(float), (size_t)H*W, fb); fclose(fb); }
    }

    /* Guardar RMSE por imagen */
    FILE *f = fopen("resultados/rmse_por_imagen.txt", "w");
    if (f) {
        fprintf(f, "RMSE por imagen (referencia = imagen_00)\n");
        fprintf(f, "----------------------------------------\n");
        for (int b = 0; b < B; b++)
            fprintf(f, "imagen_%02d : %.6f\n", b, h_rmse[b]);
        fclose(f);
    }

    /* Resumen de tiempos GPU */
    float ms_gpu   = ms_k1 + ms_k2 + ms_k3 + ms_k4;
    float ms_total = ms_htod + ms_gpu + ms_dtoh;

    printf("--- Tiempos GPU ---\n");
    printf("  H->D transfer  : %7.3f ms\n", ms_htod);
    printf("  Kernel 1       : %7.3f ms\n", ms_k1);
    printf("  Kernel 2       : %7.3f ms\n", ms_k2);
    printf("  Kernel 3       : %7.3f ms\n", ms_k3);
    printf("  Kernel 4       : %7.3f ms\n", ms_k4);
    printf("  D->H transfer  : %7.3f ms\n", ms_dtoh);
    printf("  Total          : %7.3f ms\n\n", ms_total);

    /* Speedup vs CPU */
    printf("Ejecutando pipeline CPU...\n");
    float ms_cpu = pipeline_cpu(h_batch, B, H, W);
    printf("  Tiempo CPU     : %7.3f ms\n", ms_cpu);
    printf("  Speedup GPU    : %7.2fx\n\n", ms_cpu / ms_gpu);

    /* RMSE por imagen */
    printf("--- RMSE por imagen (referencia = imagen_00) ---\n");
    for (int b = 0; b < B; b++)
        printf("  imagen_%02d : %.6f\n", b, h_rmse[b]);

    /* Liberar recursos */
    CUDA_CHECK(cudaFree(d_entrada));
    CUDA_CHECK(cudaFree(d_grises));
    CUDA_CHECK(cudaFree(d_bordes));
    CUDA_CHECK(cudaFree(d_normalizada));
    CUDA_CHECK(cudaFree(d_max_vals));
    CUDA_CHECK(cudaFree(d_rmse));

    free(h_batch); free(h_grises); free(h_bordes); free(h_normalizada); free(h_rmse);

    timer_destruir(&t_htod);
    timer_destruir(&t_k1);
    timer_destruir(&t_k2);
    timer_destruir(&t_k3);
    timer_destruir(&t_k4);
    timer_destruir(&t_dtoh);

    return 0;
}
