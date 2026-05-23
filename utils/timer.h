#pragma once
#include <cuda_runtime.h>

typedef struct {
    cudaEvent_t inicio;
    cudaEvent_t fin;
} Timer;

void timer_crear(Timer *t);
void timer_iniciar(Timer *t);
float timer_detener(Timer *t);   /* retorna milisegundos */
void timer_destruir(Timer *t);
