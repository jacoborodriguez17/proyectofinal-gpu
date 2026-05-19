#include "timer.h"

void timer_crear(Timer *t) {
    cudaEventCreate(&t->inicio);
    cudaEventCreate(&t->fin);
}

void timer_iniciar(Timer *t) {
    cudaEventRecord(t->inicio);
}

float timer_detener(Timer *t) {
    float ms = 0.0f;
    cudaEventRecord(t->fin);
    cudaEventSynchronize(t->fin);
    cudaEventElapsedTime(&ms, t->inicio, t->fin);
    return ms;
}

void timer_destruir(Timer *t) {
    cudaEventDestroy(t->inicio);
    cudaEventDestroy(t->fin);
}
