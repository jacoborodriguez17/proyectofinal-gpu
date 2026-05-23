#pragma once

/*
 * Carga B imágenes en un buffer host con layout B×3×H×W (channel-first, float [0,1]).
 * Todas las imágenes se redimensionan al tamaño de la primera.
 * El buffer se reserva internamente con malloc y el llamador debe liberarlo con free().
 */
void cargar_imagenes(const char **archivos, int B, float **h_batch, int *H, int *W);

/*
 * Guarda un arreglo de floats [0,1] como PNG en escala de grises.
 * datos: puntero a H×W floats (una sola imagen).
 */
void guardar_png_gris(const char *nombre, float *datos, int H, int W);

/*
 * Guarda la primera imagen del batch como PNG en color RGB.
 * datos: puntero a B×3×H×W floats en layout CHW.
 */
void guardar_png_rgb(const char *nombre, float *datos, int B, int H, int W);
