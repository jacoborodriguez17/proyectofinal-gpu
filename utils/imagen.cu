#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "imagen.h"

#include <stdio.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ */
/* Carga un batch de imágenes en formato B×3×H×W float [0,1]          */
/* ------------------------------------------------------------------ */
void cargar_imagenes(const char **archivos, int B, float **h_batch,
                     int *H_out, int *W_out) {
    int H, W, canales;

    /* Primera imagen para obtener dimensiones de referencia */
    unsigned char *ref = stbi_load(archivos[0], &W, &H, &canales, 3);
    if (!ref) {
        fprintf(stderr, "Error al cargar imagen: %s\n", archivos[0]);
        exit(EXIT_FAILURE);
    }
    stbi_image_free(ref);

    *H_out = H;
    *W_out = W;

    *h_batch = (float *)malloc((size_t)B * 3 * H * W * sizeof(float));
    if (!*h_batch) {
        fprintf(stderr, "Error: malloc fallo para el batch\n");
        exit(EXIT_FAILURE);
    }

    for (int b = 0; b < B; b++) {
        int w_img, h_img;
        unsigned char *img = stbi_load(archivos[b], &w_img, &h_img, &canales, 3);
        if (!img) {
            fprintf(stderr, "Error al cargar imagen: %s\n", archivos[b]);
            exit(EXIT_FAILURE);
        }
        if (h_img != H || w_img != W) {
            fprintf(stderr, "Advertencia: %s tiene tamano %dx%d, se esperaba %dx%d\n",
                    archivos[b], w_img, h_img, W, H);
        }

        /* Convertir HWC uint8 → CHW float [0,1] */
        float *base = *h_batch + (size_t)b * 3 * H * W;
        for (int i = 0; i < H * W; i++) {
            base[0 * H * W + i] = img[i * 3 + 0] / 255.0f;  /* R */
            base[1 * H * W + i] = img[i * 3 + 1] / 255.0f;  /* G */
            base[2 * H * W + i] = img[i * 3 + 2] / 255.0f;  /* B */
        }

        stbi_image_free(img);
    }
}

/* ------------------------------------------------------------------ */
/* Guarda una imagen en escala de grises                               */
/* datos: H×W floats [0,1]                                            */
/* ------------------------------------------------------------------ */
void guardar_png_gris(const char *nombre, float *datos, int H, int W) {
    unsigned char *buf = (unsigned char *)malloc(H * W);
    if (!buf) { fprintf(stderr, "Error: malloc fallo en guardar_png_gris\n"); return; }

    for (int i = 0; i < H * W; i++) {
        float v = datos[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        buf[i] = (unsigned char)(v * 255.0f + 0.5f);
    }

    if (!stbi_write_png(nombre, W, H, 1, buf, W))
        fprintf(stderr, "Error al guardar: %s\n", nombre);

    free(buf);
}

/* ------------------------------------------------------------------ */
/* Guarda una imagen RGB                                               */
/* datos: 3×H×W floats [0,1] en layout CHW                           */
/* ------------------------------------------------------------------ */
void guardar_png_rgb(const char *nombre, float *datos, int B, int H, int W) {
    (void)B;  /* se guarda siempre la primera imagen del batch */
    unsigned char *buf = (unsigned char *)malloc(H * W * 3);
    if (!buf) { fprintf(stderr, "Error: malloc fallo en guardar_png_rgb\n"); return; }

    for (int i = 0; i < H * W; i++) {
        float r = datos[0 * H * W + i];
        float g = datos[1 * H * W + i];
        float b = datos[2 * H * W + i];

        if (r < 0.0f) r = 0.0f; if (r > 1.0f) r = 1.0f;
        if (g < 0.0f) g = 0.0f; if (g > 1.0f) g = 1.0f;
        if (b < 0.0f) b = 0.0f; if (b > 1.0f) b = 1.0f;

        buf[i * 3 + 0] = (unsigned char)(r * 255.0f + 0.5f);
        buf[i * 3 + 1] = (unsigned char)(g * 255.0f + 0.5f);
        buf[i * 3 + 2] = (unsigned char)(b * 255.0f + 0.5f);
    }

    if (!stbi_write_png(nombre, W, H, 3, buf, W * 3))
        fprintf(stderr, "Error al guardar: %s\n", nombre);

    free(buf);
}
