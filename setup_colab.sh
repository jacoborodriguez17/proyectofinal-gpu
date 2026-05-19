#!/bin/bash
# Descarga las cabeceras stb_image y crea los directorios necesarios

set -e

echo "Descargando stb_image.h ..."
curl -sL https://raw.githubusercontent.com/nothings/stb/master/stb_image.h \
     -o utils/stb_image.h

echo "Descargando stb_image_write.h ..."
curl -sL https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h \
     -o utils/stb_image_write.h

echo "Creando directorios ..."
mkdir -p imagenes resultados

echo "Listo. Ahora compila con:"
echo "  nvcc -O2 -o pipeline main.cu kernels/grises.cu kernels/bordes.cu \\"
echo "       utils/imagen.cu utils/timer.cu -lm"
