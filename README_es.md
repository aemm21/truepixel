👁️ TruePixel: Motor de Verificación de Imágenes por IA de Código Abierto

👉 English version of this document

TruePixel es un motor ligero, privado y de código abierto diseñado para la detección de imágenes sintéticas generadas por Inteligencia Artificial (Deepfakes, Midjourney, DALL-E, Stable Diffusion).

A diferencia de los detectores comerciales que analizan imperfecciones visuales superficiales, TruePixel opera bajo un enfoque científico dual: analiza la geometría del píxel y busca anomalías físicas en el dominio de la frecuencia de Fourier que son matemáticamente imposibles de camuflar para los modelos de difusión generativa.

⚠️ AVISO CRÍTICO: ENTRENAMIENTO REQUERIDO

🔴 IMPORTANTE: TruePixel se entrega como una arquitectura de software. Inicialmente, los pesos del modelo se configuran de manera aleatoria.

Es estrictamente necesario entrenar el modelo con tu propio conjunto de datos de imágenes reales e imágenes generadas por IA antes de hacer uso de verify.py o api.py. Ejecutar el modelo sin entrenar dará como resultado veredictos aleatorios (por lo general, indicando siempre "Sospechoso" o "Real" porque las neuronas no entrenadas no saben diferenciar los patrones).

Para que TruePixel sea funcional, sigue las instrucciones de la Sección de Entrenamiento para generar tu archivo de pesos truepixel_v1.pth.

🧠 Arquitectura de Red Dual

TruePixel procesa cada archivo de imagen a través de dos canales paralelos independientes antes de emitir un juicio probabilístico unificado:

               ┌─────────────────────────────┐
               │     Imagen de Entrada       │
               └──────────────┬──────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
     [ Rama Espacial ]             [ Rama de Frecuencia ]
  - Análisis de píxeles/texturas -  - Transformada de Fourier (FFT) -
  - Convolución 2D (RGB)         -  - Magnitud logarítmica espectral-
              │                               │
              └───────────────┬───────────────┘
                              ▼
                   [ Fusión de Características ]
                              │
                              ▼
                     [ Clasificador Sigmoide ]
                              │
                              ▼
                     📊 Probabilidad de IA %


Rama Espacial (Visual): Una red neuronal convolucional (CNN) optimizada para identificar artefactos de texturas, incoherencias de bordes e inconsistencias de color locales.

Rama de Frecuencia (Espectral): Convierte el tensor de la imagen mediante una Transformada Rápida de Fourier 2D ($\text{FFT}_{2\text{D}}$). Las imágenes creadas por IA exhiben patrones periódicos y picos geométricos en las altas frecuencias debido a los procesos matemáticos de upsampling e interpolación de píxeles.

📦 Estructura del Repositorio

El proyecto mantiene una estructura modular y limpia para facilitar la extensibilidad del código:

Archivo

Descripción

model.py

Definición de la arquitectura de la red neuronal unificada (TruePixelNet).

fourier.py

Núcleo de cálculo de espectro de magnitud de Fourier optimizado en PyTorch.

dataset.py

Cargador de datos con transformaciones dinámicas, normalización y validaciones de ruta.

train.py

Script de entrenamiento CLI avanzado con división automática de validación y guardado inteligente.

verify.py

Herramienta local rápida para evaluar imágenes individuales desde la terminal.

api.py

Servidor FastAPI con documentación interactiva integrada para entornos web.

🚀 Guía de Inicio Rápido

1. Clonar e Instalar dependencias

Asegúrate de tener un entorno virtual activo y ejecuta:

# Clonar el proyecto de forma local
git clone [https://github.com/aemm21/truepixel.git](https://github.com/aemm21/truepixel.git)
cd truepixel

# Instalar dependencias requeridas
pip install -r requirements.txt


2. Entrenamiento Paramétrico (CLI)

Antes de verificar imágenes, necesitas entrenar la red neuronal. Organiza tus carpetas colocando fotos reales en ./data/reales e imágenes de IA en ./data/sinteticas.

Puedes ajustar el entrenamiento directamente desde la terminal pasando argumentos:

# Ejecutar entrenamiento con parámetros personalizados
python train.py --epochs 40 --batch_size 16 --lr 0.0003 --val_split 0.20


Parámetros Disponibles:

--dir_reales: Ruta al directorio de fotos reales (Por defecto: ./data/reales).

--dir_sinteticas: Ruta al directorio de imágenes generadas por IA (Por defecto: ./data/sinteticas).

--epochs: Cantidad de veces que se procesará el set de datos.

--lr: Tasa de aprendizaje (Learning Rate) para el optimizador Adam.

--val_split: Porcentaje de datos destinados a validación (Ej: 0.20 = 20%).

--output: Nombre del archivo de salida para los pesos (Por defecto: truepixel_v1.pth).

🌐 Despliegue de la API Web

Una vez entrenado, TruePixel incluye un servidor FastAPI autolocalizable para que puedas consultar el modelo desde clientes externos (aplicaciones móviles, extensiones de navegador o sistemas de moderación):

# Iniciar el servidor API localmente
uvicorn api:app --reload


Una vez encendido el servidor, accede a la documentación interactiva en:
👉 http://127.0.0.1:8000/docs

Allí podrás enviar solicitudes POST de prueba adjuntando una imagen y obteniendo una respuesta en formato JSON estructurado:

{
  "archivo": "muestra_sospechosa.png",
  "probabilidad_ia_porcentaje": 91.43,
  "veredicto": "ALTA PROBABILIDAD DE DEEPFAKE / IA"
}


🤝 Cómo Contribuir

¡Las contribuciones de la comunidad científica y de desarrollo son las que hacen fuerte a este software!
Para proponer mejoras en el análisis espectral o sugerir arquitecturas convolucionales más eficientes, por favor lee nuestra guía de Contribución.

📄 Licencia

Este proyecto está bajo la Licencia MIT. Puedes usarlo, modificarlo y distribuirlo de manera totalmente gratuita tanto para fines educativos como comerciales. Consulta el archivo LICENSE para más detalles.