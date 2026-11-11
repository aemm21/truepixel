👁️ TruePixel: Motor de Verificación de Imágenes por IA de Código Abierto

**Versión 3.0.0** — ver [CHANGELOG.md](CHANGELOG.md)

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

| Archivo | Descripción |
|---|---|
| `model.py` | Arquitectura de la red neuronal unificada (TruePixelNet). |
| `fourier.py` | Espectro de magnitud de Fourier optimizado en PyTorch. |
| `dataset.py` | Cargador de datos determinista, recursivo y tolerante a imágenes corruptas. |
| `config.py` | Configuración centralizada: rutas, umbrales, límites, CORS, device y transformaciones. |
| `checkpoint.py` | Guardado/carga de pesos con metadatos y compatibilidad con `state_dict` antiguos. |
| `explain.py` | Explicabilidad: Grad-CAM y mapa espectral coloreado. |
| `train.py` | Entrenamiento CLI con split estratificado Train/Val/Test y métricas AUC/F1. |
| `verify.py` | Herramienta CLI para evaluar una imagen (reporte legible o `--json`). |
| `api.py` | Servidor FastAPI con `/health`, CORS y documentación interactiva. |
| `tests/` | Suite de `pytest` (Fourier, modelo, dataset, métricas, checkpoints y API). |
| `Dockerfile` | Imagen de producción con PyTorch CPU y uvicorn. |
| `CHANGELOG.md` | Historial de versiones. |

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

--test_split: Porcentaje de datos reservados para el test final (Por defecto: 0.15).

--num_workers: Procesos de carga de datos (0 = sin multiproceso).

--seed: Semilla para reproducibilidad (Por defecto: 42).

--output: Nombre del archivo de salida para los pesos (Por defecto: truepixel_v1.pth).

La división Train/Validación/Test es **estratificada** para conservar la proporción de clases. Al final del entrenamiento se reportan Accuracy, AUC-ROC, F1, Precision, Recall y la matriz de confusión sobre el conjunto de test.

🔎 Verificación Local y por Lotes

`verify.py` acepta una imagen, una carpeta (recursiva) o un ZIP:

# Una imagen
python verify.py foto.jpg

# Carpeta con salida CSV y mapas de calor (Grad-CAM + espectro)
python verify.py ./imagenes --csv resultados.csv --explicar ./mapas

# ZIP con salida JSON
python verify.py lote.zip --json > resultados.json


Cada análisis puede generar dos imágenes explicativas: `*_overlay.png` (zonas que más influyen en el veredicto) y `*_espectral.png` (espectro de Fourier).

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
  "veredicto": "ALTA PROBABILIDAD DE DEEPFAKE / IA",
  "version_modelo": "3.0.0",
  "version_api": "3.0.0"
}

Endpoints disponibles:

- `GET /health`: estado del servicio, device, versión y si el modelo está cargado.
- `POST /verificar/`: analiza una imagen. Devuelve `503` si el modelo no está entrenado, `413` si supera el límite y `415` si el tipo no es una imagen soportada.
- `POST /explicar/?tipo=overlay|espectral`: devuelve un PNG con el Grad-CAM o el espectro de Fourier.

Variables de entorno:

| Variable | Por defecto | Descripción |
|---|---|---|
| `TRUEPIXEL_MODEL_PATH` | `truepixel_v1.pth` | Ruta del checkpoint. |
| `TRUEPIXEL_MAX_UPLOAD_BYTES` | `10485760` (10 MB) | Tamaño máximo de subida. |
| `TRUEPIXEL_UMBRAL_IA` | `0.70` | Umbral de "alta probabilidad de IA". |
| `TRUEPIXEL_UMBRAL_SOSPECHOSO` | `0.40` | Umbral de "sospechoso". |
| `TRUEPIXEL_CORS_ORIGINS` | `*` | Orígenes permitidos, separados por coma. |

Despliegue con Docker:

# Construir la imagen y montar tus pesos entrenados
docker build -t truepixel .
docker run -p 8000:8000 -v "$PWD/truepixel_v1.pth:/app/truepixel_v1.pth" truepixel


🧪 Pruebas

El proyecto incluye una suite con `pytest` para `fourier.py`, `model.py`, `dataset.py`, `config.py`, `checkpoint.py`, `explain.py`, `train.py` (métricas), análisis por lotes y la API (incluida una inferencia end-to-end):

# Instalar dependencias de desarrollo y ejecutar las pruebas
pip install -r requirements-dev.txt
python -m pytest -q


Las pruebas también se ejecutan automáticamente en cada push mediante GitHub Actions (`.github/workflows/ci.yml`, Ubuntu + PyTorch CPU).

🤝 Cómo Contribuir

¡Las contribuciones de la comunidad científica y de desarrollo son las que hacen fuerte a este software!
Para proponer mejoras en el análisis espectral o sugerir arquitecturas convolucionales más eficientes, por favor lee nuestra guía de Contribución.

⚠️ Limitaciones Conocidas

- **Confounder de resize/JPEG:** el reescalado a 224x224 y la compresión JPEG introducen artefactos de alta frecuencia que pueden parecerse a la firma de una IA. Entrena y evalúa con imágenes de resolución y compresión comparables entre clases para evitar que el modelo aprenda atajos.
- **El espectro depende del preprocesado:** la rama de Fourier normaliza cada espectro por imagen y elimina el componente DC, pero un cambio de pipeline (normalización, tamaño, ventana) altera las características. Mantén el mismo preprocesado en entrenamiento e inferencia.
- **Datos, no magia:** la calidad del veredicto depende del dataset. Incluye imágenes reales y sintéticas de múltiples generadores (Midjourney, DALL-E, Stable Diffusion) y evalúa con un conjunto de test separado.

📄 Licencia

Este proyecto está bajo la Licencia MIT. Puedes usarlo, modificarlo y distribuirlo de manera totalmente gratuita tanto para fines educativos como comerciales. Consulta el archivo LICENSE para más detalles.