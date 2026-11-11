# Changelog

Todas las modificaciones relevantes de este proyecto se documentan aquí.
El formato sigue una adaptación de [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/)
y el proyecto usa [Versionado Semántico](https://semver.org/lang/es/).

## [3.0.0] - 2026-11-11

### Añadido
- `explain.py`: explicabilidad con Grad-CAM sobre la rama espacial y mapa del espectro de Fourier superpuesto/coloreado.
- `verify.py`: análisis por lote de archivos, carpetas recursivas y ZIP, con salida CSV/JSON, resumen y `--explicar` para guardar mapas de calor.
- API: endpoint `POST /explicar/` que devuelve un PNG (`overlay` o `espectral`).
- `config.py`: configuración centralizada (tamaño de imagen, normalización, umbrales, límites, CORS, device y transformaciones).
- `checkpoint.py`: guardado/carga de pesos con metadatos y compatibilidad con `state_dict` antiguos.
- API de producción: `lifespan`, endpoint `/health`, CORS, límite de tamaño de subida (413), validación de tipo (415), estado degradado con 503 e inferencia en threadpool.
- Split estratificado Train/Validación/Test y métricas AUC-ROC, F1, Precision, Recall y matriz de confusión.
- Aumentos de datos solo para entrenamiento y `pos_weight` para desbalance de clases.
- `pyproject.toml`, `Dockerfile` y `.dockerignore`.
- Tests de `config`, `checkpoint`, `api` (incluida una inferencia end-to-end) y de las métricas/split.
- CI con GitHub Actions sobre Ubuntu + PyTorch CPU.

### Cambiado
- `model.py`: pooling adaptativo (384 features en vez de 301,056), sin Sigmoid en el modelo y entrada independiente del tamaño.
- `fourier.py`: luminancia perceptual, eliminación del componente DC, ventana de Hann y espectro invariante a escala/offset.
- `train.py`: `BCEWithLogitsLoss`, semilla reproducible, `num_workers`, guardado de checkpoint versionado y evaluación final en test.
- `dataset.py`: listado determinista, soporte de más extensiones, modo recursivo y tolerancia a imágenes corruptas.
- `verify.py`: usa la configuración centralizada, devuelve dict, admite `--json`, `--modelo` y falla de forma clara sin pesos.
- `README.md` / `README_es.md`: nuevas opciones, límites conocidos y sección de pruebas.

### Corregido
- Dependencia `multipart` → `python-multipart`, que rompía los uploads de FastAPI.
- Carga insegura de pesos (`weights_only=True`).
- Servir predicciones con pesos aleatorios cuando faltaba el `.pth`.
- División entre cero con `drop_last` en datasets pequeños.
- `.gitignore` con patrones mal formados y sin `__pycache__/`.

### Seguridad
- Límite de píxeles (`TRUEPIXEL_MAX_IMAGE_PIXELS`) contra decompression bombs en la API, los lotes y el dataset.
- Endurecimiento de ZIP (`TRUEPIXEL_MAX_ZIP_FILES`, `TRUEPIXEL_MAX_ZIP_BYTES`): límite de entradas y bytes descomprimidos, normalización de rutas (zip-slip) y limpieza de temporales.
- Sanitización de CSV injection en las exportaciones de `verify.py`.
- `/health` y los errores 503 ya no exponen rutas internas ni mensajes de excepción.
- Contenedor Docker ejecutado como usuario sin privilegios y CI con permisos de solo lectura.
- Cierre explícito de los archivos de imagen abiertos con Pillow (evita fugas de descriptores en lotes grandes).

### Nota de compatibilidad
- Los pesos anteriores a V3.0 no son compatibles con la nueva arquitectura. Es necesario reentrenar.
