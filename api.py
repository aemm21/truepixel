import io
import os
import threading
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from checkpoint import cargar_checkpoint
from config import (
    CORS_ORIGINS,
    MAX_UPLOAD_BYTES,
    RUTA_MODELO_POR_DEFECTO,
    __version__,
    obtener_veredicto,
    seleccionar_dispositivo,
    transform_inferencia,
)
from explain import generar_explicaciones
from model import TruePixelNet

_EXPLAIN_LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al arrancar. Si falla, la API queda en estado degradado (503)."""
    app.state.dispositivo = seleccionar_dispositivo()
    app.state.transform = transform_inferencia()
    app.state.ruta_modelo = os.getenv("TRUEPIXEL_MODEL_PATH", RUTA_MODELO_POR_DEFECTO)
    app.state.modelo = None
    app.state.metadatos = {}
    app.state.error_modelo = None

    modelo = TruePixelNet().to(app.state.dispositivo)
    try:
        metadatos = cargar_checkpoint(modelo, app.state.ruta_modelo, app.state.dispositivo)
        modelo.eval()
        app.state.modelo = modelo
        app.state.metadatos = metadatos
        print(f"Modelo v{metadatos.get('version', __version__)} cargado desde "
              f"{app.state.ruta_modelo} en {app.state.dispositivo.type}")
    except Exception as exc:
        app.state.error_modelo = str(exc)
        print(f"⚠️ Modelo no disponible ({app.state.ruta_modelo}): {exc}")

    yield

    app.state.modelo = None


app = FastAPI(
    title="TruePixel API",
    description="API Open Source para la detección de imágenes generadas por IA",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

TIPOS_IMAGEN_PERMITIDOS = {
    "image/jpeg", "image/jpg", "image/png", "image/webp",
    "image/bmp", "image/tiff", "application/octet-stream",
}


@app.get("/")
def home():
    return {
        "mensaje": "Bienvenido a TruePixel API. Usa POST /verificar/ para analizar imágenes o POST /explicar/ para ver mapas de calor.",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
        "explicar": "/explicar/",
    }


@app.get("/health")
def health():
    """Estado del servicio y del modelo. Útil para orquestadores y monitoreo."""
    cargado = getattr(app.state, "modelo", None) is not None
    return {
        "status": "ok" if cargado else "degraded",
        "version": __version__,
        "device": getattr(app.state, "dispositivo", torch.device("cpu")).type,
        "model_loaded": cargado,
        "model_path": getattr(app.state, "ruta_modelo", None),
        "model_version": getattr(app.state, "metadatos", {}).get("version"),
        "error": getattr(app.state, "error_modelo", None),
    }


def _predecir(imagen):
    """Inferencia síncrona; se ejecuta en un threadpool para no bloquear el event loop."""
    tensor = app.state.transform(imagen).unsqueeze(0).to(app.state.dispositivo)
    with torch.no_grad():
        logit = app.state.modelo(tensor)
        probabilidad = torch.sigmoid(logit).item()
    return probabilidad


async def _leer_limitado(archivo: UploadFile):
    contenido = await archivo.read(MAX_UPLOAD_BYTES + 1)
    if len(contenido) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"La imagen supera el tamaño máximo permitido ({MAX_UPLOAD_BYTES} bytes).",
        )
    return contenido


async def _leer_imagen(archivo: UploadFile):
    """Valida tipo, tamaño y decodifica la imagen subida."""
    if archivo.content_type not in TIPOS_IMAGEN_PERMITIDOS:
        raise HTTPException(
            status_code=415,
            detail=f"Tipo de archivo no soportado: {archivo.content_type}. Sube una imagen JPEG/PNG/WebP/BMP/TIFF.",
        )

    contenido = await _leer_limitado(archivo)
    try:
        return Image.open(io.BytesIO(contenido)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="El archivo no es una imagen válida.")


def _asegurar_modelo():
    if getattr(app.state, "modelo", None) is None:
        raise HTTPException(
            status_code=503,
            detail=f"Modelo no disponible: {getattr(app.state, 'error_modelo', 'desconocido')}. "
                   f"Entrena el modelo con train.py antes de verificar imágenes.",
        )


@app.post("/verificar/")
async def verificar_imagen(file: UploadFile = File(...)):
    """
    Recibe una imagen y devuelve el porcentaje de probabilidad de que sea generada por IA.
    """
    _asegurar_modelo()
    imagen = await _leer_imagen(file)

    probabilidad = await run_in_threadpool(_predecir, imagen)
    porcentaje_ia = probabilidad * 100

    return {
        "archivo": file.filename,
        "probabilidad_ia_porcentaje": round(porcentaje_ia, 2),
        "veredicto": obtener_veredicto(probabilidad),
        "version_modelo": getattr(app.state, "metadatos", {}).get("version"),
        "version_api": __version__,
    }


@app.post("/explicar/")
async def explicar_imagen(file: UploadFile = File(...), tipo: str = "overlay"):
    """
    Devuelve una imagen PNG con la explicación del veredicto.
    `tipo` puede ser `overlay` (Grad-CAM sobre la imagen) o `espectral` (espectro de Fourier).
    """
    _asegurar_modelo()
    if tipo not in ("overlay", "espectral"):
        raise HTTPException(status_code=400, detail="El parámetro 'tipo' debe ser 'overlay' o 'espectral'.")

    imagen = await _leer_imagen(file)

    def _generar():
        with _EXPLAIN_LOCK:
            tensor = app.state.transform(imagen).unsqueeze(0).to(app.state.dispositivo)
            return generar_explicaciones(app.state.modelo, tensor, imagen)

    explicaciones = await run_in_threadpool(_generar)
    buffer = io.BytesIO()
    explicaciones[tipo].save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")
