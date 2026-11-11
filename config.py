"""Configuración centralizada de TruePixel (rutas, umbrales, transformaciones y device)."""

import os

import torch
from torchvision import transforms

__version__ = "3.0.0"

# --- Imagen / normalización -------------------------------------------------
IMAGE_SIZE = 224
MEDIA_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]
EXTENSIONES_VALIDAS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif")

# --- Modelo / inferencia ----------------------------------------------------
RUTA_MODELO_POR_DEFECTO = os.getenv("TRUEPIXEL_MODEL_PATH", "truepixel_v1.pth")
UMBRAL_IA = float(os.getenv("TRUEPIXEL_UMBRAL_IA", "0.70"))
UMBRAL_SOSPECHOSO = float(os.getenv("TRUEPIXEL_UMBRAL_SOSPECHOSO", "0.40"))
MAX_UPLOAD_BYTES = int(os.getenv("TRUEPIXEL_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
CORS_ORIGINS = [origen.strip() for origen in os.getenv("TRUEPIXEL_CORS_ORIGINS", "*").split(",") if origen.strip()]

VEREDICTO_IA = "ALTA PROBABILIDAD DE DEEPFAKE / IA"
VEREDICTO_SOSPECHOSO = "SOSPECHOSO"
VEREDICTO_REAL = "IMAGEN REAL"


def seleccionar_dispositivo():
    """Prioriza CUDA, luego MPS (Mac) y finalmente CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def transform_inferencia():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEDIA_IMAGENET, std=STD_IMAGENET)
    ])


def transform_entrenamiento():
    """Aumentos ligeros para reducir sobreajuste (solo en el conjunto de train)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEDIA_IMAGENET, std=STD_IMAGENET)
    ])


def obtener_veredicto(probabilidad_ia):
    """Recibe una probabilidad en [0, 1] y devuelve el veredicto textual."""
    if probabilidad_ia > UMBRAL_IA:
        return VEREDICTO_IA
    if probabilidad_ia > UMBRAL_SOSPECHOSO:
        return VEREDICTO_SOSPECHOSO
    return VEREDICTO_REAL
