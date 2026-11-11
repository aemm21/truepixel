"""Explicabilidad: Grad-CAM sobre la rama espacial y mapa del espectro de Fourier."""

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToPILImage

from fourier import obtener_espectro_frecuencia

_A_PIL = ToPILImage()


def _normalizar(mapa):
    minimo = mapa.min()
    maximo = mapa.max()
    return (mapa - minimo) / (maximo - minimo).clamp_min(1e-8)


def _colorear(mapa):
    """Aplica un colormap tipo heatmap (negro → rojo → amarillo → blanco)."""
    x = mapa.clamp(0, 1)
    rojo = torch.clamp(1.5 * x, 0, 1)
    verde = torch.clamp(1.5 * x - 0.5, 0, 1)
    azul = torch.clamp(1.5 * x - 1.0, 0, 1)
    rgb = torch.stack([rojo, verde, azul], dim=0).cpu()
    return _A_PIL(rgb)


def _ultima_conv(modulo):
    for capa in reversed(list(modulo.modules())):
        if isinstance(capa, torch.nn.Conv2d):
            return capa
    raise ValueError("No se encontró ninguna capa Conv2d en la rama espacial.")


def grad_cam(modelo, imagen_tensor, capa_objetivo=None):
    """
    Calcula Grad-CAM de la probabilidad de IA sobre la rama espacial.
    Devuelve un tensor (H, W) normalizado en [0, 1].
    """
    capa_objetivo = capa_objetivo or _ultima_conv(modelo.rama_espacial)
    activaciones = {}

    def _hook(_modulo, _entrada, salida):
        activaciones["valor"] = salida

    handle = capa_objetivo.register_forward_hook(_hook)
    try:
        logit = modelo(imagen_tensor)
        salida = activaciones.get("valor")
        if salida is None:
            raise RuntimeError("No se capturaron activaciones para Grad-CAM.")
        gradientes = torch.autograd.grad(logit.sum(), salida)[0]
    finally:
        handle.remove()

    pesos = gradientes.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((pesos * salida).sum(dim=1, keepdim=True))
    cam = cam[0, 0]

    return _normalizar(cam.detach())


def mapa_espectral(imagen_tensor):
    """Devuelve el mapa espectral normalizado (H, W) que consume la rama de Fourier."""
    espectro = obtener_espectro_frecuencia(imagen_tensor.detach())
    return _normalizar(espectro[0, 0])


def superponer(imagen_pil, cam, alpha=0.45):
    """Mezcla el mapa de calor con la imagen original."""
    cam = F.interpolate(
        cam[None, None],
        size=(imagen_pil.height, imagen_pil.width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    calor = _colorear(_normalizar(cam))
    return Image.blend(imagen_pil.convert("RGB"), calor, alpha=alpha)


def generar_explicaciones(modelo, imagen_tensor, imagen_pil):
    """
    Genera las dos vistas explicativas:
    - overlay: Grad-CAM superpuesto a la imagen original.
    - espectral: espectro de Fourier coloreado.
    """
    cam = grad_cam(modelo, imagen_tensor)
    espectro = mapa_espectral(imagen_tensor)
    return {
        "overlay": superponer(imagen_pil, cam),
        "espectral": _colorear(espectro),
    }
