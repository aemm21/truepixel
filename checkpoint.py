"""Guardado/carga de checkpoints con metadatos y compatibilidad con pesos antiguos."""

import torch

from config import __version__


def guardar_checkpoint(modelo, ruta, metadatos=None):
    """
    Guarda pesos + metadatos en un único archivo.
    El formato es un dict con 'state_dict', por lo que sigue siendo compatible
    con herramientas estándar de PyTorch.
    """
    payload = {
        "version": __version__,
        "arquitectura": modelo.__class__.__name__,
        "state_dict": modelo.state_dict(),
    }
    if metadatos:
        payload.update(metadatos)
    torch.save(payload, ruta)


def cargar_checkpoint(modelo, ruta, device=None):
    """
    Carga un checkpoint nuevo (dict con 'state_dict') o uno antiguo
    (state_dict plano). Devuelve los metadatos disponibles.
    """
    device = device or torch.device("cpu")
    checkpoint = torch.load(ruta, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        estado = checkpoint["state_dict"]
        metadatos = checkpoint
    else:
        # Compatibilidad con versiones anteriores que guardaban solo state_dict.
        estado = checkpoint
        metadatos = {"state_dict": estado}

    modelo.load_state_dict(estado)
    return metadatos
