import torch

from checkpoint import cargar_checkpoint, guardar_checkpoint
from config import __version__
from model import TruePixelNet


def _pesos_iguales(modelo_a, modelo_b):
    pesos_a = modelo_a.state_dict()
    pesos_b = modelo_b.state_dict()
    assert set(pesos_a) == set(pesos_b)
    return all(torch.equal(pesos_a[k], pesos_b[k]) for k in pesos_a)


def test_roundtrip_con_metadatos(tmp_path):
    ruta = tmp_path / "modelo.pth"
    original = TruePixelNet()
    guardar_checkpoint(original, ruta, {"epoch": 3, "val_loss": 0.5})

    cargado = TruePixelNet()
    metadatos = cargar_checkpoint(cargado, ruta, torch.device("cpu"))

    assert _pesos_iguales(original, cargado)
    assert metadatos["version"] == __version__
    assert metadatos["arquitectura"] == "TruePixelNet"
    assert metadatos["epoch"] == 3


def test_compatibilidad_state_dict_plano(tmp_path):
    ruta = tmp_path / "legacy.pth"
    original = TruePixelNet()
    torch.save(original.state_dict(), ruta)

    cargado = TruePixelNet()
    metadatos = cargar_checkpoint(cargado, ruta, torch.device("cpu"))

    assert _pesos_iguales(original, cargado)
    assert "state_dict" in metadatos
