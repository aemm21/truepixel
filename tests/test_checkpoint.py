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
    original = TruePixelNet(pretrained=False)
    guardar_checkpoint(original, ruta, {"epoch": 3, "val_loss": 0.5})

    cargado = TruePixelNet(pretrained=False)
    metadatos = cargar_checkpoint(cargado, ruta, torch.device("cpu"))

    assert _pesos_iguales(original, cargado)
    assert metadatos["version"] == __version__
    assert metadatos["arquitectura"] == "TruePixelNet"
    assert metadatos["epoch"] == 3


def test_compatibilidad_state_dict_plano(tmp_path):
    ruta = tmp_path / "legacy.pth"
    original = TruePixelNet(pretrained=False)
    torch.save(original.state_dict(), ruta)

    cargado = TruePixelNet(pretrained=False)
    metadatos = cargar_checkpoint(cargado, ruta, torch.device("cpu"))

    assert _pesos_iguales(original, cargado)
    assert "state_dict" in metadatos


def test_roundtrip_con_estado_del_optimizador(tmp_path):
    ruta = tmp_path / "modelo_optimizador.pth"
    original = TruePixelNet(pretrained=False)
    optimizador = torch.optim.Adam(original.parameters(), lr=1e-3)
    optimizador.zero_grad()
    original(torch.randn(2, 3, 96, 96)).sum().backward()
    optimizador.step()
    guardar_checkpoint(original, ruta, {"epoch": 5}, optimizador=optimizador)

    cargado = TruePixelNet(pretrained=False)
    optimizador_cargado = torch.optim.Adam(cargado.parameters(), lr=1e-3)
    metadatos = cargar_checkpoint(cargado, ruta, torch.device("cpu"))
    optimizador_cargado.load_state_dict(metadatos["optimizer_state_dict"])

    assert metadatos["epoch"] == 5
    assert metadatos["optimizer_state_dict"]["state"]
    assert _pesos_iguales(original, cargado)
