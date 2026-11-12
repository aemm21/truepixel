import torch
from PIL import Image

from explain import generar_explicaciones, grad_cam, mapa_espectral, superponer
from model import TruePixelNet


def _entrada():
    torch.manual_seed(0)
    imagen = Image.new("RGB", (128, 96), (120, 80, 40))
    tensor = torch.randn(1, 3, 96, 96) * 0.1
    return imagen, tensor


def test_grad_cam_forma_y_rango():
    modelo = TruePixelNet(pretrained=False).eval()
    _, tensor = _entrada()
    cam = grad_cam(modelo, tensor)
    assert cam.dim() == 2
    assert torch.all(cam >= 0) and torch.all(cam <= 1)


def test_mapa_espectral_forma_y_rango():
    _, tensor = _entrada()
    espectro = mapa_espectral(tensor)
    assert espectro.shape == (96, 96)
    assert torch.all(espectro >= 0) and torch.all(espectro <= 1)


def test_superponer_respeta_tamano_original():
    modelo = TruePixelNet(pretrained=False).eval()
    imagen, tensor = _entrada()
    cam = grad_cam(modelo, tensor)
    overlay = superponer(imagen, cam)
    assert overlay.size == imagen.size
    assert overlay.mode == "RGB"


def test_generar_explicaciones_devuelve_dos_vistas():
    modelo = TruePixelNet(pretrained=False).eval()
    imagen, tensor = _entrada()
    explicaciones = generar_explicaciones(modelo, tensor, imagen)
    assert set(explicaciones) == {"overlay", "espectral"}
    assert explicaciones["overlay"].size == imagen.size
    assert explicaciones["espectral"].size == (96, 96)
