import torch

from config import IMAGE_SIZE, __version__, obtener_veredicto, transform_entrenamiento, transform_inferencia


def test_version():
    assert __version__ == "3.0.0"


def test_veredictos():
    assert "ALTA" in obtener_veredicto(0.95)
    assert obtener_veredicto(0.5) == "SOSPECHOSO"
    assert obtener_veredicto(0.1) == "IMAGEN REAL"


def test_transform_inferencia_forma():
    from PIL import Image

    imagen = Image.new("RGB", (300, 150), (10, 20, 30))
    tensor = transform_inferencia()(imagen)
    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert tensor.dtype == torch.float32


def test_transform_entrenamiento_forma():
    from PIL import Image

    imagen = Image.new("RGB", (300, 150), (10, 20, 30))
    tensor = transform_entrenamiento()(imagen)
    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
