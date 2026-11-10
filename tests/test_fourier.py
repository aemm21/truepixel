import pytest
import torch

from fourier import obtener_espectro_frecuencia


def _imagen(batch=2, canales=3, alto=64, ancho=64):
    torch.manual_seed(0)
    return torch.rand(batch, canales, alto, ancho)


def test_forma_salida():
    assert obtener_espectro_frecuencia(_imagen()).shape == (2, 1, 64, 64)


def test_error_dimensiones():
    with pytest.raises(ValueError):
        obtener_espectro_frecuencia(torch.rand(3, 64, 64))


def test_estandarizacion_por_imagen():
    salida = obtener_espectro_frecuencia(_imagen(batch=4))
    media = salida.mean(dim=(-2, -1))
    desviacion = salida.std(dim=(-2, -1))
    assert torch.allclose(media, torch.zeros_like(media), atol=1e-5)
    assert torch.allclose(desviacion, torch.ones_like(desviacion), atol=1e-4)


def test_invarianza_escala_offset():
    x = _imagen(batch=1)
    base = obtener_espectro_frecuencia(x)
    afin = obtener_espectro_frecuencia(x * 3.5 + 0.7)
    assert torch.allclose(base, afin, atol=1e-3)


def test_determinismo():
    x = _imagen()
    assert torch.allclose(
        obtener_espectro_frecuencia(x), obtener_espectro_frecuencia(x)
    )
