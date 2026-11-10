import torch
import torch.nn as nn

from model import TruePixelNet


def test_salida_es_logit_sin_sigmoid():
    modelo = TruePixelNet().eval()
    with torch.no_grad():
        salida = modelo(torch.randn(2, 3, 224, 224))
    assert salida.shape == (2, 1)
    assert isinstance(modelo.clasificador[-1], nn.Linear)


def test_tamano_entrada_flexible():
    modelo = TruePixelNet().eval()
    with torch.no_grad():
        for lado in (96, 128, 160, 224):
            salida = modelo(torch.randn(1, 3, lado, lado))
            assert salida.shape == (1, 1)


def test_modelo_ligero():
    total = sum(p.numel() for p in TruePixelNet().parameters())
    assert total < 1_000_000


def test_gradientes_fluyen():
    modelo = TruePixelNet()
    salida = modelo(torch.randn(2, 3, 96, 96))
    salida.sum().backward()
    assert all(p.grad is not None for p in modelo.parameters() if p.requires_grad)
