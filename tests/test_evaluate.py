import csv

import torch
from PIL import Image

from checkpoint import guardar_checkpoint
from evaluate import _escribir_csv, barrido_umbrales, evaluar_modelo
from model import TruePixelNet


def _crear_imagen(ruta, color):
    Image.new("RGB", (32, 32), color).save(ruta)


def test_barrido_umbrales_encuentra_f1_perfecto():
    barrido = barrido_umbrales([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])

    mejor = max(barrido, key=lambda fila: fila["f1"])

    assert len(barrido) == 99
    assert mejor["f1"] == 1.0
    assert 0.2 < mejor["umbral"] <= 0.8


def test_csv_neutraliza_formulas(tmp_path):
    ruta = tmp_path / "predicciones.csv"
    _escribir_csv(
        [{
            "archivo": "=cmd|calc",
            "etiqueta_real": "REAL",
            "probabilidad_ia": 0.1,
            "prediccion": "REAL",
        }],
        ruta,
    )

    with open(ruta, newline="", encoding="utf-8") as archivo:
        filas = list(csv.DictReader(archivo))

    assert filas[0]["archivo"].startswith("'=")


def test_evaluar_modelo_end_to_end(tmp_path):
    reales = tmp_path / "reales"
    sinteticas = tmp_path / "sinteticas"
    reales.mkdir()
    sinteticas.mkdir()
    _crear_imagen(reales / "real.jpg", (120, 80, 40))
    _crear_imagen(sinteticas / "ia.jpg", (10, 20, 30))

    ruta_modelo = tmp_path / "modelo.pth"
    guardar_checkpoint(TruePixelNet(pretrained=False), ruta_modelo)

    resultado = evaluar_modelo(
        str(reales),
        str(sinteticas),
        str(ruta_modelo),
        batch_size=2,
        num_workers=0,
        device=torch.device("cpu"),
        progreso=False,
    )
    resumen = resultado["resumen"]

    assert resumen["total"] == 2
    assert resumen["reales"] == 1
    assert resumen["sinteticas"] == 1
    assert len(resumen["barrido"]) == 99
    assert "auc" in resumen["metricas"]
    assert len(resultado["predicciones"]) == 2
