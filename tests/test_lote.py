import csv
import zipfile

import pytest
from PIL import Image

import verify
from checkpoint import guardar_checkpoint
from model import TruePixelNet
from verify import _escribir_csv, verificar_lote


@pytest.fixture
def ruta_modelo(tmp_path):
    ruta = tmp_path / "modelo.pth"
    guardar_checkpoint(TruePixelNet(pretrained=False), ruta)
    return ruta


def _crear_imagen(ruta, size=(32, 32)):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (10, 20, 30)).save(ruta)


def test_lote_carpeta_recursiva(tmp_path, ruta_modelo):
    carpeta = tmp_path / "imgs"
    _crear_imagen(carpeta / "a.jpg")
    _crear_imagen(carpeta / "sub" / "b.png")
    (carpeta / "nota.txt").write_text("ignorar")

    resultados = verificar_lote(str(carpeta), str(ruta_modelo), progreso=False)

    assert len(resultados) == 2
    assert all("error" not in r for r in resultados)
    assert all(0 <= r["probabilidad_ia_porcentaje"] <= 100 for r in resultados)


def test_lote_zip(tmp_path, ruta_modelo):
    origen = tmp_path / "origen"
    _crear_imagen(origen / "a.jpg")
    _crear_imagen(origen / "b.png")

    zip_path = tmp_path / "imgs.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(origen / "a.jpg", "a.jpg")
        zf.write(origen / "b.png", "sub/b.png")
        zf.writestr("nota.txt", "ignorar")

    resultados = verificar_lote(str(zip_path), str(ruta_modelo), progreso=False)

    assert len(resultados) == 2
    assert all("error" not in r for r in resultados)


def test_zip_no_permite_salir_del_directorio_temporal(tmp_path, ruta_modelo):
    origen = tmp_path / "x.jpg"
    _crear_imagen(origen)

    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(origen, "../../evil.jpg")

    resultados = verificar_lote(str(zip_path), str(ruta_modelo), progreso=False)

    assert len(resultados) == 1
    assert "error" not in resultados[0]
    assert not (tmp_path.parent / "evil.jpg").exists()


def test_guarda_explicaciones(tmp_path, ruta_modelo):
    carpeta = tmp_path / "imgs"
    _crear_imagen(carpeta / "a.jpg")
    salida = tmp_path / "explicaciones"

    resultados = verificar_lote(
        str(carpeta), str(ruta_modelo), explicar_dir=str(salida), progreso=False
    )

    assert len(resultados) == 1
    nombres = [p.name for p in salida.iterdir()]
    assert any(n.endswith("_overlay.png") for n in nombres)
    assert any(n.endswith("_espectral.png") for n in nombres)


def test_escribir_csv(tmp_path):
    resultados = [
        {"archivo": "a.jpg", "probabilidad_ia_porcentaje": 10.5, "veredicto": "IMAGEN REAL",
         "version_modelo": "3.4.0"},
        {"archivo": "b.jpg", "error": "corrupta"},
    ]
    ruta = tmp_path / "salida.csv"
    _escribir_csv(resultados, ruta)

    with open(ruta, newline="", encoding="utf-8") as archivo:
        filas = list(csv.DictReader(archivo))

    assert len(filas) == 2
    assert filas[0]["archivo"] == "a.jpg"
    assert filas[1]["error"] == "corrupta"


def test_csv_neutraliza_formulas(tmp_path):
    resultados = [
        {"archivo": "=cmd|calc", "probabilidad_ia_porcentaje": 1.0, "veredicto": "IMAGEN REAL",
         "version_modelo": "3.4.0"},
    ]
    ruta = tmp_path / "salida.csv"
    _escribir_csv(resultados, ruta)

    with open(ruta, newline="", encoding="utf-8") as archivo:
        filas = list(csv.DictReader(archivo))

    assert filas[0]["archivo"].startswith("'=")


def test_zip_limita_numero_de_archivos(tmp_path, ruta_modelo, monkeypatch):
    origen = tmp_path / "origen"
    _crear_imagen(origen / "a.jpg")
    _crear_imagen(origen / "b.png")

    zip_path = tmp_path / "muchos.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(origen / "a.jpg", "a.jpg")
        zf.write(origen / "b.png", "b.png")

    monkeypatch.setattr(verify, "MAX_ZIP_FILES", 1)

    with pytest.raises(ValueError):
        verificar_lote(str(zip_path), str(ruta_modelo), progreso=False)
