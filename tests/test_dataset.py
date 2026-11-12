import pytest
from PIL import Image

from dataset import TruePixelDataset


def _crear_imagen(ruta, size=(32, 32), color=(120, 80, 40)):
    Image.new("RGB", size, color).save(ruta)


def _carpetas(tmp_path):
    reales = tmp_path / "reales"
    sinteticas = tmp_path / "sinteticas"
    reales.mkdir()
    sinteticas.mkdir()
    return reales, sinteticas


def test_longitud_y_etiquetas(tmp_path):
    reales, sinteticas = _carpetas(tmp_path)
    _crear_imagen(reales / "a.jpg")
    _crear_imagen(reales / "b.png")
    _crear_imagen(sinteticas / "c.jpg")

    ds = TruePixelDataset(str(reales), str(sinteticas))

    assert len(ds) == 3
    assert sorted(ds.etiquetas) == [0.0, 0.0, 1.0]


def test_ignora_archivos_invalidos(tmp_path):
    reales, sinteticas = _carpetas(tmp_path)
    _crear_imagen(reales / "ok.jpg")
    _crear_imagen(sinteticas / "ok.jpg")
    (reales / "nota.txt").write_text("hola")

    ds = TruePixelDataset(str(reales), str(sinteticas))

    assert len(ds) == 2


def test_forma_del_tensor(tmp_path):
    reales, sinteticas = _carpetas(tmp_path)
    _crear_imagen(reales / "a.jpg")
    _crear_imagen(sinteticas / "b.jpg")

    ds = TruePixelDataset(str(reales), str(sinteticas))
    imagen, etiqueta = ds[0]

    assert imagen.shape == (3, 224, 224)
    assert etiqueta.shape == (1,)


def test_devuelve_ruta_si_se_solicita(tmp_path):
    reales, sinteticas = _carpetas(tmp_path)
    _crear_imagen(reales / "a.jpg")
    _crear_imagen(sinteticas / "b.jpg")

    ds = TruePixelDataset(str(reales), str(sinteticas), devolver_ruta=True)
    imagen, etiqueta, ruta = ds[0]

    assert imagen.shape == (3, 224, 224)
    assert etiqueta.shape == (1,)
    assert ruta == ds.rutas_imagenes[0]


def test_carpetas_inexistentes(tmp_path):
    with pytest.raises(FileNotFoundError):
        TruePixelDataset(str(tmp_path / "no_existe"), str(tmp_path))


def test_carpetas_vacias(tmp_path):
    reales, sinteticas = _carpetas(tmp_path)
    with pytest.raises(ValueError):
        TruePixelDataset(str(reales), str(sinteticas))
