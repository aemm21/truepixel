import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api
from checkpoint import guardar_checkpoint
from config import __version__
from model import TruePixelNet


def _png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def ruta_modelo_valido(tmp_path):
    ruta = tmp_path / "modelo_valido.pth"
    guardar_checkpoint(TruePixelNet(), ruta)
    return ruta


def test_health_degradado_sin_pesos(monkeypatch, tmp_path):
    monkeypatch.setenv("TRUEPIXEL_MODEL_PATH", str(tmp_path / "no_existe.pth"))
    with TestClient(api.app) as client:
        respuesta = client.get("/health")

    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["model_loaded"] is False
    assert datos["status"] == "degraded"
    assert datos["version"] == __version__
    # No debe filtrar rutas internas del servidor.
    assert "model_path" not in datos
    assert datos["model_file"] == "no_existe.pth"
    assert datos["error"] == "modelo_no_disponible"


def test_verificar_sin_modelo_devuelve_503(monkeypatch, tmp_path):
    monkeypatch.setenv("TRUEPIXEL_MODEL_PATH", str(tmp_path / "no_existe.pth"))
    with TestClient(api.app) as client:
        respuesta = client.post(
            "/verificar/",
            files={"file": ("muestra.png", _png_bytes(), "image/png")},
        )

    assert respuesta.status_code == 503


def test_inferencia_end_to_end(monkeypatch, ruta_modelo_valido):
    monkeypatch.setenv("TRUEPIXEL_MODEL_PATH", str(ruta_modelo_valido))
    with TestClient(api.app) as client:
        salud = client.get("/health")
        respuesta = client.post(
            "/verificar/",
            files={"file": ("muestra.png", _png_bytes(), "image/png")},
        )

    assert salud.json()["model_loaded"] is True
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert 0 <= datos["probabilidad_ia_porcentaje"] <= 100
    assert datos["version_api"] == __version__
    assert datos["veredicto"] in {"IMAGEN REAL", "SOSPECHOSO", "ALTA PROBABILIDAD DE DEEPFAKE / IA"}


def test_rechaza_tipo_no_soportado(monkeypatch, ruta_modelo_valido):
    monkeypatch.setenv("TRUEPIXEL_MODEL_PATH", str(ruta_modelo_valido))
    with TestClient(api.app) as client:
        respuesta = client.post(
            "/verificar/",
            files={"file": ("nota.txt", b"hola", "text/plain")},
        )

    assert respuesta.status_code == 415


def test_rechaza_archivo_demasiado_grande(monkeypatch, ruta_modelo_valido):
    monkeypatch.setenv("TRUEPIXEL_MODEL_PATH", str(ruta_modelo_valido))
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 10)
    with TestClient(api.app) as client:
        respuesta = client.post(
            "/verificar/",
            files={"file": ("muestra.png", _png_bytes(), "image/png")},
        )

    assert respuesta.status_code == 413


def test_explicar_devuelve_png(monkeypatch, ruta_modelo_valido):
    monkeypatch.setenv("TRUEPIXEL_MODEL_PATH", str(ruta_modelo_valido))
    with TestClient(api.app) as client:
        respuesta = client.post(
            "/explicar/?tipo=overlay",
            files={"file": ("muestra.png", _png_bytes(), "image/png")},
        )

    assert respuesta.status_code == 200
    assert respuesta.headers["content-type"] == "image/png"
    assert respuesta.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_explicar_tipo_invalido(monkeypatch, ruta_modelo_valido):
    monkeypatch.setenv("TRUEPIXEL_MODEL_PATH", str(ruta_modelo_valido))
    with TestClient(api.app) as client:
        respuesta = client.post(
            "/explicar/?tipo=otro",
            files={"file": ("muestra.png", _png_bytes(), "image/png")},
        )

    assert respuesta.status_code == 400


def test_rechaza_imagen_con_demasiados_pixeles(monkeypatch, ruta_modelo_valido):
    monkeypatch.setenv("TRUEPIXEL_MODEL_PATH", str(ruta_modelo_valido))

    def _rechazar(_imagen):
        raise ValueError("demasiados píxeles")

    monkeypatch.setattr(api, "validar_tamano_imagen", _rechazar)
    with TestClient(api.app) as client:
        respuesta = client.post(
            "/verificar/",
            files={"file": ("muestra.png", _png_bytes(), "image/png")},
        )

    assert respuesta.status_code == 413
