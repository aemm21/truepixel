import pytest
from PIL import Image

from verify import verificar_imagen


def test_falla_sin_pesos_entrenados(tmp_path):
    imagen = tmp_path / "muestra.jpg"
    Image.new("RGB", (32, 32), (100, 100, 100)).save(imagen)

    with pytest.raises(RuntimeError):
        verificar_imagen(str(imagen), str(tmp_path / "no_existe.pth"))
