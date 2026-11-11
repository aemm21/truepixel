import argparse
import json

import torch
from PIL import Image

from checkpoint import cargar_checkpoint
from config import (
    RUTA_MODELO_POR_DEFECTO,
    VEREDICTO_IA,
    VEREDICTO_SOSPECHOSO,
    obtener_veredicto,
    seleccionar_dispositivo,
    transform_inferencia,
)
from model import TruePixelNet


def verificar_imagen(ruta_imagen, ruta_modelo=RUTA_MODELO_POR_DEFECTO):
    """
    Analiza una imagen y devuelve un dict con la probabilidad y el veredicto.
    Lanza RuntimeError si el modelo no está disponible.
    """
    device = seleccionar_dispositivo()

    modelo = TruePixelNet().to(device)
    try:
        metadatos = cargar_checkpoint(modelo, ruta_modelo, device)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"⚠️ No se encontró '{ruta_modelo}'. Ejecuta train.py para generar los pesos."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"⚠️ No se pudo cargar '{ruta_modelo}': {exc}. Vuelve a entrenar o verifica el archivo."
        ) from exc

    modelo.eval()
    version = metadatos.get("version", "desconocida")

    transform = transform_inferencia()
    imagen = Image.open(ruta_imagen).convert('RGB')
    imagen_tensor = transform(imagen).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = modelo(imagen_tensor)
        probabilidad = torch.sigmoid(logit).item()

    return {
        "archivo": ruta_imagen,
        "probabilidad_ia_porcentaje": round(probabilidad * 100, 2),
        "veredicto": obtener_veredicto(probabilidad),
        "version_modelo": version,
    }


def _imprimir_reporte(resultado):
    porcentaje_ia = resultado["probabilidad_ia_porcentaje"]
    print("\n" + "=" * 40)
    print("🔍 REPORTE DE VERIFICACIÓN (TruePixel)")
    print("=" * 40)
    print(f"📁 Archivo: {resultado['archivo']}")
    print(f"🤖 Probabilidad de ser IA: {porcentaje_ia:.2f}%")
    print(f"🏷️ Versión del modelo: {resultado['version_modelo']}")
    print("-" * 40)

    if resultado["veredicto"] == VEREDICTO_IA:
        print("🚨 VEREDICTO: ALTA PROBABILIDAD DE DEEPFAKE / IA")
        print("Motivo: Se detectaron anomalías geométricas en las altas frecuencias del archivo.")
    elif resultado["veredicto"] == VEREDICTO_SOSPECHOSO:
        print("⚠️ VEREDICTO: SOSPECHOSO")
        print("Motivo: La firma espectral es inconsistente. Podría ser una imagen real editada digitalmente.")
    else:
        print("✅ VEREDICTO: IMAGEN REAL (FOTOGRAFÍA)")
        print("Motivo: La imagen conserva el ruido físico natural de un sensor óptico real.")
    print("=" * 40 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="TruePixel: verifica si una imagen fue generada por IA."
    )
    parser.add_argument(
        "imagen", nargs="?", default="prueba.jpg",
        help="Ruta de la imagen a analizar (por defecto: prueba.jpg)"
    )
    parser.add_argument(
        "--modelo", default=RUTA_MODELO_POR_DEFECTO,
        help=f"Ruta de los pesos entrenados (por defecto: {RUTA_MODELO_POR_DEFECTO})"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Imprime el resultado en formato JSON en lugar del reporte legible."
    )
    args = parser.parse_args()

    try:
        resultado = verificar_imagen(args.imagen, args.modelo)
    except Exception as e:
        print(f"❌ Error al procesar la imagen: {e}")
    else:
        if args.json:
            print(json.dumps(resultado, ensure_ascii=False, indent=2))
        else:
            _imprimir_reporte(resultado)


if __name__ == "__main__":
    main()
