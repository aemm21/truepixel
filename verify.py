import argparse
import csv
import json
import os
import shutil
import tempfile
import zipfile

import torch
from PIL import Image

from checkpoint import cargar_checkpoint
from config import (
    EXTENSIONES_VALIDAS,
    RUTA_MODELO_POR_DEFECTO,
    VEREDICTO_IA,
    VEREDICTO_REAL,
    VEREDICTO_SOSPECHOSO,
    obtener_veredicto,
    seleccionar_dispositivo,
    transform_inferencia,
)
from explain import generar_explicaciones
from model import TruePixelNet


def _cargar_modelo(ruta_modelo):
    """Carga el modelo una sola vez (para lotes) y devuelve modelo, device y metadatos."""
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
    return modelo, device, metadatos


def _analizar_imagen(modelo, device, transform, imagen, version, con_explicacion=False):
    tensor = transform(imagen).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = modelo(tensor)
        probabilidad = torch.sigmoid(logit).item()

    resultado = {
        "probabilidad_ia_porcentaje": round(probabilidad * 100, 2),
        "veredicto": obtener_veredicto(probabilidad),
        "version_modelo": version,
    }
    if con_explicacion:
        resultado["_explicaciones"] = generar_explicaciones(modelo, tensor, imagen)
    return resultado


def verificar_imagen(ruta_imagen, ruta_modelo=RUTA_MODELO_POR_DEFECTO):
    """
    Analiza una imagen y devuelve un dict con la probabilidad y el veredicto.
    Lanza RuntimeError si el modelo no está disponible.
    """
    modelo, device, metadatos = _cargar_modelo(ruta_modelo)
    transform = transform_inferencia()
    imagen = Image.open(ruta_imagen).convert('RGB')

    resultado = _analizar_imagen(
        modelo, device, transform, imagen, metadatos.get("version", "desconocida")
    )
    resultado.pop("_explicaciones", None)
    resultado["archivo"] = ruta_imagen
    return resultado


def _listar_directorio(directorio):
    rutas = []
    for raiz, _, archivos in os.walk(directorio):
        for archivo in sorted(archivos):
            if archivo.lower().endswith(EXTENSIONES_VALIDAS):
                rutas.append(os.path.join(raiz, archivo))
    return rutas


def _extraer_zip(ruta_zip):
    """Extrae solo imágenes, con nombres únicos, evitando rutas maliciosas (zip-slip)."""
    tempdir = tempfile.mkdtemp(prefix="truepixel_")
    with zipfile.ZipFile(ruta_zip) as zf:
        for indice, info in enumerate(zf.infolist()):
            if info.is_dir():
                continue
            nombre = os.path.basename(info.filename)
            if not nombre.lower().endswith(EXTENSIONES_VALIDAS):
                continue
            destino = os.path.join(tempdir, f"{indice:05d}_{nombre}")
            with zf.open(info) as origen, open(destino, "wb") as salida:
                shutil.copyfileobj(origen, salida)
    return tempdir


def _recolectar(ruta):
    """Devuelve (lista de imágenes, tempdir a limpiar o None) para archivo, carpeta o ZIP."""
    if os.path.isdir(ruta):
        return _listar_directorio(ruta), None
    if os.path.isfile(ruta) and ruta.lower().endswith(".zip"):
        tempdir = _extraer_zip(ruta)
        return _listar_directorio(tempdir), tempdir
    if os.path.isfile(ruta):
        return [ruta], None
    raise FileNotFoundError(f"No se encontró la ruta: {ruta}")


def _guardar_explicaciones(directorio, indice, nombre, explicaciones):
    os.makedirs(directorio, exist_ok=True)
    base = f"{indice:05d}_{os.path.splitext(nombre)[0]}"
    explicaciones["overlay"].save(os.path.join(directorio, f"{base}_overlay.png"))
    explicaciones["espectral"].save(os.path.join(directorio, f"{base}_espectral.png"))


def verificar_lote(ruta, ruta_modelo=RUTA_MODELO_POR_DEFECTO, explicar_dir=None, progreso=True):
    """
    Analiza un archivo, una carpeta (recursiva) o un ZIP de imágenes.
    Devuelve una lista de resultados; los fallos individuales no detienen el lote.
    """
    modelo, device, metadatos = _cargar_modelo(ruta_modelo)
    version = metadatos.get("version", "desconocida")
    transform = transform_inferencia()

    rutas, tempdir = _recolectar(ruta)
    if progreso and len(rutas) == 0:
        print("⚠️ No se encontraron imágenes válidas en la ruta indicada.")

    resultados = []
    try:
        for indice, ruta_img in enumerate(rutas):
            nombre = os.path.basename(ruta_img)
            try:
                imagen = Image.open(ruta_img).convert("RGB")
                resultado = _analizar_imagen(
                    modelo, device, transform, imagen, version,
                    con_explicacion=bool(explicar_dir)
                )
                if explicar_dir:
                    _guardar_explicaciones(
                        explicar_dir, indice, nombre, resultado.pop("_explicaciones")
                    )
                resultado["archivo"] = nombre
                resultados.append(resultado)
                if progreso:
                    print(f"[{indice + 1}/{len(rutas)}] {nombre} -> "
                          f"{resultado['probabilidad_ia_porcentaje']:.2f}% ({resultado['veredicto']})")
            except Exception as exc:
                resultados.append({"archivo": nombre, "error": str(exc)})
                if progreso:
                    print(f"[{indice + 1}/{len(rutas)}] {nombre} -> ERROR: {exc}")
    finally:
        if tempdir:
            shutil.rmtree(tempdir, ignore_errors=True)

    return resultados


def _resumen(resultados):
    resumen = {
        "total": len(resultados),
        "errores": 0,
        VEREDICTO_IA: 0,
        VEREDICTO_SOSPECHOSO: 0,
        VEREDICTO_REAL: 0,
    }
    for resultado in resultados:
        if "error" in resultado:
            resumen["errores"] += 1
        else:
            resumen[resultado["veredicto"]] += 1
    return resumen


def _escribir_csv(resultados, ruta_csv):
    campos = ["archivo", "probabilidad_ia_porcentaje", "veredicto", "version_modelo", "error"]
    with open(ruta_csv, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos, extrasaction="ignore")
        escritor.writeheader()
        escritor.writerows(resultados)


def _imprimir_reporte(resultado):
    porcentaje_ia = resultado["probabilidad_ia_porcentaje"]
    print("\n" + "=" * 40)
    print("🔍 REPORTE DE VERIFICACIÓN (TruePixel)")
    print("=" * 40)
    print(f"📁 Archivo: {resultado.get('archivo', 'desconocido')}")
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
        description="TruePixel: verifica una imagen, una carpeta o un ZIP de imágenes."
    )
    parser.add_argument(
        "ruta", nargs="?", default="prueba.jpg",
        help="Imagen, carpeta (recursiva) o archivo .zip a analizar (por defecto: prueba.jpg)"
    )
    parser.add_argument(
        "--modelo", default=RUTA_MODELO_POR_DEFECTO,
        help=f"Ruta de los pesos entrenados (por defecto: {RUTA_MODELO_POR_DEFECTO})"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Imprime los resultados en formato JSON."
    )
    parser.add_argument(
        "--csv", dest="ruta_csv", default=None,
        help="Guarda los resultados en un archivo CSV."
    )
    parser.add_argument(
        "--explicar", dest="explicar_dir", default=None,
        help="Directorio donde guardar los mapas de calor (overlay + espectro)."
    )
    args = parser.parse_args()

    try:
        resultados = verificar_lote(
            args.ruta, args.modelo,
            explicar_dir=args.explicar_dir,
            progreso=not args.json,
        )
    except Exception as e:
        print(f"❌ Error al procesar la ruta: {e}")
        return

    resumen = _resumen(resultados)

    if args.ruta_csv:
        _escribir_csv(resultados, args.ruta_csv)
        print(f"💾 Resultados guardados en '{args.ruta_csv}'")

    if args.json:
        print(json.dumps({"resumen": resumen, "resultados": resultados}, ensure_ascii=False, indent=2))
    elif len(resultados) == 1 and "error" not in resultados[0]:
        _imprimir_reporte(resultados[0])
    else:
        print("\n" + "=" * 50)
        print("RESUMEN DEL LOTE")
        print("=" * 50)
        print(f"Total: {resumen['total']} | Errores: {resumen['errores']}")
        print(f"IA: {resumen[VEREDICTO_IA]} | Sospechosas: {resumen[VEREDICTO_SOSPECHOSO]} | "
              f"Reales: {resumen[VEREDICTO_REAL]}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
