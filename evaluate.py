"""Evaluación de un checkpoint de TruePixel sobre carpetas etiquetadas."""

import argparse
import csv
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

from checkpoint import cargar_checkpoint
from config import (
    RUTA_MODELO_POR_DEFECTO,
    __version__,
    seleccionar_dispositivo,
    transform_inferencia,
)
from dataset import TruePixelDataset
from model import TruePixelNet
from train import calcular_metricas


def _celda_csv(valor):
    """Neutraliza fórmulas para evitar CSV injection al abrirlo en una hoja de cálculo."""
    texto = str(valor)
    if texto.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + texto
    return texto


def _especificidad(metricas):
    denominador = metricas["tn"] + metricas["fp"]
    return metricas["tn"] / denominador if denominador else 0.0


def _fila_umbral(y_true, y_prob, umbral):
    metricas = calcular_metricas(y_true, y_prob, umbral=umbral)
    especificidad = _especificidad(metricas)
    return {
        "umbral": umbral,
        "accuracy": metricas["accuracy"],
        "precision": metricas["precision"],
        "recall": metricas["recall"],
        "f1": metricas["f1"],
        "especificidad": especificidad,
        "balanced_accuracy": (metricas["recall"] + especificidad) / 2.0,
        "tp": metricas["tp"],
        "tn": metricas["tn"],
        "fp": metricas["fp"],
        "fn": metricas["fn"],
    }


def barrido_umbrales(y_true, y_prob, paso=0.01):
    """Evalúa umbrales de 0.01 a 0.99 y devuelve una fila por umbral."""
    total = int(round(1.0 / paso))
    umbrales = [round(i * paso, 4) for i in range(1, total)]
    return [_fila_umbral(y_true, y_prob, umbral) for umbral in umbrales]


def _mejor_por(barrido, clave):
    return max(barrido, key=lambda fila: (fila[clave], -abs(fila["umbral"] - 0.5)))


def _escribir_csv(predicciones, ruta_csv):
    campos = ["archivo", "etiqueta_real", "probabilidad_ia", "prediccion"]
    with open(ruta_csv, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        for prediccion in predicciones:
            fila = dict(prediccion)
            fila["archivo"] = _celda_csv(fila["archivo"])
            escritor.writerow(fila)


def evaluar_modelo(
    dir_reales,
    dir_sinteticas,
    ruta_modelo,
    batch_size=16,
    num_workers=2,
    umbral=0.5,
    device=None,
    recursivo=False,
    progreso=True,
):
    """Carga el modelo, evalúa las carpetas etiquetadas y devuelve resumen + predicciones."""
    if not 0.0 < umbral < 1.0:
        raise ValueError("El umbral debe estar entre 0 y 1 (exclusive).")
    if os.path.normcase(os.path.abspath(dir_reales)) == os.path.normcase(os.path.abspath(dir_sinteticas)):
        raise ValueError("Las carpetas de reales e IA deben ser distintas.")
    device = torch.device(device) if device is not None else seleccionar_dispositivo()
    dataset = TruePixelDataset(
        dir_reales,
        dir_sinteticas,
        transform=transform_inferencia(),
        recursivo=recursivo,
        devolver_ruta=True,
    )

    modelo = TruePixelNet(pretrained=False).to(device)
    metadatos = cargar_checkpoint(modelo, ruta_modelo, device)
    modelo.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    y_true, y_prob, rutas = [], [], []
    total_lotes = len(loader)
    if progreso:
        print(f"⏳ Evaluando {len(dataset)} imágenes...", file=sys.stderr)
    with torch.no_grad():
        for lote, (imagenes, etiquetas, rutas_lote) in enumerate(loader):
            imagenes = imagenes.to(device, non_blocking=True)
            logits = modelo(imagenes)
            y_prob.extend(torch.sigmoid(logits).view(-1).cpu().tolist())
            y_true.extend(etiquetas.view(-1).int().cpu().tolist())
            rutas.extend(rutas_lote)
            if progreso and ((lote + 1) % 20 == 0 or lote + 1 == total_lotes):
                print(
                    f"\r⏳ Evaluando... {lote + 1}/{total_lotes} lotes",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
    if progreso:
        print(file=sys.stderr)

    # El dataset sustituye imágenes corruptas/enormes por la siguiente; al
    # deduplicar por ruta evitamos contar dos veces esa imagen de reemplazo.
    vistos = set()
    predicciones = []
    y_true_limpio, y_prob_limpio = [], []
    for ruta, etiqueta, probabilidad in zip(rutas, y_true, y_prob):
        if ruta in vistos:
            continue
        vistos.add(ruta)
        y_true_limpio.append(etiqueta)
        y_prob_limpio.append(probabilidad)
        predicciones.append({
            "archivo": ruta,
            "etiqueta_real": "IA" if etiqueta == 1 else "REAL",
            "probabilidad_ia": round(probabilidad, 6),
            "prediccion": "IA" if probabilidad >= umbral else "REAL",
        })

    reales = sum(1 for etiqueta in y_true_limpio if etiqueta == 0)
    sinteticas = len(y_true_limpio) - reales

    metricas = calcular_metricas(y_true_limpio, y_prob_limpio, umbral=umbral)
    barrido = barrido_umbrales(y_true_limpio, y_prob_limpio)
    mejor_f1 = _mejor_por(barrido, "f1")
    mejor_balanceada = _mejor_por(barrido, "balanced_accuracy")

    resumen = {
        "modelo": ruta_modelo,
        "version_modelo": metadatos.get("version", "desconocida"),
        "version_api": __version__,
        "total": len(y_true_limpio),
        "reales": reales,
        "sinteticas": sinteticas,
        "umbral": umbral,
        "metricas": metricas,
        "especificidad": _especificidad(metricas),
        "mejor_f1": mejor_f1,
        "mejor_balanced_accuracy": mejor_balanceada,
        "barrido": barrido,
    }
    return {"resumen": resumen, "predicciones": predicciones}


def _imprimir_reporte(resumen):
    metricas = resumen["metricas"]
    mejor_f1 = resumen["mejor_f1"]
    mejor_balanceada = resumen["mejor_balanced_accuracy"]

    print("=" * 78)
    print("🔎 EVALUACIÓN DE TruePixel")
    print("=" * 78)
    print(f"📦 Modelo: {resumen['modelo']} (versión {resumen['version_modelo']})")
    print(f"🖼️ Imágenes evaluadas: {resumen['total']} "
          f"(reales: {resumen['reales']} | sintéticas: {resumen['sinteticas']})")
    print(f"🎚️ Umbral base: {resumen['umbral']:.2f}")
    print(f"   Accuracy: {metricas['accuracy'] * 100:.2f}% | AUC: {metricas['auc']:.4f} | "
          f"F1: {metricas['f1']:.4f}")
    print(f"   Precision: {metricas['precision']:.4f} | Recall: {metricas['recall']:.4f} | "
          f"Especificidad: {resumen['especificidad']:.4f}")
    print(f"   Matriz -> TP: {metricas['tp']} | TN: {metricas['tn']} | "
          f"FP: {metricas['fp']} | FN: {metricas['fn']}")
    print("-" * 78)
    print(f"🎯 Mejor F1: umbral {mejor_f1['umbral']:.2f} -> F1 {mejor_f1['f1']:.4f} | "
          f"Precision {mejor_f1['precision']:.4f} | Recall {mejor_f1['recall']:.4f} | "
          f"Acc {mejor_f1['accuracy'] * 100:.2f}%")
    print(f"   Matriz en mejor F1 -> TP: {mejor_f1['tp']} | TN: {mejor_f1['tn']} | "
          f"FP: {mejor_f1['fp']} | FN: {mejor_f1['fn']}")
    print(f"⚖️ Mejor balanced accuracy: umbral {mejor_balanceada['umbral']:.2f} -> "
          f"BalAcc {mejor_balanceada['balanced_accuracy']:.4f} | "
          f"Recall {mejor_balanceada['recall']:.4f} | "
          f"Especificidad {mejor_balanceada['especificidad']:.4f}")

    print("\n📊 Barrido de umbrales (cada 0.05)")
    print(f"{'Umbral':>7} | {'Acc':>7} | {'Prec':>7} | {'Recall':>7} | {'F1':>7} | {'Esp.':>7}")
    print("-" * 68)
    for fila in resumen["barrido"]:
        if round(fila["umbral"] * 100) % 5 != 0:
            continue
        print(f"{fila['umbral']:7.2f} | {fila['accuracy'] * 100:6.2f}% | "
              f"{fila['precision']:.4f} | {fila['recall']:.4f} | "
              f"{fila['f1']:.4f} | {fila['especificidad']:.4f}")

    print("\n⚠️ El umbral óptimo se calculó sobre este mismo conjunto. "
          "Para producción, ajústalo en validación y confírmalo en un test independiente.")


def main():
    parser = argparse.ArgumentParser(
        description="TruePixel: evalúa un checkpoint sobre carpetas etiquetadas de reales e IA."
    )
    parser.add_argument("--dir_reales", default="./data/reales",
                        help="Carpeta de imágenes reales (etiqueta 0).")
    parser.add_argument("--dir_sinteticas", default="./data/sinteticas",
                        help="Carpeta de imágenes IA (etiqueta 1).")
    parser.add_argument("--modelo", default=RUTA_MODELO_POR_DEFECTO,
                        help=f"Checkpoint a evaluar (por defecto: {RUTA_MODELO_POR_DEFECTO}).")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Tamaño de lote para la inferencia (por defecto: 16).")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Procesos de carga de datos (0 = sin multiproceso).")
    parser.add_argument("--umbral", type=float, default=0.5,
                        help="Umbral de decisión para el reporte base (por defecto: 0.5).")
    parser.add_argument("--recursivo", action="store_true",
                        help="Busca imágenes en subcarpetas.")
    parser.add_argument("--csv", dest="ruta_csv", default=None,
                        help="Guarda las predicciones por imagen en un CSV.")
    parser.add_argument("--json", action="store_true",
                        help="Imprime el resumen en JSON en lugar del reporte legible.")
    args = parser.parse_args()

    try:
        resultado = evaluar_modelo(
            args.dir_reales,
            args.dir_sinteticas,
            args.modelo,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            umbral=args.umbral,
            recursivo=args.recursivo,
            progreso=not args.json,
        )
    except Exception as exc:
        print(f"❌ No se pudo evaluar: {exc}", file=sys.stderr)
        raise SystemExit(1)

    if args.ruta_csv:
        _escribir_csv(resultado["predicciones"], args.ruta_csv)
        if not args.json:
            print(f"💾 Predicciones guardadas en '{args.ruta_csv}'")

    if args.json:
        print(json.dumps(resultado["resumen"], ensure_ascii=False, indent=2))
    else:
        _imprimir_reporte(resultado["resumen"])


if __name__ == "__main__":
    main()
