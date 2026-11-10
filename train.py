import os
import math
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# Importamos los componentes modulares de nuestro proyecto
from model import TruePixelNet
from dataset import TruePixelDataset

MEDIA_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]


def _transform_entrenamiento():
    """Aumentos ligeros para reducir sobreajuste (solo en el conjunto de train)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEDIA_IMAGENET, std=STD_IMAGENET)
    ])


def dividir_estratificado(etiquetas, val_split, test_split, seed=42):
    """
    Divide los índices manteniendo la proporción de clases en cada subconjunto.
    Un split aleatorio simple puede dejar una clase sin representación en validación.
    """
    rng = random.Random(seed)
    indices_por_clase = {}
    for i, etiqueta in enumerate(etiquetas):
        indices_por_clase.setdefault(int(etiqueta), []).append(i)

    train_idx, val_idx, test_idx = [], [], []
    for clase in sorted(indices_por_clase):
        idxs = indices_por_clase[clase]
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = int(n * test_split)
        n_val = int(n * val_split)
        test_idx.extend(idxs[:n_test])
        val_idx.extend(idxs[n_test:n_test + n_val])
        train_idx.extend(idxs[n_test + n_val:])

    rng.shuffle(train_idx)
    return train_idx, val_idx, test_idx


def calcular_auc(y_true, y_prob):
    """AUC-ROC por rangos (Mann-Whitney U), sin dependencias externas."""
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')

    orden = sorted(range(len(y_prob)), key=lambda i: y_prob[i])
    rangos = [0.0] * len(y_prob)
    i = 0
    while i < len(orden):
        j = i
        while j + 1 < len(orden) and y_prob[orden[j + 1]] == y_prob[orden[i]]:
            j += 1
        rango_promedio = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            rangos[orden[k]] = rango_promedio
        i = j + 1

    suma_rangos_pos = sum(r for r, y in zip(rangos, y_true) if y == 1)
    return (suma_rangos_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def calcular_metricas(y_true, y_prob, umbral=0.5):
    y_pred = [1 if p >= umbral else 0 for p in y_prob]
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    exactitud = (tp + tn) / len(y_true) if y_true else 0.0

    return {
        "accuracy": exactitud,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": calcular_auc(y_true, y_prob),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def evaluar(modelo, loader, criterio, device):
    modelo.eval()
    perdida_total = 0.0
    y_true, y_prob = [], []
    with torch.no_grad():
        for imagenes, etiquetas in loader:
            imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
            logits = modelo(imagenes)
            perdida_total += criterio(logits, etiquetas).item()
            y_prob.extend(torch.sigmoid(logits).view(-1).cpu().tolist())
            y_true.extend(etiquetas.view(-1).int().cpu().tolist())

    perdida = perdida_total / len(loader) if len(loader) > 0 else 0.0
    return perdida, calcular_metricas(y_true, y_prob)


def _resumen(metricas):
    return (f"Acc: {metricas['accuracy'] * 100:.2f}% - "
            f"AUC: {metricas['auc']:.4f} - F1: {metricas['f1']:.4f} - "
            f"Precision: {metricas['precision']:.4f} - Recall: {metricas['recall']:.4f}")


def entrenar_modelo(args):
    # 0. Semillas globales para reproducibilidad
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not (0 <= args.val_split < 1) or not (0 <= args.test_split < 1):
        raise SystemExit("❌ val_split y test_split deben estar en el rango [0, 1).")
    if args.val_split + args.test_split >= 1:
        raise SystemExit("❌ val_split + test_split debe ser menor que 1.")

    # 1. Configuración del hardware (Prioriza GPU CUDA, luego MPS para Mac, y finalmente CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"\n🚀 Dispositivo de entrenamiento seleccionado: {device.type.upper()}")

    # 2. Configuración y carga del Dataset (sin aumentos, para val/test)
    print("📦 Cargando conjunto de datos...")
    try:
        dataset_completo = TruePixelDataset(
            directorio_real=args.dir_reales,
            directorio_ia=args.dir_sinteticas
        )
    except Exception as e:
        print(f"❌ Error crítico al inicializar el dataset: {e}")
        return

    total_imagenes = len(dataset_completo)
    etiquetas = dataset_completo.etiquetas
    print(f"✅ Imágenes cargadas con éxito. Total: {total_imagenes}")

    # 3. División estratificada: Train / Validación / Test
    train_idx, val_idx, test_idx = dividir_estratificado(
        etiquetas, args.val_split, args.test_split, seed=args.seed
    )

    if len(train_idx) == 0:
        print("❌ No quedaron imágenes para entrenamiento. Ajusta val_split/test_split.")
        return
    if len(val_idx) == 0:
        print("⚠️ No hay imágenes de validación (val_split muy bajo). No se elegirá el mejor modelo por validación.")
    if len(test_idx) == 0:
        print("⚠️ No hay imágenes de test (test_split muy bajo). No se reportarán métricas finales de test.")

    print(f"📊 Distribución de datos -> Entrenamiento: {len(train_idx)} | "
          f"Validación: {len(val_idx)} | Test: {len(test_idx)}")

    # Dataset de entrenamiento con aumentos (mismo orden de archivos gracias al sorted del dataset)
    dataset_train = TruePixelDataset(
        directorio_real=args.dir_reales,
        directorio_ia=args.dir_sinteticas,
        transform=_transform_entrenamiento()
    )
    if dataset_train.etiquetas != etiquetas:
        raise RuntimeError("Las etiquetas cambiaron entre cargas del dataset. Revisa los directorios de datos.")

    train_dataset = Subset(dataset_train, train_idx)
    val_dataset = Subset(dataset_completo, val_idx)
    test_dataset = Subset(dataset_completo, test_idx)

    # 4. DataLoaders
    # drop_last solo si hay más de un lote; si no, el loader quedaría vacío
    # y provocaría una división entre cero en el cálculo de la pérdida media.
    drop_last = len(train_idx) > args.batch_size
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=drop_last, num_workers=args.num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory
    )

    if len(train_loader) == 0:
        print("❌ No hay suficientes imágenes para formar un lote de entrenamiento. Agrega más datos.")
        return

    # 5. Inicializar Modelo, Función de Pérdida y Optimizador
    modelo = TruePixelNet().to(device)

    # BCEWithLogitsLoss combina Sigmoid + BCE de forma numéricamente estable.
    # pos_weight compensa el desbalance de clases (más reales que sintéticas o viceversa).
    n_pos = sum(1 for i in train_idx if etiquetas[i] == 1)
    n_neg = len(train_idx) - n_pos
    if n_pos > 0 and n_neg > 0:
        pos_weight = torch.tensor([n_neg / n_pos], device=device)
        criterio = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"⚖️ Desbalance -> reales: {n_neg} | sintéticas: {n_pos} (pos_weight={n_neg / n_pos:.3f})")
    else:
        criterio = nn.BCEWithLogitsLoss()

    optimizador = optim.Adam(
        modelo.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay # Regularización para evitar sobreajuste
    )

    best_val_loss = float('inf')
    mejor_guardado = False

    # 6. Bucle de Entrenamiento y Validación
    print(f"\n🎮 Iniciando entrenamiento: {args.epochs} épocas | Tasa de aprendizaje: {args.lr}")
    print("=" * 100)

    for epoca in range(args.epochs):
        # --- FASE DE ENTRENAMIENTO ---
        modelo.train()
        running_loss = 0.0
        y_true, y_prob = [], []

        for imagenes, etiquetas_batch in train_loader:
            imagenes, etiquetas_batch = imagenes.to(device), etiquetas_batch.to(device)

            optimizador.zero_grad()
            logits = modelo(imagenes)
            loss = criterio(logits, etiquetas_batch)
            loss.backward()
            optimizador.step()

            running_loss += loss.item()
            y_prob.extend(torch.sigmoid(logits).detach().view(-1).cpu().tolist())
            y_true.extend(etiquetas_batch.view(-1).int().cpu().tolist())

        epoch_train_loss = running_loss / len(train_loader)
        metricas_train = calcular_metricas(y_true, y_prob)

        # --- FASE DE VALIDACIÓN ---
        epoch_val_loss, metricas_val = evaluar(modelo, val_loader, criterio, device)

        # 7. Reporte de Métricas por Época
        print(f"Época [{epoca+1:02d}/{args.epochs}] Train Loss: {epoch_train_loss:.4f} | {_resumen(metricas_train)}")
        if len(val_idx) > 0:
            print(f"{' ' * 11}Val Loss: {epoch_val_loss:.4f} | {_resumen(metricas_val)}")

        # Guardar el "Mejor Modelo" basado en la pérdida de validación (Early Saving)
        if epoch_val_loss < best_val_loss and epoch_val_loss > 0:
            best_val_loss = epoch_val_loss
            mejor_guardado = True
            torch.save(modelo.state_dict(), args.output)
            print(f"💾 ¡Nuevo mejor modelo guardado en '{args.output}'!")

    # Si no hubo conjunto de validación (val_split=0), guardamos el modelo final
    # para que verify.py y api.py tengan pesos que cargar.
    if not mejor_guardado:
        torch.save(modelo.state_dict(), args.output)
        print(f"💾 Modelo final guardado en '{args.output}' (sin validación).")

    print("=" * 100)

    # 8. Evaluación final sobre el conjunto de test (con los mejores pesos)
    if len(test_idx) > 0:
        try:
            modelo.load_state_dict(torch.load(args.output, map_location=device, weights_only=True))
            test_loss, metricas_test = evaluar(modelo, test_loader, criterio, device)
            print(f"🧪 TEST | Loss: {test_loss:.4f} | {_resumen(metricas_test)}")
            print(f"   Matriz de confusión -> TP: {metricas_test['tp']} | TN: {metricas_test['tn']} | "
                  f"FP: {metricas_test['fp']} | FN: {metricas_test['fn']}")
        except Exception as e:
            print(f"⚠️ No se pudo evaluar el conjunto de test: {e}")

    print(f"🎉 ¡Entrenamiento completado! El mejor modelo se guardó en: {args.output}\n")


if __name__ == '__main__':
    # Analizador de argumentos para facilitar las contribuciones desde la terminal
    parser = argparse.ArgumentParser(description="TruePixel: Entrenamiento de Redes Convolucionales para Verificación de Imágenes")

    # Directorios de Datos
    parser.add_argument('--dir_reales', type=str, default='./data/reales', help='Ruta a la carpeta de imágenes reales')
    parser.add_argument('--dir_sinteticas', type=str, default='./data/sinteticas', help='Ruta a la carpeta de imágenes generadas por IA')

    # Hiperparámetros de Entrenamiento
    parser.add_argument('--epochs', type=int, default=30, help='Número de épocas para entrenar')
    parser.add_argument('--batch_size', type=int, default=16, help='Tamaño del lote (batch size)')
    parser.add_argument('--lr', type=float, default=0.0003, help='Tasa de aprendizaje (learning rate)')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Coeficiente de regularización L2')
    parser.add_argument('--val_split', type=float, default=0.20, help='Porcentaje de datos destinados a validación (ej. 0.20 = 20%)')
    parser.add_argument('--test_split', type=float, default=0.15, help='Porcentaje de datos destinados a test final (ej. 0.15 = 15%)')
    parser.add_argument('--num_workers', type=int, default=0, help='Procesos de carga de datos (0 = sin multiproceso)')
    parser.add_argument('--seed', type=int, default=42, help='Semilla para reproducibilidad')

    # Salida
    parser.add_argument('--output', type=str, default='truepixel_v1.pth', help='Nombre del archivo de pesos de salida')

    args = parser.parse_args()
    entrenar_modelo(args)
