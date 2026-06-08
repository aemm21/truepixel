import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# Importamos los componentes modulares de nuestro proyecto
from model import TruePixelNet
from dataset import TruePixelDataset

def entrenar_modelo(args):
    # 1. Configuración del hardware (Prioriza GPU CUDA, luego MPS para Mac, y finalmente CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"\n🚀 Dispositivo de entrenamiento seleccionado: {device.type.upper()}")

    # 2. Configuración y carga del Dataset
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
    print(f"✅ Imágenes cargadas con éxito. Total: {total_imagenes}")

    # 3. División del Dataset: Entrenamiento (80%) y Validación (20%)
    # Esencial para que los colaboradores puedan evaluar la capacidad real del modelo
    val_size = int(total_imagenes * args.val_split)
    train_size = total_imagenes - val_size
    
    train_dataset, val_dataset = random_split(
        dataset_completo, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42) # Semilla fija para reproducibilidad científica
    )
    
    print(f"📊 Distribución de datos -> Entrenamiento: {train_size} | Validación: {val_size}")

    # 4. DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 5. Inicializar Modelo, Función de Pérdida y Optimizador
    modelo = TruePixelNet().to(device)
    criterio = nn.BCELoss()
    optimizador = optim.Adam(
        modelo.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay # Regularización para evitar sobreajuste
    )

    best_val_loss = float('inf')

    # 6. Bucle de Entrenamiento y Validación
    print(f"\n🎮 Iniciando entrenamiento: {args.epochs} épocas | Tasa de aprendizaje: {args.lr}")
    print("=" * 80)

    for epoca in range(args.epochs):
        # --- FASE DE ENTRENAMIENTO ---
        modelo.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for imagenes, etiquetas in train_loader:
            imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
            
            optimizador.zero_grad()
            predicciones = modelo(imagenes)
            loss = criterio(predicciones, etiquetas)
            loss.backward()
            optimizador.step()
            
            running_loss += loss.item()
            predicciones_binarias = (predicciones > 0.5).float()
            correct_train += (predicciones_binarias == etiquetas).sum().item()
            total_train += etiquetas.size(0)

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = (correct_train / total_train) * 100

        # --- FASE DE VALIDACIÓN ---
        modelo.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for imagenes, etiquetas in val_loader:
                imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
                predicciones = modelo(imagenes)
                loss = criterio(predicciones, etiquetas)
                
                val_loss += loss.item()
                predicciones_binarias = (predicciones > 0.5).float()
                correct_val += (predicciones_binarias == etiquetas).sum().item()
                total_val += etiquetas.size(0)

        epoch_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        epoch_val_acc = (correct_val / total_val) * 100 if total_val > 0 else 0

        # 7. Reporte de Métricas por Época
        print(f"Época [{epoca+1:02d}/{args.epochs}] "
              f"| Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.2f}% "
              f"| Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.2f}%")

        # Guardar el "Mejor Modelo" basado en la pérdida de validación (Early Saving)
        if epoch_val_loss < best_val_loss and epoch_val_loss > 0:
            best_val_loss = epoch_val_loss
            torch.save(modelo.state_dict(), args.output)
            print(f"💾 ¡Nuevo mejor modelo guardado en '{args.output}'!")

    print("=" * 80)
    print(f"🎉 ¡Entrenamiento completado! El mejor modelo de validación se guardó en: {args.output}\n")

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
    
    # Salida
    parser.add_argument('--output', type=str, default='truepixel_v1.pth', help='Nombre del archivo de pesos de salida')

    args = parser.parse_args()
    entrenar_modelo(args)