import argparse
import torch
from PIL import Image
from torchvision import transforms
from model import TruePixelNet

def verificar_imagen(ruta_imagen, ruta_modelo='truepixel_v1.pth'):
    # 1. Configurar el hardware (CPU o GPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 2. Inicializar la arquitectura del modelo
    modelo = TruePixelNet().to(device)
    
    # 3. Cargar los pesos entrenados (.pth)
    try:
        # weights_only=True evita la ejecución de código arbitrario al deserializar.
        estado = torch.load(ruta_modelo, map_location=device, weights_only=True)
        modelo.load_state_dict(estado)
        print(f"-> Cerebro del modelo cargado exitosamente desde {ruta_modelo}")
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"⚠️ No se encontró '{ruta_modelo}'. Ejecuta train.py para generar los pesos."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"⚠️ No se pudo cargar '{ruta_modelo}': {exc}. Vuelve a entrenar o verifica el archivo."
        ) from exc
    
    # Colocar el modelo en modo EVALUACIÓN
    modelo.eval()

    # 4. Preparar la transformación
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        # 5. Abrir la imagen y procesarla
        imagen = Image.open(ruta_imagen).convert('RGB')
        imagen_tensor = transform(imagen).unsqueeze(0).to(device)

        # 6. Hacer la predicción (el modelo devuelve logits)
        with torch.no_grad():
            logit = modelo(imagen_tensor)
            probabilidad = torch.sigmoid(logit).item()

        # 7. Mostrar los resultados en la terminal
        porcentaje_ia = probabilidad * 100
        print("\n" + "="*40)
        print("🔍 REPORTE DE VERIFICACIÓN (TruePixel)")
        print("="*40)
        print(f"📁 Archivo: {ruta_imagen}")
        print(f"🤖 Probabilidad de ser IA: {porcentaje_ia:.2f}%")
        print("-" * 40)
        
        if porcentaje_ia > 70:
            print("🚨 VEREDICTO: ALTA PROBABILIDAD DE DEEPFAKE / IA")
            print("Motivo: Se detectaron anomalías geométricas en las altas frecuencias del archivo.")
        elif porcentaje_ia > 40:
            print("⚠️ VEREDICTO: SOSPECHOSO")
            print("Motivo: La firma espectral es inconsistente. Podría ser una imagen real editada digitalmente.")
        else:
            print("✅ VEREDICTO: IMAGEN REAL (FOTOGRAFÍA)")
            print("Motivo: La imagen conserva el ruido físico natural de un sensor óptico real.")
        print("="*40 + "\n")

    except Exception as e:
        print(f"❌ Error al procesar la imagen: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TruePixel: verifica si una imagen fue generada por IA."
    )
    parser.add_argument(
        "imagen", nargs="?", default="prueba.jpg",
        help="Ruta de la imagen a analizar (por defecto: prueba.jpg)"
    )
    parser.add_argument(
        "--modelo", default="truepixel_v1.pth",
        help="Ruta de los pesos entrenados (por defecto: truepixel_v1.pth)"
    )
    args = parser.parse_args()
    verificar_imagen(args.imagen, args.modelo)
