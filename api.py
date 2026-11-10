from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
from torchvision import transforms
from model import TruePixelNet

# Inicializamos la aplicación FastAPI
app = FastAPI(
    title="TruePixel API",
    description="API Open Source para la detección de imágenes generadas por IA",
    version="1.0.0"
)

# 1. Configurar el hardware y el modelo al iniciar la API
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

modelo = TruePixelNet().to(device)

ruta_modelo = 'truepixel_v1.pth'
try:
    # weights_only=True evita la ejecución de código arbitrario al deserializar.
    estado = torch.load(ruta_modelo, map_location=device, weights_only=True)
    modelo.load_state_dict(estado)
    print(f"Modelo cargado desde {ruta_modelo}")
except FileNotFoundError as exc:
    raise RuntimeError(
        f"⚠️ No se encontró '{ruta_modelo}'. Entrena el modelo con train.py antes de iniciar la API."
    ) from exc
except Exception as exc:
    raise RuntimeError(
        f"⚠️ No se pudo cargar '{ruta_modelo}': {exc}. Vuelve a entrenar o verifica el archivo."
    ) from exc

modelo.eval()

# 2. Transformaciones estándar
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/")
def home():
    return {"mensaje": "Bienvenido a TruePixel API. Usa el endpoint POST /verificar/ para analizar imágenes."}

@app.post("/verificar/")
async def verificar_imagen(file: UploadFile = File(...)):
    """
    Recibe una imagen y devuelve el porcentaje de probabilidad de que sea generada por IA.
    """
    try:
        # Leer los bytes de la imagen subida
        contenido_imagen = await file.read()
        imagen = Image.open(io.BytesIO(contenido_imagen)).convert('RGB')
        
        # Procesar la imagen
        imagen_tensor = transform(imagen).unsqueeze(0).to(device)
        
        # Predicción: el modelo devuelve logits, aplicamos sigmoid aquí.
        with torch.no_grad():
            logit = modelo(imagen_tensor)
            probabilidad = torch.sigmoid(logit).item()
            
        porcentaje_ia = probabilidad * 100
        
        # Determinar el veredicto
        if porcentaje_ia > 70:
            veredicto = "ALTA PROBABILIDAD DE DEEPFAKE / IA"
        elif porcentaje_ia > 40:
            veredicto = "SOSPECHOSO"
        else:
            veredicto = "IMAGEN REAL"
            
        return JSONResponse(content={
            "archivo": file.filename,
            "probabilidad_ia_porcentaje": round(porcentaje_ia, 2),
            "veredicto": veredicto
        })
        
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Error al procesar la imagen: {str(e)}"})
