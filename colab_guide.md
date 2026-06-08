☁️ Guía de Entrenamiento Externo en Google Colab (Gratis con GPU)

Entrenar modelos de visión artificial en una CPU local con pocas imágenes suele causar que el modelo colapse y prediga siempre "Real". Google Colab te ofrece una GPU T4 gratuita que resolverá este problema al permitirte entrenar con conjuntos de datos más grandes en segundos.

Sigue estos pasos para entrenar tu modelo TruePixel en la nube.

📦 Paso 1: Preparar tu Dataset en Local

Para entrenar con éxito, necesitas más imágenes. Intenta conseguir al menos 50 imágenes reales y 50 imágenes de IA.

Asegúrate de tener tu carpeta data estructurada así en tu computadora:

data/
├── reales/       (fotos de cámara, rostros reales, etc.)
└── sinteticas/   (imágenes de Midjourney, DALL-E, etc.)


Comprime la carpeta data en un archivo ZIP llamado data.zip.

🚀 Paso 2: Crear tu Entorno en Google Colab

Ve a Google Colab e inicia sesión con tu cuenta de Google.

Haz clic en "Nuevo cuaderno" (New Notebook).

ACTIVAR GPU (Muy importante):

En el menú superior, ve a Entorno de ejecución (Runtime) -> Cambiar tipo de entorno de ejecución (Change runtime type).

En "Acelerador de hardware", selecciona T4 GPU y haz clic en Guardar.

📂 Paso 3: Subir tus datos a Colab

En la barra lateral izquierda de Google Colab, haz clic en el icono de la Carpeta 📁.

Arrastra tu archivo data.zip que creaste en el Paso 1 y suéltalo ahí dentro para subirlo.

Crea una celda de código en Colab, pega la siguiente línea y ejecútala para descomprimir tus fotos en la nube:

!unzip -q data.zip


(Verás que aparece la carpeta data en el menú lateral con tus imágenes listas).

🧠 Paso 4: El Código Unificado de Entrenamiento

Crea una nueva celda de código en Colab, pega todo el siguiente bloque de código (que unifica la red neuronal, Fourier y el cargador de datos) y presiona Play:

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# === 1. MATEMÁTICA DE FOURIER ===
def obtener_espectro_frecuencia(imagen_tensor):
    escala_grises = torch.mean(imagen_tensor, dim=1)
    fourier_transform = torch.fft.fft2(escala_grises)
    fourier_shift = torch.fft.fftshift(fourier_transform)
    magnitud = torch.log(1 + torch.abs(fourier_shift))
    return magnitud.unsqueeze(1)

# === 2. ARQUITECTURA DEL MODELO ===
class TruePixelNet(nn.Module):
    def __init__(self):
        super(TruePixelNet, self).__init__()
        
        self.rama_espacial = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.rama_frecuencia = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.clasificador = nn.Sequential(
            nn.Linear(301056, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out_espacial = self.rama_espacial(x)
        out_espacial = out_espacial.view(out_espacial.size(0), -1) 
        
        x_freq = obtener_espectro_frecuencia(x)
        out_freq = self.rama_frecuencia(x_freq)
        out_freq = out_freq.view(out_freq.size(0), -1)
        
        out_total = torch.cat((out_espacial, out_freq), dim=1)
        return self.clasificador(out_total)

# === 3. CARGADOR DE DATOS ===
class TruePixelDataset(Dataset):
    def __init__(self, directorio_real, directorio_ia):
        self.rutas_imagenes = []
        self.etiquetas = []
        
        for archivo in os.listdir(directorio_real):
            if archivo.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                self.rutas_imagenes.append(os.path.join(directorio_real, archivo))
                self.etiquetas.append(0.0)
                
        for archivo in os.listdir(directorio_ia):
            if archivo.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                self.rutas_imagenes.append(os.path.join(directorio_ia, archivo))
                self.etiquetas.append(1.0)
                
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.rutas_imagenes)

    def __getitem__(self, idx):
        ruta_img = self.rutas_imagenes[idx]
        imagen = Image.open(ruta_img).convert('RGB')
        imagen_tensor = self.transform(imagen)
        etiqueta_tensor = torch.tensor([self.etiquetas[idx]], dtype=torch.float32)
        return imagen_tensor, etiqueta_tensor

# === 4. BUCLE DE ENTRENAMIENTO CON GPU ===
def entrenar_en_colab():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Entrenando con aceleración de hardware en: {device}")
    
    dataset = TruePixelDataset('./data/reales', './data/sinteticas')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    modelo = TruePixelNet().to(device)
    criterio = nn.BCELoss()
    optimizador = optim.Adam(modelo.parameters(), lr=0.0002)
    
    epocas = 40
    for epoca in range(epocas):
        modelo.train()
        perdida_acumulada = 0.0
        correctas = 0
        
        for imagenes
