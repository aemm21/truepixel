import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class TruePixelDataset(Dataset):
    def __init__(self, directorio_real, directorio_ia, transform=None):
        """
        Inicializa el dataset leyendo las rutas de las imágenes.
        """
        self.rutas_imagenes = []
        self.etiquetas = []
        
        # Validar que las carpetas existan para evitar errores
        if not os.path.exists(directorio_real):
            raise FileNotFoundError(f"¡Atención! No se encontró la carpeta de reales: {directorio_real}")
        if not os.path.exists(directorio_ia):
            raise FileNotFoundError(f"¡Atención! No se encontró la carpeta de sintéticas: {directorio_ia}")
            
        # 1. Cargar imágenes reales (Etiqueta = 0)
        for archivo in os.listdir(directorio_real):
            if archivo.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                self.rutas_imagenes.append(os.path.join(directorio_real, archivo))
                self.etiquetas.append(0.0) # 0.0 significa "Real"
                
        # 2. Cargar imágenes generadas por IA (Etiqueta = 1)
        for archivo in os.listdir(directorio_ia):
            if archivo.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                self.rutas_imagenes.append(os.path.join(directorio_ia, archivo))
                self.etiquetas.append(1.0) # 1.0 significa "Sintética/IA"
                
        # --- NUEVA VALIDACIÓN ---
        # Verificamos si logramos cargar al menos una imagen
        if len(self.rutas_imagenes) == 0:
            raise ValueError(
                f"\n¡Atención! Las carpetas están vacías o las imágenes no tienen un formato válido (.jpg, .png).\n"
                f"Asegúrate de agregar al menos una imagen en '{directorio_real}' y otra en '{directorio_ia}'."
            )
                
        # 3. Transformaciones
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.rutas_imagenes)

    def __getitem__(self, idx):
        ruta_img = self.rutas_imagenes[idx]
        etiqueta = self.etiquetas[idx]
        
        imagen = Image.open(ruta_img).convert('RGB')
        imagen_tensor = self.transform(imagen)
        etiqueta_tensor = torch.tensor([etiqueta], dtype=torch.float32)
        
        return imagen_tensor, etiqueta_tensor