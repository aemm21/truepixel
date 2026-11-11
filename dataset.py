import os
from PIL import Image
import torch
from torch.utils.data import Dataset

from config import EXTENSIONES_VALIDAS, transform_inferencia, validar_tamano_imagen


class TruePixelDataset(Dataset):
    def __init__(self, directorio_real, directorio_ia, transform=None, recursivo=False):
        """
        Inicializa el dataset leyendo las rutas de las imágenes.

        Args:
            directorio_real: carpeta con imágenes reales (etiqueta 0).
            directorio_ia: carpeta con imágenes sintéticas (etiqueta 1).
            transform: transformación opcional de torchvision.
            recursivo: si True, busca imágenes en subcarpetas.
        """
        self.rutas_imagenes = []
        self.etiquetas = []

        # Validar que las carpetas existan para evitar errores
        if not os.path.exists(directorio_real):
            raise FileNotFoundError(f"¡Atención! No se encontró la carpeta de reales: {directorio_real}")
        if not os.path.exists(directorio_ia):
            raise FileNotFoundError(f"¡Atención! No se encontró la carpeta de sintéticas: {directorio_ia}")

        # 1. Cargar imágenes reales (Etiqueta = 0)
        for ruta in self._listar_imagenes(directorio_real, recursivo):
            self.rutas_imagenes.append(ruta)
            self.etiquetas.append(0.0)  # 0.0 significa "Real"

        # 2. Cargar imágenes generadas por IA (Etiqueta = 1)
        for ruta in self._listar_imagenes(directorio_ia, recursivo):
            self.rutas_imagenes.append(ruta)
            self.etiquetas.append(1.0)  # 1.0 significa "Sintética/IA"

        # --- VALIDACIÓN ---
        if len(self.rutas_imagenes) == 0:
            raise ValueError(
                f"\n¡Atención! Las carpetas están vacías o las imágenes no tienen un formato válido "
                f"({', '.join(EXTENSIONES_VALIDAS)}).\n"
                f"Asegúrate de agregar al menos una imagen en '{directorio_real}' y otra en '{directorio_ia}'."
            )

        # 3. Transformaciones
        self.transform = transform if transform else transform_inferencia()

    @staticmethod
    def _listar_imagenes(directorio, recursivo):
        """Devuelve rutas de imagen ordenadas (deterministas) y filtra por extensión."""
        rutas = []
        if recursivo:
            for raiz, _, archivos in os.walk(directorio):
                for archivo in sorted(archivos):
                    rutas.append(os.path.join(raiz, archivo))
        else:
            for archivo in sorted(os.listdir(directorio)):
                ruta = os.path.join(directorio, archivo)
                if os.path.isfile(ruta):
                    rutas.append(ruta)

        return [ruta for ruta in rutas if ruta.lower().endswith(EXTENSIONES_VALIDAS)]

    def __len__(self):
        return len(self.rutas_imagenes)

    def __getitem__(self, idx):
        # Si una imagen está corrupta, probamos con la siguiente (máx. 3 intentos)
        # en lugar de abortar todo el entrenamiento.
        intentos = 0
        while True:
            ruta_img = self.rutas_imagenes[idx]
            etiqueta = self.etiquetas[idx]
            try:
                with Image.open(ruta_img) as imagen:
                    validar_tamano_imagen(imagen)
                    imagen = imagen.convert('RGB')
                imagen_tensor = self.transform(imagen)
                etiqueta_tensor = torch.tensor([etiqueta], dtype=torch.float32)
                return imagen_tensor, etiqueta_tensor
            except Exception as exc:
                intentos += 1
                if intentos >= 3:
                    raise RuntimeError(f"No se pudo cargar la imagen '{ruta_img}': {exc}") from exc
                idx = (idx + 1) % len(self.rutas_imagenes)
