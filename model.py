import torch
import torch.nn as nn
from fourier import obtener_espectro_frecuencia  # <--- Importamos nuestra matemática secreta

class TruePixelNet(nn.Module):
    def __init__(self):
        super(TruePixelNet, self).__init__()
        
        # 1. Rama Espacial: Analiza la imagen normal (píxeles y texturas)
        self.rama_espacial = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # Reduce la imagen de 224x224 a 112x112
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Reduce la imagen a 56x56
        )
        
        # 2. Rama de Frecuencia: Analiza el espectro de Fourier
        self.rama_frecuencia = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Reduce el espectro de 224x224 a 112x112
        )
        
        # 3. Pooling adaptativo: desacopla la arquitectura del tamaño de entrada.
        # Antes la primera capa Linear dependía de 224x224 (301,056 features).
        self.pool_espacial = nn.AdaptiveAvgPool2d((2, 2))
        self.pool_frecuencia = nn.AdaptiveAvgPool2d((4, 4))
        
        # 4. El Clasificador Final:
        # Espacial: 32 canales * 2 * 2 = 128 | Frecuencia: 16 canales * 4 * 4 = 256
        # Total concatenado: 384 características (antes 301,056).
        self.clasificador = nn.Sequential(
            nn.Linear(in_features=384, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.5), # Evita el sobreajuste (memorización)
            nn.Linear(in_features=128, out_features=1)
            # Sin Sigmoid: usamos BCEWithLogitsLoss durante el entrenamiento y
            # aplicamos sigmoid solo en la inferencia (más estable numéricamente).
        )

    def forward(self, x):
        # Procesamos la imagen en la rama visual
        out_espacial = self.pool_espacial(self.rama_espacial(x))
        out_espacial = torch.flatten(out_espacial, 1)
        
        # Procesamos la imagen en la rama matemática de Fourier
        x_freq = obtener_espectro_frecuencia(x) # Extrae el espectro invisible
        out_freq = self.pool_frecuencia(self.rama_frecuencia(x_freq))
        out_freq = torch.flatten(out_freq, 1)
        
        # Fusión perfecta: Concatenamos los datos visuales con los datos de frecuencia
        out_total = torch.cat((out_espacial, out_freq), dim=1)
        
        # El juez final devuelve un logit (valor real sin acotar).
        return self.clasificador(out_total)
