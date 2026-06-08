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
        # Tamaño aplanado de salida: 32 canales * 56 * 56 = 100,352 características
        
        # 2. Rama de Frecuencia: Analiza el espectro de Fourier
        self.rama_frecuencia = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Reduce el espectro de 224x224 a 112x112
        )
        # Tamaño aplanado de salida: 16 canales * 112 * 112 = 200,704 características
        
        # 3. El Clasificador Final (Suma de características: 100,352 + 200,704 = 301,056)
        self.clasificador = nn.Sequential(
            nn.Linear(in_features=301056, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.5), # Evita el sobreajuste (memorización)
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid() # Devuelve un valor entre 0.0 (Real) y 1.0 (IA)
        )

    def forward(self, x):
        # Procesamos la imagen en la rama visual
        out_espacial = self.rama_espacial(x)
        out_espacial = out_espacial.view(out_espacial.size(0), -1) 
        
        # Procesamos la imagen en la rama matemática de Fourier
        x_freq = obtener_espectro_frecuencia(x) # Extrae el espectro invisible
        out_freq = self.rama_frecuencia(x_freq)
        out_freq = out_freq.view(out_freq.size(0), -1)
        
        # Fusión perfecta: Concatenamos los datos visuales con los datos de frecuencia
        out_total = torch.cat((out_espacial, out_freq), dim=1)
        
        # El juez final dictamina el porcentaje de fraude
        resultado = self.clasificador(out_total)
        return resultado