import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from fourier import obtener_espectro_frecuencia  # <--- Importamos nuestra matemática secreta


class TruePixelNet(nn.Module):
    def __init__(self, pretrained=True):
        super(TruePixelNet, self).__init__()

        # 1. Rama Espacial: ResNet18 preentrenada en ImageNet. Conservamos las
        # capas convolucionales y descartamos su pooling y clasificador.
        pesos = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=pesos)
        self.rama_espacial = nn.Sequential(*list(backbone.children())[:-2])
        self.pool_espacial = nn.AdaptiveAvgPool2d((1, 1))

        # 2. Rama de Frecuencia: Analiza el espectro de Fourier
        self.rama_frecuencia = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Reduce el espectro de 224x224 a 112x112
        )
        self.pool_frecuencia = nn.AdaptiveAvgPool2d((4, 4))

        # 3. El Clasificador Final:
        # ResNet18 aporta 512 características | Frecuencia: 16 canales * 4 * 4 = 256
        # Total concatenado: 768 características.
        self.clasificador = nn.Sequential(
            nn.Linear(in_features=512 + 256, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.5),  # Evita el sobreajuste (memorización)
            nn.Linear(in_features=128, out_features=1)
            # Sin Sigmoid: usamos BCEWithLogitsLoss durante el entrenamiento y
            # aplicamos sigmoid solo en la inferencia (más estable numéricamente).
        )

    def forward(self, x):
        # Procesamos la imagen en la rama visual
        out_espacial = self.pool_espacial(self.rama_espacial(x))
        out_espacial = torch.flatten(out_espacial, 1)

        # Procesamos la imagen en la rama matemática de Fourier
        x_freq = obtener_espectro_frecuencia(x)  # Extrae el espectro invisible
        out_freq = self.pool_frecuencia(self.rama_frecuencia(x_freq))
        out_freq = torch.flatten(out_freq, 1)

        # Fusión perfecta: Concatenamos los datos visuales con los datos de frecuencia
        out_total = torch.cat((out_espacial, out_freq), dim=1)

        # El juez final devuelve un logit (valor real sin acotar).
        return self.clasificador(out_total)
