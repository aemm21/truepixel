import torch
import torch.fft

def obtener_espectro_frecuencia(imagen_tensor):
    """
    Toma un tensor de imagen (B, C, H, W) y devuelve su espectro de magnitud en Fourier.
    Ideal para detectar artefactos de IA generativa.
    """
    # 1. Convertimos a escala de grises (promediando los canales de color RGB)
    # La forma pasa de (Batch, 3, Alto, Ancho) a (Batch, Alto, Ancho)
    escala_grises = torch.mean(imagen_tensor, dim=1)
    
    # 2. Aplicamos la Transformada Rápida de Fourier 2D (FFT2)
    # Esto convierte los datos espaciales en números complejos (frecuencias)
    fourier_transform = torch.fft.fft2(escala_grises)
    
    # 3. Desplazamos las frecuencias cero (bajas frecuencias) al centro de la cuadrícula
    # Esto es el estándar matemático para el análisis visual de espectros
    fourier_shift = torch.fft.fftshift(fourier_transform)
    
    # 4. Calculamos la magnitud logarítmica para estabilizar los valores: M(u,v) = log(1 + |F(u,v)|)
    # Usamos abs() para obtener el valor absoluto del número complejo
    magnitud = torch.log(1 + torch.abs(fourier_shift))
    
    # 5. Añadimos una dimensión extra para que actúe como un "canal de color"
    # Resultado final: (Batch, 1, Alto, Ancho) compatible con capas convolucionales
    magnitud_espectral = magnitud.unsqueeze(1)
    
    return magnitud_espectral