import torch
import torch.fft

# Pesos de luminancia perceptual (Rec. 601). Antes se promediaban los canales
# por igual, lo que no representa cómo percibe el ojo humano.
PESOS_LUMINANCIA = (0.299, 0.587, 0.114)


def obtener_espectro_frecuencia(imagen_tensor, ventana=True):
    """
    Toma un tensor de imagen (B, C, H, W) y devuelve su espectro de magnitud en Fourier.
    Ideal para detectar artefactos de IA generativa.

    Mejoras respecto a la versión anterior:
    - Escala de grises con luminancia perceptual en vez de media simple.
    - Se elimina el componente DC (media espacial), que dominaba la magnitud.
    - Ventana de Hann para reducir fugas espectrales y artefactos de borde
      introducidos por el resize.
    - Normalización por imagen: el espectro es invariante a la escala y al
      offset positivo que introduce la normalización de ImageNet.
    """
    if imagen_tensor.dim() != 4:
        raise ValueError(
            f"Se esperaba un tensor (B, C, H, W) y se recibió {tuple(imagen_tensor.shape)}"
        )

    # 1. Convertimos a escala de grises usando luminancia perceptual
    pesos = torch.tensor(
        PESOS_LUMINANCIA, device=imagen_tensor.device, dtype=imagen_tensor.dtype
    ).view(1, 3, 1, 1)
    escala_grises = torch.sum(imagen_tensor * pesos, dim=1, keepdim=True)

    # 2. Eliminamos el componente DC (media espacial)
    escala_grises = escala_grises - escala_grises.mean(dim=(-2, -1), keepdim=True)

    # 3. Suavizamos los bordes con una ventana de Hann 2D
    if ventana:
        alto, ancho = escala_grises.shape[-2:]
        w_alto = torch.hann_window(alto, periodic=False, device=escala_grises.device, dtype=escala_grises.dtype)
        w_ancho = torch.hann_window(ancho, periodic=False, device=escala_grises.device, dtype=escala_grises.dtype)
        escala_grises = escala_grises * w_alto.view(1, 1, alto, 1) * w_ancho.view(1, 1, 1, ancho)

    # 4. Aplicamos la Transformada Rápida de Fourier 2D (FFT2)
    fourier_transform = torch.fft.fft2(escala_grises, norm="ortho")

    # 5. Desplazamos las frecuencias cero al centro de la cuadrícula
    fourier_shift = torch.fft.fftshift(fourier_transform)

    # 6. Magnitud normalizada por su energía media: cancela la escala global
    # (por ejemplo, la introducida por la normalización de ImageNet) antes
    # de aplicar el logaritmo, que es no lineal.
    magnitud = torch.abs(fourier_shift)
    energia_media = magnitud.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    magnitud = magnitud / energia_media

    # 7. Magnitud logarítmica y estandarización por imagen (media 0, desviación 1)
    magnitud = torch.log1p(magnitud)
    media = magnitud.mean(dim=(-2, -1), keepdim=True)
    desviacion = magnitud.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    magnitud_espectral = (magnitud - media) / desviacion

    # Resultado: (B, 1, H, W) compatible con capas convolucionales
    return magnitud_espectral
