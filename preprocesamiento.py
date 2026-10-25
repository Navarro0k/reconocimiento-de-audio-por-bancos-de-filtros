"""Parámetros de audio y normalización (igual que los WAV de entrenamiento de audioAnalizer.py)."""
import numpy as np
from filtro import MovingAverageFilter

FS = 22050                  # frecuencia de muestreo
DURACION = 3.0              # segundos
N = int(FS * DURACION)      # muestras por audio

USAR_FILTRO = True          # filtro de media móvil sobre el audio analizado (False = sin filtro)
M_FILTRO = 3                # longitud M de la media móvil (muestras); con M = 5 el acierto bajó en las pruebas
_filtro = MovingAverageFilter(M_FILTRO)


def normalizar(x):
    """Pico a 0.9 y cuantización a 16 bits, como en el entrenamiento."""
    seg = x / max(np.max(np.abs(x)), 1e-9) * 0.9
    return (seg * 32767).astype(np.int16).astype(np.float32) / 32768.0


def filtrar(x):
    """Media móvil de M puntos. Se aplica igual al entrenar y al reconocer (después de normalizar)."""
    if not USAR_FILTRO:
        return x
    return _filtro.apply(x).astype(np.float32)