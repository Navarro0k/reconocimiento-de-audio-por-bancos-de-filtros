"""Parámetros de audio y normalización (igual que los WAV de entrenamiento de audioAnalizer.py)."""
import numpy as np

FS = 22050                  # frecuencia de muestreo
DURACION = 3.0              # segundos
N = int(FS * DURACION)      # muestras por audio


def normalizar(x):
    """Pico a 0.9 y cuantización a 16 bits, como en el entrenamiento."""
    seg = x / max(np.max(np.abs(x)), 1e-9) * 0.9
    return (seg * 32767).astype(np.int16).astype(np.float32) / 32768.0
