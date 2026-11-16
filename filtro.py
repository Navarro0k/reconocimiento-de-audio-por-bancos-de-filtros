"""Normalización, filtrado y parámetros de audio, antes de la FFT."""
import numpy as np

FS = 22050
DURACION = 3.0
N = int(FS * DURACION)

USAR_FILTRO = True
M_FILTRO = 3


def normalizar(x):
    """Pico a 0.9 y cuantización a 16 bits, como en el entrenamiento."""
    seg = x / max(np.max(np.abs(x)), 1e-9) * 0.9
    return (seg * 32767).astype(np.int16).astype(np.float32) / 32768.0


class MovingAverageFilter:
    def __init__(self, M: int = 5):
        if not isinstance(M, int) or M < 1:
            raise ValueError("M debe ser un entero positivo (M >= 1).")
        self.M = M

    def apply(self, x: np.ndarray, fs: int = None) -> np.ndarray:
        x = np.asarray(x).flatten().astype(float)
        N = x.size

        if N == 0:
            return x

        if self.M >= N:
            raise ValueError(f"M ({self.M}) debe ser menor que N ({N}).")

        y = np.zeros(N)

        for n in range(N):
            suma = 0.0
            for k in range(self.M):
                if n - k >= 0:
                    suma += x[n - k]

            y[n] = suma / self.M

        return y


_filtro = MovingAverageFilter(M_FILTRO)


def filtrar(x):
    """Media móvil de M puntos. Se aplica igual al entrenar y al reconocer (después de normalizar)."""
    if not USAR_FILTRO:
        return x
    return _filtro.apply(x).astype(np.float32)
