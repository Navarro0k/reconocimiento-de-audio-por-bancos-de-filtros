"""Parámetros de audio y filtrado de la señal, antes de la FFT."""
import numpy as np
from filtro import MovingAverageFilter

FS = 22050
DURACION = 3.0
N = int(FS * DURACION)

USAR_FILTRO = True
M_FILTRO = 3
_filtro = MovingAverageFilter(M_FILTRO)


def filtrar(x):
    """Media móvil de M puntos. Se aplica igual al entrenar y al reconocer (después de normalizar)."""
    if not USAR_FILTRO:
        return x
    return _filtro.apply(x).astype(np.float32)
