"""
core.py - Las fórmulas del documento, escritas como se aplican.

  1. X(k) = |FFT(x)|,                                k = 0, ..., N-1     espectro()
  2. C(k) = (1/M) * sum_j X_j(k)                                         espectro_promedio()
  3. E_i = (1/N_i) * sum_{k en S_i} |X(k)|^2                             energia()
  4. E_C = (1/M) * sum_j E_j                                             energia_promedio()
  5. d_C = ||E_x - E_C|| = sqrt( sum_i (E_x,i - E_C,i)^2 )               diferencia_euclidiana()
"""
import numpy as np


def espectro(x):
    """
    X(k) = |FFT(x)| para k = 0, ..., N-1.

    Como x es real, |X(N-k)| = |X(k)|: se calcula la mitad útil con rfft y
    se refleja para completar los N puntos.
    """
    mitad = np.abs(np.fft.rfft(x))
    reflejo = mitad[-2:0:-1] if len(x) % 2 == 0 else mitad[:0:-1]
    return np.concatenate([mitad, reflejo])


def espectro_promedio(espectros):
    """C(k) = (1/M) * sum_{j=1}^{M} X_j(k): promedio de los espectros de una clase."""
    return np.mean(np.asarray(espectros, dtype=np.float64), axis=0)


def energia(X, cortes):
    """
    E = (E_1, E_2, ..., E_S), con

        E_i = (1/N_i) * sum_{k en S_i} |X(k)|^2.

    `cortes` son los índices donde empieza cada subbanda S_i en X(k).
    """
    subbandas = np.split(X, cortes)
    energias = []
    for X_i in subbandas:
        N_i = len(X_i)
        if N_i == 0:
            raise ValueError("Subbanda vacía (N_i = 0): no se puede calcular su energía.")
        energias.append(float((1 / N_i) * np.sum(np.abs(X_i) ** 2)))
    return energias


def energia_promedio(energias):
    """E_C = (1/M) * sum_{j=1}^{M} E_j, componente a componente."""
    return np.mean(np.asarray(energias, dtype=np.float64), axis=0).tolist()


def diferencia_euclidiana(E_x, E_C):
    """d_C = ||E_x - E_C|| = sqrt( sum_i (E_x,i - E_C,i)^2 )."""
    E_x = np.asarray(E_x, dtype=np.float64)
    E_C = np.asarray(E_C, dtype=np.float64)
    return float(np.sqrt(np.sum((E_x - E_C) ** 2)))
