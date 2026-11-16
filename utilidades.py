"""
utilidades.py - Funciones de apoyo para ubicar los cortes y graficar.

No son fórmulas del documento: traducen frecuencias [Hz] a índices k y
comprueban que ninguna subbanda quede vacía.
"""
import numpy as np


def frecuencias(N, fs):
    """Frecuencia en Hz de cada índice k de la mitad útil del espectro (0 a fs/2)."""
    return np.fft.rfftfreq(N, d=1.0 / fs)


def indices_de_corte(frec, cortes_hz):
    """Cortes elegidos en la gráfica [Hz] -> índices k del espectro."""
    return np.searchsorted(frec, np.sort(cortes_hz))


def validar_cortes(frec, cortes_hz):
    """
    Lanza ValueError si algún corte deja una subbanda sin muestras.
    Pasa cuando dos cortes caen en el mismo índice k (p. ej. doble clic en el mismo
    punto), o un corte queda en 0 Hz o en fs/2: esa subbanda tendría N_i = 0 y
    E = 0/0 = nan.
    """
    indices = indices_de_corte(frec, cortes_hz)
    limites = np.concatenate(([0], indices, [len(frec)]))
    vacias = [i + 1 for i in range(len(limites) - 1) if limites[i + 1] <= limites[i]]
    if vacias:
        raise ValueError(f"Las subbandas {', '.join(f'S{i}' for i in vacias)} quedan vacías: "
                         "hay cortes repetidos o en los extremos del espectro.")
    return indices
