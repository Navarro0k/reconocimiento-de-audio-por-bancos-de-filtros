
import numpy as np


def frecuencias(N, fs):
    return np.fft.rfftfreq(N, d=1.0 / fs)


def indices_de_corte(frec, cortes_hz):
    return np.searchsorted(frec, np.sort(cortes_hz))


def validar_cortes(frec, cortes_hz):

    indices = indices_de_corte(frec, cortes_hz)
    limites = np.concatenate(([0], indices, [len(frec)]))
    vacias = [i + 1 for i in range(len(limites) - 1) if limites[i + 1] <= limites[i]]
    if vacias:
        raise ValueError(f"Las subbandas {', '.join(f'S{i}' for i in vacias)} quedan vacías: "
                         "hay cortes repetidos o en los extremos del espectro.")
    return indices
