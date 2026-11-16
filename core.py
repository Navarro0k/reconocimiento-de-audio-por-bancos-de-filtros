import numpy as np


def espectro(x):
    return np.abs(np.fft.fft(x))


def espectro_promedio(espectros):
    return np.mean(np.asarray(espectros, dtype=np.float64), axis=0)


def energia(X, cortes):
    subbandas = np.split(X, cortes)
    energias = []
    for X_i in subbandas:
        N_i = len(X_i)
        if N_i == 0:
            raise ValueError("Subbanda vacía (N_i = 0): no se puede calcular su energía.")
        energias.append(float((1 / N_i) * np.sum(np.abs(X_i) ** 2)))
    return energias


def energia_promedio(energias):
    return np.mean(np.asarray(energias, dtype=np.float64), axis=0).tolist()


def diferencia_euclidiana(E_x, E_C):
    E_x = np.asarray(E_x, dtype=np.float64)
    E_C = np.asarray(E_C, dtype=np.float64)
    return float(np.sqrt(np.sum((E_x - E_C) ** 2)))
