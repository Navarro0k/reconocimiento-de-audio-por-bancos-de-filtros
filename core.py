"""
core.py - Fórmulas del proyecto (PROYECTO_FINAL).

  y = fft(x);  z = abs(y)                 -> ||X(k)||
  ||C(k)||_m = promedio{ ||C(k)|| }       -> espectro promedio
  E_Ci = (1/N_i) * sum_{k=0}^{N_i-1} ||C_i(k)||_m^2
  E_XC = sum_i | E_Xi - E_Ci |
"""
import numpy as np


def espectro(x):
    """||X(k)||: y = fft(x); z = abs(y) (mitad útil, 0 a fs/2)."""
    y = np.fft.rfft(x)
    z = np.abs(y)
    return z


def espectro_promedio(lista_espectros):
    """||C(k)||_m = promedio de los espectros de la carpeta."""
    return np.mean(lista_espectros, axis=0)


def frecuencias(N, fs):
    """Frecuencia en Hz de cada índice k (para graficar y ubicar los cortes)."""
    return np.fft.rfftfreq(N, d=1.0 / fs)


def indices_de_corte(frec, cortes_hz):
    """Cortes elegidos en la gráfica [Hz] -> índices k del espectro."""
    return np.searchsorted(frec, np.sort(cortes_hz))


def energia(subbanda):
    """E = (1/N_i) * sum ||C_i(k)||_m^2"""
    return float(np.sum(subbanda ** 2) / len(subbanda))


def energias_subbandas(z, indices_corte):
    """Divide el espectro en subbandas y devuelve el vector E = [E_1 ... E_S]."""
    return [energia(sb) for sb in np.split(z, indices_corte)]


def diferencias_absolutas(E_X, E_C):
    """Vector [ |E_X1 - E_C1|, ..., |E_XS - E_CS| ]"""
    return np.abs(np.asarray(E_X) - np.asarray(E_C))


def suma_diferencias_absolutas(E_X, E_C):
    """E_XC = sum_i | E_Xi - E_Ci |"""
    return float(np.sum(diferencias_absolutas(E_X, E_C)))