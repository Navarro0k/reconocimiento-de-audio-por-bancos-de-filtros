"""
core.py - Fórmulas del proyecto (PROYECTO_FINAL).

  y = fft(x);  z = abs(y)                 -> ||X(k)||
  E_i  = (1/N_i) * sum_k ||X_i(k)||^2     -> energía promedio de la subbanda i de una señal
  E_C  = promedio{ E de cada señal de C } -> vector promedio de energías de la clase C
  d_C  = ||E_X - E_C|| = sqrt( sum_i (E_Xi - E_Ci)^2 )   -> distancia euclidiana
  La señal X se reconoce como la clase de menor distancia.
"""
import numpy as np


def espectro(x):
    """||X(k)||: y = fft(x); z = abs(y) (mitad útil, 0 a fs/2)."""
    y = np.fft.rfft(x)
    z = np.abs(y)
    return z


def frecuencias(N, fs):
    """Frecuencia en Hz de cada índice k (para graficar y ubicar los cortes)."""
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


def energia(subbanda):
    """E_i = (1/N_i) * sum_k ||X_i(k)||^2"""
    if len(subbanda) == 0:
        raise ValueError("Subbanda vacía (N_i = 0): no se puede calcular su energía.")
    return float(np.sum(subbanda ** 2) / len(subbanda))


def energias_subbandas(z, indices_corte):
    """Divide el espectro en subbandas y devuelve el vector E = [E_1 ... E_S]."""
    return [energia(sb) for sb in np.split(z, indices_corte)]


def diferencias_absolutas(E_X, E_C):
    """Vector [ |E_X1 - E_C1|, ..., |E_XS - E_CS| ]"""
    return np.abs(np.asarray(E_X) - np.asarray(E_C))


def vector_promedio(vectores):
    """E_C = promedio de los vectores de energía [E_1 ... E_S] de las señales de una clase."""
    return np.mean(np.asarray(vectores, dtype=np.float64), axis=0).tolist()


def distancia_euclidiana(E_X, E_C):
    """d = ||E_X - E_C|| = sqrt( sum_i (E_Xi - E_Ci)^2 )"""
    return float(np.sqrt(np.sum((np.asarray(E_X, dtype=np.float64) - np.asarray(E_C, dtype=np.float64)) ** 2)))
