"""
core.py - Fórmulas del documento del proyecto, en el orden en que se aplican.

  1. X(k) = fft(x),  k = 0, ..., N-1                   espectro() y vector_completo()
  2. Partición en subbandas X_1, X_2, X_3               particion()
  3. E_i = (1/N_i) * sum_k ||X_i(k)||^2                 energia() y vector_energias()
  4. E_A = promedio de los vectores E de las señales A  vector_promedio()
  5. d_A = ||E_x - E_A|| = sqrt(sum_i (E_xi - E_Ai)^2)  distancia_euclidiana() y distancias()
  6. x es del tipo con la menor distancia               tipo_reconocido()
"""
import numpy as np


def espectro(x):
    """||X(k)||: y = fft(x); z = abs(y) (mitad útil, 0 a fs/2)."""
    y = np.fft.rfft(x)
    z = np.abs(y)
    return z


def vector_completo(z, N):
    """
    ||X(k)|| para k = 0, ..., N-1 a partir de la mitad útil z que devuelve espectro().
    Como x(n) es real, ||X(N-k)|| = ||X(k)||: la segunda mitad es el reflejo de la primera.
    """
    reflejo = z[-2:0:-1] if N % 2 == 0 else z[:0:-1]
    return np.concatenate([z, reflejo])


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
    limites = np.concatenate(([0], indices, [len(frec)]))   # los cortes van en la mitad útil (0 a fs/2)
    vacias = [i + 1 for i in range(len(limites) - 1) if limites[i + 1] <= limites[i]]
    if vacias:
        raise ValueError(f"Las subbandas {', '.join(f'S{i}' for i in vacias)} quedan vacías: "
                         "hay cortes repetidos o en los extremos del espectro.")
    return indices


def particion(X, k_cortes):
    """Divide el vector X(k) en los subvectores X_1, X_2, ... en los índices k_cortes."""
    return np.split(X, k_cortes)


def energia(X_i):
    """E_i = (1/N_i) * sum_k ||X_i(k)||^2"""
    N_i = len(X_i)
    if N_i == 0:
        raise ValueError("Subbanda vacía (N_i = 0): no se puede calcular su energía.")
    return float((1 / N_i) * np.sum(np.abs(X_i) ** 2))


def vector_energias(X, k_cortes):
    """E = (E_1, E_2, ...): energía promedio de cada subbanda de X(k)."""
    return [energia(X_i) for X_i in particion(X, k_cortes)]


def vector_promedio(vectores):
    """E_A = promedio de los vectores de energía de las señales de la clase A."""
    return np.mean(np.asarray(vectores, dtype=np.float64), axis=0).tolist()


def distancia_euclidiana(E_x, E_A):
    """d_A = ||E_x - E_A|| = sqrt( sum_i (E_xi - E_Ai)^2 )"""
    E_x, E_A = np.asarray(E_x, dtype=np.float64), np.asarray(E_A, dtype=np.float64)
    return float(np.sqrt(np.sum((E_x - E_A) ** 2)))


def distancias(E_x, E_promedio):
    """{clase: d_clase} con d_clase = ||E_x - E_clase||, para cada vector promedio."""
    return {clase: distancia_euclidiana(E_x, E_clase) for clase, E_clase in E_promedio.items()}


def tipo_reconocido(d):
    """La señal es del tipo cuya distancia es la menor."""
    return min(d, key=d.get)
