"""Fase de entrenamiento (lógica). La interfaz está en app.py."""
import os
import json
import itertools
import numpy as np
from scipy.io import wavfile
import core
from preprocesamiento import acondicionar

CARPETA = "audio_3s"   # WAV de 3 s ya filtrados por audioAnalizer.py
AVES = ["Carpeta_Copeton", "Carpeta_Bichofue", "Carpeta_Mirla"]
COLORES = {"Carpeta_Copeton": "tab:blue", "Carpeta_Bichofue": "tab:orange",
           "Carpeta_Mirla": "tab:green"}
RUTA_MODELO = os.path.join("models", "umbrales_energia.json")
METODO = "energia_por_senal_euclidiana"   # identifica cómo se entrenó el modelo guardado


def espectros_de_carpeta(ruta):
    """(fs, [||X(k)|| de cada .wav de la carpeta])"""
    fs, lista = None, []
    for archivo in sorted(os.listdir(ruta)):
        if archivo.endswith(".wav"):
            fs, muestras = wavfile.read(os.path.join(ruta, archivo))
            lista.append(core.espectro(acondicionar(muestras.astype(np.float32) / 32768.0)))
    return fs, lista


def espectro_promedio(lista_espectros):
    """||C(k)||_m = promedio de los espectros de una ave. Solo para graficar: no entra al modelo."""
    return np.mean(lista_espectros, axis=0)


def cargar_promedio(ave):
    """(fs, frec, ||C(k)||_m) de una sola ave. FileNotFoundError si no hay audios."""
    fs, lista = espectros_de_carpeta(os.path.join(CARPETA, ave))
    if not lista:
        raise FileNotFoundError(f"No hay audios en '{CARPETA}/{ave}'.")
    return fs, core.frecuencias(2 * (len(lista[0]) - 1), fs), espectro_promedio(lista)


def cargar_espectros():
    """(fs, frec, {ave: [||X(k)|| de cada audio]}) de todas las aves disponibles."""
    fs, frec, espectros = None, None, {}
    for ave in AVES:
        ruta = os.path.join(CARPETA, ave)
        if not os.path.isdir(ruta):
            continue
        fs_ave, lista = espectros_de_carpeta(ruta)
        if lista:
            fs, espectros[ave] = fs_ave, lista
            frec = core.frecuencias(2 * (len(lista[0]) - 1), fs)
    if not espectros:
        raise FileNotFoundError(f"No hay audios en '{CARPETA}'. Ejecuta antes audioAnalizer.py.")
    return fs, frec, espectros


def promedios_de(espectros):
    """{ave: ||C(k)||_m}: espectro promedio de cada ave (solo para graficar)."""
    return {ave: espectro_promedio(lista) for ave, lista in espectros.items()}


def cargar_promedios():
    """(fs, frec, {ave: ||C(k)||_m}) de todas las aves disponibles."""
    fs, frec, espectros = cargar_espectros()
    return fs, frec, promedios_de(espectros)


def vectores_por_audio(frec, espectros, cortes_hz):
    """{ave: [vector de energías [E_1 ... E_S] de cada audio]}"""
    indices = core.validar_cortes(frec, cortes_hz)
    return {ave: [core.energias_subbandas(z, indices) for z in lista] for ave, lista in espectros.items()}


def vectores_promedio(frec, espectros, cortes_hz):
    """{ave: E_ave}: promedio de los vectores de energía de los audios de cada ave."""
    return {ave: core.vector_promedio(v) for ave, v in vectores_por_audio(frec, espectros, cortes_hz).items()}


def evaluar(frec, espectros, cortes_hz):
    """{ave: fracción de sus audios de entrenamiento que quedan más cerca de su propio vector}"""
    por_audio = vectores_por_audio(frec, espectros, cortes_hz)
    promedios = {ave: core.vector_promedio(v) for ave, v in por_audio.items()}
    aciertos = {}
    for ave, vectores in por_audio.items():
        bien = sum(1 for E in vectores
                   if min(promedios, key=lambda a: core.distancia_euclidiana(E, promedios[a])) == ave)
        aciertos[ave] = bien / len(vectores)
    return aciertos


def sugerir_cortes(frec, espectros, n_cortes=3, paso_hz=250, f_min=500, f_max=10000, separacion_hz=500):
    """
    Busca la partición que hace más diferentes las energías de las subbandas de cada ave:
    prueba todas las combinaciones de n_cortes sobre una rejilla de paso_hz y devuelve
    (cortes_hz, acierto) de la que más audios de entrenamiento deja junto al vector de su ave
    (mismo criterio de evaluar(): distancia euclidiana al vector promedio).
    """
    aves = list(espectros)
    # Suma acumulada de ||X(k)||^2 de cada audio: la energía de una subbanda sale de dos restas
    acum = np.concatenate([np.cumsum(np.square(np.asarray(espectros[a], dtype=np.float64)), axis=1)
                           for a in aves])
    acum = np.concatenate([np.zeros((len(acum), 1)), acum], axis=1)
    clase = np.concatenate([np.full(len(espectros[a]), i) for i, a in enumerate(aves)])
    rejilla = np.arange(f_min, f_max + 1, paso_hz, dtype=float)
    k_rejilla = core.indices_de_corte(frec, rejilla)
    salto = max(int(round(separacion_hz / paso_hz)), 1)

    mejor, mejores_cortes = -1.0, None
    for comb in itertools.combinations(range(len(rejilla)), n_cortes):
        if any(b - a < salto for a, b in zip(comb, comb[1:])):
            continue
        lim = np.concatenate(([0], k_rejilla[list(comb)], [len(frec)]))
        if np.any(np.diff(lim) <= 0):
            continue
        E = (acum[:, lim[1:]] - acum[:, lim[:-1]]) / np.diff(lim)          # un vector por audio
        prom = np.stack([E[clase == i].mean(axis=0) for i in range(len(aves))])
        cercana = np.argmin(((E[:, None, :] - prom[None]) ** 2).sum(axis=2), axis=1)
        acierto = np.mean([(cercana[clase == i] == i).mean() for i in range(len(aves))])
        if acierto > mejor:
            mejor, mejores_cortes = acierto, rejilla[list(comb)].tolist()
    return mejores_cortes, float(mejor)


def guardar_modelo(fs, frec, espectros, cortes_hz):
    """Calcula el vector promedio de energías de cada ave y lo guarda en JSON.
    ValueError si los cortes dejan alguna subbanda vacía (no se guarda nada)."""
    vectores = vectores_promedio(frec, espectros, cortes_hz)
    os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)
    with open(RUTA_MODELO, "w") as f:
        json.dump({"metodo": METODO, "fs": fs, "cortes_hz": sorted(cortes_hz), "energias": vectores},
                  f, indent=4, allow_nan=False)
    return vectores