"""Fase de entrenamiento (lógica). La interfaz está en app.py."""
import os
import json
import itertools
import numpy as np
from scipy.io import wavfile
import core
import utilidades
from preprocesamiento import acondicionar

CARPETA = "audio_3s"   # WAV de 3 s ya filtrados por audioAnalizer.py
AVES = ["Carpeta_Copeton", "Carpeta_Mirla"]
COLORES = {"Carpeta_Copeton": "tab:blue", "Carpeta_Mirla": "tab:green"}
RUTA_MODELO = os.path.join("models", "vectores_promedio.json")
METODO = "vector_completo_euclidiana"   # identifica cómo se entrenó el modelo guardado
# Cómo escoge sugerir_cortes() la partición:
#   "acierto"   -> la que más audios de entrenamiento deja junto al vector de su ave
#   "distancia" -> la que más separa los vectores promedio (criterio literal del documento,
#                  ||E_A - E_B|| máxima); con estos audios reconoce peor
CRITERIO_PARTICION = "acierto"


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
    return fs, utilidades.frecuencias(2 * (len(lista[0]) - 1), fs), espectro_promedio(lista)


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
            frec = utilidades.frecuencias(2 * (len(lista[0]) - 1), fs)
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
    indices = utilidades.validar_cortes(frec, cortes_hz)
    N = 2 * (len(frec) - 1)   # puntos de la FFT completa
    return {ave: [core.vector_energias(core.vector_completo(z, N), indices) for z in lista]
            for ave, lista in espectros.items()}


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


def separacion(frec, espectros, cortes_hz):
    """Menor distancia ||E_A - E_B|| entre los vectores promedio de dos aves distintas."""
    E_promedio = vectores_promedio(frec, espectros, cortes_hz)
    return min(core.distancia_euclidiana(E_promedio[a], E_promedio[b])
               for a, b in itertools.combinations(E_promedio, 2))


def sugerir_cortes(frec, espectros, n_cortes=2, paso_hz=250, f_min=2000, f_max=9000, separacion_hz=500):
    """
    Escoge la partición: prueba todas las combinaciones de n_cortes sobre una rejilla de
    paso_hz y devuelve (cortes_hz, acierto) de la mejor según CRITERIO_PARTICION.
    Tarda unos segundos: recalcula las energías de todos los audios con cada partición.
    """
    def acierto(cortes):
        return float(np.mean(list(evaluar(frec, espectros, cortes).values())))

    rejilla = np.arange(f_min, f_max + 1, paso_hz, dtype=float)
    mejor_valor, mejores_cortes = -1.0, None
    for cortes in itertools.combinations(rejilla, n_cortes):
        if min(np.diff(cortes), default=separacion_hz) < separacion_hz:
            continue
        valor = separacion(frec, espectros, cortes) if CRITERIO_PARTICION == "distancia" else acierto(cortes)
        if valor > mejor_valor:
            mejor_valor, mejores_cortes = valor, [float(c) for c in cortes]
    return mejores_cortes, acierto(mejores_cortes)


def guardar_modelo(fs, frec, espectros, cortes_hz):
    """Calcula el vector promedio de energías de cada ave y lo guarda en JSON.
    ValueError si los cortes dejan alguna subbanda vacía (no se guarda nada)."""
    vectores = vectores_promedio(frec, espectros, cortes_hz)
    os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)
    with open(RUTA_MODELO, "w") as f:
        json.dump({"metodo": METODO, "fs": fs, "cortes_hz": sorted(cortes_hz),
                   "vectores_promedio": vectores},
                  f, indent=4, allow_nan=False)
    return vectores