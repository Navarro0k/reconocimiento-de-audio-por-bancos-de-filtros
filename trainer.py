"""Fase de entrenamiento (lógica). La interfaz está en app.py."""
import os
import json
import numpy as np
from scipy.io import wavfile
import core
import utilidades
from filtro import normalizar

CARPETA = "audio_3s"
AVES = ["Carpeta_Copeton", "Carpeta_Mirla"]
COLORES = {"Carpeta_Copeton": "tab:blue", "Carpeta_Mirla": "tab:green"}
RUTA_MODELO = os.path.join("models", "vectores_promedio.json")
METODO = "vector_completo_euclidiana"


def espectros_de_carpeta(ruta):
    """(fs, [||X(k)|| de cada .wav de la carpeta, k = 0..N-1])"""
    fs, lista = None, []
    for archivo in sorted(os.listdir(ruta)):
        if archivo.endswith(".wav"):
            fs, muestras = wavfile.read(os.path.join(ruta, archivo))
            lista.append(core.espectro(normalizar(muestras.astype(np.float32) / 32768.0)))
    return fs, lista


def cargar_promedio(ave):
    """(fs, frec, ||C(k)||_m) de una sola ave. FileNotFoundError si no hay audios."""
    fs, lista = espectros_de_carpeta(os.path.join(CARPETA, ave))
    if not lista:
        raise FileNotFoundError(f"No hay audios en '{CARPETA}/{ave}'.")
    frec = utilidades.frecuencias(len(lista[0]), fs)
    return fs, frec, core.espectro_promedio(lista)[:len(frec)]


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
            frec = utilidades.frecuencias(len(lista[0]), fs)
    if not espectros:
        raise FileNotFoundError(f"No hay audios en '{CARPETA}'. Ejecuta antes audioAnalizer.py.")
    return fs, frec, espectros


def promedios_de(espectros):
    """{ave: ||C(k)||_m}: espectro promedio de cada ave (mitad útil, solo para graficar)."""
    return {ave: core.espectro_promedio(lista)[:len(lista[0]) // 2 + 1]
            for ave, lista in espectros.items()}


def entrenar(fs, frec, espectros, cortes_hz):
    """Calcula los vectores promedio, mide el acierto y guarda el modelo.
    Devuelve ({ave: E_ave}, {ave: fracción de audios bien clasificados}).
    ValueError si los cortes dejan alguna subbanda vacía (no se guarda nada)."""
    indices = utilidades.validar_cortes(frec, cortes_hz)
    por_audio = {ave: [core.energia(z, indices) for z in lista]
                 for ave, lista in espectros.items()}
    promedios = {ave: core.energia_promedio(vectores) for ave, vectores in por_audio.items()}
    aciertos = {}
    for ave, vectores in por_audio.items():
        bien = sum(1 for E in vectores
                   if min(promedios, key=lambda a: core.diferencia_euclidiana(E, promedios[a])) == ave)
        aciertos[ave] = bien / len(vectores)

    os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)
    with open(RUTA_MODELO, "w") as f:
        json.dump({"metodo": METODO, "fs": fs, "cortes_hz": sorted(cortes_hz),
                   "vectores_promedio": promedios},
                  f, indent=4, allow_nan=False)
    return promedios, aciertos

