"""Fase de entrenamiento (lógica). La interfaz está en app.py."""
import os
import json
import numpy as np
from scipy.io import wavfile
import core

CARPETA = "audio_3s"
AVES = ["Carpeta_Copeton", "Carpeta_Bichofue", "Carpeta_Mirla"]
COLORES = {"Carpeta_Copeton": "tab:blue", "Carpeta_Bichofue": "tab:orange",
           "Carpeta_Mirla": "tab:green"}
RUTA_MODELO = os.path.join("models", "umbrales_energia.json")


def espectros_de_carpeta(ruta):
    """(fs, [||X(k)|| de cada .wav de la carpeta])"""
    fs, lista = None, []
    for archivo in sorted(os.listdir(ruta)):
        if archivo.endswith(".wav"):
            fs, muestras = wavfile.read(os.path.join(ruta, archivo))
            lista.append(core.espectro(muestras.astype(np.float32) / 32768.0))
    return fs, lista


def cargar_promedio(ave):
    """(fs, frec, ||C(k)||_m) de una sola ave. FileNotFoundError si no hay audios."""
    fs, lista = espectros_de_carpeta(os.path.join(CARPETA, ave))
    if not lista:
        raise FileNotFoundError(f"No hay audios en '{CARPETA}/{ave}'.")
    return fs, core.frecuencias(2 * (len(lista[0]) - 1), fs), core.espectro_promedio(lista)


def cargar_promedios():
    """(fs, frec, {ave: ||C(k)||_m}) de todas las aves disponibles."""
    fs, frec, promedios = None, None, {}
    for ave in AVES:
        try:
            fs, frec, promedios[ave] = cargar_promedio(ave)
        except FileNotFoundError:
            continue
    if not promedios:
        raise FileNotFoundError(f"No hay audios en '{CARPETA}'. Ejecuta antes audioAnalizer.py.")
    return fs, frec, promedios


def guardar_modelo(fs, frec, promedios, cortes_hz):
    """Calcula el vector de energías de cada ave y lo guarda en JSON."""
    indices = core.indices_de_corte(frec, cortes_hz)
    vectores = {ave: core.energias_subbandas(z, indices) for ave, z in promedios.items()}
    os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)
    with open(RUTA_MODELO, "w") as f:
        json.dump({"fs": fs, "cortes_hz": sorted(cortes_hz), "energias": vectores}, f, indent=4)
    return vectores