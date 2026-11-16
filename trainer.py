import os
import json
import numpy as np
from scipy.io import wavfile
import core
import utilidades
from filtro import normalizar

CARPETA = "audio_3s"
AVES = ["Carpeta_Copeton", "Carpeta_Mirla"]
RUTA_MODELO = os.path.join("models", "vectores_promedio.json")

def cargar_espectros_de_carpeta(ruta):
    frecuencia_muestreo = None
    espectros = []
    for archivo in sorted(os.listdir(ruta)):
        if not archivo.endswith(".wav"):
            continue
        frecuencia_muestreo, muestras = wavfile.read(os.path.join(ruta, archivo))
        senal = muestras.astype(np.float32) / 32768.0
        espectros.append(core.espectro(normalizar(senal)))
    return frecuencia_muestreo, espectros

def cargar_espectro_promedio(ave):
    frecuencia_muestreo, espectros = cargar_espectros_de_carpeta(os.path.join(CARPETA, ave))
    if not espectros:
        raise FileNotFoundError(f"No hay audios en '{CARPETA}/{ave}'.")
    frecuencias = utilidades.frecuencias(len(espectros[0]), frecuencia_muestreo)
    promedio = core.espectro_promedio(espectros)[:len(frecuencias)]
    return frecuencia_muestreo, frecuencias, promedio

def cargar_espectros():
    frecuencia_muestreo = None
    frecuencias = None
    espectros_por_ave = {}
    for ave in AVES:
        ruta = os.path.join(CARPETA, ave)
        if not os.path.isdir(ruta):
            continue
        frecuencia_ave, espectros = cargar_espectros_de_carpeta(ruta)
        if not espectros:
            continue
        frecuencia_muestreo = frecuencia_ave
        espectros_por_ave[ave] = espectros
        frecuencias = utilidades.frecuencias(len(espectros[0]), frecuencia_muestreo)
    if not espectros_por_ave:
        raise FileNotFoundError(f"No hay audios en '{CARPETA}'. Ejecuta antes audioAnalizer.py.")
    return frecuencia_muestreo, frecuencias, espectros_por_ave

def calcular_espectros_promedio(espectros_por_ave):
    promedios = {}
    for ave, espectros in espectros_por_ave.items():
        promedios[ave] = core.espectro_promedio(espectros)[:len(espectros[0]) // 2 + 1]
    return promedios

def entrenar(frecuencia_muestreo, frecuencias, espectros_por_ave, cortes_hz):
    indices = utilidades.validar_cortes(frecuencias, cortes_hz)

    energias_por_audio = {}
    for ave, espectros in espectros_por_ave.items():
        energias_por_audio[ave] = [core.energia(espectro, indices) for espectro in espectros]

    vectores_promedio = {}
    for ave, energias in energias_por_audio.items():
        vectores_promedio[ave] = core.energia_promedio(energias)

    os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)
    with open(RUTA_MODELO, "w") as archivo:
        json.dump({"fs": frecuencia_muestreo,
                   "cortes_hz": sorted(cortes_hz),
                   "vectores_promedio": vectores_promedio},
                  archivo, indent=4, allow_nan=False)
    return vectores_promedio
