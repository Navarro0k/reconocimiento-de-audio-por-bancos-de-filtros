"""Lógica de entrenamiento: carga los audios, calcula sus espectros y guarda el modelo."""
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


def cargar_espectros_de_carpeta(ruta):
    """Devuelve la frecuencia de muestreo y el espectro de cada archivo WAV de la carpeta."""
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
    """Devuelve la frecuencia de muestreo, las frecuencias y el espectro promedio de un ave."""
    frecuencia_muestreo, espectros = cargar_espectros_de_carpeta(os.path.join(CARPETA, ave))
    if not espectros:
        raise FileNotFoundError(f"No hay audios en '{CARPETA}/{ave}'.")
    frecuencias = utilidades.frecuencias(len(espectros[0]), frecuencia_muestreo)
    promedio = core.espectro_promedio(espectros)[:len(frecuencias)]
    return frecuencia_muestreo, frecuencias, promedio


def cargar_espectros():
    """Carga las aves disponibles y devuelve su frecuencia de muestreo, frecuencias y espectros."""
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
    """Calcula el espectro promedio de cada ave. Se usa solo para las gráficas."""
    promedios = {}
    for ave, espectros in espectros_por_ave.items():
        promedios[ave] = core.espectro_promedio(espectros)[:len(espectros[0]) // 2 + 1]
    return promedios


def entrenar(frecuencia_muestreo, frecuencias, espectros_por_ave, cortes_hz):
    """Entrena con los espectros cargados y guarda el modelo.

    Valida los cortes, calcula el vector de energías de cada audio y el promedio de cada ave,
    mide cuántos audios quedan más cerca de su propia ave y escribe el resultado en el JSON.
    Devuelve los vectores promedio y el acierto de cada ave.
    Lanza ValueError si algún corte deja una subbanda vacía; en ese caso no guarda nada.
    """
    indices = utilidades.validar_cortes(frecuencias, cortes_hz)

    energias_por_audio = {}
    for ave, espectros in espectros_por_ave.items():
        energias_por_audio[ave] = [core.energia(espectro, indices) for espectro in espectros]

    vectores_promedio = {}
    for ave, energias in energias_por_audio.items():
        vectores_promedio[ave] = core.energia_promedio(energias)

    aciertos = {}
    for ave, energias in energias_por_audio.items():
        acertados = 0
        for energia in energias:
            distancias = {nombre: core.diferencia_euclidiana(energia, vector)
                          for nombre, vector in vectores_promedio.items()}
            if min(distancias, key=distancias.get) == ave:
                acertados += 1
        aciertos[ave] = acertados / len(energias)

    os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)
    with open(RUTA_MODELO, "w") as archivo:
        json.dump({"metodo": METODO,
                   "fs": frecuencia_muestreo,
                   "cortes_hz": sorted(cortes_hz),
                   "vectores_promedio": vectores_promedio},
                  archivo, indent=4, allow_nan=False)
    return vectores_promedio, aciertos
