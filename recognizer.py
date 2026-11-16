import os
import json
import math
import time
import core
import utilidades
from filtro import FS, N
from trainer import RUTA_MODELO

def cargar_modelo():
    with open(RUTA_MODELO) as f:
        m = json.load(f)
    if "vectores_promedio" not in m:
        raise ValueError("El modelo guardado no es válido o está incompleto. "
                         "Vuelve a entrenar en la pestaña Entrenamiento.")
    for ave, vector in m["vectores_promedio"].items():
        if any(not math.isfinite(e) for e in vector):
            raise ValueError(f"El modelo guardado tiene energías nan para {ave} "
                             "(subbanda vacía por cortes repetidos). Vuelve a entrenar.")
    return m["cortes_hz"], m["vectores_promedio"]

def iniciar_grabacion():
    import sounddevice as sd
    return sd.rec(N, samplerate=FS, channels=1, dtype="float32")

def terminar_grabacion(buffer):
    import sounddevice as sd
    sd.wait()
    return buffer[:, 0]

CARPETA_GRABACIONES = "grabaciones"

def guardar_grabacion(x):
    from scipy.io import wavfile
    os.makedirs(CARPETA_GRABACIONES, exist_ok=True)
    ruta = os.path.join(CARPETA_GRABACIONES, time.strftime("grabacion_%Y%m%d_%H%M%S.wav"))
    wavfile.write(ruta, FS, x.astype("float32"))
    return ruta

def reconocer(x, cortes_hz, E_promedio):
    N = len(x)
    X = core.espectro(x)
    k_cortes = utilidades.indices_de_corte(utilidades.frecuencias(N, FS), cortes_hz)
    E_x = core.energia(X, k_cortes)
    d = {ave: core.diferencia_euclidiana(E_x, E_ave)
         for ave, E_ave in E_promedio.items()}
    tipo = min(d, key=d.get)
    return tipo, d, E_x
