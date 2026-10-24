"""Fase de reconocimiento (lógica). La interfaz está en app.py."""
import json
import core
from preprocesamiento import FS, N
from trainer import RUTA_MODELO


def cargar_modelo():
    """(cortes_hz, {ave: vector de umbrales de energía})"""
    with open(RUTA_MODELO) as f:
        m = json.load(f)
    return m["cortes_hz"], m["energias"]


def iniciar_grabacion():
    """Empieza a grabar DURACION segundos del micrófono (no bloquea)."""
    import sounddevice as sd
    return sd.rec(N, samplerate=FS, channels=1, dtype="float32")


def terminar_grabacion(buffer):
    """Espera a que termine la grabación y devuelve la señal x."""
    import sounddevice as sd
    sd.wait()
    return buffer[:, 0]


def reconocer(x, cortes_hz, umbrales):
    """
    E_X = vector de energía de la señal. Por cada ave: diferencias |E_Xi - E_ave,i| y su suma E_X<ave>.
    Devuelve (resultados, E_X); resultados = [(ave, diferencias, suma)] de menor a mayor suma,
    así que resultados[0] es el tipo reconocido.
    """
    z = core.espectro(x)
    indices = core.indices_de_corte(core.frecuencias(len(x), FS), cortes_hz)
    E_X = core.energias_subbandas(z, indices)
    resultados = [(ave, core.diferencias_absolutas(E_X, E), core.suma_diferencias_absolutas(E_X, E))
                  for ave, E in umbrales.items()]
    return sorted(resultados, key=lambda r: r[2]), E_X