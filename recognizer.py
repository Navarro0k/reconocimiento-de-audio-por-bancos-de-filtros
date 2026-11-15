"""Fase de reconocimiento (lógica). La interfaz está en app.py."""
import json
import math
import core
from preprocesamiento import FS, N
from trainer import RUTA_MODELO, METODO


def cargar_modelo():
    """(cortes_hz, {ave: vector promedio de energías E_ave})"""
    with open(RUTA_MODELO) as f:
        m = json.load(f)
    if m.get("metodo") != METODO or "vectores_promedio" not in m:
        raise ValueError("El modelo guardado se entrenó con una versión anterior del método. "
                         "Vuelve a entrenar en la pestaña Entrenamiento.")
    for ave, vector in m["vectores_promedio"].items():
        if any(not math.isfinite(e) for e in vector):
            raise ValueError(f"El modelo guardado tiene energías nan para {ave} "
                             "(subbanda vacía por cortes repetidos). Vuelve a entrenar.")
    return m["cortes_hz"], m["vectores_promedio"]


def iniciar_grabacion():
    """Empieza a grabar DURACION segundos del micrófono (no bloquea)."""
    import sounddevice as sd
    return sd.rec(N, samplerate=FS, channels=1, dtype="float32")


def terminar_grabacion(buffer):
    """Espera a que termine la grabación y devuelve la señal x."""
    import sounddevice as sd
    sd.wait()
    return buffer[:, 0]


def reconocer(x, cortes_hz, E_promedio):
    """
    Reconoce la señal x (ya acondicionada) siguiendo los pasos del documento.
    Devuelve (tipo, d, E_x): el ave reconocida, {ave: distancia} y el vector de energías de x.
    """
    N = len(x)
    X = core.vector_completo(core.espectro(x), N)                          # 1. ||X(k)||, k = 0..N-1
    k_cortes = core.indices_de_corte(core.frecuencias(N, FS), cortes_hz)
    E_x = core.vector_energias(X, k_cortes)                                # 2 y 3. E_x = (E_1, E_2, E_3)
    d = core.distancias(E_x, E_promedio)                                   # 5. d_ave = ||E_x - E_ave||
    tipo = core.tipo_reconocido(d)                                         # 6. menor distancia
    return tipo, d, E_x
