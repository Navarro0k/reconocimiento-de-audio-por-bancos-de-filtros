"""
Parámetros de audio y preprocesamiento de la señal, antes de la FFT.

Nada de este archivo hace parte de las fórmulas del documento (esas están en core.py):
son agregados para que una grabación real del micrófono sea comparable con los audios
de entrenamiento. Todo se aplica igual al entrenar y al reconocer.

  normalizar()    pico a 0.9 y 16 bits, como los WAV de audioAnalizer.py
  filtrar()       media móvil de M puntos             (USAR_FILTRO)
  acondicionar()  pasa-altos y energía unitaria       (USAR_PASA_ALTOS)
"""
import numpy as np
from scipy import signal
from filtro import MovingAverageFilter

FS = 22050                  # frecuencia de muestreo
DURACION = 3.0              # segundos
N = int(FS * DURACION)      # muestras por audio

USAR_FILTRO = True          # filtro de media móvil sobre el audio analizado (False = sin filtro)
M_FILTRO = 3                # longitud M de la media móvil (muestras); con M = 5 el acierto bajó en las pruebas
_filtro = MovingAverageFilter(M_FILTRO)

USAR_PASA_ALTOS = True      # quita el ruido grave (voz, ambiente, zumbido) antes de la FFT
PASA_ALTOS_HZ = 2000        # frecuencia de corte; el canto de las aves está por encima
ORDEN_PASA_ALTOS = 4        # Butterworth
RMS_OBJETIVO = 0.1          # todas las señales quedan con la misma energía
_pasa_altos = signal.butter(ORDEN_PASA_ALTOS, PASA_ALTOS_HZ, "highpass", fs=FS, output="sos")


def normalizar(x):
    """Pico a 0.9 y cuantización a 16 bits, como en el entrenamiento."""
    seg = x / max(np.max(np.abs(x)), 1e-9) * 0.9
    return (seg * 32767).astype(np.int16).astype(np.float32) / 32768.0


def filtrar(x):
    """Media móvil de M puntos. Se aplica igual al entrenar y al reconocer (después de normalizar)."""
    if not USAR_FILTRO:
        return x
    return _filtro.apply(x).astype(np.float32)


def acondicionar(x):
    """
    Deja la señal lista para la FFT, igual al entrenar y al reconocer:
      1) pasa-altos Butterworth: el ruido grave del micrófono no entra a las subbandas;
      2) energía unitaria: x <- x * RMS_OBJETIVO / rms(x), así el volumen de la grabación
         no cambia el vector de energías (por Parseval, sum ||X(k)||^2 queda fija).
    """
    x = np.asarray(x, dtype=np.float64)
    if USAR_PASA_ALTOS:
        x = signal.sosfilt(_pasa_altos, x)
    rms = np.sqrt(np.mean(x ** 2))
    return (x * (RMS_OBJETIVO / max(rms, 1e-12))).astype(np.float32)
