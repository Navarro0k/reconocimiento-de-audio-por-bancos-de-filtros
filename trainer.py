"""
trainer.py - Parte A: análisis visual (FFT y espectro promedio).

Lee los audios de 3 s generados por audioAnalizer.py (audio_3s/<Carpeta_Ave>/*.wav),
calcula la FFT de cada uno y grafica:
  1) FFT de un audio de ejemplo por ave (señal en el tiempo + espectro).
  2) Espectro promedio de cada ave, una gráfica por especie.
  3) Espectro promedio de las tres aves superpuesto en una sola imagen, para
     elegir a ojo las frecuencias de corte de las subbandas.

Salida: carpeta graficas/
Uso: python trainer.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks

ENTRADA = "audio_3s"
SALIDA = "graficas"
FMIN = 100            # Hz: se ignora el ruido de muy baja frecuencia (viento, rumble)
FMAX = 10000          # Hz máximos a mostrar
VENTANA_MA = 31       # puntos de la media móvil (31 bins ~ 10 Hz con 3 s)
COLORES = {"Copeton": "tab:blue", "Bichofue": "tab:orange", "Mirla": "tab:green"}


def nombre_ave(carpeta):
    return carpeta.replace("Carpeta_", "")


def media_movil(y, k):
    """Media móvil simple de k puntos (suaviza el ruido del espectro)."""
    return np.convolve(y, np.ones(k) / k, mode="same")


def fft_mag(x, fs):
    """Magnitud de la FFT (ventana de Hann) y eje de frecuencias."""
    x = x - np.mean(x)
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / fs)
    return f, X


def cargar(ruta):
    fs, x = wavfile.read(ruta)
    return fs, x.astype(np.float32) / 32768.0


def main():
    if not os.path.isdir(ENTRADA):
        print(f"No existe '{ENTRADA}'. Ejecuta primero audioAnalizer.py.")
        return
    os.makedirs(SALIDA, exist_ok=True)

    carpetas = sorted(d for d in os.listdir(ENTRADA)
                      if os.path.isdir(os.path.join(ENTRADA, d)))
    promedios, ejemplos = {}, {}

    for carpeta in carpetas:
        ave = nombre_ave(carpeta)
        rutas = sorted(glob.glob(os.path.join(ENTRADA, carpeta, "*.wav")))
        if not rutas:
            continue
        acum = None
        for i, r in enumerate(rutas):
            fs, x = cargar(r)
            f, X = fft_mag(x, fs)
            banda = (f >= FMIN) & (f <= FMAX)
            Xn = X / np.linalg.norm(X[banda])   # cada audio pesa igual
            acum = Xn if acum is None else acum + Xn
            if i == 0:
                ejemplos[ave] = (x, fs, f, X)
        promedios[ave] = (f, acum / len(rutas), len(rutas))
        print(f"{ave}: {len(rutas)} audios")

    if not promedios:
        print("No hay audios recortados.")
        return

    # --- 1) FFT de un audio de ejemplo por ave ---------------------------
    fig, ejes = plt.subplots(len(ejemplos), 2, figsize=(13, 3.2 * len(ejemplos)),
                             squeeze=False)
    for fila, (ave, (x, fs, f, X)) in zip(ejes, ejemplos.items()):
        t = np.arange(len(x)) / fs
        c = COLORES.get(ave, "gray")
        fila[0].plot(t, x, color=c, lw=0.5)
        fila[0].set(title=f"{ave} - señal (3 s)", xlabel="Tiempo [s]",
                    ylabel="Amplitud")
        m = (f >= FMIN) & (f <= FMAX)
        fila[1].plot(f[m], X[m], color=c, lw=0.6)
        fila[1].set(title=f"{ave} - FFT", xlabel="Frecuencia [Hz]",
                    ylabel="|X(f)|")
        fila[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(SALIDA, "fft_ejemplos.png"), dpi=130)
    plt.close(fig)

    # --- 2) Espectro promedio, una gráfica por especie -------------------
    fig, ejes = plt.subplots(len(promedios), 1,
                             figsize=(11, 3.0 * len(promedios)), sharex=True,
                             squeeze=False)
    for ax, (ave, (f, P, n)) in zip(ejes[:, 0], promedios.items()):
        m = (f >= FMIN) & (f <= FMAX)
        c = COLORES.get(ave, "gray")
        ax.plot(f[m], P[m], color=c, lw=0.4, alpha=0.4, label="promedio")
        ax.plot(f[m], media_movil(P, VENTANA_MA)[m], color=c, lw=1.4,
                label=f"media móvil ({VENTANA_MA} pts)")
        ax.set(title=f"Espectro promedio - {ave} ({n} audios)",
               ylabel="|X(f)| normalizada")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    ejes[-1, 0].set_xlabel("Frecuencia [Hz]")
    fig.tight_layout()
    fig.savefig(os.path.join(SALIDA, "espectro_promedio_por_ave.png"), dpi=130)
    plt.close(fig)

    # --- 3) Espectros promedio superpuestos (para elegir los cortes) -----
    fig, ax = plt.subplots(figsize=(13, 6))
    for ave, (f, P, n) in promedios.items():
        m = (f >= FMIN) & (f <= FMAX)
        ax.plot(f[m], media_movil(P, VENTANA_MA)[m],
                color=COLORES.get(ave, "gray"), lw=1.5, label=f"{ave} (n={n})")
    ax.set(title="Espectro promedio de las aves (superpuestos, media móvil)",
           xlabel="Frecuencia [Hz]", ylabel="|X(f)| normalizada")
    ax.set_xticks(np.arange(0, FMAX + 1, 500))
    ax.tick_params(axis="x", labelrotation=60)
    ax.grid(alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(SALIDA, "espectro_promedio_superpuesto.png"), dpi=130)
    plt.close(fig)

    # --- picos principales de cada ave (ayuda para elegir los cortes) ----
    print("\nPicos principales del espectro promedio (media móvil):")
    for ave, (f, P, n) in promedios.items():
        S = media_movil(P, VENTANA_MA)
        m = (f >= 300) & (f <= FMAX)
        idx, _ = find_peaks(S[m], distance=int(300 / (f[1] - f[0])),
                            prominence=S[m].max() * 0.1)
        top = idx[np.argsort(S[m][idx])[::-1][:5]]
        print(f"  {ave}: " + ", ".join(f"{f[m][i]:.0f} Hz" for i in sorted(top)))

    print(f"\nGráficas guardadas en {SALIDA}/")


if __name__ == "__main__":
    main()