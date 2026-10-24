import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Mismo procesamiento que trainer.py (si cambias algo allá, cámbialo aquí)
carpeta_principal = "audio_3s"
carpetas_aves = ["Carpeta_Copeton", "Carpeta_Bichofue", "Carpeta_Mirla"]
CANTIDAD_SUBBANDAS = 4
carpeta_salida = "graficas"
colores = {"Carpeta_Copeton": "tab:blue", "Carpeta_Bichofue": "tab:orange",
           "Carpeta_Mirla": "tab:green"}

os.makedirs(carpeta_salida, exist_ok=True)
promedios = {}   # nombre_ave -> (frecuencias, espectro promedio)

for nombre_ave in carpetas_aves:
    ruta_carpeta = os.path.join(carpeta_principal, nombre_ave)
    if not os.path.exists(ruta_carpeta):
        print(f"Advertencia: no existe '{ruta_carpeta}'. Saltando...")
        continue

    espectros = []
    for archivo in os.listdir(ruta_carpeta):
        if archivo.endswith(".wav"):
            sr, audio_data = wavfile.read(os.path.join(ruta_carpeta, archivo))
            x = audio_data.astype(np.float32) / 32768.0   # señal x(n)
            y = np.fft.rfft(x)                            # y = fft(x)
            espectros.append(np.abs(y))                   # z = |y|
            frecuencias = np.fft.rfftfreq(len(x), 1 / sr)
    if not espectros:
        continue

    espectro_promedio = np.mean(espectros, axis=0)        # ||X(k)|| promedio
    promedios[nombre_ave] = (frecuencias, espectro_promedio)

    # Subbandas y energías (igual que en trainer.py)
    subbandas = np.array_split(espectro_promedio, CANTIDAD_SUBBANDAS)
    energias = [1 / len(sb) * np.sum(sb ** 2) for sb in subbandas]
    inicios = np.cumsum([0] + [len(sb) for sb in subbandas])   # índice k de cada corte

    # --- Gráfica de un tipo de audio: todos los espectros + el promedio ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for z in espectros:
        ax.plot(frecuencias, z, color="gray", alpha=0.06, lw=0.5)
    ax.plot(frecuencias, espectro_promedio, color=colores[nombre_ave], lw=1.2,
            label="espectro promedio")
    for i, sb in enumerate(subbandas):
        f_ini, f_fin = frecuencias[inicios[i]], frecuencias[inicios[i + 1] - 1]
        if i > 0:
            ax.axvline(f_ini, color="k", ls="--", lw=0.8)
        ax.text((f_ini + f_fin) / 2, 0.97, f"E{i + 1} = {energias[i]:.1f}",
                transform=ax.get_xaxis_transform(), ha="center", va="top")
    ax.set(title=f"Espectro de la FFT - {nombre_ave.replace('Carpeta_', '')} "
                 f"({len(espectros)} audios)",
           xlabel="Frecuencia [Hz]", ylabel="||X(k)||", xlim=(0, frecuencias[-1]),
           ylim=(0, 4 * espectro_promedio[frecuencias > 100].max()))  # recorta picos aislados
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.9))
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta_salida, f"espectro_{nombre_ave}.png"), dpi=130)
    plt.close(fig)
    print(f"{nombre_ave}: {len(espectros)} audios -> graficas/espectro_{nombre_ave}.png")

# --- Gráfica con los espectros promedio de todas las aves superpuestos ---
if promedios:
    fig, ax = plt.subplots(figsize=(12, 5))
    for nombre_ave, (frecuencias, prom) in promedios.items():
        ax.plot(frecuencias, prom, color=colores[nombre_ave], lw=1,
                label=nombre_ave.replace("Carpeta_", ""))
    for i in range(1, CANTIDAD_SUBBANDAS):
        corte = frecuencias[np.cumsum([len(s) for s in
                np.array_split(frecuencias, CANTIDAD_SUBBANDAS)])[i - 1]]
        ax.axvline(corte, color="k", ls="--", lw=0.8)
    ax.set(title="Espectro promedio de cada ave (líneas punteadas = cortes de subbanda)",
           xlabel="Frecuencia [Hz]", ylabel="||X(k)||", xlim=(0, frecuencias[-1]))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta_salida, "espectro_promedio_aves.png"), dpi=130)
    plt.close(fig)
    print("Superpuestas -> graficas/espectro_promedio_aves.png")