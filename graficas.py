import os
import json
import numpy as np
import matplotlib.pyplot as plt
import core
import trainer

carpeta_principal = "audio_3s"
carpetas_aves = ["Carpeta_Copeton", "Carpeta_Bichofue", "Carpeta_Mirla"]
carpeta_salida = "graficas"
colores = {"Carpeta_Copeton": "tab:blue", "Carpeta_Bichofue": "tab:orange",
           "Carpeta_Mirla": "tab:green"}

# Cortes elegidos por el usuario en trainer.py
with open(os.path.join("models", "umbrales_energia.json")) as f:
    cortes_hz = json.load(f)["cortes_hz"]

os.makedirs(carpeta_salida, exist_ok=True)
promedios = {}

for ave in carpetas_aves:
    ruta = os.path.join(carpeta_principal, ave)
    if not os.path.exists(ruta):
        print(f"Advertencia: no existe '{ruta}'. Saltando...")
        continue
    fs, espectros = trainer.espectros_de_carpeta(ruta)
    if not espectros:
        continue
    frec = core.frecuencias(2 * (len(espectros[0]) - 1), fs)
    Zp = core.espectro_promedio(espectros)
    promedios[ave] = (frec, Zp)
    indices = core.indices_de_corte(frec, cortes_hz)
    energias = core.energias_subbandas(Zp, indices)
    limites = [0] + list(indices) + [len(Zp)]

    fig, ax = plt.subplots(figsize=(12, 5))
    for Z in espectros:
        ax.plot(frec, Z, color="gray", alpha=0.06, lw=0.5)
    ax.plot(frec, Zp, color=colores[ave], lw=1.2, label="espectro promedio")
    for c in cortes_hz:
        ax.axvline(c, color="k", ls="--", lw=0.8)
    for i, E in enumerate(energias):
        centro = (frec[limites[i]] + frec[limites[i + 1] - 1]) / 2
        ax.text(centro, 0.97, f"E{i + 1} = {E:.1f}",
                transform=ax.get_xaxis_transform(), ha="center", va="top")
    ax.set(title=f"Espectro de la FFT - {ave.replace('Carpeta_', '')} ({len(espectros)} audios)",
           xlabel="Frecuencia [Hz]", ylabel="||X(k)||", xlim=(0, frec[-1]),
           ylim=(0, 4 * Zp[frec > 100].max()))
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.9))
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta_salida, f"espectro_{ave}.png"), dpi=130)
    plt.close(fig)
    print(f"{ave}: {len(espectros)} audios -> graficas/espectro_{ave}.png")

if promedios:
    fig, ax = plt.subplots(figsize=(12, 5))
    for ave, (frec, Zp) in promedios.items():
        ax.plot(frec, Zp, color=colores[ave], lw=1, label=ave.replace("Carpeta_", ""))
    for c in cortes_hz:
        ax.axvline(c, color="k", ls="--", lw=0.8)
    ax.set(title="Espectro promedio de cada ave (líneas punteadas = cortes elegidos)",
           xlabel="Frecuencia [Hz]", ylabel="||X(k)||", xlim=(0, frec[-1]))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta_salida, "espectro_promedio_aves.png"), dpi=130)
    plt.close(fig)
    print("Superpuestas -> graficas/espectro_promedio_aves.png")
