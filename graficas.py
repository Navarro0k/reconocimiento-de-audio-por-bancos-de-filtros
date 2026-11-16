import os
import json
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import core
import trainer
import utilidades
from filtro import FS

COLORES = {"Carpeta_Copeton": "tab:blue", "Carpeta_Mirla": "tab:green"}


def crear_grafica(padre):
    fig = Figure(figsize=(7, 4.2), layout="constrained")
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, padre)
    barra = NavigationToolbar2Tk(canvas, padre, pack_toolbar=False)
    barra.update()
    barra.pack(side="bottom", fill="x")
    canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
    return ax, canvas, barra


def dibujar_espectros_promedio(ax, frec, promedios):
    ax.clear()
    for ave, espectro in promedios.items():
        ax.plot(frec, espectro, color=COLORES[ave], lw=1, label=ave.replace("Carpeta_", ""))
    tope = max(espectro[frec > 100].max() for espectro in promedios.values())
    ax.set(xlabel="Frecuencia [Hz]", ylabel="||C(k)||_m", title="Espectros promedio",
           xlim=(0, frec[-1]), ylim=(0, 4 * tope))
    ax.legend(loc="upper right")


def dibujar_reconocimiento(ax, senal, frec, frec_promedio, promedio, ganador, cortes):
    ax.clear()
    ax.plot(frec, core.espectro(senal)[:len(frec)], color="tab:gray", lw=0.8,
            label="Grabación X")
    ax.plot(frec_promedio, promedio, color=COLORES[ganador], lw=1.2,
            label=f"Promedio {ganador.replace('Carpeta_', '')} (mínima distancia)")
    for corte in cortes:
        ax.axvline(corte, color="k", ls="--", lw=0.8)
    ax.set(xlabel="Frecuencia [Hz]", ylabel="||X(k)||", xlim=(0, FS / 2),
           title="Espectro de la grabación vs. ave reconocida")
    ax.legend(loc="upper right")


def dibujar_corte(ax, frecuencia):
    return ax.axvline(frecuencia, color="k", ls="--", lw=1)


def generar_informe():
    carpeta_principal = "audio_3s"
    carpeta_salida = "graficas"

    with open(trainer.RUTA_MODELO) as archivo:
        cortes_hz = json.load(archivo)["cortes_hz"]

    os.makedirs(carpeta_salida, exist_ok=True)
    promedios = {}

    for ave in trainer.AVES:
        ruta = os.path.join(carpeta_principal, ave)
        if not os.path.exists(ruta):
            print(f"Advertencia: no existe '{ruta}'. Saltando...")
            continue
        fs, espectros = trainer.cargar_espectros_de_carpeta(ruta)
        if not espectros:
            continue
        frec = utilidades.frecuencias(len(espectros[0]), fs)
        espectro_promedio = core.espectro_promedio(espectros)
        promedios[ave] = (frec, espectro_promedio[:len(frec)])
        indices = utilidades.indices_de_corte(frec, cortes_hz)
        energias = core.energia(espectro_promedio, indices)
        limites = [0] + list(indices) + [len(frec)]

        fig, ax = plt.subplots(figsize=(12, 5))
        for espectro in espectros:
            ax.plot(frec, espectro[:len(frec)], color="gray", alpha=0.06, lw=0.5)
        ax.plot(frec, espectro_promedio[:len(frec)], color=COLORES[ave], lw=1.2,
                label="espectro promedio")
        for corte in cortes_hz:
            ax.axvline(corte, color="k", ls="--", lw=0.8)
        for i, energia in enumerate(energias):
            centro = (frec[limites[i]] + frec[limites[i + 1] - 1]) / 2
            ax.text(centro, 0.97, f"E{i + 1} = {energia:.1f}",
                    transform=ax.get_xaxis_transform(), ha="center", va="top")
        ax.set(title=f"Espectro de la FFT - {ave.replace('Carpeta_', '')} ({len(espectros)} audios)",
               xlabel="Frecuencia [Hz]", ylabel="||X(k)||", xlim=(0, frec[-1]),
               ylim=(0, 4 * espectro_promedio[:len(frec)][frec > 100].max()))
        ax.legend(loc="upper right", bbox_to_anchor=(1, 0.9))
        fig.tight_layout()
        fig.savefig(os.path.join(carpeta_salida, f"espectro_{ave}.png"), dpi=130)
        plt.close(fig)
        print(f"{ave}: {len(espectros)} audios -> graficas/espectro_{ave}.png")

    if promedios:
        fig, ax = plt.subplots(figsize=(12, 5))
        for ave, (frec, espectro) in promedios.items():
            ax.plot(frec, espectro, color=COLORES[ave], lw=1, label=ave.replace("Carpeta_", ""))
        for corte in cortes_hz:
            ax.axvline(corte, color="k", ls="--", lw=0.8)
        ax.set(title="Espectro promedio de cada ave (líneas punteadas = cortes elegidos)",
               xlabel="Frecuencia [Hz]", ylabel="||X(k)||", xlim=(0, frec[-1]))
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(carpeta_salida, "espectro_promedio_aves.png"), dpi=130)
        plt.close(fig)
        print("Superpuestas -> graficas/espectro_promedio_aves.png")


if __name__ == "__main__":
    generar_informe()
