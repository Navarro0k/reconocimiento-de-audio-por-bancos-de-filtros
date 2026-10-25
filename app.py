"""Interfaz gráfica (CustomTkinter): entrenamiento y reconocimiento en vivo."""
import os
from tkinter import messagebox
import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import core
import trainer
import recognizer
from preprocesamiento import FS, DURACION, normalizar, filtrar

N_CORTES = 3   # -> 4 subbandas
VERDE = "#c8f0cf"


def nombre(ave):
    return ave.replace("Carpeta_", "")


def rangos(cortes_hz, fmax):
    """Encabezados de subbanda: 'S1\\n0.0-3.5 kHz', ..."""
    lim = [0] + sorted(cortes_hz) + [fmax]
    return [f"S{i + 1}\n{lim[i] / 1000:.1f}–{lim[i + 1] / 1000:.1f} kHz" for i in range(len(lim) - 1)]


def numeros(v):
    return [f"{e:,.2f}" for e in v]


def tabla(padre, encabezados, filas, resaltar=None):
    """Tabla de etiquetas; la fila `resaltar` se pinta de verde."""
    marco = ctk.CTkFrame(padre)
    for j, h in enumerate(encabezados):
        ctk.CTkLabel(marco, text=h, font=ctk.CTkFont(weight="bold"), justify="right",
                     anchor="w" if j == 0 else "e").grid(row=0, column=j, padx=6, pady=3, sticky="ew")
    for i, fila in enumerate(filas):
        gana = i == resaltar
        for j, v in enumerate(fila):
            ctk.CTkLabel(marco, text=v, text_color="black" if gana else None,
                         fg_color=VERDE if gana else "transparent", corner_radius=4,
                         font=ctk.CTkFont(weight="bold" if gana else "normal"),
                         anchor="w" if j == 0 else "e").grid(row=i + 1, column=j, padx=2, pady=1, sticky="ew")
    return marco


def crear_grafica(padre):
    """Figura de matplotlib con su barra de herramientas (zoom, pan, home)."""
    fig = Figure(figsize=(7, 4.2), layout="constrained")
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, padre)
    barra = NavigationToolbar2Tk(canvas, padre, pack_toolbar=False)
    barra.update()
    barra.pack(side="bottom", fill="x")
    canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
    return ax, canvas, barra


class PestanaEntrenamiento(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self.promedios, self.cortes, self.lineas = {}, [], []

        barra = ctk.CTkFrame(self, fg_color="transparent")
        barra.pack(fill="x", pady=(0, 6))
        ctk.CTkButton(barra, text="Cargar espectros", command=self.cargar).pack(side="left")
        ctk.CTkButton(barra, text="Deshacer corte", fg_color="gray50", command=self.deshacer
                      ).pack(side="left", padx=8)
        self.btn_guardar = ctk.CTkButton(barra, text="Entrenar y guardar", state="disabled",
                                         command=self.guardar)
        self.btn_guardar.pack(side="left")
        self.info = ctk.CTkLabel(self, anchor="w", text=(
            f"1) Carga los espectros  2) Clic izquierdo en la gráfica para marcar {N_CORTES} cortes "
            "(clic derecho deshace)  3) Entrena y guarda. Usa la lupa de la barra para hacer zoom."))
        self.info.pack(fill="x")

        zona = ctk.CTkFrame(self, fg_color="transparent")
        zona.pack(fill="both", expand=True)
        self.ax, self.canvas, self.toolbar = crear_grafica(zona)
        self.canvas.mpl_connect("button_press_event", self.clic)
        self.resultado = ctk.CTkFrame(self, fg_color="transparent")
        self.resultado.pack(fill="x", pady=(6, 0))

    def cargar(self):
        try:
            self.fs, self.frec, self.promedios = trainer.cargar_promedios()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        self.cortes, self.lineas = [], []
        self.ax.clear()
        for ave, z in self.promedios.items():
            self.ax.plot(self.frec, z, color=trainer.COLORES[ave], lw=1, label=nombre(ave))
        tope = max(z[self.frec > 100].max() for z in self.promedios.values())
        self.ax.set(xlabel="Frecuencia [Hz]", ylabel="||C(k)||_m", title="Espectros promedio",
                    xlim=(0, self.frec[-1]), ylim=(0, 4 * tope))
        self.ax.legend(loc="upper right")
        self.canvas.draw_idle()
        self.toolbar.update()          # "home" vuelve a esta vista
        self.actualizar()

    def clic(self, ev):
        modo = getattr(self.toolbar.mode, "value", self.toolbar.mode)
        if modo or ev.inaxes is not self.ax or not self.promedios:   # zoom/pan activo
            return
        if ev.button == 1 and len(self.cortes) < N_CORTES:
            self.cortes.append(float(ev.xdata))
            self.lineas.append(self.ax.axvline(ev.xdata, color="k", ls="--", lw=1))
            self.canvas.draw_idle()
        elif ev.button == 3:
            self.deshacer()
        self.actualizar()

    def deshacer(self):
        if self.cortes:
            self.cortes.pop()
            self.lineas.pop().remove()
            self.canvas.draw_idle()
            self.actualizar()

    def actualizar(self):
        self.btn_guardar.configure(state="normal" if len(self.cortes) == N_CORTES else "disabled")
        hz = ", ".join(f"{c:.0f} Hz" for c in sorted(self.cortes))
        self.info.configure(text=f"Cortes {len(self.cortes)}/{N_CORTES}: {hz}")

    def guardar(self):
        vectores = trainer.guardar_modelo(self.fs, self.frec, self.promedios, self.cortes)
        for w in self.resultado.winfo_children():
            w.destroy()
        ctk.CTkLabel(self.resultado, anchor="w", text=f"Modelo guardado en {trainer.RUTA_MODELO} "
                     "— vector de energías de cada ave:").pack(fill="x")
        filas = [[nombre(a)] + numeros(v) for a, v in vectores.items()]
        tabla(self.resultado, ["Ave"] + rangos(self.cortes, self.fs / 2), filas).pack(anchor="w")


class PestanaReconocimiento(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self.cache = {}     # ave -> (frec, ||C(k)||_m)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        barra = ctk.CTkFrame(self, fg_color="transparent")
        barra.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self.btn = ctk.CTkButton(barra, text=f"Grabar {DURACION:.0f} s y reconocer", command=self.grabar)
        self.btn.pack(side="left")
        self.progreso = ctk.CTkProgressBar(barra, width=200)
        self.progreso.set(0)
        self.progreso.pack(side="left", padx=12)
        self.estado = ctk.CTkLabel(barra, text="Listo")
        self.estado.pack(side="left")

        izq = ctk.CTkFrame(self, fg_color="transparent")
        izq.grid(row=1, column=0, sticky="ns", padx=(0, 10))
        self.titulo = ctk.CTkLabel(izq, text="", font=ctk.CTkFont(size=28, weight="bold"))
        self.titulo.pack(anchor="w")
        self.subtitulo = ctk.CTkLabel(izq, text="", anchor="w")
        self.subtitulo.pack(anchor="w", pady=(0, 8))
        self.tablas = ctk.CTkFrame(izq, fg_color="transparent")
        self.tablas.pack(anchor="w")

        der = ctk.CTkFrame(self, fg_color="transparent")
        der.grid(row=1, column=1, sticky="nsew")
        self.ax, self.canvas, self.toolbar = crear_grafica(der)

    # --- grabación (sin bloquear la ventana) ---
    def grabar(self):
        if not os.path.exists(trainer.RUTA_MODELO):
            messagebox.showerror("Error", "No hay modelo. Entrena primero en la otra pestaña.")
            return
        try:
            self.buffer = recognizer.iniciar_grabacion()
        except Exception as e:
            messagebox.showerror("Error de micrófono", f"{e}\n\nInstala: pip install sounddevice")
            return
        self.btn.configure(state="disabled")
        self.avance(0)

    def avance(self, tick):
        total = int(DURACION * 10)
        if tick < total:
            self.progreso.set(tick / total)
            self.estado.configure(text=f"Grabando... {DURACION - tick / 10:.1f} s")
            self.after(100, self.avance, tick + 1)
        else:
            self.after(300, self.terminar)

    # --- reconocimiento ---
    def terminar(self):
        x = recognizer.terminar_grabacion(self.buffer)
        self.btn.configure(state="normal")
        self.progreso.set(1)
        if abs(x).max() < 1e-3:
            self.estado.configure(text="No se detectó sonido")
            return
        self.estado.configure(text="Listo")
        x = filtrar(normalizar(x))
        cortes, umbrales = recognizer.cargar_modelo()
        resultados, E_X = recognizer.reconocer(x, cortes, umbrales)
        ganador, _, suma = resultados[0]
        self.mostrar_tablas(cortes, umbrales, resultados, E_X)
        self.titulo.configure(text=nombre(ganador))
        self.subtitulo.configure(text=f"Menor suma de diferencias absolutas: {suma:,.2f}")
        self.graficar(x, ganador, cortes)

    def mostrar_tablas(self, cortes, umbrales, resultados, E_X):
        for w in self.tablas.winfo_children():
            w.destroy()
        enc = rangos(cortes, FS / 2)
        ctk.CTkLabel(self.tablas, text="Vectores de energía", anchor="w",
                     font=ctk.CTkFont(weight="bold")).pack(fill="x")
        filas = [["Grabación X"] + numeros(E_X)]
        filas += [[f"Umbral {nombre(a)}"] + numeros(umbrales[a]) for a, _, _ in resultados]
        tabla(self.tablas, [""] + enc, filas).pack(anchor="w", pady=(0, 12))

        ctk.CTkLabel(self.tablas, text="Diferencias absolutas |E_Xi − E_ave,i|", anchor="w",
                     font=ctk.CTkFont(weight="bold")).pack(fill="x")
        filas = [[nombre(a)] + numeros(d) + [f"{s:,.2f}"] for a, d, s in resultados]
        tabla(self.tablas, ["Ave"] + enc + ["Σ"], filas, resaltar=0).pack(anchor="w")

    def graficar(self, x, ganador, cortes):
        if ganador not in self.cache:
            _, frec_a, z_a = trainer.cargar_promedio(ganador)
            self.cache[ganador] = (frec_a, z_a)
        frec_a, z_a = self.cache[ganador]
        self.ax.clear()
        self.ax.plot(core.frecuencias(len(x), FS), core.espectro(x), color="tab:gray", lw=0.8,
                     label="Grabación X")
        self.ax.plot(frec_a, z_a, color=trainer.COLORES[ganador], lw=1.2,
                     label=f"Promedio {nombre(ganador)} (mínima distancia)")
        for c in cortes:
            self.ax.axvline(c, color="k", ls="--", lw=0.8)
        self.ax.set(xlabel="Frecuencia [Hz]", ylabel="||X(k)||", xlim=(0, FS / 2),
                    title="Espectro de la grabación vs. ave reconocida")
        self.ax.legend(loc="upper right")
        self.canvas.draw_idle()
        self.toolbar.update()


def main():
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    raiz = ctk.CTk()
    raiz.title("Reconocimiento de aves por banco de filtros")
    raiz.geometry("1300x800")
    pestanas = ctk.CTkTabview(raiz)
    pestanas.pack(fill="both", expand=True, padx=10, pady=10)
    for titulo, clase in (("Entrenamiento", PestanaEntrenamiento),
                          ("Reconocimiento", PestanaReconocimiento)):
        clase(pestanas.add(titulo)).pack(fill="both", expand=True)
    raiz.mainloop()


if __name__ == "__main__":
    main()