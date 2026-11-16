"""Interfaz gráfica (CustomTkinter): entrenamiento y reconocimiento en vivo."""
import os
from tkinter import messagebox
import customtkinter as ctk
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import core
import trainer
import recognizer
import utilidades
from preprocesamiento import FS, DURACION, normalizar, filtrar, acondicionar, hay_canto

N_CORTES = 2   # -> 3 subbandas, como en el documento
SEPARACION_MIN_HZ = 100   # distancia mínima entre cortes (y a 0 Hz / fs/2) para no dejar subbandas vacías
VERDE = "#c8f0cf"
CARPETA_IMAGENES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagenes")
TAM_IMAGEN = 140   # px del ave reconocida


def nombre(ave):
    return ave.replace("Carpeta_", "")


def rangos(cortes_hz, fmax):
    """Encabezados de subbanda: 'S1 (0.0–3.5 kHz)', ... La última llega hasta fs porque la
    partición se hace sobre el vector FFT completo (incluye la mitad reflejada)."""
    lim = [0] + sorted(cortes_hz) + [fmax]
    return [f"S{i + 1} ({lim[i] / 1000:.1f}–{lim[i + 1] / 1000:.1f} kHz)" for i in range(len(lim) - 1)]


def imagen_ave(ave):
    """Ruta de la imagen del ave (imagenes/copeton.png, ...) o None si no existe."""
    ruta = os.path.join(CARPETA_IMAGENES, f"{nombre(ave).lower()}.png")
    return ruta if os.path.exists(ruta) else None


def numeros(v):
    return [f"{e:,.2f}" for e in v]


def numeros_cortos(v):
    """Como numeros(), pero en notación científica los valores muy grandes."""
    return [f"{e:.3e}" if abs(e) >= 1e7 else f"{e:,.2f}" for e in v]


def tabla(padre, encabezados, filas, resaltar=None):
    """Tabla con rejilla; la fila `resaltar` se pinta de verde."""
    marco = ctk.CTkFrame(padre, fg_color="gray40", corner_radius=0)
    for j, h in enumerate(encabezados):
        ctk.CTkLabel(marco, text=h, font=ctk.CTkFont(weight="bold", size=12), fg_color="#e8e8e8",
                     text_color="black", corner_radius=0, anchor="center"
                     ).grid(row=0, column=j, padx=1, pady=1, sticky="nsew", ipadx=6, ipady=2)
    for i, fila in enumerate(filas):
        gana = i == resaltar
        for j, v in enumerate(fila):
            ctk.CTkLabel(marco, text=v, text_color="black", corner_radius=0,
                         fg_color=VERDE if gana else "white",
                         font=ctk.CTkFont(weight="bold" if gana else "normal", size=12),
                         anchor="center"
                         ).grid(row=i + 1, column=j, padx=1, pady=1, sticky="nsew", ipadx=6, ipady=1)
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
        self.promedios, self.espectros, self.cortes, self.lineas = {}, {}, [], []

        barra = ctk.CTkFrame(self, fg_color="transparent")
        barra.pack(fill="x", pady=(0, 6))
        ctk.CTkButton(barra, text="Cargar espectros", command=self.cargar).pack(side="left")
        ctk.CTkButton(barra, text="Deshacer corte", fg_color="gray50", command=self.deshacer
                      ).pack(side="left", padx=8)
        self.btn_sugerir = ctk.CTkButton(barra, text="Sugerir cortes", state="disabled",
                                         command=self.sugerir)
        self.btn_sugerir.pack(side="left", padx=(0, 8))
        self.btn_guardar = ctk.CTkButton(barra, text="Entrenar y guardar", state="disabled",
                                         command=self.guardar)
        self.btn_guardar.pack(side="left")
        self.info = ctk.CTkLabel(self, anchor="w", text=(
            f"1) Carga los espectros  2) Clic izquierdo en la gráfica para marcar {N_CORTES} cortes "
            "(clic derecho deshace) o usa «Sugerir cortes»  3) Entrena y guarda. Usa la lupa de la barra para hacer zoom."))
        self.info.pack(fill="x")

        zona = ctk.CTkFrame(self, fg_color="transparent")
        zona.pack(fill="both", expand=True)
        self.ax, self.canvas, self.toolbar = crear_grafica(zona)
        self.canvas.mpl_connect("button_press_event", self.clic)
        self.resultado = ctk.CTkFrame(self, fg_color="transparent")
        self.resultado.pack(fill="x", pady=(6, 0))

    def cargar(self):
        try:
            self.fs, self.frec, self.espectros = trainer.cargar_espectros()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        self.promedios = trainer.promedios_de(self.espectros)
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
        self.btn_sugerir.configure(state="normal")
        self.actualizar()

    def clic(self, ev):
        modo = getattr(self.toolbar.mode, "value", self.toolbar.mode)
        if modo or ev.inaxes is not self.ax or not self.promedios:   # zoom/pan activo
            return
        if ev.button == 1 and len(self.cortes) < N_CORTES:
            x = float(ev.xdata)
            limites = [0.0, self.fs / 2] + self.cortes
            if min(abs(x - c) for c in limites) < SEPARACION_MIN_HZ:
                self.info.configure(text=f"Corte en {x:.0f} Hz ignorado: debe estar a más de "
                                         f"{SEPARACION_MIN_HZ} Hz de otro corte y de los extremos.")
                return
            self.cortes.append(x)
            self.lineas.append(self.ax.axvline(x, color="k", ls="--", lw=1))
            self.canvas.draw_idle()
        elif ev.button == 3:
            self.deshacer()
        self.actualizar()

    def sugerir(self):
        """Marca los cortes que mejor separan las aves (trainer.sugerir_cortes)."""
        self.info.configure(text="Buscando la mejor partición...")
        self.update_idletasks()
        cortes, acierto = trainer.sugerir_cortes(self.frec, self.espectros, N_CORTES)
        while self.cortes:
            self.cortes.pop()
            self.lineas.pop().remove()
        for c in cortes:
            self.cortes.append(c)
            self.lineas.append(self.ax.axvline(c, color="k", ls="--", lw=1))
        self.canvas.draw_idle()
        self.actualizar()
        self.info.configure(text=self.info.cget("text") + f"  (sugeridos: {100 * acierto:.0f} % de acierto "
                                                         "con los audios de entrenamiento)")

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
        try:
            vectores = trainer.guardar_modelo(self.fs, self.frec, self.espectros, self.cortes)
        except ValueError as e:
            messagebox.showerror("Cortes no válidos", f"{e}\n\nDeshaz el corte y marca otro.")
            return
        for w in self.resultado.winfo_children():
            w.destroy()
        izq = ctk.CTkFrame(self.resultado, fg_color="transparent")
        izq.pack(side="left", anchor="n")
        der = ctk.CTkFrame(self.resultado, fg_color="transparent")
        der.pack(side="left", anchor="n", padx=(24, 0))

        # Comparación de los vectores promedio (como la tabla del documento)
        ctk.CTkLabel(izq, anchor="w", text=f"Comparación de los vectores promedio (guardados en {trainer.RUTA_MODELO})",
                     font=ctk.CTkFont(weight="bold")).pack(fill="x")
        aciertos = trainer.evaluar(self.frec, self.espectros, self.cortes)
        filas = [[f"E_{nombre(a)}"] + numeros(v) + [f"{100 * aciertos[a]:.0f} %"] for a, v in vectores.items()]
        tabla(izq, ["Vector"] + rangos(self.cortes, self.fs) + ["Acierto entrenamiento"], filas).pack(anchor="w")

        # Qué tan diferentes quedaron: distancia euclidiana entre cada par de vectores
        ctk.CTkLabel(der, anchor="w", text="Distancia entre vectores promedio",
                     font=ctk.CTkFont(weight="bold")).pack(fill="x")
        aves = list(vectores)
        filas = [[f"||E_{nombre(a)} − E_{nombre(b)}||", f"{core.distancia_euclidiana(vectores[a], vectores[b]):,.2f}"]
                 for i, a in enumerate(aves) for b in aves[i + 1:]]
        tabla(der, ["Par", "Distancia"], filas).pack(anchor="w")


class PestanaReconocimiento(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self.cache = {}     # ave -> (frec, ||C(k)||_m)
        self.imagenes = {}  # ave -> CTkImage
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
        cabecera = ctk.CTkFrame(izq, fg_color="transparent")
        cabecera.pack(anchor="w", fill="x")
        self.foto = ctk.CTkLabel(cabecera, text="")
        self.foto.pack(side="left", padx=(0, 12))
        textos = ctk.CTkFrame(cabecera, fg_color="transparent")
        textos.pack(side="left", fill="y")
        self.titulo = ctk.CTkLabel(textos, text="", font=ctk.CTkFont(size=28, weight="bold"))
        self.titulo.pack(anchor="w")
        self.subtitulo = ctk.CTkLabel(textos, text="", anchor="w")
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
        try:
            recognizer.guardar_grabacion(x)
        except OSError:
            pass   # no poder guardar la copia no debe impedir el reconocimiento
        self.btn.configure(state="normal")
        self.progreso.set(1)
        if abs(x).max() < 1e-3:
            self.estado.configure(text="No se detectó sonido")
            return
        if not hay_canto(x):
            self.estado.configure(text="No se detectó canto")
            self.sin_canto()
            return
        self.estado.configure(text="Listo")
        x = acondicionar(filtrar(normalizar(x)))
        try:
            cortes, E_promedio = recognizer.cargar_modelo()
        except ValueError as e:
            messagebox.showerror("Modelo no válido", str(e))
            return
        ganador, d, E_x = recognizer.reconocer(x, cortes, E_promedio)
        self.mostrar_tablas(cortes, E_promedio, d, E_x)
        self.titulo.configure(text=nombre(ganador))
        self.mostrar_imagen(ganador)
        self.subtitulo.configure(text=f"Menor distancia euclidiana: {d[ganador]:,.2f}")
        self.graficar(x, ganador, cortes)

    def sin_canto(self):
        """La grabación es solo ruido de fondo: no se reconoce ninguna ave."""
        self.titulo.configure(text="Sin canto")
        self.subtitulo.configure(text="Solo se oyó ruido de fondo. Acerca el micrófono al canto "
                                      "y vuelve a grabar.")
        self.foto.pack_forget()
        for w in self.tablas.winfo_children():
            w.destroy()
        self.ax.clear()
        self.canvas.draw_idle()

    def mostrar_imagen(self, ave):
        """Imagen del ave reconocida al lado del nombre (nada si no hay imagen)."""
        if ave not in self.imagenes:
            ruta = imagen_ave(ave)
            self.imagenes[ave] = ctk.CTkImage(Image.open(ruta), size=(TAM_IMAGEN, TAM_IMAGEN)) if ruta else None
        img = self.imagenes[ave]
        if img:
            self.foto.configure(image=img)
            self.foto.pack(side="left", padx=(0, 12))
        else:
            self.foto.pack_forget()

    def mostrar_tablas(self, cortes, E_promedio, d, E_x):
        aves = sorted(d, key=d.get)   # de menor a mayor distancia: la primera es la reconocida
        for w in self.tablas.winfo_children():
            w.destroy()
        enc = rangos(cortes, FS)
        ctk.CTkLabel(self.tablas, text="Vectores de energía", anchor="w",
                     font=ctk.CTkFont(weight="bold")).pack(fill="x")
        filas = [["E_x (grabación)"] + numeros(E_x)]
        filas += [[f"E_{nombre(a)}"] + numeros(E_promedio[a]) for a in aves]
        tabla(self.tablas, ["Vectores"] + enc, filas).pack(anchor="w", pady=(0, 12))

        ctk.CTkLabel(self.tablas, text="Términos (E_xi − E_ave,i)² y distancia d = ||E_x − E_ave|| = √Σ", anchor="w",
                     font=ctk.CTkFont(weight="bold")).pack(fill="x")
        filas = [[nombre(a)] + numeros_cortos([(ex - ea) ** 2 for ex, ea in zip(E_x, E_promedio[a])])
                 + [f"{d[a]:,.2f}"] for a in aves]
        tabla(self.tablas, ["Ave"] + enc + ["d"], filas, resaltar=0).pack(anchor="w")

    def graficar(self, x, ganador, cortes):
        if ganador not in self.cache:
            _, frec_a, z_a = trainer.cargar_promedio(ganador)
            self.cache[ganador] = (frec_a, z_a)
        frec_a, z_a = self.cache[ganador]
        self.ax.clear()
        self.ax.plot(utilidades.frecuencias(len(x), FS), core.espectro(x), color="tab:gray", lw=0.8,
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