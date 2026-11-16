import os
import customtkinter as ctk
from PIL import Image
from graficas import crear_grafica, dibujar_corte, dibujar_espectros_promedio, dibujar_reconocimiento

VERDE = "#c8f0cf"
CARPETA_IMAGENES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagenes")
TAM_IMAGEN = 140

def nombre(ave):
    return ave.replace("Carpeta_", "")

def rangos(cortes_hz, fmax):
    lim = [0] + sorted(cortes_hz) + [fmax]
    return [f"S{i + 1} ({lim[i] / 1000:.1f}–{lim[i + 1] / 1000:.1f} kHz)" for i in range(len(lim) - 1)]

def imagen_ave(ave):
    ruta = os.path.join(CARPETA_IMAGENES, f"{nombre(ave).lower()}.png")
    return ruta if os.path.exists(ruta) else None

def numeros(v):
    return [f"{e:,.2f}" for e in v]

def numeros_cortos(v):
    return [f"{e:.3e}" if abs(e) >= 1e7 else f"{e:,.2f}" for e in v]

def tabla(padre, encabezados, filas, resaltar=None):
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

class VistaEntrenamiento(ctk.CTkFrame):

    def __init__(self, master, n_cortes, al_cargar, al_entrenar, al_marcar_corte):
        super().__init__(master, fg_color="transparent")
        self.n_cortes = n_cortes
        self.al_marcar_corte = al_marcar_corte
        self.cortes = []
        self.lineas = []
        self.hay_espectros = False

        barra = ctk.CTkFrame(self, fg_color="transparent")
        barra.pack(fill="x", pady=(0, 6))
        ctk.CTkButton(barra, text="Cargar espectros", command=al_cargar).pack(side="left")
        ctk.CTkButton(barra, text="Deshacer corte", fg_color="gray50", command=self.quitar_corte
                      ).pack(side="left", padx=8)
        self.btn_guardar = ctk.CTkButton(barra, text="Entrenar y guardar", state="disabled",
                                         command=al_entrenar)
        self.btn_guardar.pack(side="left")
        self.info = ctk.CTkLabel(self, anchor="w", text=(
            f"1) Carga los espectros  2) Clic izquierdo en la gráfica para marcar {n_cortes} cortes "
            "(clic derecho deshace)  3) Entrena y guarda. Usa la lupa de la barra para hacer zoom."))
        self.info.pack(fill="x")

        zona = ctk.CTkFrame(self, fg_color="transparent")
        zona.pack(fill="both", expand=True)
        self.ax, self.canvas, self.toolbar = crear_grafica(zona)
        self.canvas.mpl_connect("button_press_event", self.clic)
        self.resultado = ctk.CTkFrame(self, fg_color="transparent")
        self.resultado.pack(fill="x", pady=(6, 0))

    def clic(self, evento):
        modo = getattr(self.toolbar.mode, "value", self.toolbar.mode)
        if modo or evento.inaxes is not self.ax or not self.hay_espectros:
            return
        if evento.button == 1 and len(self.cortes) < self.n_cortes:
            self.al_marcar_corte(float(evento.xdata))
        elif evento.button == 3:
            self.quitar_corte()

    def mostrar_espectros_promedio(self, frecuencias, promedios):
        self.hay_espectros = True
        self.cortes, self.lineas = [], []
        dibujar_espectros_promedio(self.ax, frecuencias, promedios)
        self.canvas.draw_idle()
        self.toolbar.update()
        self.actualizar_estado()

    def agregar_corte(self, frecuencia):
        self.cortes.append(frecuencia)
        self.lineas.append(dibujar_corte(self.ax, frecuencia))
        self.canvas.draw_idle()
        self.actualizar_estado()

    def quitar_corte(self):
        if not self.cortes:
            return
        self.cortes.pop()
        self.lineas.pop().remove()
        self.canvas.draw_idle()
        self.actualizar_estado()

    def avisar_corte_invalido(self, frecuencia, separacion_hz):
        self.info.configure(text=f"Corte en {frecuencia:.0f} Hz ignorado: debe estar a más de "
                                 f"{separacion_hz} Hz de otro corte y de los extremos.")

    def actualizar_estado(self):
        self.btn_guardar.configure(state="normal" if len(self.cortes) == self.n_cortes else "disabled")
        hz = ", ".join(f"{c:.0f} Hz" for c in sorted(self.cortes))
        self.info.configure(text=f"Cortes {len(self.cortes)}/{self.n_cortes}: {hz}")

    def mostrar_vectores(self, cortes, frecuencia_muestreo, vectores, ruta_modelo):
        for w in self.resultado.winfo_children():
            w.destroy()
        izq = ctk.CTkFrame(self.resultado, fg_color="transparent")
        izq.pack(side="left", anchor="n")
        ctk.CTkLabel(izq, anchor="w", text=f"Vectores promedio guardados en {ruta_modelo}",
                     font=ctk.CTkFont(weight="bold")).pack(fill="x")
        filas = [[f"E_{nombre(a)}"] + numeros(v) for a, v in vectores.items()]
        tabla(izq, ["Vector"] + rangos(cortes, frecuencia_muestreo), filas).pack(anchor="w")

class VistaReconocimiento(ctk.CTkFrame):

    def __init__(self, master, duracion, frecuencia_muestreo, al_grabar, al_terminar):
        super().__init__(master, fg_color="transparent")
        self.duracion = duracion
        self.frecuencia_muestreo = frecuencia_muestreo
        self.al_terminar = al_terminar
        self.imagenes = {}
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        barra = ctk.CTkFrame(self, fg_color="transparent")
        barra.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self.btn = ctk.CTkButton(barra, text=f"Grabar {duracion:.0f} s y reconocer",
                                 command=al_grabar)
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

    def iniciar_grabacion(self):
        self.btn.configure(state="disabled")
        self.avance(0)

    def avance(self, tick):
        total = int(self.duracion * 10)
        if tick < total:
            self.progreso.set(tick / total)
            self.estado.configure(text=f"Grabando... {self.duracion - tick / 10:.1f} s")
            self.after(100, self.avance, tick + 1)
        else:
            self.after(300, self.al_terminar)

    def finalizar_grabacion(self):
        self.btn.configure(state="normal")
        self.progreso.set(1)

    def mostrar_sin_sonido(self):
        self.estado.configure(text="No se detectó sonido")

    def mostrar_resultado(self, senal, frecuencias_senal, ganador, distancias, energia,
                          cortes, promedios, frecuencias_promedio, promedio):
        self.estado.configure(text="Listo")
        self.mostrar_tablas(cortes, promedios, distancias, energia)
        self.titulo.configure(text=nombre(ganador))
        self.mostrar_imagen(ganador)
        self.subtitulo.configure(text=f"Menor distancia euclidiana: {distancias[ganador]:,.2f}")
        dibujar_reconocimiento(self.ax, senal, frecuencias_senal, frecuencias_promedio,
                               promedio, ganador, cortes)
        self.canvas.draw_idle()
        self.toolbar.update()

    def mostrar_imagen(self, ave):
        if ave not in self.imagenes:
            ruta = imagen_ave(ave)
            self.imagenes[ave] = ctk.CTkImage(Image.open(ruta), size=(TAM_IMAGEN, TAM_IMAGEN)) if ruta else None
        img = self.imagenes[ave]
        if img:
            self.foto.configure(image=img)
            self.foto.pack(side="left", padx=(0, 12))
        else:
            self.foto.pack_forget()

    def mostrar_tablas(self, cortes, promedios, distancias, energia):
        aves = sorted(distancias, key=distancias.get)
        for w in self.tablas.winfo_children():
            w.destroy()
        enc = rangos(cortes, self.frecuencia_muestreo)
        ctk.CTkLabel(self.tablas, text="Vectores de energía", anchor="w",
                     font=ctk.CTkFont(weight="bold")).pack(fill="x")
        filas = [["E_x (grabación)"] + numeros(energia)]
        filas += [[f"E_{nombre(a)}"] + numeros(promedios[a]) for a in aves]
        tabla(self.tablas, ["Vectores"] + enc, filas).pack(anchor="w", pady=(0, 12))

        ctk.CTkLabel(self.tablas, text="Términos (E_xi − E_ave,i)² y distancia d = ||E_x − E_ave|| = √Σ", anchor="w",
                     font=ctk.CTkFont(weight="bold")).pack(fill="x")
        filas = [[nombre(a)] + numeros_cortos([(ex - ea) ** 2 for ex, ea in zip(energia, promedios[a])])
                 + [f"{distancias[a]:,.2f}"] for a in aves]
        tabla(self.tablas, ["Ave"] + enc + ["d"], filas, resaltar=0).pack(anchor="w")
