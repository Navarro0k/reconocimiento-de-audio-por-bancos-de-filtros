"""Punto de entrada y controladores de la aplicación."""
import os
from tkinter import messagebox

import matplotlib
matplotlib.use("TkAgg")

import customtkinter as ctk

import recognizer
import trainer
import utilidades
from filtro import FS, DURACION, filtrar, normalizar
from appvista import VistaEntrenamiento, VistaReconocimiento

N_CORTES = 2   # -> 3 subbandas, como en el documento
SEPARACION_MIN_HZ = 100   # distancia mínima entre cortes (y a 0 Hz / fs/2)


class ControlEntrenamiento:
    """Conecta la pestaña de entrenamiento con la lógica de trainer."""

    def __init__(self):
        self.vista = None
        self.frecuencia_muestreo = None
        self.frecuencias = None
        self.espectros = {}

    def cargar_espectros(self):
        """Carga los audios de entrenamiento y muestra sus espectros promedio."""
        try:
            self.frecuencia_muestreo, self.frecuencias, self.espectros = trainer.cargar_espectros()
        except Exception as error:
            messagebox.showerror("Error", str(error))
            return
        promedios = trainer.calcular_espectros_promedio(self.espectros)
        self.vista.mostrar_espectros_promedio(self.frecuencias, promedios)

    def marcar_corte(self, frecuencia):
        """Valida que el corte esté separado de los demás y lo marca en la gráfica."""
        limites = [0.0, self.frecuencia_muestreo / 2] + self.vista.cortes
        if min(abs(frecuencia - corte) for corte in limites) < SEPARACION_MIN_HZ:
            self.vista.avisar_corte_invalido(frecuencia, SEPARACION_MIN_HZ)
            return
        self.vista.agregar_corte(frecuencia)

    def entrenar(self):
        """Calcula los vectores promedio de cada ave y guarda el modelo."""
        try:
            vectores = trainer.entrenar(self.frecuencia_muestreo, self.frecuencias,
                                        self.espectros, self.vista.cortes)
        except ValueError as error:
            messagebox.showerror("Cortes no válidos", f"{error}\n\nDeshaz el corte y marca otro.")
            return
        self.vista.mostrar_vectores(self.vista.cortes, self.frecuencia_muestreo,
                                    vectores, trainer.RUTA_MODELO)


class ControlReconocimiento:
    """Conecta la pestaña de reconocimiento con recognizer y trainer."""

    def __init__(self):
        self.vista = None
        self.buffer = None
        self.cache = {}

    def grabar(self):
        """Inicia la grabación del micrófono."""
        if not os.path.exists(trainer.RUTA_MODELO):
            messagebox.showerror("Error", "No hay modelo. Entrena primero en la otra pestaña.")
            return
        try:
            self.buffer = recognizer.iniciar_grabacion()
        except Exception as error:
            messagebox.showerror("Error de micrófono", f"{error}\n\nInstala: pip install sounddevice")
            return
        self.vista.iniciar_grabacion()

    def terminar(self):
        """Procesa la grabación y muestra el ave reconocida."""
        senal = recognizer.terminar_grabacion(self.buffer)
        try:
            recognizer.guardar_grabacion(senal)
        except OSError:
            pass
        self.vista.finalizar_grabacion()

        if abs(senal).max() < 1e-3:
            self.vista.mostrar_sin_sonido()
            return

        senal = filtrar(normalizar(senal))
        try:
            cortes, promedios = recognizer.cargar_modelo()
        except ValueError as error:
            messagebox.showerror("Modelo no válido", str(error))
            return

        ganador, distancias, energia = recognizer.reconocer(senal, cortes, promedios)
        frecuencias_senal = utilidades.frecuencias(len(senal), FS)
        frecuencias_promedio, promedio = self.promedio_del_ave(ganador)
        self.vista.mostrar_resultado(senal, frecuencias_senal, ganador, distancias, energia,
                                     cortes, promedios, frecuencias_promedio, promedio)

    def promedio_del_ave(self, ave):
        """Devuelve el espectro promedio del ave y lo conserva en caché."""
        if ave not in self.cache:
            _, frecuencias, promedio = trainer.cargar_espectro_promedio(ave)
            self.cache[ave] = (frecuencias, promedio)
        return self.cache[ave]


def main():
    """Construye la ventana y conecta cada vista con su controlador."""
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    raiz = ctk.CTk()
    raiz.title("Reconocimiento de aves por banco de filtros")
    raiz.geometry("1300x800")
    pestanas = ctk.CTkTabview(raiz)
    pestanas.pack(fill="both", expand=True, padx=10, pady=10)

    control_entrenamiento = ControlEntrenamiento()
    vista_entrenamiento = VistaEntrenamiento(
        pestanas.add("Entrenamiento"), N_CORTES,
        al_cargar=control_entrenamiento.cargar_espectros,
        al_entrenar=control_entrenamiento.entrenar,
        al_marcar_corte=control_entrenamiento.marcar_corte)
    control_entrenamiento.vista = vista_entrenamiento
    vista_entrenamiento.pack(fill="both", expand=True)

    control_reconocimiento = ControlReconocimiento()
    vista_reconocimiento = VistaReconocimiento(
        pestanas.add("Reconocimiento"), DURACION, FS,
        al_grabar=control_reconocimiento.grabar,
        al_terminar=control_reconocimiento.terminar)
    control_reconocimiento.vista = vista_reconocimiento
    vista_reconocimiento.pack(fill="both", expand=True)

    raiz.mainloop()


if __name__ == "__main__":
    main()
