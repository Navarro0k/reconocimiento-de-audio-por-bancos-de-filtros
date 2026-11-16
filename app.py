"""Punto de entrada de la aplicación."""
import customtkinter as ctk

from appvista import PestanaEntrenamiento, PestanaReconocimiento


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
