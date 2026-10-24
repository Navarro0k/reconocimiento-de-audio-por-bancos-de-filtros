import os
from mutagen.mp3 import MP3

# 1. Definir la carpeta principal y las subcarpetas de las aves
carpeta_principal = "audio"
carpetas_aves = ["Carpeta_Copeton", "Carpeta_Bichofue", "Carpeta_Mirla"]
duraciones = []

for nombre_carpeta in carpetas_aves:
    # 2. Construir la nueva ruta (ej. audio/Carpeta_Copeton)
    ruta_carpeta = os.path.join(carpeta_principal, nombre_carpeta)
    
    if not os.path.exists(ruta_carpeta):
        print(f"Advertencia: La carpeta '{ruta_carpeta}' no existe. Saltando...")
        continue
        
    for archivo in os.listdir(ruta_carpeta):
        if archivo.endswith(".mp3"):
            ruta_archivo = os.path.join(ruta_carpeta, archivo)
            try:
                audio = MP3(ruta_archivo)
                duraciones.append(audio.info.length)
            except Exception as e:
                print(f"Error al leer {ruta_archivo}: {e}")

if duraciones:
    duracion_minima = min(duraciones)
    duracion_maxima = max(duraciones)
    duracion_promedio = sum(duraciones) / len(duraciones)

    print("\n--- ANÁLISIS DE DURACIÓN DE AUDIOS ---")
    print(f"Total de audios procesados : {len(duraciones)}")
    print(f"Duración Mínima            : {duracion_minima:.2f} segundos")
    print(f"Duración Máxima            : {duracion_maxima:.2f} segundos")
    print(f"Duración Promedio          : {duracion_promedio:.2f} segundos")
else:
    print("No se encontraron audios MP3 para analizar en la carpeta 'audio'.")