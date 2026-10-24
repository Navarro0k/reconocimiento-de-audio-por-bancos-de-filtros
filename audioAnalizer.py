import os
from mutagen.mp3 import MP3

# 1. Definir la carpeta principal y las subcarpetas de las aves
carpeta_principal = "audio"
carpetas_aves = ["Carpeta_Copeton", "Carpeta_Bichofue", "Carpeta_Mirla"]
duraciones = []
archivos_corruptos = [] # Lista para almacenar los archivos fallidos

print("--- DURACIÓN INDIVIDUAL DE AUDIOS ---")

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
                # Forzar lectura para validar que los metadatos existan
                duracion = audio.info.length 
                duraciones.append(duracion)
                
                # NUEVO: Imprimir el nombre del archivo y su duración
                print(f"Carpeta: {nombre_carpeta} | Archivo: {archivo} -> {duracion:.2f} segundos")
                
            except Exception as e:
                # Si falla la lectura, se agrega a la lista de corruptos
                print(f"Detectado archivo corrupto: {ruta_archivo}")
                archivos_corruptos.append(ruta_archivo)

# Mostrar el análisis de tiempos para los audios válidos
if duraciones:
    duracion_minima = min(duraciones)
    duracion_maxima = max(duraciones)
    duracion_promedio = sum(duraciones) / len(duraciones)

    print("\n--- ANÁLISIS GENERAL DE DURACIÓN ---")
    print(f"Total de audios válidos procesados : {len(duraciones)}")
    print(f"Duración Mínima                    : {duracion_minima:.2f} segundos")
    print(f"Duración Máxima                    : {duracion_maxima:.2f} segundos")
    print(f"Duración Promedio                  : {duracion_promedio:.2f} segundos")
else:
    print("\nNo se encontraron audios MP3 válidos para analizar en la carpeta 'audio'.")

# 3. Interacción con el usuario para eliminar archivos corruptos
if archivos_corruptos:
    print(f"\nSe detectaron {len(archivos_corruptos)} archivos corruptos que interrumpirán el cálculo de la FFT.")
    respuesta = input("¿Deseas eliminar estos archivos permanentemente? (s/n): ").strip().lower()
    
    if respuesta == 's' or respuesta == 'si':
        archivos_eliminados = 0
        for ruta in archivos_corruptos:
            try:
                os.remove(ruta)
                archivos_eliminados += 1
            except Exception as e:
                print(f"Error al intentar borrar {ruta}: {e}")
        print(f"Se han eliminado exitosamente {archivos_eliminados} archivos corruptos.")
    else:
        print("Operación cancelada. Los archivos corruptos se conservaron en sus carpetas.")
else:
    print("\nLa base de datos está limpia y no contiene archivos corruptos.")