import os
import json
import numpy as np
from scipy.io import wavfile

# Audios ya recortados a 3 s por audioAnalizer.py (audio_3s/<Carpeta_Ave>/*.wav)
carpeta_principal = "audio_3s"
carpetas_aves = ["Carpeta_Copeton", "Carpeta_Bichofue", "Carpeta_Mirla"]
CANTIDAD_SUBBANDAS = 4  # Número de divisiones para los vectores de energía

vectores_umbrales = {}

print("--- INICIANDO FASE DE ENTRENAMIENTO ---")

for nombre_ave in carpetas_aves:
    ruta_carpeta = os.path.join(carpeta_principal, nombre_ave)
    if not os.path.exists(ruta_carpeta):
        print(f"Advertencia: no existe '{ruta_carpeta}'. Ejecuta primero audioAnalizer.py.")
        continue

    espectros_acumulados = []

    for archivo in os.listdir(ruta_carpeta):
        if archivo.endswith(".wav"):
            ruta_archivo = os.path.join(ruta_carpeta, archivo)
            try:
                sr, audio_data = wavfile.read(ruta_archivo)
                x = audio_data.astype(np.float32) / 32768.0
                y = np.fft.rfft(x)       # FFT (mitad útil: 0 a sr/2)
                z = np.abs(y)            # magnitud
                espectros_acumulados.append(z)
            except Exception as e:
                print(f"Error procesando {ruta_archivo}: {e}")

    if not espectros_acumulados:
        continue

    # Espectro promedio de la carpeta
    espectro_promedio = np.mean(espectros_acumulados, axis=0)

    # Subbandas y energía E = (1/N) * sum(|X(k)|^2)
    subbandas = np.array_split(espectro_promedio, CANTIDAD_SUBBANDAS)
    energias = [float(1 / len(sb) * np.sum(sb ** 2)) for sb in subbandas]

    vectores_umbrales[nombre_ave] = energias
    print(f"{nombre_ave}: {len(espectros_acumulados)} audios -> "
          f"energías = {[round(e, 4) for e in energias]}")

# Guardar los vectores de umbrales de energía en un archivo plano (JSON)
os.makedirs("models", exist_ok=True)
ruta_modelo = os.path.join("models", "umbrales_energia.json")
with open(ruta_modelo, "w") as f:
    json.dump(vectores_umbrales, f, indent=4)

print(f"\nEntrenamiento finalizado. Vectores de energía guardados exitosamente en '{ruta_modelo}'.")