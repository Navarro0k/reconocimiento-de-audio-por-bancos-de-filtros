import os
import shutil
import subprocess
import numpy as np
from scipy.io import wavfile
from mutagen.mp3 import MP3

# Parámetros del recorte (paso 4)
FS = 22050              # frecuencia de muestreo común (Nyquist = 11025 Hz)
DURACION = 3.0          # segundos
N = int(FS * DURACION)  # muestras por audio recortado
carpeta_salida = "audio_3s"

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
    try:
        respuesta = input("¿Deseas eliminar estos archivos permanentemente? (s/n): ").strip().lower()
    except EOFError:  # ejecución sin teclado (p. ej. en un script automático)
        respuesta = "n"
    
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


# 4. Recorte a exactamente 3 segundos
#    - Los originales de 'audio/' NO se modifican.
#    - De cada audio se toma la ventana de 3 s con mayor energía (evita silencios
#      y las locuciones del inicio de muchas grabaciones).
#    - Resultado: audio_3s/<Carpeta_Ave>/<archivo>.wav (mono, FS Hz, normalizado).
def buscar_ffmpeg():
    """Devuelve la ruta de ffmpeg: el del sistema o el que trae imageio-ffmpeg."""
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit(
            "No se encontró ffmpeg. Instálalo con:\n"
            "    pip install imageio-ffmpeg\n"
            "y vuelve a ejecutar este script.")


FFMPEG = buscar_ffmpeg()


def cargar_audio(ruta):
    """Decodifica el mp3 a mono float32 a FS Hz usando ffmpeg."""
    cmd = [FFMPEG, "-v", "error", "-i", ruta, "-f", "f32le",
           "-ac", "1", "-ar", str(FS), "-"]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0 or len(res.stdout) == 0:
        raise ValueError(res.stderr.decode(errors="ignore").strip() or "vacío")
    return np.frombuffer(res.stdout, dtype=np.float32)


def mejor_ventana(x):
    """Devuelve las N muestras consecutivas con mayor energía (paso de 0.25 s)."""
    paso = FS // 4
    acum = np.concatenate(([0.0], np.cumsum(x.astype(np.float64) ** 2)))
    mejor, ini_mejor = -1.0, 0
    for ini in range(0, len(x) - N + 1, paso):
        e = acum[ini + N] - acum[ini]
        if e > mejor:
            mejor, ini_mejor = e, ini
    return x[ini_mejor:ini_mejor + N]


print("\n--- RECORTE A 3 SEGUNDOS ---")
for nombre_carpeta in carpetas_aves:
    ruta_carpeta = os.path.join(carpeta_principal, nombre_carpeta)
    if not os.path.exists(ruta_carpeta):
        continue
    destino = os.path.join(carpeta_salida, nombre_carpeta)
    os.makedirs(destino, exist_ok=True)
    ok = descartados = 0

    for archivo in sorted(os.listdir(ruta_carpeta)):
        if not archivo.endswith(".mp3"):
            continue
        ruta_archivo = os.path.join(ruta_carpeta, archivo)
        if ruta_archivo in archivos_corruptos:
            descartados += 1
            continue
        try:
            x = cargar_audio(ruta_archivo)
        except Exception as e:
            print(f"  [no se pudo decodificar] {archivo}: {e}")
            descartados += 1
            continue
        pico = np.max(np.abs(x)) if len(x) else 0.0
        if len(x) < N or pico < 1e-4:   # muy corto o silencio total
            descartados += 1
            continue
        seg = mejor_ventana(x)
        seg = seg / max(np.max(np.abs(seg)), 1e-9) * 0.9
        wavfile.write(os.path.join(destino, archivo.replace(".mp3", ".wav")),
                      FS, (seg * 32767).astype(np.int16))
        ok += 1

    print(f"{nombre_carpeta}: {ok} recortados | {descartados} descartados")