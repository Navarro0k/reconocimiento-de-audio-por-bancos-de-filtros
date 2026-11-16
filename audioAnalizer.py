import os
import shutil
import subprocess
import numpy as np
from scipy.io import wavfile
from mutagen.mp3 import MP3
from filtro import FS, N, filtrar, normalizar

CARPETA_ORIGEN = "audio"
CARPETA_DESTINO = "audio_3s"
AVES = ["Carpeta_Copeton", "Carpeta_Mirla"]

def buscar_ffmpeg():
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("No se encontro ffmpeg. Instala imageio-ffmpeg.")

FFMPEG = buscar_ffmpeg()

def duracion_de(ruta):
    return MP3(ruta).info.length

def cargar_audio(ruta):
    comando = [FFMPEG, "-v", "error", "-i", ruta, "-f", "f32le", "-ac", "1", "-ar", str(FS), "-"]
    resultado = subprocess.run(comando, capture_output=True)
    if resultado.returncode != 0 or not resultado.stdout:
        raise ValueError(resultado.stderr.decode(errors="ignore").strip() or "vacio")
    return np.frombuffer(resultado.stdout, dtype=np.float32)

def analizar_duraciones():
    duraciones = []
    corruptos = []
    for ave in AVES:
        ruta = os.path.join(CARPETA_ORIGEN, ave)
        if not os.path.isdir(ruta):
            print(f"Advertencia: no existe {ruta}")
            continue
        for archivo in os.listdir(ruta):
            if not archivo.endswith(".mp3"):
                continue
            ruta_archivo = os.path.join(ruta, archivo)
            try:
                duracion = duracion_de(ruta_archivo)
                duraciones.append(duracion)
                print(f"{ave} | {archivo} -> {duracion:.2f} s")
            except Exception:
                print(f"Corrupto: {ruta_archivo}")
                corruptos.append(ruta_archivo)
    if duraciones:
        print(f"Total: {len(duraciones)} | min {min(duraciones):.2f} | max {max(duraciones):.2f} | prom {sum(duraciones) / len(duraciones):.2f}")
    else:
        print("No se encontraron MP3 validos.")
    if corruptos:
        try:
            respuesta = input("Eliminar corruptos? (s/n): ").strip().lower()
        except EOFError:
            respuesta = "n"
        if respuesta in ("s", "si"):
            for ruta in corruptos:
                try:
                    os.remove(ruta)
                    print(f"Eliminado {ruta}")
                except OSError as error:
                    print(f"Error al borrar {ruta}: {error}")
    return corruptos

def recortar_todo(corruptos):
    for ave in AVES:
        origen = os.path.join(CARPETA_ORIGEN, ave)
        destino = os.path.join(CARPETA_DESTINO, ave)
        if not os.path.isdir(origen):
            continue
        os.makedirs(destino, exist_ok=True)
        ok = 0
        descartados = 0
        for archivo in sorted(os.listdir(origen)):
            if not archivo.endswith(".mp3"):
                continue
            ruta_archivo = os.path.join(origen, archivo)
            if ruta_archivo in corruptos:
                descartados += 1
                continue
            try:
                x = cargar_audio(ruta_archivo)
            except ValueError as error:
                print(f"No se pudo decodificar {archivo}: {error}")
                descartados += 1
                continue
            if len(x) < N or np.max(np.abs(x)) < 1e-4:
                descartados += 1
                continue
            seg = filtrar(normalizar(x[:N]))
            wavfile.write(os.path.join(destino, archivo.replace(".mp3", ".wav")), FS, (seg * 32767).astype(np.int16))
            ok += 1
        print(f"{ave}: {ok} recortados | {descartados} descartados")

corruptos = analizar_duraciones()
recortar_todo(corruptos)