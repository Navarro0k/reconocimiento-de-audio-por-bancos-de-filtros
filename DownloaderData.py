import urllib.request
import urllib.parse
import json
import os
import time
from dotenv import load_dotenv
from mutagen.mp3 import MP3

load_dotenv()

API_KEY = os.environ.get("API_KEY_XENOCANTO")

if not API_KEY:
    print("Error: No se encontró la API Key")

opener = urllib.request.build_opener()
opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")]
urllib.request.install_opener(opener)

aves = {
    "Mirla": ("Turdus", "fuscater"),
    "Copeton": ("Zonotrichia", "capensis"),
    "Bichofue": ("Pitangus", "sulphuratus"),
}

cantidad_descargas = 100
duracion_minima_requerida = 3.0  # Segundos

for nombre_ave, (gen, sp) in aves.items():
    carpeta = os.path.join("audio", f"Carpeta_{nombre_ave}")
    os.makedirs(carpeta, exist_ok=True)
    print(f"\nAnalizando y procesando: {nombre_ave}...")

    # Conjunto para registrar los IDs de Xeno-canto ya existentes y evitar duplicados
    ids_registrados = set()
    
    # 1. Analizar audios existentes en la carpeta
    for archivo in os.listdir(carpeta):
        if archivo.endswith(".mp3"):
            ruta_existente = os.path.join(carpeta, archivo)
            try:
                audio = MP3(ruta_existente)
                duracion = audio.info.length
                
                # Si dura menos de 3 segundos o está corrupto, se borra
                if duracion < duracion_minima_requerida:
                    print(f" -> Borrando {archivo}: dura {duracion:.2f}s (menor a 3s).")
                    os.remove(ruta_existente)
                else:
                    # Extraer el ID de Xeno-canto del nombre del archivo (ej: Mirla_XC123456.mp3 -> 123456)
                    if "_XC" in archivo:
                        xc_id = archivo.split("_XC")[1].replace(".mp3", "")
                        ids_registrados.add(xc_id)
            except Exception:
                print(f" -> Borrando {archivo}: archivo corrupto.")
                os.remove(ruta_existente)

    # Contar cuántos audios válidos quedaron
    archivos_actuales = [f for f in os.listdir(carpeta) if f.endswith(".mp3")]
    descargados = len(archivos_actuales)
    print(f"Audios válidos actuales: {descargados}/{cantidad_descargas}")

    pagina = 1
    num_paginas = 1

    while descargados < cantidad_descargas and pagina <= num_paginas:
        params = {
            "query": f"gen:{gen} sp:{sp} q:A",
            "per_page": 100,
            "page": pagina,
            "key": API_KEY,
        }
        url_api = "https://xeno-canto.org/api/3/recordings?" + urllib.parse.urlencode(params)

        try:
            with urllib.request.urlopen(url_api) as r:
                datos = json.loads(r.read().decode("utf-8"))
        except Exception as e:
            print(f"Error en la API (página {pagina}): {e}")
            break

        num_paginas = int(datos.get("numPages", 1))
        grabaciones = datos.get("recordings", [])
        
        if not grabaciones:
            break

        for g in grabaciones:
            if descargados >= cantidad_descargas:
                break

            xc_id = str(g.get("id"))
            
            # Evitar repetir audios ya descargados previamente
            if xc_id in ids_registrados:
                continue

            audio_url = g.get("file")
            if not audio_url:
                continue
            if audio_url.startswith("//"):
                audio_url = "https:" + audio_url

            ruta = os.path.join(carpeta, f"{nombre_ave}_XC{xc_id}.mp3")

            try:
                urllib.request.urlretrieve(audio_url, ruta)
                
                # Verificar duración y corrupción inmediatamente tras descargar
                try:
                    audio_test = MP3(ruta)
                    duracion = audio_test.info.length
                    
                    if duracion >= duracion_minima_requerida:
                        ids_registrados.add(xc_id)
                        descargados += 1
                        print(f" -> Guardado y validado: {ruta} ({duracion:.2f}s) [{descargados}/{cantidad_descargas}]")
                    else:
                        print(f" -> Descargado pero descartado: dura {duracion:.2f}s (menor a 3s).")
                        os.remove(ruta)
                except Exception:
                    print(f" -> Archivo corrupto detectado. Descartando...")
                    os.remove(ruta)
                
                time.sleep(0.5)
            except Exception as e:
                print(f" -> Error de descarga con XC{xc_id}: {e}")

        pagina += 1

print("\n¡Proceso de descarga y análisis finalizado!")