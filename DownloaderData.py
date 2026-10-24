import urllib.request
import urllib.parse
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

# 2. Busca exactamente el mismo nombre definido en el .env (como string)
API_KEY = os.environ.get("API_KEY_XENOCANTO")

if not API_KEY:
    print("Error: No se encontró la API Key")

opener = urllib.request.build_opener()
opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")]
urllib.request.install_opener(opener)

# nombre -> (género, especie)
aves = {
    "Mirla": ("Turdus", "fuscater"),
}

cantidad_descargas = 100

for nombre_ave, (gen, sp) in aves.items():
    carpeta = f"Carpeta_{nombre_ave}"
    os.makedirs(carpeta, exist_ok=True)
    print(f"\nBuscando: {nombre_ave}...")

    descargados = 0
    pagina = 1
    num_paginas = 1

    while descargados < cantidad_descargas and pagina <= num_paginas:
        params = {
            "query": f"gen:{gen} sp:{sp} q:A",   # añade  cnt:colombia  si lo necesitas
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
        print(f"Página {pagina}/{num_paginas} - total en XC: {datos.get('numRecordings')}")
        if not grabaciones:
            break

        for g in grabaciones:
            if descargados >= cantidad_descargas:
                break

            audio_url = g.get("file")
            if not audio_url:
                continue
            if audio_url.startswith("//"):
                audio_url = "https:" + audio_url

            ruta = os.path.join(carpeta, f"{nombre_ave}_XC{g['id']}.mp3")
            if os.path.exists(ruta):          # evita repetir si relanzas el script
                descargados += 1
                continue

            try:
                urllib.request.urlretrieve(audio_url, ruta)
                descargados += 1
                print(f" -> {ruta} ({descargados}/{cantidad_descargas})")
                time.sleep(1)
            except Exception as e:
                print(f" -> Error con XC{g['id']}: {e}")

        pagina += 1

print("\n¡Descarga finalizada!")