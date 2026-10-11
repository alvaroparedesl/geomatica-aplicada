"""
Acceso a los datos preparados del taller (`talleres/fair-series-temporales/datos/`).

Los archivos se leen de la carpeta local cuando existe (en Colab, el repositorio
clonado por la celda "Preparar entorno" ya la trae) y, si no está, se descargan
desde GitHub a una carpeta temporal. Así los cuadernos funcionan igual en Colab,
en un computador con el repositorio y en uno sin él.

    from utils.datos_taller import abrir_dato
    ruta = abrir_dato("oisst_concepcion.csv")     # pathlib.Path local
"""

import tempfile
from pathlib import Path

CARPETA_DATOS = Path(__file__).resolve().parent.parent / "talleres" / "fair-series-temporales" / "datos"
URL_DATOS = "https://raw.githubusercontent.com/alvaroparedesl/geomatica-aplicada/main/talleres/fair-series-temporales/datos"
CARPETA_CACHE = Path(tempfile.gettempdir()) / "geomatica-aplicada-datos"


def abrir_dato(nombre, forzar_descarga=False):
    """
    Ruta local de un archivo de `datos/` del taller.

    Si el archivo está en la carpeta del repositorio, devuelve esa ruta. Si no,
    lo descarga de GitHub (`URL_DATOS`) a una carpeta temporal (una sola vez) y
    devuelve la ruta de la copia.

    Args:
        nombre (str): Nombre del archivo, por ejemplo "oisst_centro_sur.nc".
        forzar_descarga (bool): Si es True, ignora la copia local y descarga de nuevo
            (útil para comprobar que el archivo publicado en GitHub es el mismo).

    Returns:
        pathlib.Path: Ruta de un archivo local.
    """
    local = CARPETA_DATOS / nombre
    if local.exists() and not forzar_descarga:
        return local
    copia = CARPETA_CACHE / nombre
    if forzar_descarga or not copia.exists():
        import requests

        respuesta = requests.get(f"{URL_DATOS}/{nombre}", timeout=120)
        respuesta.raise_for_status()
        CARPETA_CACHE.mkdir(parents=True, exist_ok=True)
        copia.write_bytes(respuesta.content)
    return copia
