"""Funciones de apoyo para datos vectoriales (GeoPandas)."""

import tempfile
from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import requests

# Espejo oficial de Natural Earth en Amazon S3
_URL_NATURAL_EARTH = "https://naturalearth.s3.amazonaws.com/{escala}_{categoria}/ne_{escala}_{tema}.zip"


def descargar_natural_earth(escala="110m", tema="admin_0_countries", categoria="cultural"):
    """
    Descarga una capa de Natural Earth y la devuelve como GeoDataFrame.

    La primera vez se descarga y se guarda en una carpeta temporal; las
    siguientes veces se lee desde ahí, sin volver a descargar.

    Args:
        escala (str): "110m" (poco detalle), "50m" o "10m" (mucho detalle).
        tema (str): Nombre de la capa, por ejemplo "admin_0_countries"
            (países) o "admin_1_states_provinces" (regiones/estados).
        categoria (str): "cultural" o "physical".

    Returns:
        geopandas.GeoDataFrame: Capa en EPSG:4326.
    """
    nombre = f"ne_{escala}_{tema}"
    carpeta = Path(tempfile.gettempdir()) / "natural_earth_data" / nombre
    archivo_shp = carpeta / f"{nombre}.shp"

    if not archivo_shp.exists():
        url = _URL_NATURAL_EARTH.format(escala=escala, categoria=categoria, tema=tema)
        print(f"Descargando {nombre} desde Natural Earth...")
        respuesta = requests.get(url, timeout=60)
        respuesta.raise_for_status()

        carpeta.mkdir(parents=True, exist_ok=True)
        zip_local = carpeta / f"{nombre}.zip"
        zip_local.write_bytes(respuesta.content)
        with ZipFile(zip_local) as zip_file:
            zip_file.extractall(carpeta)

    return gpd.read_file(archivo_shp)


def calcular_area(gdf, unidad="km2"):
    """
    Calcula el área de cada geometría con una proyección de igual área.

    Un CRS geográfico (grados) no sirve para medir áreas, así que la función
    reproyecta temporalmente a "cylindrical equal area" (+proj=cea) y luego
    convierte m² a la unidad pedida. El GeoDataFrame original no se modifica.

    Args:
        gdf (geopandas.GeoDataFrame): Capa de polígonos con CRS definido.
        unidad (str): "m2", "ha" (hectáreas) o "km2".

    Returns:
        pandas.Series: Área de cada geometría en la unidad pedida.
    """
    divisores = {"m2": 1, "ha": 10_000, "km2": 1_000_000}
    if unidad not in divisores:
        raise ValueError(f"Unidad '{unidad}' no soportada. Use 'm2', 'ha' o 'km2'.")
    if gdf.crs is None:
        raise ValueError("El GeoDataFrame no tiene CRS definido.")

    area_m2 = gdf.to_crs("+proj=cea").area
    return area_m2 / divisores[unidad]
