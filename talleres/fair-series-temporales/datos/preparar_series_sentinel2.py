"""Prepara las series Sentinel-2 por punto del taller (patrón nativo del curso, paso a paso).

A partir de `puntos_concepcion.csv` construye, para 12 meses (2025-10-01 a
2026-09-30), la serie de cada punto con cuatro variables: NDWI de McFeeters (agua),
NDCI (clorofila), reflectancia roja B04 (turbidez) y NDVI (vegetación). Guarda:

* puntos_concepcion.geojson: los mismos puntos para abrirlos en QGIS.
* series_sentinel2_concepcion.parquet: una fila por punto y fecha, con la fracción
  de nubes de la escena, la fracción de píxeles válidos de la ventana y el offset
  de reflectancia de cada fecha.

Cada paso es la llamada nativa de `pystac_client` / `odc.stac`; los helpers de
`utils/` solo arman sus argumentos (ver `utils/series.py` y `utils/stac.py`). El
offset de reflectancia de Sentinel-2 nunca se fija a mano: lo decide `agregar_offset`
para cada fecha según el catálogo y el baseline de procesamiento.

Uso (necesita internet; tarda unos minutos):
    python preparar_series_sentinel2.py
"""

import sys
import time
from datetime import date
from pathlib import Path

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parents[2]))  # raíz del repositorio, para importar utils


from utils.series import (  # noqa: E402
    bbox_puntos, cargar_puntos_csv, cargar_ventanas, construir_series, guardar_series,
    puntos_a_geodataframe, resumen_validez,
)
from utils.stac import conectar_catalogo, parametros_busqueda, preparar_items, resumir_imagenes  # noqa: E402

FECHAS = ("2025-10-01", "2026-09-30")
NUBES_MAX = 60  # filtro de escena, a propósito laxo: la máscara SCL descarta los píxeles nublados
RADIO_M = 150  # ventana de unos 300 x 300 m alrededor de cada punto
FRACCION_MINIMA = 0.5  # una fecha vale si al menos la mitad de la ventana quedó sin nubes
# Escenas de Planetary Computer cuyos archivos no traen sistema de coordenadas (GDAL no puede
# leerlos y la carga completa falla). Se detectaron al construir esta serie.
ITEMS_DEFECTUOSOS = {"S2B_MSIL2A_20260407T143739_R096_T18HXD_20260407T180947"}

inicio = time.time()

# 0. Puntos (CSV) y su copia en GeoJSON para QGIS
puntos = cargar_puntos_csv(CARPETA / "puntos_concepcion.csv")
puntos_a_geodataframe(puntos).to_file(CARPETA / "puntos_concepcion.geojson", driver="GeoJSON")
print(puntos[["id", "nombre", "lat", "lon", "tipo"]].to_string(index=False))

# 1. Búsqueda (nativa): una sola para todos los puntos
bbox = bbox_puntos(puntos, RADIO_M)
catalogo = conectar_catalogo()
parametros = parametros_busqueda("sentinel-2", bbox, FECHAS, nubes_max=NUBES_MAX, catalogo=catalogo)
items = preparar_items(catalogo.search(**parametros).item_collection())  # sin duplicados, un solo catálogo
items = [item for item in items if item.id not in ITEMS_DEFECTUOSOS]
print(f"{len(items)} imágenes tras descartar {len(ITEMS_DEFECTUOSOS)} escena(s) con archivos sin CRS")
tabla_items = resumir_imagenes(items)
print(f"Baselines: {tabla_items['baseline'].value_counts().to_dict()}; offsets: {tabla_items['offset'].value_counts().to_dict()}")

# 2. Carga (nativa): una ventana pequeña por punto, con offset y nubes de escena por fecha
ventanas = cargar_ventanas(items, puntos, radio_m=RADIO_M, catalogo=catalogo)

# 3. Máscara SCL, reflectancia con el offset de cada fecha, índices y mediana de la ventana
tabla = construir_series(ventanas, puntos, radio_m=RADIO_M, fraccion_minima=FRACCION_MINIMA)

# 4. Fechas efectivas por punto
resumen = resumen_validez(tabla)
print(resumen.to_string())

# 5. Parquet con metadatos
metadatos = {
    "titulo": "Series temporales Sentinel-2 en puntos de agua y costa de Concepción (2025-10 a 2026-09)",
    "descripcion": (
        "Mediana, en una ventana circular de 150 m de radio alrededor de cada punto, del NDWI de McFeeters "
        "(B03 y B08), el NDCI (B05 y B04), la reflectancia roja B04 y el NDVI (B08 y B04) de Sentinel-2 L2A. "
        "Píxeles con nubes, sombras, cirros, saturados o sin dato (SCL) descartados antes de resumir; "
        "las fechas con menos de la mitad de la ventana válida tienen los índices en NaN."),
    "fuente": "ESA Copernicus Sentinel-2 L2A, leído como COG desde Microsoft Planetary Computer (colección sentinel-2-l2a)",
    "licencia": "Datos Copernicus Sentinel de acceso libre y abierto (Reglamento UE 1159/2013). Cita: contiene datos Copernicus Sentinel modificados (2025-2026).",
    "periodo": f"{FECHAS[0]} a {FECHAS[1]}",
    "generado": date.today().isoformat(),
    "script": "talleres/fair-series-temporales/datos/preparar_series_sentinel2.py",
    "parametros": {"nubes_max_escena": NUBES_MAX, "radio_m": RADIO_M, "estadistico": "mediana",
                   "fraccion_minima_pixeles_validos": FRACCION_MINIMA, "resolucion_m": 10,
                   "crs": "UTM de los puntos (EPSG:32718)", "catalogo": "planetary"},
    "columnas": {
        "id": "identificador del punto (puntos_concepcion.csv)", "nombre": "nombre del punto", "tipo": "tipo de cobertura",
        "lat": "latitud, grados (EPSG:4326)", "lon": "longitud, grados (EPSG:4326)",
        "fecha": "fecha y hora UTC de la toma",
        "ndwi": "(B03 - B08) / (B03 + B08), adimensional, positivo sobre agua",
        "ndci": "(B05 - B04) / (B05 + B04), adimensional, sube con la clorofila",
        "b04": "reflectancia roja (0 a 1), sube con la turbidez",
        "ndvi": "(B08 - B04) / (B08 + B04), adimensional",
        "nubes_escena": "% de nubes de la escena completa (eo:cloud_cover)",
        "pixeles_validos": "fracción (0 a 1) de la ventana que pasó la máscara SCL",
        "n_pixeles": "píxeles de la ventana",
        "offset_reflectancia": "offset aplicado a esa fecha (0 o -0.1), según baseline y catálogo"},
}
guardar_series(tabla, CARPETA / "series_sentinel2_concepcion.parquet", metadatos)
print(f"{len(tabla)} filas ({tabla['fecha'].nunique()} fechas distintas). Listo en {time.time() - inicio:.0f} s")
