"""Prepara los datos de Landsat 8 y 9 del taller: temperatura de superficie y reflectancia.

A partir de `puntos_concepcion.csv` y de la colección `landsat-c2-l2` (Landsat Collection 2
Level-2, USGS) de Microsoft Planetary Computer, construye:

* series_landsat_concepcion.parquet: una fila por punto y fecha (2013-03 a hoy) con la
  temperatura de superficie en °C (`st_c`), la reflectancia verde, roja, NIR y SWIR1, el
  NDWI y el NDVI. Son las MEDIANAS de una ventana circular de 200 m de radio, calibradas
  (DN a temperatura y DN a reflectancia) y sin los píxeles con nubes, sombras o saturación.
* landsat_malla_concepcion.nc: la temperatura de superficie en una malla de 500 m (0,8 x 0,8
  grados alrededor de la bahía de Concepción y el golfo de Arauco), con todas las fechas.
  Se guarda como entero (centésimas de °C) comprimido.

Calibración (Landsat 8/9 Collection 2 Level-2):

    temperatura (K)  = DN * 0.00341802 + 149.0      (banda `lwir11`, ST_B10)
    reflectancia     = DN * 0.0000275  - 0.2        (bandas ópticas)

No se rellenan las fechas con nubes: quedan con `pixeles_validos` bajo o NaN, a propósito,
porque el taller trabaja con series irregulares.

Uso (necesita internet; tarda unos 45 minutos: 30 para las series y 15 para la malla):
    python preparar_landsat.py          # series por punto y malla
    python preparar_landsat.py malla    # solo la malla

Necesita netCDF4 (la malla se guarda comprimida).
"""

import os

# Lectura rápida de COG por internet: caché grande (se leen unos miles de archivos) y muchas conexiones.
os.environ.setdefault("CPL_VSIL_CURL_CACHE_SIZE", "1000000000")
os.environ.setdefault("GDAL_CACHEMAX", "1024")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")

import sys  # noqa: E402
import time  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parents[2]))  # raíz del repositorio, para importar utils

import odc.stac  # noqa: E402
import xarray as xr  # noqa: E402

from utils.raster import (convertir_a_temperatura, mascara_calidad_landsat,  # noqa: E402
                          mascara_saturacion_landsat)
from utils.series import (agregar_atributos_landsat, bbox_puntos, cargar_puntos_csv,  # noqa: E402
                          cargar_ventanas_landsat, construir_series_landsat, guardar_series)
from utils.stac import conectar_catalogo, items_legibles, parametros_busqueda, parametros_carga  # noqa: E402

INICIO = "2013-03-01"  # Landsat 8 empieza a entregar imágenes en marzo de 2013
FIN = date.today().isoformat()
NUBES_MAX = 90  # filtro de escena, a propósito laxo: la máscara qa_pixel descarta los píxeles nublados
RADIO_M = 200  # ventana de unos 400 x 400 m alrededor de cada punto
FRACCION_MINIMA = 0.5  # una fecha vale si al menos la mitad de la ventana quedó sin nubes
BBOX_MALLA = [-73.6, -37.35, -72.8, -36.55]  # oeste, sur, este, norte
RESOLUCION_MALLA = 500  # metros
VALIDOS_MINIMOS_MALLA = 0.10  # una fecha entra a la malla si al menos 10 % de sus píxeles quedó libre de nubes

SOLO_MALLA = "malla" in sys.argv[1:]  # python preparar_landsat.py malla: regenera solo la malla

inicio = time.time()
catalogo = conectar_catalogo("planetary")

# 0. Puntos
puntos = cargar_puntos_csv(CARPETA / "puntos_concepcion.csv")

anios = range(int(INICIO[:4]), int(FIN[:4]) + 1)
if not SOLO_MALLA:
    # 1. Búsqueda (nativa): una sola para todos los puntos, años por separado para no perder todo si falla uno
    bbox = bbox_puntos(puntos, RADIO_M)
    items = []
    for anio in anios:
        rango = (max(INICIO, f"{anio}-01-01"), min(FIN, f"{anio}-12-31"))
        parametros = parametros_busqueda("landsat", bbox, rango, nubes_max=NUBES_MAX, catalogo=catalogo)
        items += list(catalogo.search(**parametros).items())
    print(f"{len(items)} escenas Landsat 8/9 entre {INICIO} y {FIN} (nubes de escena < {NUBES_MAX} %)")

    # 1b. Escenas con archivos dañados: Planetary Computer devuelve una página de error en vez del GeoTIFF
    BANDAS_LEIDAS = ["green", "red", "nir08", "swir16", "lwir11", "qa_pixel", "qa_radsat"]
    items, danados = items_legibles(items, BANDAS_LEIDAS)
    print(f"{len(danados)} escena(s) descartada(s) por archivos dañados: {danados}")

    # 2. Series por punto: carga, máscara, calibración y resumen (año por año)
    tablas = []
    for anio in anios:
        items_anio = [i for i in items if i.datetime.year == anio]
        if not items_anio:
            continue
        ventanas = cargar_ventanas_landsat(items_anio, puntos, radio_m=RADIO_M, trabajadores=64, verbose=False)
        tablas.append(construir_series_landsat(ventanas, puntos, radio_m=RADIO_M, fraccion_minima=FRACCION_MINIMA))
        print(f"{anio}: {len(items_anio)} escenas, {tablas[-1]['fecha'].nunique()} fechas ({time.time() - inicio:.0f} s)")
    tabla = pd.concat(tablas, ignore_index=True)

    validas = tabla.groupby("id")["st_c"].agg(fechas="size", con_dato=lambda s: int(s.notna().sum()))
    print(validas.to_string())

    metadatos = {
        "titulo": f"Series temporales Landsat 8 y 9 en puntos de agua y costa de Concepción (2013-03 a {FIN[:7]})",
        "descripcion": (
            "Mediana, en una ventana circular de 200 m de radio alrededor de cada punto, de la temperatura de superficie "
            "(banda térmica lwir11, ST_B10), de la reflectancia verde, roja, NIR y SWIR1 y de los índices NDWI de McFeeters "
            "y NDVI de Landsat 8/9 Collection 2 Level-2. Calibración: temperatura (K) = DN x 0,00341802 + 149; "
            "reflectancia = DN x 0,0000275 - 0,2. Píxeles con relleno, nubes, sombras, cirros o saturación (qa_pixel y "
            "qa_radsat) descartados antes de resumir; las fechas con menos de la mitad de la ventana válida tienen las "
            "variables en NaN. La banda térmica tiene 100 m nativos (el USGS la entrega a 30 m)."),
        "fuente": "USGS Landsat 8 y 9 Collection 2 Level-2, leído como COG desde Microsoft Planetary Computer (colección landsat-c2-l2)",
        "licencia": "Dominio público (datos Landsat del USGS). Cita: Landsat Collection 2 Level-2, cortesía del Servicio Geológico de EE. UU.",
        "periodo": f"{INICIO} a {FIN}",
        "generado": date.today().isoformat(),
        "script": "talleres/fair-series-temporales/datos/preparar_landsat.py",
        "parametros": {"nubes_max_escena": NUBES_MAX, "radio_m": RADIO_M, "estadistico": "mediana",
                       "fraccion_minima_pixeles_validos": FRACCION_MINIMA, "resolucion_m": 30,
                       "crs": "UTM de los puntos (EPSG:32718)", "catalogo": "planetary",
                       "calibracion_temperatura": "DN * 0.00341802 + 149.0 (K), menos 273.15 (°C)",
                       "calibracion_reflectancia": "DN * 0.0000275 - 0.2"},
        "columnas": {
            "id": "identificador del punto (puntos_concepcion.csv)", "nombre": "nombre del punto", "tipo": "tipo de cobertura",
            "lat": "latitud, grados (EPSG:4326)", "lon": "longitud, grados (EPSG:4326)",
            "fecha": "fecha y hora UTC del paso del satélite", "plataforma": "landsat-8 o landsat-9",
            "st_dn": "mediana de la banda térmica lwir11 en números digitales (DN), antes de calibrar",
            "st_c": "temperatura de superficie calibrada, °C (DN x 0,00341802 + 149 - 273,15)",
            "verde": "reflectancia verde (0 a 1)", "rojo": "reflectancia roja (0 a 1), sube con la turbidez",
            "nir": "reflectancia del infrarrojo cercano (0 a 1)", "swir1": "reflectancia del infrarrojo de onda corta 1 (0 a 1)",
            "ndwi": "(verde - NIR) / (verde + NIR), positivo sobre agua", "ndvi": "(NIR - rojo) / (NIR + rojo)",
            "nubes_escena": "% de nubes de la escena completa (eo:cloud_cover)",
            "pixeles_validos": "fracción (0 a 1) de la ventana que pasó la máscara de calidad",
            "n_pixeles": "píxeles de la ventana (círculo de 200 m de radio, píxeles de 30 m)"},
    }
    guardar_series(tabla, CARPETA / "series_landsat_concepcion.parquet", metadatos)
    print(f"{len(tabla)} filas ({tabla['fecha'].nunique()} fechas distintas) en {time.time() - inicio:.0f} s")

# 3. Malla de temperatura de 500 m (nativa): todas las fechas, una muestra de píxel cada 500 m
parametros = parametros_busqueda("landsat", BBOX_MALLA, (INICIO, FIN), nubes_max=NUBES_MAX, catalogo=catalogo)
items_malla = list(catalogo.search(**parametros).items())
items_malla, danados_malla = items_legibles(items_malla, ["lwir11", "qa_pixel", "qa_radsat"])
print(f"Malla: {len(items_malla)} escenas legibles, {len(danados_malla)} descartada(s)")
cortes = []
for anio in anios:
    items_anio = [i for i in items_malla if i.datetime.year == anio]
    if not items_anio:
        continue
    carga = parametros_carga("landsat", ["lwir11", "qa_pixel", "qa_radsat"], RESOLUCION_MALLA, "EPSG:32718",
                             catalogo, BBOX_MALLA, chunks={"x": 512, "y": 512})
    datos = agregar_atributos_landsat(odc.stac.load(items_anio, **carga), items_anio).compute(scheduler="threads", num_workers=64)
    valida = mascara_calidad_landsat(datos["qa_pixel"].fillna(1)) & mascara_saturacion_landsat(datos["qa_radsat"].fillna(1))
    cobertura = (datos["qa_pixel"].fillna(1).astype("uint16") & 1) == 0
    st = convertir_a_temperatura(datos["lwir11"]).where(valida)
    util = cobertura.mean(("x", "y")) > 0.05  # la escena toca la malla
    fraccion = st.notnull().sum(("x", "y")) / st.sizes["x"] / st.sizes["y"]
    corte = st.where(util & (fraccion >= VALIDOS_MINIMOS_MALLA), drop=True)
    corte = corte.assign_coords(plataforma=("time", datos["plataforma"].sel(time=corte.time).values),
                                nubes_escena=("time", datos["nubes_escena"].sel(time=corte.time).values))
    cortes.append(corte)
    print(f"malla {anio}: {corte.sizes['time']} fechas con al menos {VALIDOS_MINIMOS_MALLA:.0%} de píxeles válidos ({time.time() - inicio:.0f} s)")
malla = xr.concat(cortes, dim="time").sortby("time")
malla.name = "st_c"
malla.attrs = {"long_name": "Temperatura de superficie Landsat 8/9", "units": "degC"}
conjunto = malla.to_dataset().drop_vars("spatial_ref", errors="ignore")
conjunto.attrs = {
    "title": "Temperatura de superficie Landsat 8/9, malla de 500 m alrededor de la bahía de Concepción y el golfo de Arauco",
    "source": "USGS Landsat 8 y 9 Collection 2 Level-2 (lwir11), Microsoft Planetary Computer; "
              "temperatura (K) = DN * 0.00341802 + 149.0; muestra de un píxel cada 500 m, sin promediar",
    "license": "Dominio público (USGS)", "crs": "EPSG:32718", "created": date.today().isoformat(),
    "script": "talleres/fair-series-temporales/datos/preparar_landsat.py",
}
codificacion = {"st_c": {"dtype": "int16", "scale_factor": 0.01, "_FillValue": -32768, "zlib": True, "complevel": 9}}
conjunto.to_netcdf(CARPETA / "landsat_malla_concepcion.nc", engine="netcdf4", encoding=codificacion)
peso = (CARPETA / "landsat_malla_concepcion.nc").stat().st_size / 1e6
print(f"Malla: {malla.sizes['time']} fechas x {malla.sizes['y']} x {malla.sizes['x']} píxeles, {peso:.1f} MB. Listo en {time.time() - inicio:.0f} s")
