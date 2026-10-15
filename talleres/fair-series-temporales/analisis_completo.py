# %% [markdown]
# # Análisis completo de series temporales satelitales
#
# [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/talleres/fair-series-temporales/analisis_completo.ipynb) [![Licencia: GPL-3.0](https://img.shields.io/badge/licencia-GPL--3.0-blue.svg)](https://github.com/alvaroparedesl/geomatica-aplicada/blob/main/LICENSE) [![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://colab.research.google.com/)
#
# Un solo cuaderno, **sin `utils/`**: todo el código está a la vista y reúne lo que hicimos en los bloques 3 a 7 del taller. Cambia tus puntos, fechas y área en la celda de configuración y ejecuta el cuaderno de arriba abajo.
#
# | Módulo | Qué hace | Bloque |
# |---|---|---|
# | 0 | Configuración, puntos y herramientas comunes | |
# | 1 | Escena Sentinel-2 del área: COG, color verdadero, índices del agua y la vegetación | 3 |
# | 2 | Serie temporal Sentinel-2 en tus puntos (máscara SCL, offset por imagen), guardada en parquet | 4 |
# | 3 | Exploración: estadísticos, media móvil, tendencia, ciclo estacional y z-score | 5 |
# | 4 | Gráficos interactivos y mapa de los puntos | 6 |
# | 5 | Landsat 8/9: temperatura de superficie, climatología, episodios extremos e Isolation Forest | 7 |
# | 6 | Malla de 500 m: frecuencia de anomalías (opcional, lenta) | 7 |
#
# Cada módulo dice qué necesita y qué deja. Los módulos 1 y 5 no dependen de los demás; el 3 y el 4 usan la serie del 2. Los resultados quedan en la carpeta `salidas/`, y la serie de Landsat (la parte lenta) se reutiliza si no cambias la configuración.

# %% [markdown]
# ## Módulo 0. Configuración

# %%
# @title Instalar los paquetes que falten (ejecutar primero)
import importlib
import subprocess
import sys

PAQUETES = {"pystac-client": "pystac_client", "planetary-computer": "planetary_computer", "odc-stac": "odc.stac", "hvplot": "hvplot",
            "folium": "folium", "scikit-learn": "sklearn"}  # los dos últimos ya vienen en Colab


def falta(modulo):
    try:
        importlib.import_module(modulo)
        return False
    except ImportError:
        return True


faltan = [paquete for paquete, modulo in PAQUETES.items() if falta(modulo)]
if faltan:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *faltan], check=True)
print("Instalado:", ", ".join(faltan) if faltan else "no faltaba nada")

# %%
# @title Importar bibliotecas
import base64
import hashlib
import io
import json
import math
import os
import re
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

import dask
import folium
import hvplot
import hvplot.pandas  # noqa: F401  (activa el accesor .hvplot de pandas)
import hvplot.xarray  # noqa: F401  (y el de xarray)
import matplotlib.pyplot as plt
import numpy as np
import odc.stac
import pandas as pd
import planetary_computer
import plotly.graph_objects as go
import pyarrow as pa
import pyarrow.parquet as pq
import pystac_client
import rasterio
import xarray as xr
from pyproj import Transformer
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from scipy import stats
from shapely.geometry import box, shape
from sklearn.ensemble import IsolationForest

# Lectura rápida de COG por internet: caché de encabezados grande (los mismos archivos se abren una vez por punto) y varias peticiones por conexión
os.environ.setdefault("CPL_VSIL_CURL_CACHE_SIZE", "1000000000")
os.environ.setdefault("GDAL_CACHEMAX", "1024")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
warnings.filterwarnings("ignore", category=UserWarning)  # avisos de GDAL por escenas sin georreferencia
warnings.filterwarnings("ignore", category=RuntimeWarning)  # medias de ventanas vacías (todo nube)
hvplot.extension("bokeh")
odc.stac.configure_rio(cloud_defaults=True)  # reintentos y lectura eficiente de COG por internet
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})

# Paleta que se distingue con daltonismo (Okabe-Ito), un color por punto
PALETA = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
ROJO, AZUL, GRIS = "#c0392b", "#2471a3", "#7f8c8d"

# %%
# @title Configuración: tus datos (edita esta celda)

# --- Puntos: id, nombre, lat, lon (grados decimales, EPSG:4326) y tipo ---
PUNTOS_CSV = None  # ruta a tu CSV con esas cinco columnas (por ejemplo "mis_puntos.csv"); None usa la tabla de abajo
PUNTOS = pd.DataFrame([
    ("bahia_concepcion", "Centro de la Bahía de Concepción", -36.6900, -73.0000, "bahia"),
    ("san_vicente", "Bahía San Vicente (Talcahuano)", -36.7450, -73.1650, "bahia"),
    ("biobio_pluma", "Pluma del río Biobío", -36.8250, -73.1800, "pluma"),
    ("golfo_arauco", "Golfo de Arauco", -37.1200, -73.4000, "golfo"),
    ("laguna_grande", "Laguna Grande de San Pedro", -36.8511, -73.1070, "laguna"),
    ("humedal_rocuant", "Humedal Rocuant-Andalién", -36.7481, -73.0560, "humedal"),
], columns=["id", "nombre", "lat", "lon", "tipo"])

# --- Sentinel-2 (módulos 1 a 4) ---
FECHAS_S2 = ("2025-10-01", "2026-09-30")  # período de la serie (desde un año para ver el ciclo estacional)
NUBES_MAX_ESCENA = 60  # filtro de escena: nubes máximas (%) de la imagen completa
RADIO_S2 = 150  # radio (m) de la ventana de cada punto
FRACCION_MINIMA = 0.5  # una fecha vale si al menos esta fracción de la ventana pasó la máscara
EXCLUIR_SCL = [0, 1, 3, 8, 9, 10]  # clases SCL descartadas: sin datos, saturado, sombra, nube media, nube alta, cirrus
AREA = [-73.22, -36.89, -72.98, -36.66]  # oeste, sur, este, norte (grados): área del módulo 1
FECHA_ESCENA = None  # "AAAA-MM-DD", o None para la fecha más despejada de FECHAS_S2
RESOLUCION_AREA = 10  # m (usa 20 o 60 si tu área es grande)

# --- Un punto y una variable para los gráficos de detalle (módulos 3 a 5) ---
PUNTO = "bahia_concepcion"  # un id de tu tabla de puntos
VARIABLE = "b04"  # "ndwi", "ndci", "b04" o "ndvi"

# --- Landsat 8/9 (módulo 5) ---
FECHAS_LANDSAT = ("2013-03-01", "2026-09-30")  # período largo: la primera vez es lenta (ver el módulo 5); acórtalo para probar
BASE = (2013, 2024)  # años de la climatología: años completos de tu período
RADIO_LANDSAT = 200  # radio (m) de la ventana de cada punto
NUBES_MAX_LANDSAT = 90  # filtro de escena laxo: la máscara de píxel hace el trabajo fino
TIPOS_AGUA = ["bahia", "pluma", "golfo", "laguna"]  # valores de la columna `tipo` que son agua abierta
PERCENTIL = 90  # umbral cálido (el frío es 100 menos este valor)
MIN_OBS = 2  # observaciones seguidas mínimas de un episodio
BRECHA_MAX_DIAS = 24  # días máximos entre dos observaciones de un mismo episodio
CONTAMINACION = 0.1  # fracción de observaciones que Isolation Forest marca como atípicas

# --- Malla de 500 m (módulo 6, opcional) ---
EJECUTAR_MALLA = False  # True para correr el módulo 6 (lento: lee todas las escenas del área)
AREA_MALLA = [-73.6, -37.35, -72.8, -36.55]  # oeste, sur, este, norte
RESOLUCION_MALLA = 500  # m
PERIODO_MALLA = (2025, 2026)  # años donde se cuenta la fracción de anomalías
T_MIN_MALLA = 3  # °C: bajo este valor se asume nube sin marcar (ajústalo a tu zona)

# --- Salidas ---
SALIDAS = Path("salidas")  # carpeta de resultados (en Colab queda en /content/salidas)
REUTILIZAR = True  # reutiliza la serie de Landsat guardada si no cambiaste la configuración

# %% [markdown]
# ### Puntos y herramientas comunes
#
# Funciones que usan varios módulos: coordenadas, ventanas alrededor de un punto, búsqueda STAC y guardado en parquet.

# %%
SALIDAS.mkdir(exist_ok=True)
catalogo = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", timeout=60)  # Planetary Computer


def crs_utm(lon, lat):
    """CRS UTM (EPSG) de la zona del punto, por ejemplo EPSG:32718 en Concepción."""
    zona = int((lon + 180) // 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zona}"


def a_utm(lon, lat, crs):
    """Coordenadas en grados a metros del CRS dado."""
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)


def bbox_punto(lon, lat, radio_m):
    """Rectángulo [oeste, sur, este, norte] de `radio_m` metros alrededor de un punto."""
    dlat = radio_m / 111_320
    dlon = radio_m / (111_320 * math.cos(math.radians(lat)))
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def bbox_puntos(puntos, radio_m):
    """Rectángulo que contiene las ventanas de todos los puntos (una sola búsqueda STAC)."""
    cajas = np.array([bbox_punto(p.lon, p.lat, radio_m) for p in puntos.itertuples()])
    return [float(cajas[:, 0].min()), float(cajas[:, 1].min()), float(cajas[:, 2].max()), float(cajas[:, 3].max())]


def en_ventana(datos, x, y, radio_m):
    """Máscara de los píxeles a `radio_m` metros o menos de (x, y); si ninguno, el más cercano."""
    distancia = np.hypot(datos["x"] - x, datos["y"] - y)
    dentro = distancia <= radio_m
    return dentro if bool(dentro.any()) else distancia == distancia.min()


def diferencia_normalizada(a, b):
    """(a - b) / (a + b), entre -1 y 1; lo demás pasa a NaN."""
    indice = (a - b) / (a + b)
    return indice.where(np.abs(indice) <= 1)


def buscar(coleccion, bbox, fechas, nubes_max=100, plataformas=None):
    """Búsqueda STAC nativa: ítems de una colección con menos de `nubes_max` % de nubes de escena."""
    filtro = {"eo:cloud_cover": {"lt": nubes_max}}
    if plataformas:
        filtro["platform"] = {"in": plataformas}
    for intento in range(3):  # un corte de red pasajero no debe botar el cuaderno
        try:
            busqueda = catalogo.search(collections=[coleccion], bbox=bbox, datetime=f"{fechas[0]}/{fechas[1]}", query=filtro)
            return list(busqueda.items())
        except Exception as error:
            if intento == 2:
                raise
            print(f"Falló la búsqueda ({type(error).__name__}); reintento {intento + 1} de 2")
            time.sleep(5)


def calcular(perezosas, trabajadores=8, reintentos=2):
    """Lee en paralelo todos los Dataset perezosos de un diccionario; repite si falla una lectura de red."""
    for intento in range(reintentos + 1):
        try:
            return dask.compute(perezosas, scheduler="threads", num_workers=trabajadores)[0]
        except Exception as error:
            if intento == reintentos:
                raise
            print(f"Falló una lectura ({type(error).__name__}); reintento {intento + 1} de {reintentos}")


def guardar_parquet(tabla, ruta, metadatos):
    """Guarda la tabla en parquet con los metadatos dentro del mismo archivo (el momento FAIR)."""
    tabla_pa = pa.Table.from_pandas(tabla, preserve_index=False)
    meta = dict(tabla_pa.schema.metadata or {})
    for clave, valor in metadatos.items():
        meta[clave.encode()] = (valor if isinstance(valor, str) else json.dumps(valor, ensure_ascii=False)).encode()
    pq.write_table(tabla_pa.replace_schema_metadata(meta), ruta, compression="zstd")


def leer_metadatos(ruta):
    """Metadatos de un parquet como diccionario de texto (la clave "pandas" no interesa)."""
    return {k.decode(): v.decode() for k, v in pq.read_schema(ruta).metadata.items() if k != b"pandas"}


def serie_de(tabla, punto, variable, sin_nan=True):
    """Serie de una variable en un punto: pandas.Series con índice de fechas."""
    serie = tabla[tabla["id"] == punto].set_index("fecha")[variable].astype(float).sort_index()
    return serie.dropna() if sin_nan else serie


# %%
# Puntos: validación y mapa (el círculo es la ventana de cada punto; el rectángulo, el área del módulo 1)
puntos = pd.read_csv(PUNTOS_CSV, dtype={"id": str, "nombre": str, "tipo": str}) if PUNTOS_CSV else PUNTOS.copy()
faltan = [c for c in ["id", "nombre", "lat", "lon", "tipo"] if c not in puntos.columns]
assert not faltan, f"A tu tabla de puntos le faltan las columnas {faltan}"
assert puntos["id"].is_unique, "Hay id repetidos"
assert puntos["lat"].between(-90, 90).all() and puntos["lon"].between(-180, 180).all(), "lat o lon fuera de rango (¿están invertidas?)"
puntos = puntos[["id", "nombre", "lat", "lon", "tipo"]].reset_index(drop=True)
if PUNTO not in set(puntos["id"]):
    print(f"PUNTO = {PUNTO!r} no está en tu tabla: se usa {puntos['id'].iloc[0]!r}")
    PUNTO = puntos["id"].iloc[0]
colores = {identificador: PALETA[i % len(PALETA)] for i, identificador in enumerate(puntos["id"])}

mapa_puntos = folium.Map(tiles="OpenStreetMap")
for p in puntos.itertuples():
    folium.CircleMarker([p.lat, p.lon], radius=7, color="white", weight=2, fill=True, fill_color=colores[p.id], fill_opacity=0.95,
                        tooltip=f"{p.id} ({p.tipo})").add_to(mapa_puntos)
    folium.Circle([p.lat, p.lon], radius=RADIO_S2, color=colores[p.id], weight=1, fill=False).add_to(mapa_puntos)
folium.Rectangle([[AREA[1], AREA[0]], [AREA[3], AREA[2]]], color="black", weight=1, fill=False, tooltip="AREA").add_to(mapa_puntos)
mapa_puntos.fit_bounds([[min(puntos["lat"].min(), AREA[1]), min(puntos["lon"].min(), AREA[0])],
                        [max(puntos["lat"].max(), AREA[3]), max(puntos["lon"].max(), AREA[2])]])
print(f"{len(puntos)} puntos | Sentinel-2: {FECHAS_S2[0]} a {FECHAS_S2[1]} | Landsat: {FECHAS_LANDSAT[0]} a {FECHAS_LANDSAT[1]}")
mapa_puntos

# %% [markdown]
# ### Herramientas de Sentinel-2
#
# La reflectancia de Sentinel-2 L2A es `DN * 0,0001 + offset`. El **offset depende de la versión de procesamiento (baseline) de cada imagen**: `-0,1` desde el baseline 04.00 y `0` antes. Se decide imagen por imagen, nunca una constante para toda la serie. `grupo_dia_offset` se pasa a `odc.stac.load` para que nunca se mezclen en una fecha imágenes con offsets distintos. Estas funciones sirven solo para Sentinel-2 de Planetary Computer.

# %%
def offset_item(item):
    """Offset de reflectancia de una imagen según su baseline de procesamiento."""
    baseline = float(item.properties["s2:processing_baseline"])  # "05.11" -> 5.11
    return -0.1 if baseline >= 4.0 else 0.0  # baseline 04.00 o posterior: a los DN se les sumaron 1000


def grupo_dia_offset(item, parsed=None, idx=None):
    """Clave de agrupación de odc.stac: (día solar, offset). Las imágenes de un mismo día y offset forman una fecha."""
    lon = (item.bbox[0] + item.bbox[2]) / 2
    return ((item.datetime + timedelta(hours=lon / 15)).date().isoformat(), offset_item(item))


def quitar_duplicados(items):
    """Una sola versión de cada toma (mismo satélite, tile y datastrip): la de baseline más alto."""
    mejor = {}
    for item in items:
        datastrip = re.search(r"_S(\d{8}T\d{6})_", item.properties.get("s2:datastrip_id") or "")
        toma = (item.properties["platform"], item.properties["s2:mgrs_tile"], datastrip.group(1) if datastrip else item.datetime.isoformat())
        version = (tuple(int(n) for n in item.properties["s2:processing_baseline"].split(".")), item.properties.get("s2:generation_time", ""), item.id)
        if toma not in mejor or version > mejor[toma][0]:
            mejor[toma] = (version, item)
    return sorted((item for _, item in mejor.values()), key=lambda item: (item.datetime, item.id))


def agregar_metadatos(datos, items):
    """Agrega a `datos` el offset de reflectancia y las nubes de la escena de cada fecha (coordenadas de `time`)."""
    grupos = {}
    for item in items:
        grupos.setdefault(grupo_dia_offset(item), []).append(item)
    por_fecha = {}
    for (_, offset), lista in grupos.items():
        nubes = float(np.mean([i.properties["eo:cloud_cover"] for i in lista]))
        for item in lista:
            por_fecha[np.datetime64(item.datetime.replace(tzinfo=None), "ns")] = (offset, nubes)
    fechas = [por_fecha[t] for t in datos["time"].values.astype("datetime64[ns]")]
    return datos.assign_coords(offset_reflectancia=("time", [f[0] for f in fechas]), nubes_escena=("time", [f[1] for f in fechas]))


def reflectancia_sentinel2(datos):
    """DN a reflectancia con el offset de cada fecha. Conviértela antes de combinar fechas (por ejemplo antes de una mediana en el tiempo)."""
    refl = datos.where(datos != 0) * 0.0001 + datos["offset_reflectancia"]  # DN 0 = sin dato
    refl = refl.where((refl >= 0) & (refl <= 1))  # una reflectancia válida está entre 0 y 1
    return refl.drop_vars("offset_reflectancia")


# %% [markdown]
# ## Módulo 1. Escena Sentinel-2 del área (bloque 3)
#
# **Entra:** `AREA`, `FECHAS_S2`, `FECHA_ESCENA`. **Deja:** `refl` (reflectancia), `ndwi`, `ndci`, `ndvi`, `agua`.
#
# Buscamos la fecha más despejada del período en el área (o la que fijaste).

# %%
items_area = quitar_duplicados(buscar("sentinel-2-l2a", AREA, FECHAS_S2))
dias = pd.DataFrame({"dia": [grupo_dia_offset(i)[0] for i in items_area], "nubes": [i.properties["eo:cloud_cover"] for i in items_area]})
nubes_por_dia = dias.groupby("dia")["nubes"].mean()
fecha_escena = FECHA_ESCENA or nubes_por_dia.idxmin()
items_escena = [i for i in items_area if grupo_dia_offset(i)[0] == fecha_escena]
print(f"{len(items_area)} imágenes en el área. Escena: {fecha_escena}, {len(items_escena)} imagen(es), nubes {nubes_por_dia[fecha_escena]:.1f} %")

# %% [markdown]
# **Por qué se puede leer solo una parte: el COG.** Un COG (*Cloud Optimized GeoTIFF*) guarda la imagen en teselas y con pirámides reducidas. `rasterio.open` lee solo el encabezado; después se pide la miniatura de toda la escena o una ventana a resolución completa, sin descargar la banda.

# %%
href = planetary_computer.sign(items_escena[0].assets["B04"].href)  # enlace firmado de Planetary Computer
with rasterio.open(href) as archivo:
    print(f"Banda completa: {archivo.width} x {archivo.height} píxeles de {archivo.res[0]:.0f} m | teselas {archivo.block_shapes[0]} | pirámides {archivo.overviews(1)}")
    t0 = time.time()
    miniatura = archivo.read(1, out_shape=(archivo.height // 10, archivo.width // 10))
    print(f"Escena completa en miniatura {miniatura.shape}: {time.time() - t0:.1f} s")
    ventana = from_bounds(*transform_bounds("EPSG:4326", archivo.crs, *AREA), transform=archivo.transform)
    t0 = time.time()
    recorte = archivo.read(1, window=ventana)
    print(f"Tu área a resolución completa {recorte.shape}: {time.time() - t0:.1f} s ({recorte.size / (archivo.width * archivo.height):.1%} de la banda)")

# %% [markdown]
# Ahora cargamos el área como `xarray` con `odc.stac.load`. La carga es perezosa (dask): arma el cubo y `.compute()` lee los bloques necesarios. Cinco bandas espectrales y la clasificación de la escena (`SCL`); las bandas de 20 m se remuestrean a la resolución pedida.

# %%
bandas_escena = ["B02", "B03", "B04", "B05", "B08", "SCL"]  # azul, verde, rojo, borde rojo, NIR y clasificación
lon_c, lat_c = (AREA[0] + AREA[2]) / 2, (AREA[1] + AREA[3]) / 2
crs_area = crs_utm(lon_c, lat_c)

perezoso = odc.stac.load(items_escena, bands=bandas_escena, crs=crs_area, resolution=RESOLUCION_AREA, bbox=AREA,
                         chunks={"x": 1024, "y": 1024}, groupby=grupo_dia_offset, patch_url=planetary_computer.sign)
perezoso = agregar_metadatos(perezoso, items_escena)
print(f"{perezoso.sizes['x']} x {perezoso.sizes['y']} píxeles de {RESOLUCION_AREA} m, {perezoso.nbytes / 1e6:.0f} MB por leer")
t0 = time.time()
escena = perezoso.compute()
print(f"Lectura: {time.time() - t0:.0f} s")

# %% [markdown]
# De DN a reflectancia (con el offset de la imagen) y los cuatro índices. Casi todos son diferencias normalizadas `(A - B) / (A + B)`:
#
# | Variable | Fórmula | Qué muestra |
# |---|---|---|
# | **NDWI** | `(B03 - B08) / (B03 + B08)` | Agua abierta: positivo sobre el agua |
# | **NDCI** | `(B05 - B04) / (B05 + B04)` | Clorofila en el agua |
# | **B04** | reflectancia roja | Turbidez: el agua con sedimentos refleja más en el rojo |
# | **NDVI** | `(B08 - B04) / (B08 + B04)` | Vegetación |
#
# Los píxeles muy oscuros pueden dar reflectancia levemente negativa: quedan en `NaN` (no se inventa un valor).

# %%
espectrales = ["B02", "B03", "B04", "B05", "B08"]
refl = reflectancia_sentinel2(escena[espectrales]).mean("time")  # un solo grupo casi siempre; si hay varios, se promedian
scl = escena["SCL"].max("time")
ndwi = diferencia_normalizada(refl["B03"], refl["B08"])
ndci = diferencia_normalizada(refl["B05"], refl["B04"])
ndvi = diferencia_normalizada(refl["B08"], refl["B04"])
agua = ndwi > 0  # máscara de agua
print(f"{float(agua.mean()):.0%} del área es agua (NDWI > 0)")


# %%
def imagen_rgb(refl, bandas, brillo=3.5):
    """Apila tres bandas en (alto, ancho, 3), multiplica por un factor de brillo y recorta entre 0 y 1."""
    return np.clip(np.dstack([refl[b].fillna(0).values for b in bandas]) * brillo, 0, 1)


extension = [float(escena["x"].min()), float(escena["x"].max()), float(escena["y"].min()), float(escena["y"].max())]
puntos_utm = {p.id: a_utm(p.lon, p.lat, crs_area) for p in puntos.itertuples()}
paneles = [
    ("Color verdadero", imagen_rgb(refl, ["B04", "B03", "B02"]), {}),
    ("Falso color (NIR, rojo, verde)", imagen_rgb(refl, ["B08", "B04", "B03"], 3.0), {}),
    ("NDWI (agua > 0)", ndwi.values, {"cmap": "RdBu", "vmin": -1, "vmax": 1}),
    ("B04 sobre el agua (turbidez)", refl["B04"].where(agua).values, {"cmap": "YlOrBr", "vmin": 0, "vmax": 0.06}),
    ("NDCI sobre el agua (clorofila)", ndci.where(agua).values, {"cmap": "YlGn", "vmin": -0.2, "vmax": 0.4}),
    ("NDVI sobre la tierra", ndvi.where(~agua).values, {"cmap": "RdYlGn", "vmin": -0.2, "vmax": 0.9}),
]
fig, ejes = plt.subplots(2, 3, figsize=(16, 11))
for eje, (titulo, imagen, opciones) in zip(ejes.ravel(), paneles):
    im = eje.imshow(imagen, extent=extension, **opciones)
    if opciones:
        fig.colorbar(im, ax=eje, fraction=0.04)
    for identificador, (px, py) in puntos_utm.items():
        eje.plot(px, py, "o", ms=5, mfc=colores[identificador], mec="white")
    eje.set_title(titulo, fontsize=10)
    eje.axis("off")
plt.suptitle(f"Escena Sentinel-2 del {fecha_escena} (los puntos son los de tu tabla)")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Explorar la imagen de forma interactiva.** Un solo gráfico con zoom y desplazamiento; al pasar el cursor se ve el valor de cada banda y el `id` de tus puntos. Cambia `COMPOSICION` a `"falso color"` para ver NIR, rojo y verde (la vegetación se ve roja). La imagen se reduce a unos 1000 píxeles de ancho para que el gráfico pese poco.

# %%
COMPOSICION = "color verdadero"  # o "falso color"
combinaciones = {"color verdadero": (["B04", "B03", "B02"], 3.5), "falso color": (["B08", "B04", "B03"], 3.0)}
bandas_rgb, brillo = combinaciones[COMPOSICION]
paso = max(1, math.ceil(max(escena.sizes["x"], escena.sizes["y"]) / 1000))
pixeles = (imagen_rgb(refl, bandas_rgb, brillo) * 255).astype("uint8")[::paso, ::paso]
img = xr.DataArray(pixeles, dims=("y", "x", "banda"), coords={"y": escena["y"].values[::paso], "x": escena["x"].values[::paso], "banda": ["R", "G", "B"]})
vista = img.hvplot.rgb(x="x", y="y", bands="banda", width=800, height=int(800 * img.sizes["y"] / img.sizes["x"]),
                       xlabel="este (m)", ylabel="norte (m)", title=f"{COMPOSICION}, {fecha_escena}")
dentro_area = pd.DataFrame([{"id": k, "x": px, "y": py} for k, (px, py) in puntos_utm.items()
                            if extension[0] <= px <= extension[1] and extension[2] <= py <= extension[3]])
vista * dentro_area.hvplot.points(x="x", y="y", hover_cols=["id"], color="white", line_color="black", size=9) if len(dentro_area) else vista

# %% [markdown]
# El valor de un píxel en cada punto (los que caen dentro del área), frente a la mediana de su ventana. Un píxel suelto es ruidoso: por eso la serie del módulo 2 usa la mediana de la ventana.

# %%
filas = []
for p in puntos.itertuples():
    x, y = puntos_utm[p.id]
    if not (extension[0] <= x <= extension[1] and extension[2] <= y <= extension[3]):
        continue
    dentro = en_ventana(refl, x, y, RADIO_S2)
    pixel = {"B04": refl["B04"], "NDWI": ndwi, "NDVI": ndvi}
    fila = {"punto": p.id, "SCL": float(scl.sel(x=x, y=y, method="nearest"))}
    for nombre, capa in pixel.items():
        fila[f"{nombre} píxel"] = float(capa.sel(x=x, y=y, method="nearest"))
        fila[f"{nombre} ventana"] = float(capa.where(dentro).median())
    filas.append(fila)
pd.DataFrame(filas).set_index("punto").round(3)

# %% [markdown]
# **Resolución espacial.** Sentinel-2 mide con píxeles de 10, 20 y 60 m según la banda. Misma zona (3 x 3 km al centro del área), banda `B03` remuestreada a cada resolución: elige la que coincida con el tamaño de lo que quieres ver.

# %%
caja = bbox_punto(lon_c, lat_c, 1500)
fig, ejes = plt.subplots(1, 3, figsize=(15, 5))
for eje, resolucion in zip(ejes, [10, 20, 60]):
    cargada = odc.stac.load(items_escena, bands=["B03"], crs=crs_area, resolution=resolucion, bbox=caja,
                            groupby=grupo_dia_offset, patch_url=planetary_computer.sign).isel(time=0)["B03"].compute()
    eje.imshow(cargada, cmap="gray", interpolation="nearest", vmin=np.nanpercentile(cargada, 2), vmax=np.nanpercentile(cargada, 98))
    eje.set_title(f"B03 a {resolucion} m: {cargada.sizes['y']} x {cargada.sizes['x']} píxeles")
    eje.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Módulo 2. Serie temporal Sentinel-2 en tus puntos (bloque 4)
#
# **Entra:** `puntos`, `FECHAS_S2`, `NUBES_MAX_ESCENA`, `RADIO_S2`, `FRACCION_MINIMA`, `EXCLUIR_SCL`. **Deja:** `serie_s2` (una fila por punto y fecha) y el parquet en `salidas/`.
#
# Una sola búsqueda para toda la zona; después cada punto se queda con las imágenes que lo tocan y se carga **una ventana pequeña** (unos 300 m), no la caja que encierra todos los puntos.

# %%
bbox_zona = bbox_puntos(puntos, RADIO_S2)
items = quitar_duplicados(buscar("sentinel-2-l2a", bbox_zona, FECHAS_S2, NUBES_MAX_ESCENA))
tabla_items = pd.DataFrame({"fecha": [i.datetime.strftime("%Y-%m-%d") for i in items], "tile": [i.properties["s2:mgrs_tile"] for i in items],
                            "nubes_%": [i.properties["eo:cloud_cover"] for i in items],
                            "baseline": [i.properties["s2:processing_baseline"] for i in items], "offset": [offset_item(i) for i in items]})
print(f"{len(items)} imágenes entre {FECHAS_S2[0]} y {FECHAS_S2[1]} con menos de {NUBES_MAX_ESCENA} % de nubes de escena")
print(tabla_items.groupby(["baseline", "offset"]).size().rename("imágenes").to_string())

# %% [markdown]
# Un `Dataset` perezoso por punto, y una sola lectura en paralelo al final. Los valores quedan en **DN**, sin enmascarar ni convertir.

# %%
def cargar_ventanas_s2(items, puntos, radio_m, bandas=("B03", "B04", "B05", "B08", "SCL")):
    perezosas = {}
    for p in puntos.itertuples():
        bbox = bbox_punto(p.lon, p.lat, radio_m)
        mios = [i for i in items if shape(i.geometry).intersects(box(*bbox))]  # solo las imágenes que tocan la ventana
        if not mios:
            print(f"Aviso: ninguna imagen cubre {p.id}")
            continue
        datos = odc.stac.load(mios, bands=list(bandas), crs=crs_utm(p.lon, p.lat), resolution=10, bbox=bbox,
                              chunks={"x": 512, "y": 512}, groupby=grupo_dia_offset, patch_url=planetary_computer.sign)
        perezosas[p.id] = agregar_metadatos(datos, mios)
    return calcular(perezosas, trabajadores=32)


t0 = time.time()
ventanas_s2 = cargar_ventanas_s2(items, puntos, RADIO_S2)
peso = sum(d.nbytes for d in ventanas_s2.values()) / 1e6
print(f"{len(ventanas_s2)} ventanas, {peso:.0f} MB leídos en {time.time() - t0:.0f} s")

# %% [markdown]
# De cada ventana a una fecha, un número:
#
# 1. **Máscara SCL:** los píxeles con nube, sombra, cirrus, saturados o sin dato pasan a `NaN` (la lista es `EXCLUIR_SCL`).
# 2. **Reflectancia** con el offset de **esa fecha** y los cuatro índices.
# 3. **Mediana** de los píxeles a `RADIO_S2` metros del punto. `pixeles_validos` es la fracción de la ventana que pasó la máscara; si es menor que `FRACCION_MINIMA`, la fecha se conserva con los índices en `NaN` para que el hueco se vea.

# %%
INDICES_S2 = ["ndwi", "ndci", "b04", "ndvi"]


def serie_s2_punto(datos, punto, radio_m=RADIO_S2, fraccion_minima=FRACCION_MINIMA, excluir=EXCLUIR_SCL):
    scl = datos["SCL"]
    cubierto = scl.notnull() & (scl != 0)  # la imagen cubre el píxel
    valida = cubierto & ~scl.isin(excluir)  # y su clase no está en la lista de descartadas
    refl = reflectancia_sentinel2(datos[["B03", "B04", "B05", "B08"]].where(valida))
    indices = xr.Dataset({"ndwi": diferencia_normalizada(refl["B03"], refl["B08"]),
                          "ndci": diferencia_normalizada(refl["B05"], refl["B04"]),
                          "b04": refl["B04"],
                          "ndvi": diferencia_normalizada(refl["B08"], refl["B04"])})
    x, y = a_utm(punto.lon, punto.lat, crs_utm(punto.lon, punto.lat))
    dentro = en_ventana(datos, x, y, radio_m)
    mediana = indices.where(dentro).median(("y", "x"))
    tabla = pd.DataFrame({v: mediana[v].values for v in INDICES_S2}, index=pd.Index(datos["time"].values, name="fecha"))
    tabla["nubes_escena"] = datos["nubes_escena"].values
    tabla["pixeles_validos"] = valida.astype("float32").where(dentro).mean(("y", "x")).values
    tabla["n_pixeles"] = int(dentro.sum())
    tabla["offset_reflectancia"] = datos["offset_reflectancia"].values
    tabla = tabla[cubierto.astype("float32").where(dentro).mean(("y", "x")).values > 0.5]  # la ventana debe estar al menos a medias en la imagen
    tabla.loc[tabla["pixeles_validos"] < fraccion_minima, INDICES_S2] = np.nan
    return tabla.reset_index().assign(id=punto.id)


def construir_serie_s2(ventanas, puntos, **opciones):
    partes = [serie_s2_punto(ventanas[p.id], p, **opciones) for p in puntos.itertuples() if p.id in ventanas]
    tabla = pd.concat(partes, ignore_index=True).merge(puntos, on="id")
    columnas = ["id", "nombre", "tipo", "lat", "lon", "fecha", *INDICES_S2, "nubes_escena", "pixeles_validos", "n_pixeles", "offset_reflectancia"]
    return tabla[columnas].sort_values(["id", "fecha"]).reset_index(drop=True)


serie_s2 = construir_serie_s2(ventanas_s2, puntos)
print(f"{len(serie_s2)} filas, {serie_s2['fecha'].nunique()} fechas distintas")
serie_s2.head(6)

# %% [markdown]
# Qué clases SCL hay en cada ventana (% de los píxeles de todas las fechas) y cuántas fechas aportan un valor útil. Mira las clases antes de confiar en `EXCLUIR_SCL`: cerca del agua conviene revisar qué se descarta.

# %%
clases = {0: "sin datos", 1: "saturado", 2: "áreas oscuras", 3: "sombra de nube", 4: "vegetación", 5: "suelo desnudo", 6: "agua",
          7: "nube prob. baja", 8: "nube prob. media", 9: "nube prob. alta", 10: "cirrus", 11: "nieve"}
distribucion = pd.DataFrame({identificador: pd.Series(d["SCL"].values[~np.isnan(d["SCL"].values)].astype(int)).value_counts(normalize=True).mul(100)
                             for identificador, d in ventanas_s2.items()}).rename(index=clases).fillna(0).round(1).sort_index()
print(distribucion.to_string())
validez = serie_s2.groupby("id", sort=False).agg(fechas=("fecha", "size"), fechas_validas=("ndwi", "count"), nubes_escena=("nubes_escena", "mean"),
                                                 pixeles_validos=("pixeles_validos", "mean")).round(2)
validez["% válido"] = (100 * validez["fechas_validas"] / validez["fechas"]).round(0)
validez

# %% [markdown]
# **Filtro de escena frente a máscara SCL.** El filtro de escena (`eo:cloud_cover`) descarta fechas completas pero no sabe nada de tu punto; la máscara SCL mira tus píxeles. Con la misma carga calculamos la serie **sin máscara** (solo se quitan los píxeles sin dato) y contamos las fechas "válidas" que en realidad tienen más de la mitad de la ventana tapada: las **contaminadas**. Para ver más límites, sube `NUBES_MAX_ESCENA` a 100.

# %%
sin_mascara = construir_serie_s2(ventanas_s2, puntos, excluir=[])
unidas = sin_mascara.merge(serie_s2, on=["id", "fecha"], suffixes=("_sin", "_con"))
unidas["contaminada"] = unidas["ndwi_sin"].notna() & unidas["ndwi_con"].isna()
filas = []
for limite in [l for l in (100, 60, 30, 10) if l <= NUBES_MAX_ESCENA]:
    sub = unidas[unidas["nubes_escena_sin"] <= limite]
    filas.append({"nubes de escena (%) hasta": limite, "fechas": len(sub), "válidas con máscara": int(sub["ndwi_con"].notna().sum()),
                  "válidas sin máscara": int(sub["ndwi_sin"].notna().sum()), "de ellas, contaminadas": int(sub["contaminada"].sum()),
                  "% de píxeles que quita la máscara": round(100 * (1 - sub["pixeles_validos_con"].mean()), 1)})
pd.DataFrame(filas).set_index("nubes de escena (%) hasta")

# %% [markdown]
# **Guardar con metadatos (el momento FAIR).** En parquet la tabla conserva sus tipos y los metadatos viajan dentro del mismo archivo: título, fuente, licencia, parámetros y qué significa cada columna.

# %%
metadatos_s2 = {
    "titulo": f"Series temporales Sentinel-2 en {len(puntos)} puntos ({FECHAS_S2[0]} a {FECHAS_S2[1]})",
    "descripcion": (f"Mediana, en una ventana circular de {RADIO_S2} m de radio alrededor de cada punto, del NDWI de McFeeters (B03 y B08), el NDCI "
                    "(B05 y B04), la reflectancia roja B04 y el NDVI (B08 y B04) de Sentinel-2 L2A. Píxeles con clases SCL descartadas antes de resumir; "
                    f"las fechas con menos de {FRACCION_MINIMA:.0%} de la ventana válida tienen los índices en NaN."),
    "fuente": "ESA Copernicus Sentinel-2 L2A, leído como COG desde Microsoft Planetary Computer (colección sentinel-2-l2a)",
    "licencia": f"Datos Copernicus Sentinel de acceso libre y abierto (Reglamento UE 1159/2013). Cita: contiene datos Copernicus Sentinel modificados ({FECHAS_S2[1][:4]}).",
    "periodo": f"{FECHAS_S2[0]} a {FECHAS_S2[1]}",
    "generado": date.today().isoformat(),
    "parametros": {"nubes_max_escena": NUBES_MAX_ESCENA, "radio_m": RADIO_S2, "estadistico": "mediana", "fraccion_minima_pixeles_validos": FRACCION_MINIMA,
                   "clases_scl_descartadas": EXCLUIR_SCL, "resolucion_m": 10, "crs": "UTM de cada punto", "catalogo": "planetary",
                   "offset": "-0.1 si el baseline de la imagen es 04.00 o posterior, si no 0"},
    "columnas": {
        "id": "identificador del punto", "nombre": "nombre del punto", "tipo": "tipo de cobertura", "lat": "latitud, grados (EPSG:4326)",
        "lon": "longitud, grados (EPSG:4326)", "fecha": "fecha y hora UTC de la toma",
        "ndwi": "(B03 - B08) / (B03 + B08), positivo sobre agua", "ndci": "(B05 - B04) / (B05 + B04), sube con la clorofila",
        "b04": "reflectancia roja (0 a 1), sube con la turbidez", "ndvi": "(B08 - B04) / (B08 + B04)",
        "nubes_escena": "% de nubes de la escena completa (eo:cloud_cover)", "pixeles_validos": "fracción (0 a 1) de la ventana que pasó la máscara SCL",
        "n_pixeles": "píxeles de la ventana", "offset_reflectancia": "offset aplicado a esa fecha (0 o -0.1), según el baseline"},
}
ruta_s2 = SALIDAS / "serie_sentinel2.parquet"
guardar_parquet(serie_s2, ruta_s2, metadatos_s2)
print(f"Guardado: {ruta_s2} ({ruta_s2.stat().st_size / 1024:.0f} kB)")
print("Metadatos dentro del archivo:", ", ".join(leer_metadatos(ruta_s2)))

# %% [markdown]
# ## Módulo 3. Exploración de la serie (bloque 5)
#
# **Entra:** `serie_s2` (o cualquier parquet con las mismas columnas: `serie_s2 = pd.read_parquet("mi_serie.parquet")`), `PUNTO`, `VARIABLE`. **Deja:** tablas y gráficos.
#
# Estadísticos descriptivos por punto: mira siempre más de un número (media, mediana, percentiles) y cuántos datos hay detrás.

# %%
ETIQUETAS = {"ndwi": "NDWI (agua)", "ndci": "NDCI (clorofila)", "b04": "B04, reflectancia roja (turbidez)", "ndvi": "NDVI (vegetación)"}
ids = list(serie_s2["id"].unique())
colores_punto = [colores[i] for i in ids]


def resumen_variable(variable):
    grupo = serie_s2.groupby("id", sort=False)[variable]
    return pd.DataFrame({"n": grupo.count(), "media": grupo.mean(), "mediana": grupo.median(), "p10": grupo.quantile(0.10),
                         "p90": grupo.quantile(0.90), "desv": grupo.std(), "min": grupo.min(), "max": grupo.max()}).round(3)


print(ETIQUETAS[VARIABLE])
resumen_variable(VARIABLE)

# %%
fig, ejes = plt.subplots(1, 4, figsize=(14, 3.8))
for eje, variable in zip(ejes, ETIQUETAS):
    caja = eje.boxplot([serie_de(serie_s2, i, variable) for i in ids], tick_labels=[i.replace("_", " ") for i in ids], patch_artist=True,
                       widths=0.6, flierprops={"markersize": 3})
    for parche, color in zip(caja["boxes"], colores_punto):
        parche.set_facecolor(color)
        parche.set_alpha(0.7)
    eje.set_title(ETIQUETAS[variable], fontsize=9)
    eje.tick_params(axis="x", rotation=60, labelsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Fechas irregulares y media móvil.** Una serie de Sentinel-2 no es diaria ni regular, y hay menos fechas útiles en invierno. `rolling` acepta una ventana en **días** (`"30D"`, adecuada para fechas irregulares) o en **observaciones**; `min_periods` es el mínimo de datos para calcular la media.

# %%
mes = serie_s2["fecha"].dt.to_period("M")
print(pd.DataFrame({"fechas con imagen": serie_s2.groupby(mes)["fecha"].nunique(),
                    "observaciones válidas por punto": (serie_s2.dropna(subset=["ndwi"]).groupby(mes).size() / len(ids)).round(1)}).T.to_string())

serie = serie_de(serie_s2, PUNTO, VARIABLE)
movil_30d = serie.rolling("30D", min_periods=3, center=True).mean()  # ventana de 30 días de calendario
movil_3 = serie.rolling(3, min_periods=3, center=True).mean()  # ventana de 3 observaciones
fig, eje = plt.subplots(figsize=(10, 3.6))
eje.plot(serie.index, serie, "o", ms=4, color="0.55", label="observaciones")
eje.plot(movil_3.index, movil_3, color="#E69F00", lw=1.2, label="media móvil de 3 observaciones")
eje.plot(movil_30d.index, movil_30d, color="#0072B2", lw=2, label="media móvil de 30 días")
eje.set_ylabel(ETIQUETAS[VARIABLE])
eje.set_title(f"{PUNTO}: dos formas de suavizar")
eje.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Tendencia lineal, y qué puede decir un solo año.** Ajustamos una recta por mínimos cuadrados contra el tiempo (pendiente **por año**). Con un año de datos la pendiente mezcla estación y cambio, el valor p supone observaciones independientes (no lo son) y las fechas no están repartidas parejo: sirve para describir el período, no para concluir. Para hablar de tendencia necesitas varios años y quitar antes el ciclo estacional.

# %%
def tendencia(serie):
    """Recta de mínimos cuadrados contra el tiempo: devuelve el ajuste (pendiente por año, R², p) y la recta."""
    anios = (serie.index - serie.index[0]).total_seconds().to_numpy() / (365.25 * 86400)
    ajuste = stats.linregress(anios, serie.to_numpy(dtype=float))
    return ajuste, pd.Series(ajuste.intercept + ajuste.slope * anios, index=serie.index)


pendientes = pd.DataFrame(np.nan, index=ids, columns=list(ETIQUETAS))
for i in ids:
    for v in ETIQUETAS:
        s = serie_de(serie_s2, i, v)
        if len(s) >= 3:
            pendientes.loc[i, v] = tendencia(s)[0].slope
print("Pendiente por año (unidades de cada variable)")
display(pendientes.round(3))

mitad = serie.index[0] + (serie.index[-1] - serie.index[0]) / 2  # el período partido en dos mitades
tramos = {"período completo": serie, "primera mitad": serie[serie.index <= mitad], "segunda mitad": serie[serie.index > mitad]}
fig, eje = plt.subplots(figsize=(10, 3.6))
eje.plot(serie.index, serie, "o", ms=4, color="0.6")
for (nombre, tramo), color in zip(tramos.items(), ["black", "#D55E00", "#0072B2"]):
    if len(tramo) >= 3:
        ajuste, recta = tendencia(tramo)
        eje.plot(recta.index, recta, color=color, lw=2.2, label=f"{nombre}: {ajuste.slope:+.2f} por año (R² = {ajuste.rvalue**2:.2f}, p = {ajuste.pvalue:.3f})")
eje.set_ylabel(ETIQUETAS[VARIABLE])
eje.set_title(f"{PUNTO}: la pendiente cambia según el tramo que mires")
eje.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Ciclo estacional.** Lo que sí se ve con un año es el ciclo anual: mediana mensual de cada punto (los meses empiezan en el primero de `FECHAS_S2`). Es el ciclo de un período corto, no una climatología: fíjate en el número de observaciones por mes.

# %%
mes0 = pd.Timestamp(FECHAS_S2[0]).month
meses = [(mes0 - 1 + k) % 12 + 1 for k in range(12)]
nombres_meses = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
fig, ejes = plt.subplots(2, 2, figsize=(12, 2 + 0.5 * len(ids) * 2), sharex=True)
for eje, variable in zip(ejes.ravel(), ETIQUETAS):
    mediana_mes = serie_s2.groupby(["id", serie_s2["fecha"].dt.month])[variable].median().unstack().reindex(index=ids, columns=meses)
    imagen = eje.imshow(mediana_mes.values, aspect="auto", cmap="viridis")
    eje.set_yticks(range(len(ids)), labels=[i.replace("_", " ") for i in ids], fontsize=8)
    eje.set_xticks(range(12), labels=[nombres_meses[m - 1] for m in meses], fontsize=8)
    eje.set_title(ETIQUETAS[variable], fontsize=9)
    eje.grid(False)
    plt.colorbar(imagen, ax=eje, fraction=0.04)
plt.tight_layout()
plt.show()
print(f"Observaciones válidas de {VARIABLE} por mes y punto")
serie_s2.dropna(subset=[VARIABLE]).groupby([serie_s2["fecha"].dt.month, "id"]).size().unstack().reindex(meses).rename(index=lambda m: nombres_meses[m - 1]).fillna(0).astype(int).T

# %% [markdown]
# **Anomalías con z-score:** `z = (valor - media) / desviación`; un |z| mayor que 2 es poco habitual. Hay dos referencias: `"total"` (toda la serie, mezcla estación con anomalía) y `"mes"` (el mismo mes del año, pero necesita muchos años: con un año cada mes tiene pocas observaciones del mismo año y el valor extremo infla su propia referencia). Aquí usamos `"total"` y desconfiamos de los z extremos hasta mirar la imagen.

# %%
def zscore(serie, referencia="total"):
    if referencia == "total":
        return (serie - serie.mean()) / serie.std()
    grupo = serie.groupby(serie.index.month)
    return (serie - grupo.transform("mean")) / grupo.transform("std")


filas = []
for i in ids:
    s = serie_de(serie_s2, i, VARIABLE)
    z_total, z_mes = zscore(s, "total"), zscore(s, "mes")
    filas.append({"punto": i, "n": len(s), "|z| > 2 'total'": int((z_total.abs() > 2).sum()), "|z| > 2 'mes'": int((z_mes.abs() > 2).sum()),
                  "z máximo 'total'": round(z_total.max(), 1), "z máximo 'mes'": round(z_mes.max(), 1)})
pd.DataFrame(filas).set_index("punto")

# %%
z = zscore(serie, "total")
raras = z[z.abs() > 2]
fig, ejes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
ejes[0].plot(serie.index, serie, "o-", color="0.6", ms=3, lw=0.8)
ejes[0].plot(raras.index, serie[raras.index], "o", color="#D55E00", ms=7, label="|z| > 2")
ejes[0].set_ylabel(ETIQUETAS[VARIABLE])
ejes[0].legend(fontsize=8)
ejes[0].set_title(f"{PUNTO}: fechas atípicas")
ejes[1].bar(z.index, z, width=2, color=np.where(z.abs() > 2, "#D55E00", "0.6"))
ejes[1].axhline(2, color="k", lw=0.7, ls="--")
ejes[1].set_ylabel("z (referencia total)")
plt.tight_layout()
plt.show()

# Una fecha atípica no es todavía un evento: ¿cuánta nube tenía la escena y qué fracción de la ventana era válida?
revision = serie_s2[(serie_s2["id"] == PUNTO) & serie_s2["fecha"].isin(raras.index)][["fecha", VARIABLE, "nubes_escena", "pixeles_validos"]]
revision.assign(z=raras.values).round({VARIABLE: 3, "nubes_escena": 1, "pixeles_validos": 2, "z": 1})

# %% [markdown]
# **Interpretación.** Mediana mensual de la variable elegida en cada punto. Los números salen de los datos; la lectura es **una hipótesis**: sin lluvia, caudal o mediciones en terreno no se confirma. Antes de concluir, mira cuántos datos hay detrás de cada celda.

# %%
mensual = serie_s2.assign(mes=serie_s2["fecha"].dt.to_period("M")).pivot_table(index="mes", columns="id", values=VARIABLE, aggfunc="median")
validos = serie_s2.dropna(subset=[VARIABLE]).assign(mes=lambda d: d["fecha"].dt.to_period("M")).pivot_table(index="mes", columns="id", values=VARIABLE, aggfunc="count")
print(f"Mediana mensual de {VARIABLE} (arriba) y número de fechas válidas (abajo)")
display(mensual.round(3).T)
validos.fillna(0).astype(int).T

# %% [markdown]
# **Tu interpretación.** Elige un punto y una variable y escribe tres frases: qué ves, qué hipótesis tienes y qué dato adicional necesitarías para confirmarla.

# %% [markdown]
# ## Módulo 4. Gráficos interactivos y mapa (bloque 6)
#
# **Entra:** `serie_s2`, `puntos`, `VARIABLE`, `PUNTO`. **Deja:** gráficos de `hvplot` y un mapa de `folium` (también como HTML en `salidas/`).
#
# Un gráfico interactivo permite hacer preguntas nuevas: zoom con la rueda del mouse, valores exactos al pasar el cursor, clic en la leyenda para esconder un punto. Pon **la calidad del dato en el cuadro emergente**: nubes de la escena y fracción de la ventana válida.

# %%
grafico = serie_s2.hvplot.line(x="fecha", y=VARIABLE, by="id", width=900, height=380, xlabel="Fecha", ylabel=ETIQUETAS[VARIABLE],
                               title=f"{ETIQUETAS[VARIABLE]} por punto", legend="right", hover_cols=["nubes_escena", "pixeles_validos"])
grafico

# %% [markdown]
# Los gráficos de `hvplot` se combinan con `*` (uno sobre otro) y `+` (uno junto a otro). Aquí, las observaciones de `PUNTO` con su media móvil de 30 días.

# %%
df_punto = pd.DataFrame({"valor": serie, "media móvil 30 d": serie.rolling("30D", min_periods=3, center=True).mean()}).reset_index()
observaciones = df_punto.hvplot.scatter(x="fecha", y="valor", color="gray", size=30, label="observaciones")
suavizada = df_punto.dropna(subset=["media móvil 30 d"]).hvplot.line(x="fecha", y="media móvil 30 d", color="#0072B2", line_width=3, label="media móvil 30 d")
(observaciones * suavizada).opts(width=900, height=350, title=f"{PUNTO}: {ETIQUETAS[VARIABLE]} y media móvil", ylabel=ETIQUETAS[VARIABLE], legend_position="top_left")

# %% [markdown]
# Un gráfico por punto (con zoom compartido) y, más abajo, un solo gráfico de Plotly con un menú para elegir el punto (el menú va dentro del gráfico, sin widgets).

# %%
por_punto = serie_s2.hvplot.line(x="fecha", y=VARIABLE, by="id", subplots=True, width=430, height=230, ylabel=VARIABLE, xlabel="", shared_axes=True).cols(2)
por_punto

# %%
def serie_con_menu(tabla, variable, titulo, etiqueta):
    """Una línea por punto en un solo gráfico de Plotly; el menú elige cuál se ve (el menú va dentro del gráfico, no es un widget)."""
    ids = list(tabla["id"].unique())
    fig = go.Figure()
    for i, identificador in enumerate(ids):
        s = tabla[tabla["id"] == identificador].dropna(subset=[variable])
        fig.add_trace(go.Scatter(x=s["fecha"], y=s[variable], mode="lines+markers", name=identificador, visible=(i == 0),
                                 line={"color": colores[identificador]}, customdata=s[["nubes_escena", "pixeles_validos"]],
                                 hovertemplate=("%{x|%Y-%m-%d}<br>" + etiqueta + ": %{y:.3f}<br>nubes de la escena: %{customdata[0]:.0f} %"
                                                "<br>píxeles válidos: %{customdata[1]:.2f}<extra></extra>")))
    botones = [{"label": identificador, "method": "update", "args": [{"visible": [j == i for j in range(len(ids))]}, {"title": f"{titulo}: {identificador}"}]}
               for i, identificador in enumerate(ids)]
    fig.update_layout(title=f"{titulo}: {ids[0]}", updatemenus=[{"buttons": botones, "x": 0, "y": 1.18, "xanchor": "left"}],
                      xaxis_title="Fecha", yaxis_title=etiqueta, width=800, height=380, margin={"t": 90})
    return fig


serie_con_menu(serie_s2, VARIABLE, ETIQUETAS[VARIABLE], ETIQUETAS[VARIABLE])

# %% [markdown]
# **Mapa con folium.** Cada punto es un marcador cuyo cuadro emergente resume su serie: estadísticos y un gráfico pequeño (una imagen PNG incrustada en el HTML).

# %%
def resumen_html(identificador):
    sub = serie_s2[serie_s2["id"] == identificador]
    filas = "".join(f"<tr><td>{v.upper()}</td><td>{sub[v].median():.3f}</td><td>{sub[v].quantile(0.1):.3f}</td><td>{sub[v].quantile(0.9):.3f}</td></tr>" for v in ETIQUETAS)
    return (f"<table style='font-size:12px'><tr><th>variable</th><th>mediana</th><th>p10</th><th>p90</th></tr>{filas}</table>"
            f"<p style='font-size:12px'>{sub['ndwi'].notna().sum()} fechas válidas de {len(sub)} con imagen</p>")


def miniatura_png(identificador, variable):
    s = serie_de(serie_s2, identificador, variable)
    fig, eje = plt.subplots(figsize=(3.4, 1.4))
    eje.plot(s.index, s.values, marker="o", ms=2, lw=0.8, color=colores[identificador])
    eje.set_title(f"{variable.upper()} en el tiempo", fontsize=8)
    eje.tick_params(labelsize=6)
    memoria = io.BytesIO()
    fig.savefig(memoria, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(memoria.getvalue()).decode()


mapa = folium.Map(tiles=None)
folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(mapa)
folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                 attr="Esri, Maxar, Earthstar Geographics", name="Imagen satelital (Esri)").add_to(mapa)
for p in puntos[puntos["id"].isin(ids)].itertuples():
    html = (f"<b>{p.nombre}</b><br>tipo: {p.tipo}<br>{resumen_html(p.id)}"
            f"<img src='data:image/png;base64,{miniatura_png(p.id, VARIABLE)}' width='290'>")
    folium.CircleMarker([p.lat, p.lon], radius=8, color="white", weight=2, fill=True, fill_color=colores[p.id], fill_opacity=0.95, tooltip=p.nombre,
                        popup=folium.Popup(folium.IFrame(html, width=330, height=330), max_width=340)).add_to(mapa)
    folium.Circle([p.lat, p.lon], radius=RADIO_S2, color=colores[p.id], weight=1, fill=False).add_to(mapa)
folium.LayerControl().add_to(mapa)
mapa.fit_bounds([[puntos["lat"].min(), puntos["lon"].min()], [puntos["lat"].max(), puntos["lon"].max()]])
mapa

# %% [markdown]
# Un gráfico o un mapa se pueden guardar como HTML **independiente**, que cualquiera abre en el navegador sin Python.

# %%
hvplot.save(grafico, str(SALIDAS / "serie_interactiva.html"))
mapa.save(str(SALIDAS / "mapa_puntos.html"))
print("Guardado en", SALIDAS, ":", ", ".join(sorted(p.name for p in SALIDAS.glob("*.html"))))

# %% [markdown]
# ## Módulo 5. Landsat 8/9: temperatura de superficie y eventos extremos (bloque 7)
#
# **Entra:** `puntos`, `FECHAS_LANDSAT`, `BASE`, `TIPOS_AGUA` y los parámetros de episodios. **Deja:** `serie_landsat`, `episodios_todos` y dos parquet en `salidas/`.
#
# Landsat pasa cada 8 días con los dos satélites, pero las nubes borran muchas fechas (unas 10 a 25 observaciones útiles por año) y la banda térmica es de 100 m. Mide **temperatura de la superficie** (la piel del agua, no el agua en profundidad) con errores de 1 K o más: no es un producto oceanográfico. Esas limitaciones son las de casi cualquier serie satelital real.
#
# Los datos llegan como números enteros (DN) y hay que **calibrarlos**:
#
# | Variable | Banda | Calibración |
# |---|---|---|
# | Temperatura de superficie | `lwir11` | kelvin = DN x 0,00341802 + 149,0; °C = kelvin - 273,15 |
# | Reflectancia (verde, rojo, NIR, SWIR1) | `green`, `red`, `nir08`, `swir16` | reflectancia = DN x 0,0000275 - 0,2 |
#
# Un DN de 0 es "sin dato". La máscara de calidad usa `qa_pixel` (relleno, nube dilatada, cirrus, nube y sombra) y `qa_radsat` (saturación).

# %%
BITS_QA = (0, 1, 2, 3, 4)  # relleno, nube dilatada, cirrus, nube, sombra de nube
BITS_RADSAT = (0, 1, 2, 3, 4, 5, 6, 11)  # saturación de las bandas ópticas y terreno que tapa el píxel


def temperatura_landsat(dn):
    kelvin = dn.where(dn != 0) * 0.00341802 + 149.0
    return kelvin.where((kelvin >= 230) & (kelvin <= 340)) - 273.15  # fuera de -43 a 67 °C es un error, no una temperatura


def reflectancia_landsat(dn):
    refl = dn.where(dn != 0) * 0.0000275 - 0.2
    return refl.where((refl >= 0) & (refl <= 1))


def mascara_landsat(datos):
    """(válido, cubierto): píxeles sin nube ni saturación, y píxeles que existen (no son relleno)."""
    qa = datos["qa_pixel"].fillna(1).astype("uint16")
    radsat = datos["qa_radsat"].fillna(1).astype("uint16")
    valido = ((qa & sum(1 << b for b in BITS_QA)) == 0) & ((radsat & sum(1 << b for b in BITS_RADSAT)) == 0)
    return valido, (qa & 1) == 0


def buscar_landsat(bbox, fechas, nubes_max):
    """Una búsqueda por año (si una falla no pierdes todo). Se filtra la plataforma: la colección trae también Landsat 4, 5 y 7."""
    items = []
    for anio in range(int(fechas[0][:4]), int(fechas[1][:4]) + 1):
        rango = (max(fechas[0], f"{anio}-01-01"), min(fechas[1], f"{anio}-12-31"))
        items += buscar("landsat-c2-l2", bbox, rango, nubes_max, plataformas=["landsat-8", "landsat-9"])
    return items


def agregar_atributos_landsat(datos, items):
    """Plataforma y nubes de escena de cada fecha: los ítems a menos de una hora de ella."""
    horas = np.array([np.datetime64(i.datetime.replace(tzinfo=None), "ns") for i in items])
    plataformas = np.array([i.properties["platform"] for i in items])
    nubes = np.array([i.properties["eo:cloud_cover"] for i in items], dtype=float)
    cerca = [np.abs(horas - t) < np.timedelta64(1, "h") for t in datos["time"].values.astype("datetime64[ns]")]
    return datos.assign_coords(plataforma=("time", [plataformas[c][0] for c in cerca]), nubes_escena=("time", [nubes[c].mean() for c in cerca]))


items_l = buscar_landsat(bbox_puntos(puntos, RADIO_LANDSAT), FECHAS_LANDSAT, NUBES_MAX_LANDSAT)
print(f"{len(items_l)} escenas Landsat 8/9 entre {FECHAS_LANDSAT[0]} y {FECHAS_LANDSAT[1]} (nubes de escena < {NUBES_MAX_LANDSAT} %)")

# %% [markdown]
# **Calibrar una escena a mano.** La escena más despejada que toca la ventana de `PUNTO`: de DN a kelvin, a °C y a reflectancia. Mira qué pasa si te saltas un paso.

# %%
p = puntos[puntos["id"] == PUNTO].iloc[0]
caja = bbox_punto(p.lon, p.lat, RADIO_LANDSAT)
candidatas = [i for i in items_l if shape(i.geometry).intersects(box(*caja))]
item_demo = min(candidatas, key=lambda i: i.properties["eo:cloud_cover"])
crudo = odc.stac.load([item_demo], bands=["lwir11", "red"], crs=crs_utm(p.lon, p.lat), resolution=30, bbox=caja, patch_url=planetary_computer.sign).isel(time=0).compute()
dn = crudo["lwir11"]
kelvin = dn * 0.00341802 + 149.0
print(f"{item_demo.id} ({item_demo.properties['eo:cloud_cover']:.0f} % de nubes)")
print(f"DN térmico (mediana): {int(dn.median())} -> {float(kelvin.median()):.2f} K -> {float(kelvin.median()) - 273.15:.2f} °C")
rojo = float((crudo["red"] * 0.0000275 - 0.2).median())
print(f"Reflectancia roja: {rojo:.4f} | sin el offset de -0,2 sería {float((crudo['red'] * 0.0000275).median()):.3f}")
print(f"Sin sumar 149 a la temperatura: {float((dn * 0.00341802).median()) - 273.15:.0f} °C")

# %% [markdown]
# **La serie de Landsat en cada punto.** Para cada año: un `Dataset` perezoso por punto, una lectura en paralelo, máscara, calibración y mediana de la ventana (`RADIO_LANDSAT`). `fail_on_error=False` salta los archivos dañados que a veces entrega Planetary Computer. Es la parte lenta y depende de tu conexión, de los puntos y de los años (en una prueba, un año con 2 puntos tomó unos 7 minutos): cada año se guarda al terminar y el resultado completo se reutiliza, así que si se corta puedes volver a ejecutar y sigue donde quedó.

# %%
def serie_landsat_punto(datos, punto, radio_m=RADIO_LANDSAT, fraccion_minima=FRACCION_MINIMA):
    valida, cubierta = mascara_landsat(datos)
    refl = reflectancia_landsat(datos[["green", "red", "nir08", "swir16"]]).where(valida)
    variables = xr.Dataset({"st_c": temperatura_landsat(datos["lwir11"]).where(valida),
                            "verde": refl["green"], "rojo": refl["red"], "nir": refl["nir08"], "swir1": refl["swir16"],
                            "ndwi": diferencia_normalizada(refl["green"], refl["nir08"]),
                            "ndvi": diferencia_normalizada(refl["nir08"], refl["red"])})
    x, y = a_utm(punto.lon, punto.lat, crs_utm(punto.lon, punto.lat))
    dentro = en_ventana(datos, x, y, radio_m)
    mediana = variables.where(dentro).median(("y", "x"))
    tabla = pd.DataFrame({v: mediana[v].values for v in variables}, index=pd.Index(datos["time"].values, name="fecha"))
    tabla["plataforma"] = datos["plataforma"].values
    tabla["nubes_escena"] = datos["nubes_escena"].values
    tabla["pixeles_validos"] = valida.astype("float32").where(dentro).mean(("y", "x")).values
    tabla["n_pixeles"] = int(dentro.sum())
    tabla = tabla[cubierta.astype("float32").where(dentro).mean(("y", "x")).values > 0.5]
    tabla.loc[tabla["pixeles_validos"] < fraccion_minima, list(variables)] = np.nan
    return tabla.reset_index().assign(id=punto.id)


def construir_serie_landsat(items, puntos, fechas, huella, radio_m=RADIO_LANDSAT):
    bandas = ["green", "red", "nir08", "swir16", "lwir11", "qa_pixel", "qa_radsat"]
    carpeta = SALIDAS / "cache_landsat"  # cada año se guarda al terminar: si se corta, al volver a ejecutar sigue donde quedó
    carpeta.mkdir(exist_ok=True)
    inicio, partes = time.time(), []
    for anio in range(int(fechas[0][:4]), int(fechas[1][:4]) + 1):
        archivo_anio = carpeta / f"{huella}_{anio}.parquet"
        if REUTILIZAR and archivo_anio.exists():
            partes.append(pd.read_parquet(archivo_anio))
            print(f"{anio}: reutilizado")
            continue
        items_anio = [i for i in items if i.datetime.year == anio]
        perezosas = {}
        for p in puntos.itertuples():
            bbox = bbox_punto(p.lon, p.lat, radio_m)
            mios = [i for i in items_anio if shape(i.geometry).intersects(box(*bbox))]
            if mios:
                datos = odc.stac.load(mios, bands=bandas, crs=crs_utm(p.lon, p.lat), resolution=30, bbox=bbox, chunks={"x": 512, "y": 512},
                                      groupby="solar_day", patch_url=planetary_computer.sign, fail_on_error=False)
                perezosas[p.id] = agregar_atributos_landsat(datos, mios)
        ventanas = calcular(perezosas, trabajadores=32)
        tablas_anio = [serie_landsat_punto(ventanas[p.id], p, radio_m) for p in puntos.itertuples() if p.id in ventanas]
        if tablas_anio:
            partes.append(pd.concat(tablas_anio, ignore_index=True))
            partes[-1].to_parquet(archivo_anio)
        print(f"{anio}: {len(items_anio)} escenas ({time.time() - inicio:.0f} s)")
    tabla = pd.concat(partes, ignore_index=True).merge(puntos, on="id")
    columnas = ["id", "nombre", "tipo", "lat", "lon", "fecha", "plataforma", "st_c", "verde", "rojo", "nir", "swir1", "ndwi", "ndvi",
                "nubes_escena", "pixeles_validos", "n_pixeles"]
    tabla["fecha"] = tabla["fecha"].dt.normalize()  # solo importa el día: la hora de la toma se descarta
    return tabla[columnas].sort_values(["id", "fecha"]).reset_index(drop=True)


# La huella resume lo que define la serie: si no cambiaste nada, se reutiliza el archivo guardado
huella = hashlib.md5(json.dumps([puntos[["id", "lat", "lon", "tipo"]].values.tolist(), FECHAS_LANDSAT, RADIO_LANDSAT, FRACCION_MINIMA, NUBES_MAX_LANDSAT],
                                default=str).encode()).hexdigest()[:10]
ruta_landsat = SALIDAS / "serie_landsat.parquet"
if REUTILIZAR and ruta_landsat.exists() and leer_metadatos(ruta_landsat).get("huella") == huella:
    serie_landsat = pd.read_parquet(ruta_landsat)
    print(f"Se reutiliza {ruta_landsat} (misma configuración): {len(serie_landsat)} filas")
else:
    t0 = time.time()
    serie_landsat = construir_serie_landsat(items_l, puntos, FECHAS_LANDSAT, huella)
    guardar_parquet(serie_landsat, ruta_landsat, {
        "titulo": f"Series temporales Landsat 8 y 9 en {len(puntos)} puntos ({FECHAS_LANDSAT[0]} a {FECHAS_LANDSAT[1]})",
        "descripcion": (f"Mediana, en una ventana circular de {RADIO_LANDSAT} m de radio alrededor de cada punto, de la temperatura de superficie (banda térmica lwir11), "
                        "de la reflectancia verde, roja, NIR y SWIR1 y de los índices NDWI de McFeeters y NDVI de Landsat 8/9 Collection 2 Level-2. "
                        "Calibración: temperatura (K) = DN x 0,00341802 + 149; reflectancia = DN x 0,0000275 - 0,2. Píxeles con relleno, nubes, sombras, cirros "
                        f"o saturación descartados antes de resumir; las fechas con menos de {FRACCION_MINIMA:.0%} de la ventana válida tienen las variables en NaN."),
        "fuente": "USGS Landsat 8 y 9 Collection 2 Level-2, leído como COG desde Microsoft Planetary Computer (colección landsat-c2-l2)",
        "licencia": "Dominio público (datos Landsat del USGS). Cita: Landsat Collection 2 Level-2, cortesía del Servicio Geológico de EE. UU.",
        "periodo": f"{FECHAS_LANDSAT[0]} a {FECHAS_LANDSAT[1]}", "generado": date.today().isoformat(), "huella": huella,
        "parametros": {"nubes_max_escena": NUBES_MAX_LANDSAT, "radio_m": RADIO_LANDSAT, "estadistico": "mediana",
                       "fraccion_minima_pixeles_validos": FRACCION_MINIMA, "resolucion_m": 30, "catalogo": "planetary"},
        "columnas": {"id": "identificador del punto", "nombre": "nombre del punto", "tipo": "tipo de cobertura", "lat": "latitud, grados (EPSG:4326)",
                     "lon": "longitud, grados (EPSG:4326)", "fecha": "día del paso del satélite", "plataforma": "landsat-8 o landsat-9",
                     "st_c": "temperatura de superficie calibrada, °C", "verde": "reflectancia verde (0 a 1)",
                     "rojo": "reflectancia roja (0 a 1), sube con la turbidez", "nir": "reflectancia del infrarrojo cercano (0 a 1)",
                     "swir1": "reflectancia del infrarrojo de onda corta 1 (0 a 1)", "ndwi": "(verde - NIR) / (verde + NIR), positivo sobre agua",
                     "ndvi": "(NIR - rojo) / (NIR + rojo)", "nubes_escena": "% de nubes de la escena completa (eo:cloud_cover)",
                     "pixeles_validos": "fracción (0 a 1) de la ventana que pasó la máscara de calidad", "n_pixeles": "píxeles de la ventana"}})
    print(f"{len(serie_landsat)} filas guardadas en {ruta_landsat} ({time.time() - t0:.0f} s)")

# %% [markdown]
# **Control de calidad con la reflectancia.** La máscara `qa_pixel` marca las nubes de confianza alta; una nube delgada o la neblina pueden quedar sin marcar y medirse como "agua muy fría". Pero el agua abierta absorbe casi toda la luz roja e infrarroja: una observación sobre agua con **reflectancia roja mayor que 0,05 o SWIR1 mayor que 0,02** es sospechosa y se deja en `NaN` (la fecha se conserva). No se toca la tierra, cuya reflectancia es otra. Es una regla declarada: puede dejar pasar neblina fina y descartar alguna floración de algas.

# %%
def controlar_calidad_agua(tabla, rojo_max=0.05, swir1_max=0.02, tipos=TIPOS_AGUA):
    columnas = ["st_c", "verde", "rojo", "nir", "swir1", "ndwi", "ndvi"]
    limpia = tabla.copy()
    sospechosa = limpia["tipo"].isin(tipos) & ((limpia["rojo"] > rojo_max) | (limpia["swir1"] > swir1_max))
    limpia.loc[sospechosa, columnas] = np.nan
    return limpia


tabla_qc = controlar_calidad_agua(serie_landsat)
descartadas = (serie_landsat["st_c"].notna() & tabla_qc["st_c"].isna()).groupby(serie_landsat["id"]).sum()
print("Observaciones con dato descartadas por sospecha de nube:")
print(descartadas.to_string())

# %% [markdown]
# **Una serie irregular.** La serie de `PUNTO` con todos los pasos del satélite (las fechas con nubes quedan como `NaN`, así se ven los huecos): observaciones por año, % de pasos con dato por mes e intervalos entre observaciones válidas.

# %%
st = serie_de(tabla_qc, PUNTO, "st_c", sin_nan=False)
por_anio = st.groupby(st.index.year).agg(pasos="size", con_dato=lambda s: int(s.notna().sum()))
por_mes = st.groupby(st.index.month).apply(lambda s: s.notna().mean())
intervalos = st.dropna().index.to_series().diff().dt.days.dropna()

fig, ejes = plt.subplots(1, 4, figsize=(17, 3.4), gridspec_kw={"width_ratios": [2, 1, 1, 1]})
ejes[0].plot(st.dropna().index, st.dropna(), "o-", ms=3, lw=0.6, color=AZUL)
ejes[0].set_ylabel("temperatura de superficie (°C)")
ejes[0].set_title(f"{PUNTO}: cada punto es una pasada sin nubes")
ejes[1].bar(por_anio.index, por_anio["pasos"], color=GRIS, label="pasos del satélite")
ejes[1].bar(por_anio.index, por_anio["con_dato"], color=AZUL, label="con dato")
ejes[1].set_title("Observaciones por año")
ejes[1].legend(fontsize=8)
ejes[2].bar(por_mes.index, por_mes * 100, color=AZUL)
ejes[2].set_xticks(range(1, 13), labels=list("EFMAMJJASOND"))
ejes[2].set_title("% de pasos con dato por mes")
ejes[3].hist(intervalos, bins=range(0, int(intervalos.max()) + 8, 8), color=AZUL, rwidth=0.9)
ejes[3].set_xlabel("días entre observaciones válidas")
ejes[3].set_title("Intervalos")
plt.tight_layout()
plt.show()
print(f"{len(st)} pasos del satélite, {st.notna().sum()} con dato ({st.notna().mean():.0%}); mediana de {intervalos.median():.0f} días entre observaciones válidas, máximo {intervalos.max():.0f}")

# %% [markdown]
# **Climatología con observaciones irregulares (la idea de Hobday, adaptada).** El método de Hobday necesita una serie diaria y 30 años: Landsat no los tiene. Adaptamos la idea: para cada uno de los 366 días del año se juntan **todas las observaciones del período base a menos de 15 días de ese día, de todos los años** (ventana de 31 días), se calculan la media y los percentiles `PERCENTIL` y `100 - PERCENTIL`, y se suaviza la curva con una media móvil circular de 15 días. Con menos de 8 observaciones en la ventana no se inventa una climatología. El calendario es de 366 días: el 29 de febrero es siempre el día 60.

# %%
def dia366(indice):
    """Día del año en un calendario de 366 días (en años no bisiestos, desde el 1 de marzo se suma 1)."""
    return indice.dayofyear.to_numpy() + (~indice.is_leap_year & (indice.month >= 3)).astype(int)


def suavizar(curva, ancho):
    """Media móvil circular que ignora los NaN (un día sin climatología no contamina a los vecinos)."""
    mitad = ancho // 2
    ampliada = np.concatenate([curva[-mitad:], curva, curva[:mitad]])
    valido = np.isfinite(ampliada)
    suma = np.convolve(np.where(valido, ampliada, 0.0), np.ones(ancho), mode="valid")
    n = np.convolve(valido.astype(float), np.ones(ancho), mode="valid")
    with np.errstate(invalid="ignore"):
        suave = suma / n
    return np.where(np.isfinite(curva), suave, np.nan)


def climatologia(serie, base=BASE, ventana=31, suavizado=15, minimo_obs=8, percentil=PERCENTIL):
    """DataFrame con índice `dia` (1 a 366): media, `alto` (percentil), `bajo` (100 - percentil) y `n` (observaciones juntadas)."""
    s = serie.dropna()
    s = s[(s.index.year >= base[0]) & (s.index.year <= base[1])]
    if s.empty:
        raise ValueError(f"La serie no tiene datos entre {base[0]} y {base[1]}")
    valores, dia = s.to_numpy(dtype=float), dia366(s.index)
    curvas = {nombre: np.full(366, np.nan) for nombre in ("media", "alto", "bajo")}
    cuenta = np.zeros(366, dtype=int)
    for d in range(1, 367):
        cerca = valores[np.abs((dia - d + 183) % 366 - 183) <= ventana // 2]  # distancia circular: el día 366 es vecino del 1
        cuenta[d - 1] = cerca.size
        if cerca.size >= minimo_obs:
            curvas["media"][d - 1] = cerca.mean()
            curvas["alto"][d - 1], curvas["bajo"][d - 1] = np.percentile(cerca, [percentil, 100 - percentil])
    curvas = {nombre: suavizar(curva, suavizado) for nombre, curva in curvas.items()}
    return pd.DataFrame({**curvas, "n": cuenta}, index=pd.RangeIndex(1, 367, name="dia"))


def anomalias(serie, clima):
    """Al lado de cada observación: media y umbrales de su día del año, anomalía y si excede el umbral cálido o el frío."""
    tabla = clima.loc[dia366(serie.index), ["media", "alto", "bajo"]].set_index(serie.index)
    tabla.insert(0, "valor", serie.to_numpy(dtype=float))
    tabla["anomalia"] = tabla["valor"] - tabla["media"]
    tabla["excede_calor"] = tabla["valor"] > tabla["alto"]
    tabla["excede_frio"] = tabla["valor"] < tabla["bajo"]
    return tabla


st_qc = serie_de(tabla_qc, PUNTO, "st_c")
clima = climatologia(st_qc)
anom = anomalias(st_qc, clima)
print(f"Observaciones por ventana: de {int(clima['n'].min())} a {int(clima['n'].max())} (mediana {int(clima['n'].median())})")
print(f"{PUNTO}: {int(anom['excede_calor'].sum())} observaciones sobre el p{PERCENTIL} y {int(anom['excede_frio'].sum())} bajo el p{100 - PERCENTIL} de {len(anom)}")

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
ax.fill_between(clima.index, clima["bajo"], clima["alto"], color=GRIS, alpha=0.25, label="margen de lo normal")
ax.plot(clima["media"], color="black", label="media")
ax.plot(clima["alto"], color=ROJO, ls="--", label=f"p{PERCENTIL} (umbral cálido)")
ax.plot(clima["bajo"], color=AZUL, ls="--", label=f"p{100 - PERCENTIL} (umbral frío)")
ax.set_ylabel("temperatura de superficie (°C)")
ax.set_title(f"Climatología {BASE[0]}-{BASE[1]}: {PUNTO}")
ax.legend(ncol=2, fontsize=8)
ax2.fill_between(clima.index, clima["n"], color=AZUL, alpha=0.5)
ax2.axhline(8, color=ROJO, ls=":", label="mínimo para calcular (8)")
ax2.set_xlabel("día del año (calendario de 366 días)")
ax2.set_ylabel("observaciones")
ax2.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Episodios.** Una observación suelta no es un episodio: hace falta que **persista**. Un episodio son **al menos `MIN_OBS` observaciones válidas consecutivas** sobre el umbral cálido (o bajo el frío), con las reglas:
#
# * Las fechas con nubes (`NaN`) no cuentan como observación: ni cortan la racha ni la alargan.
# * Entre dos observaciones consecutivas del episodio pueden pasar `BRECHA_MAX_DIAS` días como máximo: una racha **no cruza un hueco largo**.
# * Una observación válida que **no** excede (un dato normal en medio) corta la racha.
# * No es un evento de Hobday: la `duracion` (de la primera a la última observación) es solo una cota mínima, y la tabla **subestima** la frecuencia real.

# %%
def episodios(serie, clima, tipo="calor", min_obs=MIN_OBS, brecha_max=BRECHA_MAX_DIAS):
    """Una fila por episodio: inicio, fin, duración (días), n_obs, día del pico, intensidades (°C) y si sigue en curso."""
    s = serie.dropna()
    dia = dia366(s.index)
    anomalia = s.to_numpy() - clima["media"].loc[dia].to_numpy()
    umbral = clima["alto" if tipo == "calor" else "bajo"].loc[dia].to_numpy()
    evaluable = ~np.isnan(umbral)  # un día sin climatología no se puede evaluar
    s, anomalia, umbral = s[evaluable], anomalia[evaluable], umbral[evaluable]
    excede = s.to_numpy() > umbral if tipo == "calor" else s.to_numpy() < umbral
    rachas, actual = [], []
    for i in range(len(s)):
        if excede[i] and actual and (s.index[i] - s.index[actual[-1]]).days <= brecha_max:
            actual.append(i)  # sigue la racha
        else:
            if len(actual) >= min_obs:
                rachas.append(actual)
            actual = [i] if excede[i] else []  # un dato normal, o un hueco largo, corta la racha
    if len(actual) >= min_obs:
        rachas.append(actual)
    filas = []
    for racha in rachas:
        a = anomalia[racha]
        pico = racha[int(np.argmax(a) if tipo == "calor" else np.argmin(a))]
        filas.append({"tipo": tipo, "inicio": s.index[racha[0]], "fin": s.index[racha[-1]], "duracion": (s.index[racha[-1]] - s.index[racha[0]]).days + 1,
                      "n_obs": len(racha), "dia_pico": s.index[pico], "intensidad_max": anomalia[pico], "intensidad_media": a.mean(),
                      "en_curso": racha[-1] == len(s) - 1})
    return pd.DataFrame(filas, columns=["tipo", "inicio", "fin", "duracion", "n_obs", "dia_pico", "intensidad_max", "intensidad_media", "en_curso"])


# Prueba en series de juguete, donde sabemos la respuesta: climatología plana (media 0, umbrales +1 y -1)
plana = pd.DataFrame({"media": 0.0, "alto": 1.0, "bajo": -1.0}, index=pd.RangeIndex(1, 367, name="dia"))


def juguete(*pares):
    """Serie irregular a partir de pares (fecha, valor); un valor None es una fecha con nubes (NaN)."""
    return pd.Series([np.nan if v is None else v for _, v in pares], index=pd.to_datetime([f for f, _ in pares]), dtype=float)


casos = {
    "2 sobre el umbral, 8 días entre ellas": juguete(("2021-03-01", 0), ("2021-03-09", 2), ("2021-03-17", 2), ("2021-03-25", 0)),
    "2 sobre el umbral, una fecha con nubes en medio": juguete(("2021-03-01", 2), ("2021-03-09", None), ("2021-03-17", 2)),
    "2 sobre el umbral, un dato normal en medio": juguete(("2021-03-01", 2), ("2021-03-09", 0), ("2021-03-17", 2)),
    "2 sobre el umbral, 24 días entre ellas": juguete(("2021-03-01", 2), ("2021-03-25", 2)),
    "2 sobre el umbral, 25 días entre ellas": juguete(("2021-03-01", 2), ("2021-03-26", 2)),
    "una sola observación sobre el umbral": juguete(("2021-03-01", 0), ("2021-03-09", 3), ("2021-03-17", 0)),
}
for nombre, serie_juguete in casos.items():
    ev = episodios(serie_juguete, plana, "calor", min_obs=2, brecha_max=24)
    print(f"{nombre:50s} -> {len(ev)} episodio(s)", [(int(n), int(d)) for n, d in zip(ev["n_obs"], ev["duracion"])])

# %% [markdown]
# Los episodios de cada punto (con su propia serie, climatología y umbrales) en una sola tabla. Cada punto se compara **solo consigo mismo**: una anomalía de 2 °C en el agua es importante, en la tierra es ruido.

# %%
partes = []
for p in puntos.itertuples():
    s = serie_de(tabla_qc, p.id, "st_c")
    try:
        c = climatologia(s)
    except ValueError as error:
        print(f"{p.id}: {error}")
        continue
    for tipo in ("calor", "frio"):
        ev = episodios(s, c, tipo)
        if not ev.empty:
            partes.append(ev.assign(punto=p.id, nombre=p.nombre))
columnas_ev = ["punto", "nombre", "tipo", "inicio", "fin", "duracion", "n_obs", "dia_pico", "intensidad_max", "intensidad_media", "en_curso"]
episodios_todos = (pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=columnas_ev))[columnas_ev].sort_values(["punto", "inicio"]).reset_index(drop=True)
print(f"{len(episodios_todos)} episodios en {episodios_todos['punto'].nunique()} puntos")
if not episodios_todos.empty:
    display(episodios_todos.groupby(["punto", "tipo"]).agg(episodios=("inicio", "size"), obs_max=("n_obs", "max"), duracion_max=("duracion", "max"),
                                                           intensidad_max=("intensidad_max", lambda x: x.loc[x.abs().idxmax()])).round(2))
    activos = episodios_todos[episodios_todos["en_curso"].astype(bool)]
    for a in activos.itertuples():
        print(f"En curso: {a.nombre} ({a.tipo}), desde el {a.inicio.date()}, {a.n_obs} observaciones. Su duración es provisoria: puede crecer o terminar.")

# %%
ev_punto = episodios_todos[episodios_todos["punto"] == PUNTO]
fig, ax = plt.subplots(figsize=(12, 4))
valida = anom["anomalia"].dropna()
ax.plot(valida.index, valida, "o", ms=3, color="black")
for ev in ev_punto.itertuples():
    ax.axvspan(ev.inicio, ev.fin + pd.Timedelta(days=1), color=ROJO if ev.tipo == "calor" else AZUL, alpha=0.5, lw=0)
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("anomalía (°C)")
ax.set_title(f"{PUNTO}: anomalía de cada observación y episodios (rojo: cálido, azul: frío)")
plt.show()

# %% [markdown]
# **Agua frente a tierra, y temperatura con reflectancia.** Landsat mide la superficie: la tierra se calienta y enfría rápido, el agua tiene inercia térmica. Y como medimos temperatura y reflectancia en el mismo instante, podemos preguntar si se mueven juntas (anomalía de la temperatura frente a anomalía de la reflectancia roja). Una correlación negativa puede ser **residuo de nube o neblina** (brilla y enfría la medida) o **agua más turbia y más fría**: los datos no permiten elegir.

# %%
resumen_puntos = tabla_qc.groupby(["id", "tipo"]).agg(st_mediana=("st_c", "median"), st_p5=("st_c", lambda s: s.quantile(0.05)),
                                                      st_p95=("st_c", lambda s: s.quantile(0.95)), ndwi=("ndwi", "median"), ndvi=("ndvi", "median"),
                                                      rojo=("rojo", "median")).round(3)
resumen_puntos["amplitud_p95_p5"] = (resumen_puntos["st_p95"] - resumen_puntos["st_p5"]).round(1)
display(resumen_puntos.sort_values("st_mediana"))

tipos = list(puntos["tipo"].unique())
fig, ejes = plt.subplots(1, 2, figsize=(14, 3.8))
for p in puntos.itertuples():
    s = serie_de(tabla_qc, p.id, "st_c")
    mensual_st = s.groupby(s.index.month).median()
    ejes[0].plot(mensual_st.index, mensual_st, "o-", ms=4, color=PALETA[tipos.index(p.tipo) % len(PALETA)], label=f"{p.id} ({p.tipo})")
ejes[0].set_xticks(range(1, 13), labels=nombres_meses)
ejes[0].set_ylabel("mediana mensual (°C)")
ejes[0].set_title("Ciclo anual de la temperatura de superficie")
ejes[0].legend(fontsize=7, ncol=2)
filas = []
for p in puntos.itertuples():
    try:
        a_st = anomalias(serie_de(tabla_qc, p.id, "st_c"), climatologia(serie_de(tabla_qc, p.id, "st_c")))["anomalia"].dropna()
        a_rojo = anomalias(serie_de(tabla_qc, p.id, "rojo"), climatologia(serie_de(tabla_qc, p.id, "rojo")))["anomalia"].dropna()
    except ValueError:
        continue
    comunes = a_st.index.intersection(a_rojo.index)
    if len(comunes) >= 5:
        filas.append({"punto": p.id, "observaciones": len(comunes), "correlación": round(np.corrcoef(a_st[comunes], a_rojo[comunes])[0, 1], 2)})
        ejes[1].scatter(a_rojo[comunes] * 1000, a_st[comunes], s=10, color=colores[p.id], label=p.id)
ejes[1].set_xlabel("anomalía de la reflectancia roja (x 1000)")
ejes[1].set_ylabel("anomalía de temperatura (°C)")
ejes[1].set_title("Temperatura y turbidez en el mismo instante")
ejes[1].legend(fontsize=7)
plt.tight_layout()
plt.show()
pd.DataFrame(filas).set_index("punto") if filas else None

# %% [markdown]
# **Umbrales clásicos frente a aprendizaje automático.** Los umbrales (p90 y p10) son un método **basado en reglas**: tú decides el percentil, las observaciones y la brecha. **Isolation Forest** es un método de detección de atípicos no supervisado: construye muchos árboles con cortes aleatorios, y los puntos raros quedan aislados con pocos cortes. No conoce el calendario, por eso le damos **anomalías** (sin estacionalidad), y puede mirar **varias variables a la vez**: temperatura, reflectancia roja y NDWI. Hay que fijar la contaminación (`CONTAMINACION`): la fracción de observaciones que esperas atípicas.
#
# El **índice de Jaccard** (observaciones en común / observaciones de la unión) va de 0 a 1 y mide cuánto coinciden los dos métodos. Responden preguntas distintas: el umbral pregunta si esta temperatura, para esta época del año, es poco común; el bosque, si esta combinación de anomalías es rara en toda la serie.

# %%
def anomalias_de(tabla_punto, variables):
    """Anomalía de cada variable respecto de su climatología; solo las fechas con todas las variables."""
    return pd.DataFrame({v: anomalias(tabla_punto[v].dropna(), climatologia(tabla_punto[v]))["anomalia"] for v in variables}).dropna()


def aislamiento(anomalia_variables, contaminacion=CONTAMINACION):
    bosque = IsolationForest(n_estimators=200, contamination=contaminacion, random_state=0).fit(anomalia_variables)
    return pd.Series(bosque.predict(anomalia_variables) == -1, index=anomalia_variables.index)


def jaccard(a, b):
    b = b.reindex(a.index, fill_value=False)
    union = int((a | b).sum())
    return int((a & b).sum()) / union if union else np.nan


filas = []
for p in puntos.itertuples():
    tabla_punto = tabla_qc[tabla_qc["id"] == p.id].set_index("fecha")[["st_c", "rojo", "ndwi"]].astype(float)
    try:
        solo_t, tres = anomalias_de(tabla_punto, ["st_c"]), anomalias_de(tabla_punto, ["st_c", "rojo", "ndwi"])
    except ValueError:
        continue
    if len(tres) < 20:
        continue
    umbral = anomalias(tabla_punto["st_c"].dropna(), climatologia(tabla_punto["st_c"]))
    marca_umbral = umbral["excede_calor"] | umbral["excede_frio"]
    a_solo_t, a_tres = aislamiento(solo_t), aislamiento(tres)
    filas.append({"punto": p.id, "sobre umbral": int(marca_umbral.sum()), "IF solo temperatura": int(a_solo_t.sum()), "IF tres variables": int(a_tres.sum()),
                  "Jaccard solo temperatura": round(jaccard(marca_umbral, a_solo_t), 2), "Jaccard tres variables": round(jaccard(marca_umbral, a_tres), 2)})
    if p.id == PUNTO:
        atipico_tres, anomalias_tres, anomalias_umbral = a_tres, tres, umbral
pd.DataFrame(filas).set_index("punto")

# %%
if "atipico_tres" in globals():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(anomalias_tres.index, anomalias_tres["st_c"], "o", ms=3, color="black", label="anomalía de temperatura")
    raros = anomalias_tres[atipico_tres]
    ax.scatter(raros.index, raros["st_c"], s=40, facecolors="none", edgecolors="#f39c12", lw=1.5, label="atípico (Isolation Forest, 3 variables)")
    marcadas = anomalias_umbral[anomalias_umbral["excede_calor"] | anomalias_umbral["excede_frio"]]
    ax.scatter(marcadas.index, marcadas["anomalia"], s=12, color=ROJO, zorder=3, label="fuera de los umbrales")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("anomalía de temperatura (°C)")
    ax.set_title(f"{PUNTO}: umbrales clásicos (rojo) frente a Isolation Forest con tres variables (naranja)")
    ax.legend(loc="lower left", fontsize=8)
    plt.show()
else:
    print(f"{PUNTO} no tiene suficientes observaciones con las tres variables")

# %% [markdown]
# **Anomalía mensual respecto de una climatología de varios años.** Con una serie larga sí funciona la referencia por mes: el z-score compara cada observación con el mismo mes de los años de `BASE`. Los meses **sin barra** no tienen ninguna observación sin nubes: un mes sin dato no es "anomalía cero", es un vacío.

# %%
base_mes = st_qc[(st_qc.index.year >= BASE[0]) & (st_qc.index.year <= BASE[1])]
grupo_base = base_mes.groupby(base_mes.index.month)
meses_serie = pd.Series(st_qc.index.month, index=st_qc.index)
z_st = (st_qc - meses_serie.map(grupo_base.mean())) / meses_serie.map(grupo_base.std())
z_mensual = z_st[z_st.index >= z_st.index[-1] - pd.DateOffset(years=4)].resample("MS").mean()

fig, eje = plt.subplots(figsize=(11, 3.6))
eje.bar(z_mensual.index, z_mensual.fillna(0), width=25, color=np.where(z_mensual.fillna(0) >= 0, "#D55E00", "#0072B2"))
for mes_vacio in z_mensual.index[z_mensual.isna()]:
    eje.axvspan(mes_vacio, mes_vacio + pd.offsets.MonthBegin(1), color="0.85")
eje.axhline(0, color="k", lw=0.7)
eje.set_ylabel("anomalía mensual (z)")
eje.set_title(f"{PUNTO}: temperatura de superficie frente a la climatología {BASE[0]}-{BASE[1]} (gris: mes sin observación válida)")
plt.show()

# %% [markdown]
# **Guardar los episodios.** Son un dato derivado: si no los guardamos con su documentación, nadie sabrá con qué parámetros salieron.

# %%
ruta_episodios = SALIDAS / "episodios_landsat.parquet"
guardar_parquet(episodios_todos, ruta_episodios, {
    "titulo": f"Episodios cálidos y fríos de la temperatura de superficie en {puntos['id'].nunique()} puntos (Landsat 8 y 9, {FECHAS_LANDSAT[0]} a {FECHAS_LANDSAT[1]})",
    "fuente": "USGS Landsat 8 y 9 Collection 2 Level-2 (banda térmica lwir11) leído desde Microsoft Planetary Computer",
    "licencia": "Dominio público (datos del USGS)",
    "periodo": f"{FECHAS_LANDSAT[0]} a {FECHAS_LANDSAT[1]}", "generado": date.today().isoformat(),
    "metodo": "Anomalías respecto de una climatología por ventana de días; inspirado en Hobday et al. 2016, pero no es una ola de calor marina: Landsat no da una serie diaria",
    "parametros": {"base": list(BASE), "ventana_dias": 31, "suavizado_dias": 15, "percentil_calor": PERCENTIL, "percentil_frio": 100 - PERCENTIL,
                   "observaciones_minimas_por_ventana": 8, "observaciones_seguidas_minimas": MIN_OBS, "brecha_maxima_dias": BRECHA_MAX_DIAS,
                   "control_de_calidad": f"puntos de tipo {TIPOS_AGUA} con reflectancia roja mayor que 0,05 o SWIR1 mayor que 0,02: descartados"},
    "limitaciones": "Landsat pasa cada 8 días y las nubes borran muchas fechas: la duración de un episodio es una cota mínima y la tabla subestima la frecuencia real. "
                    "La banda térmica tiene 100 m nativos y mide la temperatura de la piel de la superficie (errores de 1 K o más).",
    "columnas": {"punto": "id del punto", "nombre": "nombre del punto", "tipo": "calor o frio", "inicio": "fecha de la primera observación del episodio",
                 "fin": "fecha de la última observación", "duracion": "días de la primera a la última observación, ambas incluidas (cota mínima)",
                 "n_obs": "observaciones válidas consecutivas del episodio", "dia_pico": "fecha de mayor anomalía",
                 "intensidad_max": "°C sobre la media climatológica (negativa en frío)", "intensidad_media": "°C",
                 "en_curso": "True si termina en la última observación válida de la serie"}})
print(f"Guardado: {ruta_episodios} ({len(episodios_todos)} episodios)")

# %% [markdown]
# ## Módulo 6. Malla de 500 m: frecuencia de anomalías (opcional)
#
# **Entra:** `AREA_MALLA`, `FECHAS_LANDSAT`, `BASE`, `PERIODO_MALLA`, `T_MIN_MALLA`. **Deja:** mapas de la fracción de observaciones sobre el umbral cálido y bajo el frío. Pon `EJECUTAR_MALLA = True` en la configuración para correrlo.
#
# Hasta ahora miramos puntos; Landsat da imágenes. Cargamos la temperatura **cada 500 m** (una muestra de píxel, sin promediar) y repetimos la climatología **en cada píxel**. Por construcción, en el período base cerca del 10 % de las observaciones supera el p90. Si miras solo los últimos años, una fracción mucho mayor que 0,10 indica píxeles más cálidos de lo normal. Con pocas observaciones por píxel la fracción es ruidosa: mira patrones espaciales, no píxeles sueltos.

# %%
def cargar_malla(area, fechas, resolucion, validos_min=0.10):
    """Temperatura de superficie (time, y, x) de todas las fechas con al menos `validos_min` de píxeles válidos."""
    crs = crs_utm((area[0] + area[2]) / 2, (area[1] + area[3]) / 2)
    items = buscar_landsat(area, fechas, NUBES_MAX_LANDSAT)
    cortes = []
    for anio in sorted({i.datetime.year for i in items}):
        items_anio = [i for i in items if i.datetime.year == anio]
        datos = odc.stac.load(items_anio, bands=["lwir11", "qa_pixel", "qa_radsat"], crs=crs, resolution=resolucion, bbox=area, chunks={"x": 512, "y": 512},
                              groupby="solar_day", patch_url=planetary_computer.sign, fail_on_error=False).compute(scheduler="threads", num_workers=32)
        valida, cubierta = mascara_landsat(datos)
        temperatura = temperatura_landsat(datos["lwir11"]).where(valida)
        escena_util = (cubierta.mean(("x", "y")) > 0.05) & (temperatura.notnull().mean(("x", "y")) >= validos_min)
        cortes.append(temperatura.where(escena_util, drop=True).astype("float32"))
        print(f"{anio}: {int(escena_util.sum())} fechas útiles")
    return xr.concat(cortes, dim="time").sortby("time")


def climatologia_malla(malla, base=BASE, ventana=31, paso=16, minimo_obs=8, percentil=PERCENTIL):
    """Climatología por ventana de días en cada píxel, evaluada cada `paso` días del año (la curva es suave)."""
    tiempo = pd.DatetimeIndex(malla["time"].values)
    en_base = (tiempo.year >= base[0]) & (tiempo.year <= base[1])
    valores, dia = malla.isel(time=np.flatnonzero(en_base)).to_numpy(), dia366(tiempo[en_base])
    centros = np.arange(1, 367, paso)
    salida = {nombre: np.full((len(centros), *valores.shape[1:]), np.nan, dtype="float32") for nombre in ("media", "alto", "bajo")}
    for k, centro in enumerate(centros):
        junto = valores[np.abs((dia - centro + 183) % 366 - 183) <= ventana // 2]
        if junto.shape[0] == 0:
            continue
        suficiente = np.isfinite(junto).sum(axis=0) >= minimo_obs
        alto, bajo = np.nanpercentile(junto, [percentil, 100 - percentil], axis=0)
        for nombre, valor in (("media", np.nanmean(junto, axis=0)), ("alto", alto), ("bajo", bajo)):
            salida[nombre][k] = np.where(suficiente, valor, np.nan)
    return xr.Dataset({n: (("centro", "y", "x"), v) for n, v in salida.items()}, coords={"centro": centros, "y": malla["y"].values, "x": malla["x"].values})


def frecuencia_anomalias(malla, clima, tipo="calor", periodo=PERIODO_MALLA):
    """Fracción de observaciones válidas sobre el umbral cálido (o bajo el frío) en cada píxel, dentro de `periodo`."""
    tiempo = pd.DatetimeIndex(malla["time"].values)
    elegidas = np.flatnonzero((tiempo.year >= periodo[0]) & (tiempo.year <= periodo[1]))
    sub = malla.isel(time=elegidas)
    centro = np.abs((dia366(tiempo[elegidas])[:, None] - clima["centro"].values[None, :] + 183) % 366 - 183).argmin(axis=1)  # centro más cercano
    umbral = clima["alto" if tipo == "calor" else "bajo"].isel(centro=xr.DataArray(centro, dims="time")).assign_coords(time=sub["time"])
    con_dato = sub.notnull() & umbral.notnull()
    excede = (sub > umbral) if tipo == "calor" else (sub < umbral)
    n_obs = con_dato.sum("time")
    return (excede & con_dato).sum("time") / n_obs.where(n_obs > 0), n_obs


# %%
if EJECUTAR_MALLA:
    t0 = time.time()
    malla_bruta = cargar_malla(AREA_MALLA, FECHAS_LANDSAT, RESOLUCION_MALLA)
    muy_frio = float((malla_bruta < T_MIN_MALLA).sum() / malla_bruta.notnull().sum())
    print(f"{malla_bruta.sizes['time']} fechas, {malla_bruta.sizes['y']} x {malla_bruta.sizes['x']} píxeles ({time.time() - t0:.0f} s)")
    print(f"{muy_frio:.1%} de los valores son menores que {T_MIN_MALLA} °C: se descartan (casi todos nubes sin marcar)")
    malla = malla_bruta.where(malla_bruta >= T_MIN_MALLA)  # control declarado, sin reflectancia (la malla solo guarda la banda térmica)

    clima_malla = climatologia_malla(malla)
    frec_calor, n_calor = frecuencia_anomalias(malla, clima_malla, "calor")
    frec_frio, _ = frecuencia_anomalias(malla, clima_malla, "frio")
    print(f"Observaciones por píxel en {PERIODO_MALLA[0]}-{PERIODO_MALLA[1]}: mediana {int(n_calor.median())}")
    print(f"Fracción sobre el umbral cálido: mediana {float(frec_calor.median()):.2f} | bajo el frío: {float(frec_frio.median()):.2f} (esperado {1 - PERCENTIL / 100:.2f})")

    crs_malla = crs_utm((AREA_MALLA[0] + AREA_MALLA[2]) / 2, (AREA_MALLA[1] + AREA_MALLA[3]) / 2)
    x_km, y_km = malla["x"] / 1000, malla["y"] / 1000
    fig, ejes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    for eje, capa, titulo, cmap, vmax in ((ejes[0], malla.notnull().sum("time"), "Observaciones válidas por píxel", "viridis", None),
                                          (ejes[1], frec_calor, f"Fracción sobre el p{PERCENTIL}, {PERIODO_MALLA[0]}-{PERIODO_MALLA[1]}", "Reds", 0.5),
                                          (ejes[2], frec_frio, f"Fracción bajo el p{100 - PERCENTIL}, {PERIODO_MALLA[0]}-{PERIODO_MALLA[1]}", "Blues", 0.5)):
        imagen = eje.pcolormesh(x_km, y_km, capa, cmap=cmap, vmin=0, vmax=vmax, shading="nearest")
        for p in puntos.itertuples():
            px, py = a_utm(p.lon, p.lat, crs_malla)
            eje.plot(px / 1000, py / 1000, "k^", ms=5, mec="white")
        eje.set_title(titulo)
        eje.set_xlabel("este (km)")
        eje.set_aspect("equal")
        plt.colorbar(imagen, ax=eje, shrink=0.75)
    ejes[0].set_ylabel("norte (km)")
    plt.tight_layout()
    plt.show()
else:
    print("Módulo 6 omitido (EJECUTAR_MALLA = False)")

# %% [markdown]
# ## Resultados guardados

# %%
for archivo in sorted(SALIDAS.iterdir()):
    print(f"{archivo.name:32s} {archivo.stat().st_size / 1024:8.0f} kB")
