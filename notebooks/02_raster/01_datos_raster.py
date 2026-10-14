# %% [markdown]
# # Datos Raster: Conceptos Básicos
#
# [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/notebooks/02_raster/01_datos_raster.ipynb) [![Licencia: GPL-3.0](https://img.shields.io/badge/licencia-GPL--3.0-blue.svg)](https://github.com/alvaroparedesl/geomatica-aplicada/blob/main/LICENSE) [![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://colab.research.google.com/) [![Último commit](https://img.shields.io/github/last-commit/alvaroparedesl/geomatica-aplicada?label=%C3%BAltimo%20commit)](https://github.com/alvaroparedesl/geomatica-aplicada/commits/main)
#
# En este cuaderno aprenderemos los conceptos fundamentales de los datos raster, su estructura y cómo manipularlos con Python.

# %% [markdown]
# ## Configuración del entorno
#
# Primero preparamos el entorno: la celda `Preparar entorno` deja disponible la carpeta `utils/` del curso, donde guardamos funciones de apoyo (aquí usaremos `conectar_catalogo` para conectarnos a Planetary Computer).

# %%
# @title Preparar entorno (ejecutar primero)
import subprocess
import sys
from pathlib import Path

if "google.colab" in sys.modules:  # Colab: clonar el repositorio (y actualizarlo si ya estaba)
    RAIZ = Path("/content/geomatica-aplicada")
    if not RAIZ.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/alvaroparedesl/geomatica-aplicada", str(RAIZ)], check=True)
    else:  # una sesión anterior ya lo clonó: traer lo nuevo para no usar código viejo
        subprocess.run(["git", "-C", str(RAIZ), "pull", "-q", "--ff-only"], check=False)
else:  # Local: subir desde la carpeta actual hasta encontrar utils/
    RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "utils").is_dir())
sys.path.insert(0, str(RAIZ))

# %%
# @title Instalación de paquetes necesarios
# Necesitamos instalar las librerías necesarias para trabajar con datos raster.
# %pip install -q rioxarray xarray matplotlib numpy rasterio xarray-spatial geopandas pystac-client planetary-computer odc-stac

# %%
# @title Importación de bibliotecas
# Los datos reales (cobertura de suelo y modelo de elevación) los descargaremos desde el
# catálogo STAC de Microsoft Planetary Computer, por lo que no necesitamos archivos locales.
import os

import matplotlib.pyplot as plt
import numpy as np
import odc.stac
import rioxarray as rxr  # noqa: F401  (activa el accesor .rio)
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

# Importamos xarray-spatial para cálculos de productos derivados de DEM
import xrspatial

# Función de apoyo del curso (carpeta utils/)
from utils.stac import conectar_catalogo

# Configuración para visualización
plt.rcParams["figure.figsize"] = (12, 8)
plt.style.use("ggplot")

# Conexión al catálogo STAC de Microsoft Planetary Computer
# (firma automáticamente las URLs de descarga; en el cuaderno 03 vemos cómo se hace paso a paso)
catalogo = conectar_catalogo("planetary")

# Área de estudio: precordillera al este de Santiago (cuenca del Maipo), Chile central.
# Es un área pequeña, con relieve marcado y distintas coberturas (ciudad, matorral, cultivos).
bbox = [-70.60, -33.55, -70.45, -33.40]  # [oeste, sur, este, norte]


# %% [markdown]
# ## 1. ¿Qué son los datos raster?
#
# Los datos raster representan el mundo como una matriz regular de celdas o píxeles. Cada celda contiene un valor que representa información sobre esa área geográfica.
#
# Ejemplos de datos raster incluyen:
# * Imágenes satelitales
# * Modelos digitales de elevación (DEM)
# * Mapas de temperatura
# * Mapas de precipitación
# * Mapas de uso y cobertura del suelo

# %% [markdown]
# ![Representación de datos raster](https://raw.githubusercontent.com/carpentries-incubator/geospatial-python/gh-pages/fig/raster_concept.png)

# %% [markdown]
# ## 2. Estructura de los datos raster
#
# Los datos raster tienen los siguientes componentes principales:
#
# * **Celdas (píxeles)**: Unidades básicas que contienen valores
# * **Resolución espacial**: Tamaño de cada celda en unidades del mundo real
# * **Extensión**: Área geográfica cubierta por el raster
# * **Sistema de coordenadas**: Define cómo se proyecta el raster en la superficie terrestre
# * **Bandas**: Capas de información (por ejemplo, RGB en una imagen a color)
# * **Valores de datos**: Información almacenada en cada celda

# %% [markdown]
# ## 3. Creación de un raster simple
#
# Vamos a crear un raster simple utilizando NumPy y rioxarray:

# %%
# Creamos una matriz de datos simple (elevación simulada)
datos = np.zeros((100, 100))

# Agregamos algunas características
for i in range(100):
    for j in range(100):
        # Simulamos una colina
        dist_centro = np.sqrt((i - 50) ** 2 + (j - 50) ** 2)
        datos[i, j] = 100 - dist_centro * 1.5

        # Agregamos algo de ruido
        datos[i, j] += np.random.normal(0, 5)

# Valores negativos los convertimos a 0 (nivel del mar)
datos[datos < 0] = 0

# Convertimos a un DataArray de xarray
raster = xr.DataArray(
    data=datos, dims=["y", "x"], coords={"y": np.arange(100), "x": np.arange(100)}
)

# Visualizamos el raster
plt.figure(figsize=(10, 8))
im = raster.plot(cmap="terrain", add_colorbar=False)
plt.title("Modelo Digital de Elevación Simulado")
plt.xlabel("Coordenada X")
plt.ylabel("Coordenada Y")
plt.colorbar(im, label="Elevación (m)")
plt.show()

# %% [markdown]
# ## 4. Trabajando con datos reales: Cobertura de suelo de Chile
#
# Vamos a trabajar con datos de cobertura de suelo de la zona central de Chile (precordillera de Santiago, cuenca del Maipo). Usaremos el producto **ESA WorldCover** (año 2021, 10 m de resolución), que clasifica la superficie terrestre a partir de imágenes satelitales Sentinel-1 y Sentinel-2.
#
# En lugar de descargar un archivo, lo buscaremos en el catálogo STAC de Microsoft Planetary Computer (colección `esa-worldcover`) y cargaremos solo el área que nos interesa:

# %%
# Buscamos el producto de cobertura de suelo que cubre nuestra área de estudio
busqueda_cobertura = catalogo.search(
    collections=["esa-worldcover"],
    bbox=bbox,
    datetime="2021-01-01/2021-12-31",
)
items_cobertura = busqueda_cobertura.item_collection()
print(f"Productos de cobertura encontrados: {len(items_cobertura)}")

# %% [markdown]
# ### Cargando y explorando los datos de cobertura de suelo

# %%
# Cargamos la cobertura (solo el área del bbox) con odc-stac.
# Usamos remuestreo "nearest" porque son clases (categorías), no valores continuos.
cobertura = odc.stac.load(
    items_cobertura,
    bands=["map"],
    bbox=bbox,
    resampling="nearest",
)["map"].isel(time=0).compute()

# Renombramos las dimensiones a x/y y asignamos el sistema de coordenadas (lat/lon, WGS84)
cobertura = cobertura.rename({"longitude": "x", "latitude": "y"}).rio.write_crs("EPSG:4326")

# Clases de ESA WorldCover: valor -> (nombre, color)
clases_cobertura = {
    10: ("Bosque", "#006400"),
    20: ("Matorral", "#FFBB22"),
    30: ("Pastizal", "#FFFF4C"),
    40: ("Cultivos", "#F096FF"),
    50: ("Áreas construidas", "#FA0000"),
    60: ("Suelo desnudo / vegetación escasa", "#B4B4B4"),
    70: ("Nieve y hielo", "#F0F0F0"),
    80: ("Agua permanente", "#0064C8"),
    90: ("Humedal herbáceo", "#0096A0"),
    95: ("Manglar", "#00CF75"),
    100: ("Musgo y líquenes", "#FAE6A0"),
}
valores_clases = list(clases_cobertura.keys())
cmap_cobertura = ListedColormap([c for _, c in clases_cobertura.values()])
norm_cobertura = BoundaryNorm(
    [v - 0.5 for v in valores_clases] + [valores_clases[-1] + 0.5], cmap_cobertura.N
)

# Información básica del raster
print("Información del raster de cobertura de suelo:")
print(f"Dimensiones: {cobertura.shape}")
print(f"Sistema de coordenadas: {cobertura.rio.crs}")
print(f"Resolución: {cobertura.rio.resolution()}")
print(f"Bounds: {cobertura.rio.bounds()}")

# Visualizamos la cobertura
plt.figure(figsize=(12, 8))
im = cobertura.plot(cmap=cmap_cobertura, norm=norm_cobertura, add_colorbar=False)
plt.title("Cobertura de Suelo - Precordillera de Santiago (ESA WorldCover 2021)")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
presentes = [v for v in valores_clases if v in np.unique(cobertura.values)]
plt.legend(
    handles=[
        Patch(color=clases_cobertura[v][1], label=f"{v} - {clases_cobertura[v][0]}")
        for v in presentes
    ],
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)
plt.show()

# Calculamos estadísticas básicas
print("\nEstadísticas de la cobertura de suelo:")
valores_unicos = np.unique(cobertura.values)
print(f"Clases de cobertura presentes: {valores_unicos}")

# %% [markdown]
# ¿Qué significan estos valores?
#
# | Valor | Clase (ESA WorldCover) |
# |-------|------------------------|
# | 10    | Bosque                 |
# | 20    | Matorral               |
# | 30    | Pastizal               |
# | 40    | Cultivos               |
# | 50    | Áreas construidas      |
# | 60    | Suelo desnudo / vegetación escasa |
# | 70    | Nieve y hielo          |
# | 80    | Agua permanente        |
# | 90    | Humedal herbáceo       |
# | 95    | Manglar                |
# | 100   | Musgo y líquenes       |
#
# Información detallada la puedes encontrar en la [página de ESA WorldCover](https://esa-worldcover.org/en).
# Para el caso de Chile, también existe el proyecto [MapBiomas Chile](https://chile.mapbiomas.org/codigos-de-la-leyenda/), con una leyenda propia.

# %% [markdown]
# ## 5. Operaciones básicas con rasters
#
# Ahora vamos a realizar algunas operaciones básicas con nuestro raster:

# %%
# 1. Recorte de un área de interés
raster_recortado = cobertura.rio.clip_box(
    minx=cobertura.x.min() + (cobertura.x.max() - cobertura.x.min()) * 0.25,
    miny=cobertura.y.min() + (cobertura.y.max() - cobertura.y.min()) * 0.25,
    maxx=cobertura.x.min() + (cobertura.x.max() - cobertura.x.min()) * 0.75,
    maxy=cobertura.y.min() + (cobertura.y.max() - cobertura.y.min()) * 0.75,
)

# Visualizamos el recorte
plt.figure(figsize=(10, 8))
im = raster_recortado.plot(cmap=cmap_cobertura, norm=norm_cobertura, add_colorbar=False)
plt.title("Área de Interés Recortada")
plt.show()

# 2. Cálculo de estadísticas básicas
# Nota: aquí los valores son códigos de clase (categorías), por lo que el promedio
# y la desviación estándar son solo un ejercicio de sintaxis; no tienen interpretación física.
print("\nEstadísticas del raster:")
print(f"Valor mínimo: {cobertura.min().values}")
print(f"Valor máximo: {cobertura.max().values}")
print(f"Valor promedio: {cobertura.mean().values:.2f}")
print(f"Desviación estándar: {cobertura.std().values:.2f}")

# %% [markdown]
# ## 6. Análisis de terreno con xarray-spatial
#
# Si estamos trabajando con un DEM, podemos utilizar xarray-spatial para calcular productos derivados como pendiente, orientación y sombreado del relieve.
#
# Para el modelo digital de elevación usaremos el **Copernicus DEM GLO-30** (30 m de resolución), disponible en Planetary Computer como la colección `cop-dem-glo-30`.

# %%
# Buscamos y cargamos el DEM para la misma área de estudio
busqueda_dem = catalogo.search(collections=["cop-dem-glo-30"], bbox=bbox)
items_dem = busqueda_dem.item_collection()
print(f"Tiles de DEM encontrados: {len(items_dem)}")

dem = odc.stac.load(items_dem, bands=["data"], bbox=bbox)["data"].isel(time=0).compute()
dem = dem.rename({"longitude": "x", "latitude": "y"}).rio.write_crs("EPSG:4326")

# Visualizamos el DEM
plt.figure(figsize=(12, 8))
im = dem.plot(cmap="terrain", add_colorbar=False)
plt.title("Modelo Digital de Elevación (Ejemplo)")
plt.colorbar(im, label="Elevación (m)")
plt.show()

# %%
# Antes de calcular algunos derivados, vamos a reproyectar el DEM ¿por qué?
# (pista: las pendientes necesitan que x, y y la elevación estén en las mismas unidades, metros)
dem = dem.rio.reproject("EPSG:32719")
# %%
# Calculamos la pendiente con xarray-spatial
pendiente = xrspatial.slope(dem)

# Visualizamos la pendiente
plt.figure(figsize=(10, 8))
im = pendiente.plot(cmap="YlOrRd", add_colorbar=False)
plt.title("Mapa de Pendientes")
plt.colorbar(im, label="Pendiente (grados)")
plt.show()

# %%
# Calculamos la orientación (aspecto)
aspecto = xrspatial.aspect(dem)

# Visualizamos la orientación
plt.figure(figsize=(10, 8))
im = aspecto.plot(cmap="twilight", add_colorbar=False)
plt.title("Mapa de Orientación")
plt.colorbar(im, label="Orientación (grados)")
plt.show()

# %%
# Calculamos el sombreado del relieve (hillshade)
sombreado = xrspatial.hillshade(dem)

# Visualizamos el sombreado
plt.figure(figsize=(10, 8))
im = sombreado.plot(cmap="gray", add_colorbar=False)
plt.title("Sombreado del Relieve")
plt.colorbar(im)
plt.show()

# Visualización combinada: DEM con sombreado
fig, ax = plt.subplots(figsize=(12, 8))
dem_plot = dem.plot(cmap="terrain", alpha=0.6, ax=ax, add_colorbar=False)
sombreado.plot.imshow(cmap="gray", alpha=0.4, ax=ax, add_colorbar=False)
plt.title("DEM con Sombreado del Relieve")
plt.colorbar(dem_plot, label="Elevación (m)")
plt.show()

# %% [markdown]
# ## 7. Guardando un raster
#
# Finalmente, vamos a guardar nuestro raster procesado:

# %%
# Creamos un directorio para guardar los resultados si no existe
directorio_resultados = "resultados/raster"
os.makedirs(directorio_resultados, exist_ok=True)

# Guardamos el raster recortado como un nuevo archivo GeoTIFF
raster_path = os.path.join(directorio_resultados, "raster_recortado.tif")

# Asignamos el CRS del raster original
raster_recortado.rio.write_crs(cobertura.rio.crs, inplace=True)

# Guardamos el archivo
raster_recortado.rio.to_raster(raster_path)

print(f"Raster guardado en: {raster_path}")

# %% [markdown]
# ## 8. Resumen y conceptos clave
#
# En este cuaderno hemos aprendido:
#
# * Qué son los datos raster y sus componentes principales
# * Cómo crear y visualizar rasters con xarray y rioxarray
# * Cómo cargar datos reales (cobertura de suelo y DEM de Chile central) desde un catálogo STAC
# * Operaciones básicas: recorte y estadísticas
# * Análisis de terreno con xarray-spatial: pendiente, orientación y sombreado
# * Guardar rasters procesados
#
# En los próximos cuadernos, exploraremos la teledetección y cómo trabajar con imágenes satelitales.

# %% [markdown]
# ## Ejercicios
#
# 1. Cambia el `bbox` para cargar el DEM de otra región de Chile desde Planetary Computer y realiza un análisis básico.
# 2. Si trabajas con datos de cobertura de suelo, calcula el porcentaje de cada clase de cobertura en el área de estudio.
# 3. Utiliza xarray-spatial para calcular otros productos derivados del DEM, como la curvatura o la rugosidad del terreno.
# 4. Crea un mapa que combine la cobertura de suelo con el sombreado del relieve para visualizar mejor la relación entre el uso del suelo y la topografía.
