# %% [markdown]
# # Acceso a Imágenes Satelitales
#
# En este cuaderno aprenderemos a acceder y visualizar imágenes satelitales utilizando Planetary Computer.

# %% [markdown]
# ## Configuración del entorno

# %%
# @title Instalación de paquetes necesarios
# Necesitamos instalar las librerías necesarias para trabajar con imágenes satelitales
# %pip install rioxarray xarray matplotlib numpy rasterio xarray-spatial geopandas planetary-computer pystac-client odc-stac

# %%
# @title Importación de bibliotecas
# Para trabajar con este notebook en Google Colab, debemos importar las librerías y clonar el repositorio completo:
#
import os

import matplotlib.pyplot as plt
import numpy as np
import odc.stac
import planetary_computer
from pystac_client import Client

# Configuración para visualización
plt.rcParams["figure.figsize"] = (12, 8)
plt.style.use("ggplot")

# Clonamos el repositorio
os.system("git clone https://github.com/alvaroparedesl/geomatica-aplicada.git")
# %cd geomatica-aplicada

# %% [markdown]
# ## 1. Introducción a Planetary Computer
#
# Microsoft Planetary Computer es una plataforma que proporciona acceso a datos ambientales y climáticos, incluyendo:
#
# * Imágenes satelitales (Landsat, Sentinel)
# * Datos climáticos
# * Datos de elevación
# * Datos de cobertura terrestre
#
# Ventajas:
# * Acceso gratuito
# * Datos listos para usar
# * API fácil de usar
# * Integración con herramientas de análisis

# %%
# Conectamos al catálogo STAC de Planetary Computer
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# %% [markdown]
# ## 2. Búsqueda de Imágenes Sentinel-2
#
# Vamos a buscar imágenes Sentinel-2 para un área de interés en Chile.

# %%
# Definimos el área de interés
bbox = [-71.399460, -34.366111, -70.633850, -34.084512]

# Definimos el período de tiempo
time_range = "2023-01-01/2023-01-31"

# Realizamos la búsqueda
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime=time_range,
    query={"eo:cloud_cover": {"lt": 20}},  # Menos de 20% de nubes
)

# Convertimos los resultados a un DataFrame para visualización
items = list(search.get_items())
print(f"Encontradas {len(items)} imágenes")

# Mostramos información de la primera imagen
item = items[0]
print("\nInformación de la primera imagen:")
print(f"Fecha: {item.datetime.strftime('%Y-%m-%d')}")
print(f"Cobertura de nubes: {item.properties['eo:cloud_cover']}%")
print(f"ID de la imagen: {item.id}")

# %% [markdown]
# ## 3. Acceso y Visualización de Imágenes Sentinel-2
#
# Vamos a cargar y visualizar una imagen Sentinel-2. El satélite Sentinel-2 tiene 13 bandas espectrales:
#
# | Banda | Longitud de onda (nm) | Resolución (m) | Descripción |
# |-------|----------------------|----------------|-------------|
# | B01   | 443                  | 60            | Aerosoles   |
# | B02   | 490                  | 10            | Azul        |
# | B03   | 560                  | 10            | Verde       |
# | B04   | 665                  | 10            | Rojo        |
# | B05   | 705                  | 20            | Red Edge 1  |
# | B06   | 740                  | 20            | Red Edge 2  |
# | B07   | 783                  | 20            | Red Edge 3  |
# | B08   | 842                  | 10            | NIR         |
# | B8A   | 865                  | 20            | Red Edge 4  |
# | B09   | 940                  | 60            | Vapor agua  |
# | B10   | 1375                 | 60            | Cirrus      |
# | B11   | 1610                 | 20            | SWIR 1      |
# | B12   | 2190                 | 20            | SWIR 2      |

# %%
# Cargamos las bandas RGB (color verdadero)
rgb_bands = ["B04", "B03", "B02"]  # Rojo, Verde, Azul

# Cargamos los datos usando ODC
ds = odc.stac.load(
    items,
    bands=rgb_bands,
    bbox=bbox,
    crs="EPSG:32719",  # UTM Zone 19S
    resolution=10,  # 10m resolución
    group_by="solar_day",  # Agrupamos por día solar para evitar duplicados
    chunks={"x": 2048, "y": 2048},  # Tamaño de los chunks
)
ds

# %%
# Seleccionamos una fecha específica
fecha = ds.time[0]
imagen = ds.sel(time=fecha)

# Visualizamos la imagen RGB
# 1. to_array(): convierte el Dataset a DataArray
# 2. transpose(): reordena de (band,y,x) a (y,x,band) para visualización
# 3. * 0.0001: factor de escala para convertir a reflectancia
# 4. * 3.5: factor de mejora de brillo (ajustable). Otra alternativa: np.cbrt(0.6 * band)
# 5. clip(): asegura valores entre 0 y 1
rgb = np.clip(imagen.to_array().values.transpose(1, 2, 0) * 0.0001 * 3.5, 0, 1)

plt.figure(figsize=(15, 10))
im = plt.imshow(rgb)
plt.title(f"Imagen Sentinel-2 RGB - Color verdadero ({fecha.values})")
plt.axis("off")
plt.show()

# %%
# Visualización en falso color (NIR)
nir_bands = ["B08", "B04", "B03"]  # NIR, Rojo, Verde

# Cargamos los datos
ds_nir = odc.stac.load(
    items,
    bands=nir_bands,
    bbox=bbox,
    crs="EPSG:32719",
    resolution=10,
    group_by="solar_day",
    chunks={"x": 2048, "y": 2048},  # Tamaño de los chunks
)

# Seleccionamos la misma fecha
imagen_nir = ds_nir.sel(time=fecha)

# Visualizamos la imagen en falso color
nir_rgb = np.clip(imagen_nir.to_array().values.transpose(1, 2, 0) * 0.0001 * 3.5, 0, 1)

plt.figure(figsize=(15, 10))
im = plt.imshow(nir_rgb)
plt.title(f"Imagen Sentinel-2 - Falso color NIR ({fecha.values})")
plt.axis("off")
plt.show()

# %% [markdown]
# ## 4. Acceso a Imágenes Landsat
#
# Ahora vamos a buscar y visualizar imágenes Landsat para la misma región.

# %%
# Búsqueda de imágenes Landsat
search_landsat = catalog.search(
    collections=["landsat-c2-l2"],
    bbox=bbox,
    datetime=time_range,
    query={
        "eo:cloud_cover": {"lt": 20},
        "platform": {"in": ["landsat-8", "landsat-9"]},
    },
)

items_landsat = list(search_landsat.get_items())
print(f"Encontradas {len(items_landsat)} imágenes Landsat")

info = items_landsat[0].assets["blue"].to_dict()["raster:bands"][0]

if len(items_landsat) > 0:
    # Mostramos información de la primera imagen
    item_landsat = items_landsat[0]
    print("\nInformación de la primera imagen Landsat:")
    print(f"Fecha: {item_landsat.datetime.strftime('%Y-%m-%d')}")
    print(f"Cobertura de nubes: {item_landsat.properties['eo:cloud_cover']}%")
    print(f"ID de la imagen: {item_landsat.id}")

    # Cargamos las bandas RGB de Landsat
    rgb_bands_landsat = ["red", "green", "blue"]  # Rojo, Verde, Azul

    # Cargamos los datos usando ODC
    ds_landsat = odc.stac.load(
        items_landsat,
        bands=rgb_bands_landsat,
        bbox=bbox,
        crs="EPSG:32719",
        resolution=30,  # 30m resolución para Landsat
        group_by="solar_day",
        chunks={"x": 2048, "y": 2048},  # Tamaño de los chunks
    )

    # Seleccionamos una fecha
    fecha_landsat = ds_landsat.time[1]
    imagen_landsat = ds_landsat.sel(time=fecha_landsat)

    # Visualizamos la imagen RGB de Landsat
    rgb_landsat = np.clip(
        (
            imagen_landsat.to_array().values.transpose(1, 2, 0) * info["scale"]
            + info["offset"]
        )
        * 3.5,
        0,
        1,
    )

    plt.figure(figsize=(15, 10))
    im = plt.imshow(rgb_landsat)
    plt.title(f"Imagen Landsat 8 RGB - Color verdadero ({fecha_landsat.values})")
    plt.axis("off")
    plt.show()

# %% [markdown]
# ## 5. Comparación entre Sentinel-2 y Landsat
#
# Principales diferencias:
#
# * **Resolución espacial**:
#   * Sentinel-2: 10m en bandas visibles y NIR
#   * Landsat 8: 30m en la mayoría de las bandas
# * **Resolución temporal**:
#   * Sentinel-2: 5 días (con dos satélites)
#   * Landsat 8: 16 días
# * **Bandas espectrales**:
#   * Sentinel-2: 13 bandas
#   * Landsat 8: 11 bandas
# * **Cobertura**:
#   * Sentinel-2: Optimizado para tierra
#   * Landsat 8: Global, incluye océanos

# %% [markdown]
# ## Ejercicios
#
# 1. Busca imágenes Sentinel-2 para otra región de Chile y visualízalas en color verdadero.
# 2. Experimenta con diferentes combinaciones de bandas para crear visualizaciones en falso color.
# 3. Compara la misma área en imágenes Sentinel-2 y Landsat. ¿Puedes notar las diferencias en resolución?
# 4. Modifica los parámetros de búsqueda (fechas, cobertura de nubes) y analiza cómo afectan los resultados.
