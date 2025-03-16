# %% [markdown]
# # Datos Raster: Conceptos Básicos
#
# En este cuaderno aprenderemos los conceptos fundamentales de los datos raster, su estructura y cómo manipularlos con Python.

# %% [markdown]
# ## Configuración del entorno


# %%
# @title Instalación de paquetes necesarios
# Necesitamos instalar las librerías necesarias para trabajar con datos raster.
# %pip install rioxarray xarray matplotlib numpy rasterio xarray-spatial geopandas

# %%
# @title Importación de bibliotecas
# Para trabajar con este notebook en Google Colab, debemos importar las librerías y clonar el repositorio completo para tener acceso a todos los datos y archivos necesarios:
#
import os

import matplotlib.pyplot as plt
import numpy as np
import rioxarray as rxr
import xarray as xr

# Importamos xarray-spatial para cálculos de productos derivados de DEM
import xrspatial

# Configuración para visualización
plt.rcParams["figure.figsize"] = (12, 8)
plt.style.use("ggplot")

# Clonamos el repositorio
os.system("git clone https://github.com/alvaroparedesl/geomatica-aplicada.git")
# %cd geomatica-aplicada


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
# Vamos a trabajar con un archivo de cobertura de suelo de Chile, que es un subconjunto del proyecto MapBiomas para el año 2018. MapBiomas es una iniciativa que mapea la cobertura y uso del suelo en América Latina utilizando imágenes satelitales.
#
# Al haber clonado el repositorio, tenemos acceso directo al archivo de cobertura de suelo:

# %%
# Ruta al archivo de cobertura de suelo
archivo_cobertura = "data/raster/chile_coverage_2018s.tif"

# %% [markdown]
# ### Cargando y explorando los datos de cobertura de suelo

# %%
# Cargamos el archivo de cobertura con rioxarray
cobertura = rxr.open_rasterio(archivo_cobertura)

# Información básica del raster
print("Información del raster de cobertura de suelo:")
print(f"Dimensiones: {cobertura.shape}")
print(f"Sistema de coordenadas: {cobertura.rio.crs}")
print(f"Resolución: {cobertura.rio.resolution()}")
print(f"Bounds: {cobertura.rio.bounds()}")

# Visualizamos la cobertura
plt.figure(figsize=(12, 8))
im = cobertura.squeeze().plot(cmap="terrain", add_colorbar=False)
plt.title("Cobertura de Suelo de Chile (2018)")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.colorbar(im, label="Clase de cobertura")
plt.show()

# Calculamos estadísticas básicas
print("\nEstadísticas de la cobertura de suelo:")
valores_unicos = np.unique(cobertura.values)
print(f"Clases de cobertura presentes: {valores_unicos}")

# %% [markdown]
# ¿Qué significan estos valores?
#
# | Clase Nivel 2                     | DV  |
# |-----------------------------------|-----|
# | 1.1 Bosque                       | 3   |
# | 2.1 Humedal                      | 11  |
# | 2.2 Pastizal                     | 12  |
# | 2.3 Matorral                     | 66  |
# | 2.4 Afloramiento rocoso          | 29  |
# | 3.1 Plantación Forestal          | 9   |
# | 3.2 Mosaico de agricultura y pastura | 21  |
# | 4.1 Infraestructura              | 24  |
# | 4.2 Arenas, Playas y Dunas       | 23  |
# | 4.3 Salar                        | 61  |
# | 4.4 Otra área sin vegetación     | 25  |
# | 5.1 Río, lago u océano           | 33  |
# | 5.2 Hielo y nieve                | 34  |
# | No observado                     | 27  |
#
# Información detallada la podemos encontrar en este [linl](https://chile.mapbiomas.org/codigos-de-la-leyenda/)
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
im = raster_recortado.squeeze().plot(add_colorbar=False)
plt.title("Área de Interés Recortada")
plt.colorbar(im)
plt.show()

# 2. Cálculo de estadísticas básicas
print("\nEstadísticas del raster:")
print(f"Valor mínimo: {cobertura.min().values}")
print(f"Valor máximo: {cobertura.max().values}")
print(f"Valor promedio: {cobertura.mean().values:.2f}")
print(f"Desviación estándar: {cobertura.std().values:.2f}")

# %% [markdown]
# ## 6. Análisis de terreno con xarray-spatial
#
# Si estamos trabajando con un DEM, podemos utilizar xarray-spatial para calcular productos derivados como pendiente, orientación y sombreado del relieve.

# %%
archivo_dem = "data/raster/Copernicus_DSM_COG_10_S35_00_W072_00_DEM.tif.tif"

dem = rxr.open_rasterio(archivo_cobertura).squeeze()

# Visualizamos el DEM
plt.figure(figsize=(12, 8))
im = dem.plot(cmap="terrain", add_colorbar=False)
plt.title("Modelo Digital de Elevación (Ejemplo)")
plt.colorbar(im, label="Elevación (m)")
plt.show()

# %%
# Antes de calcular algunos derivados, vamos a reproyectar el DEM ¿por qué?
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
dem_data = cobertura.squeeze()
dem_plot = dem_data.plot(cmap="terrain", alpha=0.6, ax=ax, add_colorbar=False)
sombreado.plot.imshow(cmap="gray", alpha=0.4, ax=ax)
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
# * Cómo trabajar con datos reales de cobertura de suelo de Chile
# * Operaciones básicas: recorte y estadísticas
# * Análisis de terreno con xarray-spatial: pendiente, orientación y sombreado
# * Guardar rasters procesados
#
# En los próximos cuadernos, exploraremos la teledetección y cómo trabajar con imágenes satelitales.

# %% [markdown]
# ## Ejercicios
#
# 1. Descarga otro conjunto de datos raster (por ejemplo, un DEM de otra región de Chile) y realiza un análisis básico.
# 2. Si trabajas con datos de cobertura de suelo, calcula el porcentaje de cada clase de cobertura en el área de estudio.
# 3. Utiliza xarray-spatial para calcular otros productos derivados del DEM, como la curvatura o la rugosidad del terreno.
# 4. Crea un mapa que combine la cobertura de suelo con el sombreado del relieve para visualizar mejor la relación entre el uso del suelo y la topografía.
# 4. Crea un mapa que combine la cobertura de suelo con el sombreado del relieve para visualizar mejor la relación entre el uso del suelo y la topografía.
