# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Operaciones Espaciales con GeoPandas
#
# [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/notebooks/01_vector/03_operaciones_espaciales.ipynb) [![Licencia: GPL-3.0](https://img.shields.io/badge/licencia-GPL--3.0-blue.svg)](https://github.com/alvaroparedesl/geomatica-aplicada/blob/main/LICENSE) [![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://colab.research.google.com/) [![Último commit](https://img.shields.io/github/last-commit/alvaroparedesl/geomatica-aplicada?label=%C3%BAltimo%20commit)](https://github.com/alvaroparedesl/geomatica-aplicada/commits/main)
#
# En este notebook aprenderemos a realizar operaciones espaciales avanzadas utilizando GeoPandas.
#
# **Objetivos:**
# * Aprender las operaciones espaciales básicas (unión, intersección, diferencia)
# * Realizar análisis de proximidad (buffers)
# * Trabajar con joins espaciales
# * Aplicar operaciones de superposición (overlay)
# * Realizar consultas espaciales complejas

# %% [markdown]
# ## Configuración inicial
#
# La celda `Preparar entorno` deja disponible la carpeta `utils/` del curso, con funciones de apoyo para descargar datos y hacer gráficos.

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
# Importamos las bibliotecas necesarias
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

from utils.vector import descargar_natural_earth
from utils.viz import graficar_capas

# Configuración para mejorar la visualización
plt.rcParams["figure.figsize"] = (12, 8)

# %% [markdown]
# ## 1. Preparación de datos de ejemplo
#
# Crearemos algunos datos de ejemplo para ilustrar las operaciones espaciales.

# %%
# Creamos dos polígonos que se superponen
poligono1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
poligono2 = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])

# Creamos GeoDataFrames
gdf1 = gpd.GeoDataFrame({"geometry": [poligono1], "nombre": ["A"]})
gdf2 = gpd.GeoDataFrame({"geometry": [poligono2], "nombre": ["B"]})

# Visualizamos los polígonos
fig, ax = plt.subplots()
gdf1.plot(ax=ax, color="red", alpha=0.5, label="Polígono A")
gdf2.plot(ax=ax, color="blue", alpha=0.5, label="Polígono B")
plt.legend()
plt.title("Polígonos de ejemplo")
plt.axis("equal")
plt.grid(True)
plt.show()

# %% [markdown]
# ## 2. Operaciones espaciales básicas
#
# Para no repetir el código de cada figura, a partir de aquí usaremos la función `graficar_capas()` (de `utils/viz.py`): recibe una lista de capas con su color y las dibuja una encima de otra, igual que hicimos arriba con `plot(ax=ax)`.
#
# ### 2.1 Unión (Union)
# La unión combina dos geometrías incluyendo todas las áreas.

# %%
# Realizamos la unión
union = gpd.overlay(gdf1, gdf2, how="union")

# Visualizamos el resultado
graficar_capas([(union, "green")], "Unión de polígonos")

# %% [markdown]
# ### 2.2 Intersección (Intersection)
# La intersección muestra solo las áreas donde ambas geometrías se superponen.

# %%
# Realizamos la intersección
interseccion = gpd.overlay(gdf1, gdf2, how="intersection")

# Visualizamos el resultado
graficar_capas([(interseccion, "purple")], "Intersección de polígonos")

# %% [markdown]
# ### 2.3 Diferencia (Difference)
# La diferencia muestra las áreas que están en la primera geometría pero no en la segunda.

# %%
# Realizamos la diferencia
diferencia = gpd.overlay(gdf1, gdf2, how="difference")

# Visualizamos el resultado
graficar_capas([(diferencia, "orange")], "Diferencia de polígonos (A - B)")

# %% [markdown]
# ## 3. Análisis de proximidad
# Los buffers son útiles para análisis de proximidad y zonas de influencia.

# %%
# Creamos un punto central
punto = gpd.GeoDataFrame({"geometry": [Point(1, 1)], "nombre": ["Punto central"]})

# Creamos buffers de diferentes tamaños
buffer1 = punto.copy()
buffer1["geometry"] = punto.buffer(0.5)
buffer2 = punto.copy()
buffer2["geometry"] = punto.buffer(1)
buffer3 = punto.copy()
buffer3["geometry"] = punto.buffer(1.5)

# Visualizamos los buffers
fig, ax = plt.subplots()
buffer3.plot(ax=ax, color="lightblue", alpha=0.3)
buffer2.plot(ax=ax, color="blue", alpha=0.3)
buffer1.plot(ax=ax, color="darkblue", alpha=0.3)
punto.plot(ax=ax, color="red", markersize=50)
plt.title("Análisis de proximidad con buffers")
plt.axis("equal")
plt.grid(True)
plt.show()

# %% [markdown]
# ## 4. Joins espaciales
# Los joins espaciales nos permiten combinar atributos basados en relaciones espaciales.

# %%
# Creamos puntos aleatorios
np.random.seed(0)
n_puntos = 20
puntos = gpd.GeoDataFrame(
    {
        "geometry": [Point(x, y) for x, y in np.random.rand(n_puntos, 2) * 3],
        "valor": np.random.randint(1, 100, n_puntos),
    }
)

# Realizamos un join espacial para ver qué puntos están dentro de cada polígono
puntos_en_poligonos = gpd.sjoin(
    puntos,
    pd.concat([gdf1.assign(poligono="A"), gdf2.assign(poligono="B")]),
    how="left",
    predicate="within",
)

# Visualizamos el resultado
fig, ax = plt.subplots()
gdf1.plot(ax=ax, color="red", alpha=0.3)
gdf2.plot(ax=ax, color="blue", alpha=0.3)

# Coloreamos los puntos según el polígono al que pertenecen
mask_a = puntos_en_poligonos["poligono"] == "A"
mask_b = puntos_en_poligonos["poligono"] == "B"
mask_none = puntos_en_poligonos["poligono"].isna()

ax.scatter(
    puntos_en_poligonos[mask_a].geometry.x,
    puntos_en_poligonos[mask_a].geometry.y,
    c="red",
    label="En A",
)
ax.scatter(
    puntos_en_poligonos[mask_b].geometry.x,
    puntos_en_poligonos[mask_b].geometry.y,
    c="blue",
    label="En B",
)
ax.scatter(
    puntos_en_poligonos[mask_none].geometry.x,
    puntos_en_poligonos[mask_none].geometry.y,
    c="gray",
    label="Fuera",
)

plt.legend()
plt.title("Join espacial de puntos y polígonos")
plt.axis("equal")
plt.grid(True)
plt.show()

# %% [markdown]
# ## 5. Caso práctico: Análisis de áreas protegidas
#
# Vamos a trabajar con datos reales de Chile para mostrar un caso práctico.


# %%
# Cargamos los datos de países (Natural Earth, descargados con la función de utils/vector.py)
paises = descargar_natural_earth()

# Filtramos para obtener Chile
chile = paises[paises["NAME"] == "Chile"]

# Creamos algunas áreas protegidas ficticias
areas_protegidas = gpd.GeoDataFrame(
    {
        "geometry": [
            Polygon([(-70.6, -33.4), (-70.5, -33.4), (-70.5, -33.3), (-70.6, -33.3)]),
            Polygon([(-70.7, -33.5), (-70.6, -33.5), (-70.6, -33.4), (-70.7, -33.4)]),
        ],
        "nombre": ["Parque A", "Parque B"],
        "area_km2": [100, 150],
    },
    crs="EPSG:4326",
)

# Visualizamos Chile y las áreas protegidas
fig, ax = plt.subplots(figsize=(10, 15))
chile.plot(ax=ax, color="lightgray")
areas_protegidas.plot(ax=ax, color="green", alpha=0.5)
plt.title("Áreas protegidas en Chile (ejemplo)")
plt.axis("equal")
plt.show()

# %% [markdown]
# ### 5.1 Análisis de superposición
# Vamos a crear zonas de influencia alrededor de las áreas protegidas.

# %%
# Creamos un buffer de 10km alrededor de las áreas protegidas
areas_protegidas_buffer = areas_protegidas.copy()
areas_protegidas_buffer.geometry = (
    areas_protegidas.to_crs("+proj=utm +zone=19 +south")
    .buffer(10000)
    .to_crs(areas_protegidas.crs)
)

# Visualizamos el resultado
fig, ax = plt.subplots(figsize=(10, 15))
chile.plot(ax=ax, color="lightgray")
areas_protegidas.plot(ax=ax, color="green", alpha=0.5)
areas_protegidas_buffer.plot(ax=ax, color="yellow", alpha=0.3)
plt.title("Áreas protegidas y zonas de influencia")
plt.axis("equal")
plt.show()

# %% [markdown]
# ## 6. Ejercicio práctico
#
# Intenta realizar las siguientes tareas:
#
# 1. Crear tres polígonos que se superpongan y calcular su intersección común
# 2. Realizar un análisis de proximidad para encontrar todos los puntos dentro de un radio específico
# 3. Calcular el área total de la superposición entre dos geometrías
# 4. Crear un mapa de calor basado en la densidad de puntos
#
# **Pista:** Utiliza las operaciones que hemos aprendido y combínalas de forma creativa.

# %% [markdown]
# ## Referencias y recursos adicionales
#
# * [Documentación de operaciones espaciales en GeoPandas](https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.overlay.html)
# * [Tutorial de análisis espacial](https://geopandas.org/en/stable/docs/user_guide/geometric_manipulations.html)
# * [Ejemplos de joins espaciales](https://geopandas.org/en/stable/docs/user_guide/mergingdata.html)
