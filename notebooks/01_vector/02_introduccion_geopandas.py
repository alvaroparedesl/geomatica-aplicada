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
# # Introducción a GeoPandas
#
# [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/notebooks/01_vector/02_introduccion_geopandas.ipynb) [![Licencia: GPL-3.0](https://img.shields.io/badge/licencia-GPL--3.0-blue.svg)](https://github.com/alvaroparedesl/geomatica-aplicada/blob/main/LICENSE) [![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://colab.research.google.com/) [![Último commit](https://img.shields.io/github/last-commit/alvaroparedesl/geomatica-aplicada?label=%C3%BAltimo%20commit)](https://github.com/alvaroparedesl/geomatica-aplicada/commits/main)
#
# ## ¿Qué aprenderemos?
#
# En este notebook, vamos a introducirnos en el mundo del análisis de datos espaciales vectoriales con Python mediante la biblioteca GeoPandas.
#
# **Objetivos:**
# * Comprender qué son los datos vectoriales
# * Aprender a cargar y visualizar datos espaciales
# * Explorar métodos básicos de GeoPandas
# * Realizar operaciones espaciales simples
#
# > **Nota:** Este notebook está diseñado para que estudiantes con conocimientos básicos de Python puedan seguirlo sin dificultad. Si tienes dudas, ¡no dudes en preguntar!

# %% [markdown]
# ## Configuración inicial
#
# Primero preparamos el entorno: la celda `Preparar entorno` deja disponible la carpeta `utils/` del curso, donde guardamos funciones de apoyo (por ejemplo, para descargar datos).

# %%
# @title Preparar entorno (ejecutar primero)
import subprocess
import sys
from pathlib import Path

if "google.colab" in sys.modules:  # Colab: clonar el repositorio (solo la primera vez)
    RAIZ = Path("/content/geomatica-aplicada")
    if not RAIZ.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/alvaroparedesl/geomatica-aplicada", str(RAIZ)], check=True)
else:  # Local: subir desde la carpeta actual hasta encontrar utils/
    RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "utils").is_dir())
sys.path.insert(0, str(RAIZ))

# %%
# @title Instalación de paquetes necesarios
# mapclassify permite clasificar valores al colorear mapas (parámetro scheme)
# %pip install -q mapclassify

# %% [markdown]
# Ahora importamos las bibliotecas. GeoPandas extiende las funcionalidades de Pandas para trabajar con datos geoespaciales.

# %%
# Importamos las bibliotecas necesarias
import geopandas as gpd
import matplotlib.pyplot as plt

from utils.vector import descargar_natural_earth

# Configuración para mejorar la visualización
plt.rcParams["figure.figsize"] = (12, 8)

# %% [markdown]
# ## Cargando los datos
#
# Usaremos el mapa de países de [Natural Earth](https://www.naturalearthdata.com/) (escala 1:110 millones). La función `descargar_natural_earth()` (de `utils/vector.py`) descarga el archivo y lo entrega como un `GeoDataFrame`: una tabla de Pandas con una columna especial de geometrías.

# %%
# Descargamos y cargamos los datos
paises = descargar_natural_earth()

# Renombramos algunas columnas a nombres más cortos
paises = paises.rename(
    columns={"NAME": "name", "CONTINENT": "continent", "POP_EST": "pop_est"}
)

# Veamos las primeras filas
print("Primeras filas del GeoDataFrame:")
paises.head()

# %%
# Exploremos la estructura de nuestros datos
print("Información del GeoDataFrame:")
paises.info()

print("\nColumnas disponibles:")
print(paises.columns.tolist())

print("\nSistema de coordenadas:")
print(paises.crs)

# %% [markdown]
# ## Visualización básica
#
# GeoPandas facilita la visualización de datos espaciales a través de su método `.plot()`, que internamente usa Matplotlib.

# %%
# Visualizamos todos los países
ax = paises.plot(figsize=(12, 10))
ax.set_title("Países del mundo", fontsize=15)
plt.axis("off")  # Quitamos los ejes para una visualización más limpia
plt.show()

# %%
# Visualizamos los países por continente
ax = paises.plot(
    column="continent",
    legend=True,
    figsize=(12, 10),
    legend_kwds={"title": "Continentes"},
    cmap="tab10",
)
ax.set_title("Países por continente", fontsize=15)
plt.axis("off")
plt.show()

# %% [markdown]
# ## Filtrando datos espaciales
#
# Al igual que con Pandas, podemos filtrar nuestro GeoDataFrame para centrarnos en áreas específicas.

# %%
# Filtramos para mostrar solo los países de América del Sur
sudamerica = paises[paises["continent"] == "South America"]

# Visualizamos los países de América del Sur
ax = sudamerica.plot(
    figsize=(10, 10),
    column="pop_est",
    legend=True,
    cmap="YlOrRd",
    legend_kwds={"label": "Población"},
)
ax.set_title("Países de América del Sur", fontsize=15)
plt.axis("off")
plt.show()

# %% [markdown]
# ## Operaciones espaciales básicas
#
# GeoPandas proporciona múltiples operaciones espaciales, veamos algunas básicas.

# %%
# Calculamos el área de cada país (en kilómetros cuadrados)
# La función area devuelve el área en las unidades del CRS
# Como estamos usando un CRS geográfico (EPSG:4326), convertimos a km²
paises["area_km2"] = paises.geometry.to_crs("+proj=cea").area / 10**6

# Visualizamos los 10 países más grandes
paises_por_area = paises.sort_values("area_km2", ascending=False)
print("Los 10 países más grandes por área:")
paises_por_area[["name", "area_km2"]].head(10)

# %%
# Calculamos los centroides de cada país
paises["centroide"] = paises.geometry.centroid

# Creamos un nuevo GeoDataFrame con los centroides
centroides = gpd.GeoDataFrame(
    paises[["name", "centroide"]], geometry="centroide", crs=paises.crs
)

# Visualizamos países y sus centroides
ax = paises.plot(figsize=(12, 10), color="lightgray", edgecolor="gray")
centroides.plot(ax=ax, color="red", markersize=5)
ax.set_title("Países y sus centroides", fontsize=15)
plt.axis("off")
plt.show()

# %% [markdown]
# ## Ejercicio práctico
#
# Ahora es tu turno de practicar. Intenta realizar las siguientes tareas:
#
# 1. Filtrar el GeoDataFrame para mostrar solo los países de América Central
# 2. Calcular la población total por continente
# 3. Encontrar qué país tiene la mayor densidad de población (población/área)
# 4. Visualizar los países por densidad de población
#
# **Pista:** Para el ejercicio 2, puedes usar `groupby` de pandas.

# %%
# Espacio para tu solución del ejercicio

# Ejercicio 1: Filtrar países de América Central
# Tu código aquí

# Ejercicio 2: Población total por continente
# Tu código aquí

# Ejercicio 3: País con mayor densidad de población
# Tu código aquí

# Ejercicio 4: Visualizar por densidad de población
# Tu código aquí

# %% [markdown]
# ## Solución del ejercicio

# %%
# Solución 1: Filtrar países de América Central
america_central = paises[paises["continent"] == "North America"]
america_central = america_central[
    america_central["name"].isin(
        [
            "Belize",
            "Costa Rica",
            "El Salvador",
            "Guatemala",
            "Honduras",
            "Nicaragua",
            "Panama",
        ]
    )
]

ax = america_central.plot(figsize=(10, 8))
ax.set_title("Países de América Central", fontsize=15)
plt.axis("off")
plt.show()

# %%
# Solución 2: Población total por continente
poblacion_por_continente = paises.groupby("continent")["pop_est"].sum().reset_index()
print("Población total por continente:")
poblacion_por_continente

# %%
# Solución 3: País con mayor densidad de población
paises["densidad_pob"] = paises["pop_est"] / paises["area_km2"]
pais_mayor_densidad = paises.sort_values("densidad_pob", ascending=False).iloc[0]
print(
    f"El país con mayor densidad de población es {pais_mayor_densidad['name']} "
    f"con {pais_mayor_densidad['densidad_pob']:.2f} personas/km²"
)

# %%
# Solución 4: Visualizar por densidad de población
ax = paises.plot(
    column="densidad_pob",
    figsize=(12, 10),
    legend=True,
    cmap="YlOrRd",
    legend_kwds={"title": "Densidad\n(personas/km²)"},
    scheme="quantiles",
)
ax.set_title("Densidad de población por país", fontsize=15)
plt.axis("off")
plt.show()

# %% [markdown]
# ## Resumen
#
# En este notebook, hemos aprendido:
#
# * Qué son los datos vectoriales y cómo GeoPandas nos permite trabajar con ellos
# * Cargar y explorar datos espaciales con GeoPandas
# * Visualizar datos espaciales usando diferentes atributos
# * Realizar operaciones espaciales básicas como calcular áreas y centroides
# * Filtrar y analizar datos espaciales según sus propiedades
#
# En el próximo notebook, profundizaremos en operaciones espaciales más avanzadas como intersecciones, uniones y análisis de proximidad.
#
# ## Referencias y recursos adicionales
#
# * [Documentación oficial de GeoPandas](https://geopandas.org/)
# * [Tutorial de GeoPandas](https://geopandas.org/en/stable/getting_started/introduction.html)
# * [Datos geoespaciales en Python](https://geographicdata.science/book/intro.html)
