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
# # Introducción a Datos Vectoriales
#
# En este notebook aprenderemos los conceptos básicos de geometrías vectoriales usando la biblioteca Shapely.
#
# **Objetivos:**
# * Crear geometrías básicas (puntos, líneas y polígonos)
# * Visualizar geometrías
# * Acceder a atributos geométricos
# * Trabajar con geometrías complejas (multipuntos, multilíneas, multipolígonos)
# * Crear polígonos con huecos
#
# ## Configuración inicial

# %%
# Importamos las bibliotecas necesarias
from IPython.display import display
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

# %% [markdown]
# ## 1. Geometrías Básicas
#
# Shapely nos permite crear tres tipos básicos de geometrías:
# * Puntos (Point)
# * Líneas (LineString)
# * Polígonos (Polygon)

# %%
# Creamos un punto
punto = Point(0, 0)
print("Punto:", punto)
print("Coordenadas:", list(punto.coords))

# Creamos una línea
linea = LineString([(0, 0), (1, 1), (2, 0)])
print("\nLínea:", linea)
print("Coordenadas:", list(linea.coords))

# Creamos un polígono
poligono = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
print("\nPolígono:", poligono)
print("Coordenadas:", list(poligono.exterior.coords))

# %% [markdown]
# ## 2. Visualización de Geometrías

# %%
# Visualizamos cada geometría
display(punto)
display(linea)
display(poligono)

# %% [markdown]
# ## 3. Atributos Geométricos
#
# Cada tipo de geometría tiene sus propios atributos:

# %%
# @title Atributos del punto
print("Punto:")
print(f"Coordenadas: {list(punto.coords)}")
print(f"X: {punto.x}")
print(f"Y: {punto.y}")

# %%
# @title Atributos de la línea
print("\nLínea:")
print(f"Longitud: {linea.length}")
print(f"Puntos extremos: {linea.bounds}")
print(f"Centroide: {linea.centroid}")

# %%
# @title Atributos del polígono
print("\nPolígono:")
print(f"Área: {poligono.area}")
print(f"Perímetro: {poligono.length}")
print(f"Centroide: {poligono.centroid}")
print(f"Límites: {poligono.bounds}")

# %% [markdown]
# ## 4. Ejemplo Práctico: Área de un Polígono
#
# Vamos a crear un polígono con las coordenadas proporcionadas y calcular su área.

# %%
# Coordenadas del polígono
coordenadas = [(3, 5), (5, 3), (6, 9), (5, 7), (3, 6), (3, 5)]

# Creamos el polígono
poligono_ejemplo = Polygon(coordenadas)

# Calculamos el área
area = poligono_ejemplo.area
print(f"Área del polígono: {area:.2f} unidades cuadradas")

# Visualizamos el polígono
display(poligono_ejemplo)

# %% [markdown]
# ## 5. Geometrías Complejas
#
# Shapely también nos permite crear geometrías complejas que contienen múltiples elementos.

# %%
# Creamos un multipunto
multipunto = MultiPoint([(0, 0), (1, 1), (2, 0)])
print("Multipunto:", multipunto)

# Creamos una multilínea
multilinea = MultiLineString([[(0, 0), (1, 1)], [(2, 0), (3, 1)]])
print("\nMultilínea:", multilinea)

# Creamos un multipolígono
multipoligono = MultiPolygon(
    [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
        Polygon([(2, 0), (3, 0), (3, 1), (2, 1), (2, 0)]),
    ]
)
print("\nMultipolígono:", multipoligono)

# %%
# Visualizamos las geometrías complejas
display(multipunto)
display(multilinea)
display(multipoligono)

# %% [markdown]
# ## 6. Polígonos con Huecos
#
# Un polígono puede tener huecos (islas) en su interior.

# %%
# Creamos un polígono con un hueco
poligono_exterior = [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]
hueco = [(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)]
poligono_con_hueco = Polygon(poligono_exterior, [hueco])

# Calculamos el área (el hueco se resta automáticamente)
area_total_pre = Polygon(poligono_exterior).area
area_total = poligono_con_hueco.area
print(f"Área del polígono sin hueco: {area_total_pre:.2f} unidades cuadradas")
print(f"Área del polígono con hueco: {area_total:.2f} unidades cuadradas")

# Visualizamos el polígono con hueco
display(poligono_con_hueco)
