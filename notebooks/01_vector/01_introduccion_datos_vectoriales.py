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
import networkx as nx
import matplotlib.pyplot as plt

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

# %% [markdown]
# ## 7. Tipología Arco-Nodo y Análisis Topológico
#
# La tipología arco-nodo es una forma especial de organizar datos espaciales que nos permite:
# * Mantener las relaciones entre elementos geográficos
# * Realizar análisis de conectividad
# * Encontrar rutas óptimas
# * Validar la calidad de los datos
#
# En este ejemplo veremos:
# 1. La diferencia entre vértices y nodos
# 2. Cómo crear una red topológica simple
# 3. Análisis básicos que podemos hacer
# 4. Comparación con geometría simple

# %%
# 7.1 Creación de elementos básicos
# Primero, creamos algunos puntos que serán nodos (intersecciones importantes)
nodos = {
    1: Point(0, 0),    # Esquina inferior izquierda
    2: Point(2, 2),    # Centro
    3: Point(4, 0),    # Esquina inferior derecha
    4: Point(3, 3),    # Esquina superior derecha
    5: Point(1, 3)     # Esquina superior izquierda
}

# Creamos algunos vértices adicionales que NO son nodos (solo puntos de quiebre)
vertices = {
    'v1': Point(1, 1),    # Punto intermedio en línea curva
    'v2': Point(3, 1),    # Punto intermedio en línea curva
    'v3': Point(2, 3)     # Punto superior central
}

# Creamos arcos (líneas) entre los puntos
# Algunos arcos son directos entre nodos, otros pasan por vértices
arcos = {
    'a1': LineString([nodos[1].coords[0], vertices['v1'].coords[0], nodos[2].coords[0]]),  # Línea curva
    'a2': LineString([nodos[2].coords[0], vertices['v2'].coords[0], nodos[3].coords[0]]),  # Línea curva
    'a3': LineString([nodos[1].coords[0], nodos[3].coords[0]]),                            # Línea directa
    'a4': LineString([nodos[2].coords[0], nodos[4].coords[0]]),                            # Línea directa
    'a5': LineString([nodos[5].coords[0], vertices['v3'].coords[0], nodos[4].coords[0]])   # Línea con vértice
}

# Creamos dos polígonos usando los arcos
poligono1 = Polygon([nodos[1].coords[0], vertices['v1'].coords[0], nodos[2].coords[0], 
                    vertices['v2'].coords[0], nodos[3].coords[0], nodos[1].coords[0]])
poligono2 = Polygon([nodos[2].coords[0], nodos[4].coords[0], vertices['v3'].coords[0], 
                    nodos[5].coords[0], nodos[2].coords[0]])

# %% [markdown]
# ### 7.2 Visualización de Elementos
# Veamos la diferencia entre nodos (puntos de intersección) y vértices (puntos de quiebre):

# %%
# Creamos una figura más grande y clara
plt.figure(figsize=(12, 8))

# Dibujamos los arcos
for arco in arcos.values():
    xs, ys = zip(*list(arco.coords))
    plt.plot(xs, ys, 'gray', linewidth=2)

# Dibujamos los polígonos con transparencia
plt.fill(*poligono1.exterior.xy, alpha=0.2, color='blue', label='Polígono 1')
plt.fill(*poligono2.exterior.xy, alpha=0.2, color='green', label='Polígono 2')

# Dibujamos los nodos (más grandes y azules)
for nodo in nodos.values():
    plt.plot(nodo.x, nodo.y, 'o', color='blue', markersize=12, label='Nodo' if nodo == list(nodos.values())[0] else "")

# Dibujamos los vértices (más pequeños y rojos)
for vertice in vertices.values():
    plt.plot(vertice.x, vertice.y, 'o', color='red', markersize=8, label='Vértice' if vertice == list(vertices.values())[0] else "")

# Agregamos etiquetas para los nodos
for id_nodo, nodo in nodos.items():
    plt.annotate(f'N{id_nodo}', (nodo.x, nodo.y), xytext=(5, 5), textcoords='offset points')

# Agregamos etiquetas para los vértices
for id_vertice, vertice in vertices.items():
    plt.annotate(id_vertice, (vertice.x, vertice.y), xytext=(5, 5), textcoords='offset points')

plt.title("Red Topológica: Nodos, Vértices y Polígonos")
plt.legend()
plt.grid(True)
plt.axis('equal')
display(plt.gcf())
plt.close()

# %% [markdown]
# ### 7.3 Creación de la Red Topológica
# 
# Ahora crearemos una red que solo considera los nodos y sus conexiones:

# %%
# Creamos el grafo topológico
G = nx.Graph()

# Agregamos solo los nodos (no los vértices)
for id_nodo, punto in nodos.items():
    G.add_node(id_nodo, pos=punto.coords[0])

# Agregamos los arcos con sus atributos
for nombre_arco, linea in arcos.items():
    # Encontramos qué nodos conecta este arco
    nodo_inicio = None
    nodo_fin = None
    for id_nodo, punto in nodos.items():
        if punto.coords[0] == linea.coords[0]:  # Primer punto del arco
            nodo_inicio = id_nodo
        if punto.coords[0] == linea.coords[-1]:  # Último punto del arco
            nodo_fin = id_nodo
    if nodo_inicio and nodo_fin:
        # Agregamos el arco con su longitud y nombre
        G.add_edge(nodo_inicio, nodo_fin, 
                  weight=linea.length,  # Longitud del arco
                  name=nombre_arco)     # Nombre del arco

# %% [markdown]
# ### 7.4 Análisis Topológico Simple
#
# Veamos algunas preguntas que podemos responder fácilmente con la red topológica:

# %%
# 1. ¿Cuántas conexiones tiene cada nodo?
print("Conexiones por nodo:")
for nodo in G.nodes():
    num_conexiones = len(list(G.neighbors(nodo)))
    print(f"Nodo {nodo} tiene {num_conexiones} conexiones")

# 2. ¿Cuál es el camino más corto entre dos puntos?
inicio, fin = 1, 4  # Del nodo 1 al nodo 4
camino = nx.shortest_path(G, inicio, fin, weight='weight')
print(f"\nCamino más corto del nodo {inicio} al {fin}: {' → '.join(map(str, camino))}")

# 3. ¿Hay ciclos en la red?
ciclos = list(nx.cycle_basis(G))
print(f"\nNúmero de ciclos encontrados: {len(ciclos)}")
if ciclos:
    print("Ciclos encontrados:")
    for i, ciclo in enumerate(ciclos, 1):
        print(f"Ciclo {i}: {' → '.join(map(str, ciclo + [ciclo[0]]))}")

# %% [markdown]
# ### 7.5 Comparación: Topología vs Geometría
#
# Veamos la diferencia en complejidad para encontrar nodos conectados:

# %%
print("1. Usando solo geometría (Shapely):")
print("Para encontrar conexiones, debemos revisar cada línea y sus puntos extremos...")
for nombre1, punto1 in nodos.items():
    conectados = []
    for nombre2, punto2 in nodos.items():
        if nombre1 != nombre2:
            # Debemos revisar cada arco
            for arco in arcos.values():
                if (punto1.coords[0] == arco.coords[0] and punto2.coords[0] == arco.coords[-1]) or \
                   (punto1.coords[0] == arco.coords[-1] and punto2.coords[0] == arco.coords[0]):
                    conectados.append(nombre2)
    print(f"Nodo {nombre1} está conectado con: {conectados}")

print("\n2. Usando topología (NetworkX):")
print("Simplemente preguntamos por los vecinos de cada nodo...")
for nodo in G.nodes():
    print(f"Nodo {nodo} está conectado con: {list(G.neighbors(nodo))}")
