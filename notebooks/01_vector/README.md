# Datos vectoriales

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/notebooks/00_intro.ipynb) [![Licencia: GPL-3.0](https://img.shields.io/badge/licencia-GPL--3.0-blue.svg)](../../LICENSE) [![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://colab.research.google.com/) [![Último commit](https://img.shields.io/github/last-commit/alvaroparedesl/geomatica-aplicada?label=%C3%BAltimo%20commit)](https://github.com/alvaroparedesl/geomatica-aplicada/commits/main)

Esta sección introduce los datos vectoriales (puntos, líneas y polígonos): primero con Shapely, que modela las geometrías, y luego con GeoPandas, que añade atributos, sistemas de referencia y operaciones espaciales.

## Cuadernos

| Cuaderno | Descripción | Colab |
|---|---|---|
| [`01_introduccion_datos_vectoriales.py`](01_introduccion_datos_vectoriales.py) | Geometrías vectoriales con Shapely: puntos, líneas y polígonos, sus atributos, geometrías múltiples y polígonos con huecos. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/notebooks/01_vector/01_introduccion_datos_vectoriales.ipynb) |
| [`02_introduccion_geopandas.py`](02_introduccion_geopandas.py) | Primer contacto con GeoPandas: cargar, visualizar y filtrar datos espaciales, y realizar operaciones espaciales simples. Incluye un ejercicio con solución. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/notebooks/01_vector/02_introduccion_geopandas.ipynb) |
| [`03_operaciones_espaciales.py`](03_operaciones_espaciales.py) | Operaciones espaciales con GeoPandas: unión, intersección y diferencia, buffers, joins espaciales y overlay, con un caso práctico de áreas protegidas. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/notebooks/01_vector/03_operaciones_espaciales.ipynb) |

## Cómo usarlos

1. Abre el cuaderno con su botón **Abrir en Colab**.
2. Ejecuta primero la celda `Preparar entorno` (clona el repositorio para poder usar [`utils/`](../../utils/README.md)).
3. Ejecuta el resto de las celdas en orden. Se recomienda seguir los cuadernos en orden numérico.

## Datos

Los datos vectoriales no están en el repositorio: se crean en el propio cuaderno o se descargan de fuentes públicas, por ejemplo con `utils.vector.descargar_natural_earth`.

## Paquetes principales

Shapely, GeoPandas, Matplotlib, Folium, mapclassify y NetworkX. La lista completa está en [`requirements.txt`](../../requirements.txt).

Para regenerar los `.ipynb` a partir de los `.py`, consulta el [flujo de trabajo](../README.md#flujo-de-trabajo-py-es-la-fuente-ipynb-se-genera).
