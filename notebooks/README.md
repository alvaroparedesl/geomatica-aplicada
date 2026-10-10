# Cuadernos de Geomática Aplicada

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/notebooks/00_intro.ipynb) [![Licencia: GPL-3.0](https://img.shields.io/badge/licencia-GPL--3.0-blue.svg)](../LICENSE) [![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://colab.research.google.com/) [![Último commit](https://img.shields.io/github/last-commit/alvaroparedesl/geomatica-aplicada?label=%C3%BAltimo%20commit)](https://github.com/alvaroparedesl/geomatica-aplicada/commits/main)

Este directorio contiene los cuadernos del curso de Geomática Aplicada en Recursos Naturales, escritos como archivos Python en formato jupytext (`py:percent`) y publicados también como notebooks `.ipynb`.

## Estructura

* [`00_intro.py`](00_intro.py): introducción a Jupyter Notebooks y Google Colab, la celda `Preparar entorno` y la estructura del curso. [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/notebooks/00_intro.ipynb)
* [`01_vector/`](01_vector/README.md): datos vectoriales con Shapely y GeoPandas
* [`02_raster/`](02_raster/README.md): datos raster, teledetección e índices de vegetación

## Contenido del curso

### 00_intro: herramientas de trabajo

* Qué son los Jupyter Notebooks y cómo funciona Google Colab
* La celda `Preparar entorno` y la carpeta `utils/`
* Formato de los cuadernos y estructura del curso

### 01_vector: datos vectoriales

1. `01_introduccion_datos_vectoriales`: geometrías con Shapely (puntos, líneas, polígonos, geometrías múltiples y huecos)
2. `02_introduccion_geopandas`: cargar, visualizar y filtrar datos espaciales con GeoPandas
3. `03_operaciones_espaciales`: unión, intersección, buffers, joins espaciales y overlay

### 02_raster: datos raster y teledetección

0. `00_demo_imagenes`: demo de exploración de imágenes satelitales con STAC
1. `01_datos_raster`: conceptos básicos de raster, con cobertura de suelo y DEM desde Planetary Computer
2. `02_intro_teledeteccion`: fundamentos de teledetección
3. `03_acceso_imagenes`: acceso a imágenes Sentinel-2 y Landsat con pystac-client y Planetary Computer
4. `04_estadisticas_imagenes`: histogramas, estadísticas y máscaras
5. `05_ejercicio_histograma_landsat`: ejercicio interactivo con histogramas de una escena Landsat
6. `06_indices_vegetacion`: álgebra de imágenes e índices de vegetación (NDVI, EVI, NDWI)
7. `07_analisis_multitemporal`: proyecto final de análisis multitemporal

Las tablas con la descripción y el botón de Colab de cada cuaderno están en los README de [`01_vector`](01_vector/README.md) y [`02_raster`](02_raster/README.md).

## Entorno de trabajo: Google Colab

Los cuadernos están diseñados para ejecutarse en **Google Colab**, sin instalar nada en tu computador. Cada cuaderno incluye:

* La celda `Preparar entorno`, que clona el repositorio en Colab para poder usar la carpeta [`utils/`](../utils/README.md)
* Una celda `%pip install ...` con los paquetes que Colab no trae instalados
* Código para acceder a los datos en línea (por ejemplo, Planetary Computer)

Para ejecutarlos en tu computador, instala las dependencias con `pip install -r requirements.txt` (archivo en la raíz del repositorio).

## Flujo de trabajo: `.py` es la fuente, `.ipynb` se genera

Los cuadernos **no están emparejados** con jupytext. El archivo `.py` es la fuente que se edita y se versiona, y el `.ipynb` es una versión generada a partir de él (en una sola dirección). Para actualizarlo:

```bash
pip install jupytext

# Un cuaderno
jupytext --to notebook notebooks/02_raster/03_acceso_imagenes.py

# Todos los cuadernos
jupytext --to notebook notebooks/*.py notebooks/*/*.py
```

También puedes usar la herramienta [`sync_to_gdrive.py`](../sync_to_gdrive.py), que hace la misma conversión y sube los resultados a Google Drive (ver [SYNC_SETUP.md](../SYNC_SETUP.md)).

Si editas un `.ipynb` directamente, esos cambios no vuelven al `.py` y se perderán la próxima vez que se regenere el notebook. Haz las modificaciones en el `.py`.

## Acceso a datos

* **Datos vectoriales**: se descargan de fuentes públicas (por ejemplo Natural Earth) o se crean en el propio cuaderno.
* **Datos raster e imágenes satelitales**: se acceden en línea mediante **Microsoft Planetary Computer**, sin descargar archivos. Se introduce desde el inicio de la sección raster: el cuaderno `00_demo_imagenes` muestra la idea general, `01_datos_raster` descarga cobertura de suelo (ESA WorldCover) y elevación (Copernicus DEM), y `03_acceso_imagenes` profundiza en la búsqueda de imágenes Landsat y Sentinel-2.

## Recomendaciones

* Comienza con `00_intro` para familiarizarte con Jupyter y Colab
* Sigue los cuadernos en orden numérico
* Ejecuta las celdas de cada cuaderno en secuencia, comenzando por `Preparar entorno`
* Intenta resolver los ejercicios antes de revisar las soluciones
* Experimenta modificando los ejemplos para profundizar tu comprensión

## Requisitos

Para ejecutar los cuadernos en Google Colab solo necesitas una cuenta de Google, conexión a internet y un navegador web moderno. La lista completa de dependencias está en [`requirements.txt`](../requirements.txt).
