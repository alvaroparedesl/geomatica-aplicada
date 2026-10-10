# Datos raster y teledetección

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/notebooks/00_intro.ipynb) [![Licencia: GPL-3.0](https://img.shields.io/badge/licencia-GPL--3.0-blue.svg)](../../LICENSE) [![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://colab.research.google.com/) [![Último commit](https://img.shields.io/github/last-commit/alvaroparedesl/geomatica-aplicada?label=%C3%BAltimo%20commit)](https://github.com/alvaroparedesl/geomatica-aplicada/commits/main)

Esta sección cubre los datos raster y la teledetección: desde los conceptos básicos hasta el acceso a imágenes Landsat y Sentinel-2 desde Planetary Computer, el cálculo de índices de vegetación y el análisis multitemporal.

## Cuadernos

| Cuaderno | Descripción | Colab |
|---|---|---|
| [`00_demo_imagenes.py`](00_demo_imagenes.py) | Demo del catálogo STAC: buscar imágenes Sentinel-2 y Landsat, visualizarlas, comparar números digitales con reflectancia, calcular índices y extraer series temporales. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/notebooks/02_raster/00_demo_imagenes.ipynb) |
| [`01_datos_raster.py`](01_datos_raster.py) | Conceptos básicos de datos raster con xarray y rioxarray, usando cobertura de suelo (ESA WorldCover) y un modelo de elevación (Copernicus DEM) descargados desde Planetary Computer. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/notebooks/02_raster/01_datos_raster.ipynb) |
| [`02_intro_teledeteccion.py`](02_intro_teledeteccion.py) | Fundamentos de teledetección: espectro electromagnético, sensores, plataformas, resoluciones, aplicaciones y principales misiones satelitales. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/notebooks/02_raster/02_intro_teledeteccion.ipynb) |
| [`03_acceso_imagenes.py`](03_acceso_imagenes.py) | Acceso a imágenes Sentinel-2 y Landsat con pystac-client y Planetary Computer: búsqueda, carga, visualización y comparación entre sensores. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/notebooks/02_raster/03_acceso_imagenes.ipynb) |
| [`04_estadisticas_imagenes.py`](04_estadisticas_imagenes.py) | Histogramas y estadísticas por banda y por escena, máscaras con operadores lógicos y combinación con datos de elevación. Cuaderno práctico con ejercicios. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/notebooks/02_raster/04_estadisticas_imagenes.ipynb) |
| [`05_ejercicio_histograma_landsat.py`](05_ejercicio_histograma_landsat.py) | Ejercicio interactivo: descargar una escena Landsat, ver sus histogramas, seleccionar una región en un mapa y exportar la escena como GeoTIFF. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/notebooks/02_raster/05_ejercicio_histograma_landsat.ipynb) |
| [`06_indices_vegetacion.py`](06_indices_vegetacion.py) | Álgebra de imágenes e índices de vegetación (NDVI, EVI, NDWI) con configuración modular de sensor, área y período. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/notebooks/02_raster/06_indices_vegetacion.ipynb) |
| [`07_analisis_multitemporal.py`](07_analisis_multitemporal.py) | Proyecto final: análisis multitemporal de NDVI y EVI con series de Landsat, máscara de nubes, estadísticas y exportación a Excel. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/notebooks/02_raster/07_analisis_multitemporal.ipynb) |

## Cómo usarlos

1. Abre el cuaderno con su botón **Abrir en Colab**.
2. Ejecuta primero la celda `Preparar entorno` (clona el repositorio para poder usar [`utils/`](../../utils/README.md)) y luego la celda de instalación de paquetes.
3. Ejecuta el resto de las celdas en orden. Se recomienda seguir los cuadernos en orden numérico.

## Datos

Todos los datos se acceden en línea desde el catálogo STAC de Microsoft Planetary Computer (Landsat, Sentinel-2, ESA WorldCover y Copernicus DEM), por lo que necesitas conexión a internet y no hay archivos de datos en el repositorio. Los cuadernos `02_intro_teledeteccion` y parte de `01_datos_raster` usan datos generados en el propio cuaderno.

## Paquetes principales

Rioxarray, xarray, rasterio, xarray-spatial, pystac-client, planetary-computer, odc-stac, Matplotlib, Plotly e ipyleaflet. La lista completa está en [`requirements.txt`](../../requirements.txt).

Para regenerar los `.ipynb` a partir de los `.py`, consulta el [flujo de trabajo](../README.md#flujo-de-trabajo-py-es-la-fuente-ipynb-se-genera).
