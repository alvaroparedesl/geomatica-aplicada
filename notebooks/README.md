# Cuadernos de Geomática Aplicada

Este directorio contiene todos los cuadernos Jupyter (en formato jupytext) para el curso de Geomática Aplicada en Recursos Naturales.

## Estructura

- **01_vector**: Cuadernos relacionados con sistemas de referencia geoespacial, SIG y análisis de datos vectoriales
- **02_raster**: Cuadernos relacionados con teledetección, procesamiento de imágenes y análisis de datos raster

## Contenido del Curso

### 01_vector - Análisis de Datos Vectoriales y SIG
1. Sistemas de referencia geoespacial (3 sesiones)
2. Definición de SIG y Web Mapping
3. Datos geográficos y mapas temáticos
4. Datos vectoriales (2 sesiones)
5. Análisis espacial (2 sesiones)
6. Aplicaciones en recursos naturales

### 02_raster - Análisis de Datos Raster y Teledetección
1. Definición de datos raster
2. Introducción a la teledetección
3. Principios físicos de teledetección (3 sesiones)
4. Estadísticas de imágenes (2 sesiones)
5. Visualización de imágenes (2 sesiones)
6. Índices de vegetación y productos derivados
7. NDVI, EVI y álgebra de imágenes
8. Otros índices espectrales
9. Conexión a catálogos STAC

## Cómo utilizar estos cuadernos

Los cuadernos están almacenados en formato `.py` utilizando jupytext para facilitar el control de versiones. Para trabajar con ellos:

1. Asegúrate de tener jupytext instalado:
   ```
   pip install jupytext
   ```

2. Puedes abrir los archivos .py directamente en JupyterLab o Jupyter Notebook si tienes la extensión de jupytext instalada.

3. Alternativamente, puedes convertir los archivos .py a notebooks .ipynb:
   ```
   jupytext --to notebook archivo.py
   ```

4. Después de trabajar con el notebook, puedes sincronizar los cambios:
   ```
   jupytext --sync archivo.ipynb
   ```

## Recomendaciones

- Sigue los ejercicios en orden numérico
- Ejecuta todas las celdas de cada cuaderno secuencialmente
- Intenta resolver los ejercicios propuestos antes de revisar las soluciones
- Experimenta modificando los ejemplos para profundizar tu comprensión

## Requisitos

Para ejecutar estos cuadernos, necesitarás tener instaladas las bibliotecas mencionadas en la sección de tecnologías del README principal del repositorio, que incluyen:

### Para análisis vectorial:
- GeoPandas
- Folium
- Matplotlib
- Shapely

### Para análisis raster:
- Rioxarray
- Xarray
- Rasterio
- pySTAC
- Planetary Computer
- ODC (Open Data Cube) 