# Cuadernos de Geomática Aplicada

Este directorio contiene todos los cuadernos Jupyter (en formato jupytext) para el curso de Geomática Aplicada en Recursos Naturales.

## Estructura

- **00_intro.py**: Introducción a Jupyter Notebooks y Google Colab
- **01_vector**: Cuadernos relacionados con sistemas de referencia geoespacial, SIG y análisis de datos vectoriales
- **02_raster**: Cuadernos relacionados con teledetección, procesamiento de imágenes y análisis de datos raster

## Contenido del Curso

### 00_intro - Introducción a las herramientas
- Introducción a Jupyter Notebooks
- Uso de Google Colab
- Estructura del curso y recomendaciones

### 01_vector - Análisis de Datos Vectoriales y SIG
1. Sistemas de referencia geoespacial
2. Definición de SIG y Web Mapping
3. Datos geográficos y mapas temáticos
4. Ejercicios con datos geográficos
5. Datos vectoriales
6. Análisis espacial
7. Aplicaciones en recursos naturales

### 02_raster - Análisis de Datos Raster y Teledetección
1. Definición de datos raster
2. Introducción a la teledetección
3. Principios físicos de teledetección
4. Acceso a imágenes satelitales con Planetary Computer
5. Estadísticas de imágenes
6. Visualización y tratamiento de imágenes
7. Índices de vegetación
8. Análisis multitemporal

## Enfoque de consolidación

Los cuadernos han sido consolidados por temas para evitar la fragmentación y proporcionar una experiencia de aprendizaje más coherente. Si algún cuaderno resulta demasiado extenso, podrá dividirse en el futuro según sea necesario.

## Entorno de trabajo: Google Colab

Todos los cuadernos están diseñados para ejecutarse en **Google Colab**. Esto facilita el trabajo sin necesidad de instalar software adicional en el equipo del estudiante. Cada cuaderno incluye:

- Instrucciones de configuración inicial
- Celdas para instalar las bibliotecas necesarias
- Código para acceder a datos de ejemplo o conectarse a Planetary Computer
- Opciones para guardar resultados en Google Drive

## Cómo utilizar estos cuadernos

Los cuadernos están almacenados en formato `.py` utilizando jupytext para facilitar el control de versiones. Para trabajar con ellos:

1. Abre el archivo `.py` directamente en Google Colab usando la opción "Archivo > Subir cuaderno" y seleccionando el archivo.

2. Alternativamente, si prefieres trabajar localmente, asegúrate de tener jupytext instalado:
   ```
   pip install jupytext
   ```

3. Puedes convertir los archivos .py a notebooks .ipynb:
   ```
   jupytext --to notebook archivo.py
   ```

4. Después de trabajar con el notebook, puedes sincronizar los cambios:
   ```
   jupytext --sync archivo.ipynb
   ```

## Acceso a datos

- **Datos vectoriales**: Se descargarán directamente de fuentes públicas cuando sea posible. Para los primeros cuadernos, se proporcionan algunos datos de ejemplo.

- **Imágenes satelitales**: Se accederá a través de **Planetary Computer**, lo que permite trabajar con imágenes Landsat y Sentinel-2 sin necesidad de descargarlas. Esta funcionalidad se introduce temprano en el curso (cuaderno 04 de la sección raster) para permitir trabajar con imágenes reales desde el principio.

## Recomendaciones

- Comienza con el cuaderno de introducción (00_intro.py) para familiarizarte con Jupyter Notebooks y Google Colab
- Sigue los ejercicios en orden numérico
- Ejecuta todas las celdas de cada cuaderno secuencialmente
- Intenta resolver los ejercicios propuestos antes de revisar las soluciones
- Experimenta modificando los ejemplos para profundizar tu comprensión

## Requisitos

Para ejecutar estos cuadernos en Google Colab, solo necesitarás:
- Una cuenta de Google
- Conexión a internet
- Un navegador web moderno

Las celdas de configuración en cada cuaderno instalarán automáticamente:

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