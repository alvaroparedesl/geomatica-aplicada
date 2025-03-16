# Geomática Aplicada en Recursos Naturales

## 🌍 Explorando nuestro planeta a través de datos geoespaciales

Este repositorio contiene materiales para el curso de Geomática Aplicada, diseñado para estudiantes de tercer año en recursos naturales. Aquí aprenderás a utilizar Python para análisis geoespacial, procesamiento de imágenes satelitales y visualización de datos ambientales.

## 🔍 ¿Qué aprenderás?

* Manipulación y análisis de datos vectoriales con GeoPandas
* Procesamiento y análisis de datos raster con herramientas modernas
* Conexión a catálogos de imágenes satelitales y servicios geoespaciales
* Aplicación de técnicas geomáticas para resolver problemas reales en recursos naturales

## 🚀 Preparado para comenzar tu viaje geoespacial

Este curso está diseñado para llevarte desde los conceptos básicos hasta aplicaciones avanzadas, con ejemplos prácticos en cada etapa del camino. ¡Prepárate para ver el mundo desde una nueva perspectiva!

## 📝 Entorno de trabajo

Los cuadernos de este curso están optimizados para ejecutarse en **Google Colab**, lo que permite trabajar directamente en tu navegador sin necesidad de instalar software adicional. Cada cuaderno incluye las instrucciones necesarias para configurar el entorno de trabajo.

## 📁 Estructura del Repositorio

```
root/
├── notebooks/                          # Cuadernos Jupyter en formato jupytext
│   ├── 01_vector/                      # Análisis de datos vectoriales y SIG
│   │   ├── 01_sistemas_referencia.py       # Sistemas de referencia geoespacial
│   │   ├── 02_sig_webmapping.py            # Definición de SIG, Web Mapping
│   │   ├── 03_datos_geograficos.py         # Datos Geográficos y mapas temáticos
│   │   ├── 04_ejercicio_datos_geo.py       # Ejercicio de dato geográfico
│   │   ├── 05_datos_vectoriales.py         # Datos vectoriales (puntos, líneas, polígonos)
│   │   ├── 06_analisis_espacial.py         # Análisis espacial (intro, operadores)
│   │   └── 07_aplicaciones_recursos.py     # Aplicaciones en recursos naturales
│   ├── 02_raster/                      # Análisis de datos raster y teledetección
│   │   ├── 01_datos_raster.py              # Definición de datos Raster
│   │   ├── 02_intro_teledeteccion.py       # Introducción a la Teledetección
│   │   ├── 03_principios_fisicos.py        # Principios Físicos de Teledetección
│   │   ├── 04_acceso_imagenes.py           # Acceso a imágenes con pySTAC y PC
│   │   ├── 05_estadisticas_imagenes.py     # Características y estadísticas de imágenes
│   │   ├── 06_visualizacion_imagenes.py    # Tratamiento y visualización de imágenes
│   │   ├── 07_indices_vegetacion.py        # Índices de vegetación (NDVI, EVI, etc.)
│   │   └── 08_analisis_multitemporal.py    # Análisis multitemporal y detección de cambios
│   └── README.md                       # Instrucciones para los cuadernos
├── data/                               # Conjuntos de datos de ejemplo
│   ├── vector/                         # Datos vectoriales
│   └── raster/                         # Datos raster de muestra para intro
├── img/                                # Imágenes para los cuadernos
├── utils/                              # Funciones de utilidad
│   ├── vector_utils.py                 # Utilidades para datos vectoriales
│   └── raster_utils.py                 # Utilidades para datos raster
└── .gitignore                          # Archivos ignorados
```

## 🛠️ Tecnologías Utilizadas

**Análisis de datos vectoriales:**
- Python 3.12+
- GeoPandas
- Folium
- Matplotlib
- Shapely

**Análisis de datos raster:**
- Rioxarray
- Xarray
- Rasterio
- pySTAC
- Planetary Computer
- ODC (Open Data Cube)

## 🔄 Acceso a datos

- **Datos vectoriales**: Se descargarán directamente de fuentes públicas cuando sea posible.
- **Imágenes satelitales**: Se accederá a través de Planetary Computer, lo que permite trabajar con imágenes Landsat y Sentinel-2 sin necesidad de descargarlas. Esta funcionalidad se introduce temprano en el curso para permitir trabajar con imágenes reales desde el principio.
- **Datos de muestra**: Para los primeros cuadernos, se proporcionan algunos datos de ejemplo para facilitar el aprendizaje.

## 📚 Módulos del Curso

### 1. Análisis de Datos Vectoriales y Sistemas de Información Geográfica
- **Sistemas de referencia geoespacial**
  - Conceptos básicos de cartografía
  - Sistemas de coordenadas geográficas y proyectadas
  - Datums y transformaciones
- **Sistemas de Información Geográfica (SIG)**
  - Componentes y funciones de un SIG
  - Web Mapping y geoportales (ejemplo: geoportal CONAF)
- **Datos Geográficos**
  - Mapas temáticos y representación
  - Ejercicios prácticos con datos geográficos
- **Datos Vectoriales**
  - Puntos, líneas y polígonos
  - Trabajo con GeoPandas
  - Operaciones espaciales
- **Análisis Espacial**
  - Introducción y conceptos básicos
  - Operadores de atributos y operadores geométricos
  - Visualización con Folium
- **Aplicaciones en Recursos Naturales**
  - Casos de estudio aplicados

### 2. Análisis de Datos Raster y Teledetección
- **Datos Raster**
  - Conceptos básicos y estructura
  - Manipulación con rioxarray y xarray
- **Introducción a la Teledetección**
  - Principios físicos
  - Sensores y plataformas
  - Resoluciones espacial, espectral, radiométrica y temporal
- **Acceso a Imágenes Satelitales**
  - Trabajo con pySTAC y Planetary Computer
  - Acceso a datos Landsat y Sentinel-2
  - Preparación para análisis
- **Características y Estadísticas de Imágenes**
  - Análisis estadístico básico
  - Histogramas y distribuciones
- **Tratamiento y Visualización de Imágenes**
  - Correcciones y mejoras
  - Técnicas de visualización
- **Índices de Vegetación**
  - NDVI, EVI y otros índices
  - Álgebra de imágenes
  - Aplicaciones en monitoreo ambiental
- **Análisis Multitemporal**
  - Series temporales de imágenes
  - Detección de cambios

---