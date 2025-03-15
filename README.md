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

## 📁 Estructura del Repositorio

```
root/
├── notebooks/                          # Cuadernos Jupyter en formato jupytext
│   ├── 01_vector/                      # Análisis de datos vectoriales y SIG
│   │   ├── 01_sistemas_referencia_1.py     # Sistemas de referencia geoespacial 1
│   │   ├── 02_sistemas_referencia_2.py     # Sistemas de referencia geoespacial 2
│   │   ├── 03_sistemas_referencia_3.py     # Sistemas de referencia geoespacial 3
│   │   ├── 04_sig_webmapping.py            # Definición de SIG, Web Mapping
│   │   ├── 05_datos_geograficos.py         # Datos Geográficos y mapas temáticos
│   │   ├── 06_ejercicio_datos_geo.py       # Ejercicio de dato geográfico
│   │   ├── 07_datos_vectoriales_1.py       # Definición de dato vectorial
│   │   ├── 08_datos_vectoriales_2.py       # Continuación datos vectoriales
│   │   ├── 09_analisis_espacial_intro.py   # Análisis espacial introducción
│   │   ├── 10_analisis_espacial_avanz.py   # Análisis espacial operadores
│   │   └── 11_aplicaciones_recursos.py     # Aplicaciones en recursos naturales
│   ├── 02_raster/                      # Análisis de datos raster y teledetección
│   │   ├── 01_datos_raster.py              # Definición de datos Raster
│   │   ├── 02_intro_teledeteccion.py       # Introducción a la Teledetección
│   │   ├── 03_principios_fisicos_1.py      # Principios Físicos de Teledetección 1
│   │   ├── 04_principios_fisicos_2.py      # Principios Físicos de Teledetección 2
│   │   ├── 05_principios_fisicos_3.py      # Principios Físicos de Teledetección 3
│   │   ├── 06_estadisticas_imagenes_1.py   # Estadísticas de imágenes 1
│   │   ├── 07_estadisticas_imagenes_2.py   # Estadísticas de imágenes 2
│   │   ├── 08_visualizacion_imagenes_1.py  # Visualización de imágenes 1
│   │   ├── 09_visualizacion_imagenes_2.py  # Visualización de imágenes 2
│   │   ├── 10_indices_vegetacion_pp.py     # Índices de vegetación, PP
│   │   ├── 11_indices_vegetacion_ndvi.py   # NDVI, EVI, álgebra de imágenes
│   │   ├── 12_otros_indices.py             # Otros índices espectrales
│   │   └── 13_catalogo_pystac.py           # Conexión a catálogos STAC
│   └── README.md                       # Instrucciones para los cuadernos
├── data/                               # Conjuntos de datos de ejemplo
│   ├── vector/                         # Datos vectoriales
│   └── raster/                         # Datos raster
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

## 📚 Módulos del Curso

### 1. Análisis de Datos Vectoriales y Sistemas de Información Geográfica
- **Sistemas de referencia geoespacial** (3 sesiones)
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
  - Principios físicos (3 sesiones)
  - Sensores y plataformas
  - Resoluciones espacial, espectral, radiométrica y temporal
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
- **Conexión a Catálogos de Datos**
  - Trabajo con pySTAC
  - Acceso a Planetary Computer

---