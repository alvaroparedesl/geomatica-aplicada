# Geomática Aplicada en Recursos Naturales

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/notebooks/00_intro.ipynb) [![Licencia: GPL-3.0](https://img.shields.io/badge/licencia-GPL--3.0-blue.svg)](LICENSE) [![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://colab.research.google.com/) [![Último commit](https://img.shields.io/github/last-commit/alvaroparedesl/geomatica-aplicada?label=%C3%BAltimo%20commit)](https://github.com/alvaroparedesl/geomatica-aplicada/commits/main)

Materiales del curso de Geomática Aplicada, pensado para estudiantes de tercer año de recursos naturales. Aquí aprenderás a usar Python para el análisis de datos geoespaciales, el procesamiento de imágenes satelitales y la visualización de datos ambientales, con ejemplos prácticos en cada etapa.

## Qué aprenderás

* Manipulación y análisis de datos vectoriales con Shapely y GeoPandas
* Procesamiento y análisis de datos raster con xarray y rioxarray
* Acceso a imágenes satelitales (Landsat y Sentinel-2) desde el catálogo STAC de Microsoft Planetary Computer
* Cálculo de índices de vegetación y análisis multitemporal
* Aplicación de técnicas geomáticas a problemas reales en recursos naturales

## Cómo usar los cuadernos en Google Colab

Los cuadernos están pensados para ejecutarse en [Google Colab](https://colab.research.google.com/), directamente en el navegador y sin instalar nada. Solo necesitas una cuenta de Google.

1. Abre el cuaderno que quieras con su botón **Abrir en Colab** (aparece al inicio de cada cuaderno y en las tablas de los README de [`notebooks/01_vector`](notebooks/01_vector/README.md) y [`notebooks/02_raster`](notebooks/02_raster/README.md)). Se recomienda comenzar por el [cuaderno de introducción](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/notebooks/00_intro.ipynb).
2. Ejecuta primero la celda **Preparar entorno**. En Colab clona este repositorio en `/content/geomatica-aplicada` (solo la primera vez de la sesión) para que el cuaderno pueda usar la carpeta `utils/`.
3. Ejecuta la celda de instalación de paquetes (`%pip install ...`) y luego el resto de las celdas en orden.

Si prefieres trabajar en tu computador, clona el repositorio, instala las dependencias y abre los cuadernos con Jupyter o VS Code:

```bash
git clone https://github.com/alvaroparedesl/geomatica-aplicada
cd geomatica-aplicada
pip install -r requirements.txt
```

El archivo [`requirements.txt`](requirements.txt) lista todas las dependencias de los cuadernos. En Colab la mayoría ya viene instalada y cada cuaderno instala lo que falte. Como referencia, Colab usa actualmente Python 3.13.

## Estructura del repositorio

```
geomatica-aplicada/
├── notebooks/                                  # Cuadernos del curso (fuente en .py, versión .ipynb generada)
│   ├── 00_intro.py                             # Introducción a Jupyter y Google Colab
│   ├── 01_vector/                              # Datos vectoriales
│   │   ├── 01_introduccion_datos_vectoriales.py
│   │   ├── 02_introduccion_geopandas.py
│   │   ├── 03_operaciones_espaciales.py
│   │   └── README.md
│   ├── 02_raster/                              # Datos raster y teledetección
│   │   ├── 00_demo_imagenes.py
│   │   ├── 01_datos_raster.py
│   │   ├── 02_intro_teledeteccion.py
│   │   ├── 03_acceso_imagenes.py
│   │   ├── 04_estadisticas_imagenes.py
│   │   ├── 05_ejercicio_histograma_landsat.py
│   │   ├── 06_indices_vegetacion.py
│   │   ├── 07_analisis_multitemporal.py
│   │   └── README.md
│   └── README.md                               # Guía de los cuadernos y su flujo de trabajo
├── utils/                                      # Funciones de apoyo reutilizables
│   ├── vector.py                               # Descarga de capas y cálculo de áreas
│   ├── stac.py                                 # Parámetros de búsqueda y carga STAC, offset de Sentinel-2
│   ├── raster.py                               # Reflectancia, máscaras de nubes, índices, series
│   ├── viz.py                                  # Gráficos, mapas y widgets interactivos
│   ├── tests/                                  # Pruebas (pytest), con datos de prueba reales
│   └── README.md                               # Documentación de las funciones
├── talleres/                                   # (próximamente) talleres complementarios
├── scripts/                                    # Verificaciones locales (pre-commit)
├── requirements.txt                            # Dependencias de los cuadernos
├── requirements-dev.txt                        # Dependencias para desarrollo y pruebas
├── .pre-commit-config.yaml                     # Verificaciones antes de cada commit
├── CLAUDE.md                                   # Instrucciones para asistentes de código
├── requirements-sync.txt                       # Dependencias de la herramienta de sincronización
├── sync_to_gdrive.py                           # Sincronización de cuadernos con Google Drive
├── sync_to_gdrive.sh                           # Envoltorio en bash de sync_to_gdrive.py
├── SYNC_SETUP.md                               # Guía de configuración de la sincronización
├── LICENSE                                     # GNU GPL v3
└── README.md
```

Cada cuaderno existe en dos formatos: el archivo `.py` (formato jupytext `py:percent`) es la **fuente** que se edita, y el `.ipynb` se genera a partir de él. Más detalles en [`notebooks/README.md`](notebooks/README.md).

## La carpeta `utils/`

Para que los cuadernos se concentren en lo que enseñan, el código repetitivo (buscar imágenes, convertir a reflectancia, enmascarar nubes, calcular índices, hacer gráficos) vive en la carpeta [`utils/`](utils/README.md) y se importa con `from utils.<módulo> import ...`.

Para que esto funcione tanto en Colab como en local, la primera celda de código de los cuadernos es siempre la misma:

```python
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
```

En Colab clona el repositorio y en un computador local busca la carpeta `utils/` hacia arriba desde el cuaderno. En ambos casos deja definida la variable `RAIZ` (carpeta del repositorio). La documentación de cada función está en [`utils/README.md`](utils/README.md).

## Acceso a datos

* **Datos vectoriales**: se descargan directamente de fuentes públicas (por ejemplo Natural Earth) o se crean en el propio cuaderno.
* **Imágenes satelitales y datos raster**: se acceden en línea a través de Microsoft Planetary Computer (catálogo STAC), sin descargar archivos al repositorio. Esto incluye Landsat, Sentinel-2, la cobertura de suelo ESA WorldCover y el modelo de elevación Copernicus DEM. Para ejecutar los cuadernos de la sección raster necesitas conexión a internet.

## Sincronización con Google Drive (opcional)

La herramienta [`sync_to_gdrive.py`](sync_to_gdrive.py) (con su envoltorio [`sync_to_gdrive.sh`](sync_to_gdrive.sh)) convierte los cuadernos `.py` a `.ipynb` y los sube a una carpeta de Google Drive, manteniendo la estructura de carpetas y subiendo solo lo que cambió. Es útil para quienes mantienen el curso y quieren abrir las versiones actualizadas en Colab desde Drive.

```bash
pip install -r requirements-sync.txt
./sync_to_gdrive.sh setup          # configuración inicial (una sola vez)
./sync_to_gdrive.sh sync --all     # sincronizar todos los cuadernos
```

Para ver todas las opciones (`status`, `config`, `reset`, `--dry-run`, `--force`, etc.) y la configuración paso a paso de las credenciales, consulta [SYNC_SETUP.md](SYNC_SETUP.md).

## Próximamente: talleres

Se agregará una carpeta `talleres/`. El primer taller, de 2,5 días, será **Datos FAIR, series temporales satelitales y detección de anomalías** y vivirá en `talleres/fair-series-temporales/`.

## Contenido del curso

El detalle de cada cuaderno está en los README de cada sección:

* [`notebooks/01_vector`](notebooks/01_vector/README.md): datos vectoriales con Shapely y GeoPandas
* [`notebooks/02_raster`](notebooks/02_raster/README.md): datos raster, teledetección e índices de vegetación

## Desarrollo y pruebas

Esta sección es para quien mantiene el curso. Las verificaciones corren **en tu computador, antes de cada commit**, con [pre-commit](https://pre-commit.com/). No hay integración continua (GitHub Actions) a propósito: todo se revisa localmente.

Instala las dependencias de desarrollo y activa los hooks (una sola vez por clon):

```bash
pip install -r requirements-dev.txt
pre-commit install
```

Desde ese momento, cada `git commit` ejecuta los hooks sobre los archivos modificados. Los hooks usan el `python` del entorno activo, así que activa tu entorno virtual antes de hacer commit. Para ejecutarlos a mano sobre todo el repositorio:

```bash
pre-commit run --all-files
```

Qué revisa cada hook (definidos en [`.pre-commit-config.yaml`](.pre-commit-config.yaml)):

* **Pruebas de `utils/` sin internet**: ejecuta `pytest utils -m "not red" -q` cuando cambia algo dentro de `utils/`. Dura unos segundos.
* **Cuadernos sincronizados**: [`scripts/verificar_notebooks.py`](scripts/verificar_notebooks.py) comprueba que cada cuaderno `.py` de `notebooks/` y `talleres/` tenga un `.ipynb` con las mismas celdas (se ignoran salidas y metadatos). Si falla, indica el comando para regenerarlo: `jupytext --to notebook <archivo>.py`.
* **Sin rayas largas**: falla si aparece el carácter U+2014 en archivos `.py`, `.md` o `.ipynb` de `notebooks/`, `talleres/` y `utils/`, o en los `.md` de la raíz. El estilo del curso usa comas, paréntesis o dos puntos en su lugar.
* **Higiene básica**: YAML válido, marcas de conflicto de merge, claves privadas y archivos de más de 1 MB. Ninguno modifica archivos.

Las pruebas con internet (marcador `red`) consultan los catálogos STAC reales y tardan unos 12 minutos, por eso no corren en cada commit. Ejecútalas a mano antes de un release, de un curso o de un taller:

```bash
pytest utils -m red
```

También conviene una ejecución de humo de los cuadernos (que corran de principio a fin con internet) antes de un curso o un taller:

```bash
jupyter nbconvert --to notebook --execute --output-dir /tmp/humo notebooks/02_raster/*.ipynb
```

## Licencia

Este proyecto se distribuye bajo la licencia GNU GPL v3. Consulta el archivo [LICENSE](LICENSE).
