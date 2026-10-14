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
# # Introducción a Jupyter Notebooks y Google Colab
#
# [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/notebooks/00_intro.ipynb) [![Licencia: GPL-3.0](https://img.shields.io/badge/licencia-GPL--3.0-blue.svg)](https://github.com/alvaroparedesl/geomatica-aplicada/blob/main/LICENSE) [![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://colab.research.google.com/) [![Último commit](https://img.shields.io/github/last-commit/alvaroparedesl/geomatica-aplicada?label=%C3%BAltimo%20commit)](https://github.com/alvaroparedesl/geomatica-aplicada/commits/main)
#
# Bienvenidos al curso de Geomática Aplicada en Recursos Naturales. Antes de comenzar con el contenido específico, es importante familiarizarse con las herramientas que utilizaremos durante el curso.

# %% [markdown]
# ## ¿Qué son los Jupyter Notebooks?
#
# Los Jupyter Notebooks son documentos interactivos que permiten combinar:
#
# * **Código ejecutable** (Python en nuestro caso)
# * **Texto explicativo** (en formato Markdown)
# * **Visualizaciones** (gráficos, mapas, imágenes)
# * **Ecuaciones matemáticas** (usando LaTeX)
# * **Resultados de ejecución** (tablas, salidas de código)
#
# Esta combinación los hace ideales para la enseñanza, ya que permiten explicar conceptos teóricos y mostrar su aplicación práctica en un mismo documento.
#
# ### Estructura de un Notebook
#
# Un notebook está compuesto por **celdas** que pueden ser de diferentes tipos:
#
# * **Celdas de código**: Contienen código Python que puede ejecutarse y mostrar resultados
# * **Celdas Markdown**: Contienen texto formateado, títulos, listas, enlaces, imágenes, etc.
# * **Celdas Raw**: Contienen texto sin formato que no se procesa
#
# ### Ventajas de los Notebooks
#
# * **Interactividad**: Puedes ejecutar código y ver resultados inmediatamente
# * **Documentación integrada**: Código y explicaciones en un mismo lugar
# * **Reproducibilidad**: Facilitan compartir análisis completos con datos y código
# * **Visualización**: Integración directa de gráficos y mapas
# * **Narrativa científica**: Permiten contar una historia con datos

# %% [markdown]
# ## Google Colab: Jupyter en la nube
#
# Google Colaboratory (Colab) es un servicio gratuito basado en la nube que permite ejecutar notebooks de Jupyter sin necesidad de configuración local.
#
# Al ser un computador en la nube, los archivos se guardan en el navegador y no en el computador local.
#
# Cada vez que inicias sesión, te conectas a un computador en línea en los servidores de Google. Pasado cierta cantidad de tiempo (algunas horas), esta sesión se termina y se cierra; los archivos que tengas abiertos se pierden, por lo que es importante que respaldes la información si es de importancia. Por esta misma razón, cuando vuelvas a inciiar sesión, tendrás que instalar las librerías y descargar los archivos nuevamente.

# %% [markdown]
# ### Características principales de Google Colab
#
# * **Ejecución en la nube**: No requiere instalación local de Python ni bibliotecas
# * **Recursos computacionales gratuitos**: Acceso a CPU, GPU y TPU
# * **Integración con Google Drive**: Facilita guardar y compartir notebooks
# * **Bibliotecas preinstaladas**: Muchas bibliotecas científicas y de análisis de datos ya están disponibles
# * **Colaboración en tiempo real**: Similar a Google Docs
# * **Compartir fácilmente**: Mediante enlaces o incrustando en sitios web

# %% [markdown]
# ### Cómo usar Google Colab
#
# #### Acceso
#
# 1. Ve a [colab.research.google.com](https://colab.research.google.com)
# 2. Inicia sesión con tu cuenta de Google
# 3. Crea un nuevo notebook o abre uno existente

# %%
# Esta es una celda de código de ejemplo
# Puedes ejecutarla haciendo clic en el botón de reproducción a la izquierda
# o presionando Shift+Enter
print("¡Hola desde Google Colab!")

# %% [markdown]
# #### Interfaz principal
#
# ![Interfaz de Google Colab](https://colab.research.google.com/img/colab_favicon_256px.png)
#
# * **Barra de menú**: Archivo, Editar, Ver, Insertar, Tiempo de ejecución, etc.
# * **Barra de herramientas**: Botones para añadir celdas, ejecutar código, etc.
# * **Panel de navegación**: Acceso a archivos, tablas de contenido, etc.
# * **Área de trabajo**: Donde se muestran y editan las celdas

# %% [markdown]
# #### Trabajando con archivos
#
# En Google Colab puedes:
#
# 1. **Subir archivos** desde tu computadora:

# %%
# Código para subir archivos (descomenta para usar)
# from google.colab import files
# uploaded = files.upload()

# %% [markdown]
# 2. **Acceder a archivos en Google Drive**:

# %%
# Código para montar Google Drive (descomenta para usar)
# from google.colab import drive
# drive.mount('/content/drive')

# %% [markdown]
# 3. **Clonar repositorios de GitHub**: así descargamos las funciones de apoyo del curso (carpeta `utils/`) y los datos de ejemplo. Lo veremos en detalle más abajo, en la sección "Preparar el entorno en cada notebook".

# %% [markdown]
# #### Instalación de bibliotecas
#
# Puedes instalar bibliotecas adicionales usando pip:

# %%
# Ejemplo de instalación de bibliotecas (descomenta para usar)
# %pip install rioxarray xarray matplotlib numpy rasterio xarray-spatial geopandas

# %% [markdown]
# ## Preparar el entorno en cada notebook
#
# Para no repetir en cada cuaderno el mismo código (buscar imágenes satelitales, calcular índices, hacer gráficos...), las funciones comunes del curso están en la carpeta [`utils/`](https://github.com/alvaroparedesl/geomatica-aplicada/tree/main/utils) del repositorio.
#
# Por eso los notebooks que usan esas funciones comienzan con una celda que es **siempre la misma** (`Preparar entorno`). Esa celda:
#
# * En **Google Colab**: descarga (clona) el repositorio en `/content/geomatica-aplicada`, solo la primera vez de cada sesión.
# * En **tu computador**: busca la carpeta del repositorio hacia arriba desde donde está el notebook.
# * En ambos casos deja disponible la variable `RAIZ` (la carpeta del repositorio, útil para llegar a cualquier archivo del repositorio) y permite escribir `from utils... import ...`.
#
# Ejecútala antes que cualquier otra celda:

# %%
# @title Preparar entorno (ejecutar primero)
import subprocess
import sys
from pathlib import Path

if "google.colab" in sys.modules:  # Colab: clonar el repositorio (y actualizarlo si ya estaba)
    RAIZ = Path("/content/geomatica-aplicada")
    if not RAIZ.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/alvaroparedesl/geomatica-aplicada", str(RAIZ)], check=True)
    else:  # una sesión anterior ya lo clonó: traer lo nuevo para no usar código viejo
        subprocess.run(["git", "-C", str(RAIZ), "pull", "-q", "--ff-only"], check=False)
else:  # Local: subir desde la carpeta actual hasta encontrar utils/
    RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "utils").is_dir())
sys.path.insert(0, str(RAIZ))

print(f"Repositorio listo en: {RAIZ}")

# %% [markdown]
# Una vez ejecutada, ya puedes importar las funciones del curso. Por ejemplo, para descargar un mapa de países del mundo y ver sus primeras filas:

# %%
from utils.vector import descargar_natural_earth

paises = descargar_natural_earth()
paises[["NAME", "CONTINENT", "POP_EST"]].head()

# %% [markdown]
# ### Consejos para trabajar con Google Colab
#
# 1. **Guarda tu trabajo regularmente**: Usa Ctrl+S o el menú Archivo > Guardar
# 2. **Reinicia el entorno cuando sea necesario**: Tiempo de ejecución > Reiniciar entorno de ejecución
# 3. **Usa atajos de teclado**: Ctrl+M H muestra todos los atajos disponibles
# 4. **Ejecuta celdas en orden**: Para evitar errores de dependencias
# 5. **Aprovecha la GPU/TPU**: Para tareas intensivas, cambia el tipo de entorno en Tiempo de ejecución > Cambiar tipo de entorno de ejecución
# 6. **Guarda en GitHub o Drive**: Para no perder tu trabajo

# %% [markdown]
# ## Formato de los notebooks en este curso
#
# En este curso, los notebooks están almacenados en formato `.py` utilizando jupytext para facilitar el control de versiones. Esto significa que:
#
# 1. Los archivos se ven como código Python normal con comentarios especiales
# 2. Google Colab puede abrirlos directamente
# 3. Las celdas Markdown se indican con `# %% [markdown]`
# 4. Las celdas de código se indican con `# %%`
#
# ### Cómo abrir estos notebooks en Google Colab
#
# Hay varias formas de abrir estos notebooks:
#
# 1. **Directamente desde GitHub**: Usa el botón "Abrir en Colab" que aparece al inicio de cada notebook (y en los README del repositorio)
# 2. **Clonando el repositorio**: Como hace la celda `Preparar entorno` de cada notebook
# 3. **Subiendo el archivo**: Desde tu computadora a Colab

# %% [markdown]
# ## Estructura del curso
#
# Este curso está dividido en dos secciones principales:
#
# 1. **Análisis de Datos Vectoriales y SIG** (`01_vector`): geometrías con Shapely, GeoPandas y operaciones espaciales
# 2. **Análisis de Datos Raster y Teledetección** (`02_raster`): datos raster, teledetección, acceso a imágenes satelitales, estadísticas, índices de vegetación y análisis multitemporal
#
# Cada sección contiene varios notebooks que cubren temas específicos. Te recomendamos seguirlos en orden numérico para una mejor comprensión.

# %% [markdown]
# ## ¡Comencemos!
#
# Ahora que conoces las herramientas que utilizaremos, estás listo para comenzar con el contenido del curso. ¡Disfruta aprendiendo Geomática Aplicada en Recursos Naturales!
