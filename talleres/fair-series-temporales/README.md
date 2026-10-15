# Taller: Datos FAIR, series temporales satelitales y detección de anomalías con Python

[![Licencia: GPL-3.0](https://img.shields.io/badge/licencia-GPL--3.0-blue.svg)](../../LICENSE) [![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://colab.research.google.com/) [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/talleres/fair-series-temporales/01_fair_repositorios_ia.ipynb)

Taller aplicado de dos días y medio para aprender a encontrar, evaluar, analizar y publicar datos ambientales de forma reproducible. No es un curso teórico: desde el primer bloque escribes y ejecutas código en Google Colab, consultas repositorios científicos reales y construyes una serie temporal satelital desde cero.

El hilo conductor es una pregunta concreta: cómo monitorear un fenómeno ambiental a lo largo del tiempo usando datos satelitales abiertos. Para responderla combinarás los principios FAIR, los catálogos STAC, Python en Google Colab y Git. Al final tendrás un repositorio con un cuaderno reproducible que va desde la búsqueda del dato hasta la detección de eventos extremos, y un depósito de práctica con metadatos FAIR.

## A quién está dirigido

Investigadores, estudiantes de posgrado y profesionales en ciencias ambientales, teledetección, geoinformática, oceanografía, ecología y áreas afines. Necesitas conocimientos básicos de Python (variables, listas, funciones). No hace falta experiencia previa en teledetección ni en datos satelitales. El taller tiene un máximo de 10 participantes.

## Requisitos

* **Cuenta de Google**, para usar Colab y sus Secrets.
* **Clave personal de Google AI Studio** (para el bloque 1). Debe ser de tu cuenta personal y tener 18 años o más. Algunas cuentas universitarias o institucionales tienen AI Studio bloqueado por su administrador: si es tu caso, usa una cuenta personal. Cada participante usa su propia clave; no se comparten claves.
* **Cuenta de GitHub**, con un token de acceso personal de grano fino (*fine-grained PAT*) que crearás durante el bloque 8, limitado a un solo repositorio.
* **Cuenta en Zenodo Sandbox** ([sandbox.zenodo.org](https://sandbox.zenodo.org)), distinta de la cuenta de Zenodo real, para el bloque 9.
* **QGIS** (opcional). Sirve para crear o revisar puntos de muestreo y mirar una o dos imágenes, pero todo el análisis se hace en Colab.
* Conexión estable a internet. No necesitas instalar Python: todo el código se ejecuta en la nube.

## Cuadernos

Cada cuaderno se abre directamente en Colab con su botón. La fuente es el archivo `.py` (formato jupytext `py:percent`) y el `.ipynb` se genera a partir de él, igual que en el resto del repositorio.

| Bloque | Cuaderno | Colab |
|---|---|---|
| 1 | [`01_fair_repositorios_ia.py`](01_fair_repositorios_ia.py): principios FAIR y consulta de repositorios con IA | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/talleres/fair-series-temporales/01_fair_repositorios_ia.ipynb) |
| 2 | [`02_stac_busqueda.py`](02_stac_busqueda.py): catálogos STAC y búsqueda de imágenes | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/talleres/fair-series-temporales/02_stac_busqueda.ipynb) |
| 3 | [`03_acceso_sentinel2.py`](03_acceso_sentinel2.py): acceso y visualización de Sentinel-2 | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/talleres/fair-series-temporales/03_acceso_sentinel2.ipynb) |
| 4 | [`04_series_temporales.py`](04_series_temporales.py): construcción de la serie temporal | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/talleres/fair-series-temporales/04_series_temporales.ipynb) |
| 5 | [`05_exploracion_series.py`](05_exploracion_series.py): exploración, climatología y anomalías | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/talleres/fair-series-temporales/05_exploracion_series.ipynb) |
| 6 | [`06_visualizacion_interactiva.py`](06_visualizacion_interactiva.py): visualización interactiva e interpretación | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/talleres/fair-series-temporales/06_visualizacion_interactiva.ipynb) |
| 7 | [`07_eventos_extremos.py`](07_eventos_extremos.py): detección de eventos extremos, umbrales clásicos y aprendizaje automático | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/talleres/fair-series-temporales/07_eventos_extremos.ipynb) |
| 8 | [`08_git_reproducibilidad.py`](08_git_reproducibilidad.py): Git y reproducibilidad | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/talleres/fair-series-temporales/08_git_reproducibilidad.ipynb) |
| 9 | [`09_publicacion_zenodo.py`](09_publicacion_zenodo.py): publicación FAIR en Zenodo Sandbox | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/talleres/fair-series-temporales/09_publicacion_zenodo.ipynb) |

**Cuaderno completo (opcional).** [`analisis_completo.py`](analisis_completo.py) reúne los bloques 3 a 7 en un solo cuaderno, sin usar `utils/`: cambias tus puntos, fechas y área en la celda de configuración y ejecutas todo el análisis (escena Sentinel-2, serie temporal, exploración, gráficos interactivos y eventos extremos con Landsat). Cada módulo se puede ejecutar por separado. [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/talleres/fair-series-temporales/analisis_completo.ipynb)

Todos los cuadernos empiezan con la celda **Preparar entorno**, que clona este repositorio en Colab para poder usar la carpeta [`utils/`](../../utils/README.md). Las dependencias adicionales del taller están en [`requirements.txt`](requirements.txt) y cada cuaderno instala lo que falte.

## Horario

| Día | Horario | Bloque | Cuaderno |
|---|---|---|---|
| 1 | 10:00 a 12:30 | 1. Principios FAIR y consulta de repositorios con IA | `01` |
| 1 | 10:00 a 12:30 | 2. Catálogos STAC y búsqueda de imágenes | `02` |
| 1 | 14:30 a 17:30 | 3. Acceso y visualización de Sentinel-2 | `03` |
| 1 | 14:30 a 17:30 | 4. Construcción de la serie temporal | `04` |
| 2 | 10:00 a 12:30 | 5. Exploración de la serie, climatología y anomalías | `05` |
| 2 | 10:00 a 12:30 | 6. Visualización interactiva e interpretación | `06` |
| 2 | 14:30 a 17:30 | 7. Detección de eventos extremos | `07` |
| 2 | 14:30 a 17:30 | 8. Git, reproducibilidad y cierre del flujo | `08` |
| 3 | 10:00 a 12:30 | 9. Presentaciones y publicación FAIR | `09` |

Los bloques de una misma sesión se siguen uno tras otro, con una pausa entre ellos.

## Zona de estudio

El agua y la costa de Concepción y Talcahuano, hacia el sur hasta el golfo de Arauco, con algo de tierra cuando haga falta (por ejemplo, humedales y vegetación costera). Con Sentinel-2 trabajarás en puntos y ventanas pequeñas de esa zona. Con la temperatura de superficie de Landsat trabajarás en esos mismos puntos y en una malla de 500 m (0,8 grados) alrededor de la bahía de Concepción y el golfo de Arauco.

## Datos que se usan

* **Sentinel-2 L2A**, leído como COG desde el catálogo STAC de [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/). No se descarga nada al repositorio.
* **Landsat 8 y 9 Collection 2 Level-2** (USGS: temperatura de superficie y reflectancia, 30 m, un paso cada 8 días con los dos satélites, desde 2013), también desde Planetary Computer. Los datos llegan como números enteros que se **calibran** a °C y a reflectancia. Una copia preparada de las series por punto y de una malla de 500 m está en [`datos/`](datos/README.md) para que Colab no dependa de descargas largas. Las nubes y la revisita de 8 días dejan solo unas 10 a 25 observaciones válidas por año: se trabaja a propósito en el bloque 7.
* **Repositorios de datos y publicaciones**, consultados con IA en el bloque 1: Zenodo, Dataverse (incluyendo el de la Universidad de Concepción) y, de forma opcional, OpenAlex.

## Qué entregas

* **Día 1:** un cuaderno con la serie temporal de Sentinel-2 en los puntos que definiste, guardada en un archivo parquet.
* **Día 2:** un repositorio en GitHub con el cuaderno completo, commits documentados, la serie con sus anomalías y los eventos extremos detectados.
* **Día 3:** una presentación de 6 minutos y un depósito en Zenodo Sandbox con metadatos mínimos, recuperable por su DOI.

## Cómo prepararte antes del día 1

Dedica unos 30 minutos a esta lista. Los puntos 1 a 3 son los que más tiempo toman si se dejan para el taller.

- [ ] **1. Colab.** Abre [Google Colab](https://colab.research.google.com/) con tu cuenta de Google y confirma que puedes ejecutar una celda.
- [ ] **2. Clave de AI Studio.** Entra a [aistudio.google.com/apikey](https://aistudio.google.com/apikey) con tu cuenta personal (18 años o más) y crea una clave. Si tu cuenta es institucional y no puedes crearla, usa una cuenta personal.
- [ ] **3. Guárdala en Colab Secrets.** En Colab abre el panel izquierdo, pulsa el icono de la llave (**Secrets**), agrega un secreto llamado `GOOGLE_API_KEY`, pega la clave como valor y activa **Acceso desde el cuaderno**. No pegues la clave en ninguna celda ni la compartas.
- [ ] **4. GitHub.** Ten lista tu cuenta de GitHub. El token se crea durante el bloque 8, no antes.
- [ ] **5. Zenodo Sandbox.** Crea una cuenta en [sandbox.zenodo.org](https://sandbox.zenodo.org). Es independiente de zenodo.org. Si tienes ORCID, ten a mano tu identificador.
- [ ] **6. QGIS (opcional).** Instálalo solo si quieres explorar los puntos fuera de Colab. Si no lo instalas, no te pierdes nada del taller.
- [ ] **7. Conexión.** Prueba tu conexión a internet y lleva el cargador del computador.

## Licencia

Este material se distribuye bajo la licencia GNU GPL v3. Consulta el archivo [LICENSE](../../LICENSE). Los datos de Landsat (USGS) son de dominio público.
