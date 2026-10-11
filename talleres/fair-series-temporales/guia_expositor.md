# Taller: Datos FAIR, series temporales satelitales y detección de anomalías con Python

## Documento para el expositor

Esta guía es el documento original del expositor, corregido y actualizado. Mantiene su estructura y su voz, pero incorpora las decisiones tomadas después de probar cada herramienta (qué servicio sigue funcionando, qué conjunto de datos llega hasta hoy, cuánto tarda cada carga). Los participantes tienen su propio [`README.md`](README.md). Los cuadernos de cada bloque están en esta carpeta y se citan por su nombre de archivo.

## Descripción general

Taller aplicado de dos días y medio para aprender a encontrar, evaluar, analizar y publicar datos ambientales de forma reproducible. No es un curso teórico. Desde el primer bloque los participantes escriben y ejecutan código en Google Colab, consultan repositorios científicos reales y construyen una serie temporal satelital desde cero.

El hilo conductor es una pregunta concreta: cómo monitorear un fenómeno ambiental a lo largo del tiempo usando datos satelitales abiertos. Para responderla se combinan cuatro herramientas: los principios FAIR para encontrar y evaluar datos, los catálogos STAC para acceder a imágenes satelitales, Python en Google Colab para procesarlos y Git para versionar y compartir el flujo de trabajo. Se usan además asistentes de IA para consultar repositorios de datos y, de forma auxiliar, QGIS para revisar puntos de muestreo.

La zona de estudio es el agua y la costa de Concepción y Talcahuano, hacia el sur hasta el golfo de Arauco, con algo de tierra cuando haga falta. Todo el taller, de la búsqueda de datos a la publicación, ocurre sobre esa zona.

Al finalizar, cada participante tendrá un repositorio con un cuaderno reproducible que va desde la búsqueda del dato hasta la detección de eventos extremos, pasando por la visualización y la interpretación, y un depósito de práctica con metadatos FAIR.

## Dirigido a

Investigadores, estudiantes de posgrado y profesionales en ciencias ambientales, teledetección, geoinformática, oceanografía, ecología y áreas afines. Se requieren conocimientos básicos de Python (variables, listas, funciones). No se necesita experiencia previa en teledetección ni en manejo de datos satelitales. El cupo máximo es de 10 participantes: de ese número dependen los 6 minutos por presentación del bloque 9 y las cuotas de los servidores compartidos del bloque 1.

## Requisitos

* Cuenta de Google para acceder a Colab y a sus Secrets.
* Clave personal de Google AI Studio, de una cuenta con 18 años o más. Las cuentas universitarias pueden tenerla bloqueada por su administrador, así que quien no pueda crearla debe usar una cuenta personal.
* Cuenta de GitHub, con un token de acceso personal de grano fino (*fine-grained PAT*) que se crea en el bloque 8, limitado a un solo repositorio.
* Cuenta en Zenodo Sandbox (sandbox.zenodo.org), distinta de la de zenodo.org.
* QGIS es opcional. Se usa solo como apoyo y no es necesario para completar ningún bloque.
* Conexión estable a internet. No se requiere instalar Python localmente, todo el código se ejecuta en la nube.

El expositor usa su propia clave y su propia cuenta de Google en las demostraciones. Los participantes nunca usan la clave del expositor.

## Duración y horario

Dos días y medio. Días 1 y 2: sesión de mañana de 10:00 a 12:30 y sesión de tarde de 14:30 a 17:30. Día 3: sesión única de 10:00 a 12:30.

Propuesta de reparto de tiempos, que se detalla al inicio de cada bloque:

| Sesión | Bloque | Minutos | Notas |
|---|---|---|---|
| Día 1, mañana | Bienvenida y verificación de Colab | 10 | Clave en Secrets, clonar el repositorio |
| Día 1, mañana | B1 FAIR e IA | 70 | Incluye el ejercicio de metadatos de 15 min |
| Día 1, mañana | Pausa | 10 | |
| Día 1, mañana | B2 STAC | 60 | |
| Día 1, tarde | B3 Sentinel-2 | 75 | |
| Día 1, tarde | Pausa | 15 | |
| Día 1, tarde | B4 Serie temporal | 90 | Termina guardando el parquet |
| Día 2, mañana | B5 Exploración | 85 | Parte desde el parquet, sin recalcular |
| Día 2, mañana | Pausa | 15 | |
| Día 2, mañana | B6 Visualización | 50 | |
| Día 2, tarde | B7 Eventos extremos | 100 | |
| Día 2, tarde | Pausa | 15 | |
| Día 2, tarde | B8 Git | 65 | Los primeros 10 min son para el token |
| Día 3, mañana | Repaso y pulido del cuaderno | 10 | |
| Día 3, mañana | B9 Presentaciones | 65 | 10 personas por 6 min, más cambios |
| Día 3, mañana | B9 Publicación en Zenodo Sandbox | 40 | |
| Día 3, mañana | Cierre y encuesta | 25 | |

Si una pausa o un bloque se alarga, el margen está en B2 (se puede acortar la inspección de metadatos) y en B6 (se puede dejar el mapa como demostración).

## Objetivos de aprendizaje

Al terminar el taller, los participantes serán capaces de:

1. Evaluar datasets científicos según los principios FAIR y usar un asistente de IA conectado a servidores MCP para consultar repositorios como Zenodo y Dataverse, entendiendo por qué unos metadatos pobres empeoran las respuestas.
2. Buscar, filtrar y acceder a datos satelitales Sentinel-2 usando catálogos STAC y Python, tomando en cuenta duplicados por reprocesamiento y el desfase (offset) de reflectancia que depende del catálogo y de la versión de procesamiento.
3. Construir una serie temporal ambiental extrayendo valores en puntos de muestreo, distinguiendo el filtro de nubes de la escena de la máscara de nubes por píxel.
4. Calcular anomalías respecto de una climatología y detectar eventos extremos, como olas de calor y olas de frío en el agua, con un método clásico (Hobday) y con aprendizaje automático (Isolation Forest), reconociendo los límites que imponen los vacíos en los datos.
5. Versionar todo el flujo de trabajo con Git, documentarlo de forma reproducible y publicarlo con metadatos FAIR.

## Día 1. Fundamentos FAIR, IA y acceso a datos satelitales

### 10:00 a 12:30

#### Bloque 1. Principios FAIR y consulta de repositorios con IA

Cuaderno: `01_fair_repositorios_ia.py`. Presupuesto: 70 minutos (15 de explicación FAIR, 25 de demostración y consultas, 15 del ejercicio de metadatos, 15 de comparación y cierre).

Se explican los cuatro principios FAIR (Findable, Accessible, Interoperable, Reusable) con ejemplos concretos de datasets ambientales. La pregunta central es cómo saber si un dataset es realmente reutilizable. Se revisan criterios prácticos como la presencia de un identificador persistente (DOI), una licencia clara, metadatos completos y formatos abiertos.

Después se introduce el uso de asistentes de IA para consultar repositorios de datos de forma directa. La idea es simple: en lugar de buscar manualmente en la web de un repositorio, el asistente consulta la interfaz del repositorio y devuelve resultados verificados. El taller usa servidores MCP (Model Context Protocol) ya existentes. No se enseña a construir servidores MCP, solo a usarlos y a entender para qué sirven.

El cliente es el propio cuaderno: Gemini se llama con `google-genai` (viene preinstalado en Colab), con la clave personal de AI Studio que cada participante guardó en el Secret `GOOGLE_API_KEY`, y la librería `mcp` conecta los servidores (`pip install mcp`). El cuaderno imprime cada paso que da el modelo (qué herramienta llamó, con qué argumentos, cuánto devolvió y cuánto tardó), porque lo que se enseña es qué hizo el modelo y no solo su respuesta.

Servidores que se usan:

| Fuente | Servidor | Notas |
|---|---|---|
| Zenodo | Servidor de la comunidad (cyanheads), hospedado | El mejor para el taller: informa el total de coincidencias y filtra por tipo, año y licencia |
| Dataverse | Servidor de la comunidad GDCC (`https://mcp.dataverse.org/mcp`) | Informa el total, pero no tiene filtros ni facetas, así que se filtra dentro del texto de la consulta |
| Dataverse, respaldo local | Servidor pequeño que corre dentro del Colab de cada participante (`utils/mcp_dataverse.py`) | Agrega filtros, facetas y la lista de campos de metadatos vacíos por dataset. Los participantes no escriben nada |
| OpenAlex (opcional) | Servidor hospedado | Es la fuente correcta para "artículos", porque Zenodo contiene solo una parte pequeña |

Las instalaciones de Dataverse chilenas que funcionan son la de la Universidad de Concepción (`datav.udec.cl`, 17 datasets, un caso natural para esta zona), la de la Universidad de Chile, la de la USACH y datospararesiliencia.cl.

**Demostración guiada por preguntas.** La demostración parte de una pregunta como "¿cuántos artículos de 2026 son de esta región?" y muestra cómo la respuesta cambia según la fuente y según cómo se pregunta. Las respuestas ya fueron verificadas contra las APIs directamente (los totales coinciden en cada caso):

| Pregunta | Respuesta | Qué enseña |
|---|---|---|
| Publicaciones de 2026 en Zenodo que mencionan Biobío | 33 (8 artículos y 21 datasets) | "Biobío" da 29 y "Biobio" da 31: las tildes cambian el conteo |
| Artículos de 2026 de instituciones del Biobío (OpenAlex) | 3.352 | Cuenta por afiliación de los autores, no por tema |
| Datasets de 2026 en el Dataverse de la UdeC | 2 o 3 | Depende de si se cuenta por fecha de cita o por fecha de publicación |
| Datasets de temperatura del mar para Chile en Zenodo | 11 en inglés y 2 en español | El mejor tiene licencia CC BY 4.0 |

**La lección: metadatos pobres empeoran a la IA.** Un asistente solo puede responder con lo que el repositorio declara. Casos reales para mostrar en clase:

* **El dataset `FJBASE`** del Dataverse de la UdeC: un estudio de Arauco sin palabras clave ni cobertura geográfica. Buscar "Biobío" no lo encuentra, buscar "Arauco" sí.
* **El registro de Zenodo titulado solo "Database"**, de un autor de la UdeC, sin descripción ni palabras clave: ninguna búsqueda por tema puede encontrarlo.
* **166 datasets de la UdeC en Zenodo sin ubicación.** Una búsqueda de "Concepción" los devuelve por la afiliación, incluido un dataset sobre Cuba.
* **La ubicación estructurada sí ayuda:** 49 registros aparecen bajo "Chile" solo por su campo de ubicación, porque la palabra no figura en ningún otro lugar.
* **Las tildes y el idioma** cambian los conteos (ver tabla anterior).
* **En Harvard Dataverse**, "datasets sobre Chile" da 266 o 55 según el campo que se busque, y un dataset de temperatura del aire en tierra podría pasar por uno de temperatura del mar.

El asistente recibe instrucciones para responder "no consta" en lugar de inventar un DOI o una licencia, para usar el total que informa el servidor en lugar de contar filas, para decir "revisé N de M" cuando los resultados son parciales y para cerrar con "Cómo lo calculé" y "Límites".

**Ejercicio de metadatos (15 minutos).** No requiere cuentas. Cada participante parte de un registro pobre y hace cuatro preguntas que el asistente responde "no consta". Luego completa título, descripción, palabras clave en español y en inglés, lugar con coordenadas, licencia, fechas, ORCID y ROR, y repite las cuatro preguntas hasta que todas se responden. La conclusión que debe quedar es que mejorar los metadatos propios es la forma más directa de mejorar lo que cualquier asistente podrá decir de ese dato.

Para quien tenga tiempo, la comparación manual (buscar a mano en la web del repositorio, anotar DOI, licencia y formato, y comparar con lo que dijo el asistente en precisión, trazabilidad y tiempo) se hace con Zenodo.

**PANGAEA.** Queda relegado. Su servidor público permite solo 50 llamadas por día por IP, y toda la sala comparte la misma red. Se muestra únicamente como demostración del expositor, o se omite.

**Quién usa qué clave.** El expositor hace las demostraciones con su propia cuenta. Los participantes usan su propia clave desde el primer minuto, para que nadie dependa de la del expositor.

#### Bloque 2. Catálogos STAC y búsqueda de imágenes satelitales

Cuaderno: `02_stac_busqueda.py`. Presupuesto: 60 minutos (15 de conceptos, 30 de búsqueda e inspección, 15 de duplicados y offset).

Se explica qué es un catálogo STAC (SpatioTemporal Asset Catalog) y por qué se ha convertido en el estándar para publicar datos satelitales de forma accesible. Se presentan sus componentes: catálogo, colección, ítem y asset. Después se introduce `pystac-client`, la librería de Python que permite conectarse a cualquier catálogo STAC y hacer búsquedas por área, fecha y otras propiedades. Es un buen momento para señalar que los campos de STAC son metadatos FAIR en acción, el mismo tema del bloque anterior.

El ejercicio consiste en conectarse al catálogo de Microsoft Planetary Computer, definir un área de interés en Concepción, Talcahuano y el golfo de Arauco, y buscar imágenes Sentinel-2 por rango de fechas y porcentaje de nubosidad. Se inspeccionan los metadatos de los resultados: fecha de adquisición, nubosidad, huella geográfica y versión de procesamiento (*processing baseline*).

Dos puntos que se muestran con datos reales:

* **Duplicados por reprocesamiento.** La misma escena puede aparecer dos veces con identificadores de procesamiento distintos. En una prueba, 12 ítems correspondían a solo 6 fechas. Se agrupa por día solar para quedarse con una por fecha.
* **El desfase de reflectancia depende del catálogo y de la versión de procesamiento.** No existe un valor fijo (el famoso `-0.1` aplica a ciertas versiones y a ciertos catálogos). Los helpers del repositorio lo resuelven por imagen y los participantes lo ven una vez de forma explícita, para entender por qué los cuadernos antiguos con un desfase fijo subestiman los índices.

Los cuadernos siguen el patrón nativo del repositorio: conectar el catálogo, buscar con parámetros, preparar los ítems, cargar con `odc.stac.load`, agregar el offset y convertir a reflectancia. Los helpers preparan las llamadas, pero no las esconden.

### 14:30 a 17:30

#### Bloque 3. Acceso y visualización de datos Sentinel-2

Cuaderno: `03_acceso_sentinel2.py`. Presupuesto: 75 minutos (15 de COG, 20 de carga y RGB, 25 de índices, 15 de valor en un punto).

Se explica el concepto de Cloud-Optimized GeoTIFF (COG) y por qué permite leer datos satelitales directamente desde la nube, y solo la ventana que se necesita, sin descargar la escena completa. Se presentan las librerías `rioxarray`, `odc-stac` y `planetary-computer` para cargar imágenes como arreglos multidimensionales de `xarray`.

El ejercicio consiste en cargar una ventana de una escena Sentinel-2 sobre la zona de estudio, visualizar las bandas RGB y calcular índices del agua y de la costa:

* **NDWI**, para separar agua de tierra.
* **NDCI**, un índice asociado a clorofila.
* **Reflectancia en el rojo**, como indicador de turbidez.
* **NDVI**, para la vegetación costera y los humedales.

Finalmente se extrae el valor de un píxel en un punto concreto. Es el primer paso hacia la serie del bloque siguiente.

Este bloque usa solo Sentinel-2. No se incluye Sentinel-3 y tampoco se lo reemplaza por otro sensor (ver la nota técnica sobre Sentinel-3 más abajo).

#### Bloque 4. Construcción de la serie temporal

Cuaderno: `04_series_temporales.py`. Presupuesto: 90 minutos (15 de conceptos, 15 de puntos, 40 de carga y construcción, 10 de comparación de filtros, 10 de guardado). Reserva especial: la carga de 12 meses tarda más de un minuto, así que conviene que todos la lancen a la vez y trabajen mientras tanto en la comparación de filtros.

Se explica cómo pasar de una imagen única a una serie temporal. Los conceptos clave son la revisita del satélite, el filtrado por nubosidad y la extracción de valores en puntos de interés.

QGIS es solo un apoyo. Quien quiera puede crear o revisar los puntos en QGIS y exportarlos a CSV, o mirar una o dos imágenes. Pero el cubo de datos se construye siempre en Colab, y quien no use QGIS parte de un CSV de puntos ya preparado, de tres a cinco ubicaciones en la zona de estudio (por ejemplo agua abierta, bahía, humedal y vegetación costera).

El ejercicio carga los puntos en Colab, busca escenas Sentinel-2 para un período de seis a doce meses y construye la serie por punto. La estrategia es **cargar ventanas pequeñas alrededor de cada punto**, en lugar de la caja completa que los contiene. En una prueba con cinco puntos y doce meses, las ventanas tardaron 78 segundos con 313 MB de memoria, y la carga ansiosa de toda la caja tardó 306 segundos. Esos números se obtuvieron en otra zona, así que conviene volver a medirlos con los puntos del taller.

**Filtro de nubes de la escena frente a máscara SCL.** Son dos cosas distintas. El filtro de la escena (`eo:cloud_cover`) descarta fechas completas, pero no dice nada sobre un punto en particular. La máscara de clasificación de escena (SCL) trabaja por píxel. En la prueba, con escenas de menos de 20% de nubes la máscara no cambió nada, y sin filtro de escena (hasta 100%) enmascaró cerca de 40% de los valores en los puntos. Los participantes repiten la comparación con sus puntos. Conviene mencionar que la lista de clases SCL excluidas por defecto está pensada para vegetación y que cerca del agua conviene revisar qué clases se están descartando.

Para terminar, la serie se guarda en un archivo parquet con sus metadatos. Es el momento FAIR del día: el archivo viaja a los bloques siguientes y también sirve de respaldo (hay un parquet de reserva en el repositorio por si la carga en vivo falla).

Entregable del día: un cuaderno con la serie temporal de Sentinel-2 para los puntos definidos, guardada en parquet.

## Día 2. Análisis, detección de eventos extremos y flujo reproducible

### 10:00 a 12:30

#### Bloque 5. Análisis exploratorio de la serie temporal

Cuaderno: `05_exploracion_series.py`. Presupuesto: 85 minutos (10 de carga del parquet, 25 de estadísticas y suavizado, 20 de tendencia, 30 de climatología y anomalías).

El bloque parte del parquet guardado el día anterior, sin recalcular nada. Se revisan estadísticas descriptivas básicas (media, mediana, percentiles) y técnicas de suavizado como la media móvil. Se introduce el concepto de anomalía: un valor que se aleja significativamente del comportamiento esperado para esa época del año.

El ejercicio consiste en aplicar una media móvil de tres puntos, graficar la serie original, la suavizada y una línea de tendencia calculada con regresión lineal, calcular la **climatología mensual** y las **anomalías** como puntuación z respecto de esa climatología, e interpretar los patrones estacionales. Con esto la serie de Sentinel-2 queda conectada con la idea que se usa en el bloque 7.

#### Bloque 6. Visualización interactiva e interpretación

Cuaderno: `06_visualizacion_interactiva.py`. Presupuesto: 50 minutos (15 de buenas prácticas, 20 de `hvplot`, 15 de `folium`).

Se presentan buenas prácticas para gráficos de series temporales ambientales. Se introducen `hvplot` para gráficos interactivos y `folium` para mapas.

El ejercicio consiste en crear un gráfico interactivo de la serie con `hvplot` y añadir un mapa con la ubicación de los puntos de muestreo usando `folium`. Se interpreta la serie en el contexto del fenómeno estudiado. Los widgets de `hvplot` (selectores y deslizadores) pueden comportarse distinto en Colab que en un entorno local, por lo que se prueban el día anterior (ver las notas técnicas).

### 14:30 a 17:30

#### Bloque 7. Detección de eventos extremos: umbrales clásicos y aprendizaje automático

Cuaderno: `07_eventos_extremos.py`. Presupuesto: 100 minutos (15 de método, 30 de Hobday, 15 de validación, 15 de Isolation Forest, 15 de mapa de frecuencia, 10 de discusión del vacío de datos).

Este bloque introduce la detección de eventos extremos en series temporales ambientales. Se llama "umbrales clásicos y aprendizaje automático" porque el método de Hobday es estadístico y no de IA, y porque se compara con un método de aprendizaje automático. La IA queda reservada para el bloque 1.

**Datos.** Se usa temperatura superficial del mar de NOAA OISST v2.1, obtenida únicamente de Planetary Computer, y no del bucket de Sentinel-3. En el repositorio hay una copia preparada en `datos/` (una serie en un punto frente a Concepción y una malla de 34 a 40 S y 71 a 76 O) para que Colab no dependa de descargas largas.

**Método de Hobday.** Una ola de calor marina es un período de al menos cinco días consecutivos con temperaturas por encima del percentil 90 (una ola de frío, por debajo del percentil 10) respecto de una climatología de referencia. El cuaderno lo implementa en pandas legible, con estos parámetros: línea de base 1991-2020, ventana de 11 días para los percentiles, duración mínima de 5 días y unión de eventos separados por 2 días o menos. El orden importa: primero se filtran las rachas por duración y después se unen las brechas. El resultado se valida contra la librería `xmhw`.

**Comparación con aprendizaje automático.** Se aplica Isolation Forest a la misma serie y se contrasta qué detecta cada método. El punto didáctico es que Isolation Forest no conoce la estacionalidad ni la regla de los cinco días, así que se usa sobre anomalías.

**`marEx`.** Solo se menciona, o se muestra como demostración. No se usa como núcleo del bloque porque no tiene modo de punto, no aplica la regla de los 5 días, no detecta olas de frío, necesita cerca de 12 GB en una malla mediana y declara soporte solo hasta Python 3.12, mientras que Colab usa Python 3.13.

**El vacío de datos como problema didáctico.** Planetary Computer no tiene OISST entre 2024-06-20 y 2026-03-05 (624 días), más los días 2026-03-07 y 2026-03-08. No se rellenó con otra fuente, a propósito. Es un caso real de lo que pasa con una serie incompleta:

* La climatología no se ve afectada, porque se calcula con 1991-2020.
* Dentro del vacío no se pueden detectar eventos, y una racha nunca debe cruzarlo (un evento no puede "unir" dos fechas separadas por casi dos años).
* Se discuten tres opciones: rellenar, ignorar o reportar. Se espera que el grupo discuta el costo de cada una.
* Hay una ola de calor en curso desde 2026-08-17, posterior al vacío, que sirve de ejemplo actual.

**Mapa de frecuencia.** Con la malla de 34 a 40 S y 71 a 76 O se calcula la frecuencia de eventos por píxel y se muestra en un mapa.

**Resultados de referencia.** En una corrida anterior, con el vacío rellenado desde AWS, el píxel frente a Concepción dio 103 olas de calor y 136 olas de frío. Los eventos más notables fueron 1997 (66 días), 1998 (57 días), el El Niño costero de 2017 (48 días) y 1982 (hay que recalcularlo). Con el vacío sin rellenar, los números pueden variar, así que conviene correr el cuaderno completo antes del taller y anotar los valores definitivos.

#### Bloque 8. Git, reproducibilidad y cierre del flujo de trabajo

Cuaderno: `08_git_reproducibilidad.py`. Presupuesto: 65 minutos (10 de verificación del token al inicio, 10 de comandos, 25 de refactorización, 20 de commits, push y verificación).

Este bloque final integra todo lo aprendido en un solo repositorio versionado con Git. El objetivo es que cada participante termine con un proyecto reproducible que pueda compartir.

Se explican los comandos esenciales de Git (`init`, `add`, `commit`, `push`, `pull`, `branch`) y su integración con Google Colab. La trazabilidad es el concepto central: cada cambio queda registrado y se puede revertir o auditar.

**Autenticación.** El token se crea como *fine-grained PAT*, limitado a un solo repositorio, con permiso de contenido de lectura y escritura y una expiración corta. Se guarda en Colab Secrets y se lee con `google.colab.userdata`. Nunca se escribe dentro de una celda, no se imprime y no queda en las salidas ni en `.git/config`: se usa un credential helper o una URL de un solo uso que no se persiste. Dedica los primeros diez minutos del bloque a verificar que todos pueden hacer `git push`.

**Plan de respaldo.** Quien no logre que el token funcione en esos diez minutos usa la opción de Colab "Guardar una copia en GitHub", que no requiere token, y sigue con el resto del bloque.

El ejercicio consiste en refactorizar el código de los cuadernos anteriores en funciones reutilizables, inicializar un repositorio local desde Colab, hacer al menos tres commits descriptivos y subir el repositorio a GitHub. Se verifica que el cuaderno se ejecuta de principio a fin sin errores.

Entregable del día: un repositorio en GitHub con el cuaderno completo, commits documentados, la serie temporal con sus anomalías y los eventos extremos detectados.

## Día 3. Presentación de resultados y cierre

### 10:00 a 12:30

#### Bloque 9. Presentación de proyectos y publicación FAIR

Cuaderno: `09_publicacion_zenodo.py`. Presupuesto: 10 minutos de repaso y pulido, 65 minutos de presentaciones, 40 minutos de publicación y 25 minutos de cierre.

Cada participante presenta su repositorio y los hallazgos principales en **6 minutos**. Con hasta 10 personas son 60 minutos más los cambios, por lo que cabe en 65. Con más personas habría que bajar a 5 minutos. Se discuten las dificultades encontradas, las decisiones tomadas y lo aprendido sobre el fenómeno ambiental analizado. El instructor y los compañeros aportan retroalimentación.

Luego se explica cómo publicar un cuaderno y los datos derivados de forma FAIR. La práctica se hace en **Zenodo Sandbox** (sandbox.zenodo.org), no en Zenodo real: un depósito real crea un DOI público y permanente para material de práctica. En el sandbox los DOI tienen el prefijo 10.5072 y no se resuelven fuera de él. El depósito se hace por la API REST con el token de Zenodo Sandbox de cada participante, guardado en Colab Secrets. Se completan los metadatos mínimos: título, descripción, palabras clave, licencia y autores con ORCID. Si el expositor quiere mostrar un depósito real, lo hace él, como demostración y con su propia cuenta.

El ejercicio consiste en subir el cuaderno y los datos derivados al sandbox, completar los metadatos y verificar que el depósito es recuperable por su DOI.

#### Recapitulación y cierre

Resumen de los conceptos clave: principios FAIR, consulta de repositorios con IA y por qué los metadatos importan, catálogos STAC, Sentinel-2 y su desfase de reflectancia, series temporales, anomalías y eventos extremos (Hobday, Isolation Forest, vacíos de datos) y Git para reproducibilidad. Se comparte una lista de recursos para seguir aprendiendo y se aplica una encuesta final de satisfacción.

## Notas técnicas para el expositor

### Bloque 1. Configuración de IA para consulta de repositorios

Se usa MCP (Model Context Protocol), un estándar abierto que permite a un asistente de IA conectarse a sistemas externos. Los participantes no programan servidores: usan los que existen.

**Qué cambió respecto del plan original.**

* **Gemini CLI ya no sirve.** Desde 2026-06-18 no atiende cuentas personales gratuitas, así que no se usa.
* **`zenodo-mcp` no está en PyPI,** de modo que `pip install zenodo-mcp` no funciona. El servidor de Zenodo es uno hospedado por la comunidad.
* **`gemini auth login` no existe.**
* **El chat de Gemini integrado en Colab no se conecta a servidores MCP.** El "servidor MCP de Colab" que Google anunció en 2026 funciona al revés: deja que agentes externos controlen Colab. Tampoco sirve `google.colab.ai`, que solo expone `generate_text` sin llamadas a herramientas.
* **La ruta recomendada** es `google-genai` (preinstalado) con una clave gratuita de AI Studio guardada en Colab Secrets. El cuaderno es el cliente MCP y lo único que se instala es `pip install mcp`.

**Crear la clave de AI Studio (cada participante, unos 5 minutos).**

1. Entrar a aistudio.google.com/apikey con una cuenta personal de Google, de 18 años o más.
2. Crear una clave nueva.
3. En Colab, abrir el panel **Secrets** (icono de llave), crear el secreto `GOOGLE_API_KEY`, pegar la clave y activar **Acceso desde el cuaderno**.
4. En el cuaderno, la clave se lee con `from google.colab import userdata`. Nunca se escribe en una celda.

El nivel gratuito de AI Studio no tiene límites publicados por modelo, y los datos de las consultas pueden usarse para mejorar productos de Google. Conviene decirlo a los participantes. Si una cuenta universitaria tiene AI Studio bloqueado por el administrador, esa persona debe usar una cuenta personal. Si no tiene ninguna, sigue la demostración del expositor.

**Cuotas.**

* El servidor de Zenodo es hospedado y compartido, y toda la sala pasa por él.
* El servidor de OpenAlex tiene un presupuesto diario compartido. Por eso es opcional y se usa con pocas consultas.
* Los servidores de la comunidad corren en el dominio personal de alguien y pueden caerse. Por eso existe el servidor local de respaldo para Dataverse.
* PANGAEA permite 50 llamadas por día por IP: solo demostración del expositor.
* El servidor de PANGAEA de terceros expone cerca de 40 herramientas sin relación con el taller, y algunas indican al modelo que las prefiera. Si se usa, el cuaderno limita las herramientas que ve el modelo.

**Qué probar el día anterior** (con tu propia cuenta, en un Colab real):

1. Cada servidor MCP por separado: Zenodo, Dataverse GDCC, el servidor local de Dataverse y OpenAlex.
2. Una llamada real a Gemini con una clave real. Es lo único del bloque que no se pudo probar sin clave, así que es la primera prueba.
3. Los widgets de `hvplot` en Colab (selectores y deslizadores), que no se pudieron probar fuera de un entorno local.
4. Un `git push` con el token guardado en Colab Secrets, de principio a fin.
5. Si es posible, repetir la prueba de Gemini con una cuenta universitaria, para saber si el bloqueo es real.

### Bloque 3. Por qué no hay Sentinel-3

El plan original usaba Sentinel-3 desde el bucket público de AWS (MEEO, en formato COG). Se quitó por una razón concreta: **el archivo COG de MEEO no es continuo hasta el presente**. La temperatura superficial (SST) termina en 2024-02 y OLCI no tiene datos entre 2024-03 y 2026-05. Un bloque que dependiera de él fallaría hoy. El producto de Sentinel-3 SLSTR en Planetary Computer sí llega al presente, pero es un producto de franja (*swath*) en NetCDF, con archivos de unos 636 MB cada uno y sin malla regular, que no sirve para extraer el valor de un punto en pocos minutos.

No se reemplazó por otro sensor. El bloque 3 es solo Sentinel-2 y la temperatura del mar aparece en el bloque 7 con OISST.

### Bloque 4. Planetary Computer y el desfase de Sentinel-2

* **Caducidad de las firmas.** Los enlaces a los datos de Planetary Computer van firmados y caducan al cabo de un tiempo. Si una carga falla después de haber dejado el cuaderno un rato parado, el síntoma es un error de acceso (HTTP 403) en una celda que antes funcionaba. Se arregla volviendo a ejecutar la búsqueda y la preparación de los ítems para obtener enlaces nuevos.
* **El desfase (offset) de Sentinel-2 se maneja por imagen.** No existe un valor fijo. Depende del catálogo y de la versión de procesamiento (*baseline*), y los helpers lo resuelven imagen por imagen con `agregar_offset`. Los cuadernos antiguos del repositorio, con un desfase fijo de 0,0, subestiman los índices y no deben copiarse. Esto se prueba mucho porque es fácil equivocarse sin notarlo.
* **Duplicados.** Se agrupa por día solar para quedarse con una escena por fecha.

### Bloque 7. El vacío de OISST

Planetary Computer no tiene OISST del 2024-06-20 al 2026-03-05 ni los días 2026-03-07 y 2026-03-08. Los archivos de `datos/` conservan el vacío (filas vacías en el CSV, fechas ausentes en la malla). Si Planetary Computer completa algún día esos meses, `datos/preparar_oisst.py` regenera los archivos en unos 10 a 15 minutos, y los números de referencia del bloque habría que recalcularlos. La malla llega hasta 2026-09-24.

## Posibles problemas y soluciones

**Gemini no responde o da un error de clave.** Verifica que el secreto se llama exactamente `GOOGLE_API_KEY`, que el acceso desde el cuaderno está activado y que la clave se creó en una cuenta personal. Si la cuenta es institucional, es probable que AI Studio esté bloqueado: usar una cuenta personal.

**Un servidor MCP no responde.** Reintenta una vez. Los servidores de la comunidad y de terceros pueden caerse o llegar a su cuota. Para Dataverse, cambia al servidor local de respaldo. Para Zenodo y OpenAlex, continúa con la demostración del expositor y retoma más tarde.

**La respuesta de la IA no coincide con la de la web.** Es parte de la lección. Comparar el total que informa el servidor, probar variantes con y sin tilde y en inglés, y distinguir lugar de afiliación.

**Una carga de Planetary Computer falla o da un error de acceso.** Probablemente caducaron las firmas: vuelve a ejecutar la búsqueda y la preparación de los ítems. Si el problema persiste o la red está lenta (muchos participantes a la vez), usa el parquet de reserva del repositorio y continúa con el análisis. Conviene escalonar las celdas pesadas del bloque 4.

**La carga de 12 meses tarda demasiado.** Comprueba que se están cargando ventanas alrededor de los puntos y no toda la caja que los contiene.

**Los índices salen subestimados.** Revisa que se aplicó el desfase por imagen (`agregar_offset`) antes de convertir a reflectancia.

**Los widgets de `hvplot` no aparecen en Colab.** Vuelve a ejecutar la extensión de `hvplot` en la misma celda o instala `hvplot` en la sesión. Si no funciona, usa los gráficos sin widgets y deja el mapa de `folium` como demostración.

**Una racha de calor aparece uniendo años distintos en el bloque 7.** Es el vacío de datos: las rachas no deben cruzarlo. Revisa que el cuaderno parte las series en el vacío antes de buscar eventos.

**Los participantes no logran hacer `git push`.** Verifica que crearon un token de grano fino, limitado a un solo repositorio, con permiso de contenido de lectura y escritura, que lo guardaron en Colab Secrets con el nombre que usa el cuaderno y que el acceso desde el cuaderno está activado. Dedica diez minutos al inicio del bloque 8 a esta verificación. Si no resulta, usa "Guardar una copia en GitHub" desde el menú de Colab.

**Un participante pegó un token o una clave en una celda.** Bórralo de la celda y de las salidas, revócalo en GitHub o en AI Studio y crea uno nuevo.

**El depósito en Zenodo Sandbox falla.** Comprueba que el token es del sandbox y no de zenodo.org, y que tiene permisos de escritura. Si no alcanza el tiempo, el expositor completa el depósito en pantalla como demostración guiada.

**Alguien no tiene QGIS.** No hace falta. QGIS es opcional y el CSV de puntos ya está preparado.
