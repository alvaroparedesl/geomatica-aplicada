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
4. Calcular anomalías respecto de una climatología y detectar episodios extremos de temperatura de superficie con Landsat 8 y 9 (calibrando los datos a °C y a reflectancia), con umbrales y con aprendizaje automático (Isolation Forest), reconociendo los límites que imponen las nubes y la revisita de 8 días.
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
| Publicaciones de 2026 en Zenodo que mencionan Biobío | 33 (de ellas, 8 artículos de revista); además 21 datasets y 65 registros de todos los tipos | Entre las publicaciones, "Biobío" da 29 y "Biobio" da 31: las tildes cambian el conteo (con `Biobío OR Biobio` salen los 33). Cifras medidas el 10 de octubre de 2026; crecen cada día |
| Artículos de 2026 de instituciones del Biobío (OpenAlex) | Entre 3.300 y 3.400 (3.373 el 10 de octubre de 2026 con las 10 instituciones que lista OpenAlex para la región) | Cuenta por afiliación de los autores, no por tema. Sube cada semana porque se siguen indexando |
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

* **Duplicados por reprocesamiento.** La misma escena puede aparecer dos veces con identificadores de procesamiento distintos. En el cuaderno se ve con datos reales: entre el 20 de enero y el 10 de febrero de 2022, 16 ítems se reducen a 14 porque la toma del 5 de febrero existe con baseline 04.00 (procesada en 2022) y con 05.10 (reprocesada en 2024); `preparar_items` deja la más nueva. Varios ítems de una misma fecha pero de distinto tile u órbita no son duplicados.
* **El desfase de reflectancia depende del catálogo y de la versión de procesamiento.** No existe un valor fijo (el famoso `-0.1` aplica a ciertas versiones y a ciertos catálogos). Los helpers del repositorio lo resuelven por imagen y los participantes lo ven una vez de forma explícita: con ventanas reales de la Bahía de Concepción, el DN del agua salta de unas decenas o centenas (baseline 03.00, 23-01-2022) a más de 1.100 (baseline 04.00, 31-01-2022), y un -0.1 fijo daría reflectancias negativas en las fechas antiguas. Así entienden por qué los cuadernos antiguos con un desfase fijo subestiman los índices.

El bloque cierra con un caso de enseñanza: la escena `S2B_MSIL2A_20260407T143739_R096_T18HXD_20260407T180947` declara `EPSG:32718` en el ítem, pero sus archivos no traen CRS y `odc.stac.load` falla con un `AssertionError` sin mensaje. Los participantes la detectan abriendo el encabezado del archivo con `rasterio` (barato) y comparando con el CRS del ítem.

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

Finalmente se extrae el valor de un píxel en un punto concreto. Es el primer paso hacia la serie del bloque siguiente. Dos detalles que conviene anticipar: entre 1 % y 8 % de los píxeles (según la banda) tiene reflectancia levemente negativa en superficies muy oscuras (bosque, sombras) y `convertir_a_reflectancia` los deja en `NaN`, por eso la imagen RGB tiene huecos blancos hasta que se pintan de negro con `fillna(0)`; y la escena usada es la del 10 de agosto de 2026 (0,04 % de nubes), en la que se ve bien la pluma del Biobío.

Este bloque usa solo Sentinel-2. No se incluye Sentinel-3 y tampoco se lo reemplaza por otro sensor (ver la nota técnica sobre Sentinel-3 más abajo).

#### Bloque 4. Construcción de la serie temporal

Cuaderno: `04_series_temporales.py`. Presupuesto: 90 minutos (15 de conceptos, 15 de puntos, 35 de búsqueda, carga, máscara y construcción de la serie, 10 de comparación de filtros, 15 de guardado, respaldo y gráfico). Reserva especial: en clase se carga un período corto de **tres meses** (junio a agosto de 2026) y la lectura de las ventanas tarda unos dos minutos, así que conviene que todos lancen esa celda a la vez y lean mientras tanto. La serie de 12 meses no se carga en vivo: viene del parquet de respaldo, que además sirve de continuación para el bloque 5.

Se explica cómo pasar de una imagen única a una serie temporal. Los conceptos clave son la revisita del satélite, el filtrado por nubosidad y la extracción de valores en puntos de interés.

QGIS es solo un apoyo. El cuaderno lleva un recuadro opcional que explica cómo abrir `puntos_concepcion.geojson`, editar o agregar puntos y exportar a CSV; quien no use QGIS parte del CSV ya preparado (seis puntos: bahía, bahía San Vicente, pluma del Biobío, golfo de Arauco, Laguna Grande y humedal Rocuant-Andalién). El cubo de datos se construye siempre en Colab.

El cuaderno hace visible el patrón nativo: una búsqueda para toda la zona (`pystac_client`, con `preparar_items`); la carga de una ventana, escrita con `odc.stac.load` perezoso para ver su tamaño (0,8 MB por punto frente a unos 13 GB que pesaría la caja completa de 36 x 48 km con las mismas bandas y fechas); `cargar_ventanas` para los seis puntos; y luego la máscara SCL, `convertir_a_reflectancia` (con el offset de cada fecha) y los índices a mano para un punto, antes de `construir_series`. Medido con los puntos del taller (tres meses, 81 escenas, 39 fechas por punto, 5 MB en total): la búsqueda y el armado de los cubos tardan unos 6 s, la lectura de las ventanas unos 105 s y todo el cuaderno unos 2 minutos. La prueba anterior (otra zona, 12 meses, 78 s y 313 MB frente a 306 s de la caja completa) no se repite aquí: la caja completa no se lee, solo se calcula su tamaño.

**Filtro de nubes de la escena frente a máscara SCL.** Son dos cosas distintas. El filtro de la escena (`eo:cloud_cover`) descarta fechas completas, pero no dice nada sobre un punto en particular. La máscara de clasificación de escena (SCL) trabaja por píxel. Se carga una sola vez con hasta 100 % de nubes y se simulan cuatro límites, con y sin máscara (`construir_series(..., excluir=[])` es "sin máscara"). Resultado medido (junio a agosto de 2026, suma de los seis puntos; 234 fechas con cobertura con límite 100 %):

| Límite de nubes de escena | Fechas | Válidas sin máscara SCL | Válidas con máscara SCL | De las "válidas sin máscara", contaminadas |
|---|---|---|---|---|
| 100 % | 234 | 227 | 51 | 176 |
| 60 % | 73 | 71 | 49 | 22 |
| 30 % | 47 | 45 | 41 | 4 |
| 10 % | 24 | 22 | 22 | 0 |

(Fechas contaminadas: parecen válidas sin máscara, pero más de la mitad de su ventana está tapada según SCL.) La máscara quita en promedio 77 % de los píxeles con límite 100 %, 30 % con 60 %, 9 % con 30 % y 0 % con 10 %. Es invierno, la época más nublada; en verano la diferencia se achica. Estos números reemplazan a los de la prueba anterior (Maipo: con menos de 20 % de nubes la máscara no cambiaba nada; sin filtro enmascaraba cerca de 40 %). Conviene mencionar que la lista de clases SCL excluidas por defecto está pensada para vegetación y que cerca del agua conviene revisar qué clases se están descartando: el cuaderno muestra la distribución de clases por punto (en las ventanas de agua casi todo es agua o nube; el humedal tiene además suelo desnudo y sombras de nube).

Para terminar, la serie se guarda en un archivo parquet con sus metadatos (`guardar_series`). Es el momento FAIR del día: el archivo viaja a los bloques siguientes. Después se carga el parquet de respaldo de 12 meses y se comprueba que las fechas en común coinciden exactamente con la serie recién construida (diferencia máxima 0). Hay tres filas de más en el respaldo, todas del golfo de Arauco: ese punto cae donde se traslapan dos escenas, y el respaldo se buscó con filtro de escena de 60 %, así que entró solo la escena despejada de cada par; es un buen ejemplo de que el filtro de escena depende de dónde cae el punto. El bloque termina con las cuatro variables (NDWI, NDCI, B04 y NDVI) por punto.

Entregable del día: un cuaderno con la serie temporal de Sentinel-2 para los puntos definidos, guardada en parquet.

## Día 2. Análisis, detección de eventos extremos y flujo reproducible

### 10:00 a 12:30

#### Bloque 5. Análisis exploratorio de la serie temporal

Cuaderno: `05_exploracion_series.py`. Presupuesto: 85 minutos (10 de carga del parquet, 25 de estadísticas y suavizado, 15 de tendencia, 10 de patrones estacionales, 25 de anomalías, Landsat e interpretación).

El bloque parte del parquet de 12 meses, sin recalcular nada. Se revisan estadísticos descriptivos (media, mediana, percentiles) y la media móvil con `rolling_media`, con ventana en días (`"30D"`) porque las fechas son irregulares. Se calcula la tendencia lineal (`tendencia_lineal`) con una advertencia explícita: con un solo año la pendiente mezcla estación y cambio, y el cuaderno lo muestra ajustando la recta a todo el año y a dos tramos de seis meses (la pendiente de NDWI en la pluma del Biobío cambia de signo).

**Anomalías.** Con un año de datos no hay climatología mensual que valga: cada mes tiene entre 2 y 11 observaciones y el valor extremo forma parte de su propia referencia (un z con *n* valores no puede pasar de (*n* - 1) / raíz de *n*). Por eso el cuaderno usa `anomalias_zscore(referencia="total")`, y muestra que con `"mes"` el golfo de Arauco llega a z = 2,5 donde con `"total"` llega a 8,6. La climatología mensual y las anomalías mejor hechas se muestran con la serie de Landsat 8 y 9 del centro de la bahía (`cargar_landsat()`, base 2013-2024, 9 a 20 observaciones por mes), como puente al bloque 7: en mayo de 2026 la anomalía mensual media es de -0,9 y en septiembre de +0,9 (los meses sin ninguna observación sin nubes se ven como vacíos, no como cero).

Cierra con tres lecturas guiadas: turbidez de la pluma del Biobío (B04 alta en noviembre-diciembre y agosto-septiembre; sin datos de lluvia o caudal es solo una hipótesis), clorofila en las bahías (NDCI cerca de cero en primavera y verano; en invierno hay pocas fechas y agua muy oscura, el índice es ruidoso) y NDVI del humedal (rango corto, 0,5 a 0,6). Se insiste en que son hipótesis y en qué dato faltaría.

#### Bloque 6. Visualización interactiva e interpretación

Cuaderno: `06_visualizacion_interactiva.py`. Presupuesto: 50 minutos (15 de buenas prácticas, 20 de `hvplot`, 10 de `folium`, 5 de interpretación).

Se presentan buenas prácticas para gráficos de series temporales ambientales con un ejemplo mejorable y su versión corregida sobre los mismos datos. Se introducen `hvplot` (primero `serie_interactiva`, después la llamada `tabla.hvplot.line(...)` con cada parámetro a la vista, la composición con `*`, `subplots` y `groupby`, y el guardado como HTML independiente) y `folium` (`mapa_puntos` y luego un mapa armado a mano cuyos popups resumen cada serie con una tabla y un gráfico pequeño incrustado).

`hvplot` no viene en Colab: el cuaderno lo instala con una celda que lee `requirements.txt` (puede tardar un minuto; conviene que todos la lancen al empezar). Los widgets de `hvplot` (el menú de `groupby`, por ejemplo) pueden comportarse distinto en Colab que en un entorno local, por lo que se prueban el día anterior (ver las notas técnicas). El cuaderno termina con un ejercicio de interpretación por escrito, con pistas plegadas, que sirve de base para las presentaciones del día 3.

### 14:30 a 17:30

#### Bloque 7. Detección de eventos extremos: umbrales clásicos y aprendizaje automático

Cuaderno: `07_eventos_extremos.py`. Presupuesto: 100 minutos (20 de calibración y datos, 15 de serie irregular y control de calidad, 15 de climatología por ventana, 15 de episodios, 10 de agua, tierra y reflectancia, 15 de Isolation Forest, 10 de mapa de frecuencia). El cuaderno tarda unos 60 segundos en ejecutarse completo; las partes más lentas son la lectura de una escena de Landsat (sección 1) y la climatología de la malla (unos 30 segundos).

Este bloque introduce la detección de eventos extremos en series temporales ambientales con **Landsat 8 y 9**. Se llama "umbrales clásicos y aprendizaje automático" porque los percentiles son estadística y no IA, y porque se compara con un método de aprendizaje automático. La IA queda reservada para el bloque 1.

**Datos.** Se usa la temperatura de superficie (banda térmica `lwir11`) y la reflectancia de Landsat 8 y 9 Collection 2 Level-2, obtenidas de Planetary Computer. En el repositorio hay una copia preparada en `datos/` (series por punto de 2013 a 2026 y una malla de 500 m alrededor de la bahía de Concepción y el golfo de Arauco) para que Colab no dependa de descargas largas. El script `datos/preparar_landsat.py` repite todos los pasos y tarda unos 45 minutos.

**Calibración.** Lo primero que ven las personas es que Landsat no entrega grados ni reflectancia, sino números enteros: temperatura (K) = DN x 0,00341802 + 149 y reflectancia = DN x 0,0000275 - 0,2. Calibran a mano una escena despejada del 13 de febrero de 2024 en la bahía (unos 15,3 °C y reflectancia roja de 0,001), ven qué pasa si se saltan un paso (olvidar el offset de -0,2 multiplica la reflectancia por 200), comprueban que las funciones de `utils.raster` hacen lo mismo y que recalcular esa fila con `construir_series_landsat` da el mismo valor que la tabla.

**Qué no se puede hacer.** Hobday exige una serie diaria sin huecos y 30 años de base. Landsat pasa cada 8 días (16 por satélite), las nubes dejan unas 10 a 25 observaciones válidas por año (230 de 388 pasos en la bahía) y solo hay 13 años. **No se habla de olas de calor marinas**: usar ese nombre sería prometer más de lo que permiten los datos. Se adapta la idea: climatología con una ventana de 31 días (15 a cada lado) en el periodo 2013-2024, p90 y p10, y un **episodio** son al menos 2 observaciones válidas consecutivas anómalas, sin cruzar huecos de más de 24 días. La duración es solo una cota mínima. `utils/extremos.py` conserva el método de Hobday para series diarias y se prueba contra `xmhw` con una serie sintética.

**Control de calidad con la reflectancia.** Es el hallazgo del bloque. La máscara `qa_pixel` deja pasar nubes delgadas y neblina, que se miden como agua muy fría (anomalías de hasta -16 °C). La reflectancia calibrada las delata (reflectancia roja mayor que 0,05 o SWIR1 mayor que 0,02 en el agua, es decir, demasiado brillo): `controlar_calidad_agua` descarta 50 observaciones de los cinco puntos de agua (9 en la bahía). Se usa el brillo y no el NDWI porque sobre agua oscura la reflectancia NIR es a veces ligeramente negativa (se convierte en NaN) y el NDWI no se puede calcular en 1 de cada 8 observaciones; la regla también puede descartar una floración real de algas. Es una regla simple y declarada: quedan observaciones dudosas que ninguna regla resuelve, y se discuten.

**Resultados de referencia.** Con el control de calidad, en los cinco puntos de agua hay **23 episodios** (13 cálidos y 10 fríos): la bahía no tiene ninguno cálido y tiene 3 fríos (2014-03-05, 2018-09-24 y 2026-05-09), el golfo de Arauco tiene 6 cálidos (el de 2016-01-22 con +3,8 °C) y hay uno en curso desde 2026-09-06 en el golfo (3 observaciones, +1,5 °C). 10 de los 23 coinciden con otro punto: del 1 al 17 de agosto de 2016 tres puntos tienen un episodio cálido a la vez (puede ser clima regional o una nube que cubrió la zona; Landsat no lo distingue). Isolation Forest con la temperatura marca 22 observaciones de la bahía y con tres variables (temperatura, reflectancia roja y NDWI) 18; su Jaccard con el umbral es de 0,37 y 0,21. La correlación entre la anomalía de temperatura y la de la reflectancia roja es negativa (de -0,2 a -0,4): puede ser residuo de nube o agua más turbia y fría, y los datos no permiten elegir. **El muestreo no es estacionario**: antes de 2022 los intervalos entre observaciones válidas son casi siempre de 16 días y desde 2022 (Landsat 9) de 8, así que la brecha máxima de 24 días significa cosas distintas antes y después, y la climatología 2013-2024 pesa más a 2022-2024. No hay que leer tendencias en el conteo de episodios. Si los datos de Planetary Computer se actualizan, estas cifras cambian: corre el cuaderno completo antes del taller y anota los valores definitivos.

**Agua frente a tierra.** El humedal tiene una mediana de 24 °C frente a 13 a 19 °C del agua y un rango p5 a p95 de 25 °C frente a 5 a 13 °C. Por eso cada punto usa su propia climatología. Se confirma con NDWI y NDVI calibrados.

**Mapa de frecuencia.** Con la malla de 500 m se calcula la climatología por píxel (base 2013-2024) y la fracción de observaciones de 2025 y 2026 sobre el p90 y bajo el p10. En 2025-2026 la mediana de la malla es de 0,15 sobre el p90 y 0,13 bajo el p10 (esperado 0,10 en ambos): hubo más observaciones extremas de los dos signos, con las fracciones cálidas más altas en el oeste (mar abierto). Un detalle para el debate: la fracción cálida de 2013-2018 es de 0,17 y la de 2019-2024 de 0,08, así que la climatología base no es homogénea. La malla no tiene el control de calidad con la reflectancia: el cuaderno descarta lo que está bajo 3 °C (un 4,6 %, nubes sin marcar) y hay zonas de tierra con muy pocas observaciones (la máscara de nubes parece confundir superficies brillantes con nubes). Con unas 45 observaciones por píxel (de 40 a 60) el ruido de la fracción es de unos 0,05: se mira el patrón espacial, no píxeles sueltos.

**`marEx`.** No se usa en el bloque: no tiene modo de punto, no aplica la regla de los 5 días, no detecta olas de frío y declara soporte solo hasta Python 3.12, mientras que Colab usa Python 3.13.

#### Bloque 8. Git, reproducibilidad y cierre del flujo de trabajo

Cuaderno: `08_git_reproducibilidad.py`. Presupuesto: 65 minutos (10 de verificación del token al inicio, 10 de comandos, 25 de refactorización, 20 de commits, push y verificación).

Este bloque final integra todo lo aprendido en un solo repositorio versionado con Git. El objetivo es que cada participante termine con un proyecto reproducible que pueda compartir.

Se explican los comandos esenciales de Git (`init`, `add`, `commit`, `push`, `pull`, `branch`) y su integración con Google Colab. La trazabilidad es el concepto central: cada cambio queda registrado y se puede revertir o auditar.

**Autenticación.** El token se crea como *fine-grained PAT*, limitado a un solo repositorio, con permiso de contenido de lectura y escritura y una expiración corta. Se guarda en Colab Secrets y se lee con `google.colab.userdata`. Nunca se escribe dentro de una celda, no se imprime y no queda en las salidas ni en `.git/config`: el cuaderno usa `GIT_ASKPASS` con un script que no contiene el token y que lo lee de una variable de entorno válida solo para el proceso del `push` (más `credential.helper=` vacío), con la URL del remoto limpia, y después comprueba `.git/config`. Muestra antes, con un valor falso, qué pasa si el token va en la URL. Dedica los primeros diez minutos del bloque a verificar que todos pueden hacer `git push`.

**Plan de respaldo.** Quien no logre que el token funcione en esos diez minutos usa la opción de Colab "Guardar una copia en GitHub", que no requiere token, y sigue con el resto del bloque.

El ejercicio consiste en refactorizar el código de los cuadernos anteriores en funciones reutilizables, inicializar un repositorio local desde Colab, hacer al menos tres commits descriptivos y subir el repositorio a GitHub. Se verifica que el cuaderno se ejecuta de principio a fin sin errores: el cuaderno clona el repositorio de práctica en una carpeta limpia, ejecuta el análisis ahí y compara las huellas (hash) de los resultados. Las partes locales (commits, ramas, push y pull contra un remoto local) funcionan sin token y sin internet; solo la subida a GitHub necesita el Secreto `GITHUB_TOKEN`.

Entregable del día: un repositorio en GitHub con el cuaderno completo, commits documentados, la serie temporal con sus anomalías y los eventos extremos detectados.

## Día 3. Presentación de resultados y cierre

### 10:00 a 12:30

#### Bloque 9. Presentación de proyectos y publicación FAIR

Cuaderno: `09_publicacion_zenodo.py`. Presupuesto: 10 minutos de repaso y pulido, 65 minutos de presentaciones, 40 minutos de publicación y 25 minutos de cierre.

Cada participante presenta su repositorio y los hallazgos principales en **6 minutos**. Con hasta 10 personas son 60 minutos más los cambios, por lo que cabe en 65. Con más personas habría que bajar a 5 minutos. Se discuten las dificultades encontradas, las decisiones tomadas y lo aprendido sobre el fenómeno ambiental analizado. El instructor y los compañeros aportan retroalimentación.

Luego se explica cómo publicar un cuaderno y los datos derivados de forma FAIR. La práctica se hace en **Zenodo Sandbox** (sandbox.zenodo.org), no en Zenodo real: un depósito real crea un DOI público y permanente para material de práctica. En el sandbox los DOI tienen el prefijo 10.5072 y no se resuelven fuera de él. El depósito se hace por la API REST con el token de Zenodo Sandbox de cada participante, guardado en Colab Secrets. Se completan los metadatos mínimos: título, descripción, palabras clave en español e inglés (con y sin tilde), licencia, autores con ORCID, lugar e identificadores relacionados con la fuente. El cuaderno valida los metadatos antes de enviarlos y repite las preguntas del bloque 1 sobre el registro. El paso de publicar está protegido por un interruptor (`CONFIRMO_PUBLICAR`). La recuperación por DOI usa el resolvedor del propio sandbox y la búsqueda de su API, y compara la suma de verificación md5 de lo descargado. Si el expositor quiere mostrar un depósito real, lo hace él, como demostración y con su propia cuenta.

El ejercicio consiste en subir el cuaderno y los datos derivados al sandbox, completar los metadatos y verificar que el depósito es recuperable por su DOI.

#### Recapitulación y cierre

Resumen de los conceptos clave: principios FAIR, consulta de repositorios con IA y por qué los metadatos importan, catálogos STAC, Sentinel-2 y su desfase de reflectancia, series temporales, anomalías y episodios extremos (calibración de Landsat, umbrales, Isolation Forest, nubes y huecos de datos) y Git para reproducibilidad. Se comparte una lista de recursos para seguir aprendiendo y se aplica una encuesta final de satisfacción.

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

No se reemplazó por otro sensor. El bloque 3 es solo Sentinel-2 y la temperatura de superficie aparece en el bloque 7 con la banda térmica de Landsat 8 y 9.

### Bloque 4. Planetary Computer y el desfase de Sentinel-2

* **Caducidad de las firmas.** Los enlaces a los datos de Planetary Computer van firmados y caducan al cabo de un tiempo. Si una carga falla después de haber dejado el cuaderno un rato parado, el síntoma es un error de acceso (HTTP 403) en una celda que antes funcionaba. Se arregla volviendo a ejecutar la búsqueda y la preparación de los ítems para obtener enlaces nuevos.
* **El desfase (offset) de Sentinel-2 se maneja por imagen.** No existe un valor fijo. Depende del catálogo y de la versión de procesamiento (*baseline*), y los helpers lo resuelven imagen por imagen con `agregar_offset`. Los cuadernos antiguos del repositorio, con un desfase fijo de 0,0, subestiman los índices y no deben copiarse. Esto se prueba mucho porque es fácil equivocarse sin notarlo.
* **Duplicados.** Se agrupa por día solar para quedarse con una escena por fecha.

### Bloque 7. Nubes, escenas dañadas y huecos de Landsat

* **Escenas dañadas.** Dos escenas de Planetary Computer (`LC08_L2SP_001086_20140305_02_T1` y `LC08_L2SP_001086_20221106_02_T1`) tienen un archivo que el servidor devuelve como una página de error de 14 kB en vez de un GeoTIFF. La carga falla con "not recognized as being in a supported file format". `datos/preparar_landsat.py` las detecta con `utils.stac.items_legibles` y las excluye. En el cuaderno no se leen: solo se lee una escena del 13 de febrero de 2024.
* **Nubes que la máscara no ve.** Es esperable que aparezcan observaciones de agua "muy fría" con reflectancia roja alta: son nubes. El control de calidad del cuaderno las descarta. Si alguien propone una regla más dura, conviene discutir qué eventos reales se perderían.
* **Los números cambian.** Cada vez que se actualiza la colección (y desde 2022 con Landsat 9) hay más fechas. Regenerar los datos con `python preparar_landsat.py` (unos 45 minutos) cambia los números de referencia del bloque.

## Posibles problemas y soluciones

**Gemini no responde o da un error de clave.** Verifica que el secreto se llama exactamente `GOOGLE_API_KEY`, que el acceso desde el cuaderno está activado y que la clave se creó en una cuenta personal. Si la cuenta es institucional, es probable que AI Studio esté bloqueado: usar una cuenta personal.

**Un servidor MCP no responde.** Reintenta una vez. Los servidores de la comunidad y de terceros pueden caerse o llegar a su cuota. Para Dataverse, cambia al servidor local de respaldo. Para Zenodo y OpenAlex, continúa con la demostración del expositor y retoma más tarde.

**La respuesta de la IA no coincide con la de la web.** Es parte de la lección. Comparar el total que informa el servidor, probar variantes con y sin tilde y en inglés, y distinguir lugar de afiliación.

**Una carga de Planetary Computer falla o da un error de acceso.** Probablemente caducaron las firmas: vuelve a ejecutar la búsqueda y la preparación de los ítems. Si el problema persiste o la red está lenta (muchos participantes a la vez), usa el parquet de reserva del repositorio y continúa con el análisis. Conviene escalonar las celdas pesadas del bloque 4.

**La carga de 12 meses tarda demasiado.** Comprueba que se están cargando ventanas alrededor de los puntos y no toda la caja que los contiene.

**Los índices salen subestimados.** Revisa que se aplicó el desfase por imagen (`agregar_offset`) antes de convertir a reflectancia.

**Los widgets de `hvplot` no aparecen en Colab.** Vuelve a ejecutar la extensión de `hvplot` en la misma celda o instala `hvplot` en la sesión. Si no funciona, usa los gráficos sin widgets y deja el mapa de `folium` como demostración.

**Un episodio de calor del bloque 7 dura meses.** Un episodio no debe cruzar un hueco de más de 24 días entre observaciones (`brecha_max_dias`). Revisa que esa regla se aplica y recuerda que la duración es una cota mínima, no una duración real.

**Los participantes no logran hacer `git push`.** Verifica que crearon un token de grano fino, limitado a un solo repositorio, con permiso de contenido de lectura y escritura, que lo guardaron en Colab Secrets con el nombre que usa el cuaderno y que el acceso desde el cuaderno está activado. Dedica diez minutos al inicio del bloque 8 a esta verificación. Si no resulta, usa "Guardar una copia en GitHub" desde el menú de Colab.

**Un participante pegó un token o una clave en una celda.** Bórralo de la celda y de las salidas, revócalo en GitHub o en AI Studio y crea uno nuevo.

**El depósito en Zenodo Sandbox falla.** Comprueba que el token es del sandbox y no de zenodo.org, y que tiene permisos de escritura. Si no alcanza el tiempo, el expositor completa el depósito en pantalla como demostración guiada.

**Alguien no tiene QGIS.** No hace falta. QGIS es opcional y el CSV de puntos ya está preparado.
