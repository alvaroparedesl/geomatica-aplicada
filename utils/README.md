# utils: funciones de apoyo del curso

Funciones reutilizables para los notebooks de Geomática Aplicada. Los notebooks
las cargan con `from utils.<módulo> import ...`, así el código repetitivo
(armar parámetros, convertir a reflectancia, enmascarar nubes, calcular índices,
gráficos) queda en un solo lugar y el notebook se concentra en lo que enseña.

Los helpers de `utils/stac.py` **no reemplazan** las llamadas nativas de
`pystac_client` y `odc.stac`: solo las **preparan**. La búsqueda y la carga las
escribes tú; los helpers arman sus argumentos y ordenan los datos antes y después.

## Celda "Preparar entorno" (primera celda de código de cada notebook)

Funciona igual en Google Colab y en un computador local. En Colab clona el
repositorio (solo la primera vez de la sesión) en `/content/geomatica-aplicada`;
en local sube desde la carpeta actual hasta encontrar `utils/`. Además deja
definida la variable `RAIZ` (carpeta del repositorio), útil para llegar a
cualquier archivo del repositorio, por ejemplo `RAIZ / "utils"`.

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

La instalación de paquetes que Colab no trae (`rioxarray`, `odc-stac`,
`planetary-computer`, `pystac-client`, ...) sigue en la celda `# %pip install ...`
de cada notebook; la lista completa está en `requirements.txt` (raíz del repo).

## Ejemplo: de la búsqueda al NDVI

```python
import odc.stac
from utils.stac import (conectar_catalogo, parametros_busqueda, preparar_items,
                        parametros_carga, renombrar_bandas, agregar_offset)
from utils.raster import preparar_reflectancia, calcular_indices, nombres_bandas
from utils.viz import graficar_series_puntos

bbox = [-70.8, -33.7, -70.6, -33.6]

# Búsqueda: llamadas nativas de pystac_client; el helper arma el diccionario de parámetros
catalogo = conectar_catalogo()                                   # pystac_client.Client
parametros = parametros_busqueda("sentinel-2", bbox, ("2023-06-01", "2024-05-31"), nubes_max=30, catalogo=catalogo)
busqueda = catalogo.search(**parametros)
items = busqueda.item_collection()
items = preparar_items(items)                                    # sin duplicados ni catálogos mezclados

# Carga: llamada nativa de odc.stac; el helper arma sus argumentos
bandas = nombres_bandas("sentinel-2", ["red", "green", "blue", "nir", "swir1"], con_calidad=True)
parametros_ds = parametros_carga("sentinel-2", bandas, resolucion=30, crs="EPSG:32719", catalogo=catalogo, bbox=bbox)
datos = odc.stac.load(items, **parametros_ds)
datos = renombrar_bandas(datos, catalogo)                        # B04, B08, SCL en todos los catálogos
datos = agregar_offset(datos, items)                             # offset de reflectancia de cada fecha

refl = preparar_reflectancia(datos, "sentinel-2")                # enmascara nubes y convierte a reflectancia
indices = calcular_indices(refl, "sentinel-2")                   # Dataset con NDVI, EVI y NDWI
graficar_series_puntos(indices, [[342122, 6279477], [339600, 6276395]], ("NDVI", "EVI"))
```

## Módulos

### `utils/ia.py` y `utils/mcp_dataverse.py` (servidores MCP desde Colab)
Usan servidores MCP (Zenodo, Dataverse, OpenAlex, ...) sin programar MCP: el cuaderno
es el cliente. Necesitan `pip install mcp` (y `google-genai`, que Colab ya trae, solo
para `preguntar`).

| Función | Qué hace |
|---|---|
| `herramientas(servidor)` | Lista lo que ofrece un servidor (sin clave ni IA) |
| `usar(servidor, herramienta, **argumentos)` | Llama tú mismo a una herramienta y devuelve su texto (sin IA) |
| `preguntar(pregunta, servidores, modelo, max_pasos, devolver)` | Pregunta en lenguaje natural: Gemini decide qué herramientas llamar y cada paso se imprime (`[paso 1] El modelo llama a zenodo.zenodo_search_records(...)`), con un prompt de sistema en español que exige usar los totales de la herramienta y decir "no consta" cuando falta un campo |
| `SERVIDORES` | Catálogo de alias: `zenodo`, `dataverse`, `dataverse_local` (el servidor propio `utils/mcp_dataverse.py`, que se lanza como subproceso), `openalex`, `pangaea`, `nasa`, `gbif` |

La clave de Gemini es **tuya**: se lee de los Secretos de Colab (`GOOGLE_API_KEY`, con
"Acceso desde el cuaderno" activado) y, fuera de Colab, de la variable de entorno
`GOOGLE_API_KEY`. Nunca se escribe en el cuaderno. Sin clave, `preguntar` avisa cómo
guardarla y `herramientas`/`usar` siguen funcionando.

### `utils/vector.py`
| Función | Qué hace |
|---|---|
| `descargar_natural_earth(escala, tema, categoria)` | Descarga (y guarda en caché) una capa de Natural Earth como GeoDataFrame |
| `calcular_area(gdf, unidad)` | Área en m², ha o km² usando una proyección de igual área |

(Para mapas interactivos de un GeoDataFrame usa directamente `gdf.explore()`.)

### `utils/stac.py` (Planetary Computer y Earth Search)
| Función | Qué hace |
|---|---|
| `CATALOGO_SENTINEL2` | Catálogo de Sentinel-2 por defecto (`"planetary"`); ver "Catálogos" más abajo |
| `conectar_catalogo(catalogo)` | Devuelve el cliente nativo `pystac_client.Client` (en Planetary Computer, con firma automática) |
| `parametros_busqueda(sensor, bbox, fechas, nubes_max, catalogo)` | Diccionario para `catalogo.search(**parametros)`: colección, bbox, fechas y filtro `query`. No busca nada |
| `preparar_items(items)` | Quita versiones duplicadas de una toma Sentinel-2, rechaza catálogos mezclados o desactivados y ordena por fecha |
| `resumir_imagenes(items)` | Tabla con fecha, % de nubes e id (+ baseline y offset en Sentinel-2) |
| `parametros_carga(sensor, bandas, resolucion, crs, catalogo, bbox, chunks, **kwargs)` | Diccionario para `odc.stac.load(items, **parametros)`: nombres de banda del catálogo, `crs`, `resolution`, `chunks`, `patch_url` (firma de Planetary Computer) y `groupby`. Los `**kwargs` (por ejemplo `fail_on_error=False`) se agregan tal cual; en Sentinel-2 no se acepta `groupby`. No carga nada |
| `agrupar_por_dia_y_offset(item)` | El `groupby` de Sentinel-2: agrupa por (día solar, offset), así nunca se mezclan offsets distintos en una fecha |
| `renombrar_bandas(datos, catalogo)` | Deja las bandas de Earth Search como B02, B04, B08, SCL (en Planetary Computer no cambia nada) |
| `agregar_offset(datos, items)` | Agrega la coordenada `offset_reflectancia` (un valor por fecha) a lo que cargó `odc.stac.load` |
| `cargar_dem(bbox, epsg, resolucion)` | DEM Copernicus 30 m |
| `offset_sentinel2(item)` | Offset de reflectancia correcto de un ítem Sentinel-2 (0 o -0.1) |
| `quitar_duplicados(items)` | Deja una sola versión (la más nueva) de cada toma Sentinel-2 |
| `escala_y_offset(item, banda)` | Factor y offset para pasar de DN a reflectancia |
| `items_legibles(items, bandas)` | Separa los ítems con algún archivo dañado (Planetary Computer devuelve una página de error en vez del GeoTIFF) de los legibles |

### `utils/raster.py`
| Función | Qué hace |
|---|---|
| `SENSORES`, `obtener_configuracion_sensor(sensor)` | Bandas, factores y colección de "landsat" y "sentinel-2" |
| `nombres_bandas(sensor, nombres, con_calidad)` | "nir" → `nir08` (Landsat) o `B08` (Sentinel-2) |
| `convertir_a_reflectancia(imagen, sensor, offset)` | DN * factor + offset (offset por fecha en Sentinel-2); DN 0 y valores fuera de 0 a 1 pasan a NaN |
| `convertir_a_temperatura(imagen, celsius)` | Banda térmica `lwir11` de Landsat 8/9: K = DN * 0.00341802 + 149, luego °C; DN 0 y valores fuera de 230 a 340 K pasan a NaN |
| `mascara_calidad_landsat(qa_pixel, excluir)` | Máscara True=válido desde QA_PIXEL (`BITS_LANDSAT`) |
| `mascara_saturacion_landsat(qa_radsat)` | Máscara True=sin saturación (bandas ópticas 1 a 7 y terreno que tapa el píxel) desde QA_RADSAT |
| `mascara_calidad_sentinel2(scl, excluir)` | Máscara True=válido desde SCL (`CLASES_SCL`) |
| `enmascarar_nubes(imagen, sensor, excluir)` | Pone NaN en nubes/sombras/relleno |
| `preparar_reflectancia(imagen, sensor, enmascarar)` | Atajo: enmascara + convierte |
| `calcular_ndvi / calcular_evi / calcular_ndwi(imagen, sensor)` | Índices espectrales (DataArray) |
| `calcular_ndwi_agua(imagen, sensor)` | NDWI de McFeeters (verde y NIR), para agua abierta (no es el `NDWI` de arriba) |
| `calcular_ndci(imagen, sensor)` | NDCI (borde rojo y rojo), clorofila en agua (solo Sentinel-2) |
| `calcular_indices(imagen, sensor, indices)` | Varios índices en un Dataset ("NDVI", "EVI", "NDWI", "NDWI_AGUA", "NDCI") |
| `lonlat_a_crs(lon, lat, crs)` | Transforma coordenadas geográficas a otro CRS |
| `extraer_series_puntos(datos, puntos, columna_id, radio, estadistico, crs)` | Serie temporal en formato largo para puntos con x, y o lon, lat: el píxel más cercano o, con `radio` (m), la mediana (o media, min, max) de la ventana ignorando NaN |
| `recortar_region(imagen, limites, crs)` | Recorta con un rectángulo lon/lat |
| `calcular_estadisticas(dataset, bandas)` | Tabla de estadísticos descriptivos por banda |

### `utils/viz.py`
| Función | Qué hace |
|---|---|
| `crear_imagen_rgb(dataset, bandas_rgb, factor_brillo)` | Arreglo (alto, ancho, 3) para color verdadero o falso color |
| `mostrar_imagen(datos, titulo, cmap, vmin, vmax, barra)` | Una imagen sin ejes |
| `mostrar_paneles(paneles, ncols, titulo)` | Varias imágenes en cuadrícula |
| `mostrar_mascaras(mascaras, ncols)` | Máscaras booleanas con conteo de píxeles |
| `graficar_histogramas(dataset, bandas, ...)` | Histogramas por banda (matplotlib) |
| `histograma_interactivo(dataset, bandas, rango, bins)` | Histogramas superpuestos (plotly) |
| `graficar_tabla_series(tabla, variables, columna_id, fecha, puntos, suavizado, ...)` | Tabla de series en formato ancho (`cargar_series`): un panel por variable, una línea con marcadores por punto, leyenda con `nombre`, filtro opcional de puntos y media móvil opcional (`suavizado="30D"`). Devuelve `(fig, ejes)` (matplotlib) |
| `graficar_series_temporales(serie, ...)` | Series temporales por punto o grupo (plotly) |
| `graficar_series_puntos(datos, coordenadas, variables)` | Atajo: series temporales desde una lista de [x, y] |
| `mostrar_imagen_interactiva(datos, modo, indice_tiempo)` | RGB / falso color / NDVI / EVI interactivo |
| `explorador_imagenes(datos)` | Menús para elegir fecha y tipo de imagen |
| `graficar_capas(capas, titulo)` | Varias capas vectoriales superpuestas |
| `mapa_bbox(bbox, zoom)` | Mapa folium con el área de estudio |
| `mapa_puntos(puntos, zoom, radio_m, ancho, alto, popup, ancho_popup)` | Mapa folium con los puntos de muestreo, una capa y un color por tipo (folium se importa al usarla). `popup` personaliza el cuadro emergente: una lista de columnas (texto escapado), una función `fila -> html` o un diccionario `{id: html}`; sin `popup` muestra nombre, id, tipo y coordenadas |
| `serie_interactiva(tabla, variable, ...)` | Series por punto con hvplot/Bokeh (zoom, valores al pasar el mouse; hvplot se importa al usarla) |
| `SelectorRegion(imagen_rgb, bbox)` | Mapa para dibujar un rectángulo (ipyleaflet) |

Las funciones que dibujan aceptan `mostrar=False` para devolver la figura en
vez de mostrarla.

### `utils/datos_taller.py` (datos preparados del taller)
| Función | Qué hace |
|---|---|
| `abrir_dato(nombre)` | Ruta local de un archivo de `talleres/fair-series-temporales/datos/`: la del repositorio si existe y, si no, una copia descargada de GitHub (`URL_DATOS`) a una carpeta temporal |

### `utils/series.py` (series temporales por punto)
Del cubo Sentinel-2 a la tabla de series y de la tabla al análisis. Los imports
pesados (odc.stac, geopandas, pyarrow, scipy) están dentro de las funciones.

| Función | Qué hace |
|---|---|
| `cargar_puntos_csv(origen)` | Lee el CSV de puntos (`id, nombre, lat, lon, tipo`), valida columnas, `id` únicos y rangos |
| `puntos_a_geodataframe(puntos)` | GeoDataFrame EPSG:4326 (para `.explore()` o guardar GeoJSON) |
| `bbox_punto(lon, lat, radio_m)`, `bbox_puntos(puntos, radio_m)` | Rectángulo en grados alrededor de un punto, o que cubre todos (para la búsqueda STAC) |
| `items_de_punto(items, bbox)` | Los ítems cuya huella toca la ventana de un punto |
| `crs_utm(lon, lat)` | CRS UTM de la zona de un punto (Concepción: EPSG:32718) |
| `agregar_nubes_escena(datos, items)` | Coordenada `nubes_escena` (% de nubes de la escena por fecha), como `agregar_offset` |
| `cargar_ventanas(items, puntos, bandas, radio_m, resolucion, crs, catalogo, **kwargs)` | Para cada punto: `bbox_punto`, `parametros_carga`, `odc.stac.load`, `renombrar_bandas`, `agregar_offset`, `agregar_nubes_escena`; devuelve `{id: Dataset}` leído en paralelo. Los pasos están en el docstring del módulo para que un cuaderno los escriba a la vista |
| `construir_series(ventanas, puntos, radio_m, estadistico, fraccion_minima)` | Máscara SCL + reflectancia con el offset de cada fecha + NDWI, NDCI, B04 y NDVI + mediana de la ventana. Tabla ancha `COLUMNAS_SERIES` |
| `resumen_validez(tabla, variable)` | Fechas con cobertura y fechas válidas por punto |
| `guardar_series(tabla, ruta, metadatos)`, `cargar_series(origen)`, `leer_metadatos_series(origen)` | Parquet con metadatos dentro del archivo |
| `serie_de_punto(tabla, punto, variable)` | La serie de un punto como `pandas.Series` |
| `rolling_media(serie, ventana, minimo, centrada)` | Media móvil (`"30D"` sirve con fechas irregulares) |
| `tendencia_lineal(serie)` | `Tendencia`: pendiente por año, intercepto, r2, p, n y la recta ajustada |
| `climatologia_mensual(serie, base)` | Media, desviación y n de cada mes |
| `anomalias_zscore(serie, referencia, base)` | z-score respecto del mismo mes (`"mes"`) o de toda la serie (`"total"`) |

La tabla de series tiene una fila por punto y fecha, con `id, nombre, tipo, lat,
lon, fecha, ndwi, ndci, b04, ndvi, nubes_escena, pixeles_validos, n_pixeles,
offset_reflectancia`. Con un solo año de datos la climatología mensual tiene pocas
observaciones por mes: sirve para ver el ciclo, no para anomalías robustas
(para eso, la serie larga de Landsat 8 y 9 de `cargar_landsat`).

### `utils/series.py`: Landsat 8 y 9
| Función | Qué hace |
|---|---|
| `cargar_ventanas_landsat(items, puntos, radio_m, resolucion, **kwargs)` | Una ventana de Landsat alrededor de cada punto con todas las fechas (verde, rojo, NIR, SWIR1, `lwir11`, `qa_pixel`, `qa_radsat`, en DN). Mismo patrón que `cargar_ventanas`; los `**kwargs` van a `odc.stac.load` |
| `construir_series_landsat(ventanas, puntos, radio_m, fraccion_minima)` | Máscara, **calibración** (temperatura y reflectancia), NDWI y NDVI, y mediana de la ventana. Una fila por punto y fecha; las fechas con nubes se conservan con NaN |
| `agregar_atributos_landsat(datos, items)` | Agrega la plataforma (landsat-8 o landsat-9) y las nubes de la escena a cada fecha |

### `utils/extremos.py` (anomalías y eventos extremos de temperatura)
Dos partes. **Landsat 8 y 9** (series irregulares: un paso cada 8 días, con nubes): climatología por ventana de días, anomalías y *episodios* de observaciones seguidas (no son eventos de Hobday). **Hobday** (series diarias sin huecos), validado contra `xmhw` (ver el docstring del módulo). Un vacío (NaN) nunca se rellena ni se puentea.

| Función | Qué hace |
|---|---|
| `cargar_landsat(tipo)` | Datos de Landsat del taller: `"puntos"` (tabla por punto y fecha, calibrada, con las fechas nubladas como NaN) o `"malla"` (DataArray de temperatura cada 500 m) |
| `serie_landsat(tabla, punto, variable)` | Una variable de un punto como Serie con índice de fechas |
| `climatologia_ventana(serie, base, ventana, percentiles, suavizado, minimo_obs)` | Media, p90, p10 y `n` por día del año, juntando las observaciones a +-15 días de todos los años de la base (2013-2024) |
| `anomalias_observaciones(serie, clima)` | Cada observación con su media, umbrales, anomalía y marcas de calor y frío |
| `detectar_episodios(serie, clima, tipo, min_obs, brecha_max_dias)` | Tabla de episodios: al menos 2 observaciones válidas seguidas sobre el p90 (o bajo el p10), sin cruzar huecos de más de 24 días. La duración es una cota mínima |
| `aislamiento_forest_observaciones(tabla, columnas, clima_por_variable, contaminacion)` | Isolation Forest sobre varias variables (temperatura, reflectancia...) de las observaciones válidas |
| `climatologia_malla(malla, base, ventana, paso)`, `anomalia_malla`, `frecuencia_anomalias_malla` | Lo mismo píxel por píxel: climatología, anomalía y fracción de observaciones anómalas |
| `climatologia_hobday(serie, base, ventana, percentiles, suavizado)` | Hobday: media, p90 y p10 por día del año (1991-2020, ventana de 11 días, suavizado de 31) para una serie **diaria** |
| `aplicar_climatologia(serie, clima)`, `dia_del_anio_366(indice)` | La climatología alineada con la serie; calendario de 366 días |
| `detectar_eventos(serie, clima, tipo, duracion_min, brecha_max)` | Hobday: tabla de eventos de `"calor"` o `"frio"`: rachas de al menos 5 días y unión de brechas de 2 días o menos (primero filtrar, después unir). Duración, intensidad, día del pico, categoría y `en_curso` |
| `resumen_eventos(eventos, por)`, `evento_en_curso(eventos)` | Conteos por década o año; el evento activo en el último día |
| `marcar_dias_evento(serie, eventos)`, `solapamiento(a, b)` | Días dentro de eventos (booleana) y comparación entre dos marcas (Jaccard) |
| `frecuencia_eventos(malla, tipo, base, periodo)` | Eventos por píxel de una malla: `n_eventos`, `dias_en_evento`, `eventos_por_decada` |
| `aislamiento_forest(serie, clima, contaminacion)` | Días atípicos con Isolation Forest (scikit-learn), para compararlos con Hobday |

## Sentinel-2: offset de reflectancia

Desde el 25 de enero de 2022 la ESA procesa Sentinel-2 L2A con el *baseline*
04.00 o posterior, que suma 1000 a los números digitales (DN) para poder
guardar reflectancias negativas. Por eso:

* baseline < 04.00: `reflectancia = DN * 0.0001`
* baseline >= 04.00: `reflectancia = DN * 0.0001 - 0.1`

El baseline no depende solo de la fecha: la ESA reprocesó imágenes antiguas
(por ejemplo, diciembre de 2021 ya existe en 04.00 en Planetary Computer, y
Earth Search tiene casi todo 2018-2021 en 05.00). Y cada catálogo entrega los
DN de forma distinta:

| Catálogo (`CATALOGO_SENTINEL2`) | Colección | Offset que corresponde |
|---|---|---|
| `"planetary"` | `sentinel-2-l2a` | 0 si baseline < 04.00; -0.1 si >= 04.00 |
| `"earthsearch"` | `sentinel-2-c1-l2a` (Colección 1) | siempre -0.1 (todo reprocesado, baseline >= 05.00). **Hueco:** casi no tiene imágenes entre el 29-12-2021 y el 07-12-2022 (solo escenas sueltas); `parametros_busqueda` avisa |
| `"earthsearch-l2a"` (DESACTIVADO) | `sentinel-2-l2a` | 0 casi siempre (Earth Search ya restó los 1000); -0.1 solo en baseline 04.00 entre el 25-01-2022 y fines de febrero de 2022 sin `earthsearch:boa_offset_applied` = True |

**Por qué Earth Search `sentinel-2-l2a` está desactivado.** Sus metadatos de
offset son inconsistentes: `earthsearch:boa_offset_applied` vale False en muchas
imágenes que ya vienen sin el +1000 (en la escena de Concepción, 33 de 45 ítems
dicen False y los píxeles están bien), y el `offset` de `raster:bands` dice
-0.1 aunque ya lo haya restado. Elegirlo (`catalogo="earthsearch-l2a"`) lanza
un `ValueError`. La regla queda en `offset_sentinel2` y sus pruebas, para no
perder lo que se midió en los píxeles. Planetary Computer no informa offset en
los metadatos.

La regla se comprobó mirando los píxeles (agua y zonas oscuras, donde un error
de offset se nota más) y contra Landsat como referencia independiente. La
explicación completa, con las pruebas que respaldan cada regla, está al inicio
de `utils/stac.py` (`help(utils.stac)`).

Cómo se aplica, paso a paso (el flujo del ejemplo de arriba):

1. `preparar_items` quita las versiones repetidas de una misma toma (deja el
   baseline más alto) y exige un solo catálogo.
2. `parametros_carga` entrega un `groupby` que agrupa por (día solar, offset):
   `odc.stac.load` nunca junta en una misma fecha imágenes con offsets
   distintos (por ejemplo, una 03.00 y una 04.00 del mismo día quedan como dos
   fechas).
3. `agregar_offset` decide el offset de cada imagen con `offset_sentinel2` y lo
   guarda en la coordenada `offset_reflectancia` (un valor por fecha).
   (Los cuadernos del taller no llaman a estas funciones: escriben la misma regla a la vista.
   `test_cuadernos_taller.py` comprueba que dan el mismo resultado.)
4. `convertir_a_reflectancia` / `preparar_reflectancia` usan esa coordenada.
   Al seleccionar una fecha (`ds.sel(time=...)`) el offset se conserva; al
   combinar fechas (`ds.median("time")`) se pierde, así que **convierte a
   reflectancia antes de combinar fechas** (si no, hay un error claro). Si los
   datos no traen la coordenada (por ejemplo, no llamaste a `agregar_offset`),
   se avisa y se usa la regla por fecha, que puede fallar en imágenes
   reprocesadas.

Los píxeles sin datos (DN = 0) quedan como NaN, igual que los valores fuera
de 0 a 1 (en agua muy oscura puede haber algunos negativos que quedan NaN).

## Catálogos

`CATALOGO_SENTINEL2` (en `utils/stac.py`) define dónde se busca Sentinel-2:
`"planetary"` (por defecto) o `"earthsearch"` (Colección 1 de Earth Search en
AWS, `sentinel-2-c1-l2a`). Cambiar esa línea cambia todos los notebooks; también
se puede pasar `catalogo="earthsearch"` a `conectar_catalogo`,
`parametros_busqueda` y `parametros_carga`. Las bandas se llaman igual en todos
tras `renombrar_bandas` (B02, B04, B08, SCL, ...). `"earthsearch-l2a"` está
desactivado (ver arriba). Landsat, el DEM y ESA WorldCover se buscan siempre en
Planetary Computer (Earth Search no tiene WorldCover ni Sentinel-3, y su Landsat
está en un bucket de AWS que se paga por descarga).

## Pruebas

```bash
pip install pytest
pytest utils -m "not red"       # sin internet (unos segundos)
pytest utils -m red             # solo las de red: descargan imágenes y tardan (unos 12 minutos)
pytest utils -v                 # todas
```

Las pruebas están en `utils/tests/`; cada archivo empieza con un docstring que
dice qué demuestra. Las ***offline*** (marca implícita, sin internet) usan datos
reales grabados, así que la regla del offset se prueba sin conexión:

* `fixtures/items/`: **metadatos reales de 33 ítems** STAC (Planetary Computer
  `pc_*`, Earth Search `sentinel-2-l2a` `es_*` y `sentinel-2-c1-l2a` `c1_*`) del
  tile 18HXE (Concepción y Talcahuano), de 2018 a 2026. Incluyen reprocesamientos
  duplicados, la ventana de febrero de 2022 y casos en que Earth Search dice
  `boa_offset_applied=False` con los píxeles ya corregidos.
* `fixtures/pixeles/`: **ventanas pequeñas de píxeles reales** (`.npz` con B02,
  B04, B08 y SCL, mar abierto y tierra) de la misma toma en los **tres
  catálogos**, para comparar la reflectancia entre ellos.
* GeoTIFF pequeños creados en el momento (`tmp_path`), con los DN que guardaría
  cada catálogo, para probar la carga completa con `odc.stac.load`.

| Archivo | Qué demuestra |
|---|---|
| `test_offset_sentinel2.py` | La regla del offset sobre los 33 ítems reales (tabla de offsets esperados), límites de la ventana de Earth Search l2a, caso sin baseline, reconocimiento del catálogo y `quitar_duplicados` |
| `test_pixeles_reales.py` | Con píxeles reales, la reflectancia coincide entre catálogos (sesgo < 0.005, un error de offset da 0.1), el agua oscura no queda negativa, y un offset equivocado se detecta |
| `test_busqueda_offline.py` | `parametros_busqueda`, elección de catálogo (`earthsearch` es c1, `earthsearch-l2a` desactivado, Landsat solo en Planetary Computer), aviso del hueco de c1 y `preparar_items` |
| `test_carga_offline.py` | `parametros_carga`, `groupby`, `renombrar_bandas` y `agregar_offset` con `odc.stac.load` real sobre archivos locales; dedupe, catálogos mezclados, offsets distintos en un mismo día |
| `test_reflectancia.py` | `convertir_a_reflectancia` con cubos sintéticos: offset por fecha, DN 0 a NaN, error al combinar fechas antes de convertir; `convertir_a_temperatura` de Landsat (DN a K y a °C) y la máscara de saturación |
| `test_cuadernos_taller.py` | El offset y la reflectancia que los cuadernos 02 a 04 del taller escriben a la vista (extraídos de los `.py`) dan lo mismo que `offset_sentinel2`, `agregar_offset` y `convertir_a_reflectancia` |
| `test_general.py` | Vectorial, raster y gráficos con datos sintéticos |
| `test_series.py` | Puntos, `extraer_series_puntos` con ventana, el flujo Sentinel-2 completo (`cargar_ventanas` + `construir_series`) con GeoTIFF locales y offsets por fecha, `construir_series_landsat` con ventanas sintéticas (calibración, máscara, fechas nubladas), `items_legibles` (archivos dañados), análisis de series, parquet, la serie preparada del taller y los gráficos (folium, hvplot) |
| `test_extremos.py` | Hobday con series sintéticas (racha de 5 y de 4 días, brechas de 2 y 3, filtrar antes de unir, vacíos que no se puentean, frío, 29 de febrero, evento en curso), comparación con `xmhw` (si está instalado), malla diaria e Isolation Forest. Landsat: climatología por ventana, episodios (racha de observaciones, nubes que no cortan, brecha máxima), control de calidad con la reflectancia, malla de 500 m y los datos reales preparados del taller (calibración, huecos, episodios de referencia) |
| `test_ia.py` | Cliente MCP con un servidor falso por stdio y un "LLM" guionizado (sin internet ni clave); la parte con servidores reales lleva la marca `red` |
| `test_red_catalogos.py` | (marca `red`, con internet) La misma toma en los catálogos reales, comparada entre sí y con Landsat; serie que cruza el cambio de baseline; flujo nativo completo; hueco de c1 |

Cómo correr cada grupo:

* Offline: `pytest utils -m "not red"` (unos segundos, sin internet).
* Un solo archivo: `pytest utils/tests/test_pixeles_reales.py -v`.
* Red: `pytest utils -m red` (necesita acceso a Planetary Computer y AWS; descarga
  imágenes, unos 12 minutos). Córrelas antes de un curso o taller.

**Qué probaron los controles de mutación.** Para comprobar que las pruebas
muerden, se rompió cada regla a propósito, se confirmó que fallaban y se
restauró el código:

Se probaron 23 mutaciones; en todas fallaron las pruebas offline (ninguna
sobrevivió), con estos resultados:

| Regla que se rompió | Pruebas que fallaron |
|---|---|
| c1 devuelve 0 en vez de -0.1 | 23 (tabla de offsets, píxeles reales, cargas de c1) |
| Planetary Computer con umbral 05.00 en vez de 04.00 | 34 |
| Planetary Computer decide por fecha en vez de baseline | 17 (las tomas de 2021 reprocesadas con 04.00) |
| Earth Search l2a: se ignora la ventana de fechas | 20 |
| Earth Search l2a: la ventana termina el 20-02-2022 | 2 (límites de la ventana) |
| Earth Search l2a: se ignora `boa_offset_applied` = True | 3 |
| `quitar_duplicados` conserva el baseline más bajo | 17 |
| `preparar_items` no quita duplicados | 7 |
| Se permiten catálogos mezclados | 2 |
| `earthsearch-l2a` se puede elegir (sin desactivar) | 5 |
| El `groupby` ignora el offset (solo el día) | 10 |
| `agregar_offset` asigna 0 a todas las fechas | 9 |
| `agregar_offset` acepta versiones duplicadas | 1 |
| `agregar_offset` no detecta una carga sin el `groupby` del curso | 1 |
| `renombrar_bandas` no renombra | 5 |
| `parametros_carga` no traduce los nombres de banda de Earth Search | 7 |
| `parametros_carga` sin `patch_url` (sin firma de Planetary Computer) | 4 |
| `firmar_url` no descarta el token viejo | 1 |
| Se apaga el aviso del hueco de c1 | 4 |
| Landsat se acepta con un cliente de otro catálogo | 1 |
| DN 0 ya no pasa a NaN | 1 |
| Se combinan fechas antes de convertir sin error | 1 |
| `convertir_a_reflectancia` ignora `offset_reflectancia` (usa -0.1 fijo) | 14 |

Además, `test_un_offset_equivocado_se_detecta` comprueba con píxeles reales que
un offset equivocado se nota: mueve la mediana del agua en exactamente 0.1.
