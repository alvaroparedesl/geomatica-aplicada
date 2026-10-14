# %% [markdown]
# # Bloque 1. Principios FAIR y consulta de repositorios con IA
#
# [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/talleres/fair-series-temporales/01_fair_repositorios_ia.ipynb) [![Licencia: GPL-3.0](https://img.shields.io/badge/licencia-GPL--3.0-blue.svg)](https://github.com/alvaroparedesl/geomatica-aplicada/blob/main/LICENSE) [![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://colab.research.google.com/)
#
# Taller *Datos FAIR, series temporales satelitales y detección de anomalías con Python*. Día 1, mañana (unos 70 minutos).
#
# Antes de analizar datos hay que encontrarlos, y antes de encontrarlos alguien tuvo que describirlos bien. En este cuaderno vemos las dos caras de esa idea:
#
# * Los **principios FAIR** (*Findable, Accessible, Interoperable, Reusable*), aplicados a registros reales de Zenodo y de un Dataverse chileno.
# * Cómo un **asistente de IA** puede consultar esos repositorios por ti mediante **MCP**, y por qué sus respuestas son tan buenas (o tan malas) como los metadatos que encuentra.
#
# **Qué vas a hacer**
#
# | Parte | Tema | Minutos |
# |---|---|---|
# | 1 | Principios FAIR | 15 |
# | 2 | Tu clave de Google AI Studio | 5 |
# | 3 | Qué es MCP y para qué sirve | 5 |
# | 4 | Ver y llamar herramientas a mano (sin IA) | 10 |
# | 5 | Preguntar con IA y mirar cada paso | 15 |
# | 6 | La lección: metadatos pobres, respuestas pobres | 5 |
# | 7 | Ejercicio de metadatos | 15 |
#
# > **Importante:** este cuaderno se ejecuta de principio a fin aunque no tengas clave de Gemini. Las partes 4, 6 y 7 no la necesitan. Si no tienes clave, la parte 5 te lo avisa y sigues con la demostración del expositor.

# %% [markdown]
# ## Preparar el entorno
#
# La celda `Preparar entorno` clona el repositorio del curso (para poder usar la carpeta `utils/`) y la siguiente instala el único paquete que Colab no trae para este bloque: `mcp`, el cliente del protocolo que veremos más abajo.

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

# %%
# @title Instalar los paquetes de este bloque (solo si faltan)
import importlib.util

# Cada línea de requirements.txt es un paquete; instalamos solo los que este bloque necesita.
NECESARIOS = ["mcp"]
requisitos = [linea.strip() for linea in (RAIZ / "talleres/fair-series-temporales/requirements.txt").read_text().splitlines()
              if linea.strip() and not linea.startswith("#")]
requisitos = [r for r in requisitos if r.split()[0] in NECESARIOS]

if any(importlib.util.find_spec(nombre) is None for nombre in NECESARIOS):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *requisitos], check=True)
    print("Instalado:", ", ".join(requisitos))
else:
    print("Ya está todo instalado:", ", ".join(NECESARIOS))

# %% [markdown]
# ## 1. Principios FAIR
#
# Los principios FAIR (Wilkinson et al., 2016) dicen qué necesita un dato para que **otra persona, o un programa, lo encuentre, lo obtenga, lo entienda y lo reutilice**. No hablan de datos "bonitos": hablan de metadatos, identificadores, formatos y licencias.
#
# | Principio | Pregunta práctica | Qué mirar en un registro |
# |---|---|---|
# | **F**indable (encontrable) | ¿Alguien que no me conoce puede llegar a mi dato? | Identificador persistente (DOI), título descriptivo, descripción, palabras clave, lugar y fechas |
# | **A**ccessible (accesible) | ¿Se puede obtener con una dirección estable y un protocolo abierto? | Acceso abierto o restringido (y por qué), archivos descargables, los metadatos siguen existiendo aunque el archivo ya no esté |
# | **I**nteroperable | ¿Se entiende y se combina con otros datos? | Formato abierto (NetCDF, CSV, GeoTIFF), vocabularios y unidades declaradas, enlaces a otros registros (artículo, código) |
# | **R**eusable (reutilizable) | ¿Sé con qué condiciones puedo usarlo y cómo se produjo? | Licencia clara (CC BY 4.0, CC0), autoría con ORCID y afiliación con ROR, procedencia y método |
#
# Un truco para recordarlo: **F** y **A** sirven para llegar al dato, **I** y **R** para poder usarlo cuando llegaste.
#
# Más abajo (parte 4) vamos a leer tres registros reales con esta lista de control en la mano:
#
# * un registro **rico** de Zenodo (una climatología de temperatura del mar en el norte de la Patagonia chilena, con datos para detectar olas de calor marinas);
# * un dataset del **Dataverse de la Universidad de Concepción** al que le faltan campos;
# * un registro de Zenodo titulado solo **"Database"**.
#
# Este taller entero es FAIR en práctica: hoy buscaremos datos con metadatos, el día 2 guardaremos una serie con sus metadatos dentro del archivo y el día 3 publicaremos un depósito con un DOI.

# %% [markdown]
# ## 2. Tu clave de Google AI Studio
#
# Para que un modelo de lenguaje (Gemini) consulte repositorios por ti necesitas una **clave de API**. Cada participante usa **la suya**; nunca se comparte ni se usa la del expositor.
#
# **Cómo conseguirla (unos 5 minutos)**
#
# 1. Entra a [aistudio.google.com/apikey](https://aistudio.google.com/apikey) con una **cuenta personal de Google**.
# 2. Pulsa **Crear clave de API** (si te pide un proyecto, crea uno nuevo) y copia la clave.
# 3. En Colab abre el panel izquierdo, pulsa el icono de la llave (**Secrets**), agrega un secreto llamado exactamente `GOOGLE_API_KEY`, pega la clave como valor y activa **Acceso desde el cuaderno**.
# 4. Vuelve a esta celda y ejecútala: te dirá si la encontró.
# 5. Si trabajas fuera de Colab, escribe la clave en un archivo `.env` junto al cuaderno (una línea: `GOOGLE_API_KEY=tu_clave`; el archivo no se sube a GitHub) o expórtala como variable de entorno antes de abrir Jupyter.
#
# **Antes de empezar, ten en cuenta**
#
# * Debes tener **18 años o más**.
# * Algunas **cuentas universitarias o institucionales** tienen AI Studio bloqueado por su administrador. Si te pasa, usa una cuenta personal de Google. Si no tienes ninguna, sigue la demostración del expositor: todo lo demás del cuaderno funciona igual.
# * El nivel gratuito tiene límites de uso y, según las condiciones de Google para ese nivel, **las consultas pueden usarse para mejorar sus productos**. No envíes datos sensibles.
# * La clave **nunca se escribe en una celda** ni se pega en un cuaderno compartido. Si se te escapa, bórrala en AI Studio y crea otra.

# %%
# @title Comprobar si tengo la clave (no la muestra)
import os


def clave_gemini():
    """La clave GOOGLE_API_KEY de los Secrets de Colab (o, fuera de Colab, del entorno o de un archivo .env); None si no existe."""
    try:
        from google.colab import userdata
        return userdata.get("GOOGLE_API_KEY") or None
    except Exception:  # no estamos en Colab, el secreto no existe o no tiene acceso desde el cuaderno
        from utils.ia import _clave_gemini
        return _clave_gemini()


def hay_clave_gemini():
    """True si existe la clave (no la muestra)."""
    return clave_gemini() is not None


TIENE_CLAVE = hay_clave_gemini()
if TIENE_CLAVE:
    print("Encontré tu clave GOOGLE_API_KEY (no se muestra). Puedes usar la parte 5 con IA.")
else:
    print("No encontré la clave GOOGLE_API_KEY.\n"
          "  1. Crea una en aistudio.google.com/apikey (cuenta personal, 18 años o más).\n"
          "  2. En Colab, icono de la llave (Secrets): agrega GOOGLE_API_KEY y activa 'Acceso desde el cuaderno'.\n"
          "  3. Ejecuta de nuevo esta celda.\n"
          "Mientras tanto el cuaderno sigue sin IA: las partes 4, 6 y 7 funcionan igual.")

# %% [markdown]
# ## 3. Qué es MCP y para qué sirve
#
# **MCP** (*Model Context Protocol*) es un estándar abierto para que un asistente de IA use **herramientas externas**: buscar en un repositorio, leer un archivo, consultar una base de datos. Un servidor MCP publica una lista de herramientas, cada una con su nombre, una descripción y sus parámetros; un **cliente** las llama.
#
# ```text
#   tú ──pregunta──▶  cuaderno (cliente MCP)  ──▶  Gemini (modelo de lenguaje)
#                            │                          │
#                            │◀── "llama a zenodo_search_records(...)" ──┘
#                            │
#                            └──▶ servidor MCP de Zenodo ──▶ API de Zenodo ──▶ datos reales
# ```
#
# Aquí **el cuaderno es el cliente**: lo único que instalamos es el paquete `mcp`. Tú **no programas ningún servidor**: usamos servidores que ya existen. Lo que importa es entender qué te dan:
#
# * **Datos verificables.** Gemini no responde "de memoria": decide qué herramienta llamar, la herramienta consulta el repositorio real y devuelve un **total** y una lista de registros.
# * **Trazabilidad.** El cuaderno imprime cada paso (qué herramienta, con qué argumentos, cuánto devolvió). Lo que aprendes es **qué hizo el modelo**, no solo qué contestó.
# * **El mismo servidor para muchas IA.** Una herramienta publicada una vez sirve a cualquier cliente MCP.
#
# **Y lo que no resuelve**
#
# * La IA solo ve **metadatos** (título, descripción, palabras clave, ubicación, licencia), no el contenido de tus archivos.
# * Puede equivocarse al elegir la consulta (por ejemplo, olvidar que "Biobío" se escribe de dos maneras). Por eso **siempre se compara con el total que informa el servidor**.
# * Los servidores públicos tienen cuotas y a veces se caen. Cuando eso pase, repites la celda o sigues con la demostración.
#
# Servidores que usamos hoy:
#
# | Alias | Qué es | Notas |
# |---|---|---|
# | `zenodo` | Zenodo (artículos, datasets, software) | Servidor de la comunidad, alojado por un tercero. Informa el total y filtra por tipo y año |
# | `dataverse` | Dataverse (repositorios universitarios, como el de la UdeC) | Servidor de la comunidad GDCC. Informa el total, pero casi no tiene filtros |
# | `dataverse_local` | El mismo Dataverse, con un servidor pequeño que corre **dentro de tu Colab** | Es el respaldo si el público se cae. Agrega filtros, conteos por año y licencia, y la lista de campos vacíos de un dataset |
# | `openalex` | OpenAlex (índice de artículos científicos) | Opcional. Es la fuente correcta para contar **artículos**, porque Zenodo solo tiene una parte |

# %% [markdown]
# ## 4. Ver y llamar herramientas a mano (sin IA)
#
# `utils/ia.py` esconde la parte técnica de MCP (conexión, protocolo, asincronía). Con dos funciones puedes **ver** qué ofrece un servidor y **llamar tú mismo** a una herramienta. No necesitan clave ni IA: es lo que Gemini hace por ti en la parte 5.

# %%
# @title Cargar las funciones de utils.ia
from utils.ia import SERVIDORES, herramientas, mensaje_de_error, preguntar, usar


def intentar(funcion, *args, **kwargs):
    """Llama a una función de utils.ia; si el servidor falla (caído, cuota), avisa y el cuaderno sigue."""
    for intento in (1, 2):
        try:
            return funcion(*args, **kwargs)
        except Exception as error:  # noqa: BLE001
            if intento == 2:
                print(f"No hubo respuesta ({mensaje_de_error(error)}).\n"
                      "Los servidores son servicios públicos compartidos: espera un minuto y vuelve a ejecutar la celda, "
                      "o sigue con la demostración del expositor.")


print("Servidores disponibles:", ", ".join(SERVIDORES))

# %%
# 1. Qué herramientas ofrece el servidor de Zenodo (* = parámetro obligatorio)
intentar(herramientas, "zenodo")

# %% [markdown]
# Cada herramienta tiene un nombre, una descripción en lenguaje natural (la que lee el modelo para decidir si le sirve) y sus parámetros. `zenodo_search_records` busca; `zenodo_get_record` trae un registro completo por su identificador.
#
# Ahora llamamos a una herramienta nosotros. Buscamos **datasets de 2026 que mencionan el Biobío**. Es la misma llamada que haría el modelo.

# %%
# 2. Llamada manual: tres datasets de 2026 que mencionan el Biobío
texto = intentar(
    usar, "zenodo", "zenodo_search_records",
    query="Biobío", resource_type=["dataset"], published_from="2026", published_to="2026", size=3,
)

# %% [markdown]
# La primera línea de la respuesta dice **cuántos registros coinciden en total** (`N matches`), aunque solo se muestren 3. Ese total es la cifra que importa cuando alguien pregunta "¿cuántos?". Cuando contestes con IA, vas a comparar siempre con él.
#
# Armemos una función que extrae ese total y úsala para contar de distintas maneras. Los repositorios crecen todos los días, así que tus cifras pueden diferir un poco de las del expositor (las de la tabla se midieron el 10 de octubre de 2026).

# %%
import re


def total_zenodo(**filtros):
    """Total de registros de Zenodo que coinciden (la cifra 'N matches' que informa el servidor)."""
    respuesta = usar("zenodo", "zenodo_search_records", mostrar=False, size=1, published_from="2026", published_to="2026", **filtros)
    return int(re.search(r"(\d+) match", respuesta).group(1))


consultas = {
    "Todo lo de 2026 que dice Biobío": dict(query="Biobío"),
    "... y que dice Biobio (sin tilde)": dict(query="Biobio"),
    "... con cualquiera de las dos escrituras": dict(query="Biobío OR Biobio"),
    "Solo datasets (cualquiera de las dos)": dict(query="Biobío OR Biobio", resource_type=["dataset"]),
    "Solo artículos de revista (cualquiera de las dos)": dict(query="Biobío OR Biobio", resource_type=["publication-article"]),
    "Todas las publicaciones (cualquiera de las dos)": dict(query="Biobío OR Biobio", resource_type=["publication"]),
}
totales = {}
try:
    for nombre, filtros in consultas.items():
        totales[nombre] = total_zenodo(**filtros)
        print(f"{totales[nombre]:>4}  {nombre}")
except Exception as error:  # noqa: BLE001
    print(f"El servidor de Zenodo no respondió ({type(error).__name__}); repite la celda en un minuto.")

# %% [markdown]
# Mira lo que pasó:
#
# * **Con tilde y sin tilde dan totales distintos** y la unión con `OR` da más que cada una. Un buscador no sabe que `Biobío` y `Biobio` son lo mismo, y cada persona que depositó datos escribió como quiso.
# * **El tipo de recurso importa.** Las "publicaciones" incluyen los artículos de revista y otras cosas (informes, preprints, tesis). Los datasets van aparte.
# * La pregunta "¿cuántas publicaciones de 2026 mencionan el Biobío?" **no tiene una única respuesta correcta**: depende de qué cuentas como publicación y de cómo escribes el nombre. Por eso conviene preguntar con las variantes y decir cuáles usaste.
#
# **Tu turno (3 min).** Cambia la consulta por `Concepción`, por `Arauco` y por `"Golfo de Arauco"` (con comillas, para buscar la frase). ¿Qué le pasa al total? ¿Por qué `Concepción` es una consulta peligrosa? (Pista: hay ciudades con ese nombre en Argentina, Paraguay, Colombia y Filipinas.)

# %%
# Tu turno: cambia la consulta y mira el total
try:
    print(total_zenodo(query="Concepción"), "registros de 2026 mencionan Concepción")
except Exception as error:  # noqa: BLE001
    print(f"El servidor de Zenodo no respondió ({type(error).__name__}); repite la celda en un minuto.")

# %% [markdown]
# ### Tres registros reales con la lista de control FAIR
#
# `zenodo_get_record` trae el registro completo de un depósito. Veamos uno **rico**: una climatología de la temperatura del mar en el norte de la Patagonia chilena, con umbrales para detectar olas de calor y de frío marinas (es el tema del bloque 7). Mostramos solo la parte de los metadatos.

# %%
# Registro rico de Zenodo: climatología de temperatura del mar en el norte de la Patagonia
registro_rico_zenodo = intentar(usar, "zenodo", "zenodo_get_record", id="14845077", mostrar=False)
if registro_rico_zenodo:
    print(registro_rico_zenodo.split("### Description")[0].strip())

# %% [markdown]
# Aplicando la lista de la parte 1:
#
# * **F:** tiene DOI (`10.5281/zenodo.14845077`), un título que dice qué, dónde y qué incluye, **palabras clave** (`marine heatwave`, `sea surface temperature`, `chile`...) y la zona con coordenadas en la descripción.
# * **A:** acceso abierto, archivos públicos. Los metadatos de Zenodo se pueden leer por una API abierta, que es justo lo que usa el servidor MCP.
# * **I:** archivos NetCDF (un formato abierto y estándar en oceanografía), con la resolución y las unidades descritas.
# * **R:** licencia **CC BY 4.0**, autoría con **ORCID** y afiliación con **ROR**, financiamiento declarado.
#
# Ahora un dataset del **Dataverse de la Universidad de Concepción** (`datav.udec.cl`). El servidor local, además del registro, entrega la lista de **campos vacíos**. Es un estudio de caso en la provincia de Arauco.

# %%
# Dataset FJBASE del Dataverse de la UdeC: qué metadatos tiene y cuáles faltan
intentar(usar, "dataverse_local", "dataverse_dataset", persistent_id="doi:10.48665/udec/FJBASE", host="datav.udec.cl")

# %% [markdown]
# Este dataset **sí** tiene DOI, licencia (CC0) y una buena descripción. Pero no tiene **palabras clave**, ni **cobertura geográfica**, ni **período** cubierto. Para una persona que lo lee, el título dice "Arauco" y listo. Para un buscador, es un registro **sin lugar**.
#
# Veamos qué pasa cuando alguien lo busca por región. Misma instalación, tres consultas:

# %%
for consulta in ["Biobío", "Biobio", "Arauco"]:
    print(f"--- {consulta}")
    respuesta = intentar(usar, "dataverse_local", "dataverse_search", q=consulta, host="datav.udec.cl", per_page=1, mostrar=False)
    if respuesta:
        print(respuesta.splitlines()[0])

# %% [markdown]
# **"Biobío" no encuentra el dataset y "Arauco" sí.** Arauco **es** una provincia del Biobío, pero el registro nunca lo dice. Quien busque datos "del Biobío" no llega a él, aunque estos datos sean precisamente del Biobío.
#
# Por último, el caso extremo: un registro de Zenodo de 2026, de un autor de la Universidad de Concepción, titulado solo **"Database"**.

# %%
# Un registro de Zenodo titulado "Database": ¿qué se puede saber de él?
registro_database = intentar(usar, "zenodo", "zenodo_get_record", id="22260072", mostrar=False)
if registro_database:
    # Mostramos solo el encabezado: título, tipo, acceso y licencia (sin los datos de la persona autora)
    print(registro_database.split("### Creators")[0].strip())

# %% [markdown]
# No tiene descripción, ni palabras clave, ni ubicación, ni fechas de observación. Tiene licencia y un DOI (y sus archivos son de acceso restringido), pero **ninguna búsqueda por tema puede encontrarlo**: ni "Biobío", ni "temperatura", ni "datos oceanográficos". Es técnicamente un registro público, y prácticamente invisible. Tampoco sabemos de qué trata.
#
# **Tu turno (2 min).** Con la lista de control de la parte 1, ¿qué letras de FAIR (F, A, I, R) cumple cada uno de los tres registros? Anota una frase por registro.

# %% [markdown]
# ## 5. Preguntar con IA y mirar cada paso
#
# `preguntar()` conecta Gemini con uno o varios servidores. El modelo recibe tu pregunta y la lista de herramientas, **decide qué llamar**, el cuaderno ejecuta la llamada y le devuelve el resultado, y así hasta que tiene una respuesta. Cada paso se imprime así (los números son de ejemplo):
#
# ```text
# [paso 1] El modelo llama a  zenodo.zenodo_search_records(query='Biobío OR Biobio', ...)
#           <- 5230 caracteres en 2.1 s: ## Zenodo search: 33 matches ...
# ```
#
# Además le damos una instrucción de sistema en español: debe usar **los totales que informa la herramienta** (no contar filas), probar variantes con y sin tilde, escribir **"no consta"** cuando un dato no aparece en el registro, y terminar con *Cómo lo calculé* y *Límites*.
#
# Para no frenar el cuaderno si todavía no tienes clave, envolvemos `preguntar` en una función que comprueba antes.

# %%
# @title Función para preguntar (avisa si no hay clave)
def consultar(pregunta, servidores=("zenodo",), **kwargs):
    """Pregunta con IA si hay clave; si no, explica qué hacer en vez de fallar."""
    if not hay_clave_gemini():
        print("Esta celda necesita tu clave de Gemini (secreto GOOGLE_API_KEY) y no la encontré.\n"
              "Sigue la demostración del expositor y, cuando tengas la clave, vuelve a ejecutarla.\n"
              f"Pregunta que se habría hecho: {pregunta}")
        return None
    return intentar(preguntar, pregunta, servidores=list(servidores), **kwargs)


# %% [markdown]
# ### Pregunta 1: Zenodo
#
# Es la pregunta de la parte 4, pero ahora la decide el modelo. **Antes de ejecutar**, mira de nuevo la tabla de totales de arriba: sabemos qué esperar.

# %%
consultar("¿Cuántas publicaciones de 2026 en Zenodo mencionan el Biobío? Dime cuántas son artículos y cuántos son datasets.",
          servidores=["zenodo"])

# %% [markdown]
# **Comprueba la respuesta.** Compara los números con los de la parte 4:
#
# * ¿Usó el total que informa la herramienta (`N matches`) o contó filas?
# * ¿Probó **Biobío** y **Biobio**? ¿Los sumó con `OR` o los dio por separado?
# * ¿Separó artículos y datasets con el filtro de tipo de recurso, o los mezcló?
# * ¿Dijo "Cómo lo calculé" y "Límites"?
#
# Si no coinciden, no es un fallo del cuaderno: es el tema de este bloque. Pregunta de nuevo con más precisión ("usa Biobío OR Biobio") y mira qué cambia.
#
# ### Pregunta 2: el Dataverse de la UdeC
#
# El Dataverse de la Universidad de Concepción (`datav.udec.cl`) es pequeño (17 datasets al día de hoy) y está en nuestra zona. Si el servidor público de la comunidad no responde, cambia `servidores=["dataverse"]` por `servidores=["dataverse_local"]`.

# %%
consultar("En el Dataverse de la Universidad de Concepción (datav.udec.cl), ¿cuántos datasets mencionan Biobío y cuántos mencionan Arauco? "
          "Si alguno de Arauco no aparece al buscar Biobío, dime por qué.",
          servidores=["dataverse"])

# %% [markdown]
# ### Pregunta 3 (opcional): artículos con OpenAlex
#
# Zenodo contiene solo una parte de lo que se publica. Para contar **artículos**, la fuente correcta es OpenAlex, que cuenta por la **afiliación de los autores** (no por el tema). Cada llamada tiene un costo diminuto en un presupuesto diario compartido por toda la sala, así que haz **una sola** pregunta.

# %%
consultar("¿Cuántos artículos de 2026 son de autores con afiliación en instituciones de la Región del Biobío (Chile)? "
          "Dime qué instituciones consideraste.",
          servidores=["openalex"], max_pasos=8)

# %% [markdown]
# **Cómo se verifica sin IA.** El mismo conteo se puede hacer a mano con las dos herramientas que el modelo usó: primero listar las instituciones de la región y luego contar los artículos de 2026 con afiliación en cualquiera de ellas. Con los datos de hoy (10 de octubre de 2026) el total está entre **3.300 y 3.400 artículos**, y sube cada semana porque se siguen indexando.

# %%
# Opcional: la verificación a mano (dos llamadas; cuenta contra el presupuesto compartido de OpenAlex)
IDS_BIOBIO = ["I172787465", "I3124150290", "I221450849", "I38685851", "I4210155123",
              "I4405259846", "I4405270024", "I4405276063", "I4405263578", "I4405261347"]  # instituciones del Biobío en OpenAlex
respuesta = intentar(
    usar, "openalex", "openalex_search_entities", mostrar=False,
    entity_type="works", per_page=1, select=["id"],
    filters={"institutions.id": "|".join(IDS_BIOBIO), "publication_year": "2026", "type": "article"},
)
if respuesta:
    print("Total informado por OpenAlex:", re.search(r"\*\*Total:\*\* (\d+)", respuesta).group(1), "artículos de 2026")

# %% [markdown]
# **Tu turno (3 min).** Escribe **tu propia pregunta** para la IA sobre un dato que te interese de la zona (por ejemplo, "¿hay datasets de temperatura del mar para Chile con licencia abierta?"). Antes de ejecutar, escribe en una frase qué esperas encontrar. Después revisa los pasos: ¿qué filtros usó? ¿qué dijo que no sabía?

# %%
# Tu turno: tu propia pregunta
consultar("Escribe aquí tu pregunta", servidores=["zenodo"])

# %% [markdown]
# ## 6. La lección: metadatos pobres, respuestas pobres
#
# Un asistente solo puede responder con lo que el repositorio **declara**. Lo que no está en los metadatos no existe para él, y la IA no puede inventar lo que falta (de hecho, le pedimos que escriba "no consta"). Los casos reales que acabas de ver:
#
# | Caso | Qué pasa | Qué lo arregla |
# |---|---|---|
# | Dataset `FJBASE` (UdeC) | Es de Arauco, pero no tiene palabras clave ni cobertura geográfica: **"Biobío" no lo encuentra** | Agregar región, provincia y coordenadas |
# | Registro "Database" (Zenodo) | Sin descripción, palabras clave ni ubicación: ninguna búsqueda por tema puede llegar a él | Un título que diga qué, dónde y cuándo, y una descripción |
# | 166 registros de la UdeC en Zenodo sin ubicación | Una búsqueda de "Concepción" los devuelve por la **afiliación** de los autores (incluido un dataset sobre Cuba) | Declarar el lugar de la observación, no solo el de la institución |
# | `Biobío` y `Biobio` | Los totales cambian según la tilde | Palabras clave en español **y** en inglés, con y sin tilde |
# | Ubicación estructurada | 49 registros de Zenodo aparecen bajo "Chile" **solo** porque llenaron el campo de ubicación | Llenar el campo, no solo mencionarlo en el texto |
#
# Un asistente con metadatos pobres no solo contesta peor: **da respuestas que parecen buenas y están incompletas**. Y quien tiene los datos más relevantes (como `FJBASE`) es quien sale perdiendo, porque nadie los encuentra.
#
# La buena noticia es que la solución depende de ti: **mejorar los metadatos de tus propios datos es la forma más directa de mejorar lo que cualquier asistente, o cualquier persona, podrá decir de ellos**. Es lo que haces en el ejercicio.

# %% [markdown]
# ## 7. Ejercicio de metadatos (15 minutos)
#
# No necesita cuentas ni clave. Trabajamos con un registro como los que vimos, escrito como un diccionario de Python con campos parecidos a los de Zenodo y DataCite.
#
# 1. Mira el registro **de partida**: pobre, como "Database".
# 2. Haz **cuatro preguntas** a un buscador de metadatos simulado. Verás cuatro veces `no consta`.
# 3. Completa los campos del registro en la celda de **Tu turno**.
# 4. Repite las preguntas hasta que las cuatro se respondan y no queden carencias.
#
# El buscador simulado es más tolerante que uno real (no distingue tildes ni mayúsculas), pero tiene la misma limitación: solo ve el **título, la descripción, las palabras clave y los lugares**. La afiliación de la persona que depositó **no dice dónde se tomó el dato**.

# %%
# @title Registro de partida y revisor
import copy
import unicodedata

# Un registro pobre, del tipo que se ve a menudo (inventado, basado en patrones reales)
registro_pobre = {
    "title": "Database",
    "description": "",
    "keywords": [],
    "locations": [],            # [{"place": "...", "lat": .., "lon": ..}]
    "license": None,
    "creators": [{"name": "Pérez, Ana", "affiliation": "Universidad de Concepción"}],
    "dates": [],                # [{"type": "Collected", "range": "2024-01-01/2024-12-31"}]
    "related_identifiers": [],  # [{"relation": "isSupplementTo", "doi": "10.xxxx/articulo"}]
}


def revisar(registro):
    """Lista de carencias que impiden que el dato se encuentre o se entienda."""
    falta = []
    if len(registro["title"].split()) < 5:
        falta.append("título vago (menos de 5 palabras; di QUÉ, DÓNDE y CUÁNDO)")
    if len(registro["description"]) < 150:
        falta.append("descripción ausente o corta (menos de 150 caracteres)")
    if len(registro["keywords"]) < 3:
        falta.append("menos de 3 palabras clave (agrega en español e inglés)")
    if not registro["locations"]:
        falta.append("sin ubicación geográfica (lugar y coordenadas)")
    if not registro["license"]:
        falta.append("sin licencia: nadie sabe si puede reutilizarlo")
    if not registro["dates"]:
        falta.append("sin fechas de la observación")
    if not any("orcid" in autor for autor in registro["creators"]):
        falta.append("autoría sin ORCID")
    if not registro["related_identifiers"]:
        falta.append("sin enlace al artículo asociado")
    return falta


def _normalizar(texto):  # quita tildes y mayúsculas, como haría un buscador tolerante
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()


def texto_buscable(registro):
    """Lo que ve un buscador de metadatos: título, descripción, palabras clave y lugares."""
    partes = [registro["title"], registro["description"], *registro["keywords"], *[lugar["place"] for lugar in registro["locations"]]]
    return _normalizar(" ".join(partes))


PREGUNTAS = {
    "¿Hay datos del Biobío?": ["biobio"],
    "¿Hay datos de la Bahía de Concepción?": ["bahia de concepcion"],
    "¿Hay datos de temperatura del mar en Chile?": ["temperatura", "mar", "chile"],
    "¿Hay datos con licencia abierta (CC BY)?": None,  # se responde con el campo de licencia
}


def probar(registro, nombre):
    """Hace las cuatro preguntas al registro y lista lo que falta."""
    texto = texto_buscable(registro)
    print(f"\n== {nombre}")
    for pregunta, terminos in PREGUNTAS.items():
        if terminos is None:
            encontrado = bool(registro["license"] and registro["license"].startswith("CC"))
        else:
            encontrado = all(termino in texto for termino in terminos)
        print(f"  {'ENCONTRADO' if encontrado else 'no consta '}  {pregunta}")
    print("  Carencias:", "; ".join(revisar(registro)) or "ninguna")


probar(registro_pobre, "Registro de partida")

# %% [markdown]
# **Tu turno (10 min).** Completa el registro de abajo como si los datos fueran **tuyos**: una serie horaria de temperatura superficial del mar medida con una boya fija en la Bahía de Concepción durante 2024. (Puedes inventar los detalles que falten, pero que sean verosímiles.)
#
# Piensa en lo que necesita alguien que **no te conoce** y no puede preguntarte nada:
#
# * **Título:** qué, dónde y cuándo.
# * **Descripción:** variable, unidades, instrumento y profundidad, control de calidad, formato de los archivos, región.
# * **Palabras clave:** en **español y en inglés** (`temperatura superficial del mar` y `sea surface temperature`), con el nombre de la región.
# * **Lugar:** nombre y coordenadas.
# * **Licencia**, **fechas** de la observación, **ORCID** (y ROR de la institución) y el **artículo** relacionado.
#
# Ejecuta la celda las veces que quieras. Cuando las cuatro preguntas den `ENCONTRADO` y no queden carencias, terminaste.

# %%
# Tu turno: completa los campos (parten vacíos, así que la celda se ejecuta aunque aún no los llenes)
registro_mio = copy.deepcopy(registro_pobre)

registro_mio["title"] = "Database"                  # reemplaza: ¿qué, dónde y cuándo?
registro_mio["description"] = ""                    # escribe al menos 150 caracteres
registro_mio["keywords"] = []                       # por ejemplo ["...", "...", "..."]; incluye español e inglés
registro_mio["locations"] = []                      # por ejemplo [{"place": "...", "lat": -36.7, "lon": -73.0}]
registro_mio["license"] = None                      # por ejemplo "CC-BY-4.0"
registro_mio["dates"] = []                          # por ejemplo [{"type": "Collected", "range": "2024-01-01/2024-12-31"}]
registro_mio["creators"] = [{"name": "Pérez, Ana", "affiliation": "Universidad de Concepción"}]  # agrega "orcid" y "ror"
registro_mio["related_identifiers"] = []            # por ejemplo [{"relation": "isSupplementTo", "doi": "10.xxxx/articulo"}]

probar(registro_mio, "Mi registro")

# %% [markdown]
# ### ¿Y qué diría la IA de tu registro?
#
# Si tienes clave, puedes darle a Gemini **solo tu registro** como contexto y hacerle las mismas cuatro preguntas. Es lo que le pasa a un asistente cuando solo tiene los metadatos. Prueba primero con el registro de partida y luego con el tuyo.

# %%
# @title Preguntar a Gemini usando solo un registro (necesita clave)
import json


def preguntar_al_registro(registro):
    """Le da a Gemini solo los metadatos de un registro y le hace las cuatro preguntas del ejercicio."""
    if not hay_clave_gemini():
        print("Esta celda necesita tu clave de Gemini (secreto GOOGLE_API_KEY) y no la encontré. "
              "Sigue la demostración del expositor.")
        return
    from utils.ia import explicar_error_gemini, generar_texto, mensaje_de_error

    instruccion = (
        "Eres un asistente que responde SOLO con los metadatos del registro siguiente. Si un dato no aparece en ellos, "
        "responde 'no consta' y no lo supongas. Responde en español, una línea por pregunta.\n\n"
        f"Registro:\n{json.dumps(registro, ensure_ascii=False, indent=2)}\n\nPreguntas:\n"
        + "\n".join(f"{i}. {p}" for i, p in enumerate(PREGUNTAS, start=1))
    )
    try:
        print(generar_texto(instruccion))
    except Exception as error:  # noqa: BLE001
        print(explicar_error_gemini(error) or f"Gemini no respondió ({mensaje_de_error(error)}).")


print("--- Registro de partida")
preguntar_al_registro(registro_pobre)

# %%
print("--- Mi registro")
preguntar_al_registro(registro_mio)

# %% [markdown]
# ### Solución de referencia
#
# Una manera de completar el registro (hay muchas buenas). Compara con el tuyo: ¿qué campos pusiste que aquí faltan? ¿cuáles agregaste mejor?

# %%
registro_referencia = {
    "title": "Temperatura superficial del mar en la Bahía de Concepción (Biobío, Chile), datos de boya 2024",
    "description": ("Serie horaria de temperatura superficial del mar (°C) medida con una boya fija en la Bahía de "
                    "Concepción, Región del Biobío, Chile, entre enero y diciembre de 2024. Sensor CTD a 1 m de profundidad; "
                    "datos con control de calidad (valores fuera de 5-25 °C marcados). Formato CSV, UTF-8, separador coma."),
    "keywords": ["temperatura superficial del mar", "sea surface temperature", "Bahía de Concepción", "Biobío",
                 "Chile", "oceanografía", "boya"],
    "locations": [{"place": "Bahía de Concepción, Región del Biobío, Chile", "lat": -36.75, "lon": -73.03}],
    "license": "CC-BY-4.0",
    "creators": [{"name": "Pérez, Ana", "affiliation": "Universidad de Concepción",
                  "orcid": "0000-0000-0000-0000", "ror": "0460jpj73"}],
    "dates": [{"type": "Collected", "range": "2024-01-01/2024-12-31"}],
    "related_identifiers": [{"relation": "isSupplementTo", "doi": "10.xxxx/articulo"}],
}
probar(registro_referencia, "Registro de referencia")

# %% [markdown]
# Fíjate en que este registro **responde las cuatro preguntas** y no depende de que alguien conozca a la autora. Es también la lista de lo que pediremos el día 3, cuando publiques tu propio depósito en Zenodo Sandbox.
#
# ## Cierre
#
# * **FAIR** es una lista de control práctica: identificador, metadatos ricos, acceso, formato abierto, licencia, autoría con ORCID/ROR.
# * **MCP** deja que un asistente consulte repositorios reales, y tú puedes ver cada paso. La IA no reemplaza al total que informa el servidor: lo usa y lo cita.
# * **Metadatos pobres, respuestas pobres.** Lo que no está declarado no se puede encontrar, ni por una persona ni por una IA.
# * Tu mejor herramienta para que **tus** datos se usen es describirlos bien.
#
# **Para llevarte:** una lista de comprobación de metadatos para tus propios datos (los ocho puntos de `revisar`), y el hábito de comparar toda respuesta de una IA con el total que informa la fuente.
#
# Siguiente bloque: **catálogos STAC**, donde los metadatos FAIR son la base para buscar imágenes satelitales (`02_stac_busqueda`).
