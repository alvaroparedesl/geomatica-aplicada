# %% [markdown]
# # Presentaciones y publicación FAIR en Zenodo Sandbox
#
# [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/talleres/fair-series-temporales/09_publicacion_zenodo.ipynb) [![Licencia: GPL-3.0](https://img.shields.io/badge/licencia-GPL--3.0-blue.svg)](https://github.com/alvaroparedesl/geomatica-aplicada/blob/main/LICENSE) [![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://colab.research.google.com/)
#
# **Bloque 9 del taller "Datos FAIR, series temporales satelitales y detección de anomalías con Python"** (unos 40 minutos de publicación, además de las presentaciones).
#
# El primer día aprendiste a **encontrar** datos de otras personas y viste que la calidad de sus metadatos decide si una IA (o tú) los encuentra y los entiende. Hoy te toca el otro lado: **publicar tu propio trabajo** para que otras personas lo encuentren, lo citen y lo reutilicen.
#
# | Parte | Minutos |
# |---|---|
# | Repaso y pulido del proyecto | 10 |
# | Presentaciones (6 minutos por persona) | 65 |
# | Publicación en Zenodo Sandbox | 40 |
# | Cierre | 25 |
#
# En este cuaderno vas a:
#
# 1. Preparar tu **presentación de 6 minutos** (guía en la sección 1).
# 2. Repasar qué hace que un depósito sea **FAIR** y por qué los metadatos importan (el vínculo con el bloque 1).
# 3. **Publicar en Zenodo Sandbox** por la API REST: crear el depósito, subir tu cuaderno y tus datos derivados, completar los metadatos, publicar y **recuperar el registro por su DOI**.

# %% [markdown]
# ## Configuración del entorno

# %%
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

# %% [markdown]
# ## 1. Guía para tu presentación de 6 minutos
#
# Cada persona presenta su repositorio y sus hallazgos principales en **6 minutos**, con unos 2 minutos para preguntas y cambio de turno. Con 10 participantes son unos 65 minutos. Con 6 minutos no se cuenta *todo lo que hiciste*: se cuenta **una historia corta con una conclusión**.
#
# ### Estructura sugerida
#
# | Minuto | Qué cuentas | Apoyo |
# |---|---|---|
# | 0:00 a 0:30 | **La pregunta.** ¿Qué fenómeno ambiental miraste y por qué importa? Una frase | Un mapa con tus puntos o tu zona |
# | 0:30 a 1:30 | **Los datos.** ¿De dónde salen (STAC, Landsat o Sentinel-2), qué periodo, qué resolución, qué limitaciones? | La tabla de tu serie o su metadato |
# | 1:30 a 3:00 | **El método.** Qué hiciste con los datos, en 3 pasos (por ejemplo: máscara de nubes, serie por punto, anomalías o eventos) | Un esquema simple o el código clave, no todo el cuaderno |
# | 3:00 a 4:30 | **El resultado principal.** Un solo gráfico, bien explicado: qué se ve y qué significa | **Un** gráfico, con título y ejes legibles |
# | 4:30 a 5:30 | **Lo que no sabes.** Una limitación o una sorpresa (por ejemplo, las nubes de Landsat y Sentinel-2, o la resolución de la banda térmica) y cómo la trataste | Una frase honesta |
# | 5:30 a 6:00 | **Reproducibilidad.** El enlace a tu repositorio y qué metadatos tiene tu depósito | La URL de GitHub y el DOI (que obtendrás en este cuaderno) |
#
# ### Consejos
#
# * **Un mensaje.** Si el público recuerda una sola cosa, ¿cuál quieres que sea? Escríbela primero.
# * **Un gráfico bueno vale más que cinco regulares.** Tamaño de letra legible, título que diga la conclusión, unidades en los ejes.
# * **Muestra tus decisiones, no solo tus resultados.** "Usé la máscara SCL en vez del filtro de la escena porque..." es lo más valioso para quien te escucha.
# * **Cuenta lo que no funcionó.** Un vacío, una imagen mala, una prueba que descartaste: eso es ciencia, y ayuda a otros.
# * **Ensaya con cronómetro.** Seis minutos pasan muy rápido. Si te sobran 30 segundos, bien; si te faltan 2 minutos, recorta.
#
# ### Qué hace el público
#
# Anota para cada presentación una cosa **que te gustó**, una **pregunta** y una **sugerencia concreta**. Las usaremos en la retroalimentación.

# %% [markdown]
# ## 2. ¿Qué hace FAIR a un depósito?
#
# Recordemos los principios FAIR y qué ofrece un repositorio como Zenodo para cada uno:
#
# | Principio | Qué significa | Qué hace Zenodo |
# |---|---|---|
# | **F**indable (encontrable) | Un identificador único y persistente, y metadatos ricos para buscarlo | Asigna un **DOI** y indexa título, descripción, palabras clave, ubicaciones y fechas |
# | **A**ccessible (accesible) | Se recupera por su identificador con un protocolo abierto | El DOI redirige al registro; la **API REST** y OAI-PMH son abiertas |
# | **I**nteroperable | Metadatos en formatos estándar y que apuntan a otros recursos | Exporta a DataCite y otros; permite **identificadores relacionados** (la fuente de tus datos) y ORCID |
# | **R**eusable (reutilizable) | Licencia clara y procedencia documentada | **Licencia** obligatoria para registros abiertos; descripción, versión y autoría |
#
# ### El vínculo con el bloque 1: los metadatos deciden qué encuentra la IA
#
# En el bloque 1 una IA conectada a Zenodo respondió mal (o dijo "no consta") cuando los registros tenían metadatos pobres: sin ubicación, sin palabras clave en el idioma correcto, sin una descripción que dijera qué contiene. También viste que **una tilde cambia los resultados**: buscar "Biobío" o "Biobio" daba conteos distintos.
#
# Lo mismo te va a pasar a ti, pero al revés. Quien (o la IA que) busque datos de temperatura del mar o de Sentinel-2 en Concepción encontrará tu registro **solo si tus metadatos lo dicen**. Por eso, al completar los metadatos de abajo:
#
# * Pon la **ubicación** como dato estructurado (`locations`), no solo en el texto.
# * Escribe **palabras clave en español y en inglés**, y con y sin tilde (`Biobío` y `Biobio`).
# * Describe qué contiene, de dónde viene (la fuente, con un identificador relacionado), qué periodo cubre y qué **limitaciones** tiene.
# * Indica la **licencia**, sin ambigüedad.
# * Identifica a las personas con **ORCID**.
#
# Más abajo, una pequeña prueba que repite las preguntas del bloque 1 sobre **tu** registro, antes de publicarlo.

# %% [markdown]
# ## 3. Zenodo Sandbox, no Zenodo real
#
# > **Un DOI real es permanente.** Un registro publicado en [zenodo.org](https://zenodo.org) no se puede borrar: puedes publicar versiones nuevas, pero el registro y su DOI quedan para siempre, a la vista de todo el mundo. Por eso **hoy practicamos solo en el Sandbox**.
#
# [sandbox.zenodo.org](https://sandbox.zenodo.org) es una copia de Zenodo **para pruebas**: funciona igual y tiene la misma API, pero sus registros no son permanentes (se limpia cada cierto tiempo) y sus DOI tienen el prefijo **10.5072**, un prefijo de prueba que **no se resuelve en `doi.org`** (en el real el prefijo es 10.5281). Tiene cuentas y tokens **propios**: tu cuenta de zenodo.org no sirve aquí.
#
# Haz un depósito **real** solo cuando tengas un trabajo real que merezca un DOI: un análisis terminado, con datos y código que quieres que se citen. Al final del cuaderno hay una lista para revisar antes de dar ese paso.
#
# ### 3.1 Cuenta y token (si aún no lo hiciste)
#
# 1. Crea una cuenta en [sandbox.zenodo.org](https://sandbox.zenodo.org) (puedes iniciar sesión con GitHub u ORCID).
# 2. Ve a **tu nombre > Applications > Personal access tokens > New token**.
# 3. Ponle un nombre (`taller-colab`) y marca **dos permisos (scopes)**:
#     * `deposit:write` (crear depósitos y subir archivos)
#     * `deposit:actions` (publicar)
# 4. **Create** y copia el valor **ahora**: no se vuelve a mostrar.
# 5. En Colab, panel **Secretos** (icono de llave) > agrega un secreto llamado **`ZENODO_SANDBOX_TOKEN`**, pega el valor y activa **Acceso desde el cuaderno**.
#
# Como con el token de GitHub: **nunca** lo escribas en una celda, ni lo imprimas, ni lo pongas en una URL (`?access_token=...` queda en registros y mensajes de error). Aquí viaja en un encabezado `Authorization` y no se muestra.

# %%
# @title Leer el Secreto ZENODO_SANDBOX_TOKEN (sin mostrarlo)
import os


def leer_secreto(nombre):
    """El valor de un Secreto de Colab (o de una variable de entorno fuera de Colab). None si no existe."""
    try:
        from google.colab import userdata

        valor = userdata.get(nombre)
        if valor:
            return valor
    except Exception:  # fuera de Colab, o el secreto no existe, o no tiene acceso desde el cuaderno
        pass
    return os.environ.get(nombre) or None


TOKEN = leer_secreto("ZENODO_SANDBOX_TOKEN")
if TOKEN is None:
    print("No encuentro el Secreto ZENODO_SANDBOX_TOKEN.\n"
          "  1. En sandbox.zenodo.org crea un token con los permisos deposit:write y deposit:actions.\n"
          "  2. En Colab abre el panel de Secretos (icono de llave) y agrega ZENODO_SANDBOX_TOKEN con ese valor.\n"
          "  3. Activa 'Acceso desde el cuaderno' y vuelve a ejecutar esta celda.\n"
          "Mientras tanto, el cuaderno sigue en modo de ensayo: prepara y valida todo, pero no se conecta a Zenodo con tu cuenta.")
else:
    print(f"Token encontrado ({len(TOKEN)} caracteres). No se muestra.")

# %% [markdown]
# ## 4. Qué vamos a publicar
#
# Un depósito FAIR junta **el código y los datos derivados**, con la documentación necesaria para entenderlos y la procedencia de las fuentes. Subiremos:
#
# 1. **Tu cuaderno** (`.ipynb`): el análisis completo.
# 2. **Tu parquet de datos derivados**: la tabla que produjiste, con sus metadatos dentro del archivo.
#
# Para que el cuaderno funcione para todo el grupo, por defecto usa el cuaderno del bloque 7 y su tabla de **episodios cálidos y fríos de la temperatura de superficie** (`episodios_landsat_concepcion.parquet`; si no la guardaste, se genera aquí en pocos segundos). **Cámbialos por los tuyos**: la ruta de tu cuaderno (el que subiste a GitHub en el bloque 8) y la de tu parquet (por ejemplo, la serie de Sentinel-2 del bloque 4). Si cambias los archivos, cambia también la descripción de la sección 5: debe decir lo que realmente subes.

# %%
# @title Qué archivos subir y quién eres (cámbialos por los tuyos)
import tempfile

import pandas as pd

from utils.extremos import cargar_landsat, controlar_calidad_agua, episodios_por_punto
from utils.series import guardar_series, leer_metadatos_series

CARPETA = Path(tempfile.mkdtemp(prefix="zenodo_"))      # carpeta de trabajo temporal: aquí van las copias
SALIDAS_BLOQUE7 = Path("/content/salidas") if "google.colab" in sys.modules else Path(tempfile.gettempdir()) / "salidas_taller"

RUTA_CUADERNO = RAIZ / "talleres" / "fair-series-temporales" / "07_eventos_extremos.ipynb"   # cambia por tu cuaderno
RUTA_PARQUET = SALIDAS_BLOQUE7 / "episodios_landsat_concepcion.parquet"                        # cambia por tu parquet

if not RUTA_PARQUET.exists():     # no ejecutaste el bloque 7 en esta sesión: lo generamos con las mismas funciones
    episodios = episodios_por_punto(controlar_calidad_agua(cargar_landsat()), tipos=["bahia", "pluma", "golfo", "laguna"])
    RUTA_PARQUET = CARPETA / "episodios_landsat_concepcion.parquet"
    guardar_series(episodios, RUTA_PARQUET, metadatos={
        "titulo": "Episodios cálidos y fríos de la temperatura de superficie en el agua de la bahía de Concepción y alrededores (Landsat 8 y 9, 2013-2026)",
        "fuente": "USGS Landsat 8 y 9 Collection 2 Level-2 leído desde Microsoft Planetary Computer; cinco puntos de agua",
        "licencia": "Dominio público (datos del USGS)",
        "periodo": "2013-03-28 a 2026-10-08",
        "metodo": "Anomalías respecto de una climatología por ventana de días (2013-2024); utils/extremos.py",
        "limitaciones": "Landsat pasa cada 8 días y las nubes borran muchas fechas; no se interpoló",
    })

AUTOR_NOMBRE = "Apellido, Nombre"            # @param {type:"string"}  (formato "Apellido, Nombre")
AUTOR_AFILIACION = "Universidad o institución"  # @param {type:"string"}
AUTOR_ORCID = "0000-0002-1825-0097"          # @param {type:"string"}  (ORCID de ejemplo: reemplázalo por el tuyo)
USUARIO_GITHUB = "tu-usuario"                # @param {type:"string"}
REPO_GITHUB = "analisis-taller"              # @param {type:"string"}

# Trabajamos sobre copias en la carpeta temporal
ARCHIVOS = []
for ruta in (RUTA_CUADERNO, RUTA_PARQUET):
    ruta = Path(ruta)
    if not ruta.exists() and ruta.suffix == ".ipynb" and ruta.with_suffix(".py").exists():
        ruta = ruta.with_suffix(".py")      # si el .ipynb aún no existe, el cuaderno en formato .py
    destino = CARPETA / ruta.name
    if ruta.resolve() != destino.resolve():
        destino.write_bytes(ruta.read_bytes())
    ARCHIVOS.append(destino)
    print(f"{destino.name:42s} {destino.stat().st_size / 1024:8.1f} KB")

# %% [markdown]
# Mira los metadatos que el parquet lleva **dentro del archivo** (los escribió `guardar_series`). Esa es la primera capa de documentación; los metadatos de Zenodo son la segunda, la que ve el buscador.

# %%
meta_parquet = leer_metadatos_series(RUTA_PARQUET)
for clave in ("titulo", "fuente", "licencia", "periodo", "limitaciones"):
    print(f"{clave:9s}: {meta_parquet.get(clave, '(no consta)')}")

# %% [markdown]
# ## 5. Los metadatos mínimos
#
# La API de Zenodo recibe los metadatos como un diccionario JSON. Armamos uno completo. Lee cada campo: es la parte del bloque donde más vale la pena detenerse.
#
# | Campo | Para qué sirve |
# |---|---|
# | `upload_type` | Tipo de recurso (`dataset`, `software`, `publication`, ...) |
# | `title`, `description` | Lo primero que ve quien busca (y lo que lee la IA). La descripción acepta HTML básico |
# | `creators` | Autoría: `name` ("Apellido, Nombre"), `affiliation`, `orcid` |
# | `keywords` | Palabras clave, **en español y en inglés** |
# | `license` | Identificador de licencia (`cc-by-4.0`) |
# | `language` | Idioma del registro, código ISO 639-3 (`spa`) |
# | `locations` | Dónde: lugar y coordenadas, como dato estructurado |
# | `related_identifiers` | Con qué se relaciona: la **fuente** de los datos, el código en GitHub, el artículo |
# | `version`, `publication_date` | Versión y fecha |
#
# ### Licencia
#
# Elegimos **CC BY 4.0**: cualquiera puede reutilizar tu material con la condición de que te cite. Los datos de origen (USGS Landsat) son de dominio público. Si publicas **código**, suele usarse una licencia de software (GPL, MIT); para un registro mixto de cuaderno y datos, CC BY es habitual.
#
# ### Identificadores relacionados
#
# La relación `isDerivedFrom` ("deriva de") apunta a la **fuente**; `cites` al producto Landsat Collection 2 Level-2 del USGS y al artículo que describe su temperatura de superficie (Malakar et al. 2018); `isSupplementedBy` ("está complementado por") al código en GitHub. Así cualquiera puede seguir la cadena de procedencia: de tu dato a su fuente.

# %%
import json
from datetime import date

PERIODO = meta_parquet.get("periodo", "2013-03-01 a 2026-10-13")
TABLA_EPISODIOS = pd.read_parquet(RUTA_PARQUET)
N_CALOR, N_FRIO = int((TABLA_EPISODIOS["tipo"] == "calor").sum()), int((TABLA_EPISODIOS["tipo"] == "frio").sum())


def construir_metadatos(nombre, afiliacion, orcid, usuario_github, repo_github, periodo, n_calor, n_frio):
    """El diccionario de metadatos de Zenodo (API REST de depósitos)."""
    descripcion = (
        "<p>Cuaderno de análisis y tabla de datos derivados del taller <em>Datos FAIR, series temporales satelitales y "
        "detección de anomalías con Python</em>. Contiene los <strong>episodios cálidos y fríos</strong> de la temperatura "
        "de superficie (banda térmica de Landsat 8 y 9, calibrada a °C) en cinco puntos de agua de la bahía de Concepción, el "
        f"golfo de Arauco y la laguna Grande de San Pedro, región del Biobío, Chile: {n_calor} episodios cálidos y {n_frio} fríos, "
        "con su fecha de inicio y fin, duración mínima, número de observaciones e intensidad.</p>"
        f"<p>Periodo: {periodo}. Climatología de referencia 2013-2024 con una ventana de 15 días a cada lado y percentiles "
        "90 y 10. Un episodio son al menos 2 observaciones válidas consecutivas sobre el percentil 90 (o bajo el 10), "
        "separadas por 24 días o menos. No es el método de Hobday et al. (2016), que exige una serie diaria sin huecos: "
        "aquí la duración es solo una cota mínima.</p>"
        "<p>Limitaciones: Landsat pasa cada 8 días con sus dos satélites y las nubes borran muchas fechas (unas decenas "
        "de observaciones válidas al año); no se interpoló. La banda térmica tiene 100 m nativos y la temperatura de "
        "superficie del USGS mide la temperatura de la piel de la superficie (errores de 1 K o más) y no es un producto oceanográfico.</p>"
        "<p>Datos del USGS, de dominio público.</p>"
    )
    return {
        "upload_type": "dataset",
        "title": "Episodios cálidos y fríos de la temperatura de superficie en el agua de la bahía de Concepción y alrededores (Biobío, Chile), Landsat 8 y 9, 2013-2026: tabla y cuaderno",
        "description": descripcion,
        "creators": [{"name": nombre, "affiliation": afiliacion, "orcid": orcid}],
        "keywords": [
            "temperatura de superficie", "episodios extremos", "anomalías", "series temporales", "Concepción", "Biobío",
            "Biobio", "Chile", "Landsat", "teledetección", "FAIR",
            "land surface temperature", "surface temperature", "extreme events", "anomalies", "time series", "remote sensing",
        ],
        "license": "cc-by-4.0",
        "language": "spa",
        "access_right": "open",
        "version": "1.0.0",
        "publication_date": date.today().isoformat(),
        "locations": [
            {"place": "Bahía de Concepción y golfo de Arauco, Región del Biobío, Chile", "lat": -36.8, "lon": -73.1},
        ],
        "related_identifiers": [
            {"identifier": "https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2",
             "relation": "isDerivedFrom", "resource_type": "dataset"},
            {"identifier": "10.5066/P9OGBGM6", "relation": "cites", "resource_type": "dataset"},
            {"identifier": "10.1109/TGRS.2018.2824828", "relation": "cites", "resource_type": "publication-article"},
            {"identifier": f"https://github.com/{usuario_github}/{repo_github}",
             "relation": "isSupplementedBy", "resource_type": "software"},
        ],
    }


metadatos = construir_metadatos(AUTOR_NOMBRE, AUTOR_AFILIACION, AUTOR_ORCID, USUARIO_GITHUB, REPO_GITHUB, PERIODO, N_CALOR, N_FRIO)
print(json.dumps(metadatos, ensure_ascii=False, indent=2)[:2400], "\n...")

# %% [markdown]
# ### Validar antes de enviar
#
# Un depósito mal descrito es peor que uno que no existe, porque ya quedó publicado. Antes de enviar nada, revisamos lo que podemos revisar **sin conectarnos**: campos obligatorios, el formato del **ORCID** (incluido su dígito verificador), palabras clave en los dos idiomas y la licencia.

# %%
import re


def orcid_valido(orcid):
    """Formato 0000-0000-0000-000X y dígito verificador (ISO 7064 mod 11-2)."""
    if not re.fullmatch(r"\d{4}-\d{4}-\d{4}-\d{3}[\dX]", orcid or ""):
        return False
    digitos = orcid.replace("-", "")
    total = 0
    for d in digitos[:-1]:
        total = (total + int(d)) * 2
    resto = total % 11
    verificador = (12 - resto) % 11
    return digitos[-1] == ("X" if verificador == 10 else str(verificador))


def validar_metadatos(m):
    """Devuelve (errores, avisos): lo que impide publicar y lo que conviene mejorar."""
    errores, avisos = [], []
    for campo in ("upload_type", "title", "description", "creators", "license"):
        if not m.get(campo):
            errores.append(f"falta '{campo}'")
    for c in m.get("creators", []):
        if "," not in c.get("name", ""):
            errores.append(f"el nombre '{c.get('name')}' debe tener el formato 'Apellido, Nombre'")
        if not orcid_valido(c.get("orcid")):
            errores.append(f"el ORCID '{c.get('orcid')}' no tiene un formato válido")
    if m.get("license") not in {"cc-by-4.0", "cc-by-sa-4.0", "cc-zero", "mit", "gpl-3.0-or-later", "apache-2.0"}:
        avisos.append(f"la licencia '{m.get('license')}' no es de las habituales: confirma que existe en Zenodo")
    if len(m.get("keywords", [])) < 5:
        avisos.append("menos de 5 palabras clave")
    if not any(re.search(r"[áéíóúñ]", k.lower()) or k.lower() in {"chile", "series temporales"} for k in m.get("keywords", [])):
        avisos.append("no veo palabras clave en español")
    if not any(k.lower() in {"time series", "surface temperature", "land surface temperature", "remote sensing"} for k in m.get("keywords", [])):
        avisos.append("no veo palabras clave en inglés")
    if not m.get("locations"):
        avisos.append("sin 'locations': la búsqueda por lugar no encontrará este registro")
    if not any(r["relation"] == "isDerivedFrom" for r in m.get("related_identifiers", [])):
        avisos.append("sin identificador de la fuente (isDerivedFrom)")
    if AUTOR_ORCID == "0000-0002-1825-0097":
        avisos.append("el ORCID es el de ejemplo (el de la persona ficticia de la documentación de ORCID): reemplázalo por el tuyo")
    if AUTOR_NOMBRE == "Apellido, Nombre":
        avisos.append("el nombre sigue siendo el de ejemplo: escribe el tuyo")
    return errores, avisos


errores, avisos = validar_metadatos(metadatos)
print("Errores (impiden publicar):", errores or "ninguno")
print("Avisos (conviene mejorar):")
for a in avisos:
    print("  -", a)

# %% [markdown]
# ### La prueba de las cuatro preguntas (el vínculo con el bloque 1)
#
# En el bloque 1, una IA tenía que responder preguntas sobre registros y contestaba **"no consta"** cuando faltaba el dato. Hagamos lo mismo con tu registro: ¿puede una persona (o una IA) que **solo** lee tus metadatos responder cada pregunta?

# %%
def preguntas_de_busqueda(m):
    """Qué preguntas típicas de una búsqueda puede responder este registro, solo con sus metadatos."""
    texto = " ".join([m["title"], m["description"], " ".join(m.get("keywords", []))]).lower()
    return {
        "¿Dónde es? (lugar)": bool(m.get("locations")),
        "¿Qué periodo cubre?": bool(re.search(r"20\d\d", m["description"])),
        "¿Qué se midió y con qué producto?": "landsat" in texto and "temperatura de superficie" in texto,
        "¿De dónde vienen los datos? (fuente)": any(r["relation"] == "isDerivedFrom" for r in m.get("related_identifiers", [])),
        "¿Con qué licencia puedo reutilizarlo?": bool(m.get("license")),
        "¿Quién es la autoría y cómo la identifico? (ORCID)": all(c.get("orcid") for c in m["creators"]),
        "¿Aparece buscando 'Biobío' y 'Biobio'?": "biobío" in texto and "biobio" in texto,
        "¿Aparece buscando en inglés?": "surface temperature" in texto or "time series" in texto,
    }


for pregunta, responde in preguntas_de_busqueda(metadatos).items():
    print(f"  {'consta   ' if responde else 'NO CONSTA'}  {pregunta}")

# %% [markdown]
# ### Tu turno 1: mejora tus metadatos
#
# Cada "NO CONSTA" es una pregunta que una persona (o una IA) no podrá responder sobre tu registro. Edita la función `construir_metadatos` o el diccionario `metadatos` para que:
#
# 1. Tus datos de autoría (nombre, ORCID, afiliación) sean los **tuyos**.
# 2. La **descripción** diga qué hay en **tu** parquet (tus puntos, tu periodo, tus variables) y una **limitación** de tus datos, como hicimos aquí con las nubes y la resolución de la banda térmica de Landsat.
# 3. Las **palabras clave** reflejen tu tema (si subes la serie de Sentinel-2: `Sentinel-2`, `NDWI`, `clorofila`, `chlorophyll`).
# 4. El **lugar** y las **fuentes relacionadas** (`related_identifiers`) correspondan a tus datos: Sentinel-2 en Planetary Computer, por ejemplo.
#
# Después vuelve a ejecutar las dos celdas de validación y comprueba que **todas** las preguntas quedan con "consta".

# %% [markdown]
# ## 6. Publicar por la API REST
#
# Publicar un registro en Zenodo es una secuencia de cinco llamadas HTTP:
#
# | Paso | Llamada | Permiso |
# |---|---|---|
# | 1. Crear un depósito vacío (un **borrador**) | `POST /api/deposit/depositions` | `deposit:write` |
# | 2. Subir cada archivo | `PUT {bucket}/{nombre}` | `deposit:write` |
# | 3. Poner los metadatos | `PUT /api/deposit/depositions/{id}` | `deposit:write` |
# | 4. **Publicar** (asigna el DOI) | `POST /api/deposit/depositions/{id}/actions/publish` | `deposit:actions` |
# | 5. Recuperar el registro por su DOI | `GET /api/records/...` (pública) | ninguno |
#
# Los pasos 1 a 3 solo dejan un **borrador**, que puedes borrar sin consecuencias. El paso 4 es el que no se deshace, así que lo separamos y lo protegemos con un interruptor.
#
# El token viaja en el encabezado `Authorization: Bearer ...` y **nunca se imprime**. Si algo falla, mostramos solo el mensaje de error de Zenodo.

# %%
import requests

BASE = "https://sandbox.zenodo.org"      # el Sandbox. El real sería https://zenodo.org (y otro token)


class ZenodoError(RuntimeError):
    pass


def _revisar(respuesta, paso):
    """Lanza un error claro (sin encabezados ni token) si Zenodo no respondió con éxito."""
    if respuesta.ok:
        return respuesta
    try:
        cuerpo = respuesta.json()
        detalle = cuerpo.get("message", "") + " " + "; ".join(
            f"{e.get('field')}: {e.get('message')}" for e in cuerpo.get("errors", []))
    except ValueError:
        detalle = respuesta.text[:200]
    ayuda = {401: "El token no es válido o no corresponde al Sandbox.",
             403: "El token no tiene el permiso necesario (revisa deposit:write y deposit:actions).",
             400: "Zenodo rechazó los datos enviados (revisa los metadatos)."}.get(respuesta.status_code, "")
    raise ZenodoError(f"{paso}: Zenodo respondió {respuesta.status_code}. {detalle.strip()} {ayuda}".strip())


def sesion_zenodo(token):
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    return s


def crear_deposito(s):
    r = _revisar(s.post(f"{BASE}/api/deposit/depositions", json={}, timeout=60), "crear el depósito")
    return r.json()


def subir_archivo(s, deposito, ruta):
    ruta = Path(ruta)
    with ruta.open("rb") as f:
        r = _revisar(s.put(f"{deposito['links']['bucket']}/{ruta.name}", data=f, timeout=600), f"subir {ruta.name}")
    return r.json()


def poner_metadatos(s, deposito, metadatos):
    r = _revisar(s.put(f"{BASE}/api/deposit/depositions/{deposito['id']}", json={"metadata": metadatos}, timeout=60),
                 "poner los metadatos")
    return r.json()


def publicar(s, deposito):
    r = _revisar(s.post(f"{BASE}/api/deposit/depositions/{deposito['id']}/actions/publish", timeout=120), "publicar")
    return r.json()


def borrar_borrador(s, deposito):
    _revisar(s.delete(f"{BASE}/api/deposit/depositions/{deposito['id']}", timeout=60), "borrar el borrador")


# %% [markdown]
# ### Pasos 1 a 3: borrador, archivos y metadatos
#
# Esta celda necesita el token. Si no hay, te lo dice y muestra qué enviaría (modo de ensayo). Si hay errores de validación, no se conecta.

# %%
deposito = None
if errores:
    print("Corrige primero estos errores de los metadatos:", errores)
elif TOKEN is None:
    print("MODO DE ENSAYO: no hay token, así que no me conecto a Zenodo. Esto es lo que se enviaría:\n")
    print(f"  1. POST {BASE}/api/deposit/depositions                  (crear el borrador)")
    for a in ARCHIVOS:
        print(f"  2. PUT  <bucket>/{a.name}   ({a.stat().st_size / 1024:.1f} KB)")
    print(f"  3. PUT  {BASE}/api/deposit/depositions/<id>   con los metadatos de arriba")
else:
    s = sesion_zenodo(TOKEN)
    try:
        deposito = crear_deposito(s)
        print(f"1. Borrador creado: id {deposito['id']}")
        for a in ARCHIVOS:
            subida = subir_archivo(s, deposito, a)
            print(f"2. Subido {subida['key']} ({subida['size'] / 1024:.1f} KB, md5 {subida['checksum'].split(':')[-1]})")
        deposito = poner_metadatos(s, deposito, metadatos)
        print(f"3. Metadatos guardados. DOI reservado: {deposito['metadata'].get('prereserve_doi', {}).get('doi', '(no reservado)')}")
        print(f"\nRevisa el borrador en: {deposito['links']['html']}")
    except ZenodoError as error:
        print("No se pudo completar:", error)

# %% [markdown]
# Abre el enlace del borrador y míralo como lo vería otra persona: título, descripción, palabras clave, ubicaciones, relaciones. Aún no está publicado, así que puedes volver atrás y corregir.
#
# ### Paso 4: publicar
#
# **Esto no se deshace** (ni siquiera en el Sandbox: el registro publicado no se puede borrar, solo se puede publicar una versión nueva). Cuando estés conforme con el borrador, cambia el interruptor a `True` y ejecuta la celda.

# %%
CONFIRMO_PUBLICAR = False   # @param {type:"boolean"}

registro = None
if deposito is None:
    print("No hay borrador (falta el token o la celda anterior no se completó): no hay nada que publicar.")
elif not CONFIRMO_PUBLICAR:
    print("El borrador está listo pero NO se publicó. Revísalo en el enlace de arriba; cuando estés conforme,\n"
          "cambia CONFIRMO_PUBLICAR a True y ejecuta esta celda de nuevo.")
else:
    try:
        registro = publicar(s, deposito)
        print("Publicado.")
        print(f"  DOI:      {registro['doi']}")
        print(f"  Registro: {registro.get('links', {}).get('record_html') or registro.get('doi_url')}")
    except ZenodoError as error:
        print("No se pudo publicar:", error)

# %% [markdown]
# Si te equivocaste en el borrador y quieres empezar de nuevo, esta celda lo borra (solo funciona **antes** de publicar):

# %%
BORRAR_BORRADOR = False   # @param {type:"boolean"}
if BORRAR_BORRADOR and deposito is not None and registro is None:
    borrar_borrador(s, deposito)
    print("Borrador eliminado.")

# %% [markdown]
# ## 7. Recuperar el registro por su DOI
#
# Publicar no termina el trabajo: hay que comprobar que **otra persona, solo con el DOI, puede encontrar y descargar** lo que publicaste. Es la **A** de FAIR (accesible). La verificación hace cuatro cosas:
#
# 1. **El DOI lleva al registro.** En el Sandbox el prefijo es `10.5072` y **no se resuelve en `doi.org`**; el propio Sandbox tiene un resolvedor (`sandbox.zenodo.org/doi/<doi>`). En Zenodo real bastaría `https://doi.org/<doi>`.
# 2. **La búsqueda por DOI lo encuentra** (así lo encontraría un buscador o una IA).
# 3. **Los metadatos recuperados son los que enviaste** (título, autoría, palabras clave, licencia).
# 4. **Los archivos descargados son idénticos a los originales** (misma suma de verificación md5).
#
# Una suma de verificación (*checksum*) es una "huella" del archivo: si un solo byte cambia, la huella cambia. Es la forma de saber que lo descargado es lo subido.

# %%
import hashlib
import time


def md5(ruta_o_bytes):
    contenido = ruta_o_bytes if isinstance(ruta_o_bytes, bytes) else Path(ruta_o_bytes).read_bytes()
    return hashlib.md5(contenido).hexdigest()


def _n_lugares(ubicaciones):
    """Número de lugares del registro (la API los devuelve como lista o como un objeto GeoJSON)."""
    if isinstance(ubicaciones, dict):
        return len(ubicaciones.get("features", []))
    return len(ubicaciones or [])


def verificar_recuperacion(doi, base=BASE, locales=None, intentos=6):
    """
    Recupera un registro de Zenodo por su DOI y comprueba que es accesible. No necesita token (es público).

    locales: lista de archivos locales cuyo md5 debe coincidir con el de los archivos del registro.
    Devuelve el registro (dict) o None si no se pudo recuperar.
    """
    print(f"DOI: {doi}")
    # 1. El resolvedor del propio Zenodo lleva al registro
    r = requests.get(f"{base}/doi/{doi}", timeout=60)
    print(f"1. Resolvedor {base}/doi/...: {'OK, lleva a ' + r.url if r.ok else 'falló (' + str(r.status_code) + ')'}")
    # 2. La búsqueda por DOI lo encuentra (puede tardar unos segundos en indexarse)
    registro = None
    for i in range(intentos):
        b = requests.get(f"{base}/api/records", params={"q": f'doi:"{doi}"'}, timeout=60)
        hits = b.json()["hits"]["hits"] if b.ok else []
        if hits:
            registro = hits[0]
            break
        if i < intentos - 1:
            time.sleep(5)
    print(f"2. Búsqueda por DOI: {'1 registro (id ' + str(registro['id']) + ')' if registro else 'sin resultados (aún no indexado o DOI incorrecto)'}")
    if registro is None:
        return None
    meta = registro["metadata"]
    print(f"3. Metadatos: título «{meta.get('title', '')[:70]}»; licencia {meta.get('license', {}).get('id')}; "
          f"{len(meta.get('creators', []))} autor(es); {len(meta.get('keywords', []))} palabras clave; "
          f"{_n_lugares(meta.get('locations'))} lugar(es)")
    # 4. Los archivos descargados tienen la misma huella
    locales = {Path(p).name: md5(p) for p in (locales or [])}
    for archivo in registro.get("files", []):
        d = requests.get(archivo["links"]["self"], timeout=300)
        huella = md5(d.content)
        esperado = archivo["checksum"].split(":")[-1]
        propio = locales.get(archivo["key"].split("/")[-1])
        estado = "coincide con la huella del registro"
        if huella != esperado:
            estado = "NO coincide con la huella del registro"
        elif propio is not None:
            estado += " y con tu archivo local" if propio == huella else " pero NO con tu archivo local"
        print(f"4. {archivo['key']}: {len(d.content) / 1024:.1f} KB, md5 {huella[:10]}... {estado}")
    return registro


if registro is not None:
    recuperado = verificar_recuperacion(registro["doi"], locales=ARCHIVOS)
else:
    print("Aún no hay un registro tuyo publicado: cuando lo publiques (sección 6), ejecuta esta celda de nuevo.")

# %% [markdown]
# ### Un ejemplo con un registro cualquiera del Sandbox
#
# Para ver la verificación funcionando (con o sin tu registro), la aplicamos a un registro **público cualquiera** del Sandbox con archivos. No es tuyo ni lo elegimos por su contenido: solo sirve para mostrar lo que verías con tu DOI.

# %%
try:
    publicos = requests.get(f"{BASE}/api/records", params={"size": 25, "sort": "mostrecent"}, timeout=60).json()["hits"]["hits"]
    ejemplo = next((h for h in publicos if h.get("files")), None)
    if ejemplo:
        verificar_recuperacion(ejemplo["doi"])
    else:
        print("No encontré un registro público con archivos para el ejemplo.")
except Exception as error:   # sin internet o el Sandbox no responde
    print("No pude consultar el Sandbox para el ejemplo:", type(error).__name__)

# %% [markdown]
# ### Tu turno 2: tu registro, visto desde fuera
#
# Cuando hayas publicado:
#
# 1. Abre el registro en una **ventana de incógnito** (sin tu sesión). ¿Se ve todo? ¿Se pueden descargar los archivos?
# 2. Cópiale el DOI a un compañero o compañera para que ejecute `verificar_recuperacion("10.5072/zenodo.XXXX")` con **tu** DOI en **su** cuaderno. Es la prueba real de accesibilidad.
# 3. Busca tu registro en el Sandbox con una palabra clave en español, otra en inglés y una con y sin tilde (`Biobío` / `Biobio`): ¿aparece en todas?
# 4. En la página del registro, mira la sección de **Citar** y de **Exportar** (DataCite, JSON): eso es la **interoperabilidad**.
#
# **Una sola palabra sobre el DOI.** Cada registro tiene un DOI de su **versión** (el que acabas de obtener) y otro "de concepto" que representa a **todas** las versiones. Si luego mejoras tus datos, publicas una **versión nueva**: el DOI de concepto siempre lleva a la última, y las citas a una versión concreta siguen funcionando. Por eso la versión importa y está en los metadatos.

# %% [markdown]
# ## 8. Antes de publicar en Zenodo real
#
# **Un DOI real es permanente.** El registro no se puede borrar ni "despublicar" salvo en casos excepcionales (y aun así queda una página que dice que existió). Cualquiera, en cualquier parte, puede citarlo. Haz esto solo para trabajo real y revisa antes:
#
# - [ ] ¿Es un trabajo **terminado** y que quieres que se cite? Un borrador o una prueba no merece un DOI.
# - [ ] ¿Tienes **derecho a publicar** los datos y el código? Revisa la licencia de las fuentes y los acuerdos con tu institución o financiamiento.
# - [ ] ¿Los datos contienen información **personal, sensible o protegida** (ubicación de especies amenazadas, sitios sagrados, personas)? Si es así, no se publica tal cual.
# - [ ] ¿Elegiste una **licencia** conscientemente (CC BY, CC0, GPL, MIT) y la citas en el registro?
# - [ ] ¿La **autoría** está completa y con ORCID? Cambiar la autoría después es posible pero queda registrado.
# - [ ] ¿La descripción dice **qué hay, de dónde viene, qué periodo cubre y qué limitaciones tiene**, incluidos los vacíos de datos?
# - [ ] ¿Las **fuentes** están como identificadores relacionados?
# - [ ] ¿Probaste todo en el **Sandbox** primero, con el mismo cuaderno?
# - [ ] ¿Tu **repositorio de GitHub** está en el estado que quieres que quede asociado al DOI? (Zenodo puede archivar automáticamente cada *release* de GitHub: es otra forma de publicar, con la integración GitHub-Zenodo.)
#
# Para publicar en el real, cambia `BASE` a `https://zenodo.org`, usa un token **de zenodo.org** (otro Secreto, por ejemplo `ZENODO_TOKEN`) y el resto del código es el mismo.

# %% [markdown]
# ## Resumen
#
# * Un depósito **FAIR** tiene un **DOI**, **metadatos ricos** (título, descripción, palabras clave en dos idiomas, lugar, relaciones), una **licencia** clara y **procedencia** (la fuente de los datos).
# * Los **metadatos deciden lo que encuentra y entiende una IA**, como viste en el bloque 1: sin lugar, sin palabras clave y sin descripción, el registro responde "no consta".
# * Zenodo se usa por una **API REST** en cinco pasos: crear el borrador, subir archivos, poner metadatos, publicar y verificar. Los tres primeros se pueden deshacer; el cuarto, no.
# * **Practica siempre en el Sandbox** (`sandbox.zenodo.org`, DOI `10.5072`). Un DOI real es permanente: solo para trabajo real.
# * Verifica siempre por el DOI y por la **huella md5** que lo publicado es lo que subiste.
# * Una presentación de 6 minutos tiene **un mensaje, un gráfico bueno, una limitación honesta y un enlace** a tu repositorio y tu DOI.
#
# ### Cierre del taller
#
# En dos días y medio recorriste el flujo completo: encontraste datos con ayuda de la IA, los buscaste con STAC, construiste series temporales de Sentinel-2, las exploraste y las visualizaste, detectaste eventos extremos, versionaste tu trabajo en Git y lo publicaste con un DOI. Cada paso es un eslabón de la misma cadena: **si alguien puede encontrarlo, entenderlo, reproducirlo y citarlo, tu trabajo es FAIR**.
