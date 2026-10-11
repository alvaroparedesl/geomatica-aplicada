"""Usar servidores MCP (Zenodo, Dataverse, OpenAlex...) desde Google Colab, sin programar MCP.

Uso en un cuaderno de Colab (una sola vez por sesión):

    %pip install -q mcp
    from utils.ia import herramientas, usar, preguntar

1. Ver qué ofrece un servidor (no necesita clave ni IA):
       herramientas("zenodo")

2. Llamar tú mismo a una herramienta (tampoco necesita IA):
       usar("zenodo", "zenodo_search_records", query="Biobío", resource_type=["dataset"], size=3)

3. Preguntar en lenguaje natural; Gemini decide qué herramientas llamar y cada
   llamada se muestra en pantalla. Necesita tu propia clave de Google AI Studio,
   guardada en los Secretos de Colab (icono de la llave) con el nombre
   GOOGLE_API_KEY y con "Acceso desde el cuaderno" activado. Fuera de Colab se lee
   de la variable de entorno GOOGLE_API_KEY. La clave nunca se escribe en el cuaderno:
       preguntar("¿Cuántos datasets publicados en 2026 mencionan el Biobío?",
                 servidores=["zenodo", "dataverse"])

Toda la "fontanería" asíncrona de MCP queda oculta en este módulo.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------
# Catálogo de servidores.
#   url / comando : servidor remoto (HTTP) o local (se lanza como subproceso, stdio)
#   tools         : lista blanca de lo que se ofrece al modelo (None = todas)
#   resumir       : función que recorta la respuesta antes de dárselo al modelo
#   prefijo       : prefijo del nombre de herramienta que ve el modelo
#   guia          : párrafo que se añade al prompt de sistema cuando se usa ese servidor
# Estado verificado el 2026-10-09; ver notas en cada entrada.
# --------------------------------------------------------------------------
HOSTS_DATAVERSE = ("dataverse.harvard.edu (internacional), datav.udec.cl (Universidad de Concepción), "
                   "datos.uchile.cl (U. de Chile), datos.usach.cl (USACH), datospararesiliencia.cl (Itrend)")

SERVIDORES: dict[str, dict] = {
    "zenodo": {
        "url": "https://zenodo.caseyjhand.com/mcp",
        "tools": ["zenodo_search_records", "zenodo_get_record", "zenodo_list_files", "zenodo_lookup_vocabulary"],
        "nota": "Comunitario (cyanheads, Apache-2.0), instancia alojada por un tercero. Sin clave. "
                "Devuelve total de resultados y facetas (tipo, año, materias, licencia).",
        "guia": (
            "ZENODO (zenodo_search_records): la cabecera dice 'N matches', ese N es el total real (no cuentes filas). "
            "Filtra con los parámetros, no con texto: resource_type (['publication'] incluye artículos, informes, etc.; "
            "['publication-article'] solo artículos; ['dataset'] datasets) y published_from / published_to (YYYY). "
            "Los términos del query se unen con OR salvo que escribas AND: para una región usa frases entre comillas y "
            "AND, y prueba variantes con y sin tilde (Biobío OR Biobio). Un nombre como 'Concepción' es ambiguo "
            "(Argentina, Paraguay, Colombia, Filipinas...) y 'Universidad de Concepción' es una afiliación, no un lugar. "
            "Solo busca en título, descripción, palabras clave, ubicación y afiliaciones, no dentro de los PDF. "
            "La ubicación geográfica estructurada es rara: pruébala con metadata.locations.features.place:\"Chile\"."),
    },
    "dataverse": {
        "url": "https://mcp.dataverse.org/mcp",  # también hay /sse; el de HTTP en streaming funciona igual
        "tools": ["search_datasets", "get_croissant_record"],  # 'overview' devuelve ~115 KB: se excluye a propósito
        "resumir": "dataverse_gdcc",
        "nota": "Comunidad Dataverse (gdcc/mcp-dataverse, 15 estrellas, sin licencia declarada), alojado en mcp.dataverse.org. "
                "Sin clave. Solo busca datasets; sin filtros ni facetas, pero devuelve total_count.",
        "guia": (
            "DATAVERSE (dataverse_search_datasets): requiere host. Instalaciones disponibles: " + HOSTS_DATAVERSE + ". "
            "Responde siempre 'total_count': úsalo como total (la lista trae como mucho 10). No hay parámetros de "
            "filtro, pero 'query' acepta sintaxis Solr: country:Chile, state:..., city:..., keywordValue:\"...\", "
            "subject:..., publicationDate:2026 (año de la cita del dataset, puede diferir del año en que se publicó en "
            "el repositorio: para esto último usa dateSort:[2026-01-01T00:00:00Z TO *]), unidos con AND. "
            "Sin AND, varios términos se tratan como OR y el total se infla. No uses los caracteres & ni # en query (el servidor corta la consulta ahí). La búsqueda es sensible a la tilde "
            "('Biobío' y 'Biobio' dan resultados distintos). La licencia no viene en la búsqueda: sale en "
            "dataverse_get_croissant_record (campo license). Busca solo en metadatos, no en el contenido de los archivos."),
    },
    "dataverse_local": {
        "comando": [sys.executable, str(Path(__file__).with_name("mcp_dataverse.py"))],
        "prefijo": "dataverse",
        "tools": None,
        "nota": "Servidor propio del curso (utils/mcp_dataverse.py, stdio, sin clave). Misma API pública de Dataverse, "
                "pero con filtros fq, facetas (año, licencia, materia), paginación y lista de campos vacíos.",
        "guia": (
            "DATAVERSE (dataverse_search, dataverse_dataset): dataverse_search devuelve 'Total: N' y facetas sobre todo "
            "el conjunto (año de publicación, licencia, materia): úsalos para contar y resumir en vez de contar filas. "
            "Filtra con fq (['publicationDate:2026', 'country:Chile']) y no con texto libre. En q usa AND y comillas; "
            "campos: title, keywordValue, subject, country, state, city, otherGeographicCoverage. La búsqueda es sensible "
            "a la tilde (prueba Biobío y Biobio). Instalaciones (parámetro host): " + HOSTS_DATAVERSE + ". "
            "dataverse_dataset da la licencia y los CAMPOS VACÍOS de un dataset: si faltan palabras clave o cobertura "
            "geográfica, dilo, porque explica por qué una búsqueda por región no lo encuentra."),
    },
    "openalex": {
        "url": "https://openalex.caseyjhand.com/mcp",
        "tools": None,
        "nota": "Comunitario (cyanheads), instancia alojada por un tercero. Presupuesto diario compartido (cada llamada cuesta ~0,0001 USD). "
                "Es la fuente adecuada para ARTÍCULOS (Zenodo solo contiene una fracción pequeña).",
        "guia": (
            "OPENALEX: es la fuente para contar ARTÍCULOS científicos. Primero openalex_resolve_name para obtener el id de "
            "una institución (p. ej. University of Concepción = I172787465); luego openalex_analyze_trends (group_by "
            "publication_year) o openalex_search_entities con filters {'institutions.id': ..., 'publication_year': '2026', "
            "'type': 'article'}; el total sale en 'Entity Total' / 'result(s)'. Para una REGIÓN, lista las instituciones con "
            "openalex_search_entities(entity_type='institutions', filters={'country_code':'CL','geo.region':'Biobío'}) y "
            "únelas con '|' en institutions.id. Cuenta por afiliación de los autores, no por el tema del artículo."),
    },
    "pangaea": {
        "url": "https://gateway.pipeworx.io/pangaea/mcp",
        # el gateway expone ~45 herramientas de otros dominios (finanzas, etc.): solo ofrecemos las de PANGAEA.
        "tools": ["search", "dataset", "dataset_by_doi", "recent", "facets"],
        "nota": "Terceros (Pipeworx), ~50 llamadas/día por IP. Sin clave. Opcional.",
        "guia": "PANGAEA (ciencias de la Tierra y el mar): úsalo para datos oceanográficos; indica siempre el DOI.",
    },
    "nasa": {
        "url": "https://cmr.earthdata.nasa.gov/mcp/v1",
        "tools": None,
        "nota": "Oficial NASA (catálogo CMR). Sin clave para buscar. Opcional.",
        "guia": "NASA CMR: catálogo de productos satelitales (colecciones, no artículos).",
    },
    "gbif": {
        "url": "https://gbif-biodiversity.caseyjhand.com/mcp",
        "tools": None,
        "nota": "Comunitario (cyanheads), instancia alojada por un tercero. Sin clave. Opcional.",
        "guia": "GBIF: datasets y registros de biodiversidad.",
    },
}

MODELO_POR_DEFECTO = "gemini-3.5-flash"  # gratuito en AI Studio (con límites de uso)
MAX_CARACTERES_RESULTADO = 8000  # lo que se devuelve al modelo por llamada (ahorra cuota)

INSTRUCCIONES_BASE = (
    "Eres un asistente que responde preguntas sobre artículos y datasets de repositorios abiertos usando "
    "herramientas (servidores MCP). Reglas:\n"
    "1. Usa SIEMPRE las herramientas; no respondas de memoria ni inventes DOI, licencias, fechas o cifras.\n"
    "2. Preguntas de 'cuántos': NO cuentes los elementos de la lista que ves. Usa el total que devuelve la herramienta "
    "('N matches', 'total_count', 'Total: N', 'Entity Total') y los filtros de fecha y tipo de la herramienta. "
    "Si el total es mayor que lo que ves, dilo (por ejemplo 'revisé 10 de 377').\n"
    "3. Los términos de búsqueda amplios inflan el total. Usa AND y frases entre comillas, y prueba variantes con y sin "
    "tilde (Biobío/Biobio, Concepción/Concepcion). Los metadatos están casi siempre en inglés: busca también "
    "los términos en inglés (sea surface temperature, no solo temperatura del mar). Si los totales cambian mucho "
    "entre variantes, muéstralos todos.\n"
    "4. Distingue dónde aparece el lugar: en el título, las palabras clave, la ubicación geográfica del registro, "
    "o solo en la afiliación de los autores. Una afiliación no prueba que el dato sea de esa zona.\n"
    "5. Solo ves los metadatos, no el contenido de los archivos ni de los PDF. Si un campo no aparece en la "
    "respuesta de la herramienta escribe 'no consta' (no lo supongas). Si un registro tiene metadatos pobres "
    "(sin descripción, sin palabras clave, sin licencia, sin ubicación), señálalo: puede existir un dato relevante "
    "que la búsqueda no puede encontrar.\n"
    "6. Termina siempre con una línea 'Cómo lo calculé:' que indique herramienta, filtros y consulta usados, "
    "y otra 'Límites:' con lo que pudo quedar fuera (resultados parciales, variantes, metadatos incompletos).\n"
    "7. Responde en español, de forma breve, con DOI y licencia cuando la herramienta los devuelva."
)


def _instrucciones(cfgs: list[dict]) -> str:
    guias = [c["guia"] for c in cfgs if c.get("guia")]
    return INSTRUCCIONES_BASE + ("\n\nHerramientas disponibles:\n" + "\n".join(f"- {g}" for g in guias) if guias else "")


# Compatibilidad: algunos cuadernos antiguos importaban INSTRUCCIONES
INSTRUCCIONES = INSTRUCCIONES_BASE


# --------------------------------------------------------------------------
# Utilidades internas
# --------------------------------------------------------------------------
def _resolver(servidor: str) -> dict:
    """Acepta un alias del catálogo o una URL https://.../mcp directamente."""
    if servidor in SERVIDORES:
        return {"nombre": servidor, **SERVIDORES[servidor]}
    if servidor.startswith(("http://", "https://")):
        return {"nombre": servidor.split("//")[1].split("/")[0].split(".")[0], "url": servidor, "tools": None}
    raise ValueError(f"Servidor desconocido: {servidor!r}. Opciones: {', '.join(SERVIDORES)} o una URL.")


# --------------------------------------------------------------------------
# Recortes: dejan al modelo solo lo útil (y dentro de su cuota gratuita)
# --------------------------------------------------------------------------
def _resumir_dataverse_gdcc(texto: str) -> str:
    """search_datasets de mcp.dataverse.org devuelve el JSON completo de la API (hasta ~130 KB con 10
    resultados). Se queda con el total y con los campos que importan."""
    try:
        d = json.loads(texto)["datasets"]
    except Exception:  # no es la respuesta esperada: se deja tal cual
        return texto
    items = d.get("items", [])
    out = [f"Total: {d.get('total_count')} resultados (q={d.get('q')!r}); se muestran {len(items)} desde el {d.get('start', 0) + 1}."]
    for it in items:
        desc = " ".join((it.get("description") or "").split())[:200]
        out.append(f"- {it.get('global_id')} | {(it.get('published_at') or '')[:10]} | {it.get('name')} | "
                   f"colección: {it.get('name_of_dataverse', '-')} | palabras clave: {', '.join(it.get('keywords') or []) or 'no consta'} | "
                   f"materias: {', '.join(it.get('subjects') or []) or 'no consta'} | {desc}")
    if (d.get("total_count") or 0) > len(items):
        out.append(f"Parcial: faltan {d['total_count'] - len(items)} resultados (la herramienta no permite paginar).")
    return "\n".join(out)


_RESUMIDORES = {"dataverse_gdcc": _resumir_dataverse_gdcc}


def _recortar(cfg: dict, herramienta: str, texto: str) -> str:
    r = cfg.get("resumir")
    if r and herramienta in ("search_datasets",):
        return _RESUMIDORES[r](texto)
    return texto


def _importar_mcp():
    """Importa el cliente MCP tolerando las versiones 1.x y 2.x del paquete `mcp`."""
    try:
        from mcp import ClientSession
    except ImportError as e:  # pragma: no cover
        raise ImportError("Falta el paquete 'mcp'. Ejecuta antes:  !pip install -q mcp") from e
    try:  # mcp 2.x
        from mcp.client.streamable_http import streamable_http_client as http_client
    except ImportError:  # mcp 1.x
        from mcp.client.streamable_http import streamablehttp_client as http_client
    from mcp.client.stdio import StdioServerParameters, stdio_client

    class Sesion(ClientSession):
        # Algunos servidores publican esquemas de salida inválidos y el cliente
        # oficial aborta tras ejecutar la herramienta. Aquí no validamos la salida.
        async def validate_tool_result(self, *a, **k):  # mcp 2.x
            return None

        async def _validate_tool_result(self, *a, **k):  # mcp 1.x
            return None

    return Sesion, http_client, StdioServerParameters, stdio_client


@contextlib.asynccontextmanager
async def _conectar(cfg: dict):
    """Abre una sesión MCP con un servidor remoto (url) o local (comando, por stdio)."""
    Sesion, http_client, Params, stdio_client = _importar_mcp()
    async with contextlib.AsyncExitStack() as stack:
        if cfg.get("comando"):
            comando, *args = cfg["comando"]
            flujos = await stack.enter_async_context(stdio_client(Params(command=comando, args=args)))
        else:
            flujos = await stack.enter_async_context(http_client(cfg["url"]))
        sesion = await stack.enter_async_context(Sesion(flujos[0], flujos[1]))
        await sesion.initialize()
        yield sesion


def _texto(resultado) -> str:
    partes = [getattr(c, "text", None) for c in (resultado.content or [])]
    texto = "\n".join(p for p in partes if p)
    if not texto:
        estructurado = getattr(resultado, "structured_content", None) or getattr(resultado, "structuredContent", None)
        texto = json.dumps(estructurado, ensure_ascii=False) if estructurado is not None else ""
    return texto


def _correr(corutina, salida):
    """Ejecuta una corutina desde código normal, también dentro de Colab/Jupyter
    (donde ya hay un bucle asyncio en marcha)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(corutina)
    try:
        import nest_asyncio  # preinstalado en Colab

        nest_asyncio.apply()
        return asyncio.get_event_loop().run_until_complete(corutina)
    except ImportError:  # plan B: hilo aparte (la salida se imprime al terminar)
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            return pool.submit(asyncio.run, corutina).result()


def _corto(obj, n=300) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + "..."


# --------------------------------------------------------------------------
# 1) y 2) Uso directo, sin IA
# --------------------------------------------------------------------------
def herramientas(servidor: str, mostrar: bool = True) -> list[dict]:
    """Lista las herramientas que ofrece un servidor MCP (nombre, descripción y parámetros)."""
    cfg = _resolver(servidor)

    async def _ir():
        async with _conectar(cfg) as s:
            return (await s.list_tools()).tools

    tools = _correr(_ir(), print)
    out = []
    for t in tools:
        esquema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {}
        out.append({
            "nombre": t.name,
            "descripcion": (t.description or "").strip(),
            "parametros": {k: v.get("type", "?") for k, v in esquema.get("properties", {}).items()},
            "obligatorios": esquema.get("required", []),
        })
    if mostrar:
        print(f"Servidor '{cfg['nombre']}' ({cfg.get('url') or 'local'}): {len(out)} herramientas")
        for t in out:
            print(f"\n- {t['nombre']}\n    {_corto(t['descripcion'], 160)}")
            print("    parametros:", ", ".join(f"{k}{'*' if k in t['obligatorios'] else ''}" for k in t["parametros"]))
        print("\n(* = obligatorio)")
        return None
    return out


def usar(servidor: str, herramienta: str, mostrar: bool = True, **argumentos) -> str:
    """Llama tú mismo a una herramienta MCP y devuelve su respuesta como texto."""
    cfg = _resolver(servidor)

    async def _ir():
        async with _conectar(cfg) as s:
            return await s.call_tool(herramienta, argumentos)

    res = _correr(_ir(), print)
    texto = _texto(res)
    if getattr(res, "is_error", None) or getattr(res, "isError", None):
        print("La herramienta devolvió un error:")
    if mostrar:
        print(texto)
    return texto


# --------------------------------------------------------------------------
# 3) Preguntar en lenguaje natural (LLM + MCP)
# --------------------------------------------------------------------------
def _clave_gemini() -> str | None:
    """Clave de Google AI Studio: Secretos de Colab (GOOGLE_API_KEY) y, si no hay, variable de entorno."""
    try:
        from google.colab import userdata

        clave = userdata.get("GOOGLE_API_KEY")
        if clave:
            return clave
    except Exception:  # fuera de Colab, o el secreto no existe / no tiene acceso desde el cuaderno
        pass
    return os.environ.get("GOOGLE_API_KEY") or None


def _limpiar_esquema(esquema):
    """Quita palabras clave de JSON Schema que la API de Gemini no acepta."""
    if isinstance(esquema, dict):
        return {k: _limpiar_esquema(v) for k, v in esquema.items()
                if k not in ("$schema", "additionalProperties", "$defs", "definitions")}
    if isinstance(esquema, list):
        return [_limpiar_esquema(v) for v in esquema]
    return esquema


class _Gemini:
    """Bucle de function calling con google-genai (clave de AI Studio). Sin llamada automática:
    así se ve cada paso."""

    def __init__(self, modelo, cliente=None):
        from google import genai
        from google.genai import types

        self.types = types
        self.modelo = modelo or MODELO_POR_DEFECTO
        if cliente is None:
            clave = _clave_gemini()
            if not clave:
                raise RuntimeError("SIN_CLAVE")
            cliente = genai.Client(api_key=clave)
        self.cliente = cliente

    async def correr(self, pregunta, tools, llamar, max_pasos, emit, instrucciones=INSTRUCCIONES_BASE):
        t = self.types
        decl = [t.FunctionDeclaration(name=x["name"], description=x["description"][:1000],
                                      parameters_json_schema=_limpiar_esquema(x["schema"])) for x in tools]
        cfg = t.GenerateContentConfig(
            system_instruction=instrucciones,
            tools=[t.Tool(function_declarations=decl)],
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
        )
        contenidos = [t.Content(role="user", parts=[t.Part(text=pregunta)])]
        for paso in range(1, max_pasos + 1):
            resp = self.cliente.models.generate_content(model=self.modelo, contents=contenidos, config=cfg)
            contenido = resp.candidates[0].content
            contenidos.append(contenido)  # se conserva tal cual (firmas de razonamiento de Gemini 3)
            llamadas = [p.function_call for p in (contenido.parts or []) if p.function_call]
            if not llamadas:
                return "".join(p.text for p in (contenido.parts or []) if p.text)
            partes = []
            for fc in llamadas:
                salida = await llamar(fc.name, dict(fc.args or {}), paso)
                partes.append(t.Part.from_function_response(name=fc.name, response={"resultado": salida}))
            contenidos.append(t.Content(role="user", parts=partes))
        return "(se alcanzó el máximo de pasos sin respuesta final)"


def preguntar(pregunta: str, servidores=("zenodo", "dataverse"), modelo: str | None = None,
              max_pasos: int = 10, devolver: bool = False, _backend=None) -> str | None:
    """Hace una pregunta a Gemini conectado a uno o varios servidores MCP.

    Imprime cada herramienta que el modelo decide llamar, con sus argumentos,
    y al final la respuesta. Con devolver=True también la devuelve como texto
    (por defecto no, para que Colab no la muestre dos veces).

    servidores: alias del catálogo `SERVIDORES` (o una URL https://.../mcp).
    modelo: modelo de Gemini; por defecto `MODELO_POR_DEFECTO`.
    max_pasos: máximo de rondas de llamadas a herramientas antes de rendirse.
    """
    if isinstance(servidores, str):
        servidores = [servidores]
    cfgs = [_resolver(s) for s in servidores]

    try:
        if _backend is None:
            _backend = _Gemini(modelo)
    except RuntimeError as e:
        if str(e) != "SIN_CLAVE":
            raise
        print("No hay clave de Gemini. Guarda tu clave de Google AI Studio en Colab:\n"
              "  icono de la llave (Secretos) -> 'Agregar secreto' con el nombre GOOGLE_API_KEY\n"
              "  y activa 'Acceso desde el cuaderno'.\n"
              "Mientras tanto puedes usar los servidores a mano: herramientas('zenodo'), usar('zenodo', ...).")
        return None

    llamadas: list[str] = []  # para el resumen final

    async def _ir():
        async with contextlib.AsyncExitStack() as stack:
            sesiones: dict[str, tuple] = {}  # nombre expuesto al modelo -> (sesion, nombre real, cfg)
            tools = []
            activos = []
            for cfg in cfgs:
                try:
                    s = await stack.enter_async_context(_conectar(cfg))
                    lista = (await s.list_tools()).tools
                except Exception as e:  # un servidor caído no debe tumbar la pregunta
                    print(f"AVISO: no pude conectar con '{cfg['nombre']}' ({type(e).__name__}); sigo sin él.")
                    continue
                activos.append(cfg)
                nombres = {t.name for t in lista}
                for falta in set(cfg["tools"] or []) - nombres:
                    print(f"AVISO: '{cfg['nombre']}' ya no ofrece la herramienta '{falta}'.")
                for t in lista:
                    if cfg["tools"] is not None and t.name not in cfg["tools"]:
                        continue
                    pre = cfg.get("prefijo", cfg["nombre"])
                    expuesto = t.name if t.name.startswith(pre) else f"{pre}_{t.name}"
                    esquema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {"type": "object"}
                    sesiones[expuesto] = (s, t.name, cfg)
                    tools.append({"name": expuesto, "description": f"[{cfg['nombre']}] {t.description or ''}",
                                  "schema": esquema})
            if not tools:
                return "No hay ningún servidor disponible; prueba de nuevo en unos minutos."
            print(f"Pregunta: {pregunta}")
            print(f"Servidores: {', '.join(c['nombre'] for c in activos)} | herramientas ofrecidas al modelo: "
                  f"{len(tools)} ({', '.join(t['name'] for t in tools)})\n")

            async def llamar(nombre, args, paso):
                if nombre not in sesiones:
                    return f"Error: herramienta desconocida {nombre}"
                s, real, cfg = sesiones[nombre]
                print(f"[paso {paso}] El modelo llama a  {cfg['nombre']}.{real}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
                llamadas.append(cfg["nombre"])
                t0 = time.time()
                texto = ""
                for intento in (1, 2):  # un reintento si hay límite de tasa o fallo transitorio
                    try:
                        res = await s.call_tool(real, args)
                        texto = _texto(res)
                        if getattr(res, "is_error", None) or getattr(res, "isError", None):
                            texto = "ERROR: " + texto
                    except Exception as e:  # el modelo puede corregir y reintentar
                        texto = f"ERROR: {type(e).__name__}: {e}"
                    if intento == 1 and texto.startswith("ERROR") and any(x in texto for x in ("429", "rate limit", "timeout", "503")):
                        print("          (límite de tasa o fallo transitorio: espero 5 s y reintento)")
                        await asyncio.sleep(5)
                        continue
                    break
                bruto = len(texto)
                texto = _recortar(cfg, real, texto)
                nota = f", resumido a {len(texto)}" if len(texto) != bruto else ""
                print(f"          <- {bruto} caracteres{nota} en {time.time() - t0:.1f} s: {_corto(texto, 160)}")
                if len(texto) > MAX_CARACTERES_RESULTADO:
                    print(f"          (se recorta a {MAX_CARACTERES_RESULTADO} caracteres para el modelo)")
                    texto = texto[:MAX_CARACTERES_RESULTADO] + "\n[...recortado: respuesta parcial...]"
                return texto

            return await _backend.correr(pregunta, tools, llamar, max_pasos, print, _instrucciones(activos))

    try:
        respuesta = _correr(_ir(), print)
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            print("Cuota gratuita agotada (error 429). Espera un minuto o reduce la pregunta; "
                  "el límite diario se reinicia a medianoche (hora del Pacífico).")
            return None
        raise
    print("\n=== Respuesta ===\n" + (respuesta or ""))
    if llamadas:
        cuenta = {n: llamadas.count(n) for n in dict.fromkeys(llamadas)}
        print(f"\n[{len(llamadas)} llamadas a herramientas: " + ", ".join(f"{n} x{c}" for n, c in cuenta.items()) + "]")
    return respuesta if devolver else None
