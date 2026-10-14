"""
Pruebas de `utils/ia.py` y `utils/mcp_dataverse.py` (servidores MCP desde Colab).

Qué demuestran:

* Offline (sin internet ni clave de Gemini): el catálogo de servidores y el
  prompt de sistema, la lectura de la clave (Secretos de Colab y variable de
  entorno), `herramientas` y `usar` contra un servidor MCP falso que corre como
  subproceso por stdio, y `preguntar` con un "LLM" guionizado (se simula lo que
  decidiría el modelo; el bucle de herramientas, el recorte de respuestas y el
  manejo de errores son los reales). Sin el paquete `mcp` o `google-genai` estas
  pruebas se omiten.
* Con red (marca `red`): los servidores reales (Zenodo, Dataverse, OpenAlex) y el
  servidor local `dataverse_local` contra datav.udec.cl. Los totales cambian con
  el tiempo; se comprueban formatos y cotas, no cifras exactas.
"""

import asyncio
import json
import re
import sys
import types as pytipos
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp")
genai_tipos = pytest.importorskip("google.genai.types")

from utils import ia  # noqa: E402

FALSO = Path(__file__).parent / "fixtures" / "servidor_mcp_falso.py"
VOSEO = re.compile(r"\b(tenés|podés|corré|saltás|preferís|vos|querés|sabés|hacé|mirá|fijate|usá|ponés|elegí)\b", re.I)


@pytest.fixture
def servidor_falso(monkeypatch):
    """Registra el alias "falso": un servidor MCP local por stdio, sin red."""
    monkeypatch.setitem(ia.SERVIDORES, "falso", {
        "comando": [sys.executable, str(FALSO)],
        "tools": ["falso_buscar", "falso_licencia", "falso_error"],
        "nota": "Servidor de prueba.",
        "guia": "FALSO: servidor de prueba con totales fijos.",
    })
    return "falso"


# Un servidor que muere al arrancar (el equivalente rápido de un servidor caído: una URL
# sin respuesta tarda unos 30 s en rendirse).
CAIDO = {"comando": [sys.executable, "-c", "raise SystemExit(1)"], "tools": None, "nota": "x"}


class Guion:
    """'LLM' simulado: en cada ronda devuelve las llamadas guionizadas; al final, el texto."""

    def __init__(self, pasos, final="Respuesta final simulada"):
        self.pasos, self.final, self.n = pasos, final, 0
        self.vistas, self.sistema, self.ofrecidas = [], None, []

    def generate_content(self, model, contents, config):
        t = genai_tipos
        self.sistema = config.system_instruction
        self.ofrecidas = [d.name for d in config.tools[0].function_declarations]
        if contents[-1].parts and contents[-1].parts[0].function_response:
            self.vistas += [p.function_response.response["resultado"] for p in contents[-1].parts]
        if self.n < len(self.pasos):
            partes = [t.Part.from_function_call(name=n, args=a) for n, a in self.pasos[self.n]]
            self.n += 1
        else:
            partes = [t.Part(text=self.final)]
        contenido = t.Content(role="model", parts=partes)
        return pytipos.SimpleNamespace(candidates=[pytipos.SimpleNamespace(content=contenido)])


def correr(guion, servidores, **kw):
    """Llama a `preguntar` con el LLM simulado, dentro de un bucle asyncio en marcha (como en Colab)."""
    backend = ia._Gemini("gemini-3.5-flash", cliente=pytipos.SimpleNamespace(models=guion))

    async def en_bucle():
        return ia.preguntar("pregunta de prueba", servidores, _backend=backend, devolver=True, **kw)

    return asyncio.run(en_bucle())


# ---------------------------------------------------------------------------
# Catálogo, prompt y texto
# ---------------------------------------------------------------------------
def test_catalogo_y_alias():
    for nombre in ("zenodo", "dataverse", "dataverse_local", "openalex"):
        assert ia._resolver(nombre)["nombre"] == nombre
    assert ia._resolver("https://ejemplo.org/mcp")["url"] == "https://ejemplo.org/mcp"
    with pytest.raises(ValueError, match="desconocido"):
        ia._resolver("inventado")


def test_prompt_incluye_solo_las_guias_de_los_servidores_usados():
    p = ia._instrucciones([ia._resolver("zenodo"), ia._resolver("dataverse")])
    assert "N matches" in p and "total_count" in p and "Límites:" in p and "dataverse_search_datasets" in p
    assert "ZENODO" in p and "DATAVERSE" in p and "OPENALEX" not in p


def test_textos_sin_voseo_ni_rayas_largas():
    for modulo in ("ia.py", "mcp_dataverse.py"):
        fuente = (Path(ia.__file__).parent / modulo).read_text(encoding="utf-8")
        assert not VOSEO.search(fuente), VOSEO.search(fuente)
        assert chr(0x2014) not in fuente  # raya larga: el hook del repositorio también la prohíbe


def test_limpiar_esquema_quita_palabras_que_gemini_no_acepta():
    esquema = {"$schema": "x", "type": "object", "additionalProperties": False,
               "properties": {"a": {"type": "array", "items": {"type": "string", "additionalProperties": True}}}}
    assert ia._limpiar_esquema(esquema) == {"type": "object",
                                            "properties": {"a": {"type": "array", "items": {"type": "string"}}}}


# ---------------------------------------------------------------------------
# Clave de Gemini
# ---------------------------------------------------------------------------
def _colab_falso(monkeypatch, secretos):
    """Simula `google.colab.userdata` con un diccionario de secretos."""
    def get(nombre):
        if nombre not in secretos:
            raise KeyError(nombre)
        return secretos[nombre]

    monkeypatch.setitem(sys.modules, "google.colab", pytipos.SimpleNamespace(userdata=pytipos.SimpleNamespace(get=get)))
    monkeypatch.setitem(sys.modules, "google.colab.userdata", pytipos.SimpleNamespace(get=get))


def test_clave_desde_secretos_de_colab_antes_que_el_entorno(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "del-entorno")
    _colab_falso(monkeypatch, {"GOOGLE_API_KEY": "de-colab"})
    assert ia._clave_gemini() == "de-colab"


def test_clave_desde_el_entorno_si_no_hay_secreto(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "del-entorno")
    _colab_falso(monkeypatch, {})
    assert ia._clave_gemini() == "del-entorno"
    monkeypatch.setitem(sys.modules, "google.colab", None)  # fuera de Colab: el import falla
    assert ia._clave_gemini() == "del-entorno"


def test_sin_clave_avisa_y_no_falla(monkeypatch, capsys):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "google.colab", None)
    assert ia.preguntar("x") is None
    salida = capsys.readouterr().out
    assert "GOOGLE_API_KEY" in salida and "herramientas('zenodo')" in salida


# ---------------------------------------------------------------------------
# Recorte de la respuesta de Dataverse (mcp.dataverse.org)
# ---------------------------------------------------------------------------
def _respuesta_gdcc(n=2, total=2):
    items = [{"global_id": f"doi:10.1/{i}", "published_at": "2026-03-01T00:00:00Z", "name": f"Dataset {i}",
              "name_of_dataverse": "Colección", "keywords": [], "subjects": ["Earth and Environmental Sciences"],
              "description": "x " * 500, "relleno": "y" * 5000} for i in range(n)]
    return json.dumps({"datasets": {"total_count": total, "q": "Arauco", "start": 0, "items": items}})


def test_recorte_gdcc_deja_total_y_campos_utiles():
    cfg = ia._resolver("dataverse")
    bruto = _respuesta_gdcc(n=2, total=25)
    corto = ia._recortar(cfg, "search_datasets", bruto)
    assert corto.startswith("Total: 25 resultados") and len(corto) < len(bruto) / 5
    assert "palabras clave: no consta" in corto and "Parcial: faltan 23" in corto
    assert ia._recortar(cfg, "get_croissant_record", bruto) == bruto  # otras herramientas no se tocan
    assert ia._recortar(cfg, "search_datasets", "no es JSON") == "no es JSON"


# ---------------------------------------------------------------------------
# herramientas, usar y preguntar contra un servidor MCP falso (stdio)
# ---------------------------------------------------------------------------
def test_herramientas_lista_nombre_y_parametros(servidor_falso):
    lista = ia.herramientas(servidor_falso, mostrar=False)
    por_nombre = {t["nombre"]: t for t in lista}
    assert set(por_nombre) == {"falso_buscar", "falso_licencia", "falso_error"}
    assert por_nombre["falso_buscar"]["obligatorios"] == ["query"]
    assert set(por_nombre["falso_buscar"]["parametros"]) == {"query", "size"}


def test_herramientas_imprime_resumen(servidor_falso, capsys):
    assert ia.herramientas(servidor_falso) is None
    salida = capsys.readouterr().out
    assert "3 herramientas" in salida and "falso_buscar" in salida and "(* = obligatorio)" in salida


def test_usar_llama_a_la_herramienta(servidor_falso, capsys):
    texto = ia.usar(servidor_falso, "falso_buscar", query="Biobío", size=2)
    assert texto.startswith("Total: 3 resultados para 'Biobío' (size=2)")
    assert texto in capsys.readouterr().out
    assert ia.usar(servidor_falso, "falso_licencia", mostrar=False, doi="10.1/x") == "Licencia de 10.1/x: CC-BY-4.0"


def test_preguntar_muestra_los_pasos_y_devuelve_la_respuesta(servidor_falso, capsys):
    g = Guion([[("falso_buscar", {"query": "Biobío"})], [("falso_licencia", {"doi": "10.1/x"})]])
    assert correr(g, [servidor_falso]) == "Respuesta final simulada"
    assert g.ofrecidas == ["falso_buscar", "falso_licencia", "falso_error"]
    assert g.vistas[0].startswith("Total: 3 ") and g.vistas[1] == "Licencia de 10.1/x: CC-BY-4.0"
    assert "FALSO" in g.sistema and "Cómo lo calculé:" in g.sistema
    salida = capsys.readouterr().out
    assert "[paso 1] El modelo llama a  falso.falso_buscar(query='Biobío')" in salida
    assert "[paso 2]" in salida and "=== Respuesta ===" in salida
    assert "[2 llamadas a herramientas: falso x2]" in salida


def test_preguntar_sin_devolver_no_repite_la_respuesta(servidor_falso):
    backend = ia._Gemini("m", cliente=pytipos.SimpleNamespace(models=Guion([])))
    assert ia.preguntar("p", [servidor_falso], _backend=backend) is None


def test_un_error_de_la_herramienta_llega_al_modelo_como_texto(servidor_falso):
    g = Guion([[("falso_error", {})], [("falso_inexistente", {})]])
    assert correr(g, [servidor_falso]) == "Respuesta final simulada"
    assert g.vistas[0].startswith("ERROR")
    assert g.vistas[1].startswith("Error: herramienta desconocida")


def test_servidor_caido_no_tumba_la_pregunta(servidor_falso, monkeypatch, capsys):
    monkeypatch.setitem(ia.SERVIDORES, "roto", CAIDO)
    g = Guion([[("falso_buscar", {"query": "x"})]])
    assert correr(g, ["roto", servidor_falso]) == "Respuesta final simulada"
    salida = capsys.readouterr().out
    assert "AVISO: no pude conectar con 'roto'" in salida and "Servidores: falso" in salida


def test_sin_ningun_servidor_responde_sin_llamar_al_modelo(monkeypatch):
    monkeypatch.setitem(ia.SERVIDORES, "roto", CAIDO)
    g = Guion([])
    assert "No hay ningún servidor disponible" in correr(g, ["roto"])
    assert g.n == 0 and g.sistema is None


def test_recorta_respuestas_largas_para_el_modelo(servidor_falso, monkeypatch):
    monkeypatch.setattr(ia, "MAX_CARACTERES_RESULTADO", 20)
    g = Guion([[("falso_buscar", {"query": "x"})]])
    correr(g, [servidor_falso])
    assert g.vistas[0].endswith("[...recortado: respuesta parcial...]") and len(g.vistas[0]) < 80


def test_max_pasos(servidor_falso):
    g = Guion([[("falso_buscar", {"query": "x"})]] * 5)
    assert "máximo de pasos" in correr(g, [servidor_falso], max_pasos=2)
    assert g.n == 2


class GuionQueFalla(Guion):
    """'LLM' simulado cuya llamada a la API falla, como una cuota agotada o un modelo inexistente."""

    def __init__(self, error):
        super().__init__([])
        self.error = error

    def generate_content(self, model, contents, config):
        raise self.error


class GuionSaturado(Guion):
    """'LLM' simulado que responde 503 (servicio saturado) las primeras `fallos` veces."""

    def __init__(self, fallos, final="Respuesta tras reintentar"):
        super().__init__([], final=final)
        self.fallos, self.intentos = fallos, 0

    def generate_content(self, model, contents, config):
        self.intentos += 1
        if self.intentos <= self.fallos:
            from google.genai import errors

            raise errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE",
                                                     "message": "The model is overloaded. Please try again later."}})
        return super().generate_content(model, contents, config)


def test_gemini_saturado_se_reintenta_y_responde(servidor_falso, monkeypatch, capsys):
    monkeypatch.setattr(ia, "ESPERAS_REINTENTO", (0, 0, 0))
    g = GuionSaturado(fallos=2)
    assert correr(g, [servidor_falso]) == "Respuesta tras reintentar"
    assert g.intentos == 3 and capsys.readouterr().out.count("Gemini está saturado") == 2


def test_gemini_saturado_para_siempre_da_un_mensaje_claro(servidor_falso, monkeypatch, capsys):
    monkeypatch.setattr(ia, "ESPERAS_REINTENTO", (0, 0))
    g = GuionSaturado(fallos=99)
    assert correr(g, [servidor_falso]) is None
    assert g.intentos == 3  # el intento inicial y dos reintentos
    assert "Gemini sigue saturado" in capsys.readouterr().out


def test_un_error_que_no_es_pasajero_no_se_reintenta(servidor_falso, monkeypatch):
    monkeypatch.setattr(ia, "ESPERAS_REINTENTO", (0, 0))
    g = GuionQueFalla(RuntimeError("404 modelo inexistente"))
    with pytest.raises(RuntimeError, match="404"):
        correr(g, [servidor_falso])


def test_error_de_gemini_se_desenvuelve_en_vez_de_un_exceptiongroup(servidor_falso):
    # El error ocurre con la sesión MCP abierta: anyio lo envuelve en un ExceptionGroup que no dice nada.
    with pytest.raises(RuntimeError, match="404 modelo inexistente") as info:
        correr(GuionQueFalla(RuntimeError("404 modelo inexistente")), [servidor_falso])
    assert not isinstance(info.value, BaseExceptionGroup)


def test_cuota_agotada_de_gemini_se_reconoce_aunque_venga_envuelta(servidor_falso, capsys):
    g = GuionQueFalla(RuntimeError("429 RESOURCE_EXHAUSTED: cuota"))
    assert correr(g, [servidor_falso]) is None
    assert "Cuota gratuita agotada" in capsys.readouterr().out


def test_mensaje_de_error_muestra_la_causa_real_de_un_grupo():
    grupo = ExceptionGroup("unhandled errors in a TaskGroup", [ValueError("uno"), ExceptionGroup("g", [KeyError("dos")])])
    texto = ia.mensaje_de_error(grupo)
    assert "ValueError: uno" in texto and "KeyError" in texto and "TaskGroup" not in texto
    assert ia.mensaje_de_error(RuntimeError("solo")) == "RuntimeError: solo"


def test_servidor_local_funciona_aunque_stderr_no_sea_un_archivo(servidor_falso):
    # En Jupyter y Colab sys.stderr no tiene descriptor de archivo (fileno() lanza UnsupportedOperation) y el
    # cliente MCP toma sys.stderr al importarse. Se reproduce en un intérprete nuevo que lo cambia antes de importar.
    import subprocess

    codigo = (
        "import io, sys\n"
        "sys.stderr = io.StringIO()\n"
        f"sys.path.insert(0, {str(Path(ia.__file__).parent.parent)!r})\n"
        "from utils import ia\n"
        f"ia.SERVIDORES['falso'] = {{'comando': [sys.executable, {str(FALSO)!r}], 'tools': None, 'nota': 'x'}}\n"
        "print(ia.usar('falso', 'falso_buscar', mostrar=False, query='x'))\n"
    )
    r = subprocess.run([sys.executable, "-I", "-c", codigo], capture_output=True, text=True, timeout=60)
    assert "Total: 3" in r.stdout, r.stdout + r.stderr


# ---------------------------------------------------------------------------
# Con red: servidores reales (marca `red`)
# ---------------------------------------------------------------------------
@pytest.mark.red
def test_red_zenodo_devuelve_el_total():
    texto = ia.usar("zenodo", "zenodo_search_records", mostrar=False, query="Biobío OR Biobio",
                    resource_type=["publication"], published_from="2026", published_to="2026", size=1)
    assert int(re.search(r"Zenodo search: (\d+) matches", texto).group(1)) > 0


@pytest.mark.red
def test_red_dataverse_local_udec():
    texto = ia.usar("dataverse_local", "dataverse_search", mostrar=False, q="*", host="datav.udec.cl",
                    fq=["publicationDate:2026"])
    assert re.match(r"Total: \d+ resultados en datav.udec.cl", texto), texto[:100]
    assert "Facetas" in texto


@pytest.mark.red
def test_red_dataverse_dataset_informa_campos_vacios():
    texto = ia.usar("dataverse_local", "dataverse_dataset", mostrar=False,
                    persistent_id="doi:10.48665/udec/FJBASE", host="datav.udec.cl")
    assert "CAMPOS VACÍOS:" in texto and "Licencia:" in texto


@pytest.mark.red
def test_red_dataverse_gdcc_se_recorta():
    bruto = ia.usar("dataverse", "search_datasets", mostrar=False, host="datav.udec.cl", query="publicationDate:2026")
    corto = ia._recortar(ia._resolver("dataverse"), "search_datasets", bruto)
    assert corto.startswith("Total: ") and len(corto) < len(bruto) / 3


@pytest.mark.red
def test_red_openalex_resuelve_una_institucion():
    texto = ia.usar("openalex", "openalex_search_entities", mostrar=False, entity_type="institutions",
                    filters={"country_code": "CL", "geo.region": "Biobío"}, per_page=12)
    assert "University of Concepción" in texto


@pytest.mark.red
def test_red_preguntar_con_llm_guionizado_sobre_servidores_reales():
    g = Guion([[("zenodo_search_records", {"query": "Biobío OR Biobio", "size": 1}),
                ("dataverse_search_datasets", {"host": "datav.udec.cl", "query": "Arauco"})]])
    assert correr(g, ["zenodo", "dataverse"]) == "Respuesta final simulada"
    assert "matches" in g.vistas[0] and g.vistas[1].startswith("Total: ")
