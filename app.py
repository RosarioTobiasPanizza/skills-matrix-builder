"""
Skills Matrix Builder
=====================
Aplicación Streamlit que ayuda al equipo de RR.HH. a construir, con IA generativa
(API de Claude), una matriz de habilidades por rol con niveles de madurez por
seniority y, a partir de ella:
  - comparar dos niveles (base de un plan de carrera)
  - generar una descripción de puesto

Principio de diseño: la IA propone, la persona revisa, edita y aprueba cada paso.
La app trabaja sobre ROLES, nunca sobre personas, y no puntúa el desempeño.
"""

from __future__ import annotations

import io
import json
import re
from typing import Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pydantic import BaseModel, Field, ValidationError

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Skills Matrix Builder", page_icon="🧭", layout="wide")

DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_LEVELS = ["Junior", "Semi Senior", "Senior", "Lead"]
MATURITY_LABELS = {1: "Inicial", 2: "En desarrollo", 3: "Competente", 4: "Avanzado", 5: "Experto"}
TABS = ["🏠 Inicio", "1. Rol y skills", "2. Matriz", "3. Comparar niveles", "4. Descripción de puesto"]
DISCLAIMER = (
    "Los resultados son propuestas generadas por IA y deben ser revisados por RR.HH. "
    "antes de usarse."
)


def get_secret(name: str, default: str | None = None) -> str | None:
    """Lee un valor de st.secrets sin romper si no existe secrets.toml."""
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Esquemas de salida de la IA (validados con Pydantic)
# ---------------------------------------------------------------------------


class Skill(BaseModel):
    name: str = Field(min_length=2)
    type: Literal["Técnica", "Conductual"]
    definition: str = Field(min_length=5)


class SkillList(BaseModel):
    skills: list[Skill] = Field(min_length=3, max_length=12)


class LevelCell(BaseModel):
    level: str
    maturity: int = Field(ge=1, le=5)
    descriptor: str = Field(min_length=5)


class MatrixRow(BaseModel):
    skill: str
    levels: list[LevelCell]


class Matrix(BaseModel):
    rows: list[MatrixRow]


class GapItem(BaseModel):
    skill: str
    what_changes: str
    evidence: list[str]
    development_actions: list[str]


class Comparison(BaseModel):
    summary: str
    items: list[GapItem]


class JobDescription(BaseModel):
    title: str
    purpose: str
    responsibilities: list[str]
    requirements: list[str]
    we_offer: list[str]


# ---------------------------------------------------------------------------
# Datos del modo demo (no usan la API)
# ---------------------------------------------------------------------------

DEMO_CONFIG = {
    "role": "Customer Success Manager",
    "area": "Customer Success",
    "industry": "Software B2B (SaaS)",
    "company_size": "51-200 empleados",
    "framework": "Valores: foco en el cliente, ownership, transparencia.",
    "levels": DEFAULT_LEVELS,
    "n_skills": 8,
    "tech_share": 50,
    "language": "Español",
}

DEMO_SKILLS = [
    ("Gestión de cartera y renovaciones", "Técnica", "Administra cuentas para asegurar renovación y expansión."),
    ("Onboarding de clientes", "Técnica", "Lleva al cliente desde la firma hasta el primer valor obtenido."),
    ("Análisis de salud del cliente", "Técnica", "Usa datos de uso y señales para anticipar riesgo y oportunidad."),
    ("Conocimiento del producto", "Técnica", "Domina el producto para recomendar el uso que genera valor."),
    ("Comunicación con stakeholders", "Conductual", "Adapta el mensaje a cada interlocutor, del usuario al C-level."),
    ("Negociación y manejo de objeciones", "Conductual", "Sostiene conversaciones difíciles cuidando la relación."),
    ("Resolución de problemas", "Conductual", "Diagnostica causas y resuelve con autonomía creciente."),
    ("Colaboración interfuncional", "Conductual", "Coordina con Producto, Ventas y Soporte en favor del cliente."),
]

# skill -> [(madurez, descriptor) por nivel: Junior, Semi Senior, Senior, Lead]
DEMO_MATRIX = {
    "Gestión de cartera y renovaciones": [
        (2, "Gestiona una cartera pequeña con acompañamiento y sigue el playbook de renovación."),
        (3, "Gestiona su cartera de forma autónoma y anticipa renovaciones con 90 días."),
        (4, "Gestiona cuentas estratégicas y detecta oportunidades de expansión con Ventas."),
        (5, "Define la estrategia de cartera del equipo y es responsable del NRR del segmento."),
    ],
    "Onboarding de clientes": [
        (2, "Ejecuta onboardings estándar siguiendo el checklist definido."),
        (3, "Lidera onboardings completos y adapta el plan a las necesidades del cliente."),
        (4, "Gestiona onboardings complejos o multi-área y reduce el tiempo al primer valor."),
        (5, "Diseña y mejora el proceso de onboarding y lo mide con indicadores."),
    ],
    "Análisis de salud del cliente": [
        (1, "Consulta el health score y reporta alertas a su referente."),
        (3, "Interpreta datos de uso y define acciones para cuentas en riesgo."),
        (4, "Cruza varias fuentes de datos y prioriza la cartera según riesgo y potencial."),
        (5, "Define los indicadores de salud del cliente y su modelo de alertas."),
    ],
    "Conocimiento del producto": [
        (2, "Conoce las funcionalidades principales y resuelve consultas frecuentes."),
        (3, "Recomienda configuraciones y buenas prácticas según el caso de uso."),
        (4, "Traduce necesidades del cliente en propuestas de producto fundamentadas."),
        (4, "Es referente del equipo y canaliza el feedback estratégico hacia Producto."),
    ],
    "Comunicación con stakeholders": [
        (2, "Se comunica con claridad con usuarios operativos, por escrito y en reuniones."),
        (3, "Conduce reuniones de seguimiento con responsables del cliente."),
        (4, "Presenta resultados y business reviews a niveles directivos."),
        (5, "Gestiona relaciones ejecutivas y representa a la empresa en cuentas clave."),
    ],
    "Negociación y manejo de objeciones": [
        (1, "Identifica objeciones y las escala con contexto claro."),
        (2, "Responde objeciones frecuentes con argumentos preparados."),
        (4, "Negocia renovaciones y condiciones comerciales dentro de su mandato."),
        (5, "Lidera negociaciones complejas y forma al equipo en técnicas de negociación."),
    ],
    "Resolución de problemas": [
        (2, "Resuelve problemas conocidos y escala los nuevos a tiempo."),
        (3, "Diagnostica la causa de los problemas y propone soluciones sin supervisión."),
        (4, "Resuelve situaciones ambiguas y previene que se repitan."),
        (5, "Detecta patrones entre cuentas e impulsa cambios de proceso o de producto."),
    ],
    "Colaboración interfuncional": [
        (2, "Colabora con Soporte y Ventas en los casos que se le asignan."),
        (3, "Coordina con otras áreas para resolver casos de sus cuentas."),
        (4, "Alinea a varias áreas en planes de cuenta y sostiene los acuerdos."),
        (5, "Construye procesos de trabajo compartidos entre Customer Success y otras áreas."),
    ],
}


def demo_state() -> dict:
    """Arma un estado completo de ejemplo (skills y matriz aprobadas)."""
    skills_df = pd.DataFrame(DEMO_SKILLS, columns=["Skill", "Tipo", "Definición"])
    levels = DEMO_CONFIG["levels"]
    desc = pd.DataFrame({lvl: [DEMO_MATRIX[s][i][1] for s, _, _ in DEMO_SKILLS] for i, lvl in enumerate(levels)},
                        index=[s for s, _, _ in DEMO_SKILLS])
    mat = pd.DataFrame({lvl: [DEMO_MATRIX[s][i][0] for s, _, _ in DEMO_SKILLS] for i, lvl in enumerate(levels)},
                       index=[s for s, _, _ in DEMO_SKILLS])
    desc.index.name = mat.index.name = "Skill"
    return {"config": dict(DEMO_CONFIG), "skills_df": skills_df, "desc_df": desc, "mat_df": mat}


# ---------------------------------------------------------------------------
# Estado de la sesión
# ---------------------------------------------------------------------------


def go_to(tab: str) -> None:
    """Callback de navegación: cambia la pestaña activa y pide volver arriba."""
    st.session_state.active_tab = tab
    st.session_state.scroll_top = True


def nav_button(tab: str, label: str, key: str, primary: bool = True) -> None:
    """Botón que lleva a otra pestaña sin tener que subir y hacer clic arriba."""
    st.button(label, key=key, type="primary" if primary else "secondary", on_click=go_to, args=(tab,))


def show_md(md: str) -> None:
    """Muestra Markdown con títulos de menor tamaño dentro de un recuadro."""
    md = re.sub(r"^## ", "#### ", md, flags=re.MULTILINE)
    md = re.sub(r"^# ", "### ", md, flags=re.MULTILINE)
    with st.container(border=True):
        st.markdown(md)


def init_state() -> None:
    defaults = {
        "config": None,
        "skills_df": None,           # propuesta de skills (editable)
        "skills_approved": None,     # skills aprobadas por RR.HH.
        "desc_df": None,             # matriz de descriptores (editable)
        "mat_df": None,              # matriz de madurez 1-5 (editable)
        "matrix_approved": False,
        "comparison_md": None,
        "jd_md": None,
        "imported_file_id": None,
        "editor_version": 0,         # fuerza a recrear los editores al cargar datos nuevos
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def load_full_state(data: dict) -> None:
    """Carga un estado completo (demo o archivo importado) y lo marca como aprobado."""
    st.session_state.config = data["config"]
    st.session_state.skills_df = data["skills_df"]
    st.session_state.skills_approved = data["skills_df"].copy()
    st.session_state.desc_df = data["desc_df"]
    st.session_state.mat_df = data["mat_df"]
    st.session_state.matrix_approved = True
    st.session_state.comparison_md = None
    st.session_state.jd_md = None
    st.session_state.editor_version += 1


# ---------------------------------------------------------------------------
# Cliente de IA
# ---------------------------------------------------------------------------


def extract_json(text: str) -> dict:
    """Extrae el primer objeto JSON del texto (tolera bloques ```json)."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("La respuesta no contiene JSON.")
    return json.loads(text[start : end + 1])


def call_claude(system: str, user: str, schema: type[BaseModel], max_tokens: int = 8000) -> BaseModel:
    """Llama a Claude pidiendo JSON, lo valida con Pydantic y reintenta una vez si falla."""
    import anthropic  # import local: el modo demo funciona aunque falte la clave

    api_key = st.session_state.get("api_key")
    if not api_key:
        raise RuntimeError("Falta la clave de API. Cárgala en la barra lateral o activa el modo demo.")
    client = anthropic.Anthropic(api_key=api_key)
    model = get_secret("ANTHROPIC_MODEL", DEFAULT_MODEL)

    messages = [{"role": "user", "content": user}]
    last_error = None
    for _ in range(2):  # intento original + 1 reintento
        response = client.messages.create(model=model, max_tokens=max_tokens, system=system, messages=messages)
        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        try:
            return schema.model_validate(extract_json(text))
        except (ValueError, ValidationError, json.JSONDecodeError) as exc:
            last_error = exc
            messages += [
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"Tu respuesta no cumple el formato pedido ({exc}). "
                                            "Devuelve SOLO el JSON válido, sin texto adicional."},
            ]
    raise RuntimeError(f"La IA no devolvió un JSON válido tras reintentar: {last_error}")


def system_prompt(language: str) -> str:
    return (
        "Eres experto/a en gestión del talento, diseño organizacional y marcos de competencias. "
        "Trabajas para el equipo de RR.HH. y propones contenido que luego una persona revisa. "
        "Escribe conductas observables y verificables; evita adjetivos vagos (\"excelente\", \"sólido\"). "
        "Usa lenguaje inclusivo y no incluyas criterios de edad, género ni otros sesgos. "
        f"Redacta todo el contenido en {language}. "
        "Responde SOLO con un objeto JSON válido que siga exactamente el esquema pedido, sin texto adicional."
    )


def context_block(cfg: dict) -> str:
    return (
        f"Rol: {cfg['role']}\nÁrea: {cfg['area']}\nSector: {cfg['industry']}\n"
        f"Tamaño de empresa: {cfg['company_size']}\n"
        f"Niveles de seniority (de menor a mayor): {', '.join(cfg['levels'])}\n"
        f"Valores / marco de competencias de la empresa: {cfg['framework'] or 'No informado'}"
    )


# ---------------------------------------------------------------------------
# Funciones de IA (una por paso del proceso)
# ---------------------------------------------------------------------------


def ai_propose_skills(cfg: dict) -> pd.DataFrame:
    if st.session_state.demo:
        return demo_state()["skills_df"]
    n_tech = round(cfg["n_skills"] * cfg["tech_share"] / 100)
    user = (
        f"{context_block(cfg)}\n\n"
        f"Propón exactamente {cfg['n_skills']} skills clave para este rol: unas {n_tech} de tipo \"Técnica\" "
        f"y el resto de tipo \"Conductual\". Cada definición en una línea.\n"
        'Esquema: {"skills": [{"name": str, "type": "Técnica" | "Conductual", "definition": str}]}'
    )
    result: SkillList = call_claude(system_prompt(cfg["language"]), user, SkillList)
    return pd.DataFrame([(s.name, s.type, s.definition) for s in result.skills],
                        columns=["Skill", "Tipo", "Definición"])


def ai_generate_matrix(cfg: dict, skills: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    levels = cfg["levels"]
    names = skills["Skill"].tolist()
    if st.session_state.demo:
        demo = demo_state()
        if set(names) <= set(demo["desc_df"].index) and levels == DEFAULT_LEVELS:
            return demo["desc_df"].loc[names], demo["mat_df"].loc[names]
        # Demo con skills editadas: descriptores genéricos de ejemplo
        desc = pd.DataFrame({lvl: [f"[Demo] Descriptor de '{n}' para {lvl}." for n in names] for lvl in levels}, index=names)
        mat = pd.DataFrame({lvl: [min(5, 1 + i) for _ in names] for i, lvl in enumerate(levels)}, index=names)
        desc.index.name = mat.index.name = "Skill"
        return desc, mat

    skills_txt = "\n".join(f"- {r.Skill} ({r.Tipo}): {r.Definición}" for r in skills.itertuples())
    scale = ", ".join(f"{k} {v}" for k, v in MATURITY_LABELS.items())
    user = (
        f"{context_block(cfg)}\n\nSkills aprobadas:\n{skills_txt}\n\n"
        f"Para CADA skill y CADA nivel ({', '.join(levels)}) asigna un nivel de madurez esperado en la escala "
        f"1-5 ({scale}). La madurez no puede bajar al subir de seniority. Redacta un descriptor de 1 a 2 frases "
        "con conductas observables y verificables que distingan claramente ese nivel del anterior.\n"
        "Usa los nombres de skills y niveles exactamente como se indican.\n"
        'Esquema: {"rows": [{"skill": str, "levels": [{"level": str, "maturity": int, "descriptor": str}]}]}'
    )
    result: Matrix = call_claude(system_prompt(cfg["language"]), user, Matrix, max_tokens=12000)

    by_skill = {row.skill.strip().lower(): row for row in result.rows}
    desc = pd.DataFrame(index=names, columns=levels, dtype=object)
    mat = pd.DataFrame(index=names, columns=levels, dtype="Int64")
    for name in names:
        row = by_skill.get(name.strip().lower())
        if row is None:
            raise RuntimeError(f"La IA no devolvió la skill '{name}'. Vuelve a generar la matriz.")
        cells = {c.level.strip().lower(): c for c in row.levels}
        previous = 1
        for lvl in levels:
            cell = cells.get(lvl.strip().lower())
            if cell is None:
                raise RuntimeError(f"Falta el nivel '{lvl}' en la skill '{name}'. Vuelve a generar la matriz.")
            previous = max(previous, cell.maturity)  # garantiza madurez no decreciente
            desc.loc[name, lvl] = cell.descriptor
            mat.loc[name, lvl] = previous
    desc.index.name = mat.index.name = "Skill"
    return desc, mat


def ai_compare(cfg: dict, desc: pd.DataFrame, mat: pd.DataFrame, origin: str, target: str, skills: list[str]) -> str:
    gaps = {s: int(mat.loc[s, target]) - int(mat.loc[s, origin]) for s in skills}
    with_gap = [s for s in skills if gaps[s] > 0]
    header = f"# De {origin} a {target} — {cfg['role']}\n\n"
    if not with_gap:
        return header + "No hay brechas de madurez entre estos niveles para las skills elegidas."

    if st.session_state.demo:
        body = "**Resumen (demo):** el salto se concentra en las skills con mayor brecha de madurez.\n\n"
        for s in with_gap:
            body += (f"## {s} (+{gaps[s]})\n\n**Qué cambia:** pasa de \"{desc.loc[s, origin]}\" a "
                     f"\"{desc.loc[s, target]}\"\n\n**Evidencias:**\n- Ejemplos concretos donde ya actúa como {target}.\n\n"
                     "**Acciones de desarrollo (70-20-10):**\n- 70: asumir un caso real del nivel siguiente.\n"
                     "- 20: mentoría con alguien del nivel destino.\n- 10: formación específica.\n\n")
        return header + body

    lines = "\n".join(
        f"- {s}: {origin} (madurez {mat.loc[s, origin]}) = \"{desc.loc[s, origin]}\" → "
        f"{target} (madurez {mat.loc[s, target]}) = \"{desc.loc[s, target]}\"" for s in with_gap
    )
    user = (
        f"{context_block(cfg)}\n\nCompara los niveles {origin} y {target} en estas skills con brecha:\n{lines}\n\n"
        "Para cada skill explica qué cambia en el comportamiento esperado, 2-3 evidencias observables que "
        "demostrarían que la persona ya dio el salto y 2-3 acciones de desarrollo siguiendo el modelo 70-20-10. "
        "Agrega un resumen de 2 frases. No evalúes a ninguna persona concreta.\n"
        'Esquema: {"summary": str, "items": [{"skill": str, "what_changes": str, "evidence": [str], '
        '"development_actions": [str]}]}'
    )
    result: Comparison = call_claude(system_prompt(cfg["language"]), user, Comparison)
    body = f"**Resumen:** {result.summary}\n\n"
    for item in result.items:
        body += f"## {item.skill} (+{gaps.get(item.skill, '?')})\n\n**Qué cambia:** {item.what_changes}\n\n**Evidencias:**\n"
        body += "".join(f"- {e}\n" for e in item.evidence)
        body += "\n**Acciones de desarrollo:**\n" + "".join(f"- {a}\n" for a in item.development_actions) + "\n"
    return header + body


def ai_job_description(cfg: dict, desc: pd.DataFrame, level: str, skills: list[str]) -> str:
    if st.session_state.demo:
        jd = JobDescription(
            title=f"{cfg['role']} — {level}",
            purpose=f"[Demo] Acompañar a los clientes de {cfg['industry']} para que obtengan valor del producto y renueven.",
            responsibilities=[desc.loc[s, level] for s in skills],
            requirements=[f"{s}: nivel {int(st.session_state.mat_df.loc[s, level])} "
                          f"({MATURITY_LABELS[int(st.session_state.mat_df.loc[s, level])]})." for s in skills],
            we_offer=["[Demo] Completar con la propuesta de valor de la empresa."],
        )
    else:
        reqs = "\n".join(f"- {s}: \"{desc.loc[s, level]}\"" for s in skills)
        user = (
            f"{context_block(cfg)}\n\nRedacta la descripción de puesto para el nivel {level}. "
            f"Usa SOLO estas skills y descriptores de la matriz aprobada como base de responsabilidades y requisitos:\n{reqs}\n\n"
            "No inventes requisitos que no surjan de la matriz, no pidas años de experiencia arbitrarios, "
            "edad ni género. En 'we_offer' propone puntos genéricos que RR.HH. debe completar.\n"
            'Esquema: {"title": str, "purpose": str, "responsibilities": [str], "requirements": [str], "we_offer": [str]}'
        )
        jd = call_claude(system_prompt(cfg["language"]), user, JobDescription)
    md = f"# {jd.title}\n\n## Propósito del rol\n\n{jd.purpose}\n\n## Responsabilidades\n\n"
    md += "".join(f"- {r}\n" for r in jd.responsibilities)
    md += "\n## Requisitos\n\n" + "".join(f"- {r}\n" for r in jd.requirements)
    md += "\n## Qué ofrecemos\n\n" + "".join(f"- {r}\n" for r in jd.we_offer)
    return md


# ---------------------------------------------------------------------------
# Exportación / importación
# ---------------------------------------------------------------------------


def to_excel(skills: pd.DataFrame, desc: pd.DataFrame, mat: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        desc.to_excel(writer, sheet_name="Descriptores")
        mat.to_excel(writer, sheet_name="Niveles de madurez")
        skills.to_excel(writer, sheet_name="Skills", index=False)
        pd.DataFrame(list(MATURITY_LABELS.items()), columns=["Nivel", "Etiqueta"]).to_excel(
            writer, sheet_name="Escala", index=False)
    return buffer.getvalue()


def to_json(cfg: dict, skills: pd.DataFrame, desc: pd.DataFrame, mat: pd.DataFrame) -> str:
    payload = {
        "app": "skills-matrix-builder",
        "version": 1,
        "config": cfg,
        "skills": skills.to_dict(orient="records"),
        "descriptors": desc.to_dict(orient="index"),
        "maturity": {k: {lvl: int(v) for lvl, v in row.items()} for k, row in mat.to_dict(orient="index").items()},
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def from_json(raw: bytes) -> dict:
    data = json.loads(raw)
    if data.get("app") != "skills-matrix-builder":
        raise ValueError("El archivo no fue exportado por Skills Matrix Builder.")
    cfg = data["config"]
    skills_df = pd.DataFrame(data["skills"])[["Skill", "Tipo", "Definición"]]
    names, levels = skills_df["Skill"].tolist(), cfg["levels"]
    desc = pd.DataFrame.from_dict(data["descriptors"], orient="index").loc[names, levels]
    mat = pd.DataFrame.from_dict(data["maturity"], orient="index").loc[names, levels].astype("Int64")
    desc.index.name = mat.index.name = "Skill"
    return {"config": cfg, "skills_df": skills_df, "desc_df": desc, "mat_df": mat}


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------


def sidebar() -> None:
    with st.sidebar:
        st.subheader("Tu recorrido")
        st.session_state.progress_box = st.empty()  # se completa al final de main()
        st.divider()
        st.subheader("Configuración")
        secret_key = get_secret("ANTHROPIC_API_KEY")
        # Sin clave en Secrets, la app arranca en modo demo para que cualquiera pueda probarla.
        st.session_state.demo = st.toggle(
            "Modo demo (sin IA)", value=st.session_state.get("demo", not secret_key),
            help="Usa un ejemplo precargado (Customer Success Manager) y no llama a la API.")
        if st.session_state.demo:
            if st.button("Cargar ejemplo completo", width="stretch"):
                load_full_state(demo_state())
                st.rerun()
            st.caption("Atajo: carga el ejemplo con skills y matriz ya aprobadas.")
            st.session_state.api_key = None
        else:
            if secret_key:
                st.session_state.api_key = secret_key
                st.success("Clave de API cargada desde Secrets.")
            else:
                st.session_state.api_key = st.text_input("Clave de API de Anthropic", type="password") or None

        st.divider()
        st.subheader("Retomar una matriz")
        uploaded = st.file_uploader("Importar JSON exportado", type=["json"])
        if uploaded is not None and uploaded.file_id != st.session_state.imported_file_id:
            try:
                load_full_state(from_json(uploaded.getvalue()))
                st.session_state.imported_file_id = uploaded.file_id
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"No se pudo importar: {exc}")



PROGRESS_CSS = """
<style>
.jr {font-family: inherit;}
.jr-badge {background: linear-gradient(135deg, #1E293B, #475569); color: #F8FAFC; border-radius: 14px; padding: 12px 14px; margin-bottom: 12px;}
.jr-badge small {color: #CBD5E1; font-size: 11.5px; letter-spacing: .4px; text-transform: uppercase;}
.jr-badge div {font-size: 17px; font-weight: 700; margin-top: 2px;}
.jr-dots {display: flex; gap: 4px; margin-top: 8px;}
.jr-dots span {flex: 1; height: 6px; border-radius: 3px; background: rgba(255,255,255,.18);}
.jr-dots span.on {background: #FF8A8A;}
.jr-step {display: flex; gap: 10px; position: relative; padding-bottom: 14px;}
.jr-step:not(:last-child)::after {content: ""; position: absolute; left: 13px; top: 28px; bottom: 0; width: 2px; background: #E2E8F0;}
.jr-step.done:not(:last-child)::after {background: #22C55E;}
.jr-dot {flex: 0 0 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-size: 13px; font-weight: 700; background: #E2E8F0; color: #64748B; z-index: 1;}
.jr-step.done .jr-dot {background: #22C55E; color: #FFF;}
.jr-step.now .jr-dot {background: #FF4B4B; color: #FFF; animation: jr-pulse 1.6s infinite;}
@keyframes jr-pulse {0% {box-shadow: 0 0 0 0 rgba(255,75,75,.55);} 70% {box-shadow: 0 0 0 9px rgba(255,75,75,0);} 100% {box-shadow: 0 0 0 0 rgba(255,75,75,0);}}
.jr-txt b {display: block; font-size: 14px; color: #0F172A; line-height: 1.25; margin-top: 4px;}
.jr-txt span {font-size: 12px; color: #64748B;}
.jr-step.now .jr-txt b {color: #FF4B4B;}
.jr-step.todo .jr-txt b {color: #94A3B8;}
</style>
"""

JOURNEY = [  # (pestaña, título, qué se consigue)
    (1, "Skills aprobadas", "La IA propone, tú decides"),
    (2, "Matriz aprobada", "Madurez y conductas por nivel"),
    (3, "Niveles comparados", "Base del plan de carrera"),
    (4, "Descripción de puesto", "Lista para revisar y publicar"),
]
JOURNEY_MESSAGES = [
    "¡Arranquemos! Define el rol y deja que la IA proponga skills 🚀",
    "Skills listas ✨ Ahora, a darles niveles en la matriz.",
    "Matriz aprobada 🎯 Mira qué cambia al subir de nivel.",
    "¡Casi! Solo falta la descripción de puesto 📝",
    "¡Recorrido completo! Descarga tus resultados 🏆",
]


def render_progress() -> None:
    """Recorrido interactivo en la barra lateral. Se llama al final de main(),
    cuando el estado ya refleja la última acción del usuario."""
    ss = st.session_state
    done = [ss.skills_approved is not None, bool(ss.matrix_approved),
            ss.comparison_md is not None, ss.jd_md is not None]
    n_done = sum(done)
    current = next((i for i, ok in enumerate(done) if not ok), None)  # primer paso pendiente

    # Guiño a la escala de la matriz: el "nivel de madurez" de quien usa la app sube con cada paso.
    maturity = n_done + 1
    dots = "".join(f'<span class="{"on" if i < maturity else ""}"></span>' for i in range(5))
    steps_html = ""
    for i, (_, title, sub) in enumerate(JOURNEY):
        state = "done" if done[i] else ("now" if i == current else "todo")
        icon = "✓" if done[i] else str(i + 1)
        steps_html += (f'<div class="jr-step {state}"><div class="jr-dot">{icon}</div>'
                       f'<div class="jr-txt"><b>{title}</b><span>{sub}</span></div></div>')

    with ss.progress_box.container():
        st.html(PROGRESS_CSS + f"""
        <div class="jr">
          <div class="jr-badge"><small>Tu madurez en Skills Matrix Builder</small>
            <div>Nivel {maturity} · {MATURITY_LABELS[maturity]}</div>
            <div class="jr-dots">{dots}</div></div>
          {steps_html}
        </div>""")
        st.progress(n_done / 4, text=f"{n_done} de 4 pasos · {n_done * 25}%")
        st.caption(JOURNEY_MESSAGES[n_done])
        if current is not None:
            tab = TABS[JOURNEY[current][0]]
            if ss.get("active_tab") != tab:
                st.button(f"Continuar: {tab} →", key="journey_go", on_click=go_to, args=(tab,), width="stretch")

    # Celebración, una sola vez, al completar el recorrido.
    if n_done == 4 and not ss.get("celebrated"):
        ss.celebrated = True
        st.balloons()
    elif n_done < 4:
        ss.celebrated = False


def tab_role() -> None:
    st.subheader("Paso 1 · Definir el rol y proponer skills")
    demo = st.session_state.demo
    cfg = st.session_state.config or (DEMO_CONFIG if demo else {})
    if demo:
        st.info(
            "**Cómo probar la demo:** pulsa «Proponer skills», revisa y aprueba la tabla, y sigue por las "
            "pestañas 2, 3 y 4. En modo demo los resultados corresponden siempre al ejemplo de "
            "Customer Success Manager; con una clave de API, la IA los genera para cualquier rol.",
            icon="🧪",
        )
    with st.form("config_form"):
        c1, c2 = st.columns(2)
        role = c1.text_input("Rol o familia de puestos *", cfg.get("role", ""), placeholder="Ej.: Customer Success Manager")
        area = c2.text_input("Área", cfg.get("area", ""))
        industry = c1.text_input("Sector", cfg.get("industry", ""), placeholder="Ej.: Software B2B")
        sizes = ["1-50 empleados", "51-200 empleados", "201-1000 empleados", "Más de 1000 empleados"]
        company_size = c2.selectbox("Tamaño de empresa", sizes, index=sizes.index(cfg.get("company_size", sizes[1])))
        framework = st.text_area("Valores o marco de competencias de la empresa (opcional)", cfg.get("framework", ""),
                                 placeholder="Pega aquí valores o competencias corporativas para alinear la matriz.")
        c3, c4, c5 = st.columns(3)
        levels_txt = c3.text_input("Niveles (3 a 5, separados por coma)", ", ".join(cfg.get("levels", DEFAULT_LEVELS)))
        n_skills = c4.slider("Cantidad de skills", 6, 10, cfg.get("n_skills", 8))
        tech_share = c5.slider("% de skills técnicas", 0, 100, cfg.get("tech_share", 50), step=10)
        language = st.radio("Idioma de salida", ["Español", "Inglés"], horizontal=True,
                            index=0 if cfg.get("language", "Español") == "Español" else 1)
        submitted = st.form_submit_button("✨ Proponer skills", type="primary")

    if submitted:
        levels = [lvl.strip() for lvl in levels_txt.split(",") if lvl.strip()]
        if not role.strip():
            st.error("Indica el rol.")
        elif not 3 <= len(levels) <= 5 or len(set(levels)) != len(levels):
            st.error("Indica entre 3 y 5 niveles distintos.")
        else:
            st.session_state.config = dict(DEMO_CONFIG) if demo else {
                "role": role.strip(), "area": area.strip(), "industry": industry.strip(),
                "company_size": company_size, "framework": framework.strip(), "levels": levels,
                "n_skills": n_skills, "tech_share": tech_share, "language": language,
            }
            try:
                with st.spinner("La IA está proponiendo skills..."):
                    st.session_state.skills_df = ai_propose_skills(st.session_state.config)
                # Reiniciar pasos posteriores
                st.session_state.skills_approved = None
                st.session_state.matrix_approved = False
                st.session_state.desc_df = st.session_state.mat_df = None
                st.session_state.comparison_md = st.session_state.jd_md = None
                st.session_state.editor_version += 1
            except Exception as exc:  # noqa: BLE001
                st.error(f"Error al llamar a la IA: {exc}")

    if st.session_state.skills_df is not None:
        st.markdown("**Revisa la propuesta:** edita, agrega o elimina filas y luego aprueba.")
        edited = st.data_editor(
            st.session_state.skills_df, num_rows="dynamic", width="stretch", hide_index=True,
            key=f"skills_editor_{st.session_state.editor_version}",
            column_config={
                "Tipo": st.column_config.SelectboxColumn(options=["Técnica", "Conductual"], required=True),
                "Definición": st.column_config.TextColumn(width="large"),
            },
        )
        if st.button("✅ Aprobar skills", type="primary"):
            clean = edited.dropna(subset=["Skill"])
            clean = clean[clean["Skill"].str.strip() != ""]
            if len(clean) < 3:
                st.error("Necesitas al menos 3 skills.")
            elif clean["Skill"].duplicated().any():
                st.error("Hay skills repetidas.")
            else:
                st.session_state.skills_approved = clean.reset_index(drop=True)
                st.session_state.matrix_approved = False
                st.session_state.desc_df = st.session_state.mat_df = None
                st.session_state.comparison_md = st.session_state.jd_md = None
        if st.session_state.skills_approved is not None:
            st.success("Skills aprobadas.")
            nav_button(TABS[2], "Siguiente paso: generar la matriz →", "next_1")


def heatmap(mat: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        mat.astype(int), text_auto=True, aspect="auto", zmin=1, zmax=5,
        color_continuous_scale="Blues", labels={"color": "Madurez"},
    )
    fig.update_layout(height=80 + 40 * len(mat), margin=dict(l=10, r=10, t=30, b=10),
                      coloraxis_colorbar=dict(tickvals=list(MATURITY_LABELS), ticktext=list(MATURITY_LABELS.values())))
    fig.update_xaxes(side="top", title=None)
    fig.update_yaxes(title=None)
    return fig


def tab_matrix() -> None:
    st.subheader("Paso 2 · Matriz de habilidades")
    if st.session_state.skills_approved is None:
        st.info("🔒 Aprueba primero las skills en la pestaña «1. Rol y skills».")
        nav_button(TABS[1], "← Ir al paso 1", "back_2", primary=False)
        return
    cfg = st.session_state.config
    if st.button("✨ Generar matriz" if st.session_state.desc_df is None else "🔄 Volver a generar matriz",
                 type="primary"):
        try:
            with st.spinner("La IA está redactando los descriptores por nivel..."):
                desc, mat = ai_generate_matrix(cfg, st.session_state.skills_approved)
            st.session_state.desc_df, st.session_state.mat_df = desc, mat
            st.session_state.matrix_approved = False
            st.session_state.editor_version += 1
        except Exception as exc:  # noqa: BLE001
            st.error(f"Error al generar la matriz: {exc}")

    if st.session_state.desc_df is None:
        return

    v = st.session_state.editor_version
    st.markdown("**Descriptores por nivel** (doble clic en una celda para editar)")
    desc_edit = st.data_editor(
        st.session_state.desc_df, width="stretch", key=f"desc_editor_{v}",
        column_config={lvl: st.column_config.TextColumn(width="medium") for lvl in cfg["levels"]},
    )
    with st.expander("📖 Ver descriptores completos (lectura)"):
        for skill, row in desc_edit.iterrows():
            st.markdown(f"**{skill}**\n" + "".join(f"\n- *{lvl}:* {txt}" for lvl, txt in row.items()))
    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Nivel de madurez esperado (1-5)**")
        mat_edit = st.data_editor(
            st.session_state.mat_df, width="stretch", key=f"mat_editor_{v}",
            column_config={lvl: st.column_config.NumberColumn(min_value=1, max_value=5, step=1, required=True)
                           for lvl in cfg["levels"]},
        )
        st.caption(" · ".join(f"{k} {v}" for k, v in MATURITY_LABELS.items()))
    with right:
        st.markdown("**Mapa de calor de madurez**")
        st.plotly_chart(heatmap(mat_edit), width="stretch")

    c1, c2, c3 = st.columns(3)
    if c1.button("✅ Aprobar matriz", type="primary"):
        st.session_state.desc_df, st.session_state.mat_df = desc_edit, mat_edit.astype("Int64")
        st.session_state.matrix_approved = True
    skills = st.session_state.skills_approved
    c2.download_button("⬇️ Descargar Excel", to_excel(skills, desc_edit, mat_edit),
                       file_name="skills_matrix.xlsx", width="stretch")
    c3.download_button("⬇️ Descargar JSON (para retomar)", to_json(cfg, skills, desc_edit, mat_edit),
                       file_name="skills_matrix.json", mime="application/json", width="stretch")
    if st.session_state.matrix_approved:
        st.success("Matriz aprobada. Ya puedes comparar niveles y generar descripciones de puesto.")
        nav_button(TABS[3], "Siguiente paso: comparar niveles →", "next_2")


def tab_compare() -> None:
    st.subheader("Paso 3 · Comparar niveles (base del plan de carrera)")
    if not st.session_state.matrix_approved:
        st.info("🔒 Aprueba primero la matriz en la pestaña «2. Matriz».")
        nav_button(TABS[2], "← Ir al paso 2", "back_3", primary=False)
        return
    cfg, desc, mat = st.session_state.config, st.session_state.desc_df, st.session_state.mat_df
    levels = cfg["levels"]
    c1, c2 = st.columns(2)
    origin = c1.selectbox("Nivel de origen", levels, index=max(0, len(levels) - 3))
    target = c2.selectbox("Nivel de destino", levels, index=len(levels) - 2)
    skills = st.multiselect("Skills a comparar", list(desc.index), default=list(desc.index))
    if levels.index(target) <= levels.index(origin):
        st.warning("El nivel de destino debe ser superior al de origen.")
        return
    if not skills:
        return

    gap = pd.DataFrame({"Skill": skills, "Brecha": [int(mat.loc[s, target]) - int(mat.loc[s, origin]) for s in skills]})
    gap = gap.sort_values("Brecha")
    fig = px.bar(gap, x="Brecha", y="Skill", orientation="h", text="Brecha",
                 title=f"Brecha de madurez: {origin} → {target}")
    fig.update_layout(height=120 + 36 * len(skills), margin=dict(l=10, r=10, t=50, b=10), yaxis_title=None)
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, width="stretch")

    if st.button("✨ Generar análisis de la brecha", type="primary"):
        try:
            with st.spinner("La IA está analizando la brecha entre niveles..."):
                st.session_state.comparison_md = ai_compare(cfg, desc, mat, origin, target, skills)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Error al generar la comparación: {exc}")
    if st.session_state.comparison_md:
        show_md(st.session_state.comparison_md)
        st.download_button("⬇️ Descargar comparación (.md)", st.session_state.comparison_md,
                           file_name="comparacion_niveles.md", mime="text/markdown")
    st.divider()
    nav_button(TABS[4], "Siguiente paso: descripción de puesto →", "next_3",
               primary=st.session_state.comparison_md is not None)


def tab_jd() -> None:
    st.subheader("Paso 4 · Descripción de puesto")
    if not st.session_state.matrix_approved:
        st.info("🔒 Aprueba primero la matriz en la pestaña «2. Matriz».")
        nav_button(TABS[2], "← Ir al paso 2", "back_4", primary=False)
        return
    cfg, desc = st.session_state.config, st.session_state.desc_df
    level = st.selectbox("Nivel del puesto", cfg["levels"], index=len(cfg["levels"]) - 2)
    skills = st.multiselect("Skills a incluir", list(desc.index), default=list(desc.index), key="jd_skills")
    if st.button("✨ Generar descripción de puesto", type="primary", disabled=not skills):
        try:
            with st.spinner("La IA está redactando la descripción de puesto..."):
                st.session_state.jd_md = ai_job_description(cfg, desc, level, skills)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Error al generar la descripción: {exc}")
    if st.session_state.jd_md:
        show_md(st.session_state.jd_md)
        st.download_button("⬇️ Descargar descripción (.md)", st.session_state.jd_md,
                           file_name="descripcion_de_puesto.md", mime="text/markdown")
        st.divider()
        st.success("¡Proceso completo! Puedes volver a la matriz para descargarla o empezar con otro rol.")
        c1, c2 = st.columns(2)
        with c1:
            nav_button(TABS[2], "← Volver a la matriz", "back_jd", primary=False)
        with c2:
            nav_button(TABS[1], "Empezar con otro rol", "restart", primary=False)


HOME_CSS = """
<style>
.smb-hero {background: linear-gradient(135deg, #1E293B 0%, #334155 60%, #475569 100%); color: #F8FAFC;
  border-radius: 18px; padding: 40px 44px; margin-bottom: 8px;}
.smb-kicker {display: inline-block; background: rgba(255,75,75,.18); color: #FFB4B4; border: 1px solid rgba(255,75,75,.45);
  border-radius: 999px; padding: 4px 12px; font-size: 13px; font-weight: 600; letter-spacing: .3px;}
.smb-hero h1 {color: #F8FAFC; font-size: 40px; line-height: 1.15; margin: 16px 0 12px; padding: 0;}
.smb-hero h1 span {color: #FF8A8A;}
.smb-hero p {color: #CBD5E1; font-size: 18px; line-height: 1.55; max-width: 760px; margin: 0;}
.smb-section {font-size: 13px; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; color: #FF4B4B; margin: 28px 0 10px;}
.smb-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px;}
.smb-card {background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 18px 18px 16px; color: #1E293B;}
.smb-card .ico {font-size: 26px; line-height: 1;}
.smb-card h4 {color: #0F172A; font-size: 17px; margin: 10px 0 6px; padding: 0;}
.smb-card p {color: #475569; font-size: 14.5px; line-height: 1.5; margin: 0;}
.smb-problem {background: #FFF5F5; border-color: #FECACA;}
.smb-flow {display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; counter-reset: step;}
.smb-step {position: relative; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 14px; padding: 16px; color: #1E293B;}
.smb-step b {display: inline-flex; align-items: center; justify-content: center; width: 28px; height: 28px; border-radius: 50%;
  background: #FF4B4B; color: #FFF; font-size: 14px; margin-bottom: 8px;}
.smb-step h4 {color: #0F172A; font-size: 15.5px; margin: 0 0 4px; padding: 0;}
.smb-step p {color: #475569; font-size: 13.5px; line-height: 1.45; margin: 0;}
.smb-who {display: flex; flex-wrap: wrap; gap: 10px;}
.smb-pill {background: #F1F5F9; border: 1px solid #E2E8F0; border-radius: 999px; padding: 8px 14px; color: #1E293B; font-size: 14px;}
.smb-pill b {color: #0F172A;}
</style>
"""

HOME_HERO_HTML = """
<div class="smb-hero">
  <span class="smb-kicker">IA aplicada a Recursos Humanos</span>
  <h1>Qué se espera de cada rol, <span>en cada nivel</span>, en minutos.</h1>
  <p>Skills Matrix Builder usa IA generativa para que RR.HH. construya la matriz de habilidades de un rol con niveles
  de madurez por seniority, y la convierta en planes de carrera y descripciones de puesto coherentes.</p>
</div>
"""

HOME_BODY_HTML = """
<div class="smb-section">El problema</div>
<div class="smb-grid">
  <div class="smb-card smb-problem"><div class="ico">⏳</div><h4>Semanas de talleres</h4>
    <p>Armar una matriz a mano exige muchas reuniones con managers y redactar descriptores para cada skill y cada nivel.</p></div>
  <div class="smb-card smb-problem"><div class="ico">🧩</div><h4>Criterios dispares</h4>
    <p>Sin matriz, promociones, planes de desarrollo y búsquedas se definen distinto según quién decide.</p></div>
  <div class="smb-card smb-problem"><div class="ico">🌫️</div><h4>Expectativas vagas</h4>
    <p>“Ser proactivo” o “tener buena comunicación” no dicen qué conducta concreta separa a un nivel del siguiente.</p></div>
</div>

<div class="smb-section">Qué obtienes</div>
<div class="smb-grid">
  <div class="smb-card"><div class="ico">🧭</div><h4>Matriz de habilidades</h4>
    <p>Skills técnicas y conductuales, con un nivel de madurez de 1 a 5 y conductas observables por seniority.</p></div>
  <div class="smb-card"><div class="ico">📈</div><h4>Comparación entre niveles</h4>
    <p>Qué cambia de un nivel al siguiente, qué evidencias lo demuestran y qué acciones de desarrollo ayudan.</p></div>
  <div class="smb-card"><div class="ico">📝</div><h4>Descripción de puesto</h4>
    <p>Redactada a partir de la matriz aprobada, con lenguaje inclusivo y requisitos coherentes con el nivel.</p></div>
  <div class="smb-card"><div class="ico">📤</div><h4>Todo exportable</h4>
    <p>Excel y JSON para la matriz (y retomarla después), Markdown para comparaciones y descripciones.</p></div>
</div>

<div class="smb-section">Cómo funciona</div>
<div class="smb-flow">
  <div class="smb-step"><b>1</b><h4>Define el rol</h4><p>Rol, contexto de la empresa, niveles y cantidad de skills.</p></div>
  <div class="smb-step"><b>2</b><h4>Aprueba las skills</h4><p>La IA propone; RR.HH. edita, agrega o quita.</p></div>
  <div class="smb-step"><b>3</b><h4>Aprueba la matriz</h4><p>Descriptores y madurez por nivel, con mapa de calor.</p></div>
  <div class="smb-step"><b>4</b><h4>Usa los resultados</h4><p>Compara niveles y genera descripciones de puesto.</p></div>
</div>

<div class="smb-section">Principios de diseño</div>
<div class="smb-grid">
  <div class="smb-card"><div class="ico">🙋</div><h4>La IA propone, la persona decide</h4>
    <p>Nada se usa sin revisión y aprobación de RR.HH. en cada paso.</p></div>
  <div class="smb-card"><div class="ico">🔒</div><h4>Roles, no personas</h4>
    <p>La app no recibe ni procesa datos personales de empleados.</p></div>
  <div class="smb-card"><div class="ico">⚖️</div><h4>No puntúa a nadie</h4>
    <p>Evaluar el desempeño de trabajadores es de alto riesgo según el AI Act europeo: esa decisión queda en manos del manager.</p></div>
</div>

<div class="smb-section">Para quién</div>
<div class="smb-who">
  <div class="smb-pill"><b>RR.HH.</b> la opera y aprueba</div>
  <div class="smb-pill"><b>Managers</b> definen expectativas y búsquedas</div>
  <div class="smb-pill"><b>Empleados</b> saben qué se espera para crecer</div>
</div>
"""


def start_demo_example() -> None:
    """Callback: carga el ejemplo completo y lleva a la matriz."""
    load_full_state(demo_state())
    go_to(TABS[2])


def home_actions(suffix: str) -> None:
    """Botones de entrada: empezar el proceso o ver el ejemplo ya resuelto."""
    c1, c2, _ = st.columns([1.1, 1.3, 2])
    with c1:
        nav_button(TABS[1], "Empezar ahora →", f"home_start_{suffix}")
    with c2:
        st.button("Ver un ejemplo completo", key=f"home_example_{suffix}", on_click=start_demo_example,
                  help="Carga el ejemplo de Customer Success Manager con la matriz ya aprobada.")


def tab_home() -> None:
    st.html(HOME_CSS + HOME_HERO_HTML)
    home_actions("top")
    if st.session_state.demo:
        st.caption("🧪 La app está en modo demo: usa datos de ejemplo y no llama a la API de IA.")
    st.html(HOME_CSS + HOME_BODY_HTML)
    st.write("")
    home_actions("bottom")


def main() -> None:
    init_state()
    sidebar()
    st.caption("🧭 **Skills Matrix Builder** · IA aplicada a Recursos Humanos")
    # Pestañas con estado: los botones de navegación cambian la pestaña activa.
    home, t1, t2, t3, t4 = st.tabs(TABS, key="active_tab", on_change="rerun")
    with home:
        tab_home()
    for tab, render in ((t1, tab_role), (t2, tab_matrix), (t3, tab_compare), (t4, tab_jd)):
        with tab:
            st.warning(DISCLAIMER, icon="⚠️")
            render()

    render_progress()

    # Tras navegar con un botón, volver al inicio de la página.
    if st.session_state.pop("scroll_top", False):
        st.html(
            "<script>for (const sel of ['[data-testid=\"stMain\"]', '[data-testid=\"stAppViewContainer\"]']) {"
            "const el = document.querySelector(sel); if (el) el.scrollTo({top: 0, behavior: 'smooth'}); }"
            "window.scrollTo({top: 0, behavior: 'smooth'});</script>",
            unsafe_allow_javascript=True,
        )


main()
