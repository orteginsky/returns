# src/services/back_questions.py
from __future__ import annotations
from io import BytesIO
from typing import List, Dict, Any, Optional
import json
import re
import time
from collections import defaultdict

from PyPDF2 import PdfReader, PdfWriter
from google.api_core import exceptions as gex

from src.clients.drive_client import (
    assert_sa_has_access,
    parse_drive_url_to_id,
    download_file_bytes,
)
from src.clients.gdocs_client import get_document_content, write_qas_native
from src.clients.vertex_client import generate_text
from src.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ======= Patrones de variantes de encabezados =======
_VARIANTS = [
    r"preguntas?\s+de\s+regreso",
    r"preguntas?\s+regreso",
    r"preguntas?\s+de\s+seguimiento",
    r"preguntas?\s+para\s+el\s+cliente",
    r"\bPREGUNTAS\s+REGRESO\b",
    r"seguimiento",
    r"back\s*questions?",
    r"follow[-\s]up",
    r"preguntas?\s+pendientes?",
]

# ---------- Helpers de logging (NUEVO) ----------
def _log_detected_questions(tag: str, questions: List[Dict[str, Any]], *, max_len: int = 160) -> None:
    """
    Loggea preguntas detectadas de forma compacta:
      • idx, id, preview(text), page_hint, section_heading
    """
    if not questions:
        logger.info(f"{tag}: 0 preguntas.")
        return
    logger.info(f"{tag}: {len(questions)} preguntas detectadas.")
    for i, q in enumerate(questions, 1):
        qid = q.get("id") or f"q{i}"
        txt = (q.get("text") or "").strip().replace("\n", " ")
        if len(txt) > max_len:
            txt = txt[:max_len] + "…"
        ph  = q.get("page_hint")
        sh  = q.get("section_heading")
        logger.info(f"{tag}  [{i:02d}] id={qid} page_hint={ph} heading={sh!r} :: {txt}")

def _first_heading_variant_hit(text: str) -> str | None:
    for pat in _VARIANTS:
        if re.search(pat, text, re.I):
            return pat
    return None


# ================= Utilidades JSON robustas =================

def _extract_first_json_object(raw: str) -> str | None:
    """
    Devuelve el primer objeto JSON balanceado a partir del primer '{'.
    Respeta comillas y escapes. Soporta texto antes/después y fenced code.
    """
    s = (raw or "").strip()
    # Quita fences tipo ```json ... ```
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I | re.S).strip()

    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
    return None


def _safe_json_loads(raw: str) -> dict:
    s = (raw or "").strip()
    # Intento directo
    try:
        return json.loads(s)
    except Exception:
        pass
    # Intento con extractor balanceado
    frag = _extract_first_json_object(s)
    if frag:
        return json.loads(frag)
    raise ValueError("No se encontró JSON válido en la salida del modelo.")

# ================= Detección de preguntas =================

def _detect_back_questions_via_model_text(sample_text: str, *, max_questions: int) -> List[Dict[str, Any]]:
    """
    * Detector ML usando SOLO TEXTO (sin adjuntos). Se pasa el sample P40+U40 como texto plano.
    """
    prompt = f"""
Eres un extractor de 'Preguntas regreso' en documentos legales.
Busca secciones y encabezados que indiquen preguntas para el cliente, seguimiento o back questions.
Entrega SOLO JSON válido (sin comentarios, sin texto adicional):

{{
  "questions": [
    {{"id": "q1", "text": "<pregunta exactamente como aparece>", "page_hint": <int|null>, "section_heading": "<encabezado_o_null>"}}
  ]
}}

Reglas:
- Incluye solo preguntas (frases con '?' o bullets interrogativos) o elementos bajo encabezados relevantes.
- Acepta variantes: "Preguntas de regreso", "Preguntas regreso", "PREGUNTAS REGRESO", "Preguntas de seguimiento", "Preguntas para el cliente", "Seguimiento", "Back questions", "Follow-up".
- 'page_hint' si el texto sugiere la página; si no, null.
- Máximo {max_questions} preguntas.

[SAMPLE_TEXT]
<<<
{sample_text}
>>>
""".strip()
    model_for_detection = getattr(settings, "map_model_id", settings.vertex_model_id)
    raw = generate_text(prompt, model_id=model_for_detection)
    try:
        data = _safe_json_loads(raw)
        questions = data.get("questions", [])
    except Exception as e:
        logger.warning(f"No se pudo parsear JSON de detección (len={len(raw)}). Error: {e}")
        questions = []

    out = []
    for idx, q in enumerate(questions, 1):
        txt = (q.get("text") or "").strip()
        if not txt:
            continue
        if "?" not in txt:
            heading = (q.get("section_heading") or "").lower()
            if not any(re.search(v, heading, re.I) for v in _VARIANTS):
                continue
        out.append({
            "id": q.get("id") or f"q{idx}",
            "text": txt,
            "page_hint": q.get("page_hint"),
            "section_heading": q.get("section_heading"),
        })
    return out[:max_questions]

def _extract_full_text(data: bytes) -> str:
    """
    Extrae TEXTO de TODO el PDF (concatenado por páginas).
    Prioriza PyMuPDF con 'blocks' (mejor orden visual); fallback a PyPDF2.
    """
    try:
        import fitz
        doc = fitz.open(stream=data, filetype="pdf")
        parts = []
        for page in doc:
            blocks = page.get_text("blocks") or []
            blk_texts = []
            for b in blocks:
                if isinstance(b, (list, tuple)) and len(b) >= 5 and isinstance(b[4], str):
                    blk_texts.append(b[4])
            if blk_texts:
                parts.append("\n".join(blk_texts).replace("\r", ""))
            else:
                parts.append((str(page.get_text("text")) or "").replace("\r", ""))
        doc.close()
        logger.info("📄 Extracción de texto usando PyMuPDF (fitz)")
        return "\n".join(parts)
    except Exception as e:
        logger.info(f"⚠️ PyMuPDF no disponible o falló ({type(e).__name__}), usando PyPDF2 fallback")

    r = PdfReader(BytesIO(data))
    buf = []
    for i in range(len(r.pages)):
        try:
            t = r.pages[i].extract_text() or ""
            buf.append(t.replace("\r", ""))
        except Exception:
            buf.append("")
    logger.info("📄 Extracción de texto usando PyPDF2 (fallback)")
    return "\n".join(buf)



def _detect_back_questions_regex(sample_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Fallback local: si detecta encabezados candidatos en el sample, extrae líneas con '¿...?'.
    """
    r = PdfReader(BytesIO(sample_bytes))
    buf = []
    for p in range(len(r.pages)):
        try:
            buf.append(r.pages[p].extract_text() or "")
        except Exception:
            buf.append("")
    txt = "\n".join(buf)

    if any(re.search(v, txt, re.I) for v in _VARIANTS):
        qs = []
        for line in txt.splitlines():
            s = line.strip()
            if s and re.search(r"¿.+\?", s):
                logger.info(f"la pregunta agregada es {s}")
                qs.append(s)
        return [{"id": f"q{i+1}", "text": q, "page_hint": None, "section_heading": None} for i, q in enumerate(qs)]
    return []

# =============== Resolución del base_prompt dinámico ===============

def _resolve_base_prompt_doc_id(
    *, explicit_base_prompt_doc_id: str | None,
    visa_type: str | None,
    base_prompt_ids_from_req: Dict[str, str] | None
) -> str | None:
    """
    Prioridad:
    1) base_prompt_doc_id explícito en el request.
    2) mapping en el request (additional_params.base_prompt_ids) + visa_type.
    3) mapping en env (settings.base_prompt_ids()) + visa_type (si existe helper).
    4) mapping con 'default'.
    5) None.
    """
    if explicit_base_prompt_doc_id:
        return explicit_base_prompt_doc_id

    vt = (visa_type or "").strip().lower()
    req_map = {(k or "").lower(): v for k, v in (base_prompt_ids_from_req or {}).items()}
    env_map = {}
    # defensivo: solo si existe el helper en settings
    if hasattr(settings, "base_prompt_ids") and callable(getattr(settings, "base_prompt_ids")):
        try:
            env_map = settings.base_prompt_ids()
        except Exception:
            env_map = {}

    if vt and vt in req_map:
        return req_map[vt]
    if vt and vt in env_map:
        return env_map[vt]
    if "default" in req_map:
        return req_map["default"]
    if "default" in env_map:
        return env_map["default"]
    return None

# ================== Helpers de PDF/Chunking ==================

def _extract_sample_pdf_bytes(data: bytes, take_first: int, take_last: int) -> bytes:
    """
    Devuelve un PDF en bytes con primeras `take_first` y últimas `take_last` páginas (sin solapar).
    PyMuPDF primero (rápido), PyPDF2 como fallback.
    """
    try:
        import fitz
        src = fitz.open(stream=data, filetype="pdf")
        total = src.page_count
        first = max(0, min(int(take_first or 0), total))
        last  = max(0, min(int(take_last  or 0), total))
        first_end  = min(first, total)
        last_start = max(0, total - last)
        if last_start < first_end:
            last_start = first_end
        out = fitz.open()
        if first_end > 0:
            out.insert_pdf(src, from_page=0, to_page=first_end - 1)
        if last_start < total:
            out.insert_pdf(src, from_page=last_start, to_page=total - 1)
        result = out.tobytes()
        out.close()
        src.close()
        logger.info("📄 Muestreo de PDF usando PyMuPDF (fitz)")
        return result
    except Exception as e:
        logger.info(f"⚠️ PyMuPDF no disponible o falló ({type(e).__name__}), usando PyPDF2 fallback para muestreo")

    r = PdfReader(BytesIO(data))
    n = len(r.pages)
    w = PdfWriter()
    first_count = min(max(0, int(take_first or 0)), n)
    for i in range(first_count):
        w.add_page(r.pages[i])
    last_count = min(max(0, int(take_last or 0)), max(0, n - first_count))
    for i in range(n - last_count, n):
        if i >= first_count:
            w.add_page(r.pages[i])
    out = BytesIO()
    w.write(out)
    logger.info("📄 Muestreo de PDF usando PyPDF2 (fallback)")
    return out.getvalue()


def _split_pdf_to_text_chunks(data: bytes, pages_per_chunk: int) -> List[str]:
    r = PdfReader(BytesIO(data))
    n = len(r.pages)
    chunks = []
    for start in range(0, n, pages_per_chunk):
        end = min(start + pages_per_chunk, n)
        texts = []
        for i in range(start, end):
            try:
                texts.append(r.pages[i].extract_text() or "")
            except Exception:
                pass
        chunks.append("\n".join(texts))
    return chunks

def _select_topk_chunks_for_question(question: str, chunk_texts: List[str], k: int = 4) -> List[int]:
    """
    Selección heurística por coincidencias de tokens (rápido, sin embeddings).
    """
    q = re.sub(r"[^\w\sáéíóúüñÁÉÍÓÚÜÑ]", " ", (question or "").lower())
    tokens = [t for t in q.split() if len(t) > 2]
    scores = []
    for idx, txt in enumerate(chunk_texts):
        t = (txt or "").lower()
        sc = sum(t.count(tok) for tok in tokens)
        scores.append((sc, idx))
    scores.sort(reverse=True)
    top = [idx for sc, idx in scores[:k] if sc > 0]
    return top or list(range(0, min(k, len(chunk_texts))))

# ================== Router de preguntas → chunks ==================

def _route_questions_to_chunks(
    questions: List[Dict[str, Any]],
    chunk_texts: List[str],
    *,
    k_top: int,
    min_cover: int,
    chunk_cap: int
) -> Dict[int, List[Dict[str, str]]]:
    num_chunks = len(chunk_texts)
    if num_chunks == 0:
        return {}
    sentry = [0, max(0, num_chunks - 1)]

    prelim = defaultdict(list)    # chunk_idx -> [qid]
    assigned = defaultdict(list)  # qid -> [chunk_idx]

    # Pre-asignación por scorer
    for q in questions:
        qid, qtext = q["id"], q["text"]
        top_idx = _select_topk_chunks_for_question(qtext, chunk_texts, k=k_top)
        if not top_idx:
            top_idx = sentry[:]
        for c in top_idx:
            c = max(0, min(num_chunks - 1, c))
            prelim[c].append(qid)
            assigned[qid].append(c)

    # Cobertura mínima
    for q in questions:
        qid = q["id"]
        cur = list(dict.fromkeys(assigned[qid]))
        while len(cur) < min_cover:
            for sc in sentry:
                if sc not in cur:
                    prelim[sc].append(qid)
                    cur.append(sc)
                if len(cur) >= min_cover:
                    break
        assigned[qid] = cur

    # Cap por chunk y derrame con preferencia
    routing = defaultdict(list)
    q_order = {}
    for q in questions:
        qid, qtext = q["id"], q["text"]
        order = _select_topk_chunks_for_question(qtext, chunk_texts, k=max(k_top, min_cover)) or sentry[:]
        q_order[qid] = order

    overflow = []
    for cidx in range(num_chunks):
        qids = prelim[cidx]
        if len(qids) <= chunk_cap:
            for qid in qids:
                qobj = next(x for x in questions if x["id"] == qid)
                routing[cidx].append({"id": qobj["id"], "text": qobj["text"]})
        else:
            def priority(qid): return q_order[qid].index(cidx) if cidx in q_order[qid] else 999
            qids.sort(key=priority)
            keep, spill = qids[:chunk_cap], qids[chunk_cap:]
            for qid in keep:
                qobj = next(x for x in questions if x["id"] == qid)
                routing[cidx].append({"id": qobj["id"], "text": qobj["text"]})
            for qid in spill:
                placed = False
                for alt in q_order[qid]:
                    if alt == cidx:
                        continue
                    if len(routing[alt]) < chunk_cap:
                        qobj = next(x for x in questions if x["id"] == qid)
                        routing[alt].append({"id": qobj["id"], "text": qobj["text"]})
                        placed = True
                        break
                if not placed:
                    overflow.append(qid)

    # Derrame final al último chunk si quedó algo
    for qid in overflow:
        last = max(0, num_chunks - 1)
        qobj = next(x for x in questions if x["id"] == qid)
        routing[last].append({"id": qobj["id"], "text": qobj["text"]})

    # Log de cobertura por pregunta
    cover = defaultdict(list)
    for cidx, lst in routing.items():
        for q in lst:
            cover[q["id"]].append(cidx)
    for q in questions:
        logger.info(f"🧭 Q[{q['id']}] cubierta en chunks: {sorted(cover[q['id']])}")

    return routing

# ================== MAP/REDUCE específicos del híbrido ==================

def _map_chunk_answers_json_from_text(chunk_text: str, chunk_id: int, q_subset: List[Dict[str, str]]) -> Dict[str, Any]:
    prompt = (
        "Eres analista. Te doy el TEXTO de un fragmento (chunk) de un PDF legal y una lista de preguntas.\n"
        "Responde SOLO las preguntas cuya respuesta esté sustentada EN ESTE CHUNK (texto adjunto).\n"
        "Devuelve JSON estricto:\n"
        "{ \"chunk_id\": <int>, \"answers\": [ {\"id\":\"<qid>\", \"answer\":\"<texto>\"} ] }\n"
        "Si no hay evidencia para una pregunta en este chunk, NO la incluyas. No inventes."
    )
    qs_json = json.dumps([{"id": q["id"], "text": q["text"]} for q in q_subset], ensure_ascii=False)
    full_prompt = (
        f"{prompt}\n\n"
        f"Preguntas:\n{qs_json}\n\n"
        f"[CHUNK_TEXT]\n<<<\n{chunk_text}\n>>>"
    )
    raw = generate_text(full_prompt, model_id=settings.map_model_id)
    try:
        data = _safe_json_loads(raw)
    except Exception:
        logger.warning(f"MAP chunk {chunk_id}: JSON inválido.")
        data = {"chunk_id": chunk_id, "answers": []}
    out = {"chunk_id": chunk_id, "answers": []}
    for a in data.get("answers", []):
        qid = (a.get("id") or "").strip()
        ans = (a.get("answer") or "").strip()
        if qid and ans:
            out["answers"].append({"id": qid, "answer": ans})
    return out

def _reduce_answers_for_question(system_text: str, base_prompt: str, qtext: str, candidates: List[str]) -> str:
    if not candidates:
        return "_(No se encontró evidencia suficiente en este documento para responder esta pregunta)_"
    prompt = (
        f"[SYSTEM]\n{system_text}\n\n"
        f"[PROMPT_BASE]\n{base_prompt}\n\n"
        f"Pregunta: {qtext}\n\n"
        "Candidatos (extractos provenientes de distintos fragmentos):\n" +
        "\n---\n".join(candidates) + "\n\n"
        "Instrucción: sintetiza UNA respuesta final clara basada SOLO en los candidatos. No inventes."
    )
    return generate_text(prompt, model_id=settings.reduce_model_id)

# ================== Fallback per-pregunta ==================

def _answer_one_question_over_text_chunks(
    *,
    question_text: str,
    system_text: str,
    base_prompt: str,
    selected_chunk_texts: List[str],
    params: Dict[str, Any]
) -> str:
    enriched_params = dict(params or {})
    enriched_params["question"] = question_text
    enriched_params["objetivo"] = "responder_pregunta_de_regreso"

    # MAP: respuestas parciales por chunk (texto)
    partials: List[str] = []
    total = len(selected_chunk_texts)
    for i, txt in enumerate(selected_chunk_texts, start=1):
        sub_prompt = (
            f"[SYSTEM]\n{system_text}\n\n"
            f"[PROMPT_BASE]\n{base_prompt}\n\n"
            f"[INPUT_CHUNK {i}/{total}]\n<<<\n{txt}\n>>>\n\n"
            f"[PARAMS]\n{enriched_params}\n"
        )
        partial = generate_text(sub_prompt, model_id=getattr(settings, "map_model_id", settings.vertex_model_id))
        partials.append(f"### CHUNK {i}\n{partial}")

    # REDUCE
    reduce_prompt = (
        f"[SYSTEM]\n{system_text}\n\n"
        f"[PROMPT_BASE]\n{base_prompt}\n\n"
        f"Pregunta: {question_text}\n\n"
        "[PARTIALS]\n" + "\n\n".join(partials) + "\n\n"
        "Instrucción: Fusiona y sintetiza una sola respuesta final basada SOLO en los parciales. No inventes."
    )
    return generate_text(reduce_prompt, model_id=getattr(settings, "reduce_model_id", settings.vertex_model_id))

# =============== Helpers de progreso (Google Sheets) ===============

def _col_to_letter(n: int) -> str:
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s

def _make_sheet_updater(sheet_id: Optional[str], row: Optional[int], col: Optional[int]):
    """
    Devuelve una función _sheet_update(status?, link?) que escribe:
      - link en (row, col)
      - status en (row, col+1)
    Si faltan parámetros, la función no hace nada.
    """
    def _sheet_update(status: Optional[str] = None, link: Optional[str] = None):
        if not sheet_id or not row or not col:
            return
        try:
            from src.clients.sheets_client import set_values
            # link en (row, col)
            if link is not None:
                rng_link = f"{_col_to_letter(col)}{row}"
                set_values(sheet_id, rng_link, [[link]])
            # status en (row, col+1)
            if status is not None:
                rng_status = f"{_col_to_letter(col + 1)}{row}"
                set_values(sheet_id, rng_status, [[status]])
        except Exception as e:
            logger.warning(f"No se pudo escribir progreso en Sheet: {e}")
    return _sheet_update

# ================== Orquestación principal ==================

def process_back_questions_job(
    *,
    system_instructions_doc_id: str,
    base_prompt_doc_id: Optional[str],
    pdf_url: str,
    output_doc_id: str,
    drive_file_id: Optional[str],
    sampling_first_pages: int,
    sampling_last_pages: int,
    sheet_id: Optional[str] = None,
    row: Optional[int] = None,
    col: Optional[int] = None,
    additional_params: Dict[str, Any],
) -> Dict[str, Any]:
    logger.info("🏁 Back-Questions: inicio de job.")

    # Sheet progress helper
    _sheet_update = _make_sheet_updater(sheet_id, row, col)
    _sheet_update(status="10% Inicio")

    # Accesos mínimos
    for fid in (system_instructions_doc_id, output_doc_id):
        assert_sa_has_access(fid)

    # System + base prompt dinámico
    system_text = get_document_content(system_instructions_doc_id)
    visa_type = (additional_params or {}).get("visa_type")
    base_prompt_ids_from_req = (additional_params or {}).get("base_prompt_ids") or {}
    resolved_base_prompt_id = _resolve_base_prompt_doc_id(
        explicit_base_prompt_doc_id=base_prompt_doc_id,
        visa_type=visa_type,
        base_prompt_ids_from_req=base_prompt_ids_from_req,
    )
    if not resolved_base_prompt_id:
        raise ValueError("No se pudo resolver base_prompt_doc_id (ni explícito, ni por visa_type, ni por 'default').")
    assert_sa_has_access(resolved_base_prompt_id)
    base_prompt = get_document_content(resolved_base_prompt_id)
    _sheet_update(status="20% Prompts listos")

    # Resolver PDF (Drive) → bytes locales
    if pdf_url.startswith("gs://"):
        raise RuntimeError("Para muestreo por páginas con 'gs://', implemente descarga GCS o use Drive.")
    fid = drive_file_id or parse_drive_url_to_id(pdf_url)
    if not fid:
        raise ValueError("pdf_url no es gs:// y no se pudo extraer drive_file_id.")
    assert_sa_has_access(fid, use_docs_api=False)
    bytes_local = download_file_bytes(fid)
    _sheet_update(status="30% PDF descargado")

    # Si es chico, reutiliza pipeline existente (opcional; mantiene compat)
    reader = PdfReader(BytesIO(bytes_local))
    n_pages = len(reader.pages)
    if n_pages < 80:
        logger.info(f"PDF con {n_pages} páginas (<80). Usando pipeline existente.")
        from src.services.pdf_processing import process_pdf_documents
        resp = process_pdf_documents(
            system_instructions_doc_id=system_instructions_doc_id,
            base_prompt_doc_id=resolved_base_prompt_id,
            pdf_url=pdf_url,
            output_doc_id=output_doc_id,
            drive_file_id=drive_file_id,
            additional_params=additional_params or {},
        )
        # Escribimos link y estado final
        try:
            link = resp.get("output_doc_link")
        except Exception:
            link = f"https://docs.google.com/document/d/{output_doc_id}/edit"
        _sheet_update(status="100% ✔️", link=link)
        return resp

    # Sample P40 + U40 para detectar preguntas (solo TEXTO, sin GCS)
    take_first = max(1, sampling_first_pages or settings.backq_first_pages_default)
    take_last = max(1, sampling_last_pages or settings.backq_last_pages_default)

    logger.info(f"📄 PDF n={n_pages} páginas; sample first/last = {take_first}/{take_last}")
    sample_bytes = _extract_sample_pdf_bytes(bytes_local, take_first=take_first, take_last=take_last)
    sample_text = _extract_full_text(sample_bytes)
    hit = _first_heading_variant_hit(sample_text)
    logger.info(f"HEADINGS: primer patrón que hizo match = {hit!r}")
    _sheet_update(status="40% Muestra procesada")

    max_q = int((additional_params or {}).get("detect_limit") or settings.backq_detect_limit)
    questions = _detect_back_questions_via_model_text(sample_text, max_questions=max_q)
    _log_detected_questions("DET-ML", questions)
    if not questions:
        logger.warning("⚠️ Detector ML no devolvió preguntas. Probando fallback regex local sobre el sample…")
        questions = _detect_back_questions_regex(sample_bytes)[:max_q]
        _log_detected_questions("DET-REGEX", questions)

    if not questions:
        # Sin preguntas → escribir doc básico y salir
        write_qas_native(output_doc_id, title="Respuestas", qas=[])
        output_link = f"https://docs.google.com/document/d/{output_doc_id}/edit"
        _sheet_update(status="100% ✔️ (sin preguntas detectadas)", link=output_link)
        return {
            "status": "success",
            "message": "No se detectaron preguntas regreso en el documento.",
            "output_doc_link": output_link,
        }

    # Preparar PDF completo → SOLO TEXTO + textos por chunk (sin GCS)
    pages_per_chunk = max(5, settings.pdf_max_pages_per_chunk)
    chunk_texts = _split_pdf_to_text_chunks(bytes_local, pages_per_chunk)
    logger.info(f"Chunking: {len(chunk_texts)} chunks a ~{pages_per_chunk} páginas/chunk.")
    _sheet_update(status=f"50% {len(questions)} preguntas detectadas")

    strategy = (additional_params or {}).get("strategy") or settings.backq_strategy

    # --------- Estrategia híbrida (router + batch por chunk) ---------
    if strategy != "per_question":
        k_top = int((additional_params or {}).get("k_top_chunks") or settings.backq_k_top_chunks)
        min_cov = int((additional_params or {}).get("min_cover") or settings.backq_min_cover)
        cap = int((additional_params or {}).get("chunk_cap") or settings.backq_chunk_cap)
        throttle_s = float((additional_params or {}).get("throttle_s") or settings.backq_throttle_s)

        routing = _route_questions_to_chunks(
            questions=[{"id": q["id"], "text": q["text"], "page_hint": q.get("page_hint")} for q in questions],
            chunk_texts=chunk_texts,
            k_top=k_top,
            min_cover=min_cov,
            chunk_cap=cap,
        )
        _sheet_update(status="60% Ruteo de preguntas listo")

        # MAP por chunk (Flash/JSON) — ahora basado en TEXTO
        partials = defaultdict(list)  # qid -> [respuestas parciales]
        total_chunks = len(routing)
        done = 0
        for cidx, q_subset in routing.items():
            if not q_subset:
                continue
            logger.info(f"🗺️ MAP chunk {cidx}: {len(q_subset)} preguntas")
            try:
                out = _map_chunk_answers_json_from_text(chunk_texts[cidx], cidx, q_subset)
                for a in out.get("answers", []):
                    partials[a["id"]].append(a["answer"])
            except gex.ResourceExhausted:
                logger.warning(f"429 en MAP chunk {cidx}. Reintentando con subset reducido…")
                # Degradación: mitad de preguntas
                small_subset = q_subset[: max(1, len(q_subset) // 2)]
                try:
                    out = _map_chunk_answers_json_from_text(chunk_texts[cidx], cidx, small_subset)
                    for a in out.get("answers", []):
                        partials[a["id"]].append(a["answer"])
                except Exception as e2:
                    logger.warning(f"MAP chunk {cidx} degradado también falló: {e2}")
            except Exception as e:
                logger.warning(f"MAP chunk {cidx} falló: {e}")

            time.sleep(throttle_s)
            done += 1
            # progreso entre 60% y 85% durante MAP
            pct = 60 + int(25 * (done / max(1, total_chunks)))
            _sheet_update(status=f"{pct}% MAP {done}/{total_chunks}")

        # REDUCE por pregunta (Pro)
        _sheet_update(status="90% REDUCE por pregunta")
        qas: List[Dict[str, str]] = []
        missing: List[Dict[str, Any]] = []
        for q in questions:
            qid, qtext = q["id"], q["text"]
            candidates = partials.get(qid, [])
            final_ans = _reduce_answers_for_question(system_text, base_prompt, qtext, candidates)
            if "No se encontró evidencia" in final_ans:
                missing.append(q)
            qas.append({"question": qtext, "answer": final_ans})

        # Fallback dirigido Top-2 (opcional, barato)
        if missing:
            logger.info(f"🛟 Fallback: {len(missing)} preguntas sin candidatos. Intento dirigido Top-2.")
            for q in missing:
                top_idx = _select_topk_chunks_for_question(q["text"], chunk_texts, k=2) or [0]
                sel_txt = [chunk_texts[i] for i in top_idx]
                try:
                    ans = _answer_one_question_over_text_chunks(
                        question_text=q["text"],
                        system_text=system_text,
                        base_prompt=base_prompt,
                        selected_chunk_texts=sel_txt,
                        params={},
                    )
                    for qa in qas:
                        if qa["question"] == q["text"] and "No se encontró evidencia" in qa["answer"]:
                            qa["answer"] = ans
                            break
                except Exception as e:
                    logger.warning(f"Fallback dirigido falló para {q['id']}: {e}")

        write_qas_native(output_doc_id, title="Respuestas", qas=qas)
        output_link = f"https://docs.google.com/document/d/{output_doc_id}/edit"
        logger.info("✅ Back-Questions completado (híbrido).")
        _sheet_update(status="95% Escribiendo Doc", link=output_link)
        _sheet_update(status="100% ✔️")
        return {
            "status": "success",
            "message": "Q/A escritos en el documento (modo híbrido).",
            "output_doc_link": output_link,
        }

    # --------- Fallback: per-pregunta (más lento) ---------
    qas: List[Dict[str, str]] = []
    throttle_s = float((additional_params or {}).get("throttle_s") or settings.backq_throttle_s)
    for idx, q in enumerate(questions, 1):
        q_text = q["text"]
        # Reducimos contexto por pregunta: Top-K chunks más relevantes
        top_idx = _select_topk_chunks_for_question(q_text, chunk_texts, k=3) or [0]
        selected_texts = [chunk_texts[i] for i in top_idx]
        logger.info(f"→ Respondiendo ({idx}/{len(questions)}): {q_text[:80]}… (chunks {top_idx})")
        try:
            ans = _answer_one_question_over_text_chunks(
                question_text=q_text,
                system_text=system_text,
                base_prompt=base_prompt,
                selected_chunk_texts=selected_texts,
                params={**(additional_params or {})},
            )
        except gex.ResourceExhausted:
            logger.warning(f"429 en pregunta {idx}. Reintentando con solo 1 chunk…")
            try:
                ans = _answer_one_question_over_text_chunks(
                    question_text=q_text,
                    system_text=system_text,
                    base_prompt=base_prompt,
                    selected_chunk_texts=selected_texts[:1],
                    params={**(additional_params or {})},
                )
            except Exception as e2:
                logger.error(f"❌ Pregunta {idx} falló tras degradación: {e2}")
                ans = "_(No se pudo responder por límite temporal de cuota; intente más tarde)_"
        except Exception as e:
            logger.error(f"❌ Error en pregunta {idx}: {e}")
            ans = "_(error al procesar esta pregunta)_"

        qas.append({"question": q_text, "answer": ans})
        time.sleep(throttle_s)
        # progreso entre 60% y 95% en modo per_question
        pct = 60 + int(35 * (idx / max(1, len(questions))))
        _sheet_update(status=f"{pct}% ({idx}/{len(questions)})")

    write_qas_native(output_doc_id, title="Respuestas", qas=qas)
    output_link = f"https://docs.google.com/document/d/{output_doc_id}/edit"
    logger.info("✅ Back-Questions completado (per_question).")
    _sheet_update(status="95% Escribiendo Doc", link=output_link)
    _sheet_update(status="100% ✔️")
    return {
        "status": "success",
        "message": "Q/A escritos en el documento (modo per_question).",
        "output_doc_link": output_link,
    }
