"""
Task 2: Data extraction (fields per paper).

A reference set of field labels (the canonical extracted data under the
configured data directory) supports scoring the extraction output.
Arms:
  - direct: whole paper in context (paper_text).
  - rag:    retrieved relevant chunks (retrieved_context).
Output: per-field {value, source} JSON, feeding field-level F1 and the
source-grounding hallucination check.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import EXTRACTED_DATA, LLM_MODEL
from src.llm import (call_llm, SYSTEM_ROLE, extraction_prompt, extraction_prompt_strict,
                     extraction_prompt_v3, extraction_prompt_v4)
from src.llm.parsing import parse_json


def load_gold():
    """Load the reference extraction labels as {paper_id: record} from the configured data file."""
    data = json.loads(Path(EXTRACTED_DATA).read_text(encoding="utf-8"))
    return {d["paper_id"]: d for d in data}


def format_context(hits):
    """Join retrieval hits (RetrievedChunk) into a context block for the rag arm."""
    return "\n\n".join(f"[{h.chunk_id}] {h.text}" for h in hits)


def get_value(obj, field):
    """Pull a field's value from a parsed extraction object ({field: {value, source}} or flat)."""
    v = (obj or {}).get(field)
    return v.get("value") if isinstance(v, dict) else v


def extraction_status(parsed, finish_reason: str = "") -> str:
    """Classify an extraction outcome so a failed parse is never silently all-null.

    'ok'           -> parsed JSON available.
    'truncated'    -> no parse AND the model hit its token cap (finish_reason='length'):
                      the output was cut off mid-JSON, not a genuine all-null record.
    'parse_failed' -> no parse for any other reason (malformed JSON).
    Callers surface this in the result + UI instead of emitting a silent {field: null}.
    """
    if parsed is not None:
        return "ok"
    return "truncated" if finish_reason == "length" else "parse_failed"


_PROMPT_BUILDERS = {
    "v1": extraction_prompt,            # original
    "v2": extraction_prompt_strict,     # stricter rules + null-preference; regressed in testing
    "v3": extraction_prompt_v3,         # exhaustive enumeration (all fields)
    "v4": extraction_prompt_v4,         # field-scoped exhaustive (best-measured)
}


def extract(fields, paper_text=None, retrieved_context=None, field_hints=None, model=LLM_MODEL,
            prompt="v4"):
    """Extract fields. paper_text -> direct arm; retrieved_context -> rag arm.

    field_hints injects allowed-value taxonomies for closed fields (domain, hybrid_type).
    prompt selects the extraction prompt; the default v4 (extraction_prompt_v4,
    field-scoped exhaustive) is the best-measured prompt: it counters multi-value
    under-listing on the named-entity fields while keeping taxonomy assignment
    conservative, and raised extraction accuracy over both v1 and the stricter v2
    (which regressed). The offline comparison benchmarks are run separately with an
    explicit prompt=v1, so this product default does not change those reported numbers.
    Returns (parsed_dict_or_None, response).
    """
    builder = _PROMPT_BUILDERS.get(prompt, extraction_prompt_v4)
    prompt_str = builder(fields, paper_text=paper_text,
                         retrieved_context=retrieved_context, field_hints=field_hints)
    r = call_llm(prompt_str, system=SYSTEM_ROLE, model=model)
    return parse_json(r.text), r
