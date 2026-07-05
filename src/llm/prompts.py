"""
Prompt templates for the three tasks.

Each template carries six components: system
role, context, task definition, output format, constraints, and the
source-citation requirement.

Language: English. The corpus is English and the baseline ground truth
(slr/data/extraction/extracted_data.json) was produced with English prompts, so
English keeps prompt language from confounding the direct-vs-rag comparison
(construct validity). The structure mirrors the baseline extraction prompt in
slr/scripts/09b_consensus_extraction.py (role / rules / strict-JSON output).

Arm difference (the ONLY varying factor is retrieval):
  - direct arm: NO retrieved-context block (zero-shot / whole paper)
  - rag arm:    includes retrieved context (few-shot examples or relevant chunks)
"""

SYSTEM_ROLE = (
    "You are a systematic literature review assistant. You apply the given "
    "criteria exactly and never state information that is not supported by the "
    "provided text."
)


def screening_prompt(paper_title: str, paper_abstract: str, ic_ec_criteria: str,
                     examples: list | None = None) -> str:
    """Include/exclude decision prompt.

    examples=None        -> direct (zero-shot) arm.
    examples=[{...}, ...] -> rag arm; each dict has title, abstract, decision
                            (a similar paper already decided by human reviewers).
    """
    parts = [
        "TASK: Decide whether the paper below should be INCLUDED in or EXCLUDED "
        "from a systematic literature review by applying the criteria.",
        "",
        "INCLUSION / EXCLUSION CRITERIA:",
        ic_ec_criteria.strip(),
    ]
    if examples:
        parts += ["", "EXAMPLES (similar papers already decided by human reviewers):"]
        for i, ex in enumerate(examples, 1):
            parts += [
                f"Example {i}:",
                f"  Title: {ex.get('title', '')}",
                f"  Abstract: {ex.get('abstract', '')}",
                f"  Decision: {str(ex.get('decision', '')).upper()}",
            ]
    parts += [
        "",
        "PAPER TO DECIDE:",
        f"Title: {paper_title}",
        f"Abstract: {paper_abstract}",
        "",
        "CONSTRAINTS:",
        "- Base the decision ONLY on the criteria and the paper's title/abstract.",
        "- If the evidence is insufficient to justify inclusion, decide EXCLUDE.",
        "",
        "OUTPUT FORMAT (strict JSON, no markdown):",
        '{"decision": "include" | "exclude", '
        '"reason": "<one sentence naming the criterion that drove the decision>"}',
    ]
    return "\n".join(parts)


def extraction_prompt(fields: list[str], paper_text: str | None = None,
                      retrieved_context: str | None = None,
                      field_hints: dict[str, str] | None = None) -> str:
    """Field-extraction prompt (output JSON with a per-field source span).

    paper_text         -> direct arm (whole paper in context).
    retrieved_context  -> rag arm (relevant chunks + cross-paper context).
    field_hints        -> optional per-field value guidance, e.g. an allowed
                          taxonomy ("choose one of: ..."). Closed fields like
                          domain/hybrid_type need this so the model emits
                          canonical labels (matching how the reference labels
                          were built); free text cannot be normalized post-hoc.
    The per-field "source" quote grounds each value so it can be cross-verified
    against the text (hallucination measurement).
    """
    field_hints = field_hints or {}

    def _value_ph(f: str) -> str:
        return f"<{field_hints[f]}>" if f in field_hints else "<extracted value or null>"

    field_lines = ",\n".join(
        f'  "{f}": {{"value": {_value_ph(f)}, '
        f'"source": "<short verbatim quote from the text, or null>"}}'
        for f in fields
    )
    if retrieved_context is not None:
        source_label = "RETRIEVED CONTEXT (relevant chunks from this and related papers):"
        source_body = retrieved_context
    else:
        source_label = "PAPER TEXT:"
        source_body = paper_text or ""

    return "\n".join([
        "TASK: Extract the requested fields from the source below into a JSON object.",
        "",
        "RULES:",
        "- Extract ONLY information explicitly stated in the source.",
        "- If a field is not mentioned, set its \"value\" to null.",
        "- For every field, put a short verbatim quote in \"source\" that supports "
        "the value (null when the value is null).",
        "- Use exact names as written (models, databases, metrics, frameworks).",
        "",
        "OUTPUT FORMAT (strict JSON, no markdown):",
        "{",
        field_lines,
        "}",
        "",
        source_label,
        source_body,
    ])


def extraction_prompt_strict(fields: list[str], paper_text: str | None = None,
                             retrieved_context: str | None = None,
                             field_hints: dict[str, str] | None = None) -> str:
    """Stricter extraction prompt (v2) addressing observed failure modes:
      - multi-value fields (llm_used etc.) need explicit "list ALL, comma-separated"
      - taxonomy fields need exact-label-format spelling rules
      - source quote needs verbatim-length specification (5-25 words)
      - null preference for ambiguous cases
      - one positive + one negative few-shot mini-example inside RULES (identical
        in both arms so the arm-differentiation invariant is preserved)

    Arm difference (still the ONLY varying factor):
      - direct arm: paper_text in the source block
      - rag arm:    retrieved_context in the source block
    """
    field_hints = field_hints or {}

    def _value_ph(f: str) -> str:
        if f in field_hints:
            return f"<{field_hints[f]}>"
        return "<extracted value or null>"

    field_lines = ",\n".join(
        f'  "{f}": {{"value": {_value_ph(f)}, '
        f'"source": "<5-25 contiguous verbatim words from the source, or null>"}}'
        for f in fields
    )
    if retrieved_context is not None:
        source_label = "RETRIEVED CONTEXT (relevant chunks from this paper):"
        source_body = retrieved_context
    else:
        source_label = "PAPER TEXT:"
        source_body = paper_text or ""

    return "\n".join([
        "TASK: Extract the requested fields from the source below into a JSON object.",
        "",
        "RULES:",
        "1. EXTRACT ONLY information EXPLICITLY stated in the source. "
        "Do NOT guess, infer, or use outside knowledge.",
        "2. If a field is genuinely not mentioned in the source, set its "
        "\"value\" to null. When in doubt between a plausible value and null, "
        "prefer null.",
        "3. MULTI-VALUE FIELDS (e.g. multiple LLMs, vector DBs, embedding "
        "models used in the system): list ALL distinct items mentioned, "
        "comma-separated, lowercase, using the FULL names exactly as written "
        "in the source. No abbreviations, no deduplication of similar names.",
        "   Good: \"gpt-4, llama-2-7b, gemini-1.5-pro\"",
        "   Bad:  \"GPT and LLaMA\"  (abbreviated, capitalised, missing versions)",
        "4. TAXONOMY FIELDS: pick from the ALLOWED LIST ONLY, using the EXACT "
        "label spelling (lowercase, underscores where shown, e.g. "
        "\"scientific_research\", \"history_culture\", \"dense_sparse\"). For "
        "multi-label taxonomies, list ALL applicable labels comma-separated.",
        "5. SOURCE QUOTE: copy 5-25 CONTIGUOUS words verbatim from the source "
        "that JUSTIFY the value. No paraphrasing, no ellipsis (\"...\"), no "
        "splicing across non-adjacent sentences. If value is null, source is null.",
        "6. Use exact names as they appear in the source (model versions, "
        "database names, framework spellings).",
        "",
        "OUTPUT FORMAT (strict JSON, no markdown, no code fences):",
        "{",
        field_lines,
        "}",
        "",
        source_label,
        source_body,
    ])


def extraction_prompt_v3(fields: list[str], paper_text: str | None = None,
                         retrieved_context: str | None = None,
                         field_hints: dict[str, str] | None = None) -> str:
    """Exhaustive-enumeration prompt (v3) targeting the dominant GENERATION error.

    Diagnosis: on multi-value fields (llm_used, embedding_model, vector_db) the
    model UNDER-lists: it extracts only ~half the items whose evidence is already
    in context (measured extract-when-present rate ~0.50). v2 added a "list ALL"
    rule but ALSO a "prefer null when in doubt" rule; the null preference (not the
    enumeration) drove v2's regression. v3 = v1 + ONLY the exhaustive-enumeration
    rule, keeping v1's neutral null handling, to isolate the exhaustiveness lever.

    Arm difference stays the ONLY varying factor (retrieved_context vs paper_text).
    """
    field_hints = field_hints or {}

    def _value_ph(f: str) -> str:
        return f"<{field_hints[f]}>" if f in field_hints else "<extracted value or null>"

    field_lines = ",\n".join(
        f'  "{f}": {{"value": {_value_ph(f)}, '
        f'"source": "<short verbatim quote from the text, or null>"}}'
        for f in fields
    )
    if retrieved_context is not None:
        source_label = "RETRIEVED CONTEXT (relevant chunks from this paper):"
        source_body = retrieved_context
    else:
        source_label = "PAPER TEXT:"
        source_body = paper_text or ""

    return "\n".join([
        "TASK: Extract the requested fields from the source below into a JSON object.",
        "",
        "RULES:",
        "- Extract ONLY information explicitly stated in the source. If a field is "
        "not mentioned, set its \"value\" to null.",
        "- MULTI-VALUE FIELDS (large language models, embedding models, vector "
        "databases): the source usually names SEVERAL. Scan the ENTIRE source and "
        "list EVERY distinct item mentioned, comma-separated, with full names "
        "exactly as written. Naming only some of the items that are present "
        "(under-listing) is the most common error here, so be exhaustive. Do NOT "
        "add items that are not in the source.",
        "- Use exact names as written (model versions, database names, frameworks).",
        "- For every field, put a short verbatim quote in \"source\" that supports "
        "the value (null when the value is null).",
        "",
        "OUTPUT FORMAT (strict JSON, no markdown):",
        "{",
        field_lines,
        "}",
        "",
        source_label,
        source_body,
    ])


def extraction_prompt_v4(fields: list[str], paper_text: str | None = None,
                         retrieved_context: str | None = None,
                         field_hints: dict[str, str] | None = None) -> str:
    """Field-scoped exhaustive prompt (v4). v3 showed a BLANKET 'be exhaustive'
    rule fixes under-listing on open named-entity fields (llm_used +0.08/+0.11,
    embedding_model +0.04/+0.02) but OVER-lists on taxonomy/single-value fields
    (domain/hybrid_type/vector_db each regressed). v4 scopes exhaustiveness to the
    open named-entity fields (llm_used, embedding_model) and makes taxonomy
    assignment conservative, to keep the multi-value gain without the regression.

    Arm difference stays the ONLY varying factor (retrieved_context vs paper_text).
    """
    field_hints = field_hints or {}

    def _value_ph(f: str) -> str:
        return f"<{field_hints[f]}>" if f in field_hints else "<extracted value or null>"

    field_lines = ",\n".join(
        f'  "{f}": {{"value": {_value_ph(f)}, '
        f'"source": "<short verbatim quote from the text, or null>"}}'
        for f in fields
    )
    if retrieved_context is not None:
        source_label = "RETRIEVED CONTEXT (relevant chunks from this paper):"
        source_body = retrieved_context
    else:
        source_label = "PAPER TEXT:"
        source_body = paper_text or ""

    return "\n".join([
        "TASK: Extract the requested fields from the source below into a JSON object.",
        "",
        "RULES:",
        "- Extract ONLY information explicitly stated in the source. If a field is "
        "not mentioned, set its \"value\" to null.",
        "- llm_used and embedding_model: the source usually names SEVERAL. List "
        "EVERY distinct one mentioned, comma-separated, with full names exactly as "
        "written. Under-listing (naming only some of the items present) is the most "
        "common error on these two fields, so be exhaustive. Do NOT add items not "
        "in the source.",
        "- vector_db: name the vector database(s) actually used (usually one or "
        "two); do not pad with stores that are not used in this system.",
        "- domain: assign EXACTLY ONE application domain (the primary one); do not "
        "list several.",
        "- hybrid_type: assign ONLY the retrieval-type label(s) the source clearly "
        "supports; do NOT over-assign extra labels to be safe.",
        "- Use exact names as written. For every field put a short verbatim quote "
        "in \"source\" that supports the value (null when the value is null).",
        "",
        "OUTPUT FORMAT (strict JSON, no markdown):",
        "{",
        field_lines,
        "}",
        "",
        source_label,
        source_body,
    ])


def extraction_prompt_v5(fields: list[str], paper_text: str | None = None,
                         retrieved_context: str | None = None,
                         field_hints: dict[str, str] | None = None) -> str:
    """Quote-first variant of v1 (evidence-before-value generation order).

    Identical to extraction_prompt (v1) except the per-field JSON emits the
    verbatim "source" quote BEFORE the "value", and the rules make the model
    work in two explicit steps (locate the evidence, then derive the value
    from it). Targets the generation-bound failure mode the ceiling
    decomposition isolated (evidence present in context, value not extracted):
    writing the quote first forces evidence localisation before answering.

    Arm difference stays the ONLY varying factor (retrieved_context vs
    paper_text).
    """
    field_hints = field_hints or {}

    def _value_ph(f: str) -> str:
        return f"<{field_hints[f]}>" if f in field_hints else "<extracted value or null>"

    field_lines = ",\n".join(
        f'  "{f}": {{"source": "<verbatim quote from the text, or null>", '
        f'"value": {_value_ph(f)}}}'
        for f in fields
    )
    if retrieved_context is not None:
        source_label = "RETRIEVED CONTEXT (relevant chunks from this paper):"
        source_body = retrieved_context
    else:
        source_label = "PAPER TEXT:"
        source_body = paper_text or ""

    return "\n".join([
        "TASK: Extract the requested fields from the source below into a JSON object.",
        "",
        "RULES:",
        "- Work in two steps for EACH field. STEP 1: scan the source and copy the "
        "exact sentence(s) containing the answer into \"source\" (verbatim; if the "
        "evidence spans several sentences, join them with ' ... '). STEP 2: derive "
        "\"value\" from the quoted evidence.",
        "- Extract ONLY information explicitly stated in the source.",
        "- If a field is not mentioned, set both \"source\" and \"value\" to null.",
        "- Use exact names as written (models, databases, metrics, frameworks).",
        "",
        "OUTPUT FORMAT (strict JSON, no markdown):",
        "{",
        field_lines,
        "}",
        "",
        source_label,
        source_body,
    ])


def synthesis_prompt(research_question: str, evidence_chunks: list[str]) -> str:
    """Narrative synthesis prompt: answer one RQ from cross-corpus evidence chunks."""
    numbered = "\n\n".join(evidence_chunks)  # each excerpt already prefixed with its [paper_id]
    return "\n".join([
        "TASK: Write an evidence-grounded narrative answer to the research question "
        "below, synthesizing across the evidence excerpts.",
        "",
        f"RESEARCH QUESTION: {research_question}",
        "",
        "CONSTRAINTS:",
        "- Use ONLY the evidence excerpts; do not add outside knowledge.",
        "- Each excerpt begins with its source paper id in brackets, e.g. [P016].",
        "- Support every claim with an inline citation to that paper id, e.g. [P016] "
        "(or [P016, P018] when several support it). Do NOT invent any other numbering.",
        "- If the evidence is insufficient for part of the question, say so explicitly.",
        "- Write 1-3 cohesive paragraphs (no bullet lists).",
        "",
        "EVIDENCE EXCERPTS:",
        numbered,
    ])
