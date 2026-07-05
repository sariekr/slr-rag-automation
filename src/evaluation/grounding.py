"""Deterministic source-grounding check (no LLM-judge, no RAGAS).

A field's extracted value carries a verbatim "source" quote. The value is
grounded when that quote is genuinely a CONTIGUOUS span of the source text
(exact substring, or >=50% of its 5-word windows present), or when the value
itself appears verbatim. A non-null value whose source is not contiguously in
the text = fabricated evidence = hallucination.

Why contiguous windows (not scattered tokens): in a ~7k-word paper almost any
phrase's individual words appear somewhere, so a token-overlap test trivially
passes. Contiguous 5-word windows tolerate minor paraphrase but reject fabricated
quotes. Centralizing this deterministic check here lets the product verify
grounding on its own path, not only in offline evaluation scripts.

Pure / dependency-free so it can be unit-tested without an LLM call.
"""


# Separator set mirrors normalize._norm so a hyphenated identifier in a source
# quote ("GPT-3.5", "e5-base-v2") matches the same name spelled with a space/dot
# in the paper ("gpt 3.5"). Without this the grounding check counted correct
# dashed model/embedding names as fabricated, inflating the hallucination rate.
_SEPARATORS = ("\u2013", "\u2014", "-", "_", ",", ".", "/")


def normalize_text(s) -> str:
    """Lowercase, normalize separators to spaces, collapse whitespace: the
    canonical form for the contiguous-span substring checks. Applied identically
    to the source quote, the value, and the paper text (the caller normalizes the
    paper text with this same function), so the comparison stays consistent."""
    s = str(s or "").lower()
    for ch in _SEPARATORS:
        s = s.replace(ch, " ")
    return " ".join(s.split())


def grounded(source, value, text_norm: str) -> bool:
    """True if `source` (or `value`) is a contiguous span of the normalized text.

    text_norm must already be normalize_text()-ed (the caller normalizes the
    source text once and reuses it across fields).
    """
    s = normalize_text(source)
    if s:
        if s in text_norm:
            return True
        words = s.split()
        if len(words) >= 5:
            wins = [" ".join(words[i:i + 5]) for i in range(len(words) - 4)]
            if sum(1 for w in wins if w in text_norm) / len(wins) >= 0.5:
                return True
    v = normalize_text(value)
    return len(v) >= 4 and v in text_norm
