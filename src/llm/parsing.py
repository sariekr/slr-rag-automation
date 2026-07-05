"""
LLM response parsing: strip markdown fences, parse JSON, fall back to the first
balanced {...} object.

Shared by the task modules (screening / extraction). The first-balanced-object
fallback handles long-context responses that occasionally emit their answer twice
(```json{..}``` ```json{..}```), where a naive outermost-brace span would break.
"""

import json


def first_json_object(t):
    """Return the first complete balanced {...} substring (string-aware), or None."""
    start = t.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for k in range(start, len(t)):
        c = t[k]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return t[start:k + 1]
    return None


def parse_json(text):
    """Strip ```json fences and parse; fall back to the first balanced object. None on failure."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.endswith("```"):
            t = t[:-3]
        t = t.strip()
        if t.startswith("json"):
            t = t[4:].strip()
    try:
        return json.loads(t)
    except Exception:
        obj = first_json_object(t)
        if obj is not None:
            try:
                return json.loads(obj)
            except Exception:
                return None
        return None
