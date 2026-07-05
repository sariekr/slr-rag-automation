"""Characterization tests for LLM response parsing (pure logic, no LLM/network).

Covers the two functions that are the ONLY path from raw model output to a
parsed dict, and that the screening / extraction tasks depend on:
  src.llm.parsing.first_json_object  -> string-aware first balanced {...}
  src.llm.parsing.parse_json         -> fence stripping + balanced-object fallback

These are CHARACTERIZATION tests: they lock in the current behavior so a later
refactor cannot silently change how a model response becomes parsed / parse_failed
without a test going red. They assert what the code does today; they do not propose
new behavior.
"""

from src.llm.parsing import first_json_object, parse_json


# ── first_json_object: string-aware balanced-brace scan ──────────────────────

def test_first_object_returns_simple_object():
    assert first_json_object('{"a": 1}') == '{"a": 1}'


def test_first_object_none_when_no_brace():
    assert first_json_object("no braces here") is None


def test_first_object_none_when_unbalanced():
    # Opening brace never closes -> no complete object.
    assert first_json_object('{"a": 1') is None


def test_first_object_picks_first_of_two():
    assert first_json_object('{"a": 1} {"b": 2}') == '{"a": 1}'


def test_first_object_returns_full_nested_outer():
    assert first_json_object('{"a": {"b": 1}}') == '{"a": {"b": 1}}'


def test_first_object_ignores_braces_inside_string():
    # Braces inside a JSON string value must not affect depth counting.
    assert first_json_object('{"k": "a { b } c"}') == '{"k": "a { b } c"}'


def test_first_object_handles_escaped_quote_in_string():
    # The string contains an escaped quote; the scan must stay in-string until
    # the real closing quote, then close the object.
    s = '{"k": "a\\"b"}'
    assert first_json_object(s) == s


def test_first_object_skips_leading_prose():
    assert first_json_object('answer: {"x": 1} end') == '{"x": 1}'


# ── parse_json: fence stripping + fallback ───────────────────────────────────

def test_parse_plain_json():
    assert parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_fenced_block():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_plain_fenced_block_without_json_tag():
    assert parse_json('```\n{"a": 1}\n```') == {"a": 1}


def test_parse_none_returns_none():
    assert parse_json(None) is None


def test_parse_empty_returns_none():
    assert parse_json("") is None


def test_parse_garbage_returns_none():
    assert parse_json("totally not json") is None


def test_parse_object_embedded_in_prose():
    # Not fenced and not bare JSON -> json.loads fails, balanced-object fallback fires.
    assert parse_json('Result: {"decision": "include"} done') == {"decision": "include"}


def test_parse_duplicated_fenced_blocks_picks_first():
    # The documented long-context case: the answer emitted twice. Fence strip
    # leaves invalid JSON, so the fallback returns the FIRST balanced object.
    twice = '```json\n{"decision": "include"}\n``` ```json\n{"decision": "exclude"}\n```'
    assert parse_json(twice) == {"decision": "include"}


def test_parse_two_bare_objects_picks_first():
    assert parse_json('{"a": 1} {"b": 2}') == {"a": 1}


def test_parse_keeps_braces_inside_string_value():
    assert parse_json('{"text": "use { and }"}') == {"text": "use { and }"}
