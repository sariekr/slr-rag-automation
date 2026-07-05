"""LLM layer (Stage 5): OpenRouter client + prompt templates."""

from .client import LLMResponse, call_llm
from .prompts import (
    SYSTEM_ROLE,
    screening_prompt,
    extraction_prompt,
    extraction_prompt_strict,
    extraction_prompt_v3,
    extraction_prompt_v4,
    extraction_prompt_v5,
    synthesis_prompt,
)
from .parsing import parse_json, first_json_object

__all__ = [
    "LLMResponse",
    "call_llm",
    "SYSTEM_ROLE",
    "screening_prompt",
    "extraction_prompt",
    "extraction_prompt_strict",
    "extraction_prompt_v3",
    "extraction_prompt_v4",
    "extraction_prompt_v5",
    "synthesis_prompt",
    "parse_json",
    "first_json_object",
]
