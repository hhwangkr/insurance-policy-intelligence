from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, ValidationError

from insurance_ai_generation.answer_prompt import GroundedAnswerPrompt
from insurance_ai_generation.grounded_answer import (
    GroundedAnswer,
    GroundedAnswerValidation,
    extract_citation_ids_from_text,
    grounded_answer_json_schema,
    validate_answer_citations,
)
from insurance_ai_generation.llm_provider import LLMProvider, LLMRequest


class GroundedAnswerGenerationResult(BaseModel):
    """Outcome of one grounded generation attempt (parse + local citation checks)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    answer: GroundedAnswer
    validation: GroundedAnswerValidation
    provider_name: str
    model_name: str | None = None
    raw_text: str


def parse_provider_text_to_grounded_answer(raw: str) -> GroundedAnswer:
    """Parse provider output as JSON (``answer`` / ``citations_used`` / ``insufficient_context``) or
    plain text with ``[Cn]`` citation inference.
    """
    stripped = raw.strip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(data, dict) and "answer" in data:
                try:
                    return GroundedAnswer.model_validate(data)
                except ValidationError:
                    pass

    ids = extract_citation_ids_from_text(stripped)
    return GroundedAnswer(
        answer=stripped,
        citations_used=sorted(ids),
        insufficient_context=False,
    )


def generate_grounded_answer(
    prompt: GroundedAnswerPrompt,
    provider: LLMProvider,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> GroundedAnswerGenerationResult:
    """Call ``provider`` with ``prompt.messages``, parse JSON or plaintext, validate citations."""
    request = LLMRequest(
        messages=list(prompt.messages),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_schema=grounded_answer_json_schema(),
    )
    response = provider.complete(request)
    answer = parse_provider_text_to_grounded_answer(response.text)
    allowed = set(prompt.citation_ids)
    validation = validate_answer_citations(answer, allowed)
    return GroundedAnswerGenerationResult(
        answer=answer,
        validation=validation,
        provider_name=response.provider_name,
        model_name=response.model_name,
        raw_text=response.text,
    )
