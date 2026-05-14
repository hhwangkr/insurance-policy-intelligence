"""Grounded answer prompt construction, provider contracts, and answer validation."""

from insurance_ai_generation.answer_prompt import (
    ChatMessage,
    GroundedAnswerPrompt,
    build_grounded_answer_prompt,
    format_grounded_answer_prompt_json,
)
from insurance_ai_generation.generate_grounded_answer import (
    GroundedAnswerGenerationResult,
    generate_grounded_answer,
    parse_provider_text_to_grounded_answer,
)
from insurance_ai_generation.grounded_answer import (
    GroundedAnswer,
    GroundedAnswerValidation,
    extract_citation_ids_from_text,
    grounded_answer_json_schema,
    grounded_answer_json_schema_for_citations,
    render_citation_summary,
    render_grounded_answer_with_citations,
    validate_answer_citations,
)
from insurance_ai_generation.inspect_grounded_answer import (
    CitationContextPreview,
    GroundedAnswerInspection,
    build_grounded_answer_inspection,
    generation_result_from_generate_answer_json,
    inspection_report_to_markdown,
)
from insurance_ai_generation.llm_provider import (
    LLMProvider,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    StaticLLMProvider,
)
from insurance_ai_generation.provider_registry import (
    GenerationProviderConfig,
    create_llm_provider,
)
from insurance_ai_generation.providers.ollama_provider import (
    DEFAULT_OLLAMA_BASE_URL,
    OLLAMA_DEFAULT_MODEL_EXAMPLE,
    OllamaProvider,
)

__all__ = [
    "CitationContextPreview",
    "DEFAULT_OLLAMA_BASE_URL",
    "ChatMessage",
    "GenerationProviderConfig",
    "GroundedAnswer",
    "GroundedAnswerGenerationResult",
    "GroundedAnswerInspection",
    "GroundedAnswerPrompt",
    "GroundedAnswerValidation",
    "LLMProvider",
    "LLMProviderError",
    "LLMRequest",
    "LLMResponse",
    "OLLAMA_DEFAULT_MODEL_EXAMPLE",
    "OllamaProvider",
    "StaticLLMProvider",
    "build_grounded_answer_inspection",
    "build_grounded_answer_prompt",
    "create_llm_provider",
    "extract_citation_ids_from_text",
    "format_grounded_answer_prompt_json",
    "generate_grounded_answer",
    "generation_result_from_generate_answer_json",
    "grounded_answer_json_schema",
    "grounded_answer_json_schema_for_citations",
    "inspection_report_to_markdown",
    "parse_provider_text_to_grounded_answer",
    "render_citation_summary",
    "render_grounded_answer_with_citations",
    "validate_answer_citations",
]
