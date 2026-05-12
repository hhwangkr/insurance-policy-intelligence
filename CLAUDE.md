# Insurance AI Copilot — Engineering Guidelines

## Project overview

This repository is a production-grade insurance AI platform focused on:

- Insurance policy intelligence
- Retrieval-augmented generation (RAG)
- Citation-grounded reasoning
- Agentic workflows
- Evaluation and observability

This is not a toy chatbot. The target is an enterprise-grade system that resembles real applied AI infrastructure: maintainable, observable, and grounded in sources.

---

## How we work (execution discipline)

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

These habits matter as much as stack choices. They apply to every change; the architecture sections below describe the **system**—not a license to overbuild a single task.

### 1. Think before coding

Do not assume. Do not hide confusion. Surface tradeoffs.

- Restate the goal and trace the smallest change that could work before touching files. Prefer a short plan over exploratory refactors.
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them—do not pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop, name what is confusing, and ask.

### 2. Simplicity first

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No flexibility or configurability that was not requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.
- Ask: would a senior engineer say this is overcomplicated? If yes, simplify.

### 3. Surgical changes

Touch only what you must. Clean up only your own mess.

- Do not “improve” adjacent code, comments, or formatting unless the task requires it.
- Do not refactor things that are not broken.
- Match existing style, even if you would do it differently.
- If you notice unrelated dead code, mention it—do not delete it unless asked.
- When your changes create orphans: remove imports, variables, or functions that **your** changes made unused. Do not remove pre-existing dead code unless asked.
- Test: every changed line should trace directly to the user’s request.

### 4. Goal-driven execution, success criteria, and verification

Define success criteria. Loop until verified.

- Turn tasks into verifiable goals, for example:
  - “Add validation” → write tests for invalid inputs, then make them pass.
  - “Fix the bug” → write a test that reproduces it, then make it pass.
  - “Refactor X” → ensure tests pass before and after.
- Before implementing, define what “done” means (behavior, tests, API shape, docs if required). If criteria are missing, ask or propose them, then hold the work to that bar.
- Strong criteria let you loop independently; weak criteria (“make it work”) require constant clarification.
- Run the relevant tests, linters, or manual checks the task implies. Do not claim completion without evidence that the success criteria are met.

For multi-step tasks, state a brief plan:

```text
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Iterative development is still the default: iterate in **small, verified steps**, not large speculative rewrites. If the goal shifts, stop and realign instead of accumulating diff.

When uncertain: ask clarifying questions; do not guess requirements.

**These guidelines are working if:** diffs contain fewer unnecessary changes, there are fewer rewrites from overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Engineering principles

- Prefer clean architecture and explicitness over hidden magic and quick hacks.
- Strong typing is mandatory; business logic must be testable.
- Prefer modular, composable systems; avoid tight coupling.
- Design for maintainability and observability from the start.
- Hallucination reduction is a core requirement: AI outputs must be traceable and citation-grounded.

---

## Tech stack

### Backend

- Python
- FastAPI
- Pydantic v2

### AI orchestration

- LangGraph

### Retrieval

- Qdrant
- Hybrid retrieval (BM25 + vector search)

### Infrastructure

- Docker
- Docker Compose

### Frontend

- Next.js

### Observability

- OpenTelemetry

---

## Code style

- Use type hints everywhere.
- Use Pydantic models for structured data; avoid raw dictionaries when a model fits.
- Keep API routers thin; place business logic in services or dedicated modules.
- Prefer composition over inheritance; prefer pure functions where practical.
- Avoid giant files; use descriptive naming; minimize unnecessary comments.

---

## AI architecture

- All LLM outputs must use structured outputs.
- Separate retrieval, reranking, and reasoning layers.
- Agent state transitions must be explicit; avoid hidden prompt chains.
- Keep prompts in dedicated files or modules.
- All AI responses must support citations.

---

## RAG

- Use hybrid retrieval.
- Support semantic chunking; preserve document hierarchy and metadata.
- Support clause-level citations.
- Retrieval quality matters more than stylistic generation.

---

## Evaluation

Design evaluation pipelines early. Track at least:

- Hallucination rate
- Retrieval accuracy
- Citation correctness
- Latency

Every major workflow should be benchmarkable.

---

## Frontend

- Keep the UI minimal and functional.
- Prioritize workflow visualization over visual polish.
- Display citations clearly; show reasoning traces when appropriate.

---

## Development workflow (when a larger change is warranted)

Before substantial implementation:

1. Propose architecture and file layout.
2. Explain tradeoffs.
3. Implement in small increments, each verifiable on its own.

Avoid giant rewrites unless explicitly agreed.

---

## North star

Optimize for maintainability, extensibility, observability, reliability, evaluation, and production readiness — in service of **correct, grounded** insurance intelligence, not demo polish alone.
