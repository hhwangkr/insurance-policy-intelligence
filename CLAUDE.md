```md
# Insurance AI Copilot - Engineering Guidelines

## Project Overview

This repository contains a production-grade insurance AI platform focused on:

- insurance policy intelligence
- retrieval-augmented generation (RAG)
- citation-grounded reasoning
- agentic workflows
- evaluation and observability

This is NOT a toy chatbot project.

The goal is to build an enterprise-grade AI system that resembles real-world applied AI infrastructure.

---

# Engineering Principles

- Prioritize clean architecture over quick hacks.
- Prefer explicitness over hidden abstractions.
- Strong typing is mandatory.
- All business logic must be testable.
- Avoid tightly coupled code.
- Prefer modular and composable systems.
- Design for maintainability and observability from the beginning.
- Hallucination reduction is a core requirement.
- Every AI-generated response must be traceable and citation-grounded.

---

# Tech Stack

## Backend
- Python
- FastAPI
- Pydantic v2

## AI Orchestration
- LangGraph

## Retrieval
- Qdrant
- Hybrid retrieval (BM25 + vector search)

## Infrastructure
- Docker
- Docker Compose

## Frontend
- Next.js

## Observability
- OpenTelemetry

---

# Code Style Requirements

- Use type hints everywhere.
- Use Pydantic models for structured data.
- Avoid raw dictionaries whenever possible.
- Keep API routers thin.
- Place business logic inside services/modules.
- Prefer composition over inheritance.
- Prefer pure functions when possible.
- Avoid giant files.
- Use descriptive naming.
- Minimize unnecessary comments.

---

# AI Architecture Requirements

- All LLM outputs must use structured outputs.
- Retrieval, reranking, and reasoning layers must be separated.
- Agent state transitions must be explicit.
- Avoid hidden prompt chains.
- Prompts should live in dedicated files/modules.
- All AI responses must support citations.

---

# RAG Requirements

- Use hybrid retrieval.
- Support semantic chunking.
- Preserve document hierarchy and metadata.
- Support clause-level citations.
- Retrieval quality is more important than generation style.

---

# Evaluation Requirements

Design evaluation pipelines early.

Track:
- hallucination rate
- retrieval accuracy
- citation correctness
- latency

Every major workflow should be benchmarkable.

---

# Frontend Requirements

- Keep UI minimal and functional.
- Prioritize workflow visualization over visual polish.
- Display citations clearly.
- Display reasoning traces when appropriate.

---

# Development Workflow

Before writing code:
1. Propose architecture
2. Propose file structure
3. Explain tradeoffs
4. Then implement incrementally

Avoid giant rewrites.

Prefer iterative development.

When uncertain:
- ask clarifying questions
- do not guess requirements

---

# Important

This project should resemble a real enterprise AI platform,
not a demo chatbot.

Optimize for:
- maintainability
- extensibility
- observability
- reliability
- evaluation
- production readiness
```