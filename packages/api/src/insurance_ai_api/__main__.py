from __future__ import annotations


def main() -> None:
    import uvicorn

    uvicorn.run("insurance_ai_api.main:app", host="127.0.0.1", port=8000, reload=True)
