from __future__ import annotations

import argparse

import uvicorn

from insurance_ai_api.config import get_settings


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run the Insurance AI API server.")
    parser.add_argument("--host", default=settings.api_host)
    parser.add_argument("--port", type=int, default=settings.api_port)
    args = parser.parse_args()
    uvicorn.run(
        "insurance_ai_api.main:app",
        host=args.host,
        port=args.port,
        factory=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
