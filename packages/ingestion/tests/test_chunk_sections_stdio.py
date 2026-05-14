from insurance_ai_ingestion.chunk_sections import configure_stdout_utf8 as chunk_stdout
from insurance_ai_shared.stdio_utf8 import configure_stdout_utf8


def test_chunk_sections_imports_shared_stdout_utf8_helper() -> None:
    assert chunk_stdout is configure_stdout_utf8
