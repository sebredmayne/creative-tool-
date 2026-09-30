"""Parses plain-text and Markdown files into text chunks. No structured/tabular data."""
from core.ingestion.base import Parser
from core.ingestion.chunking import chunk_text
from core.models import ParsedDocument


class TextParser(Parser):
    def parse(self, file, filename: str) -> ParsedDocument:
        raw = file.read()
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        return ParsedDocument(filename=filename, file_type="text", text_chunks=chunk_text(text), dataframes=None)
