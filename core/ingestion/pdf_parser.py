"""Parses PDF files into text chunks. PDFs have no structured/tabular data."""
from pypdf import PdfReader

from core.ingestion.base import Parser
from core.ingestion.chunking import chunk_text
from core.models import ParsedDocument


class PDFParser(Parser):
    def parse(self, file, filename: str) -> ParsedDocument:
        reader = PdfReader(file)
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return ParsedDocument(
            filename=filename,
            file_type="pdf",
            text_chunks=chunk_text(full_text),
            dataframes=None,
        )
