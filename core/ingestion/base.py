"""Common interface every file parser implements.

Adding a new file type (e.g. DOCX, TXT) means writing one new class here
that implements `parse()` and registering it in `core/ingestion/__init__.py`
- nothing else in the app needs to change.
"""
from abc import ABC, abstractmethod

from core.models import ParsedDocument


class Parser(ABC):
    @abstractmethod
    def parse(self, file, filename: str) -> ParsedDocument:
        """Parse a file (path or file-like object) into a ParsedDocument."""
        raise NotImplementedError
