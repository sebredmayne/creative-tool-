"""Ingestion layer: turns uploaded files into ParsedDocuments.

`get_parser` is the single dispatch point the rest of the app should use.
To support a new file type, add its extension + Parser class to the mapping
below (and implement the class per `core/ingestion/base.py`).
"""
from core.ingestion.csv_parser import CSVParser
from core.ingestion.pdf_parser import PDFParser
from core.ingestion.xlsx_parser import XLSXParser

_PARSERS_BY_EXTENSION = {
    "csv": CSVParser,
    "xlsx": XLSXParser,
    "xls": XLSXParser,
    "pdf": PDFParser,
}


def get_parser(filename: str):
    """Return a Parser instance for `filename`'s extension, or None if unsupported."""
    extension = filename.rsplit(".", 1)[-1].lower()
    parser_class = _PARSERS_BY_EXTENSION.get(extension)
    return parser_class() if parser_class else None
