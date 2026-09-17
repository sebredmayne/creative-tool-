"""Parses XLSX files (all sheets) into DataFrames and text chunks."""
import pandas as pd

from core.ingestion.base import Parser
from core.ingestion.chunking import dataframe_to_chunks
from core.models import ParsedDocument


class XLSXParser(Parser):
    def parse(self, file, filename: str) -> ParsedDocument:
        sheets: dict[str, pd.DataFrame] = pd.read_excel(file, sheet_name=None, engine="openpyxl")

        text_chunks = []
        for sheet_name, df in sheets.items():
            text_chunks.extend(dataframe_to_chunks(df, table_name=f"{filename}:{sheet_name}"))

        return ParsedDocument(
            filename=filename,
            file_type="xlsx",
            text_chunks=text_chunks,
            dataframes=sheets,
        )
