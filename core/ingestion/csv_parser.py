"""Parses CSV files into a DataFrame (structured analysis) and text chunks (semantic search)."""
import pandas as pd

from core.ingestion.base import Parser
from core.ingestion.chunking import dataframe_to_chunks
from core.models import ParsedDocument


class CSVParser(Parser):
    def parse(self, file, filename: str) -> ParsedDocument:
        df = pd.read_csv(file)
        return ParsedDocument(
            filename=filename,
            file_type="csv",
            text_chunks=dataframe_to_chunks(df, table_name=filename),
            dataframes={filename: df},
        )
