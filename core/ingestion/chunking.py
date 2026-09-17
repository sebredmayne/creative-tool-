"""Helpers for turning raw content into small text chunks suitable for embedding."""
import pandas as pd


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """Split text into overlapping chunks of roughly `chunk_size` words."""
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def dataframe_to_chunks(df: pd.DataFrame, table_name: str, rows_per_chunk: int = 10) -> list[str]:
    """Turn a DataFrame into text chunks for semantic search.

    One summary chunk (shape + columns) plus one chunk per small group of
    rows, written out as `column: value` pairs. Numeric/structured analysis
    of the same data happens separately via StructuredStore - this is only
    so free-text columns (e.g. reviews) are semantically searchable.
    """
    summary = f"Table '{table_name}' has {len(df)} rows and columns: {', '.join(df.columns.astype(str))}."
    chunks = [summary]

    for start in range(0, len(df), rows_per_chunk):
        rows = df.iloc[start : start + rows_per_chunk]
        row_text = "\n".join(
            "; ".join(f"{col}: {row[col]}" for col in df.columns) for _, row in rows.iterrows()
        )
        chunks.append(f"Rows {start}-{start + len(rows) - 1} of '{table_name}':\n{row_text}")

    return chunks
