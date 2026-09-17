"""Keeps uploaded tabular data available for pandas-based structured analysis.

Semantic search (vector_store) is good at finding relevant text, but numeric
questions ("what's our average order value?") need real computation, not
similarity search. This is a deliberately simple in-memory store for V0 -
no persistence, no multi-user handling.
"""
import pandas as pd


class StructuredStore:
    def __init__(self):
        self._tables: dict[str, pd.DataFrame] = {}

    def add(self, name: str, df: pd.DataFrame) -> None:
        self._tables[name] = df

    def has_data(self) -> bool:
        return len(self._tables) > 0

    def clear(self) -> None:
        self._tables.clear()

    def summarize_all(self, max_sample_rows: int = 5) -> str:
        """Plain-text summary (shape, columns, dtypes, sample rows, basic stats) for every table.

        Intended to be handed to an LLM as context - not a general-purpose analytics engine.
        """
        if not self._tables:
            return ""

        sections = []
        for name, df in self._tables.items():
            lines = [
                f"Table: {name}",
                f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
                f"Columns: {', '.join(f'{c} ({df[c].dtype})' for c in df.columns)}",
                f"Sample rows:\n{df.head(max_sample_rows).to_string(index=False)}",
            ]

            numeric_df = df.select_dtypes(include="number")
            if not numeric_df.empty:
                lines.append(f"Numeric summary:\n{numeric_df.describe().to_string()}")

            sections.append("\n".join(lines))

        return "\n\n---\n\n".join(sections)
