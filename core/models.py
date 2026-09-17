"""Shared data structures used across the ingestion, retrieval, and reasoning layers."""
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class ParsedDocument:
    """Output of a Parser: text for semantic search, and (if tabular) DataFrames for structured analysis."""

    filename: str
    file_type: str
    text_chunks: list[str]
    dataframes: Optional[dict[str, pd.DataFrame]] = None


@dataclass
class RetrievedChunk:
    """A single chunk returned by a vector store query."""

    text: str
    source: str
    distance: float


@dataclass
class QueryContext:
    """Everything needed to answer one user request, for either mode."""

    mode: str  # "explore" or "connect"
    query: str
    brand: str = ""
    product: str = ""
    customer: str = ""
    category: str = ""
    objective: str = ""


@dataclass
class CreativeIdea:
    """A single structured piece of V0 output."""

    concept: str
    rationale: str
    recommended_format: str
    source_context: str
    script: Optional[str] = None
