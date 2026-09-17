"""Tests for the ingestion layer, using small synthetic data (no real company data)."""
import io
from unittest.mock import MagicMock, patch

import pandas as pd

from core.ingestion import get_parser
from core.ingestion.csv_parser import CSVParser
from core.ingestion.pdf_parser import PDFParser
from core.ingestion.xlsx_parser import XLSXParser


def _synthetic_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product": ["Vitamin C Face Wash", "Niacinamide Serum", "Sunscreen SPF50"],
            "units_sold": [120, 340, 210],
            "review_snippet": [
                "Left my skin feeling fresh, will repurchase.",
                "A little pricey but noticeable results after two weeks.",
                "Doesn't leave a white cast, love it for daily use.",
            ],
        }
    )


def test_csv_parser_produces_chunks_and_dataframe():
    df = _synthetic_df()
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    parsed = CSVParser().parse(buffer, "sales.csv")

    assert parsed.file_type == "csv"
    assert parsed.dataframes["sales.csv"].shape == df.shape
    assert len(parsed.text_chunks) >= 1
    assert any("product" in chunk for chunk in parsed.text_chunks)


def test_xlsx_parser_handles_multiple_sheets():
    df1 = _synthetic_df()
    df2 = pd.DataFrame({"month": ["Jan", "Feb"], "retention_rate": [0.62, 0.58]})

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df1.to_excel(writer, sheet_name="sales", index=False)
        df2.to_excel(writer, sheet_name="retention", index=False)
    buffer.seek(0)

    parsed = XLSXParser().parse(buffer, "company_data.xlsx")

    assert set(parsed.dataframes.keys()) == {"sales", "retention"}
    assert any("retention" in chunk.lower() for chunk in parsed.text_chunks)


def test_pdf_parser_extracts_and_chunks_text():
    fake_page = MagicMock()
    fake_page.extract_text.return_value = "word " * 500  # long enough to span multiple chunks

    with patch("core.ingestion.pdf_parser.PdfReader") as mock_reader:
        mock_reader.return_value.pages = [fake_page]
        parsed = PDFParser().parse(io.BytesIO(b"%PDF-1.4 fake"), "brand_deck.pdf")

    assert parsed.file_type == "pdf"
    assert parsed.dataframes is None
    assert len(parsed.text_chunks) > 1


def test_get_parser_dispatches_by_extension():
    assert isinstance(get_parser("data.csv"), CSVParser)
    assert isinstance(get_parser("data.xlsx"), XLSXParser)
    assert isinstance(get_parser("deck.pdf"), PDFParser)
    assert get_parser("notes.docx") is None
