"""Tests for the pure-logic pieces of ui/components.py - just the video-format heuristic that
gates the "Export edit brief" button, which needs no Streamlit runtime to test."""
import pytest

from ui.components import _is_video_format


@pytest.mark.parametrize(
    "format_text", ["Instagram Reel", "UGC video", "TikTok", "9:16 short-form video", "Reels", "UGC"]
)
def test_video_formats_are_detected(format_text):
    assert _is_video_format(format_text) is True


@pytest.mark.parametrize("format_text", ["Carousel", "Static ad", "Email", "Product page copy"])
def test_non_video_formats_are_not_detected(format_text):
    assert _is_video_format(format_text) is False
