import os
from pdf_loader import load_and_chunk_pdf
import pytest
# Path to a small sample PDF in tests/
SAMPLE_PDF = os.path.join(os.path.dirname(__file__), "Warehousing Agreement 2(66756714) 1 (1).pdf")

def test_load_and_chunk_pdf_returns_list_of_strings():
    """
    Basic sanity: the loader should return a non-empty list of str.
    """
    chunks = load_and_chunk_pdf(SAMPLE_PDF)
    assert isinstance(chunks, list), "Expected a list of chunks"
    assert chunks, "Expected at least one chunk"
    assert all(isinstance(c, str) for c in chunks), "All chunks should be strings"

def test_chunk_size_respected():
    """
    When specifying small chunk_size and chunk_overlap, no chunk should exceed 
    chunk_size + chunk_overlap characters.
    """
    chunk_size = 50
    chunk_overlap = 10
    chunks = load_and_chunk_pdf(SAMPLE_PDF, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    max_len = max(len(c) for c in chunks)
    assert max_len <= chunk_size + chunk_overlap, (
        f"Chunk too long ({max_len} > {chunk_size + chunk_overlap})"
    )