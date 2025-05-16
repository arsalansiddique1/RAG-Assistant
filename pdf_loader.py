
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk_pdf(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> list[str]:
    """
    Load a PDF and split it into fixed-size character chunks.

    Args:
        file_path: Path to the PDF file.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters of overlap between chunks.

    Returns:
        A list of text chunks.
    """
    # 1. Extract full text
    reader = PdfReader(file_path)
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

    # 2. Split into fixed-size chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(full_text)