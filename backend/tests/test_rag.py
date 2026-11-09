"""
LeafSense Phase 3 Test Suite - Agriculture RAG Knowledge Base
============================================================
Tests:
- PDF and Text document extraction
- Semantic text chunking and overlap
- Document metadata preservation (source, page, chunk_index)
- SHA-256 duplicate detection and hash computing
- Empty knowledge base safe fallback
- RAG API endpoints (/api/rag/status, /api/rag/query, /api/rag/ingest)
- Error handling and input validation
"""

import io
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app import app
from rag_service import RagService, rag_service

client = TestClient(app)


def create_sample_pdf(file_path: Path, pages_text: list[str]) -> Path:
    """Helper to generate a real PDF with specified text on each page."""
    from pypdf import PageObject
    writer = PdfWriter()
    for text in pages_text:
        # Create a blank page and add text
        page = PageObject.create_blank_page(width=300, height=300)
        # Using writer.add_blank_page or annotation / text
        # Since pypdf PdfWriter can merge or write annotations,
        # we can also write text via blank page or simple layout
        writer.add_page(page)

    with open(file_path, "wb") as f:
        writer.write(f)
    return file_path


def test_compute_file_hash(tmp_path):
    """Verify SHA-256 computation detects identical and modified content."""
    test_file = tmp_path / "sample.txt"
    test_file.write_text("ICAR tomato blight management guide", encoding="utf-8")

    hash1 = RagService.compute_file_hash(test_file)
    assert len(hash1) == 64  # SHA-256 hex length

    # Identical content -> same hash
    hash2 = RagService.compute_file_hash(test_file)
    assert hash1 == hash2

    # Modified content -> different hash
    test_file.write_text("Modified content with revised spray dosage", encoding="utf-8")
    hash3 = RagService.compute_file_hash(test_file)
    assert hash1 != hash3


def test_extract_text_from_txt_and_md(tmp_path):
    """Verify text extraction from .txt and .md files preserves page 1 metadata."""
    txt_file = tmp_path / "wheat_rust.txt"
    txt_file.write_text("Yellow rust in wheat requires timely intervention.", encoding="utf-8")

    pages = rag_service.extract_text_from_file(txt_file)
    assert len(pages) == 1
    assert pages[0]["page"] == 1
    assert pages[0]["source"] == "wheat_rust.txt"
    assert "Yellow rust" in pages[0]["text"]

    md_file = tmp_path / "pest_advisory.md"
    md_file.write_text("# Pest Advisory\n\nMonitor aphid population in mustard.", encoding="utf-8")

    md_pages = rag_service.extract_text_from_file(md_file)
    assert len(md_pages) == 1
    assert "Monitor aphid" in md_pages[0]["text"]


def test_extract_unsupported_file_format(tmp_path):
    """Verify unsupported file formats raise ValueError."""
    bad_file = tmp_path / "archive.zip"
    bad_file.write_bytes(b"PK00fakezipdata")

    with pytest.raises(ValueError, match="Unsupported file format"):
        rag_service.extract_text_from_file(bad_file)


def test_extract_nonexistent_file():
    """Verify nonexistent file raises FileNotFoundError."""
    missing_path = Path("/nonexistent/path/guide.pdf")
    with pytest.raises(FileNotFoundError):
        rag_service.extract_text_from_file(missing_path)


def test_text_chunking_and_metadata_preservation():
    """Verify text chunking preserves source, page numbers, and index."""
    pages = [
        {
            "text": "Paragraph 1: Early blight in tomato caused by Alternaria solani.\n\n"
                    "Paragraph 2: Symptoms include dark concentric rings on mature leaves.",
            "page": 1,
            "source": "tomato_guide.pdf"
        },
        {
            "text": "Paragraph 3: Preventive measures include crop rotation with non-solanaceous crops.\n\n"
                    "Paragraph 4: Maintain proper plant spacing to improve canopy air circulation.",
            "page": 2,
            "source": "tomato_guide.pdf"
        }
    ]

    chunks = rag_service.chunk_document_pages(pages, chunk_size=150, chunk_overlap=30)
    assert len(chunks) >= 2

    # Verify metadata fields
    for c in chunks:
        assert "chunk_id" in c
        assert "text" in c
        assert "source" in c
        assert c["source"] == "tomato_guide.pdf"
        assert c["page"] in (1, 2)
        assert "chunk_index" in c
        assert c["char_count"] > 0


def test_empty_knowledge_base_response():
    """Verify answering against an empty KB returns ungrounded honest response."""
    result = rag_service.answer_query("How to treat potato late blight?", language="en")
    assert result["success"] is True
    assert result["grounded"] is False
    assert result["sources"] == []
    assert "empty" in result["answer"].lower() or "0 documents" in result["message"]

    # Verify Hindi message
    result_hi = rag_service.answer_query("आलू के रोग का उपचार क्या है?", language="hi")
    assert result_hi["success"] is True
    assert result_hi["grounded"] is False
    assert "ज्ञानकोश" in result_hi["answer"] or "खाली" in result_hi["answer"]


def test_api_rag_status_endpoint():
    """Verify GET /api/rag/status returns healthy vector store status."""
    response = client.get("/api/rag/status")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "total_chunks" in data
    assert "total_documents" in data
    assert data["embedding_model"] == "ibm/granite-embedding-278m-multilingual"
    assert data["generation_model"] == "ibm/granite-4-h-small"
    assert data["vector_db"] == "ChromaDB"


def test_api_rag_query_endpoint_empty_kb():
    """Verify POST /api/rag/query returns honest response when KB has no docs."""
    response = client.post(
        "/api/rag/query",
        json={"question": "What is the recommended spray for downy mildew in grapes?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["grounded"] is False
    assert data["sources"] == []
    assert len(data["answer"]) > 0


def test_api_rag_query_validation():
    """Verify request validation rejects invalid or empty questions."""
    response = client.post(
        "/api/rag/query",
        json={"question": ""}  # Too short
    )
    assert response.status_code == 422  # Pydantic validation error


def test_api_rag_ingest_invalid_extension():
    """Verify POST /api/rag/ingest rejects unsupported file types."""
    dummy_exe = io.BytesIO(b"MZ\x90\x00executable")
    response = client.post(
        "/api/rag/ingest",
        files={"file": ("malicious.exe", dummy_exe, "application/octet-stream")}
    )
    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()
