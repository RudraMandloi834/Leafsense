#!/usr/bin/env python3
"""
LeafSense - Agriculture Knowledge Base Document Ingestion CLI
=============================================================
Scans `backend/knowledge_base/documents/` for agricultural documents (.pdf, .txt, .md),
chunks text, computes embeddings using IBM Granite Multilingual Embedding,
and updates the local ChromaDB vector database.

Uses SHA-256 duplicate detection so unchanged documents are skipped.
"""

import sys
import argparse
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))

from rag_service import rag_service
from watsonx_service import watsonx_service, WatsonxServiceError


def run_ingestion(force: bool = False):
    print("=" * 70)
    print("  LeafSense - Agriculture Knowledge Base Document Ingestion")
    print("=" * 70)

    # 1. Check configuration
    status = watsonx_service.get_credential_status()
    print(f"\n[1/3] Checking IBM watsonx Configuration...")
    print(f"  • Endpoint:        {status['url']}")
    print(f"  • Embedding Model: {rag_service.embedding_model_id}")
    print(f"  • Generation Model:{rag_service.generation_model_id}")
    print(f"  • Credentials:     {'[CONFIGURED]' if status['ready'] else '[MISSING]'}")

    if not status["ready"]:
        print("\n❌ Error: Missing watsonx credentials in backend/.env!")
        print("Please configure WATSONX_API_KEY and WATSONX_PROJECT_ID before ingesting documents.")
        sys.exit(1)

    # 2. Inspect Documents Directory
    docs_dir = rag_service.documents_dir
    print(f"\n[2/3] Inspecting Document Directory: {docs_dir}")
    doc_files = [
        f for f in docs_dir.iterdir()
        if f.is_file() and f.suffix.lower() in (".pdf", ".txt", ".md") and not f.name.startswith(".") and f.name != "README.md"
    ]

    print(f"  Found {len(doc_files)} document(s) in directory:")
    if not doc_files:
        print("  ℹ No agricultural documents found in 'backend/knowledge_base/documents/'.")
        print("\nTo add reference documents:")
        print("  1. Copy official PDFs or text guides (e.g., from ICAR or state universities) into:")
        print(f"     {docs_dir}")
        print("  2. Re-run: python backend/ingest_documents.py\n")

    for f in doc_files:
        size_kb = round(f.stat().st_size / 1024, 1)
        print(f"   • {f.name} ({size_kb} KB)")

    # 3. Execute Ingestion Pipeline
    print(f"\n[3/3] Running Ingestion Pipeline (Force reindex: {force})...")
    try:
        result = rag_service.ingest_documents(force_reindex=force)
        print("\n" + "=" * 70)
        print("  ✔ INGESTION COMPLETE")
        print("=" * 70)
        print(f"  • Added:        {len(result['added_files'])} ({', '.join(result['added_files']) if result['added_files'] else 'None'})")
        print(f"  • Updated:      {len(result['updated_files'])} ({', '.join(result['updated_files']) if result['updated_files'] else 'None'})")
        print(f"  • Skipped (Unchanged): {len(result['skipped_files'])} ({', '.join(result['skipped_files']) if result['skipped_files'] else 'None'})")
        print(f"  • Deleted:      {len(result['deleted_files'])} ({', '.join(result['deleted_files']) if result['deleted_files'] else 'None'})")
        print(f"  • Total Chunks in Index: {result['total_chunks']}")
        print(f"  • Storage Path: {rag_service.chroma_db_dir}")
        print("=" * 70)
        return True

    except WatsonxServiceError as e:
        print(f"\n❌ Ingestion Failed [{e.category}]:")
        print(f"   {e}")
        if e.suggestions:
            print("\nRecommended Actions:")
            for s in e.suggestions:
                print(f"   - {s}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Ingestion Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest agricultural documents into LeafSense RAG knowledge base.")
    parser.add_argument("--force", action="store_true", help="Force re-indexing of all documents even if unchanged.")
    args = parser.parse_args()

    run_ingestion(force=args.force)
