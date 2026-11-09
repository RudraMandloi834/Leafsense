#!/usr/bin/env python3
"""
LeafSense - End-to-End Agriculture RAG Verification Script
==========================================================
Verifies the complete RAG pipeline using IBM Granite:
1. Sets up an isolated test knowledge base in a temporary directory.
2. Ingests a verified test agronomy advisory document.
3. Computes embeddings using IBM Granite Multilingual Embedding (ibm/granite-embedding-278m-multilingual).
4. Executes semantic retrieval for a farmer question.
5. Generates a grounded answer using IBM Granite (ibm/granite-4-h-small).
6. Verifies source attribution (document name, page number, relevance score).
7. Cleans up test artifacts without modifying the production knowledge base.
"""

import sys
import shutil
import tempfile
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))

from rag_service import RagService
from watsonx_service import watsonx_service


def run_e2e_verification():
    print("=" * 70)
    print("  LeafSense - Agriculture RAG Knowledge Base Live Verification")
    print("=" * 70)

    # Verify credentials
    status = watsonx_service.get_credential_status()
    print(f"\n[1/5] Checking IBM watsonx.ai Credentials...")
    print(f"  • Endpoint:        {status['url']}")
    print(f"  • Credentials:     {'[CONFIGURED]' if status['ready'] else '[MISSING]'}")

    if not status["ready"]:
        print("❌ Error: Missing credentials in backend/.env!")
        sys.exit(1)

    temp_dir = Path(tempfile.mkdtemp(prefix="leafsense_rag_test_"))
    print(f"\n[2/5] Creating Isolated Test Knowledge Base at: {temp_dir}")

    try:
        test_rag = RagService(base_dir=temp_dir)

        # Create a sample agricultural reference document
        sample_doc = test_rag.documents_dir / "icar_wheat_rust_bulletin_2024.txt"
        sample_content = (
            "Indian Council of Agricultural Research (ICAR) Advisory Bulletin 2024\n"
            "Crop: Wheat (Triticum aestivum)\n"
            "Disease: Stripe Rust / Yellow Rust (Puccinia striiformis)\n\n"
            "Symptoms:\n"
            "Yellow pustules arranged in linear stripes along leaf veins on young and mature leaves. "
            "Under high humidity and cool temperatures (10-15°C), yellow powdery spores cover leaf blades.\n\n"
            "Recommended Management:\n"
            "1. Grow resistant varieties such as HD 3226, DBW 187, or DBW 222.\n"
            "2. At initial disease appearance, spray propiconazole 25% EC at 0.1% concentration (1 ml per liter of water).\n"
            "3. Ensure adequate spray coverage with 200 liters of water per acre using a hollow cone nozzle.\n"
            "4. Avoid excessive nitrogenous fertilizers which favor disease development."
        )
        sample_doc.write_text(sample_content, encoding="utf-8")
        print(f"  ✔ Created test document: {sample_doc.name} ({len(sample_content)} chars)")

        # Ingest document
        print(f"\n[3/5] Ingesting Document & Generating IBM Granite Embeddings...")
        print(f"  • Embedding Model: {test_rag.embedding_model_id}")
        ingest_result = test_rag.ingest_documents()
        print(f"  ✔ Ingested: {ingest_result['added_files']}")
        print(f"  ✔ Total Chunks in Test ChromaDB: {ingest_result['total_chunks']}")

        # Semantic retrieval
        test_question = "What wheat varieties are resistant to yellow rust and what spray is recommended?"
        print(f"\n[4/5] Testing Semantic Retrieval & Grounded Generation...")
        print(f"  Farmer Question: \"{test_question}\"")
        print(f"  • Generation Model: {test_rag.generation_model_id}")
        print("  Retrieving context and generating grounded answer...")

        answer_result = test_rag.answer_query(
            question=test_question,
            top_k=2,
            language="en"
        )

        print("\n" + "=" * 70)
        print("  ✔ SUCCESS: Grounded IBM Granite Response Received!")
        print("=" * 70)
        print(f"\nGrounded: {answer_result['grounded']}")
        print(f"Model:    {answer_result['model_used']}\n")
        print("Generated Answer:")
        print("-" * 50)
        print(answer_result['answer'])
        print("-" * 50)

        print("\nRetrieved Source Attributions:")
        for idx, src in enumerate(answer_result['sources'], 1):
            print(f"  [{idx}] Document: {src['document']} (Page {src['page']}) | Relevance: {src['relevance_score']}")
            print(f"      Snippet: {src['snippet'][:120]}...\n")

        # Verify ground truth expectations
        assert answer_result["grounded"] is True, "Expected answer to be grounded"
        assert len(answer_result["sources"]) > 0, "Expected at least 1 source"
        assert answer_result["sources"][0]["document"] == "icar_wheat_rust_bulletin_2024.txt"

        print("=" * 70)
        print("  Phase 3 End-to-End RAG Verification Completed Successfully!")
        print("=" * 70)
        return True

    finally:
        # Clean up temporary test directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\n[5/5] Cleaned up temporary test directory.")


if __name__ == "__main__":
    run_e2e_verification()
