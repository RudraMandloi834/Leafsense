# LeafSense Agricultural Knowledge Base Documents
===================================================

This directory holds official, trusted agricultural reference documents used by the LeafSense Retrieval-Augmented Generation (RAG) system.

## Supported Document Formats
- **PDF (`.pdf`)**: Agricultural crop advisories, ICAR technical bulletins, extension pamphlets, university research reports.
- **Text (`.txt`)**: Plain text guidelines, advisory notes.
- **Markdown (`.md`)**: Structured agronomy reference sheets.

## Recommended Trusted Sources
To avoid misleading agricultural advice, ensure documents added here are sourced from verified institutions:
1. **ICAR (Indian Council of Agricultural Research)** — [icar.org.in](https://icar.org.in)
2. **State Agricultural Universities (SAUs)** (e.g., TNAU Agritech Portal, PAU, GBPUAT)
3. **Ministry of Agriculture & Farmers Welfare, Government of India** — [agricoop.gov.in](https://agricoop.gov.in)
4. **Directorate of Plant Protection, Quarantine & Storage (DPPQ&S)** — [ppqs.gov.in](https://ppqs.gov.in)

## How It Works
1. Place your `.pdf`, `.txt`, or `.md` files in this directory:
   `backend/knowledge_base/documents/`
2. Run document ingestion via script:
   ```bash
   python backend/ingest_documents.py
   ```
   Or via the backend API endpoint:
   ```bash
   curl -X POST http://localhost:8000/api/rag/ingest
   ```
3. The system extracts text page-by-page, generates semantic chunks, computes embeddings using **IBM Granite Multilingual Embedding** (`ibm/granite-embedding-278m-multilingual`), and stores vectors in local persistent ChromaDB.
4. **Duplicate Prevention**: The system calculates SHA-256 hashes of each document and will not re-embed unchanged documents.
