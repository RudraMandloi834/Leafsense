# Agriculture RAG Knowledge Base Setup

LeafSense incorporates a localized **Retrieval-Augmented Generation (RAG)** knowledge base that indexes authentic agricultural research, extension bulletins, and government plant pathology advisories. This ensures that the Smart Farming Assistant generates responses grounded in verified literature rather than relying solely on parametric foundation model recall.

---

## 1. Directory Structure

The knowledge base components reside inside `backend/knowledge_base/`:

```
backend/
├── knowledge_base/
│   ├── documents/             # Place agricultural PDFs, TXT, and Markdown files here
│   │   ├── icar_tomato_management.pdf
│   │   ├── kvk_potato_blight_guide.pdf
│   │   └── organic_pest_control.md
│   └── chroma_db/             # Persistent vector store (managed by ChromaDB, gitignored)
├── ingest_documents.py        # CLI ingestion script
└── rag_service.py             # ChromaDB vector retrieval and authority classification service
```

---

## 2. Supported Formats & Extraction

The ingestion engine (`backend/rag_service.py`) supports three document formats:

| Format | Extension | Extractor | Extraction Details |
|:---|:---:|:---|:---|
| **Portable Document Format** | `.pdf` | `pypdf.PdfReader` | Extracts per-page text, tracks exact page numbers in chunk metadata |
| **Plain Text** | `.txt` | Native Python UTF-8 | Chunks continuous text, assigns page number `1` |
| **Markdown** | `.md` | Native Python UTF-8 | Strips formatting headers, preserves structural paragraphs |

---

## 3. Ingestion Workflow

### Step 1: Place Documents
Copy your authentic agricultural PDF, TXT, or MD documents into `backend/knowledge_base/documents/`:
```bash
cp /path/to/icar_crop_guidelines.pdf backend/knowledge_base/documents/
```

### Step 2: Run CLI Ingestion
Activate the backend virtual environment and execute the ingestion script:
```bash
cd backend
source .venv/bin/activate
python ingest_documents.py
```

**Expected Console Output:**
```
============================================================
LeafSense - Agriculture Knowledge Base Ingestion Pipeline
============================================================
📁 Documents Directory : .../backend/knowledge_base/documents
💾 ChromaDB Directory : .../backend/knowledge_base/chroma_db
🤖 Embedding Model    : ibm/granite-embedding-278m-multilingual
============================================================

📄 Processing: icar_tomato_management.pdf (pdf)
   Extracted 12 pages, created 36 chunks.
   Ingesting 36 chunks into ChromaDB...
   ✅ Successfully ingested icar_tomato_management.pdf!

============================================================
🎉 Ingestion Complete!
📊 Total Documents Indexed: 1
📦 Total Chunks in Collection: 36
============================================================
```

Alternatively, documents can be uploaded dynamically at runtime via the REST API:
```bash
curl -X POST http://localhost:8000/api/rag/ingest \
  -F "file=@icar_tomato_management.pdf"
```

---

## 4. Chunking Strategy & Metadata Schema

Each document is divided into manageable segments:
- **Chunk Size:** $\sim 500$ characters / words with contextual sentence boundaries.
- **Chunk Overlap:** $\sim 100$ characters to maintain transitional coherence across boundaries.
- **Metadata Fields:**
  - `document`: Filename of the source document (e.g., `icar_tomato_management.pdf`).
  - `page`: Page index (1-indexed for PDFs).
  - `chunk_index`: Sequential integer index within the document.
  - `content_hash`: SHA-256 hash preventing duplicate chunk indexing.
  - `authority_type`: Classified authority tier.

---

## 5. Source Authority Tiers

To safeguard farmers against unreliable advice, LeafSense automatically categorizes retrieved snippets into three authority tiers:

| Authority Tier | Qualifying Identifiers | Visual UI Badge |
|:---|:---|:---:|
| **`official_guidance`** | ICAR, KVK, Krishi Vigyan Kendra, State Agricultural Universities (TNAU, PAU, GBPUAT), CIBRC, Department of Agriculture | 🏛️ Official / KVK |
| **`verified_research`** | Peer-reviewed agricultural journals, agronomy trials, university bulletins | 🔬 Verified Research |
| **`agricultural_reference`** | General extension bulletins, farming manuals, reference handbooks | 📚 Knowledge Base |

---

## 6. Recommended Document Sources

To maintain high advisory quality, maintainers should ingest authentic, copyright-cleared public agricultural materials:
- **ICAR (Indian Council of Agricultural Research):** Package of Practices for Horticultural and Field Crops.
- **Krishi Vigyan Kendras (KVK):** District-level agro-advisory bulletins and regional disease management guides.
- **State Agricultural Universities:** Extension manuals from PAU Ludhiana, TNAU Coimbatore, UAS Bengaluru, etc.
- **CIBRC (Central Insecticide Board & Registration Committee):** Approved pesticide labels and chemical formulations.

> **Important Note:** If the knowledge base is empty (`total_documents: 0`), LeafSense falls back to general agronomic educational guidance with an explicit **ungrounded** transparency warning, and will not fabricate reference citations.
