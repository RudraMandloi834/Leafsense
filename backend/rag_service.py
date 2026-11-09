"""
LeafSense - Agriculture RAG (Retrieval-Augmented Generation) Knowledge Base
===========================================================================
Grounded agricultural question answering powered by:
- IBM Granite Foundation Model (ibm/granite-4-h-small)
- IBM Granite Multilingual Embedding Model (ibm/granite-embedding-278m-multilingual)
- Persistent ChromaDB vector database

Guiding Principles:
1. Strict factual grounding: Answers are drawn SOLELY from retrieved verified agricultural documents.
2. No fabricated facts or citations: If information is not in the knowledge base, state it clearly.
3. Source attribution: Every response includes document names, page numbers, and relevance scores.
4. Smart ingestion: SHA-256 duplicate detection avoids re-embedding unchanged files.
5. Multilingual capability: Supports both English and Hindi farming queries.
"""

import os
import re
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from watsonx_service import watsonx_service, WatsonxServiceError

logger = logging.getLogger("leafsense.rag")


class RagService:
    """
    Agricultural Knowledge Base service providing document processing,
    vector storage, semantic retrieval, and grounded Granite Q&A.
    """

    DEFAULT_EMBEDDING_MODEL = "ibm/granite-embedding-278m-multilingual"
    DEFAULT_GENERATION_MODEL = "ibm/granite-4-h-small"
    COLLECTION_NAME = "leafsense_agriculture_kb"

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        embedding_model_id: Optional[str] = None,
        generation_model_id: Optional[str] = None
    ):
        """
        Initialize the RAG service directories and model configurations.
        """
        if base_dir is None:
            # Default to backend/knowledge_base
            self.base_dir = Path(__file__).resolve().parent / "knowledge_base"
        else:
            self.base_dir = Path(base_dir).resolve()

        self.documents_dir = self.base_dir / "documents"
        self.chroma_db_dir = self.base_dir / "chroma_db"
        self.manifest_path = self.base_dir / "kb_manifest.json"

        # Ensure directories exist
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_db_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_model_id = (
            embedding_model_id
            or os.getenv("WATSONX_EMBEDDING_MODEL_ID")
            or self.DEFAULT_EMBEDDING_MODEL
        ).strip()

        self.generation_model_id = (
            generation_model_id
            or os.getenv("WATSONX_MODEL_ID")
            or self.DEFAULT_GENERATION_MODEL
        ).strip()

        self._chroma_client = None
        self._collection = None
        self._embedding_client = None

    # -------------------------------------------------------------------------
    # Hashing & Duplicate Management
    # -------------------------------------------------------------------------

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """Compute SHA-256 hash of a file to detect content modifications."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _load_manifest(self) -> Dict[str, Any]:
        """Load the file ingestion manifest."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load manifest, creating fresh: {e}")
                return {}
        return {}

    def _save_manifest(self, manifest: Dict[str, Any]) -> None:
        """Persist the ingestion manifest."""
        try:
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save KB manifest: {e}")

    # -------------------------------------------------------------------------
    # Document Text Extraction
    # -------------------------------------------------------------------------

    def extract_text_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Extracts text from a supported document (.pdf, .txt, .md).
        Returns list of pages: [{'text': str, 'page': int, 'source': str}]
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = file_path.suffix.lower()
        pages: List[Dict[str, Any]] = []

        if ext == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError:
                raise RuntimeError("pypdf is required for PDF extraction. Run 'pip install pypdf'.")

            try:
                reader = PdfReader(str(file_path))
                total_pages = len(reader.pages)
                for page_idx, page in enumerate(reader.pages):
                    raw_text = page.extract_text() or ""
                    cleaned = self._clean_text(raw_text)
                    if cleaned:
                        pages.append({
                            "text": cleaned,
                            "page": page_idx + 1,
                            "source": file_path.name,
                            "total_pages": total_pages
                        })
            except Exception as e:
                logger.error(f"Error reading PDF {file_path.name}: {e}")
                raise ValueError(f"Unable to parse PDF '{file_path.name}': {e}")

        elif ext in (".txt", ".md"):
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    raw_text = f.read()
                cleaned = self._clean_text(raw_text)
                if cleaned:
                    pages.append({
                        "text": cleaned,
                        "page": 1,
                        "source": file_path.name,
                        "total_pages": 1
                    })
            except Exception as e:
                logger.error(f"Error reading text document {file_path.name}: {e}")
                raise ValueError(f"Unable to read file '{file_path.name}': {e}")
        else:
            raise ValueError(f"Unsupported file format '{ext}'. Supported formats: .pdf, .txt, .md")

        return pages

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize whitespace, remove excess newlines, and clean raw text."""
        if not text:
            return ""
        # Replace non-breaking spaces
        text = text.replace("\xa0", " ")
        # Collapse multiple empty lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Collapse multiple horizontal whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    # -------------------------------------------------------------------------
    # Text Chunking
    # -------------------------------------------------------------------------

    def chunk_document_pages(
        self,
        pages: List[Dict[str, Any]],
        chunk_size: int = 800,
        chunk_overlap: int = 150
    ) -> List[Dict[str, Any]]:
        """
        Splits document pages into semantic chunks while preserving source, page,
        and paragraph structure.
        """
        all_chunks: List[Dict[str, Any]] = []
        global_chunk_idx = 0

        for page in pages:
            text = page["text"]
            source = page["source"]
            page_num = page["page"]

            # Paragraph-aware chunking
            paragraphs = text.split("\n\n")
            current_chunk = ""

            for p in paragraphs:
                p = p.strip()
                if not p:
                    continue

                # If adding paragraph exceeds chunk_size, emit current chunk
                if len(current_chunk) + len(p) + 2 > chunk_size and current_chunk:
                    all_chunks.append({
                        "chunk_id": f"{source}_p{page_num}_c{global_chunk_idx}",
                        "text": current_chunk.strip(),
                        "source": source,
                        "page": page_num,
                        "chunk_index": global_chunk_idx,
                        "char_count": len(current_chunk.strip())
                    })
                    global_chunk_idx += 1

                    # Keep overlap from the end of current_chunk
                    if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                        current_chunk = current_chunk[-chunk_overlap:].strip() + " " + p
                    else:
                        current_chunk = p
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + p
                    else:
                        current_chunk = p

            # Flush any remaining text for the page
            if current_chunk.strip():
                all_chunks.append({
                    "chunk_id": f"{source}_p{page_num}_c{global_chunk_idx}",
                    "text": current_chunk.strip(),
                    "source": source,
                    "page": page_num,
                    "chunk_index": global_chunk_idx,
                    "char_count": len(current_chunk.strip())
                })
                global_chunk_idx += 1

        return all_chunks

    # -------------------------------------------------------------------------
    # Embedding & Vector Database
    # -------------------------------------------------------------------------

    def _get_embedding_client(self):
        """Initializes IBM Granite Embeddings client."""
        if self._embedding_client is not None:
            return self._embedding_client

        try:
            from ibm_watsonx_ai.foundation_models.embeddings import Embeddings
            client = watsonx_service._get_api_client()
            self._embedding_client = Embeddings(
                model_id=self.embedding_model_id,
                api_client=client,
                project_id=watsonx_service.project_id
            )
            return self._embedding_client
        except Exception as e:
            logger.error(f"Failed to initialize IBM Granite Embeddings: {e}")
            raise WatsonxServiceError(
                f"Embedding client initialization failed: {e}",
                category="EMBEDDING_INIT_FAILED"
            )

    def _get_chroma_collection(self):
        """Initializes persistent ChromaDB client and retrieves/creates agricultural collection."""
        if self._collection is not None:
            return self._collection

        try:
            import chromadb
            from chromadb.config import Settings
            client = chromadb.PersistentClient(
                path=str(self.chroma_db_dir),
                settings=Settings(anonymized_telemetry=False)
            )
            # Use cosine distance for normalized embeddings
            self._collection = client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            self._chroma_client = client
            return self._collection
        except ImportError:
            raise RuntimeError("chromadb is required for vector storage. Run 'pip install chromadb'.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise RuntimeError(f"ChromaDB initialization failed: {e}")

    def embed_texts(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """
        Embeds a list of texts in batches using IBM Granite Multilingual Embedding.
        """
        if not texts:
            return []

        emb_client = self._get_embedding_client()
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_vectors = emb_client.embed_documents(texts=batch)
                all_embeddings.extend(batch_vectors)
            except Exception as e:
                logger.error(f"Error during embedding batch {i}-{i+len(batch)}: {e}")
                raise WatsonxServiceError(
                    f"Embedding generation failed: {e}",
                    category="EMBEDDING_GENERATION_FAILED"
                )

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embeds a single query string using IBM Granite Multilingual Embedding."""
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")
        emb_client = self._get_embedding_client()
        try:
            return emb_client.embed_query(text=query.strip())
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise WatsonxServiceError(
                f"Query embedding failed: {e}",
                category="EMBEDDING_QUERY_FAILED"
            )

    # -------------------------------------------------------------------------
    # Ingestion Pipeline
    # -------------------------------------------------------------------------

    def ingest_documents(self, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Scans documents directory, processes new/updated PDFs and text files,
        computes IBM Granite embeddings, and updates ChromaDB.
        Avoids re-embedding unchanged documents based on SHA-256 hash.
        """
        collection = self._get_chroma_collection()
        manifest = self._load_manifest()

        # Discover all valid documents (ignore README or dotfiles)
        supported_exts = {".pdf", ".txt", ".md"}
        existing_files = {
            f.name: f for f in self.documents_dir.iterdir()
            if f.is_file() and f.suffix.lower() in supported_exts and not f.name.startswith(".") and f.name != "README.md"
        }

        added_files: List[str] = []
        updated_files: List[str] = []
        skipped_files: List[str] = []
        deleted_files: List[str] = []

        # 1. Clean up deleted files from vector store
        manifest_files = list(manifest.keys())
        for doc_name in manifest_files:
            if doc_name not in existing_files:
                logger.info(f"Removing deleted document from vector index: {doc_name}")
                try:
                    collection.delete(where={"source": doc_name})
                except Exception as e:
                    logger.warning(f"Error deleting chunks for {doc_name}: {e}")
                del manifest[doc_name]
                deleted_files.append(doc_name)

        # 2. Ingest new or modified files
        for filename, file_path in existing_files.items():
            file_hash = self.compute_file_hash(file_path)
            prev_info = manifest.get(filename)

            if not force_reindex and prev_info and prev_info.get("sha256") == file_hash:
                skipped_files.append(filename)
                continue

            is_update = prev_info is not None
            logger.info(f"Processing {'updated' if is_update else 'new'} document: {filename}")

            # If update, remove old chunks first
            if is_update:
                try:
                    collection.delete(where={"source": filename})
                except Exception as e:
                    logger.warning(f"Error clearing previous chunks for {filename}: {e}")

            # Extract pages & chunk
            pages = self.extract_text_from_file(file_path)
            if not pages:
                logger.warning(f"No extractable text found in {filename}. Skipping vector indexing.")
                continue

            chunks = self.chunk_document_pages(pages)
            if not chunks:
                logger.warning(f"No chunks created for {filename}.")
                continue

            chunk_texts = [c["text"] for c in chunks]
            chunk_ids = [c["chunk_id"] for c in chunks]
            metadatas = [
                {
                    "source": c["source"],
                    "page": int(c["page"]),
                    "chunk_index": int(c["chunk_index"]),
                    "file_hash": file_hash,
                }
                for c in chunks
            ]

            # Compute IBM Granite embeddings
            embeddings = self.embed_texts(chunk_texts)

            # Upsert into ChromaDB
            collection.upsert(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunk_texts,
                metadatas=metadatas
            )

            # Update manifest
            manifest[filename] = {
                "sha256": file_hash,
                "chunk_count": len(chunks),
                "total_pages": len(pages),
                "last_modified": file_path.stat().st_mtime
            }

            if is_update:
                updated_files.append(filename)
            else:
                added_files.append(filename)

        # Persist updated manifest
        self._save_manifest(manifest)

        total_chunks = collection.count()

        return {
            "success": True,
            "added_files": added_files,
            "updated_files": updated_files,
            "skipped_files": skipped_files,
            "deleted_files": deleted_files,
            "total_documents": len(manifest),
            "total_chunks": total_chunks,
            "message": (
                f"Ingestion complete: {len(added_files)} added, {len(updated_files)} updated, "
                f"{len(skipped_files)} skipped unchanged, {len(deleted_files)} deleted. "
                f"Total chunks in knowledge base: {total_chunks}."
            )
        }

    # -------------------------------------------------------------------------
    # Retrieval & Semantic Search
    # -------------------------------------------------------------------------

    def _classify_authority_type(self, source_name: str, text: str) -> str:
        """Categorizes document credibility to distinguish official extension bulletins."""
        combined = f"{source_name} {text}".lower()
        official_indicators = [
            "icar", "iari", "kvk", "krishi vigyan", "tnau", "pau", "hau", "uas",
            "icrisat", "cibrc", "ministry of agriculture", "state agricultural university",
            "package of practices", "extension bulletin", "कृषि विज्ञान केंद्र", "भाकृअनुप"
        ]
        if any(ind in combined for ind in official_indicators):
            return "official_guidance"
        elif any(ind in combined for ind in ["journal", "research", "fao", "cgiar", "ciat", "cimmyt"]):
            return "verified_research"
        return "agricultural_reference"

    def retrieve_context(
        self,
        query: str,
        top_k: int = 4,
        similarity_threshold: float = 0.20,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves semantically relevant document chunks with intelligent ranking,
        source diversification, deduplication, and authority tagging.
        """
        collection = self._get_chroma_collection()
        total_chunks = collection.count()

        if total_chunks == 0:
            return []

        # Embed user query using IBM Granite Multilingual Embedding
        query_vector = self.embed_query(query)

        # Retrieve a broader candidate pool (up to 3x top_k) for deduplication & diversification
        candidate_k = min(max(top_k * 3, 6), total_chunks)
        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": candidate_k,
            "include": ["documents", "metadatas", "distances"]
        }
        if filter_metadata:
            query_kwargs["where"] = filter_metadata

        results = collection.query(**query_kwargs)

        candidates: List[Dict[str, Any]] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        query_tokens = set(re.findall(r'\b[a-zA-Z\u0900-\u097F]{4,}\b', query.lower()))

        for doc_text, meta, dist in zip(docs, metas, distances):
            base_similarity = max(0.0, 1.0 - float(dist))
            if base_similarity < similarity_threshold:
                continue

            # Token overlap boost for domain precision (crops, diseases, specific terms)
            doc_tokens = set(re.findall(r'\b[a-zA-Z\u0900-\u097F]{4,}\b', doc_text.lower()))
            overlap = len(query_tokens.intersection(doc_tokens))
            boost = min(0.08, overlap * 0.02)
            adjusted_score = min(1.0, base_similarity + boost)

            doc_name = meta.get("source", "Unknown Document")
            authority = self._classify_authority_type(doc_name, doc_text)

            candidates.append({
                "document": doc_name,
                "page": int(meta.get("page", 1)),
                "chunk_index": int(meta.get("chunk_index", 0)),
                "relevance_score": round(adjusted_score, 4),
                "base_similarity": round(base_similarity, 4),
                "authority_type": authority,
                "snippet": doc_text
            })

        # Sort candidates by adjusted relevance score
        candidates.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Deduplication & Source Diversification
        final_chunks: List[Dict[str, Any]] = []
        seen_snippets: List[set] = []
        doc_counts: Dict[str, int] = {}

        for cand in candidates:
            cand_tokens = set(re.findall(r'\b[a-zA-Z\u0900-\u097F]{4,}\b', cand["snippet"].lower()))
            
            # Check overlap against already chosen chunks (Jaccard similarity > 0.70 = duplicate)
            is_dup = False
            for prev_tokens in seen_snippets:
                intersection = len(cand_tokens.intersection(prev_tokens))
                union = len(cand_tokens.union(prev_tokens))
                if union > 0 and (intersection / union) > 0.70:
                    is_dup = True
                    break

            if is_dup:
                continue

            # Cap chunks per single document if multiple documents exist
            doc_name = cand["document"]
            if len(candidates) > top_k and doc_counts.get(doc_name, 0) >= 2:
                continue

            final_chunks.append(cand)
            seen_snippets.append(cand_tokens)
            doc_counts[doc_name] = doc_counts.get(doc_name, 0) + 1

            if len(final_chunks) >= top_k:
                break

        return final_chunks

    # -------------------------------------------------------------------------
    # Grounded Q&A with IBM Granite
    # -------------------------------------------------------------------------

    def answer_query(
        self,
        question: str,
        top_k: int = 4,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Answers an agricultural question strictly grounded in retrieved knowledge base documents.
        Never fabricates recommendations, chemicals, or citations.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        collection = self._get_chroma_collection()
        total_chunks = collection.count()
        lang = (language or "en").lower().strip()

        # Case 1: Knowledge base is empty
        if total_chunks == 0:
            msg_en = (
                "The LeafSense agricultural knowledge base is currently empty. "
                "No verified agricultural reference documents have been indexed yet. "
                "Please add trusted agricultural guides (such as ICAR crop bulletins or university package of practices) "
                "to 'backend/knowledge_base/documents/' and trigger ingestion to receive grounded recommendations."
            )
            msg_hi = (
                "LeafSense कृषि ज्ञानकोश (Knowledge Base) वर्तमान में खाली है। "
                "अभी तक कोई सत्यापित कृषि संदर्भ दस्तावेज़ अनुक्रमित नहीं किया गया है। "
                "कृपया आधिकारिक और सटीक जानकारी प्राप्त करने के लिए 'backend/knowledge_base/documents/' में "
                "ICAR या कृषि विश्वविद्यालय के दिशानिर्देश जोड़ें और इनजेशन (ingestion) चलाएं।"
            )
            return {
                "success": True,
                "question": question,
                "answer": msg_hi if lang.startswith("hi") else msg_en,
                "grounded": False,
                "sources": [],
                "model_used": self.generation_model_id,
                "embedding_model": self.embedding_model_id,
                "message": "Knowledge base contains 0 documents."
            }

        # Case 2: Retrieve relevant context
        retrieved_chunks = self.retrieve_context(
            query=question,
            top_k=top_k,
            similarity_threshold=0.20
        )

        # Case 3: No sufficiently relevant chunks found
        if not retrieved_chunks:
            msg_en = (
                "The LeafSense knowledge base does not contain sufficiently verified information "
                f"to answer the question: '{question}'. To avoid inaccurate recommendations, "
                "an answer cannot be generated without supporting documentation from trusted agricultural sources."
            )
            msg_hi = (
                f"LeafSense ज्ञानकोश में आपके प्रश्न '{question}' से संबंधित पर्याप्त सत्यापित जानकारी उपलब्ध नहीं है। "
                "गलत सलाह से बचने के लिए, बिना आधिकारिक कृषि संदर्भ के उत्तर उत्पन्न नहीं किया जा सकता।"
            )
            return {
                "success": True,
                "question": question,
                "answer": msg_hi if lang.startswith("hi") else msg_en,
                "grounded": False,
                "sources": [],
                "model_used": self.generation_model_id,
                "embedding_model": self.embedding_model_id,
                "message": "No matching document chunks exceeded the relevance threshold."
            }

        # Case 4: Format context and prompt IBM Granite
        context_blocks = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            context_blocks.append(
                f"[Source {idx}: Document: {chunk['document']}, Page {chunk['page']}]\n{chunk['snippet']}"
            )
        formatted_context = "\n\n---\n\n".join(context_blocks)

        target_lang_str = "Hindi (हिंदी)" if lang.startswith("hi") else "simple, farmer-friendly English"

        system_instruction = (
            f"You are the LeafSense Agricultural Knowledge Assistant. "
            f"Your responsibility is to assist Indian farmers with crop diseases, pests, and agronomy.\n\n"
            f"CRITICAL GROUNDING RULES:\n"
            f"1. You MUST answer the question SOLELY and STRICTLY using the provided Agricultural Reference Context.\n"
            f"2. Cite the source document name and page number for every specific recommendation or symptom.\n"
            f"3. NEVER fabricate pesticide doses, chemicals, fertilizers, or agricultural facts that are not explicitly stated in the context.\n"
            f"4. If the provided context does not have enough information to fully address the question, explicitly state what is missing.\n"
            f"5. Respond in {target_lang_str}. Keep the language practical, polite, and easy to understand for a farmer."
        )

        user_prompt = (
            f"AGRICULTURAL REFERENCE CONTEXT:\n"
            f"-----------------------------------------\n"
            f"{formatted_context}\n"
            f"-----------------------------------------\n\n"
            f"FARMER'S QUESTION:\n"
            f"{question.strip()}\n\n"
            f"Please provide a grounded answer with citations to the document and page numbers."
        )

        try:
            generated_answer = watsonx_service.generate_text(
                prompt=user_prompt,
                system_prompt=system_instruction,
                max_new_tokens=600,
                temperature=0.3,
                decoding_method="greedy",
                model_id=self.generation_model_id
            )
        except Exception as e:
            logger.error(f"Error generating answer with IBM Granite: {e}")
            raise WatsonxServiceError(
                f"Granite generation failed during RAG answer: {e}",
                category="RAG_GENERATION_FAILED"
            )

        # Prepare clean structured sources for frontend/API
        sources_summary = [
            {
                "document": c["document"],
                "page": c["page"],
                "relevance_score": c["relevance_score"],
                "snippet": c["snippet"][:250] + "..." if len(c["snippet"]) > 250 else c["snippet"]
            }
            for c in retrieved_chunks
        ]

        return {
            "success": True,
            "question": question,
            "answer": generated_answer,
            "grounded": True,
            "sources": sources_summary,
            "model_used": self.generation_model_id,
            "embedding_model": self.embedding_model_id,
            "message": f"Successfully answered using {len(sources_summary)} grounded source chunk(s)."
        }

    # -------------------------------------------------------------------------
    # Knowledge Base Status
    # -------------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Returns the current state and health of the knowledge base."""
        manifest = self._load_manifest()
        collection = self._get_chroma_collection()
        total_chunks = collection.count()

        return {
            "success": True,
            "is_configured": watsonx_service.is_configured(),
            "total_documents": len(manifest),
            "total_chunks": total_chunks,
            "documents": list(manifest.keys()),
            "embedding_model": self.embedding_model_id,
            "generation_model": self.generation_model_id,
            "vector_db": "ChromaDB",
            "storage_path": str(self.chroma_db_dir),
            "documents_path": str(self.documents_dir)
        }


# Singleton instance for application reuse
rag_service = RagService()
