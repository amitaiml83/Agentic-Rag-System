from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Dict, Any, Optional

from loguru import logger

from config import (
    CHROMA_PATH, COLLECTION_NAME,
    TOP_K, RERANK_TOP_N, HYBRID_ALPHA
)
from src.embeddings import EmbeddingManager
from src.document_processor import Chunk


class ChromaStore:
    """
    ChromaDB store - simple, local, works on all platforms.
    """

    def __init__(self, embed_mgr: EmbeddingManager):
        self.embed = embed_mgr
        self.client = None
        self.collection = None
        self.connected = False
        self._connect()

    # ── Connection ────────────────────────────────────────────────────────────

    def _connect(self):
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Ensure the data directory exists
            Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=CHROMA_PATH,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.success(f"✓ ChromaDB connected - local DB: {CHROMA_PATH}")
            self.connected = True
            
        except ImportError as e:
            logger.error(f"ChromaDB import failed: {e}")
            logger.warning("Install with: pip install chromadb")
            self.connected = False
        except Exception as e:
            logger.error(f"ChromaDB init failed: {e}")
            logger.warning("System will run in degraded mode (no vector DB)")
            self.connected = False

    # ── Ingest ────────────────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: List[Chunk]) -> int:
        if not self.connected or not chunks:
            return 0
        
        try:
            # Delete existing chunks from same doc
            doc_ids = list({c.doc_id for c in chunks})
            for did in doc_ids:
                try:
                    # Get existing IDs for this doc
                    existing = self.collection.get(
                        where={"doc_id": did},
                        include=[]
                    )
                    if existing['ids']:
                        self.collection.delete(ids=existing['ids'])
                except Exception:
                    pass
            
            # Embed texts
            texts = [c.text for c in chunks]
            embeddings = self.embed.embed_texts(texts)
            
            # Prepare data for ChromaDB
            ids = [c.chunk_id for c in chunks]
            metadatas = [
                {
                    "doc_id": c.doc_id,
                    "chunk_id": c.chunk_id,
                    "file": c.metadata.get("file", ""),
                    "page": str(c.metadata.get("page", "")),
                    "slide": str(c.metadata.get("slide", "")),
                    "sheet": c.metadata.get("sheet", ""),
                    "source_type": c.metadata.get("source_type", ""),
                    "metadata_json": json.dumps(c.metadata)[:2000],
                }
                for c in chunks
            ]
            
            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.success(f"✓ Upserted {len(chunks)} chunks")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Upsert error: {e}")
            return 0

    # ── Dense Search ──────────────────────────────────────────────────────────

    def dense_search(
        self,
        query: str,
        top_k: int = TOP_K,
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Standard similarity search."""
        if not self.connected:
            return []
        
        try:
            q_vec = self.embed.embed_query(query)
            
            # Build where filter
            where = {}
            if source_filter:
                where = {"file": {"$contains": source_filter}}
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[q_vec],
                n_results=top_k,
                where=where if where else None,
                include=["documents", "metadatas", "distances"]
            )
            
            return self._format_results(results)
            
        except Exception as e:
            logger.error(f"Dense search error: {e}")
            return []

    # ── Keyword Search ────────────────────────────────────────────────────────

    def keyword_search(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """Keyword matching via text search."""
        if not self.connected:
            return []
        
        try:
            # Get all documents
            all_docs = self.collection.get(
                include=["documents", "metadatas"]
            )
            
            if not all_docs['ids']:
                return []
            
            # Extract keywords
            terms = [t.lower().strip() for t in query.split() if 2 <= len(t) <= 8]
            if not terms:
                return []
            
            # Score by term frequency
            scored = []
            for i, doc in enumerate(all_docs['documents']):
                text_lower = doc.lower()
                score = sum(text_lower.count(t) for t in terms)
                if score > 0:
                    scored.append({
                        "chunk_id": all_docs['ids'][i],
                        "doc_id": all_docs['metadatas'][i].get('doc_id', ''),
                        "text": doc,
                        "metadata": self._parse_metadata(all_docs['metadatas'][i]),
                        "score": score / (len(text_lower.split()) + 1),
                    })
            
            scored.sort(key=lambda x: x['score'], reverse=True)
            return scored[:top_k]
            
        except Exception as e:
            logger.error(f"Keyword search error: {e}")
            return []

    # ── Hybrid Search (RRF) ───────────────────────────────────────────────────

    def hybrid_search(
        self,
        query: str,
        top_k: int = TOP_K,
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion of dense + keyword results."""
        dense_res = self.dense_search(query, top_k=top_k * 2, source_filter=source_filter)
        keyword_res = self.keyword_search(query, top_k=top_k * 2)
        
        k = 60
        scores: Dict[str, float] = {}
        docs: Dict[str, Dict] = {}
        
        for rank, hit in enumerate(dense_res):
            cid = hit["chunk_id"]
            scores[cid] = scores.get(cid, 0) + HYBRID_ALPHA * (1 / (k + rank + 1))
            docs[cid] = hit
        
        for rank, hit in enumerate(keyword_res):
            cid = hit["chunk_id"]
            scores[cid] = scores.get(cid, 0) + (1 - HYBRID_ALPHA) * (1 / (k + rank + 1))
            docs[cid] = hit
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [docs[cid] for cid, _ in ranked]

    # ── MMR (diversity) ───────────────────────────────────────────────────────

    def mmr_search(
        self,
        query: str,
        top_k: int = TOP_K,
        lambda_mult: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """Maximal Marginal Relevance – balances relevance and diversity."""
        candidates = self.dense_search(query, top_k=top_k * 3)
        if not candidates:
            return []
        
        q_vec = self.embed.embed_query(query)
        c_vecs = self.embed.embed_texts([c["text"] for c in candidates])
        
        selected_idx: List[int] = []
        remaining = list(range(len(candidates)))
        
        for _ in range(min(top_k, len(candidates))):
            best, best_score = -1, float('-inf')
            for i in remaining:
                rel = _cosine(q_vec, c_vecs[i])
                sim = max(_cosine(c_vecs[i], c_vecs[j]) for j in selected_idx) if selected_idx else 0.0
                mmr = lambda_mult * rel - (1 - lambda_mult) * sim
                if mmr > best_score:
                    best, best_score = i, mmr
            selected_idx.append(best)
            remaining.remove(best)
        
        return [candidates[i] for i in selected_idx]

    # ── Stats ─────────────────────────────────────────────────────────────────

    def count(self) -> int:
        if not self.connected:
            return 0
        try:
            return self.collection.count()
        except Exception:
            return 0

    def list_documents(self) -> List[Dict[str, str]]:
        if not self.connected:
            return []
        try:
            all_data = self.collection.get(include=["metadatas"])
            seen: Dict[str, str] = {}
            
            for meta in all_data['metadatas']:
                doc_id = meta.get('doc_id', '')
                file_name = meta.get('file', doc_id)
                if doc_id and doc_id not in seen:
                    seen[doc_id] = file_name
            
            return [{"doc_id": k, "file": v} for k, v in seen.items()]
        except Exception as e:
            logger.error(f"list_documents error: {e}")
            return []

    def delete_document(self, doc_id: str) -> bool:
        if not self.connected:
            return False
        try:
            existing = self.collection.get(
                where={"doc_id": doc_id},
                include=[]
            )
            if existing['ids']:
                self.collection.delete(ids=existing['ids'])
            return True
        except Exception as e:
            logger.error(f"delete error: {e}")
            return False

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _format_results(results: Dict) -> List[Dict[str, Any]]:
        """Format ChromaDB query results"""
        out = []
        if not results['ids'] or not results['ids'][0]:
            return out
        
        for i in range(len(results['ids'][0])):
            meta = ChromaStore._parse_metadata(results['metadatas'][0][i])
            out.append({
                "chunk_id": results['ids'][0][i],
                "doc_id": results['metadatas'][0][i].get('doc_id', ''),
                "text": results['documents'][0][i],
                "metadata": meta,
                "score": 1.0 - results['distances'][0][i],  # Convert distance to similarity
            })
        return out

    @staticmethod
    def _parse_metadata(meta: Dict) -> Dict:
        """Parse metadata from ChromaDB format"""
        try:
            full_meta = json.loads(meta.get('metadata_json', '{}'))
            return full_meta
        except Exception:
            return {
                "file": meta.get('file', ''),
                "source_type": meta.get('source_type', ''),
            }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
