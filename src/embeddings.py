"""
Embedding Manager  –  local SentenceTransformers (default) or Ollama embeddings.
Provides a unified interface for both options.
"""
from __future__ import annotations

from typing import List
import numpy as np

from loguru import logger
from config import EMBEDDING_MODEL, USE_OLLAMA_EMBEDDINGS, OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL


class EmbeddingManager:
    """
    Handles text → vector conversion.
    Uses SentenceTransformers by default (100% offline, fast).
    Falls back to Ollama embeddings if USE_OLLAMA_EMBEDDINGS=True.
    """

    def __init__(self):
        self._model = None
        self._use_ollama = USE_OLLAMA_EMBEDDINGS
        self._init()

    # ── Init ──────────────────────────────────────────────────────────────────

    def _init(self):
        if self._use_ollama:
            logger.info(f"Embedding backend: Ollama ({OLLAMA_EMBED_MODEL})")
        else:
            logger.info(f"Embedding backend: SentenceTransformers ({EMBEDDING_MODEL})")
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(EMBEDDING_MODEL)
                logger.success("SentenceTransformer loaded")
            except Exception as e:
                logger.error(f"SentenceTransformer load failed: {e}")
                logger.warning("Falling back to Ollama embeddings")
                self._use_ollama = True

    # ── Core ──────────────────────────────────────────────────────────────────

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Return list of float vectors (one per text)."""
        if not texts:
            return []
        if self._use_ollama:
            return self._ollama_embed(texts)
        return self._st_embed(texts)

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]

    # ── Backends ──────────────────────────────────────────────────────────────

    def _st_embed(self, texts: List[str]) -> List[List[float]]:
        try:
            vecs = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            return vecs.tolist()
        except Exception as e:
            logger.error(f"SentenceTransformer encode error: {e}")
            return [[0.0] * 384] * len(texts)

    def _ollama_embed(self, texts: List[str]) -> List[List[float]]:
        import requests
        results = []
        for text in texts:
            try:
                r = requests.post(
                    f"{OLLAMA_BASE_URL}/api/embeddings",
                    json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
                    timeout=30
                )
                r.raise_for_status()
                results.append(r.json()["embedding"])
            except Exception as e:
                logger.error(f"Ollama embed error: {e}")
                results.append([0.0] * 384)
        return results

    # ── Dimension ─────────────────────────────────────────────────────────────

    @property
    def dim(self) -> int:
        if self._use_ollama:
            # probe once
            try:
                v = self._ollama_embed(["probe"])
                return len(v[0])
            except:
                return 768
        if self._model:
            return self._model.get_sentence_embedding_dimension()
        return 384
