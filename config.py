"""
Configuration settings for the Agentic RAG System
"""
import os

# ── Milvus Lite (local, no Docker required) ───────────────────────────────────
# Data is stored in a local file — zero infrastructure needed.
# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_PATH       = os.getenv("CHROMA_PATH", "./data/chroma_db")  # ✅ Changed from MILVUS_URI
COLLECTION_NAME   = "agentic_rag_v1"
VECTOR_DIM      = 384          # all-MiniLM-L6-v2 output dimension

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")  
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "llama3.2:latest")   # change to any local model
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # local SentenceTransformers model
USE_OLLAMA_EMBEDDINGS = False         # set True to use Ollama for embeddings

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE        = 512
CHUNK_OVERLAP     = 64
MAX_TOKENS        = 2048

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K             = 5       # base top-k
HYBRID_ALPHA      = 0.7     # weight for dense vs sparse (0=sparse, 1=dense)
RERANK_TOP_N      = 3       # after reranking

# ── Agent ─────────────────────────────────────────────────────────────────────
MAX_AGENT_STEPS   = 6
AGENT_TEMPERATURE = 0.2

# ── MCP Server ────────────────────────────────────────────────────────────────
MCP_HOST = "0.0.0.0"
MCP_PORT = 8765

# ── File Support ─────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".ppt", ".xlsx", ".xls", ".txt", ".csv"}

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
