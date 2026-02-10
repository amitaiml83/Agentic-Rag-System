"""
MCP (Model Context Protocol) Server
Exposes the Agentic RAG system as an MCP-compatible API.

Endpoints:
  POST /mcp/tools/list         –  list available tools
  POST /mcp/tools/call         –  invoke a tool
  GET  /mcp/resources/list     –  list ingested documents
  GET  /health                 –  health check
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.document_processor import process_file
from src.embeddings          import EmbeddingManager
# from vectorstore             import ChromaStore          # ChromaDB (local, no Docker)
from src.vector_store        import ChromaStore          # ChromaDB (local, no Docker)
from agents.rag_agent        import AgenticRAG           # LangChain-orchestrated agent
from config                  import MCP_HOST, MCP_PORT

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agentic RAG MCP Server",
    description="Model Context Protocol server for Agentic RAG",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Lazy-init globals ─────────────────────────────────────────────────────────
_embed_mgr: EmbeddingManager | None = None
_store:     ChromaStore      | None = None
_agent:     AgenticRAG       | None = None


def _get_agent() -> AgenticRAG:
    global _embed_mgr, _store, _agent
    if _agent is None:
        _embed_mgr = EmbeddingManager()
        _store     = ChromaStore(_embed_mgr)
        # AgenticRAG now builds its own LangChain ChatOllama internally
        _agent     = AgenticRAG(_store)
    return _agent


# ── MCP Schema ────────────────────────────────────────────────────────────────

MCP_TOOLS = [
    {
        "name": "query_documents",
        "description": (
            "Query the document knowledge base using the Agentic RAG pipeline. "
            "Supports aggregation questions (SUM, AVERAGE, COUNT, MAX, MIN) on Excel data."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query":    {"type": "string", "description": "Natural language question"},
                "strategy": {
                    "type": "string",
                    "enum": ["hybrid", "dense", "mmr", "keyword"],
                    "default": "hybrid",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "ingest_file",
        "description": "Ingest a file (PDF/DOCX/PPTX/XLSX/TXT) into the knowledge base.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path to the file"},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "list_documents",
        "description": "List all documents currently indexed in the knowledge base.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "delete_document",
        "description": "Remove a document from the knowledge base.",
        "inputSchema": {
            "type": "object",
            "properties": {"doc_id": {"type": "string"}},
            "required": ["doc_id"],
        },
    },
    {
        "name": "get_stats",
        "description": "Get statistics about the vector store.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

# ── Request / Response models ─────────────────────────────────────────────────

class ToolCallRequest(BaseModel):
    name:      str
    arguments: Dict[str, Any] = {}


class MCPResponse(BaseModel):
    content: List[Dict[str, Any]]
    isError: bool = False


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "agentic-rag-mcp", "version": "2.0.0"}


@app.post("/mcp/tools/list")
async def tools_list():
    return {"tools": MCP_TOOLS}


@app.post("/mcp/tools/call", response_model=MCPResponse)
async def tools_call(req: ToolCallRequest):
    agent = _get_agent()

    try:
        if req.name == "query_documents":
            query  = req.arguments.get("query", "")
            result = agent.run(query)
            return MCPResponse(content=[{
                "type": "text",
                "text": json.dumps({
                    "answer":      result.answer,
                    "confidence":  result.confidence,
                    "strategy":    result.strategy,
                    "num_sources": len(result.sources),
                    "sources": [
                        {
                            "file":  s.get("metadata", {}).get("file", "?"),
                            "score": s.get("score", 0),
                        }
                        for s in result.sources
                    ],
                }),
            }])

        elif req.name == "ingest_file":
            from pathlib import Path
            path   = Path(req.arguments["file_path"])
            chunks = process_file(path)
            count  = agent.store.upsert_chunks(chunks)
            return MCPResponse(content=[{
                "type": "text",
                "text": json.dumps({"chunks_indexed": count, "file": path.name}),
            }])

        elif req.name == "list_documents":
            docs = agent.store.list_documents()
            return MCPResponse(content=[{
                "type": "text",
                "text": json.dumps({"documents": docs, "count": len(docs)}),
            }])

        elif req.name == "delete_document":
            ok = agent.store.delete_document(req.arguments["doc_id"])
            return MCPResponse(content=[{
                "type": "text",
                "text": json.dumps({"deleted": ok}),
            }])

        elif req.name == "get_stats":
            count = agent.store.count()
            docs  = agent.store.list_documents()
            return MCPResponse(content=[{
                "type": "text",
                "text": json.dumps({
                    "total_chunks":    count,
                    "total_documents": len(docs),
                }),
            }])

        else:
            raise HTTPException(status_code=404, detail=f"Tool '{req.name}' not found")

    except Exception as e:
        return MCPResponse(
            content=[{"type": "text", "text": f"Tool execution error: {e}"}],
            isError=True,
        )


@app.get("/mcp/resources/list")
async def resources_list():
    agent = _get_agent()
    docs  = agent.store.list_documents()
    return {
        "resources": [
            {
                "uri":         f"rag://documents/{d['doc_id']}",
                "name":        d["file"],
                "description": f"Indexed document: {d['file']}",
                "mimeType":    "text/plain",
            }
            for d in docs
        ]
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host=MCP_HOST, port=MCP_PORT, log_level="info")
