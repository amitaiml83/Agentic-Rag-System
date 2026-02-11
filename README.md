# 🤖 Agentic RAG System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-Orchestrated-green.svg)](https://langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Local-orange.svg)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/Ollama-LLM-red.svg)](https://ollama.ai/)

An intelligent document Q&A system powered by **Agentic RAG** (Retrieval-Augmented Generation) with LangChain orchestration. Upload documents and ask questions - the AI agent will intelligently search, analyze, and provide accurate answers with source citations.

---

## ✨ Features

### 🎯 Core Capabilities
- **Multi-format Support**: PDF, DOCX, PPTX, XLSX, TXT, CSV
- **Intelligent Retrieval**: Dense, keyword, hybrid (RRF), and MMR diversity search
- **Excel Aggregation**: Direct SUM, AVERAGE, COUNT, MAX, MIN queries on spreadsheet data
- **Source Citations**: Inline citations with page/slide/sheet references
- **Confidence Scoring**: Quality assessment for every answer
- **Chat History**: Context-aware multi-turn conversations

### 🧠 Agentic Architecture
The system uses a **6-step agent loop**:
1. **PLAN** → Analyze query, decide retrieval strategy
2. **RETRIEVE** → Call appropriate tools (dense/hybrid/MMR/keyword)
3. **ANALYSE** → Evaluate if context is sufficient
4. **REFINE** → Optionally decompose and re-retrieve sub-questions
5. **GENERATE** → Produce final grounded answer with citations
6. **REFLECT** → Score confidence, flag unsupported claims

### 🛠️ Technology Stack
- **LLM Backend**: [Ollama](https://ollama.ai/) (local, privacy-first)
- **Orchestration**: [LangChain](https://langchain.com/) (typed chains, structured outputs)
- **Vector Store**: [ChromaDB](https://www.trychroma.com/) (local, no Docker required)
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`) or Ollama embeddings
- **UI**: Streamlit (modern, responsive chat interface)
- **MCP Server**: FastAPI (Model Context Protocol compatible)

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai/download) installed and running

### Step 1: Clone & Install Dependencies
```bash
git clone <your-repo-url>
cd agentic-rag-system
pip install -r requirements.txt
```

### Step 2: Install Ollama Models
```bash
# Install LLM model (default: llama3.2)
ollama pull llama3.2

# Optional: Install embedding model if using Ollama embeddings
ollama pull nomic-embed-text
```

### Step 3: Start Ollama Server
```bash
ollama serve
```

---

## 🚀 Quick Start

### Option 1: Streamlit UI (Recommended)
```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` and:
1. Upload documents via the sidebar
2. Click "⚡ Ingest Documents"
3. Ask questions in the chat interface

### Option 2: MCP Server (API)
```bash
python mcp/server.py
```

Server runs at `http://localhost:8765`. See [MCP API Documentation](#mcp-api) below.

---

## 📚 Usage Examples

### Basic Document Q&A
```
Upload: quarterly_report.pdf

Q: "Summarize the key findings from Q4"
A: Based on the Q4 report, revenue increased by 23%... [Source: quarterly_report.pdf, page 3]

Q: "What were the main challenges mentioned?"
A: The report highlighted three primary challenges... [Source: quarterly_report.pdf, page 5]
```

### Excel Aggregation Queries
```
Upload: sales_data.xlsx

Q: "What is the total sales amount?"
A: The total sales across all regions is $1,247,893.00 [Source: sales_data.xlsx, sheet 'Q1 Sales']

Q: "What's the average deal size?"
A: The average deal size is $34,219.50 (mean across 365 transactions) [Source: sales_data.xlsx, sheet 'Deals']

Q: "Show me the maximum and minimum revenue by region"
A: 
- Maximum: North America - $523,450
- Minimum: APAC - $89,234
[Source: sales_data.xlsx, sheet 'Regional Summary']
```

### Multi-document Analysis
```
Upload: contract_v1.docx, contract_v2.docx

Q: "What are the differences between the two contracts?"
A: Comparing the contracts:
- Payment terms: v1 has Net-30, v2 changed to Net-45 [Source: contract_v1.docx, contract_v2.docx]
- Scope: v2 added cloud hosting services [Source: contract_v2.docx, page 2]
```

---

## 🏗️ Architecture

### Project Structure
```
agentic-rag-system/
├── agents/
│   └── rag_agent.py          # LangChain-orchestrated agent loop
├── mcp/
│   └── server.py             # FastAPI MCP server
├── src/
│   ├── document_processor.py # Multi-format extraction & chunking
│   ├── embeddings.py         # SentenceTransformers / Ollama embeddings
│   ├── llm_client.py         # Ollama LLM client
│   └── vector_store.py       # ChromaDB store with hybrid retrieval
├── app.py                    # Streamlit UI
├── config.py                 # Configuration settings
└── requirements.txt
```

### Data Flow
```
┌─────────────┐
│  Documents  │
│ (PDF/DOCX)  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Document        │
│ Processor       │  → Extract text, tables, metadata
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Chunker         │  → Sentence-aware sliding window
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Embedding       │  → SentenceTransformers / Ollama
│ Manager         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ChromaDB        │  → Local vector store
│ (Persistent)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Agentic RAG     │  → 6-step agent loop
│ (LangChain)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Ollama LLM      │  → Generate grounded answer
└─────────────────┘
```

---

## ⚙️ Configuration

Edit `config.py` to customize:

### Ollama Settings
```python
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "llama3.2:latest"  # Change to any local model
OLLAMA_EMBED_MODEL = "nomic-embed-text"
```

### Embedding Backend
```python
# Option 1: Local SentenceTransformers (default, faster)
USE_OLLAMA_EMBEDDINGS = False
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Option 2: Ollama embeddings
USE_OLLAMA_EMBEDDINGS = True
```

### Retrieval Parameters
```python
TOP_K        = 5      # Number of chunks to retrieve
HYBRID_ALPHA = 0.7    # Dense vs. keyword weight (0=keyword, 1=dense)
RERANK_TOP_N = 3      # Final chunks after reranking
```

### Chunking Strategy
```python
CHUNK_SIZE    = 512   # Words per chunk
CHUNK_OVERLAP = 64    # Overlap between chunks
```

### Agent Behavior
```python
MAX_AGENT_STEPS   = 6    # Max retrieval iterations
AGENT_TEMPERATURE = 0.2  # LLM temperature (lower = more factual)
```

---

## 🔌 MCP API

The system exposes an MCP-compatible API for programmatic access.

### Start Server
```bash
python mcp/server.py
# Server: http://localhost:8765
```

### Available Tools

#### 1. Query Documents
```bash
POST /mcp/tools/call
Content-Type: application/json

{
  "name": "query_documents",
  "arguments": {
    "query": "What is the total revenue?",
    "strategy": "hybrid"  # optional: dense | keyword | mmr | hybrid
  }
}
```

Response:
```json
{
  "content": [{
    "type": "text",
    "text": "{\"answer\": \"...\", \"confidence\": 0.87, \"strategy\": \"hybrid\", \"num_sources\": 3}"
  }],
  "isError": false
}
```

#### 2. Ingest File
```bash
POST /mcp/tools/call
{
  "name": "ingest_file",
  "arguments": {
    "file_path": "/path/to/document.pdf"
  }
}
```

#### 3. List Documents
```bash
POST /mcp/tools/call
{
  "name": "list_documents",
  "arguments": {}
}
```

#### 4. Delete Document
```bash
POST /mcp/tools/call
{
  "name": "delete_document",
  "arguments": {
    "doc_id": "document_abc123"
  }
}
```

#### 5. Get Stats
```bash
POST /mcp/tools/call
{
  "name": "get_stats",
  "arguments": {}
}
```

### Health Check
```bash
GET /health
# Response: {"status": "ok", "service": "agentic-rag-mcp", "version": "2.0.0"}
```

---

## 🧪 Advanced Features

### Retrieval Strategies

#### 1. Dense Search (Semantic)
Best for: Conceptual questions, paraphrasing detection
```python
docs = store.dense_search("What are the main risks?", top_k=5)
```

#### 2. Keyword Search
Best for: Exact term matching, technical queries
```python
docs = store.keyword_search("API rate limit", top_k=5)
```

#### 3. Hybrid Search (RRF)
Best for: General queries (combines dense + keyword)
```python
docs = store.hybrid_search("quarterly revenue growth", top_k=5)
```

#### 4. MMR (Maximal Marginal Relevance)
Best for: Diverse results, exploratory queries
```python
docs = store.mmr_search("market trends", top_k=5, lambda_mult=0.6)
```

### Excel Aggregation

The system automatically extracts numeric statistics from Excel files:

**Auto-generated summary block:**
```
Sheet: Sales_Data
Total rows: 1,247
Columns: Region, Product, Revenue, Quantity, Date

=== NUMERIC COLUMN STATISTICS ===
Column 'Revenue': count=1247, sum=1247893.50, mean=1000.72, min=45.00, max=25430.00
Column 'Quantity': count=1247, sum=15892, mean=12.74, min=1, max=500

=== JSON STATS (machine-readable) ===
{
  "sheet": "Sales_Data",
  "row_count": 1247,
  "columns": {
    "Revenue": {"count": 1247, "sum": 1247893.50, "mean": 1000.72, "min": 45.00, "max": 25430.00}
  }
}
```

**Query examples:**
- "What is the total revenue?" → Uses `sum` value
- "What's the average sale?" → Uses `mean` value
- "What's the largest order?" → Uses `max` value

---

## 🐛 Troubleshooting

### Ollama Not Running
```
Error: ❌ Ollama is not running
Solution: Start Ollama with `ollama serve` in a terminal
```

### Model Not Found
```
Error: Model 'llama3.2' not found
Solution: Pull the model with `ollama pull llama3.2`
```

### ChromaDB Permission Error
```
Error: Permission denied writing to ./data/chroma_db
Solution: mkdir -p data/chroma_db && chmod -R 755 data/
```

### Out of Memory (Large Documents)
```
Solution: Reduce CHUNK_SIZE in config.py or split large files
```

### Slow Embeddings
```
Solution: Set USE_OLLAMA_EMBEDDINGS = False to use local SentenceTransformers
```

---

## 📊 Performance Tips

1. **Embedding Backend**: Local SentenceTransformers is ~5x faster than Ollama embeddings
2. **Chunk Size**: Smaller chunks (256-512) = better precision; larger (1024+) = better recall
3. **Top-K Tuning**: Start with 5, increase to 10 for complex queries
4. **Hybrid Alpha**: 0.7 works well for most cases; increase for semantic queries, decrease for exact matching
5. **Model Selection**: Smaller models (llama3.2) are faster; larger (llama3.1:70b) are more accurate

---

## 🔒 Privacy & Security

- **100% Local**: All data stays on your machine (no cloud APIs)
- **No Telemetry**: ChromaDB telemetry disabled by default
- **File Isolation**: Uploaded files processed in temp directories, auto-cleaned
- **MCP CORS**: Configured for localhost only (update for production)

---

## 🛣️ Roadmap

- [ ] Multi-language support
- [ ] Image/diagram extraction from PDFs
- [ ] Graph RAG for entity relationships
- [ ] Fine-tuning on domain-specific data
- [ ] Web scraping integration
- [ ] Export chat history to markdown

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) - Agent orchestration
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Ollama](https://ollama.ai/) - Local LLM runtime
- [SentenceTransformers](https://www.sbert.net/) - Embeddings
- [Streamlit](https://streamlit.io/) - UI framework

---

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: support@yourproject.com

---

**Built with ❤️ using LangChain, ChromaDB, and Ollama**
