"""
Agentic RAG System Streamlit UI
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import time
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background: #0f1117; color: #e0e0e0; }

  .user-bubble {
    background: #2d3250;
    border-radius: 16px 16px 4px 16px;
    padding: 0.8rem 1.2rem;
    margin: 0.8rem 0;
    max-width: 80%;
    margin-left: auto;
    color: #e0e0e0;
    font-size: 0.95rem;
  }
  .bot-bubble {
    background: #1a2332;
    border: 1px solid #2d5a8e;
    border-radius: 4px 16px 16px 16px;
    padding: 1rem 1.4rem;
    margin: 0.8rem 0;
    max-width: 90%;
    color: #c8d8e8;
    font-size: 0.95rem;
    line-height: 1.6;
  }

  .rag-card {
    background: #1e2130;
    border: 1px solid #2d3250;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
  }

  .file-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1a2540;
    border: 1px solid #2d4a7a;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.85rem;
    color: #8ab4f8;
    margin: 4px;
  }

  h1, h2, h3 { color: #8ab4f8 !important; }

  .stButton > button {
    background: #2d5a9e !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
  }
  .stButton > button:hover {
    background: #3a6fb5 !important;
    transform: translateY(-1px);
    transition: all 0.2s;
  }

  section[data-testid="stSidebar"] { background: #141824 !important; }

  .metric-row {
    display: flex;
    gap: 8px;
    margin: 8px 0;
    font-size: 0.8rem;
  }
  .metric-badge {
    background: #1e2130;
    border: 1px solid #2d3250;
    border-radius: 6px;
    padding: 4px 10px;
    color: #aaa;
  }
  .metric-value { color: #8ab4f8; font-weight: bold; }

  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "messages":      [],
    "agent":         None,
    "store":         None,
    "llm":           None,
    "indexed_files": [],
    "system_ready":  False,
    "total_chunks":  0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Ollama health check ───────────────────────────────────────────────────────
def _ollama_is_alive() -> bool:
    """Ping Ollama HTTP API. Returns False immediately if the server is down."""
    import requests
    from config import OLLAMA_BASE_URL
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ── Lazy init (NOT cached — we want a live check every call) ──────────────────
def init_system():
    from src.embeddings import EmbeddingManager
    # from vectorstore import ChromaStore
    from src.vector_store import ChromaStore
    from agents.rag_agent import AgenticRAG

    embed_mgr = EmbeddingManager()
    store     = ChromaStore(embed_mgr)
    agent     = AgenticRAG(store)
    return store, agent


def get_system():
    """
    Returns the agent ONLY if Ollama is currently reachable.
    Resets system_ready if Ollama has gone down since last init.
    """
    # Live health check on every call — catches stopped Ollama immediately
    if not _ollama_is_alive():
        st.session_state.system_ready = False
        st.error(
            "❌ Ollama is not running. "
            "Start it with `ollama serve` in your terminal, then retry."
        )
        return None

    if not st.session_state.system_ready:
        try:
            store, agent = init_system()
            st.session_state.store        = store
            st.session_state.agent        = agent
            st.session_state.system_ready = True
            st.session_state.total_chunks = store.count()
        except Exception as e:
            st.error(f"❌ System initialization error: {e}")
            return None

    return st.session_state.agent


# ── Helpers ───────────────────────────────────────────────────────────────────
FILE_ICONS = {
    ".pdf": "📄", ".docx": "📝", ".pptx": "📊", ".ppt": "📊",
    ".xlsx": "📈", ".xls": "📈", ".txt": "📃", ".csv": "📋",
}

def _file_icon(name: str) -> str:
    return FILE_ICONS.get(Path(name).suffix.lower(), "📎")

def _confidence_color(conf: float) -> str:
    if conf >= 0.8: return "#27ae60"
    if conf >= 0.6: return "#f39c12"
    return "#e74c3c"


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:1.5rem 0;">'
            '<span style="font-size:3rem;">🤖</span><br>'
            '<span style="font-size:1.4rem;font-weight:bold;color:#8ab4f8;">Agentic RAG</span><br>'
            '<span style="font-size:0.8rem;color:#666;">Intelligent Document Q&A</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Upload ─────────────────────────────────────────────────────────
        st.markdown("### 📁 Upload Documents")
        uploaded = st.file_uploader(
            "Drop files here",
            type=["pdf", "docx", "pptx", "ppt", "xlsx", "xls", "txt", "csv"],
            accept_multiple_files=True,
            help="Supported: PDF, DOCX, PPTX, XLSX, TXT, CSV",
            label_visibility="collapsed",
        )

        if uploaded and st.button("⚡ Ingest Documents", use_container_width=True):
            agent = get_system()
            if agent:
                progress  = st.progress(0, "Processing...")
                total_new = 0

                for i, uf in enumerate(uploaded):
                    progress.progress(i / len(uploaded), f"Processing {uf.name}...")
                    try:
                        suffix   = Path(uf.name).suffix
                        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                            tmp.write(uf.read())
                            tmp_path = Path(tmp.name)

                        from src.document_processor import process_file
                        chunks = process_file(tmp_path)

                        # Patch metadata with original filename
                        for c in chunks:
                            c.metadata["file"] = uf.name

                        count      = agent.store.upsert_chunks(chunks)
                        total_new += count

                        if uf.name not in [f["name"] for f in st.session_state.indexed_files]:
                            st.session_state.indexed_files.append({"name": uf.name, "chunks": count})

                        tmp_path.unlink(missing_ok=True)
                    except Exception as e:
                        st.error(f"Error: {uf.name} - {e}")

                progress.progress(1.0, "✅ Complete!")
                st.session_state.total_chunks = agent.store.count()
                st.success(f"✅ Indexed {total_new} chunks from {len(uploaded)} file(s)")
                time.sleep(1.5)
                st.rerun()

        st.divider()

        # ── Indexed files ──────────────────────────────────────────────────
        if st.session_state.indexed_files:
            st.markdown("**📚 Indexed Documents**")
            for f in st.session_state.indexed_files:
                icon = _file_icon(f["name"])
                st.markdown(
                    f'<div class="file-chip">{icon} {f["name"]} '
                    f'<span style="color:#666;">· {f["chunks"]} chunks</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No documents indexed yet. Upload files to get started.")

        st.divider()

        # ── Quick stats ────────────────────────────────────────────────────
        st.markdown(
            f'<div class="metric-row">'
            f'<div class="metric-badge">📊 Chunks: <span class="metric-value">'
            f'{st.session_state.total_chunks}</span></div>'
            f'<div class="metric-badge">📁 Files: <span class="metric-value">'
            f'{len(st.session_state.indexed_files)}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if len(st.session_state.messages) > 0:
            st.divider()
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()


# ── Main chat ─────────────────────────────────────────────────────────────────
def render_chat():
    st.markdown(
        '<h1 style="margin-bottom:0.3rem;">🤖 Agentic RAG System</h1>'
        '<p style="color:#666;margin-top:0;font-size:0.9rem;">Ask questions about your documents</p>',
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        st.markdown("""
        <div class="rag-card">
            <h3 style="margin-top:0;color:#8ab4f8;">👋 Welcome!</h3>
            <p style="color:#aaa;line-height:1.6;">
                Upload your documents using the sidebar, then ask any questions.<br>
                The AI agent will intelligently search and provide accurate answers with source citations.
            </p>
            <p style="color:#666;font-size:0.85rem;margin-bottom:0;">
                ✨ <strong>Features:</strong> LangChain orchestration · Hybrid retrieval ·
                Excel aggregation · Source tracking · Confidence scoring
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.total_chunks > 0:
            st.markdown("**💡 Try asking:**")
            examples = [
                "Summarize the key points",
                "What is the total sales amount?",
                "What is the average value per column?",
                "Show me the maximum and minimum values",
            ]
            cols = st.columns(2)
            for i, q in enumerate(examples):
                with cols[i % 2]:
                    if st.button(q, key=f"ex_{i}", use_container_width=True):
                        st.session_state.messages.append({"role": "user", "content": q})
                        st.rerun()

    # Render history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">👤 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            content = msg["content"].replace("\n", "<br>")
            st.markdown(
                f'<div class="bot-bubble">🤖 {content}</div>',
                unsafe_allow_html=True,
            )
            if "confidence" in msg:
                conf  = msg["confidence"]
                color = _confidence_color(conf)
                strat = msg.get("strategy", "auto")
                st.markdown(
                    f'<div style="margin-top:-8px;margin-bottom:12px;font-size:0.75rem;">'
                    f'<span style="background:{color}22;border:1px solid {color};color:{color};'
                    f'padding:2px 8px;border-radius:10px;margin-right:6px;">📊 {conf*100:.0f}%</span>'
                    f'<span style="background:#8ab4f822;border:1px solid #8ab4f8;color:#8ab4f8;'
                    f'padding:2px 8px;border-radius:10px;">🎯 {strat}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Chat input
    st.markdown("<br>", unsafe_allow_html=True)
    query = st.chat_input("💬 Ask anything about your documents...")

    if query:
        agent = get_system()
        if not agent:
            st.error("❌ System not ready. Please check Ollama is running.")
            return

        if st.session_state.total_chunks == 0:
            st.warning("⚠️ No documents indexed yet. Upload files in the sidebar first.")
            return

        st.session_state.messages.append({"role": "user", "content": query})

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[-6:]
            if m["role"] in ("user", "assistant")
        ]

        # ── Pre-stream health guard ───────────────────────────────────────
        if not _ollama_is_alive():
            st.session_state.system_ready = False
            st.session_state.messages.pop()   # remove the user msg we just added
            st.error(
                "❌ Ollama is not running. "
                "Start it with `ollama serve` in your terminal, then retry."
            )
            return

        with st.spinner("🤔 Thinking..."):
            placeholder  = st.empty()
            full_text    = ""
            final_result = None
            stream_error = None

            try:
                for token, result in agent.stream(query, history):
                    if token is not None:
                        full_text += token
                        display_text = full_text.replace("\n", "<br>")
                        placeholder.markdown(
                            f'<div class="bot-bubble">🤖 {display_text}▌</div>',
                            unsafe_allow_html=True,
                        )
                    if result is not None:
                        final_result = result
            except Exception as e:
                stream_error = str(e)

            placeholder.empty()

        if stream_error:
            st.session_state.messages.pop()   # discard incomplete user msg
            st.error(f"❌ LLM error — Ollama may have stopped mid-response: {stream_error}")
            st.session_state.system_ready = False
            return

        # Only save if we actually got a response
        if full_text.strip():
            bot_msg = {
                "role":       "assistant",
                "content":    full_text,
                "confidence": final_result.confidence if final_result else 0.7,
                "strategy":   final_result.strategy   if final_result else "auto",
            }
            st.session_state.messages.append(bot_msg)

        st.rerun()


# ── Run ───────────────────────────────────────────────────────────────────────
render_sidebar()
render_chat()
