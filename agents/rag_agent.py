from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from loguru import logger

# ── LangChain imports ─────────────────────────────────────────────────────────
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# from vectorstore import ChromaStore
from src.vector_store import ChromaStore
from config import (
    MAX_AGENT_STEPS, RERANK_TOP_N, TOP_K,
    OLLAMA_BASE_URL, OLLAMA_MODEL, AGENT_TEMPERATURE,
)


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class AgentStep:
    step_num:  int
    action:    str
    input:     str
    output:    str
    tool_used: str = ""


@dataclass
class AgentResult:
    answer:     str
    sources:    List[Dict[str, Any]]
    steps:      List[AgentStep]
    confidence: float = 0.0
    strategy:   str   = ""


# ── Prompt templates ───────────────────────────────────────────────────────────

_PLANNER_SYSTEM = """You are a retrieval planner for an Agentic RAG system.
Given the user query, output ONLY a valid JSON plan with these keys:
{{
  "intent": "factual|analytical|comparative|summarization|exploratory|aggregation",
  "retrieval_strategy": "dense|hybrid|mmr|keyword",
  "sub_questions": [],
  "source_hint": ""
}}

Use "aggregation" as intent when the query asks for SUM, AVERAGE, COUNT, MAX, MIN,
totals, or any mathematical operation on data — especially from spreadsheets.
Reply ONLY with valid JSON, no explanation."""

_PLANNER_HUMAN = "Query: {query}"

_ANALYSER_SYSTEM = """You are a context analyser. Given retrieved document chunks and a query,
decide if the context is sufficient to answer the query.
Output ONLY JSON: {{"sufficient": true, "missing": ""}}
Reply ONLY with valid JSON."""

_ANALYSER_HUMAN = "Query: {query}\n\nContext snippet:\n{context_snippet}"

_GENERATOR_SYSTEM = """You are an expert AI assistant answering questions from retrieved documents.

Rules:
- Answer ONLY from the provided context.
- Cite sources inline as [Source: filename, page/slide/sheet].
- If the answer is not in the context, say so explicitly.
- Be concise but complete.
- For aggregation questions (SUM, AVERAGE, COUNT, MAX, MIN) on Excel/spreadsheet data:
  * Use the numeric stats provided in the context (min, max, mean, sum).
  * Present them clearly in a table or bullet list.
  * If asked for a total/sum, use the "sum" value from the context.
  * If asked for average/mean, use the "mean" value.
- Structure longer answers with headings."""

_GENERATOR_HUMAN = """Context from documents:

{context}

---

Question: {query}"""

_REFLECTOR_SYSTEM = """Review this answer and rate its quality (0.0–1.0).
Output ONLY JSON: {{"confidence": 0.85, "issues": []}}
Consider: factual grounding in sources, completeness, clarity.
Reply ONLY with valid JSON."""

_REFLECTOR_HUMAN = "Query: {query}\n\nAnswer:\n{answer_snippet}"


# ── Chain builder ──────────────────────────────────────────────────────────────

def _make_json_chain(llm: ChatOllama, system_template: str, human_template: str):
    """Return a chain that always outputs parsed JSON (dict)."""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ])
    return prompt | llm | StrOutputParser() | RunnableLambda(_safe_parse_json)


def _make_text_chain(llm: ChatOllama, system_template: str, human_template: str):
    """Return a chain that outputs plain text."""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ])
    return prompt | llm | StrOutputParser()


def _safe_parse_json(text: str) -> dict:
    """Robustly extract a JSON dict from LLM output."""
    # Strip markdown fences
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        text = match.group(1)
    # Find first {...} block
    match = re.search(r"\{[\s\S]+\}", text)
    if match:
        text = match.group(0)
    try:
        return json.loads(text)
    except Exception:
        return {}


# ── Agent ──────────────────────────────────────────────────────────────────────

class AgenticRAG:
    """
    Orchestrates plan → retrieve → analyse → generate → reflect loop.
    All LLM calls go through typed LangChain chains.
    """

    def __init__(self, store: ChromaStore, llm=None):
        self.store = store

        # Build a ChatOllama instance (LangChain)
        self._llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=AGENT_TEMPERATURE,
        )

        # Keep a reference to the raw OllamaClient if provided (used for streaming)
        self._raw_llm = llm

        # Build reusable chains
        self._planner_chain   = _make_json_chain(self._llm, _PLANNER_SYSTEM,   _PLANNER_HUMAN)
        self._analyser_chain  = _make_json_chain(self._llm, _ANALYSER_SYSTEM,  _ANALYSER_HUMAN)
        self._generator_chain = _make_text_chain(self._llm, _GENERATOR_SYSTEM, _GENERATOR_HUMAN)
        self._reflector_chain = _make_json_chain(self._llm, _REFLECTOR_SYSTEM, _REFLECTOR_HUMAN)

    # ── Public entry point ─────────────────────────────────────────────────────

    def run(self, query: str, chat_history: Optional[List[Dict]] = None) -> AgentResult:
        steps: List[AgentStep] = []
        logger.info(f"Agent query: '{query}'")

        # Step 1 – Plan
        plan     = self._plan(query, steps)
        strategy = plan.get("retrieval_strategy", "hybrid")
        sub_qs   = plan.get("sub_questions", [])
        src_hint = plan.get("source_hint", "")
        intent   = plan.get("intent", "factual")

        # Step 2 – Retrieve
        all_docs = self._retrieve(query, strategy, src_hint, steps)

        # Step 3 – Sub-question retrieval
        if sub_qs and len(all_docs) < RERANK_TOP_N:
            for sq in sub_qs[:3]:
                sub_docs = self._retrieve(sq, "hybrid", src_hint, steps, tag=f"sub-q: {sq[:40]}")
                all_docs = _merge_deduplicate(all_docs, sub_docs)

        # Step 4 – No docs early exit
        if not all_docs:
            return AgentResult(
                answer    ="I couldn't find relevant information in the uploaded documents to answer this query.",
                sources   =[],
                steps     =steps,
                confidence=0.0,
                strategy  =strategy,
            )

        # Step 5 – Analyse sufficiency
        sufficient = self._analyse_context(query, all_docs[:RERANK_TOP_N * 2], steps)

        if not sufficient and len(all_docs) < RERANK_TOP_N:
            fallback = self._retrieve(query, "mmr", "", steps, tag="fallback-mmr")
            all_docs = _merge_deduplicate(all_docs, fallback)

        # Step 6 – Generate
        top_docs = all_docs[:RERANK_TOP_N + 2]
        answer   = self._generate(query, top_docs, chat_history, intent, steps)

        # Step 7 – Reflect
        confidence = self._reflect(query, answer, top_docs, steps)

        return AgentResult(
            answer    =answer,
            sources   =top_docs,
            steps     =steps,
            confidence=confidence,
            strategy  =strategy,
        )

    # ── Stream entry point ─────────────────────────────────────────────────────

    def stream(self, query: str, chat_history: Optional[List[Dict]] = None):
        """
        Yields (token_str | None, AgentResult | None).
        First yields are text tokens; final yield is (None, AgentResult).
        """
        steps: List[AgentStep] = []

        plan     = self._plan(query, steps)
        strategy = plan.get("retrieval_strategy", "hybrid")
        src_hint = plan.get("source_hint", "")
        sub_qs   = plan.get("sub_questions", [])
        intent   = plan.get("intent", "factual")

        all_docs = self._retrieve(query, strategy, src_hint, steps)
        if sub_qs:
            for sq in sub_qs[:2]:
                sub_docs = self._retrieve(sq, "hybrid", src_hint, steps)
                all_docs = _merge_deduplicate(all_docs, sub_docs)

        if not all_docs:
            yield None, AgentResult(
                answer="No relevant documents found.",
                sources=[], steps=steps, confidence=0.0, strategy=strategy,
            )
            return

        top_docs = all_docs[:RERANK_TOP_N + 2]
        context  = _build_context(top_docs)

        # Build the streaming prompt using LangChain ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(_GENERATOR_SYSTEM),
            HumanMessagePromptTemplate.from_template(_GENERATOR_HUMAN),
        ])

        full_answer = ""

        # Add history turns as extra messages if available
        messages = prompt.format_messages(query=query, context=context)
        if chat_history:
            from langchain_core.messages import HumanMessage, AIMessage
            history_msgs = []
            for h in chat_history[-6:]:
                if h["role"] == "user":
                    history_msgs.append(HumanMessage(content=h["content"]))
                elif h["role"] in ("assistant", "ai"):
                    history_msgs.append(AIMessage(content=h["content"]))
            # Insert history before the final user message
            messages = messages[:-1] + history_msgs + [messages[-1]]

        # Stream via LangChain
        for chunk in self._llm.stream(messages):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                full_answer += token
                yield token, None

        confidence = self._reflect_quick(full_answer, top_docs)
        yield None, AgentResult(
            answer    =full_answer,
            sources   =top_docs,
            steps     =steps,
            confidence=confidence,
            strategy  =strategy,
        )

    # ── Internal steps ─────────────────────────────────────────────────────────

    def _plan(self, query: str, steps: List[AgentStep]) -> Dict[str, Any]:
        try:
            plan = self._planner_chain.invoke({"query": query})
        except Exception as e:
            logger.warning(f"Planner chain error: {e}")
            plan = {"retrieval_strategy": "hybrid", "sub_questions": [], "source_hint": "", "intent": "factual"}

        if not isinstance(plan, dict) or not plan:
            plan = {"retrieval_strategy": "hybrid", "sub_questions": [], "source_hint": "", "intent": "factual"}

        steps.append(AgentStep(
            step_num=1, action="PLAN",
            input=query, output=json.dumps(plan), tool_used="planner_chain",
        ))
        logger.debug(f"Plan: {plan}")
        return plan

    def _retrieve(
        self,
        query:    str,
        strategy: str,
        src_hint: str,
        steps:    List[AgentStep],
        tag:      str = "",
    ) -> List[Dict[str, Any]]:
        label = f"RETRIEVE[{strategy}]" + (f" – {tag}" if tag else "")

        if strategy == "dense":
            docs = self.store.dense_search(query, top_k=TOP_K, source_filter=src_hint or None)
        elif strategy == "keyword":
            docs = self.store.keyword_search(query, top_k=TOP_K)
        elif strategy == "mmr":
            docs = self.store.mmr_search(query, top_k=TOP_K)
        else:
            docs = self.store.hybrid_search(query, top_k=TOP_K, source_filter=src_hint or None)

        steps.append(AgentStep(
            step_num=len(steps) + 1, action=label,
            input=query, output=f"Retrieved {len(docs)} chunks", tool_used=strategy,
        ))
        return docs

    def _analyse_context(
        self, query: str, docs: List[Dict], steps: List[AgentStep],
    ) -> bool:
        context_snippet = "\n".join(d["text"][:200] for d in docs[:4])
        try:
            result = self._analyser_chain.invoke({"query": query, "context_snippet": context_snippet})
            sufficient = result.get("sufficient", True) if isinstance(result, dict) else True
        except Exception as e:
            logger.warning(f"Analyser chain error: {e}")
            sufficient = True

        steps.append(AgentStep(
            step_num=len(steps) + 1, action="ANALYSE",
            input=f"{len(docs)} docs", output=f"Sufficient: {sufficient}",
            tool_used="analyser_chain",
        ))
        return sufficient

    def _generate(
        self,
        query:       str,
        docs:        List[Dict],
        chat_history: Optional[List[Dict]],
        intent:      str,
        steps:       List[AgentStep],
    ) -> str:
        context = _build_context(docs)

        # For aggregation intent, add an explicit instruction in the context header
        if intent == "aggregation":
            context = (
                "⚠️  This is an AGGREGATION query. "
                "Use the min/max/mean/sum statistics from Excel summary blocks to answer.\n\n"
                + context
            )

        try:
            # If chat history exists, build full messages manually
            if chat_history:
                from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
                msgs = [SystemMessage(content=_GENERATOR_SYSTEM)]
                for h in chat_history[-6:]:
                    if h["role"] == "user":
                        msgs.append(HumanMessage(content=h["content"]))
                    elif h["role"] in ("assistant", "ai"):
                        msgs.append(AIMessage(content=h["content"]))
                msgs.append(HumanMessage(content=_GENERATOR_HUMAN.format(context=context, query=query)))
                answer = (self._llm | StrOutputParser()).invoke(msgs)
            else:
                answer = self._generator_chain.invoke({"query": query, "context": context})
        except Exception as e:
            logger.error(f"Generator chain error: {e}")
            answer = f"Error generating answer: {e}"

        steps.append(AgentStep(
            step_num=len(steps) + 1, action="GENERATE",
            input=f"context={len(docs)} docs, intent={intent}",
            output=answer[:100] + "...", tool_used="generator_chain",
        ))
        return answer

    def _reflect(
        self, query: str, answer: str, docs: List[Dict], steps: List[AgentStep],
    ) -> float:
        try:
            result = self._reflector_chain.invoke({
                "query": query,
                "answer_snippet": answer[:600],
            })
            confidence = float(result.get("confidence", 0.7)) if isinstance(result, dict) else 0.7
        except Exception as e:
            logger.warning(f"Reflector chain error: {e}")
            confidence = 0.7

        steps.append(AgentStep(
            step_num=len(steps) + 1, action="REFLECT",
            input=answer[:80], output=f"Confidence: {confidence:.2f}",
            tool_used="reflector_chain",
        ))
        return confidence

    def _reflect_quick(self, answer: str, docs: List[Dict]) -> float:
        """Fast heuristic confidence score for streaming mode."""
        if not answer or len(answer) < 50:
            return 0.3
        has_source = any(
            d["text"][:80].lower() in answer.lower()[:500]
            for d in docs[:2]
        )
        base = 0.75 if has_source else 0.5
        return min(base + len(answer) / 5000, 0.95)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _merge_deduplicate(a: List[Dict], b: List[Dict]) -> List[Dict]:
    seen   = {d["chunk_id"] for d in a}
    result = list(a)
    for d in b:
        if d["chunk_id"] not in seen:
            seen.add(d["chunk_id"])
            result.append(d)
    return result


def _build_context(docs: List[Dict]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        meta  = doc.get("metadata", {})
        fname = meta.get("file", "unknown")
        extra = ""
        if "page" in meta:
            extra = f", page {meta['page']}"
        elif "slide" in meta:
            extra = f", slide {meta['slide']}"
        elif "sheet" in meta:
            extra = f", sheet '{meta['sheet']}'"
        parts.append(f"[{i}] Source: {fname}{extra}\n{doc['text']}")
    return "\n\n---\n\n".join(parts)
