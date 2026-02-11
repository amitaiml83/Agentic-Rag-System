from __future__ import annotations

import json
from typing import List, Dict, Any, Optional, Generator

import requests
from loguru import logger

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, AGENT_TEMPERATURE


class OllamaClient:
    """Thin wrapper around Ollama's /api/chat and /api/generate endpoints."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model:    str = OLLAMA_MODEL,
    ):
        self.base_url = base_url.rstrip("/")
        self.model    = model
        self._check_availability()

    # ── Health ────────────────────────────────────────────────────────────────

    def _check_availability(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            
            if models:
                logger.success(f"Ollama available. Models: {models}")
            
            # ✅ FIX: Flexible model matching (handles both "llama3.2" and "llama3.2:latest")
            if self.model not in models:
                # Try finding partial match without tag
                base_model = self.model.split(":")[0]  # Extract base name
                matching = [m for m in models if m.startswith(base_model)]
                
                if matching:
                    logger.info(f"Using '{matching[0]}' for requested model '{self.model}'")
                    self.model = matching[0]
                elif models:
                    logger.warning(
                        f"Model '{self.model}' not found. Available: {models}. "
                        f"Using first: '{models[0]}'"
                    )
                    self.model = models[0]
            return True
        except Exception as e:
            logger.error(f"Ollama not reachable at {self.base_url}: {e}")
            return False


    # ── Non-streaming completion ──────────────────────────────────────────────

    def chat(
        self,
        messages:    List[Dict[str, str]],
        temperature: float = AGENT_TEMPERATURE,
        max_tokens:  int   = 2048,
        json_mode:   bool  = False,
    ) -> str:
        """Send messages, return full response string."""
        payload = {
            "model":    self.model,
            "messages": messages,
            "stream":   False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"

        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
            return data.get("message", {}).get("content", "")
        except requests.Timeout:
            logger.error("Ollama request timed out")
            return "Error: LLM request timed out. Please try again."
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            return f"Error communicating with Ollama: {e}"

    # ── Streaming completion ──────────────────────────────────────────────────

    def stream_chat(
        self,
        messages:    List[Dict[str, str]],
        temperature: float = AGENT_TEMPERATURE,
        max_tokens:  int   = 2048,
    ) -> Generator[str, None, None]:
        """Stream tokens as they arrive."""
        payload = {
            "model":    self.model,
            "messages": messages,
            "stream":   True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            with requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=120,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            token = chunk.get("message", {}).get("content", "")
                            if token:
                                yield token
                            if chunk.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"\n[Streaming error: {e}]"

    # ── Convenience ───────────────────────────────────────────────────────────

    def simple_completion(self, prompt: str, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)
