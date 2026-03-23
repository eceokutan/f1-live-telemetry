"""Local GGUF inference client for Jarvis Post post-race analysis."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .client import LLMError


@dataclass
class StreamComplete:
    """Sentinel yielded as the final item from generate_stream()."""
    tokens_used: int
    finish_reason: str

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = "postrace_gguf/granite-postrace-analyst-Q4_K_M.gguf"


class LocalGGUFClient:
    """
    Client that runs a quantized GGUF model locally via llama-cpp-python.

    Implements the same async ``generate()`` interface as ``HFClient`` so that
    Jarvis Post agents work without any changes.
    """

    def __init__(self) -> None:
        self._model: Optional[object] = None
        self._loaded = False
        self._lock = threading.Lock()

        # Resolve model path
        env_path = (os.getenv("POSTRACE_GGUF_MODEL_PATH") or "").strip()
        if env_path:
            self._model_path = Path(env_path)
        else:
            self._model_path = Path(_DEFAULT_MODEL_PATH)

        if not self._model_path.is_absolute():
            self._model_path = Path(__file__).resolve().parent.parent.parent / self._model_path

        # Context / threading config
        env_n_ctx = (os.getenv("POSTRACE_GGUF_N_CTX") or "").strip()
        self._n_ctx = int(env_n_ctx) if env_n_ctx else 8192

        env_n_threads = (os.getenv("POSTRACE_GGUF_N_THREADS") or "").strip()
        if env_n_threads:
            self._n_threads = max(1, int(env_n_threads))
        else:
            # Match live mode default: half of available logical CPUs.
            self._n_threads = max(1, (os.cpu_count() or 2) // 2)

        logger.info(
            "Post-race GGUF threading configured: n_threads=%d (logical_cpus=%s, env_override=%s)",
            self._n_threads,
            os.cpu_count(),
            "set" if env_n_threads else "unset",
        )

    def _load_model(self) -> None:
        """Load the GGUF model into memory (lazy, thread-safe)."""
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            if not self._model_path.exists():
                auto_download_enabled = (
                    os.getenv("POSTRACE_GGUF_AUTO_DOWNLOAD", "1").strip().lower()
                    not in {"0", "false", "no", "off"}
                )
                if auto_download_enabled:
                    # Try auto-downloading from Hugging Face Hub
                    try:
                        from ai.model_downloader import ensure_postrace_model
                        logger.info("Post-race model not found locally, attempting auto-download...")
                        downloaded = ensure_postrace_model()
                        self._model_path = downloaded
                    except Exception as dl_err:
                        raise FileNotFoundError(
                            f"Post-race GGUF model not found at {self._model_path} "
                            f"and auto-download failed: {dl_err}. "
                            "Run the conversion script to generate it."
                        ) from dl_err
                else:
                    raise FileNotFoundError(
                        f"Post-race GGUF model not found at {self._model_path}. "
                        "Set POSTRACE_GGUF_AUTO_DOWNLOAD=1 to allow automatic download."
                    )

            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise LLMError(
                    "llama-cpp-python is not installed. "
                    "Install with: pip install llama-cpp-python"
                ) from exc

            logger.info(
                "Loading post-race GGUF model from %s (n_ctx=%d, n_threads=%d) ...",
                self._model_path,
                self._n_ctx,
                self._n_threads,
            )
            self._model = Llama(
                model_path=str(self._model_path),
                n_ctx=self._n_ctx,
                n_threads=self._n_threads,
                n_gpu_layers=0,
                n_batch=512,
                verbose=False,
            )
            self._loaded = True
            logger.info("Post-race GGUF model loaded successfully")

    def _format_prompt(self, prompt: str, system_prompt: str) -> str:
        """Format prompt using Granite chat template role markers."""
        parts: list[str] = []
        if system_prompt.strip():
            parts.append(
                f"<|start_of_role|>system<|end_of_role|>{system_prompt.strip()}<|end_of_text|>"
            )
        parts.append(
            f"<|start_of_role|>user<|end_of_role|>{prompt}<|end_of_text|>"
        )
        parts.append("<|start_of_role|>assistant<|end_of_role|>")
        return "".join(parts)

    async def generate(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.5,
    ) -> dict:
        """
        Generate a completion using the local GGUF model.

        Returns the same dict shape as HFClient:
            {"content": str, "tokens_used": int, "finish_reason": str}
        """
        self._load_model()

        formatted = self._format_prompt(prompt, system_prompt)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            self._sync_generate,
            formatted,
            max_tokens,
            temperature,
        )
        return result

    def _sync_generate(
        self,
        formatted_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> dict:
        """Run the actual llama-cpp completion (blocking, called from executor)."""
        with self._lock:
            try:
                response = self._model.create_completion(
                    formatted_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=50,
                    top_p=0.95,
                    stop=["<|end_of_text|>", "<|start_of_role|>"],
                )
            except Exception as exc:
                raise LLMError(f"Local GGUF generation failed: {exc}") from exc

        choice = response["choices"][0]
        content = (choice.get("text") or "").strip()
        finish_reason = choice.get("finish_reason", "stop")
        tokens_used = response.get("usage", {}).get("completion_tokens", 0)

        return {
            "content": content,
            "tokens_used": tokens_used,
            "finish_reason": finish_reason,
        }

    def _format_messages(self, messages: list[dict], system_prompt: str) -> str:
        """Format multi-turn messages using Granite chat template role markers.

        Args:
            messages: List of {"role": "user"|"assistant", "content": str}.
                      The final entry must have role "user".
            system_prompt: System prompt placed at the start of the template.
        """
        parts: list[str] = []
        if system_prompt.strip():
            parts.append(
                f"<|start_of_role|>system<|end_of_role|>{system_prompt.strip()}<|end_of_text|>"
            )
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(
                f"<|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>"
            )
        # Trailing assistant marker (no end_of_text) to prompt generation
        parts.append("<|start_of_role|>assistant<|end_of_role|>")
        return "".join(parts)

    def generate_stream_messages(self, messages, system_prompt, max_tokens=300, temperature=0.6):
        """Synchronous generator for multi-turn conversations.

        Same lock/yield/StreamComplete pattern as generate_stream(), but accepts
        a list of message dicts instead of a single prompt string.
        """
        self._load_model()
        formatted = self._format_messages(messages, system_prompt)
        self._lock.acquire()
        try:
            response_iter = self._model.create_completion(
                formatted,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                stop=["<|end_of_text|>", "<|start_of_role|>"],
                stream=True,
            )
            tokens_used = 0
            finish_reason = "stop"
            for chunk in response_iter:
                choice = chunk["choices"][0]
                text = choice.get("text", "")
                if text:
                    yield text
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                usage = chunk.get("usage")
                if usage:
                    tokens_used = usage.get("completion_tokens", 0)
            yield StreamComplete(tokens_used=tokens_used, finish_reason=finish_reason)
        except Exception as exc:
            raise LLMError(f"Local GGUF streaming failed: {exc}") from exc
        finally:
            self._lock.release()

    def generate_stream(self, prompt, system_prompt, max_tokens=1000, temperature=0.5):
        """Synchronous generator yielding str chunks, then a final StreamComplete."""
        self._load_model()
        formatted = self._format_prompt(prompt, system_prompt)
        self._lock.acquire()
        try:
            response_iter = self._model.create_completion(
                formatted,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                stop=["<|end_of_text|>", "<|start_of_role|>"],
                stream=True,
            )
            tokens_used = 0
            finish_reason = "stop"
            for chunk in response_iter:
                choice = chunk["choices"][0]
                text = choice.get("text", "")
                if text:
                    yield text
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                usage = chunk.get("usage")
                if usage:
                    tokens_used = usage.get("completion_tokens", 0)
            yield StreamComplete(tokens_used=tokens_used, finish_reason=finish_reason)
        except Exception as exc:
            raise LLMError(f"Local GGUF streaming failed: {exc}") from exc
        finally:
            self._lock.release()

    def warmup_sync(self) -> bool:
        """
        Synchronous warmup: loads model into memory.

        Returns True on success, False on failure.
        """
        try:
            self._load_model()
            return True
        except Exception as exc:
            logger.warning("Post-race GGUF warmup failed: %s", exc)
            return False
