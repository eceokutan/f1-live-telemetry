"""
Local LLM inference using llama-cpp-python with GGUF models.

Loads a quantized GGUF model (converted from Granite-4.0-micro + QLoRA adapter)
for fast CPU or GPU inference without torch/transformers/peft dependencies.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level singleton so the model stays in memory across restarts
# within the same process (e.g. launcher loop -> run_jarvis_live -> back to launcher).
_shared_instance: Optional["LocalLLMInference"] = None
_shared_lock = threading.Lock()


class LocalLLMInference:
    """
    Loads and runs a GGUF-quantized model via llama-cpp-python.

    Supports both CPU (n_gpu_layers=0) and GPU (n_gpu_layers=-1) inference.
    """

    def __init__(
        self,
        model_path: str = "race_engineer_gguf/granite-race-engineer-Q4_K_M.gguf",
        n_gpu_layers: int = 0,
        max_tokens: int = 48,
        temperature: float = 0.3,
        max_time_seconds: float = 5.0,
        max_prompt_tokens: int = 1024,
    ):
        """
        Initialize local LLM inference.

        Args:
            model_path: Path to GGUF model file (relative to project root)
            n_gpu_layers: Number of layers to offload to GPU (0=CPU, -1=all)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            max_time_seconds: Hard generation time cap per response
            max_prompt_tokens: Maximum input prompt tokens (truncate if longer)

        Raises:
            FileNotFoundError: If GGUF model file does not exist.
        """
        # Resolve model path relative to project root
        self.model_path = Path(model_path)
        if not self.model_path.is_absolute():
            self.model_path = Path(__file__).parent.parent / self.model_path

        if not self.model_path.exists():
            # Try auto-downloading from Hugging Face Hub
            try:
                from ai.model_downloader import ensure_model
                logger.info("Model not found locally, attempting auto-download...")
                downloaded = ensure_model(model_path)
                self.model_path = downloaded
            except Exception as dl_err:
                raise FileNotFoundError(
                    f"GGUF model not found at {self.model_path} and auto-download failed: {dl_err}. "
                    "Run 'python scripts/convert_to_gguf.py' to generate it from the QLoRA adapter, "
                    "or place the model file manually."
                ) from dl_err

        # Env var overrides
        env_model_path = os.getenv("LOCAL_LLM_MODEL_PATH", "").strip()
        if env_model_path:
            self.model_path = Path(env_model_path)

        env_n_gpu_layers = os.getenv("LOCAL_LLM_N_GPU_LAYERS", "").strip()
        if env_n_gpu_layers:
            try:
                n_gpu_layers = int(env_n_gpu_layers)
            except ValueError:
                logger.warning(
                    "Invalid LOCAL_LLM_N_GPU_LAYERS=%s, using %d",
                    env_n_gpu_layers,
                    n_gpu_layers,
                )

        env_n_threads = os.getenv("LOCAL_LLM_N_THREADS", "").strip()
        if env_n_threads:
            try:
                self.n_threads = max(1, int(env_n_threads))
            except ValueError:
                self.n_threads = max(1, (os.cpu_count() or 2) // 2)
        else:
            self.n_threads = max(1, (os.cpu_count() or 2) // 2)

        self.n_gpu_layers = n_gpu_layers
        self.max_tokens = max_tokens
        self.temperature = temperature

        env_max_time = os.getenv("LOCAL_LLM_MAX_TIME_SECONDS", "").strip()
        if env_max_time:
            try:
                max_time_seconds = float(env_max_time)
            except ValueError:
                logger.warning(
                    "Invalid LOCAL_LLM_MAX_TIME_SECONDS=%s, using %.2f",
                    env_max_time,
                    max_time_seconds,
                )
        self.max_time_seconds = max(0.0, float(max_time_seconds))

        env_max_prompt_tokens = os.getenv("LOCAL_LLM_MAX_PROMPT_TOKENS", "").strip()
        if env_max_prompt_tokens:
            try:
                max_prompt_tokens = int(env_max_prompt_tokens)
            except ValueError:
                logger.warning(
                    "Invalid LOCAL_LLM_MAX_PROMPT_TOKENS=%s, using %d",
                    env_max_prompt_tokens,
                    max_prompt_tokens,
                )
        self.max_prompt_tokens = max(32, int(max_prompt_tokens))

        logger.info(
            "GGUF model: %s (n_gpu_layers=%d, n_threads=%d)",
            self.model_path.name,
            self.n_gpu_layers,
            self.n_threads,
        )

        # Lazy-loaded model
        self._model = None
        self._loaded = False

    def load(self) -> None:
        """Load the GGUF model into memory."""
        if self._loaded:
            return

        try:
            from llama_cpp import Llama

            logger.info(f"Loading GGUF model from {self.model_path}...")
            self._model = Llama(
                model_path=str(self.model_path),
                n_ctx=2048,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                n_batch=512,
                verbose=False,
            )

            self._prewarm_generation()
            self._loaded = True
            logger.info("Model loaded successfully")

        except ImportError as e:
            logger.error(f"Missing required library: {e}")
            raise RuntimeError(
                f"Cannot load local model: {e}. Install with: pip install llama-cpp-python"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise RuntimeError(f"Failed to load local model: {e}") from e

    def _format_with_chat_template(self, prompt: str) -> str:
        """
        Wrap a raw prompt in Granite chat template role markers.

        The Granite 4.0 model expects prompts delimited by role tokens.
        Without these markers, the model sees unstructured text and
        hallucinates data continuations instead of generating responses.
        """
        system_msg = (
            "You are an expert F1 race engineer communicating with your "
            "driver over team radio. Use ONLY numbers from the provided data. "
            "Do NOT derive, calculate, or invent new values (percentages, rates, totals). "
            "If key data is missing, say so. Reply in one short sentence."
        )
        return (
            f"<|start_of_role|>system<|end_of_role|>{system_msg}<|end_of_text|>"
            f"<|start_of_role|>user<|end_of_role|>{prompt}<|end_of_text|>"
            f"<|start_of_role|>assistant<|end_of_role|>"
        )

    def generate(self, prompt: str) -> str:
        """
        Generate a response for the given prompt.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            # Wrap prompt in Granite chat template before tokenization
            formatted_prompt = self._format_with_chat_template(prompt)

            # Truncate prompt if it exceeds max_prompt_tokens
            tokens = self._model.tokenize(formatted_prompt.encode())
            if len(tokens) > self.max_prompt_tokens:
                tokens = tokens[:self.max_prompt_tokens]
                formatted_prompt = self._model.detokenize(tokens).decode(errors="replace")

            logger.debug("Formatted prompt (first 200 chars): %s", formatted_prompt[:200])

            response = self._model.create_completion(
                formatted_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_k=50,
                top_p=0.95,
                stop=["<|end_of_text|>", "\n\n", "<|start_of_role|>"],
            )

            return response["choices"][0]["text"].strip()

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}") from e

    def _prewarm_generation(self) -> None:
        """Run a representative generation to absorb first-inference overhead."""
        try:
            warmup_prompt = self._format_with_chat_template(
                "Radio check. Fuel and tires are stable. Reply in one short sentence."
            )
            self._model.create_completion(
                warmup_prompt,
                max_tokens=min(16, self.max_tokens),
                temperature=0.0,
                stop=["<|end_of_text|>", "\n\n", "<|start_of_role|>"],
            )
            logger.info("Local LLM generation warmup complete")
        except Exception as e:
            logger.warning("Local LLM generation warmup failed: %s", e)

    def generate_stream(self, prompt: str):
        """
        Yield token chunks as they are generated.

        Same prompt formatting, tokenization, truncation, and generation
        params as generate(), but with stream=True so tokens arrive
        incrementally.

        Yields:
            str: Token text chunks as they are produced.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            formatted_prompt = self._format_with_chat_template(prompt)

            tokens = self._model.tokenize(formatted_prompt.encode())
            if len(tokens) > self.max_prompt_tokens:
                tokens = tokens[:self.max_prompt_tokens]
                formatted_prompt = self._model.detokenize(tokens).decode(errors="replace")

            logger.debug("Streaming formatted prompt (first 200 chars): %s", formatted_prompt[:200])

            for chunk in self._model.create_completion(
                formatted_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_k=50,
                top_p=0.95,
                stop=["<|end_of_text|>", "\n\n", "<|start_of_role|>"],
                stream=True,
            ):
                text = chunk["choices"][0]["text"]
                if text:
                    yield text

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise RuntimeError(f"Streaming generation failed: {e}") from e

    def __call__(self, prompt: str) -> str:
        """Allow calling the inference object directly."""
        return self.generate(prompt)

    # ------------------------------------------------------------------
    # Singleton helpers
    # ------------------------------------------------------------------

    @classmethod
    def get_shared(
        cls,
        model_path: str = "race_engineer_gguf/granite-race-engineer-Q4_K_M.gguf",
        n_gpu_layers: int = 0,
        max_tokens: int = 48,
        temperature: float = 0.3,
        max_time_seconds: float = 5.0,
        max_prompt_tokens: int = 1024,
    ) -> "LocalLLMInference":
        """
        Return the shared singleton instance, creating and loading it on first call.

        Thread-safe: if the background prewarm thread is already loading the
        model, a second caller (e.g. the AI worker thread) will block on the
        lock and then receive the already-loaded instance instead of loading
        a duplicate.
        """
        global _shared_instance

        # Fast path -- no lock needed if already loaded.
        if _shared_instance is not None and _shared_instance._loaded:
            return _shared_instance

        with _shared_lock:
            # Re-check after acquiring the lock (another thread may have finished).
            if _shared_instance is not None and _shared_instance._loaded:
                return _shared_instance

            logger.info("Loading shared local LLM instance...")
            instance = cls(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                max_tokens=max_tokens,
                temperature=temperature,
                max_time_seconds=max_time_seconds,
                max_prompt_tokens=max_prompt_tokens,
            )
            instance.load()
            _shared_instance = instance
            return instance

    @classmethod
    def get_shared_if_loaded(cls) -> Optional["LocalLLMInference"]:
        """Return the shared instance only if it is already loaded, else None."""
        if _shared_instance is not None and _shared_instance._loaded:
            return _shared_instance
        return None
