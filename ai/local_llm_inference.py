"""
Local LLM inference using transformers + PEFT for QLoRA-finetuned models.

Loads Granite-4.0-micro base model from Hugging Face Hub and applies
the local QLoRA adapter from race_engineer_llm/.
"""

import logging
import os
import threading
from pathlib import Path
from typing import ClassVar, Optional

import torch

logger = logging.getLogger(__name__)

# Module-level singleton so the model stays in GPU memory across restarts
# within the same process (e.g. launcher loop → run_jarvis_live → back to launcher).
_shared_instance: Optional["LocalLLMInference"] = None
_shared_lock = threading.Lock()


class LocalLLMInference:
    """
    Loads and runs a QLoRA-finetuned model locally.

    Requires CUDA (NVIDIA GPU). Raises RuntimeError if CUDA is unavailable.
    """

    def __init__(
        self,
        base_model_id: str = "ibm-granite/granite-4.0-micro",
        adapter_path: str = "race_engineer_llm",
        max_tokens: int = 24,
        temperature: float = 0.3,
        max_time_seconds: float = 5.0,
        max_prompt_tokens: int = 256,
    ):
        """
        Initialize local LLM inference.

        Args:
            base_model_id: Hugging Face model ID for base model
            adapter_path: Path to QLoRA adapter directory (relative to project root)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            max_time_seconds: Hard generation time cap per response
            max_prompt_tokens: Maximum input prompt tokens (truncate if longer)

        Raises:
            RuntimeError: If CUDA is not available.
        """
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for local LLM inference but is not available. "
                "Install a CUDA-enabled PyTorch build or use rule-based fallback."
            )

        self.base_model_id = base_model_id
        self.adapter_path = Path(adapter_path)
        if not self.adapter_path.is_absolute():
            # Make relative to project root
            self.adapter_path = Path(__file__).parent.parent / self.adapter_path

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

        self.device = "cuda"
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))

        try:
            if torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
        except Exception:
            self.dtype = torch.float16

        # Lazy-loaded model and tokenizer
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def load(self) -> None:
        """
        Pre-load model and tokenizer.

        This is called during initialization by default.
        """
        if self._loaded:
            return

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel

            logger.info(f"Loading base model {self.base_model_id}...")
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
            try:
                # Newer transformers prefers `dtype`; keep a compatibility fallback.
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_id,
                    dtype=self.dtype,
                    low_cpu_mem_usage=True,
                )
            except TypeError:
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_id,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                )

            if self.device != "cpu":
                base_model = base_model.to(self.device)

            logger.info(f"Loading QLoRA adapter from {self.adapter_path}...")
            model = PeftModel.from_pretrained(base_model, str(self.adapter_path))
            model = model.to(self.device)

            self._tokenizer = tokenizer
            self._model = model

            # Prime first-token latency so initial live query is responsive.
            self._prewarm_generation()

            self._loaded = True
            logger.info("Model loaded successfully")

        except ImportError as e:
            logger.error(f"Missing required library: {e}")
            raise RuntimeError(f"Cannot load local model: {e}") from e
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise RuntimeError(f"Failed to load local model: {e}") from e

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
            # Use truncation to cap prompt prefill latency for live response budgets.
            encoded = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_prompt_tokens,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # Generate
            generation_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": self.max_tokens,
                "temperature": self.temperature,
                "do_sample": self.temperature > 0.0,
                "top_k": 50,
                "top_p": 0.95,
                "pad_token_id": self._tokenizer.eos_token_id,
            }
            if self.max_time_seconds > 0:
                generation_kwargs["max_time"] = self.max_time_seconds

            with torch.no_grad():
                outputs = self._model.generate(**generation_kwargs)

            # Decode response (excluding the input prompt)
            response = self._tokenizer.decode(
                outputs[0][input_ids.shape[-1]:],
                skip_special_tokens=True,
            )

            return response.strip()

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}") from e

    def _prewarm_generation(self) -> None:
        """Run a tiny generation to absorb first-token initialization overhead."""
        if os.getenv("LOCAL_LLM_PREWARM_GENERATE", "0").strip().lower() not in {"1", "true", "yes", "on"}:
            return

        try:
            encoded = self._tokenizer(
                "Radio check.",
                return_tensors="pt",
                truncation=True,
                max_length=64,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            with torch.no_grad():
                self._model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=2,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                    max_time=max(self.max_time_seconds, 3.0),
                )
            logger.info("Local LLM generation warmup complete")
        except Exception as e:
            logger.warning("Local LLM generation warmup failed: %s", e)

    def __call__(self, prompt: str) -> str:
        """Allow calling the inference object directly."""
        return self.generate(prompt)

    # ------------------------------------------------------------------
    # Singleton helpers
    # ------------------------------------------------------------------

    @classmethod
    def get_shared(
        cls,
        base_model_id: str = "ibm-granite/granite-4.0-micro",
        adapter_path: str = "race_engineer_llm",
        max_tokens: int = 24,
        temperature: float = 0.3,
        max_time_seconds: float = 5.0,
        max_prompt_tokens: int = 256,
    ) -> "LocalLLMInference":
        """
        Return the shared singleton instance, creating and loading it on first call.

        Thread-safe: if the background prewarm thread is already loading the
        model, a second caller (e.g. the AI worker thread) will block on the
        lock and then receive the already-loaded instance instead of loading
        a duplicate.

        Raises RuntimeError if CUDA is not available.
        """
        global _shared_instance

        # Fast path — no lock needed if already loaded.
        if _shared_instance is not None and _shared_instance._loaded:
            return _shared_instance

        with _shared_lock:
            # Re-check after acquiring the lock (another thread may have finished).
            if _shared_instance is not None and _shared_instance._loaded:
                return _shared_instance

            logger.info("Loading shared local LLM instance...")
            instance = cls(
                base_model_id=base_model_id,
                adapter_path=adapter_path,
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
