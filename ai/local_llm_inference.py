"""
Local LLM inference using transformers + PEFT for QLoRA-finetuned models.

Loads Granite-4.0-micro base model from Hugging Face Hub and applies
the local QLoRA adapter from granite_f1_finetuned_live/.
"""

import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class LocalLLMInference:
    """
    Loads and runs a QLoRA-finetuned model locally.

    Supports GPU (with automatic fallback to CPU) and pre-loads on initialization.
    """

    def __init__(
        self,
        base_model_id: str = "ibm-granite/granite-4.0-micro",
        adapter_path: str = "granite_f1_finetuned_live",
        max_tokens: int = 75,
        temperature: float = 0.7,
        use_gpu: bool = True,
    ):
        """
        Initialize local LLM inference.

        Args:
            base_model_id: Hugging Face model ID for base model
            adapter_path: Path to QLoRA adapter directory (relative to project root)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_gpu: Try to use GPU if available, else CPU
        """
        self.base_model_id = base_model_id
        self.adapter_path = Path(adapter_path)
        if not self.adapter_path.is_absolute():
            # Make relative to project root
            self.adapter_path = Path(__file__).parent.parent / self.adapter_path

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_gpu = use_gpu

        # Determine device
        self.device = self._select_device()

        # Lazy-loaded model and tokenizer
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def _select_device(self) -> str:
        """Select GPU or CPU based on availability and user preference."""
        if not self.use_gpu:
            logger.info("GPU disabled, using CPU")
            return "cpu"

        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            return device
        elif torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon
            logger.info("MPS (Apple Silicon) available")
            return device
        else:
            logger.info("No GPU found, falling back to CPU")
            return "cpu"

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
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                torch_dtype=torch.float32,
                device_map=self.device if self.device != "cpu" else None,
            )

            logger.info(f"Loading QLoRA adapter from {self.adapter_path}...")
            model = PeftModel.from_pretrained(base_model, str(self.adapter_path))
            model = model.to(self.device)

            self._tokenizer = tokenizer
            self._model = model
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
            # Encode prompt
            inputs = self._tokenizer.encode(prompt, return_tensors="pt").to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            # Decode response (excluding the input prompt)
            response = self._tokenizer.decode(
                outputs[0][inputs.shape[-1]:],
                skip_special_tokens=True,
            )

            return response.strip()

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}") from e

    def __call__(self, prompt: str) -> str:
        """Allow calling the inference object directly."""
        return self.generate(prompt)
