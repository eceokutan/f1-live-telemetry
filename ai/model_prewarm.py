"""
Model prewarm utilities for STT/TTS.

These helpers download/cache model artifacts ahead of first use so
voice features feel instant during normal app interaction.
"""

from __future__ import annotations

import time
from pathlib import Path

DEFAULT_KOKORO_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "kokoro_cache"


def _default_hf_hub_cache() -> Path:
    """Return Hugging Face hub cache root."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        return Path(HF_HUB_CACHE)
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "hub"


def _resolve_faster_whisper_repo(model_size: str) -> str:
    """
    Resolve faster-whisper model alias (e.g. 'base') to repo id.

    Falls back to the input string if mapping is unavailable.
    """
    try:
        from faster_whisper.utils import _MODELS
        return _MODELS.get(model_size, model_size)
    except Exception:
        return model_size


def is_faster_whisper_cached(model_size: str = "base") -> bool:
    """Whether a faster-whisper model is already present in local cache."""
    # Local directory path case (already "cached" by user).
    model_path = Path(model_size)
    if model_path.exists():
        return True

    repo_id = _resolve_faster_whisper_repo(model_size)
    if "/" not in repo_id:
        return False

    cache_root = _default_hf_hub_cache()
    repo_cache = cache_root / f"models--{repo_id.replace('/', '--')}"
    snapshots_dir = repo_cache / "snapshots"
    if not snapshots_dir.exists():
        return False

    required_files = ("model.bin", "config.json", "tokenizer.json")
    for snapshot in snapshots_dir.iterdir():
        if not snapshot.is_dir():
            continue
        if all((snapshot / name).exists() for name in required_files):
            return True
    return False


def needs_faster_whisper_prewarm(model_size: str = "base") -> bool:
    """True when faster-whisper prewarm is still needed."""
    return not is_faster_whisper_cached(model_size=model_size)


def is_kokoro_cached(cache_dir: Path | None = None, variant: str = "v1.0") -> bool:
    """Whether Kokoro core model/voice/config cache files are present."""
    cache_path = cache_dir or DEFAULT_KOKORO_CACHE_DIR
    model_file = cache_path / "models" / "huggingface" / variant / "onnx" / "model.onnx"
    voices_archive = cache_path / "voices" / "huggingface" / variant / "voices.bin.npz"
    config_file = cache_path / "config" / variant / "config.json"

    if not (model_file.exists() and voices_archive.exists() and config_file.exists()):
        return False

    # Lightweight sanity checks to avoid treating partial files as warm cache.
    if model_file.stat().st_size < 1_000_000:
        return False
    if voices_archive.stat().st_size < 100_000:
        return False
    if config_file.stat().st_size < 100:
        return False
    return True


def needs_kokoro_prewarm(cache_dir: Path | None = None, variant: str = "v1.0") -> bool:
    """True when Kokoro prewarm is still needed."""
    return not is_kokoro_cached(cache_dir=cache_dir, variant=variant)


def _clear_stale_locks(cache_dir: Path, max_age_seconds: int = 1200) -> None:
    """Remove stale Kokoro lock files from interrupted downloads."""
    now = time.time()
    for lock_file in cache_dir.rglob("*.lock"):
        try:
            age = now - lock_file.stat().st_mtime
            if age > max_age_seconds:
                lock_file.unlink(missing_ok=True)
        except Exception:
            continue


def needs_local_llm_prewarm() -> bool:
    """True when the local LLM singleton is not yet loaded into GPU memory."""
    try:
        from ai.local_llm_inference import LocalLLMInference
        return LocalLLMInference.get_shared_if_loaded() is None
    except Exception:
        return True


def prewarm_local_llm(adapter_path: str = "race_engineer_llm") -> None:
    """Load the local LLM into the shared singleton (downloads base model on first run)."""
    from ai.local_llm_inference import LocalLLMInference
    LocalLLMInference.get_shared(adapter_path=adapter_path)


def prewarm_faster_whisper(model_size: str = "base") -> None:
    """Download/cache faster-whisper model files."""
    from faster_whisper import WhisperModel

    _ = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
    )


def prewarm_kokoro(
    voice_id: str = "bm_lewis",
    lang: str = "en-gb",
    speed: float = 0.97,
    use_cuda: bool = False,
    cache_dir: Path | None = None,
) -> None:
    """
    Download/cache Kokoro model + voices and run one tiny synthesis.

    This forces first-run assets to be prepared before live TTS usage.
    """
    cache_path = cache_dir or DEFAULT_KOKORO_CACHE_DIR
    cache_path.mkdir(parents=True, exist_ok=True)
    _clear_stale_locks(cache_path)

    from pykokoro import GenerationConfig, PipelineConfig, build_pipeline
    from pykokoro import onnx_backend as kokoro_onnx_backend
    from pykokoro import utils as kokoro_utils
    from pykokoro.ssmd_parser import (
        DEFAULT_PAUSE_NONE,
        DEFAULT_PAUSE_WEAK,
        parse_ssmd_to_segments,
    )
    from pykokoro.stages.doc_parsers.ssmd import SsmdDocumentParser
    from pykokoro.stages.protocols import DocumentResult
    from pykokoro.tokenizer import TokenizerConfig

    class NoSpacySsmdDocumentParser(SsmdDocumentParser):
        """Doc parser variant that forces non-spaCy sentence splitting."""

        def parse(self, text: str, cfg, trace):
            generation = cfg.generation
            initial_pause, segments = parse_ssmd_to_segments(
                text,
                lang=generation.lang,
                pause_none=DEFAULT_PAUSE_NONE,
                pause_weak=DEFAULT_PAUSE_WEAK,
                pause_clause=generation.pause_clause,
                pause_sentence=generation.pause_sentence,
                pause_paragraph=generation.pause_paragraph,
                use_spacy=False,
            )
            clean_text, spans, boundaries, doc_segments = self._build_document(
                segments, initial_pause, trace
            )
            if generation.pause_mode == "auto":
                boundaries.extend(self._sentence_boundaries(doc_segments, boundaries))
            return DocumentResult(
                clean_text=clean_text,
                annotation_spans=spans,
                boundary_events=boundaries,
                segments=doc_segments,
            )

    def _project_cache_path(folder: str | None = None) -> Path:
        base = cache_path.resolve()
        base.mkdir(parents=True, exist_ok=True)
        if folder:
            out = base / folder
            out.mkdir(parents=True, exist_ok=True)
            return out
        return base

    # Force pykokoro to use project-local cache path.
    kokoro_utils.get_user_cache_path = _project_cache_path
    kokoro_onnx_backend.get_user_cache_path = _project_cache_path

    provider = "cuda" if use_cuda else "cpu"
    cfg = PipelineConfig(
        voice=voice_id,
        model_source="huggingface",
        provider=provider,
        cache_dir=str(cache_path.resolve()),
        tokenizer_config=TokenizerConfig(
            use_spacy=False,
            spacy_model_size="sm",
        ),
        generation=GenerationConfig(
            lang=lang,
            speed=float(speed),
            pause_mode="tts",
        ),
    )

    pipeline = build_pipeline(
        config=cfg,
        eager=True,
        doc_parser=NoSpacySsmdDocumentParser(),
    )
    try:
        _ = pipeline.run("System ready.")
    finally:
        close_fn = getattr(pipeline, "close", None)
        if callable(close_fn):
            close_fn()
