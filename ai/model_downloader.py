"""
Auto-download GGUF models from Hugging Face Hub on first run.

Downloads the quantized race engineer and post-race analyst models
if they are not found locally. Caches files for future launches.
Shows a Qt progress dialog so the user knows what's happening.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default HF repo and filenames
HF_REPO_ID = "okutanece/f1-race-engineer-gguf"

# Live race engineer model
HF_FILENAME = "granite-race-engineer-Q4_K_M.gguf"
DEFAULT_LOCAL_DIR = "race_engineer_gguf"
DEFAULT_LOCAL_PATH = f"{DEFAULT_LOCAL_DIR}/{HF_FILENAME}"

# Post-race analyst model
POSTRACE_HF_FILENAME = "granite-postrace-analyst-Q4_K_M.gguf"
POSTRACE_LOCAL_DIR = "postrace_gguf"
POSTRACE_LOCAL_PATH = f"{POSTRACE_LOCAL_DIR}/{POSTRACE_HF_FILENAME}"


def _get_project_root() -> Path:
    """Return the project root directory."""
    # When running from PyInstaller bundle, use the exe's directory
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


def get_model_path(local_path: str = DEFAULT_LOCAL_PATH) -> Path:
    """Resolve the model path relative to project root."""
    p = Path(local_path)
    if not p.is_absolute():
        p = _get_project_root() / p
    return p


def is_model_available(local_path: str = DEFAULT_LOCAL_PATH) -> bool:
    """Check if the GGUF model file exists locally."""
    return get_model_path(local_path).exists()


def _show_download_dialog(model_name: str, repo_id: str, filename: str, dest: Path) -> Path:
    """
    Download the model silently.

    The startup loading screen (ui/startup_loader.py stage 6) now owns the
    user-facing download progress, so a separate popup dialog is no longer
    needed.  All late callers (post-race, fallback paths) just download
    in the background without blocking the UI.
    """
    return _download_silent(repo_id, filename, dest)


def _download_silent(repo_id: str, filename: str, dest: Path) -> Path:
    """Download without UI (used as fallback or from non-GUI contexts)."""
    from huggingface_hub import hf_hub_download

    cached_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(dest.parent),
    )

    result = Path(cached_path)
    if result.exists():
        return result
    if dest.exists():
        return dest

    raise RuntimeError(f"Download completed but file not found at {dest}")


def download_model(
    local_path: str = DEFAULT_LOCAL_PATH,
    repo_id: str = HF_REPO_ID,
    filename: str = HF_FILENAME,
    model_name: str = "AI Race Engineer model",
) -> Path:
    """
    Download the GGUF model from Hugging Face Hub.

    Shows a progress dialog if a Qt application is running.

    Args:
        local_path: Local path to save the model (relative to project root)
        repo_id: Hugging Face repository ID
        filename: Filename within the HF repo
        model_name: Human-readable name for the progress dialog

    Returns:
        Path to the downloaded model file

    Raises:
        RuntimeError: If download fails
    """
    dest = get_model_path(local_path)

    if dest.exists():
        logger.info("Model already exists at %s", dest)
        return dest

    # Create directory if needed
    dest.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s from %s/%s ...", model_name, repo_id, filename)

    try:
        result = _show_download_dialog(model_name, repo_id, filename, dest)
        logger.info("Model downloaded to %s", result)
        return result

    except ImportError:
        raise RuntimeError(
            "huggingface-hub is required for auto-download. "
            "Install with: pip install huggingface-hub"
        )
    except Exception as e:
        # Clean up partial download
        if dest.exists() and dest.stat().st_size == 0:
            dest.unlink()
        raise RuntimeError(f"Failed to download model: {e}") from e


def ensure_model(
    local_path: str = DEFAULT_LOCAL_PATH,
    repo_id: str = HF_REPO_ID,
    filename: str = HF_FILENAME,
) -> Path:
    """
    Ensure the GGUF model is available locally, downloading if needed.

    This is the main entry point for the live race engineer model.

    Returns:
        Path to the model file
    """
    if is_model_available(local_path):
        return get_model_path(local_path)

    return download_model(local_path, repo_id, filename, model_name="AI Race Engineer model")


def ensure_postrace_model(
    local_path: str = POSTRACE_LOCAL_PATH,
    repo_id: str = HF_REPO_ID,
    filename: str = POSTRACE_HF_FILENAME,
) -> Path:
    """
    Ensure the post-race analyst GGUF model is available locally.

    Returns:
        Path to the model file
    """
    if is_model_available(local_path):
        return get_model_path(local_path)

    return download_model(local_path, repo_id, filename, model_name="Post-Race Analyst model")
