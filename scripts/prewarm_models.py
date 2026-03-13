#!/usr/bin/env python3
"""
Pre-download/cache voice models used by the app.

Usage:
    python scripts/prewarm_models.py
    python scripts/prewarm_models.py --skip-stt
    python scripts/prewarm_models.py --skip-tts --stt-model small
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure repository root is on sys.path when running as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.model_prewarm import (
    needs_faster_whisper_prewarm,
    needs_kokoro_prewarm,
    prewarm_faster_whisper,
    prewarm_kokoro,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prewarm STT/TTS model caches")
    parser.add_argument("--skip-stt", action="store_true", help="Skip faster-whisper prewarm")
    parser.add_argument("--skip-tts", action="store_true", help="Skip Kokoro TTS prewarm")
    parser.add_argument("--stt-model", default="base", help="Whisper model size (default: base)")
    parser.add_argument("--tts-voice", default="bm_lewis", help="Kokoro voice id (default: bm_lewis)")
    parser.add_argument("--tts-lang", default="en-gb", help="Kokoro language (default: en-gb)")
    parser.add_argument("--tts-speed", type=float, default=0.97, help="Kokoro speed (default: 0.97)")
    parser.add_argument("--tts-cuda", action="store_true", help="Use CUDA for Kokoro prewarm")
    parser.add_argument("--force", action="store_true", help="Run prewarm even if cache is already warm")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("prewarm")

    errors: list[str] = []

    if not args.skip_stt:
        try:
            if not args.force and not needs_faster_whisper_prewarm(model_size=args.stt_model):
                logger.info("Skipping faster-whisper prewarm (%s already cached)", args.stt_model)
            else:
                logger.info("Prewarming faster-whisper (%s)...", args.stt_model)
                prewarm_faster_whisper(model_size=args.stt_model)
                logger.info("faster-whisper prewarm complete")
        except Exception as e:
            errors.append(f"STT prewarm failed: {e}")

    if not args.skip_tts:
        try:
            if not args.force and not needs_kokoro_prewarm():
                logger.info("Skipping Kokoro TTS prewarm (cache already warm)")
            else:
                logger.info("Prewarming Kokoro TTS (voice=%s)...", args.tts_voice)
                prewarm_kokoro(
                    voice_id=args.tts_voice,
                    lang=args.tts_lang,
                    speed=args.tts_speed,
                    use_cuda=args.tts_cuda,
                )
                logger.info("Kokoro TTS prewarm complete")
        except Exception as e:
            errors.append(f"TTS prewarm failed: {e}")

    if errors:
        for err in errors:
            logger.error(err)
        return 1

    logger.info("All requested model caches are ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
