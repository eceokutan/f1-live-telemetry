#!/usr/bin/env python3
"""
Quick Hugging Face model connectivity test for this project.

Usage:
    python scripts/test_huggingface_model.py
    python scripts/test_huggingface_model.py --prompt "Give me one race tip."
    python scripts/test_huggingface_model.py --model-id ibm/granite-3-8b-instruct
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


def _parse_bool(value: str) -> bool:
    """Parse common truthy string values."""
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test a Hugging Face model call using the project's LLM client.",
    )
    parser.add_argument(
        "--prompt",
        default="You are a race engineer. Give one short setup tip for next lap.",
        help="Prompt to send to the model.",
    )
    parser.add_argument(
        "--model-id",
        default="",
        help="Override HUGGINGFACE_MODEL_ID from environment.",
    )
    parser.add_argument(
        "--endpoint-url",
        default="",
        help="Override HUGGINGFACE_ENDPOINT_URL from environment.",
    )
    parser.add_argument(
        "--space-url",
        default="",
        help="Override HUGGINGFACE_SPACE_URL from environment.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=75,
        help="Maximum new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    return parser.parse_args()


def main() -> int:
    # Ensure imports work when running this file directly from scripts/
    repo_root = Path(__file__).resolve().parents[1]
    eima_root = repo_root / "eima_ai"

    if str(eima_root) not in sys.path:
        sys.path.insert(0, str(eima_root))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    load_dotenv(repo_root / ".env")
    args = _build_args()

    token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_API_KEY", "")
    model_id = args.model_id or os.getenv("HUGGINGFACE_MODEL_ID", "")
    endpoint_url = args.endpoint_url or os.getenv("HUGGINGFACE_ENDPOINT_URL", "")
    space_url = args.space_url or os.getenv("HUGGINGFACE_SPACE_URL", "")
    space_skip_ssl = _parse_bool(os.getenv("HUGGINGFACE_SPACE_SKIP_SSL_VERIFY", ""))

    if not token:
        print("ERROR: Missing HUGGINGFACE_TOKEN (or HUGGINGFACE_API_KEY) in .env.")
        return 1

    if not model_id and not endpoint_url and not space_url:
        print("ERROR: Set one of HUGGINGFACE_MODEL_ID, HUGGINGFACE_ENDPOINT_URL, or HUGGINGFACE_SPACE_URL.")
        return 1

    try:
        from huggingface_hub import InferenceClient  # noqa: F401
    except ImportError:
        print("ERROR: huggingface-hub is not installed. Run: pip install huggingface-hub")
        return 1

    from jarvis_granite.llm.llm_client import LLMClient, LLMError

    client = LLMClient(
        huggingface_token=token,
        model_id=model_id or "unused/model-id",
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_retries=2,
        endpoint_url=endpoint_url or None,
        space_url=space_url or None,
        space_skip_ssl_verify=space_skip_ssl,
    )

    print("Hugging Face test starting...")
    print(f"Model ID: {model_id or '(not set)'}")
    print(f"Endpoint URL: {endpoint_url or '(not set)'}")
    print(f"Space URL: {space_url or '(not set)'}")
    print(f"Prompt: {args.prompt}")

    start = time.perf_counter()
    try:
        response = asyncio.run(client.invoke(args.prompt))
    except LLMError as exc:
        print(f"ERROR: Model call failed: {exc}")
        return 1
    except Exception as exc:  # Defensive: unexpected runtime error
        print(f"ERROR: Unexpected failure: {exc}")
        return 1
    elapsed = time.perf_counter() - start

    print("\nResponse:")
    print(response.strip() if response else "(empty response)")
    print(f"\nLatency: {elapsed:.2f}s")
    print("Success: Hugging Face request completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
