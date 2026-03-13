#!/usr/bin/env python3
"""
Test script for local Granite-4.0-micro QLoRA model.

Tests model loading, inference, and performance.
"""

import sys
import time
import os
from pathlib import Path

# Add project root to path (tests/../ = project root)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

os.environ.setdefault("USE_LOCAL_LLM", "true")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


LOCAL_BASE_MODEL_ID = os.getenv("LOCAL_BASE_MODEL_ID", "ibm-granite/granite-4.0-micro")
LOCAL_ADAPTER_PATH = os.getenv("LOCAL_ADAPTER_PATH", "race_engineer_llm")
LOCAL_USE_GPU = _env_bool("LOCAL_LLM_USE_GPU", True)
LOCAL_REQUIRE_CUDA = _env_bool("LOCAL_REQUIRE_CUDA", False)
LOCAL_MAX_TOKENS = int(os.getenv("LOCAL_MAX_TOKENS", "24"))
LOCAL_NUM_PROMPTS = max(1, int(os.getenv("LOCAL_NUM_PROMPTS", "1")))


def test_local_llm_inference():
    """Test local LLM loading and inference."""
    print("=" * 80)
    print("Testing Local Granite-4.0-micro + QLoRA Inference")
    print("=" * 80)

    # Test 1: Import and initialization
    print("\n[1/4] Importing LocalLLMInference...")
    try:
        from ai.local_llm_inference import LocalLLMInference
        print("✓ Import successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("  Required: torch, transformers, peft")
        return False

    # Test 2: Initialize inference engine
    print("\n[2/4] Initializing local LLM inference engine...")
    print(f"  base_model_id = {LOCAL_BASE_MODEL_ID}")
    print(f"  adapter_path  = {LOCAL_ADAPTER_PATH}")
    print(f"  use_gpu       = {LOCAL_USE_GPU}")
    print(f"  max_tokens    = {LOCAL_MAX_TOKENS}")
    print(f"  num_prompts   = {LOCAL_NUM_PROMPTS}")
    try:
        llm = LocalLLMInference(
            base_model_id=LOCAL_BASE_MODEL_ID,
            adapter_path=LOCAL_ADAPTER_PATH,
            max_tokens=LOCAL_MAX_TOKENS,
            temperature=0.7,
            use_gpu=LOCAL_USE_GPU,
        )
        print(f"✓ Initialized (device: {llm.device})")
        if LOCAL_REQUIRE_CUDA and llm.device != "cuda":
            print("✗ CUDA required but not selected")
            return False
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False

    # Test 3: Load model
    print("\n[3/4] Pre-loading model and QLoRA adapter...")
    start = time.time()
    try:
        llm.load()
        load_time = time.time() - start
        print(f"✓ Model loaded in {load_time:.2f}s")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

    # Test 4: Generate responses using actual race engineer prompt format
    print("\n[4/4] Testing inference (actual race engineer prompts)...")

    # Real session context (from context.to_prompt_context())
    session_context = """Track: Monza
Lap: 8 | Position: P2
Speed: 285 km/h | Gear: 5 | RPM: 12500
Throttle: 87% | Brake: 0% | Steering: 0.05
G-Forces: Lat 1.8g | Lon 0.2g
Gap Ahead: 0.80s | Gap Behind: 1.20s
Nearby Opponents: P1(car 5, 288 km/h), P3(car 2, 282 km/h)
Fuel: 3.2L (8.0 laps)
Tire Temps: FL:105°C FR:103°C RL:98°C RR:101°C
Tire Wear: FL:65% FR:62% RL:68% RR:64%
Wheel Slip: FL:0.15 FR:0.12 RL:0.08 RR:0.09
Suspension: FL:0.045m FR:0.047m RL:0.042m RR:0.043m
Ride Height: Front 0.135m | Rear 0.142m
Car Damage: No damage
Best Lap: 1:35.234 | Last Lap: 1:35.567"""

    # Real prompts formatted with actual templates
    test_prompts = [
        # Reactive prompt (moderate): "How's my fuel?"
        f"""F1 race engineer on radio. Answer the driver's question in one short sentence using the data below.

Question: "How's my fuel?"

Data:
{session_context}

Example answers: "Fuel for 8 more laps." / "Fuel looks good, manage and push."

Answer:""",

        # Proactive prompt (moderate): tire temperature warning
        f"""F1 race engineer on radio. Alert the driver in one short sentence.

Event: tire_temperature_warning
Details: Tire Temperature Current Value: 105.0, Tire Temperature Warning Threshold: 100.0

Data:
{session_context}

Example alerts: "Front left is overheating, ease off." / "Tire temps rising, manage pace."

Alert:""",

        # Reactive prompt (moderate): "Can I push harder?"
        f"""F1 race engineer on radio. Answer the driver's question in one short sentence using the data below.

Question: "Can I push harder?"

Data:
{session_context}

Example answers: "Tires at 75 degrees, looking good." / "Manage pace, tires degrading."

Answer:""",

        # Proactive prompt (moderate): gap change warning
        f"""F1 race engineer on radio. Alert the driver in one short sentence.

Event: gap_change_detected
Details: Gap Change: -0.4, Gap Ahead: 0.8, Gap Behind: 1.2

Data:
{session_context}

Example alerts: "Gap closing! Focus up front." / "You're pulling away from P3, extend the lead."

Alert:""",
    ]

    selected_prompts = test_prompts[:LOCAL_NUM_PROMPTS]

    for i, prompt in enumerate(selected_prompts, 1):
        print(f"\n  [{i}/{len(selected_prompts)}] Testing race engineer prompt...")
        start = time.time()
        try:
            response = llm.generate(prompt)
            inference_time = time.time() - start
            print(f"  → {response}")
            print(f"  ⏱ {inference_time:.2f}s")
        except Exception as e:
            print(f"  ✗ Generation failed: {e}")
            return False

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
    return True


def test_llm_client():
    """Test LLMClient with local LLM enabled."""
    print("\n" + "=" * 80)
    print("Testing LLMClient with Local LLM")
    print("=" * 80)

    print("\n[1/2] Importing LLMClient...")
    try:
        from ai.race_engineer_core import LLMClient
        print("✓ Import successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

    print("\n[2/2] Initializing LLMClient with local LLM enabled...")
    print(f"  local_adapter_path = {LOCAL_ADAPTER_PATH}")
    try:
        client = LLMClient(
            huggingface_token="dummy_token",
            model_id="dummy_model",
            use_local_llm=True,
            local_adapter_path=LOCAL_ADAPTER_PATH,
        )
        print("✓ LLMClient initialized")
        if client._local_llm is not None:
            print("✓ Local LLM pre-loaded successfully")
        else:
            print("⚠ Local LLM not loaded")
    except Exception as e:
        print(f"✗ LLMClient initialization failed: {e}")
        return False

    print("\n" + "=" * 80)
    print("✓ LLMClient test passed!")
    print("=" * 80)
    return True


def check_dependencies():
    """Check required dependencies."""
    print("=" * 80)
    print("Checking Dependencies")
    print("=" * 80)

    deps = {
        "torch": "torch",
        "transformers": "transformers",
        "peft": "peft",
        "pydantic": "pydantic",
    }

    all_ok = True
    for name, module in deps.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - MISSING")
            all_ok = False

    if not all_ok:
        print("\n⚠ Missing dependencies. Install with:")
        print("  pip install -r requirements.txt")

    try:
        import torch
        print("\nCUDA check:")
        print(f"  torch version: {torch.__version__}")
        print(f"  torch cuda: {torch.version.cuda}")
        print(f"  cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  cuda device: {torch.cuda.get_device_name(0)}")
        if LOCAL_REQUIRE_CUDA and not torch.cuda.is_available():
            print("✗ LOCAL_REQUIRE_CUDA=true but CUDA is unavailable")
            return False
    except Exception as e:
        print(f"\n⚠ Could not run CUDA diagnostics: {e}")

    print()
    return all_ok


if __name__ == "__main__":
    # Check dependencies first
    if not check_dependencies():
        print("Cannot proceed without installing dependencies.")
        sys.exit(1)

    # Test local inference
    if not test_local_llm_inference():
        sys.exit(1)

    # Test LLMClient integration
    if not test_llm_client():
        sys.exit(1)

    print("\n" + "=" * 80)
    print("🎉 All tests passed! Local deployment is ready.")
    print("=" * 80)
    print("\nRun the app with:")
    print("  python main.py --ai")
