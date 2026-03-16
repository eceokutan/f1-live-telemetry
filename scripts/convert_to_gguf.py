#!/usr/bin/env python3
"""
Convert QLoRA adapter + base model into a quantized GGUF file.

Three stages:
1. Merge LoRA adapter into base model
2. Convert merged model to GGUF (f16)
3. Quantize to Q4_K_M

Prerequisites:
    - pip install -r requirements-convert.txt
    - Clone and build llama.cpp (https://github.com/ggerganov/llama.cpp)

Usage:
    python scripts/convert_to_gguf.py --llama-cpp-path /path/to/llama.cpp
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def find_quantize_binary(llama_cpp_path: Path) -> Path:
    """Find the llama-quantize binary in the llama.cpp build tree."""
    candidates = [
        # Visual Studio multi-config builds place binaries under configuration dirs.
        llama_cpp_path / "build" / "bin" / "Release" / "llama-quantize.exe",
        llama_cpp_path / "build" / "bin" / "RelWithDebInfo" / "llama-quantize.exe",
        llama_cpp_path / "build" / "bin" / "Debug" / "llama-quantize.exe",
        llama_cpp_path / "build" / "bin" / "llama-quantize",
        llama_cpp_path / "build" / "bin" / "llama-quantize.exe",
        llama_cpp_path / "llama-quantize",
        llama_cpp_path / "llama-quantize.exe",
        llama_cpp_path / "build" / "llama-quantize",
        llama_cpp_path / "build" / "llama-quantize.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            # On Windows, prefer an executable with nearby runtime DLLs.
            if sys.platform == "win32":
                required_dlls = ["ggml-base.dll", "llama.dll"]
                if not all((candidate.parent / dll).exists() for dll in required_dlls):
                    continue
            return candidate
    return None


def validate_llama_cpp(llama_cpp_path: Path) -> tuple:
    """Validate llama.cpp path contains required tools. Returns (convert_script, quantize_binary)."""
    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print(f"ERROR: convert_hf_to_gguf.py not found at {convert_script}")
        print("Make sure --llama-cpp-path points to the llama.cpp root directory.")
        sys.exit(1)

    quantize_bin = find_quantize_binary(llama_cpp_path)
    if quantize_bin is None:
        print(f"ERROR: llama-quantize binary not found in {llama_cpp_path}")
        print("Build llama.cpp first: cd llama.cpp && cmake -B build && cmake --build build")
        sys.exit(1)

    return convert_script, quantize_bin


def validate_adapter_files(adapter_path: Path) -> None:
    """Validate adapter directory contains the minimum files required for merge."""
    config_file = adapter_path / "adapter_config.json"
    has_weights = any(
        (adapter_path / name).exists()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    )

    if not config_file.exists() or not has_weights:
        print(f"ERROR: Adapter directory is missing required files: {adapter_path}")
        print("Expected:")
        print("  - adapter_config.json")
        print("  - adapter_model.safetensors (or adapter_model.bin)")
        print(
            "Note: large adapter/model artifacts are often kept out of Git. "
            "Provide --adapter-path to a local directory that contains them."
        )
        sys.exit(1)


def merge_lora(base_model_id: str, adapter_path: Path, output_dir: Path) -> None:
    """Merge LoRA adapter into base model and save."""
    print(f"\n{'='*60}")
    print("Stage 1: Merging LoRA adapter into base model")
    print(f"{'='*60}")
    print(f"  Base model: {base_model_id}")
    print(f"  Adapter:    {adapter_path}")
    print(f"  Output:     {output_dir}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    print("  Loading base model...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype="float16",
            low_cpu_mem_usage=True,
        )
    except TypeError:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype="float16",
            low_cpu_mem_usage=True,
        )

    print(f"  Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    print("  Merging and unloading LoRA weights...")
    model = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Saving merged model to {output_dir}...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("  [OK] Merge complete")


def convert_to_gguf(convert_script: Path, merged_dir: Path, output_gguf: Path) -> None:
    """Convert merged HF model to GGUF format (f16)."""
    print(f"\n{'='*60}")
    print("Stage 2: Converting to GGUF (f16)")
    print(f"{'='*60}")
    print(f"  Input:  {merged_dir}")
    print(f"  Output: {output_gguf}")

    output_gguf.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(convert_script),
        str(merged_dir),
        "--outfile", str(output_gguf),
        "--outtype", "f16",
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    if not output_gguf.exists():
        print("  ERROR: GGUF file was not created")
        sys.exit(1)
    print(f"  [OK] GGUF created ({output_gguf.stat().st_size / 1e6:.1f} MB)")


def quantize_gguf(quantize_bin: Path, input_gguf: Path, output_gguf: Path, quantization: str) -> None:
    """Quantize GGUF to target quantization level."""
    print(f"\n{'='*60}")
    print(f"Stage 3: Quantizing to {quantization}")
    print(f"{'='*60}")
    print(f"  Input:  {input_gguf}")
    print(f"  Output: {output_gguf}")

    cmd = [str(quantize_bin), str(input_gguf), str(output_gguf), quantization]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    if not output_gguf.exists():
        print("  ERROR: Quantized GGUF file was not created")
        sys.exit(1)

    print(f"  [OK] Quantized ({output_gguf.stat().st_size / 1e6:.1f} MB)")

    # Clean up intermediate f16 GGUF
    if input_gguf != output_gguf and input_gguf.exists():
        print(f"  Removing intermediate f16 GGUF: {input_gguf}")
        input_gguf.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter and convert to quantized GGUF"
    )
    parser.add_argument(
        "--llama-cpp-path", required=True, type=Path,
        help="Path to llama.cpp root directory (must contain convert_hf_to_gguf.py and built llama-quantize)"
    )
    parser.add_argument(
        "--base-model", default="ibm-granite/granite-4.0-micro",
        help="Hugging Face model ID for base model (default: ibm-granite/granite-4.0-micro)"
    )
    parser.add_argument(
        "--adapter-path", default="race_engineer_llm", type=Path,
        help="Path to QLoRA adapter directory (default: race_engineer_llm)"
    )
    parser.add_argument(
        "--output-dir", default="race_engineer_gguf", type=Path,
        help="Output directory for GGUF files (default: race_engineer_gguf)"
    )
    parser.add_argument(
        "--quantization", default="Q4_K_M",
        help="Quantization level (default: Q4_K_M)"
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    # Resolve relative paths against project root
    adapter_path = args.adapter_path
    if not adapter_path.is_absolute():
        adapter_path = project_root / adapter_path

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    llama_cpp_path = args.llama_cpp_path.resolve()

    # Validate
    if not adapter_path.exists():
        print(f"ERROR: Adapter path does not exist: {adapter_path}")
        sys.exit(1)
    validate_adapter_files(adapter_path)

    convert_script, quantize_bin = validate_llama_cpp(llama_cpp_path)

    # Paths
    merged_dir = project_root / "race_engineer_llm_merged"
    f16_gguf = output_dir / "granite-race-engineer-f16.gguf"
    quant_name = f"granite-race-engineer-{args.quantization}.gguf"
    final_gguf = output_dir / quant_name

    print(f"\nConversion pipeline:")
    print(f"  1. Merge:    {adapter_path} -> {merged_dir}")
    print(f"  2. Convert:  {merged_dir} -> {f16_gguf}")
    print(f"  3. Quantize: {f16_gguf} -> {final_gguf}")

    # Stage 1: Merge
    merge_lora(args.base_model, adapter_path, merged_dir)

    # Stage 2: Convert to GGUF
    convert_to_gguf(convert_script, merged_dir, f16_gguf)

    # Stage 3: Quantize
    quantize_gguf(quantize_bin, f16_gguf, final_gguf, args.quantization)

    # Clean up merged model directory
    print(f"\nCleaning up merged model directory: {merged_dir}")
    shutil.rmtree(merged_dir, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"[OK] Done! GGUF model ready at: {final_gguf}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
