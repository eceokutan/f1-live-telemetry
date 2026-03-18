"""
Build script for creating the Jarvis Windows executable.

Usage:
    python scripts/build_exe.py

Steps:
    1. Runs PyInstaller with jarvis.spec
    2. Creates a zip archive for distribution

Requires: PyInstaller
    pip install pyinstaller
"""

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPEC_FILE = PROJECT_ROOT / "jarvis.spec"
DIST_DIR = PROJECT_ROOT / "dist"


def step_build():
    """Run PyInstaller."""
    print("[BUILD] Running PyInstaller...")
    subprocess.run(
        [
            sys.executable, "-m", "PyInstaller",
            "--noconfirm",
            str(SPEC_FILE),
        ],
        cwd=str(PROJECT_ROOT),
        check=True,
    )
    print(f"[OK] Build complete: {DIST_DIR / 'Jarvis'}")


def step_zip():
    """Create distributable zip archive."""
    output_dir = DIST_DIR / "Jarvis"
    if not output_dir.exists():
        print("[ERROR] Build output not found")
        return

    zip_path = DIST_DIR / "Jarvis-Windows"
    print(f"[BUILD] Creating zip archive: {zip_path}.zip")
    shutil.make_archive(str(zip_path), "zip", str(DIST_DIR), "Jarvis")
    print(f"[OK] Archive created: {zip_path}.zip")


def main():
    print("=" * 60)
    print("  Jarvis F1 Telemetry Suite - Build Script")
    print("=" * 60)

    step_build()
    step_zip()

    print()
    print("=" * 60)
    print("  Build complete!")
    print(f"  Executable: {DIST_DIR / 'Jarvis' / 'Jarvis.exe'}")
    print(f"  Archive:    {DIST_DIR / 'Jarvis-Windows.zip'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
