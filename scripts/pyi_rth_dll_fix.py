"""
PyInstaller runtime hook: register native DLL directories BEFORE any imports.

This runs before main.py, ensuring onnxruntime/ctranslate2/llama_cpp DLLs
are discoverable even after PyQt5 changes the DLL search path.
"""
import os
import sys

if hasattr(os, "add_dll_directory"):
    _base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    for _subdir in (
        os.path.join(_base, "onnxruntime", "capi"),
        os.path.join(_base, "ctranslate2"),
        os.path.join(_base, "llama_cpp", "lib"),
        _base,
    ):
        if os.path.isdir(_subdir):
            try:
                os.add_dll_directory(_subdir)
            except OSError:
                pass

# Force-import onnxruntime before anything else can interfere
try:
    import onnxruntime
except Exception:
    pass
