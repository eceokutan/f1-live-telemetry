"""
PyInstaller runtime hook: register native DLL directories and fix SSL
certificates BEFORE any imports.

This runs before main.py, ensuring:
1. onnxruntime/ctranslate2/llama_cpp DLLs are discoverable
2. SSL certificates are found for HTTPS downloads (huggingface_hub)
"""
import os
import sys

_base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))

# Register DLL directories
if hasattr(os, "add_dll_directory"):
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

# Fix SSL certificates for bundled app
_cert_file = os.path.join(_base, "certifi", "cacert.pem")
if os.path.isfile(_cert_file):
    os.environ.setdefault("SSL_CERT_FILE", _cert_file)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _cert_file)

# Disable hf_xet in bundled app — its native .pyd can fail in PyInstaller,
# causing NoneType errors. huggingface_hub falls back to normal HTTP downloads.
if getattr(sys, "frozen", False):
    os.environ["HF_HUB_DISABLE_XET"] = "1"

# Force-import onnxruntime before anything else can interfere
try:
    import onnxruntime
except Exception:
    pass
