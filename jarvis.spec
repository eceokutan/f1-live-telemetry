# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Jarvis F1 Telemetry Suite.

Build with:
    pyinstaller jarvis.spec

Produces: dist/Jarvis/Jarvis.exe (one-folder mode for faster startup)
"""

import os
import sys
from pathlib import Path

block_cipher = None

# Project root
ROOT = os.path.dirname(os.path.abspath(SPEC))

# Icon path (generate with: python scripts/convert_icon.py)
icon_path = os.path.join(ROOT, 'ui', 'img', 'jarvis.ico')
if not os.path.exists(icon_path):
    icon_path = None  # Build without icon if not converted yet

a = Analysis(
    [os.path.join(ROOT, 'main.py')],
    pathex=[ROOT],
    binaries=[],
    datas=[
        # UI assets
        (os.path.join(ROOT, 'ui', 'fonts'), 'ui/fonts'),
        (os.path.join(ROOT, 'ui', 'img'), 'ui/img'),
        # SSL certificates for HTTPS downloads (huggingface_hub)
        (os.path.join(
            ROOT, 'venv', 'lib', 'site-packages', 'certifi', 'cacert.pem'
        ), 'certifi'),
    ],
    hiddenimports=[
        # PyQt5
        'PyQt5.QtWidgets',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtSvg',
        # Matplotlib
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.collections',
        'matplotlib.colors',
        'matplotlib.cm',
        # AI modules
        'ai.race_engineer',
        'ai.race_engineer_core',
        'ai.voice_input',
        'ai.tts_output',
        'ai.ptt_controller',
        'ai.local_llm_inference',
        'ai.model_downloader',
        'ai.model_prewarm',
        'ai.fuel_lookup',
        # Post-race analysis
        'analysis.ai_pipeline_bridge',
        'jarvis_post.llm.local_client',
        'jarvis_post.llm.client',
        # Data (lazy imports in __init__.py)
        'data.session_recorder',
        'data.session_exporter',
        'data.telemetry_loader',
        'data.lap',
        'data.session',
        'data.models',
        # Telemetry backends
        'telemetry.backends.ac_backend',
        'telemetry.lap_buffer',
        # LLM and model inference
        'llama_cpp',
        'huggingface_hub',
        # Voice and TTS
        'faster_whisper',
        'pykokoro',
        'pykokoro.onnx_backend',
        'pykokoro.ssmd_parser',
        'pykokoro.stages.doc_parsers.ssmd',
        'pykokoro.stages.protocols',
        'pykokoro.tokenizer',
        'pykokoro.utils',
        # Input control
        'pynput',
        'pygame',
        # Native runtimes (pre-loaded before PyQt5)
        'onnxruntime',
        'ctranslate2',
        # HTTP / networking
        'httpx',
        'requests',
        'urllib3',
        'certifi',
        'charset_normalizer',
        # Standard deps PyInstaller may miss
        'pydantic',
        'pydantic.deprecated.decorator',
        'dotenv',
        'numpy',
        'pandas',
        'ctypes',
        'sqlite3',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy packages not needed at runtime
        'torch',
        'transformers',
        'peft',
        'tensorflow',
        'keras',
        'spacy',
        'langchain',
        'langchain_core',
        'langgraph',
        'ibm_watsonx_ai',
        'ibm_botocore',
        # Exclude test frameworks
        'pytest',
        '_pytest',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Jarvis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No terminal window — progress shown in LoadingScreen
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_path,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Jarvis',
)
