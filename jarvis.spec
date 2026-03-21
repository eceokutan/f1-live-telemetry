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
        # Config example (user can copy to config.json)
        (os.path.join(ROOT, 'config.example.json'), '.'),
    ],
    hiddenimports=[
        # PyQt5
        'PyQt5.QtWidgets',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        # Matplotlib backends
        'matplotlib.backends.backend_qt5agg',
        # AI modules (optional - app works without them)
        'ai.race_engineer',
        'ai.race_engineer_core',
        'ai.voice_input',
        'ai.tts_output',
        'ai.ptt_controller',
        'ai.local_llm_inference',
        'ai.model_downloader',
        'ai.model_prewarm',
        # Post-race analysis
        'analysis.ai_pipeline_bridge',
        'jarvis_post.llm.local_client',
        'jarvis_post.llm.client',
        # Data recording
        'data.session_recorder',
        'data.session_exporter',
        # Telemetry backends
        'telemetry.backends.ac_backend',
        'telemetry.lap_buffer',
        # Dependencies that PyInstaller may miss
        'pydantic',
        'pydantic.deprecated.decorator',
        'dotenv',
        'numpy',
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
        'pandas',
        # Exclude test frameworks
        'pytest',
        'unittest',
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
    console=False,  # No console window
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
