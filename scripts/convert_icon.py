"""
Convert SVG icon to ICO format for PyInstaller Windows executable.

Usage:
    python scripts/convert_icon.py

Requires: Pillow, cairosvg (pip install Pillow cairosvg)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SVG_PATH = PROJECT_ROOT / "ui" / "img" / "f1_jarvis_topdown_massive_tyres.svg"
ICO_PATH = PROJECT_ROOT / "ui" / "img" / "jarvis.ico"


def convert_svg_to_ico():
    try:
        import cairosvg
        from PIL import Image
        import io
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install Pillow cairosvg")
        sys.exit(1)

    if not SVG_PATH.exists():
        print(f"SVG not found: {SVG_PATH}")
        sys.exit(1)

    # Render SVG to PNG at 256x256
    png_data = cairosvg.svg2png(
        url=str(SVG_PATH),
        output_width=256,
        output_height=256,
    )

    img = Image.open(io.BytesIO(png_data))

    # Create ICO with multiple sizes
    sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    img.save(str(ICO_PATH), format="ICO", sizes=sizes)
    print(f"Icon saved to: {ICO_PATH}")


if __name__ == "__main__":
    convert_svg_to_ico()
