"""
Styling constants and theme configuration for the dashboard UI.
"""
import os
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Font Configuration
# =============================================================================

FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")

# Font family names (after loading)
FONT_HEADING = "Bebas Neue"     # All-caps headers, titles
FONT_BODY = "Rajdhani"          # Buttons, labels, body text

_fonts_loaded = False


def load_fonts():
    """Load custom fonts into the Qt font database. Call once at app startup."""
    global _fonts_loaded
    if _fonts_loaded:
        return

    try:
        from PyQt5.QtGui import QFontDatabase
        font_files = [
            "BebasNeue-Regular.ttf",
            "Rajdhani-Regular.ttf",
            "Rajdhani-SemiBold.ttf",
            "Rajdhani-Bold.ttf",
        ]
        for fname in font_files:
            path = os.path.join(FONT_DIR, fname)
            if os.path.exists(path):
                font_id = QFontDatabase.addApplicationFont(path)
                if font_id < 0:
                    logger.warning("Failed to load font: %s", fname)
            else:
                logger.warning("Font file not found: %s", path)
        _fonts_loaded = True
    except Exception as e:
        logger.warning("Could not load custom fonts: %s", e)

# =============================================================================
# Color Palette
# =============================================================================

# Dark theme colors
BG_COLOR = "#111111"          # Main background
BG_COLOR_LIGHT = "#181818"    # Lighter background (axes, panels)
TEXT_COLOR = "#EEEEEE"        # Main text
TEXT_COLOR_DIM = "#CCCCCC"    # Dimmed text (axis labels, etc.)
TEXT_COLOR_DARK = "#888888"   # Dark text (timestamps)
BORDER_COLOR = "#555555"      # Borders
GRID_COLOR = "#333333"        # Grid lines

# Accent colors
ACCENT_PRIMARY = "#E10600"    # F1 Racing Red - primary accent
ACCENT_RED = "#FF6B6B"        # FL tire, critical alerts
ACCENT_CYAN = "#4ECDC4"       # FR tire
ACCENT_YELLOW = "#FFD93D"     # RL tire, warnings
ACCENT_GREEN = "#6BCB77"      # RR tire

# Graph line color (kept blue for readability on dark backgrounds)
GRAPH_LINE_COLOR = "#6FA8FF"

# Legacy alias (some files still reference ACCENT_BLUE)
ACCENT_BLUE = ACCENT_PRIMARY

# Priority colors (for AI commentary)
PRIORITY_CRITICAL = "#FF3B30"
PRIORITY_HIGH = "#FF9500"
PRIORITY_MEDIUM = "#FFCC00"
PRIORITY_LOW = "#8E8E93"

# =============================================================================
# Matplotlib Theme
# =============================================================================

MATPLOTLIB_DARK_THEME = {
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": BG_COLOR_LIGHT,
    "axes.edgecolor": TEXT_COLOR_DIM,
    "axes.labelcolor": TEXT_COLOR_DIM,
    "axes.titlecolor": "#FFFFFF",
    "xtick.color": TEXT_COLOR_DIM,
    "ytick.color": TEXT_COLOR_DIM,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.6,
    "text.color": TEXT_COLOR,
}

# =============================================================================
# PyQt5 Stylesheet
# =============================================================================

DARK_STYLESHEET = f"""
    QMainWindow {{
        background-color: {BG_COLOR};
        color: {TEXT_COLOR};
    }}
    QWidget {{
        background-color: {BG_COLOR};
        color: {TEXT_COLOR};
        font-family: '{FONT_BODY}';
    }}
    QGroupBox {{
        border: 1px solid {BORDER_COLOR};
        border-radius: 4px;
        margin-top: 8px;
        padding-top: 10px;
        font-weight: bold;
        color: {TEXT_COLOR};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 8px;
        padding: 0 4px 0 4px;
    }}
    QLabel {{
        color: {TEXT_COLOR};
        font-size: 10pt;
    }}
    QTableWidget {{
        background-color: {BG_COLOR_LIGHT};
        color: {TEXT_COLOR};
        gridline-color: {BORDER_COLOR};
        border: 1px solid {BORDER_COLOR};
    }}
    QTableWidget::item {{
        padding: 4px;
    }}
    QHeaderView::section {{
        background-color: {BG_COLOR};
        color: {TEXT_COLOR};
        padding: 4px;
        border: 1px solid {BORDER_COLOR};
        font-weight: bold;
    }}
    QTextEdit {{
        background-color: {BG_COLOR_LIGHT};
        color: {TEXT_COLOR};
        border: 1px solid {BORDER_COLOR};
        border-radius: 4px;
        padding: 4px;
    }}
    QListWidget {{
        background-color: {BG_COLOR_LIGHT};
        color: {TEXT_COLOR};
        border: 1px solid {BORDER_COLOR};
    }}
    QListWidget::item:selected {{
        background-color: {ACCENT_PRIMARY};
    }}
    QTabWidget::pane {{
        border: 1px solid {BORDER_COLOR};
        background-color: {BG_COLOR};
    }}
    QTabBar::tab {{
        background-color: {BG_COLOR_LIGHT};
        color: {TEXT_COLOR};
        padding: 8px 20px;
        border: 1px solid {BORDER_COLOR};
        border-bottom: none;
    }}
    QTabBar::tab:selected {{
        background-color: {BG_COLOR};
    }}
    QPushButton {{
        background-color: {ACCENT_PRIMARY};
        color: #FFFFFF;
        border: none;
        border-radius: 4px;
        padding: 6px 12px;
        font-weight: bold;
    }}
    QPushButton:hover {{
        background-color: #C00500;
    }}
    QPushButton:pressed {{
        background-color: #A00400;
    }}
    QMenuBar {{
        background-color: {BG_COLOR};
        color: {TEXT_COLOR};
        border-bottom: 1px solid {BORDER_COLOR};
        font-size: 11pt;
        padding: 2px 0;
    }}
    QMenuBar::item {{
        padding: 6px 14px;
    }}
    QMenuBar::item:selected {{
        background-color: {BG_COLOR_LIGHT};
    }}
    QMenu {{
        background-color: {BG_COLOR_LIGHT};
        color: {TEXT_COLOR};
        border: 1px solid {BORDER_COLOR};
        font-size: 11pt;
        padding: 4px 0;
    }}
    QMenu::item {{
        padding: 8px 24px;
    }}
    QMenu::item:selected {{
        background-color: {ACCENT_PRIMARY};
    }}
    QMenu::separator {{
        height: 1px;
        background: {BORDER_COLOR};
        margin: 4px 8px;
    }}
    QSlider::groove:horizontal {{
        border: 1px solid {BORDER_COLOR};
        height: 8px;
        background: {BG_COLOR_LIGHT};
        margin: 2px 0;
    }}
    QSlider::handle:horizontal {{
        background: {ACCENT_PRIMARY};
        border: 1px solid {ACCENT_PRIMARY};
        width: 18px;
        margin: -5px 0;
        border-radius: 9px;
    }}
    QScrollArea {{
        border: none;
    }}
    QComboBox {{
        background-color: {BG_COLOR_LIGHT};
        color: {TEXT_COLOR};
        border: 1px solid {BORDER_COLOR};
        border-radius: 4px;
        padding: 4px;
    }}
    QComboBox::drop-down {{
        border: none;
    }}
    QComboBox QAbstractItemView {{
        background-color: {BG_COLOR_LIGHT};
        color: {TEXT_COLOR};
        selection-background-color: {ACCENT_PRIMARY};
    }}
"""

# =============================================================================
# Tire Colors (for multi-line charts)
# =============================================================================

TIRE_COLORS = [ACCENT_RED, ACCENT_CYAN, ACCENT_YELLOW, ACCENT_GREEN]
TIRE_LABELS = ["FL", "FR", "RL", "RR"]
