"""
Styling constants and theme configuration for the dashboard UI.
"""

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
ACCENT_BLUE = "#6FA8FF"       # Primary accent (speed line, etc.)
ACCENT_RED = "#FF6B6B"        # FL tire, critical alerts
ACCENT_CYAN = "#4ECDC4"       # FR tire
ACCENT_YELLOW = "#FFD93D"     # RL tire, warnings
ACCENT_GREEN = "#6BCB77"      # RR tire

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
    QPushButton {{
        background-color: {ACCENT_BLUE};
        color: #FFFFFF;
        border: none;
        border-radius: 4px;
        padding: 6px 12px;
        font-weight: bold;
    }}
    QPushButton:hover {{
        background-color: #5A98EF;
    }}
    QPushButton:pressed {{
        background-color: #4A88DF;
    }}
    QMenuBar {{
        background-color: {BG_COLOR};
        color: {TEXT_COLOR};
        border-bottom: 1px solid {BORDER_COLOR};
    }}
    QMenuBar::item:selected {{
        background-color: {BG_COLOR_LIGHT};
    }}
"""

# =============================================================================
# Tire Colors (for multi-line charts)
# =============================================================================

TIRE_COLORS = [ACCENT_RED, ACCENT_CYAN, ACCENT_YELLOW, ACCENT_GREEN]
TIRE_LABELS = ["FL", "FR", "RL", "RR"]
