"""
Styling constants and CSS classes for the dashboard.

Uses the Catppuccin Mocha color scheme for a cohesive dark theme.
"""

# =============================================================================
# FONT CONFIGURATION
# =============================================================================

FONT_FAMILY = "Iosevka, monospace"
FONT_SIZE = 12
FONT_SIZE_TITLE = 16

# =============================================================================
# CATPPUCCIN MOCHA PALETTE
# =============================================================================

# Base Colors
CATPPUCCIN_BASE = "#1e1e2e"        # Main background
CATPPUCCIN_MANTLE = "#181825"      # Darker background
CATPPUCCIN_CRUST = "#11111b"       # Darkest background
CATPPUCCIN_TEXT = "#cdd6f4"        # Main text
CATPPUCCIN_SUBTEXT1 = "#bac2de"    # Secondary text
CATPPUCCIN_SUBTEXT0 = "#a6adc8"    # Tertiary text
CATPPUCCIN_OVERLAY0 = "#6c7086"    # Overlay/dimmed
CATPPUCCIN_SURFACE0 = "#313244"    # Surface
CATPPUCCIN_SURFACE1 = "#45475a"    # Surface raised

# Accent Colors
CATPPUCCIN_GREEN = "#a6e3a1"       # Success
CATPPUCCIN_RED = "#f38ba8"         # Failure
CATPPUCCIN_PEACH = "#fab387"       # Error/Warning
CATPPUCCIN_YELLOW = "#f9e2af"      # Timeout
CATPPUCCIN_BLUE = "#89b4fa"        # Links/Info
CATPPUCCIN_SAPPHIRE = "#74c7ec"    # Accents
CATPPUCCIN_MAUVE = "#cba6f7"       # Highlights

# =============================================================================
# STATUS COLORS
# =============================================================================

COLOR_SUCCESS = CATPPUCCIN_GREEN
COLOR_FAILURE = CATPPUCCIN_RED
COLOR_ERROR = CATPPUCCIN_PEACH
COLOR_TIMEOUT = CATPPUCCIN_YELLOW

# =============================================================================
# STATUS SYMBOLS
# =============================================================================

# Matching progress.py symbols
SYMBOL_SUCCESS = "\u2713"      # checkmark
SYMBOL_FAILURE = "\u2717"      # x
SYMBOL_ERROR = "\u26a0"        # warning
SYMBOL_TIMEOUT = "\u23f1"      # stopwatch

# =============================================================================
# VIOLIN PLOT STYLING
# =============================================================================

VIOLIN_FILL = f"rgba(137, 180, 250, 0.1)"  # Blue with transparency
VIOLIN_LINE = f"rgba(137, 180, 250, 0.5)"  # Blue
VIOLIN_LINE_WIDTH = 0.35
VIOLIN_WIDTH = 0.25

# =============================================================================
# DATA POINT MARKERS
# =============================================================================

POINT_SIZE = 5
POINT_COLOR = f"rgba(205, 214, 244, 0.4)"  # Text color with transparency
POINT_OUTLINE = f"rgba(49, 50, 68, 0.8)"   # Surface0
POINT_OUTLINE_WIDTH = 0.5

# =============================================================================
# MEAN MARKER
# =============================================================================

MEAN_MARKER_SIZE = 7
MEAN_MARKER_COLOR = CATPPUCCIN_SAPPHIRE
MEAN_MARKER_WIDTH = 2

# =============================================================================
# FAILED RUN MARKERS
# =============================================================================

FAILED_MARKER_SYMBOL = "x-thin"
FAILED_MARKER_SIZE = 5
FAILED_MARKER_WIDTH = 1
FAILED_MARKER_COLOR = f"rgba(243, 139, 168, 0.8)"  # Red with transparency

# =============================================================================
# LAYOUT & SPACING
# =============================================================================

MARGIN = dict(l=0, r=10, t=0, b=0)
HEIGHT_PER_CASE = 40
MIN_PLOT_HEIGHT = 80
ANNOTATION_PADDING = 0.5
ANNOTATION_X_OFFSET = 0.3  # Fraction of dx to offset annotations

# =============================================================================
# GRID STYLING
# =============================================================================

GRID_COLOR = f"rgba(108, 112, 134, 0.2)"  # Overlay0 with transparency
GRID_WIDTH = 0.5
PLOT_BGCOLOR = CATPPUCCIN_BASE
PAPER_BGCOLOR = CATPPUCCIN_BASE
ANNOTATION_BGCOLOR = PLOT_BGCOLOR

# =============================================================================
# TEXT COLORS
# =============================================================================

TEXT_COLOR = CATPPUCCIN_TEXT
TEXT_DIMMED = CATPPUCCIN_SUBTEXT0
TEXT_SECONDARY = CATPPUCCIN_OVERLAY0

# =============================================================================
# CSS STYLE CLASSES
# =============================================================================

STYLE_CLASSES = """
.text-base {
    margin: 0;
    padding: 0;
    color: """ + TEXT_COLOR + """;
}

.text-dimmed {
    color: """ + TEXT_DIMMED + """;
}

.text-secondary {
    color: """ + TEXT_SECONDARY + """;
    font-style: italic;
}

.text-bold {
    font-weight: bold;
}

.text-underline {
    text-decoration: underline;
}

.link {
    color: """ + CATPPUCCIN_BLUE + """;
    text-decoration: underline;
}

.pre-text {
    font-family: """ + FONT_FAMILY + """;
    font-size: """ + str(FONT_SIZE) + """px;
    margin: 0;
    padding: 0;
}

.status-success {
    color: """ + COLOR_SUCCESS + """;
    font-weight: bold;
}

.status-failure {
    color: """ + COLOR_FAILURE + """;
    font-weight: bold;
}

.status-error {
    color: """ + COLOR_ERROR + """;
    font-weight: bold;
}

.status-timeout {
    color: """ + COLOR_TIMEOUT + """;
    font-weight: bold;
}
"""
