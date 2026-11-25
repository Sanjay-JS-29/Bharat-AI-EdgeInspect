"""
Configuration module for BharatPCB-AI.
Defines model paths, classes, color maps, thresholds, and alignment/comparison safeguards.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = OUTPUT_DIR / "reports"
VISUAL_LOGS_DIR = OUTPUT_DIR / "visual_logs"
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"

# Target Component Classes from models/best.pt
CLASSES = [
    "capacitor",
    "connector",
    "diode",
    "ic",
    "inductor",
    "led",
    "resistor",
    "switch",
    "transistor"
]

CLASS_MAP = {idx: name for idx, name in enumerate(CLASSES)}

# High-visibility BGR color palette for PCB visualization (OpenCV format)
CLASS_COLORS = {
    "capacitor": (255, 140, 0),     # Deep Sky Blue
    "connector": (205, 90, 106),    # Slate Blue
    "diode": (0, 215, 255),         # Gold
    "ic": (0, 0, 255),              # Red / Critical IC
    "inductor": (255, 0, 255),      # Magenta
    "led": (0, 255, 255),           # Yellow
    "resistor": (0, 255, 0),        # Lime Green
    "switch": (238, 130, 238),      # Violet
    "transistor": (255, 191, 0)     # Amber Cyan
}

# Defect Overlay Colors (BGR)
DEFECT_COLORS = {
    "missing": (0, 0, 255),         # Bright Red
    "extra": (255, 0, 128),         # Purple
    "misplaced": (0, 165, 255),     # Orange
    "low_confidence": (0, 255, 255) # Yellow
}

# Fallback color for unmapped classes
DEFAULT_COLOR = (200, 200, 200)

# Detection Thresholds
DEFAULT_CONF_THRESHOLD = 0.35
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_IMG_SIZE = 640

# Alignment Safeguard Thresholds (Stage 2)
ALIGNMENT_MIN_FEATURES = 1000
ALIGNMENT_MIN_GOOD_MATCHES = 15
ALIGNMENT_MIN_INLIER_RATIO = 0.25
ALIGNMENT_RANSAC_REPROJ_THRESH = 3.0

# Component Comparison Safeguards (Stage 3)
MATCH_IOU_THRESHOLD = 0.30
MAX_CENTROID_DRIFT_PIXELS = 35.0  # Allowable shift before flagging as misplaced
LOW_CONFIDENCE_FLAG_THRESHOLD = 0.40
