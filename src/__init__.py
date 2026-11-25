"""
BharatPCB-AI / EdgeInspect AI Package
"""

__version__ = "1.0.0"

from src.config import CLASSES, CLASS_MAP, MODEL_PATH
from src.detector import PCBDetector, UltralyticsDetector, Detection
from src.aligner import PCBAligner, AlignmentResult, AlignmentMetrics
from src.comparator import ComponentComparator, ComparisonConfig, ComparisonSummary, ComponentDiscrepancy
from src.inspector import PCBInspector, InspectionResult, run_inspection, extract_defect_records, get_board_region
from src.visualizer import PCBVisualizer
from src.reporter import InspectionReporter, ReportPaths
from src.advisor import PCBAdvisor
from src.camera import decode_image_bytes, validate_image, bgr_to_rgb

