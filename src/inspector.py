"""
Inspection Orchestrator for BharatPCB-AI.
Combines image alignment, YOLO component detection, and discrepancy analysis
into a unified inspection workflow with strict safety gating.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from src.config import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_IMG_SIZE
)
from src.detector import PCBDetector, Detection
from src.aligner import PCBAligner, AlignmentResult
from src.comparator import (
    ComponentComparator,
    ComparisonConfig,
    ComparisonSummary,
    ComponentDiscrepancy
)


@dataclass
class InspectionResult:
    """Complete inspection outcome container."""
    verdict: str  # 'PASS', 'FAIL', 'REVIEW', 'HALTED_ALIGNMENT_REJECTED'
    verdict_reason: str
    can_proceed_to_comparison: bool
    timestamp: str
    alignment: AlignmentResult
    comparison: Optional[ComparisonSummary]
    ref_detections: List[Detection]
    test_detections: List[Detection]
    aligned_test_image: Optional[np.ndarray]

    @property
    def is_passed(self) -> bool:
        return self.verdict == "PASS"

    @property
    def total_defects(self) -> int:
        if self.comparison is None:
            return 0
        return self.comparison.total_discrepancies

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "verdict_reason": self.verdict_reason,
            "can_proceed_to_comparison": self.can_proceed_to_comparison,
            "timestamp": self.timestamp,
            "alignment": self.alignment.to_dict(),
            "comparison": self.comparison.to_dict() if self.comparison else None,
            "ref_detections_count": len(self.ref_detections),
            "test_detections_count": len(self.test_detections),
            "ref_detections": [d.to_dict() for d in self.ref_detections],
            "test_detections": [d.to_dict() for d in self.test_detections]
        }


class PCBInspector:
    """
    Main inspection system coordinating alignment, detection, and defect evaluation.
    Enforces strict safety barriers to avoid generating false defect claims.
    """

    def __init__(
        self,
        detector: Optional[PCBDetector] = None,
        aligner: Optional[PCBAligner] = None,
        comparator: Optional[ComponentComparator] = None
    ):
        self.detector = detector or PCBDetector()
        self.aligner = aligner or PCBAligner()
        self.comparator = comparator or ComponentComparator()

    def inspect(
        self,
        ref_image: np.ndarray,
        test_image: np.ndarray,
        manual_fiducials: Optional[Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]] = None,
        conf: float = DEFAULT_CONF_THRESHOLD,
        iou: float = DEFAULT_IOU_THRESHOLD,
        imgsz: int = DEFAULT_IMG_SIZE,
        comparison_config: Optional[ComparisonConfig] = None
    ) -> InspectionResult:
        """
        Execute full end-to-end inspection:
        1. Align test image onto reference image coordinate frame.
        2. SAFETY GATE: Abort comparison if alignment is rejected.
        3. Detect components on reference and aligned test images using YOLO.
        4. Compare detections and derive missing, extra, misplaced, and low-confidence items.
        5. Compute PASS / FAIL / REVIEW verdict.
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        # Step 1: Image Alignment
        if manual_fiducials is not None:
            ref_pts, test_pts = manual_fiducials
            alignment_result = self.aligner.align_manual(ref_image, test_image, ref_pts, test_pts)
        else:
            alignment_result = self.aligner.align(ref_image, test_image)

        # Step 2: CRITICAL SAFETY GATE
        # Never generate missing/extra component verdicts when alignment quality is insufficient
        if not alignment_result.can_proceed_to_comparison or alignment_result.aligned_image is None:
            reason = (
                f"Inspection HALTED: Image alignment rejected ({alignment_result.metrics.status}). "
                f"Reason: {alignment_result.metrics.rejection_reason}. "
                "Component comparison aborted to prevent false defect claims."
            )
            return InspectionResult(
                verdict="REVIEW",
                verdict_reason=reason,
                can_proceed_to_comparison=False,
                timestamp=now_iso,
                alignment=alignment_result,
                comparison=None,
                ref_detections=[],
                test_detections=[],
                aligned_test_image=None
            )

        # Step 3: YOLO Inference on Reference and Aligned Test Images
        aligned_test = alignment_result.aligned_image
        ref_detections = self.detector.predict(ref_image, conf=conf, iou=iou, imgsz=imgsz)
        test_detections = self.detector.predict(aligned_test, conf=conf, iou=iou, imgsz=imgsz)

        # Step 4: Component Comparison
        active_comparator = self.comparator
        if comparison_config is not None:
            active_comparator = ComponentComparator(comparison_config)

        comparison_summary = active_comparator.compare(ref_detections, test_detections)

        # Step 5: Verdict Determination
        verdict, verdict_reason = self._determine_verdict(comparison_summary)

        return InspectionResult(
            verdict=verdict,
            verdict_reason=verdict_reason,
            can_proceed_to_comparison=True,
            timestamp=now_iso,
            alignment=alignment_result,
            comparison=comparison_summary,
            ref_detections=ref_detections,
            test_detections=test_detections,
            aligned_test_image=aligned_test
        )

    def _determine_verdict(self, summary: ComparisonSummary) -> Tuple[str, str]:
        """
        Evaluate comparison summary into PASS, FAIL, or REVIEW with actionable justification.
        """
        missing_count = len(summary.missing)
        extra_count = len(summary.extra)
        misplaced_count = len(summary.misplaced)
        mismatch_count = len(summary.class_mismatches)
        low_conf_count = len(summary.low_confidence_warnings)

        reasons = []

        # Critical Defects -> FAIL
        if missing_count > 0:
            reasons.append(f"{missing_count} missing component(s)")
        if mismatch_count > 0:
            reasons.append(f"{mismatch_count} class mismatch(es)")
        if misplaced_count > 0:
            reasons.append(f"{misplaced_count} misplaced component(s)")

        if reasons:
            return "FAIL", f"FAIL: Board has critical defects: {', '.join(reasons)}."

        # Secondary Discrepancies or Low Confidence -> REVIEW
        if extra_count > 0:
            reasons.append(f"{extra_count} uncataloged extra component(s)")
        if low_conf_count > 0:
            reasons.append(f"{low_conf_count} low-confidence detection(s) requiring verification")

        if reasons:
            return "REVIEW", f"REVIEW: Manual inspection advised: {', '.join(reasons)}."

        return "PASS", f"PASS: All {len(summary.matched)} components verified within tolerances."


def get_board_region(cx: float, cy: float, width: int, height: int) -> str:
    """Classify 2D centroid into one of 9 spatial board regions."""
    if width <= 0 or height <= 0:
        return "Center"
    rx = cx / max(1, width)
    ry = cy / max(1, height)
    row = "Top" if ry < 0.33 else ("Bottom" if ry > 0.67 else "Center")
    col = "Left" if rx < 0.33 else ("Right" if rx > 0.67 else "Center")
    if row == "Center" and col == "Center":
        return "Center"
    if row == "Center":
        return f"Mid-{col}"
    if col == "Center":
        return f"{row}-Mid"
    return f"{row}-{col}"


def extract_defect_records(
    result: InspectionResult,
    img_w: int = 640,
    img_h: int = 640
) -> List[Dict[str, Any]]:
    """Extract tabular defect records with Defect ID, region, positions, and explanations."""
    records = []
    if result.comparison is None:
        return records

    summary = result.comparison
    idx = 1

    # Missing
    for d in summary.missing:
        cx, cy = d.component.centroid
        region = get_board_region(cx, cy, img_w, img_h)
        records.append({
            "defect_id": f"DEF-{idx:03d}",
            "defect_type": "MISSING",
            "component_name": d.component.class_name.upper(),
            "confidence": f"{d.component.confidence:.1%}",
            "region": region,
            "expected_pos": f"({cx:.0f}, {cy:.0f})",
            "actual_pos": "ABSENT",
            "explanation": d.reason,
            "box": d.component.box,
            "source_image": "reference"
        })
        idx += 1

    # Misplaced
    for d in summary.misplaced:
        cx, cy = d.component.centroid
        exp_pos = f"({d.paired_component.centroid[0]:.0f}, {d.paired_component.centroid[1]:.0f})" if d.paired_component else "N/A"
        region = get_board_region(cx, cy, img_w, img_h)
        records.append({
            "defect_id": f"DEF-{idx:03d}",
            "defect_type": "MISPLACED",
            "component_name": d.component.class_name.upper(),
            "confidence": f"{d.component.confidence:.1%}",
            "region": region,
            "expected_pos": exp_pos,
            "actual_pos": f"({cx:.0f}, {cy:.0f})",
            "explanation": d.reason,
            "box": d.component.box,
            "source_image": "test"
        })
        idx += 1

    # Class Mismatch
    for d in summary.class_mismatches:
        cx, cy = d.component.centroid
        exp_name = d.paired_component.class_name.upper() if d.paired_component else "UNKNOWN"
        exp_pos = f"({d.paired_component.centroid[0]:.0f}, {d.paired_component.centroid[1]:.0f})" if d.paired_component else "N/A"
        region = get_board_region(cx, cy, img_w, img_h)
        records.append({
            "defect_id": f"DEF-{idx:03d}",
            "defect_type": "CLASS_MISMATCH",
            "component_name": f"{d.component.class_name.upper()} (Exp: {exp_name})",
            "confidence": f"{d.component.confidence:.1%}",
            "region": region,
            "expected_pos": exp_pos,
            "actual_pos": f"({cx:.0f}, {cy:.0f})",
            "explanation": d.reason,
            "box": d.component.box,
            "source_image": "test"
        })
        idx += 1

    # Extra
    for d in summary.extra:
        cx, cy = d.component.centroid
        region = get_board_region(cx, cy, img_w, img_h)
        records.append({
            "defect_id": f"DEF-{idx:03d}",
            "defect_type": "EXTRA",
            "component_name": d.component.class_name.upper(),
            "confidence": f"{d.component.confidence:.1%}",
            "region": region,
            "expected_pos": "NONE (Uncataloged)",
            "actual_pos": f"({cx:.0f}, {cy:.0f})",
            "explanation": d.reason,
            "box": d.component.box,
            "source_image": "test"
        })
        idx += 1

    return records


def run_inspection(
    ref_img: np.ndarray,
    test_img: np.ndarray,
    backend: str = "yolo",
    **kwargs
) -> InspectionResult:
    """
    Public entry point for full reference-vs-test PCB inspection.
    Uses existing safety-gated OpenCV alignment, YOLO detection, and discrepancy logic.
    """
    inspector = PCBInspector()
    return inspector.inspect(ref_img, test_img, **kwargs)

