"""
Component Comparison Engine for BharatPCB-AI.
Compares reference PCB detections against aligned test PCB detections.
Identifies matched components, misplaced components, class mismatches,
missing components, extra components, and low-confidence warnings.
"""

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple
import math

from src.detector import Detection
from src.config import (
    MATCH_IOU_THRESHOLD,
    MAX_CENTROID_DRIFT_PIXELS,
    LOW_CONFIDENCE_FLAG_THRESHOLD
)


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    return float(inter_area / union_area) if union_area > 0.0 else 0.0


def compute_centroid_distance(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
    """Compute Euclidean distance between two (cx, cy) centroids in pixels."""
    return float(math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2))


@dataclass
class MatchedPair:
    """Pair of matched reference and test components."""
    reference_component: Detection
    test_component: Detection
    iou: float
    centroid_distance: float
    is_misplaced: bool
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_name": self.reference_component.class_name,
            "reference": self.reference_component.to_dict(),
            "test": self.test_component.to_dict(),
            "iou": round(self.iou, 4),
            "centroid_distance": round(self.centroid_distance, 2),
            "is_misplaced": self.is_misplaced,
            "notes": self.notes
        }


@dataclass
class ComponentDiscrepancy:
    """
    Algorithmically derived discrepancy between Reference and Test boards.
    Explicitly marked to distinguish from raw model detections.
    """
    discrepancy_type: str  # 'MISSING', 'EXTRA', 'MISPLACED', 'CLASS_MISMATCH'
    component: Detection   # Reference detection for MISSING, Test detection for EXTRA/MISPLACED/MISMATCH
    paired_component: Optional[Detection] = None
    centroid_drift: Optional[float] = None
    iou: Optional[float] = None
    severity: str = "HIGH"  # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
    reason: str = ""
    is_model_detection: bool = False  # Distinguishes derived discrepancy from raw model detections

    def to_dict(self) -> Dict[str, Any]:
        return {
            "discrepancy_type": self.discrepancy_type,
            "severity": self.severity,
            "reason": self.reason,
            "is_model_detection": self.is_model_detection,
            "centroid_drift": round(self.centroid_drift, 2) if self.centroid_drift is not None else None,
            "iou": round(self.iou, 4) if self.iou is not None else None,
            "component": self.component.to_dict(),
            "paired_component": self.paired_component.to_dict() if self.paired_component else None
        }


@dataclass
class ComparisonConfig:
    """Configurable spatial tolerances and confidence thresholds for comparison."""
    match_iou_threshold: float = MATCH_IOU_THRESHOLD
    max_centroid_drift: float = MAX_CENTROID_DRIFT_PIXELS
    min_confidence_threshold: float = LOW_CONFIDENCE_FLAG_THRESHOLD
    spatial_search_radius: float = 75.0
    allow_class_mismatch_check: bool = True


@dataclass
class ComparisonSummary:
    """Complete summary of component comparison."""
    ref_count: int
    test_count: int
    matched: List[MatchedPair] = field(default_factory=list)
    missing: List[ComponentDiscrepancy] = field(default_factory=list)
    extra: List[ComponentDiscrepancy] = field(default_factory=list)
    misplaced: List[ComponentDiscrepancy] = field(default_factory=list)
    class_mismatches: List[ComponentDiscrepancy] = field(default_factory=list)
    low_confidence_warnings: List[Detection] = field(default_factory=list)

    @property
    def total_discrepancies(self) -> int:
        return len(self.missing) + len(self.extra) + len(self.misplaced) + len(self.class_mismatches)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ref_count": self.ref_count,
            "test_count": self.test_count,
            "matched_count": len(self.matched),
            "missing_count": len(self.missing),
            "extra_count": len(self.extra),
            "misplaced_count": len(self.misplaced),
            "class_mismatch_count": len(self.class_mismatches),
            "low_confidence_count": len(self.low_confidence_warnings),
            "total_discrepancies": self.total_discrepancies,
            "matched": [m.to_dict() for m in self.matched],
            "missing": [d.to_dict() for d in self.missing],
            "extra": [d.to_dict() for d in self.extra],
            "misplaced": [d.to_dict() for d in self.misplaced],
            "class_mismatches": [d.to_dict() for d in self.class_mismatches],
            "low_confidence_warnings": [d.to_dict() for d in self.low_confidence_warnings]
        }


class ComponentComparator:
    """
    Compares reference and aligned test detections with multi-metric matching:
    Component class, IoU, centroid Euclidean distance, and configurable tolerances.
    """

    def __init__(self, config: Optional[ComparisonConfig] = None):
        self.config = config or ComparisonConfig()

    def compare(
        self,
        ref_detections: List[Detection],
        test_detections: List[Detection]
    ) -> ComparisonSummary:
        """
        Execute component comparison between reference and test board detections.
        """
        summary = ComparisonSummary(
            ref_count=len(ref_detections),
            test_count=len(test_detections)
        )

        # 1. Audit detections for low confidence
        for det in test_detections:
            if det.confidence < self.config.min_confidence_threshold:
                summary.low_confidence_warnings.append(det)

        # Track which detections have been matched
        unmatched_ref = set(range(len(ref_detections)))
        unmatched_test = set(range(len(test_detections)))

        # 2. First Pass: Match components of the exact same class
        # Candidate pairs with scoring
        candidates = []
        for r_idx in unmatched_ref:
            r_det = ref_detections[r_idx]
            for t_idx in unmatched_test:
                t_det = test_detections[t_idx]
                if r_det.class_name == t_det.class_name:
                    iou = compute_iou(r_det.box, t_det.box)
                    dist = compute_centroid_distance(r_det.centroid, t_det.centroid)
                    # Candidate if reasonable proximity or overlap
                    if iou > 0.0 or dist <= self.config.spatial_search_radius:
                        candidates.append((iou, -dist, r_idx, t_idx, dist))

        # Sort candidates: highest IoU first, then smallest centroid distance
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

        for iou, neg_dist, r_idx, t_idx, dist in candidates:
            if r_idx in unmatched_ref and t_idx in unmatched_test:
                r_det = ref_detections[r_idx]
                t_det = test_detections[t_idx]

                # Check if matched within shift tolerance
                if iou >= self.config.match_iou_threshold and dist <= self.config.max_centroid_drift:
                    # Valid match within acceptable tolerance
                    summary.matched.append(
                        MatchedPair(
                            reference_component=r_det,
                            test_component=t_det,
                            iou=iou,
                            centroid_distance=dist,
                            is_misplaced=False,
                            notes=f"Verified match (IoU: {iou:.2f}, shift: {dist:.1f}px)"
                        )
                    )
                    unmatched_ref.remove(r_idx)
                    unmatched_test.remove(t_idx)
                elif dist > self.config.max_centroid_drift and dist <= self.config.spatial_search_radius:
                    # Same class detected, but shifted beyond acceptable tolerance
                    summary.misplaced.append(
                        ComponentDiscrepancy(
                            discrepancy_type="MISPLACED",
                            component=t_det,
                            paired_component=r_det,
                            centroid_drift=dist,
                            iou=iou,
                            severity="HIGH",
                            reason=(
                                f"Component '{t_det.class_name}' position shifted {dist:.1f}px "
                                f"(allowable tolerance: {self.config.max_centroid_drift:.1f}px)"
                            ),
                            is_model_detection=False
                        )
                    )
                    unmatched_ref.remove(r_idx)
                    unmatched_test.remove(t_idx)

        # 3. Second Pass: Check remaining components for Class Mismatch
        if self.config.allow_class_mismatch_check:
            mismatch_candidates = []
            for r_idx in list(unmatched_ref):
                r_det = ref_detections[r_idx]
                for t_idx in list(unmatched_test):
                    t_det = test_detections[t_idx]
                    iou = compute_iou(r_det.box, t_det.box)
                    dist = compute_centroid_distance(r_det.centroid, t_det.centroid)
                    if iou >= 0.20 or dist <= self.config.max_centroid_drift:
                        mismatch_candidates.append((iou, -dist, r_idx, t_idx, dist))

            mismatch_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            for iou, neg_dist, r_idx, t_idx, dist in mismatch_candidates:
                if r_idx in unmatched_ref and t_idx in unmatched_test:
                    r_det = ref_detections[r_idx]
                    t_det = test_detections[t_idx]
                    summary.class_mismatches.append(
                        ComponentDiscrepancy(
                            discrepancy_type="CLASS_MISMATCH",
                            component=t_det,
                            paired_component=r_det,
                            centroid_drift=dist,
                            iou=iou,
                            severity="CRITICAL",
                            reason=(
                                f"Class mismatch at centroid {r_det.centroid}: "
                                f"Expected '{r_det.class_name}', detected '{t_det.class_name}'"
                            ),
                            is_model_detection=False
                        )
                    )
                    unmatched_ref.remove(r_idx)
                    unmatched_test.remove(t_idx)

        # 4. Remaining unmatched reference components are Missing
        for r_idx in unmatched_ref:
            r_det = ref_detections[r_idx]
            summary.missing.append(
                ComponentDiscrepancy(
                    discrepancy_type="MISSING",
                    component=r_det,
                    paired_component=None,
                    centroid_drift=None,
                    iou=0.0,
                    severity="CRITICAL",
                    reason=f"Expected component '{r_det.class_name}' at {r_det.centroid} was not detected on test PCB",
                    is_model_detection=False
                )
            )

        # 5. Remaining unmatched test components are Extra
        for t_idx in unmatched_test:
            t_det = test_detections[t_idx]
            summary.extra.append(
                ComponentDiscrepancy(
                    discrepancy_type="EXTRA",
                    component=t_det,
                    paired_component=None,
                    centroid_drift=None,
                    iou=0.0,
                    severity="MEDIUM",
                    reason=f"Uncataloged component '{t_det.class_name}' detected at {t_det.centroid} not present on reference PCB",
                    is_model_detection=False
                )
            )

        return summary
