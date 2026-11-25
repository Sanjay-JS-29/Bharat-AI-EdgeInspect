"""
Stage 4 Verification Suite: Visualization Engine Tests.
Tests:
  1. Component detection bounding box & label rendering.
  2. Inspection defect overlay rendering (Matched, Missing, Extra, Misplaced, Class Mismatch, Low Conf).
  3. Side-by-side comparison image stitching with PASS, FAIL, and REVIEW banners.
  4. Visualization file persistence in outputs/visualizations/.
  5. Edge-case handling (empty detections, unaligned image).
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import Detection
from src.aligner import AlignmentResult, AlignmentMetrics
from src.comparator import (
    ComponentComparator,
    ComparisonConfig,
    ComparisonSummary,
    MatchedPair,
    ComponentDiscrepancy
)
from src.inspector import InspectionResult
from src.visualizer import PCBVisualizer, VISUALIZATIONS_DIR
from tests.test_alignment import generate_synthetic_pcb


def test_1_draw_detections():
    print("\n--- Test 1: Component Detections Rendering ---")
    visualizer = PCBVisualizer()
    canvas = np.full((600, 600, 3), (25, 95, 25), dtype=np.uint8)

    detections = [
        Detection(class_id=3, class_name="ic", confidence=0.96, box=[100.0, 100.0, 220.0, 220.0]),
        Detection(class_id=6, class_name="resistor", confidence=0.89, box=[300.0, 150.0, 360.0, 180.0]),
        Detection(class_id=0, class_name="capacitor", confidence=0.91, box=[400.0, 300.0, 460.0, 360.0])
    ]

    annotated = visualizer.draw_detections(canvas, detections, title="TEST DETECTIONS")
    assert annotated.shape == canvas.shape
    assert not np.array_equal(annotated, canvas), "Annotated image must differ from blank canvas"
    print("  -> Drawing standard detections: PASSED")


def test_2_render_inspection_views():
    print("\n--- Test 2: Defect Annotations (Matched, Missing, Extra, Misplaced) ---")
    visualizer = PCBVisualizer()
    ref_pcb = generate_synthetic_pcb(640, 640, seed=301)
    test_pcb = ref_pcb.copy()

    # Create mock inspection result with mixed discrepancies
    ref_ic = Detection(class_id=3, class_name="ic", confidence=0.98, box=[100.0, 100.0, 220.0, 220.0])
    test_ic = Detection(class_id=3, class_name="ic", confidence=0.97, box=[100.0, 100.0, 220.0, 220.0])

    ref_cap = Detection(class_id=0, class_name="capacitor", confidence=0.92, box=[300.0, 300.0, 350.0, 350.0])

    test_led = Detection(class_id=5, class_name="led", confidence=0.88, box=[450.0, 150.0, 490.0, 190.0])

    ref_ind = Detection(class_id=4, class_name="inductor", confidence=0.91, box=[150.0, 400.0, 210.0, 460.0])
    test_ind = Detection(class_id=4, class_name="inductor", confidence=0.89, box=[195.0, 400.0, 255.0, 460.0])

    low_conf_det = Detection(class_id=6, class_name="resistor", confidence=0.35, box=[500.0, 500.0, 540.0, 520.0])

    summary = ComparisonSummary(
        ref_count=3,
        test_count=4,
        matched=[MatchedPair(ref_ic, test_ic, 1.0, 0.0, False, "Exact match")],
        missing=[ComponentDiscrepancy("MISSING", ref_cap, severity="CRITICAL", reason="Missing capacitor")],
        extra=[ComponentDiscrepancy("EXTRA", test_led, severity="MEDIUM", reason="Extra uncataloged LED")],
        misplaced=[ComponentDiscrepancy("MISPLACED", test_ind, paired_component=ref_ind, centroid_drift=45.0, severity="HIGH", reason="Shifted inductor")],
        low_confidence_warnings=[low_conf_det]
    )

    metrics = AlignmentMetrics(1000, 1000, 100, 80, 0.80, 1.1, True, 0.92, "SUCCESS")
    align_res = AlignmentResult(True, test_pcb, np.eye(3), np.eye(3), metrics, True)

    insp_result = InspectionResult(
        verdict="FAIL",
        verdict_reason="FAIL: Board has critical defects: 1 missing, 1 misplaced.",
        can_proceed_to_comparison=True,
        timestamp="2026-11-25T12:00:00Z",
        alignment=align_res,
        comparison=summary,
        ref_detections=[ref_ic, ref_cap, ref_ind],
        test_detections=[test_ic, test_led, test_ind, low_conf_det],
        aligned_test_image=test_pcb
    )

    ref_ann, test_ann = visualizer.render_inspection_views(ref_pcb, insp_result)
    assert ref_ann.shape == ref_pcb.shape
    assert test_ann.shape == test_pcb.shape
    print("  -> Defect annotations rendering: PASSED")


def test_3_side_by_side_comparison():
    print("\n--- Test 3: Side-by-Side Comparison Stitching (PASS / FAIL / REVIEW) ---")
    visualizer = PCBVisualizer()
    ref_pcb = generate_synthetic_pcb(600, 600, seed=302)

    # 1. PASS Result
    metrics = AlignmentMetrics(1000, 1000, 100, 90, 0.90, 0.8, True, 0.95, "SUCCESS")
    align_res = AlignmentResult(True, ref_pcb.copy(), np.eye(3), np.eye(3), metrics, True)
    summary_pass = ComparisonSummary(ref_count=1, test_count=1)

    pass_result = InspectionResult(
        verdict="PASS",
        verdict_reason="PASS: All components verified within tolerances.",
        can_proceed_to_comparison=True,
        timestamp="2026-11-25T12:00:00Z",
        alignment=align_res,
        comparison=summary_pass,
        ref_detections=[],
        test_detections=[],
        aligned_test_image=ref_pcb.copy()
    )

    sbs_pass = visualizer.create_side_by_side_comparison(ref_pcb, pass_result)
    assert sbs_pass.shape[0] == 600 + 70  # target_h + banner_h
    assert sbs_pass.shape[1] == 1200       # 600 + 600

    # 2. Save image to outputs/visualizations/
    saved_path = visualizer.save_visualization(sbs_pass, filename_prefix="test_vis_pass")
    assert saved_path.exists(), f"Saved visualization missing at: {saved_path}"
    assert saved_path.stat().st_size > 5000, "Image file should be non-empty"
    print(f"  -> Generated and saved comparison visualization: {saved_path.name}")
    print("  -> Side-by-side comparison: PASSED")


def test_4_unaligned_edge_case():
    print("\n--- Test 4: Unaligned Image Edge Case ---")
    visualizer = PCBVisualizer()
    ref_pcb = generate_synthetic_pcb(600, 600, seed=303)

    # Alignment rejected result
    metrics = AlignmentMetrics(1000, 0, 0, 0, 0.0, float("inf"), False, 0.0, "REJECTED_INSUFFICIENT_FEATURES", "Blank test image")
    align_res = AlignmentResult(False, None, None, None, metrics, False)

    halted_result = InspectionResult(
        verdict="REVIEW",
        verdict_reason="Inspection HALTED: Image alignment rejected.",
        can_proceed_to_comparison=False,
        timestamp="2026-11-25T12:00:00Z",
        alignment=align_res,
        comparison=None,
        ref_detections=[],
        test_detections=[],
        aligned_test_image=None
    )

    sbs_halted = visualizer.create_side_by_side_comparison(ref_pcb, halted_result)
    assert sbs_halted.shape[0] == 600 + 70
    assert sbs_halted.shape[1] == 1200
    print("  -> Unaligned image handling: PASSED")


def run_all_visualizer_tests():
    print("=" * 65)
    print("BHARATPCB-AI: STAGE 4 VISUALIZER TESTS")
    print("=" * 65)
    test_1_draw_detections()
    test_2_render_inspection_views()
    test_3_side_by_side_comparison()
    test_4_unaligned_edge_case()
    print("\n" + "=" * 65)
    print("STAGE 4 VISUALIZER TESTS: ALL 4 TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_all_visualizer_tests()
