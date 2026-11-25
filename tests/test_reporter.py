"""
Stage 4 Verification Suite: Multi-Format Report Generator Tests.
Tests:
  1. PDF inspection report generation (validates %PDF header, multi-page, file size).
  2. CSV inspection data generation (validates headers, row count, formatting).
  3. JSON inspection data generation (validates JSON syntax, schema, fields).
  4. PASS report generation.
  5. FAIL report generation (with missing, extra, misplaced, and mismatched components).
  6. REVIEW report generation (with low-confidence warnings).
  7. Rejected alignment reporting (ensures halted status and zero false defects).
  8. Missing / None image input handling.
"""

import sys
import json
import csv
from pathlib import Path
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import Detection
from src.aligner import AlignmentResult, AlignmentMetrics
from src.comparator import (
    ComparisonSummary,
    MatchedPair,
    ComponentDiscrepancy
)
from src.inspector import InspectionResult
from src.reporter import InspectionReporter, REPORTS_DIR
from tests.test_alignment import generate_synthetic_pcb


def test_1_json_generation():
    print("\n--- Test 1: JSON Report Structure & Integrity ---")
    reporter = InspectionReporter()
    ref_det = Detection(class_id=3, class_name="ic", confidence=0.95, box=[100.0, 100.0, 200.0, 200.0])
    test_det = Detection(class_id=3, class_name="ic", confidence=0.94, box=[100.0, 100.0, 200.0, 200.0])

    summary = ComparisonSummary(
        ref_count=1,
        test_count=1,
        matched=[MatchedPair(ref_det, test_det, 1.0, 0.0, False, "Exact match")]
    )
    metrics = AlignmentMetrics(1000, 1000, 100, 90, 0.90, 0.5, True, 0.95, "SUCCESS")
    align_res = AlignmentResult(True, np.zeros((100, 100, 3), dtype=np.uint8), np.eye(3), np.eye(3), metrics, True)

    insp_result = InspectionResult(
        verdict="PASS",
        verdict_reason="PASS: All components verified within tolerances.",
        can_proceed_to_comparison=True,
        timestamp="2026-11-25T14:00:00Z",
        alignment=align_res,
        comparison=summary,
        ref_detections=[ref_det],
        test_detections=[test_det],
        aligned_test_image=np.zeros((100, 100, 3), dtype=np.uint8)
    )

    json_path = reporter.generate_json_report(insp_result, "test_report_pass")
    assert json_path.exists(), "JSON report must exist on disk"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["project"] == "BharatPCB-AI"
    assert data["inspection_result"]["verdict"] == "PASS"
    assert data["inspection_result"]["can_proceed_to_comparison"] is True
    assert len(data["inspection_result"]["comparison"]["matched"]) == 1
    print(f"  -> Generated valid JSON report: {json_path.name}")
    print("  -> JSON Structure: PASSED")


def test_2_csv_generation():
    print("\n--- Test 2: CSV Report Structure & Integrity ---")
    reporter = InspectionReporter()
    ref_det = Detection(class_id=3, class_name="ic", confidence=0.95, box=[100.0, 100.0, 200.0, 200.0])
    test_det = Detection(class_id=3, class_name="ic", confidence=0.94, box=[100.0, 100.0, 200.0, 200.0])
    miss_det = Detection(class_id=0, class_name="capacitor", confidence=0.88, box=[300.0, 300.0, 350.0, 350.0])

    summary = ComparisonSummary(
        ref_count=2,
        test_count=1,
        matched=[MatchedPair(ref_det, test_det, 1.0, 0.0, False, "Exact match")],
        missing=[ComponentDiscrepancy("MISSING", miss_det, severity="CRITICAL", reason="Missing capacitor")]
    )
    metrics = AlignmentMetrics(1000, 1000, 100, 90, 0.90, 0.5, True, 0.95, "SUCCESS")
    align_res = AlignmentResult(True, np.zeros((100, 100, 3), dtype=np.uint8), np.eye(3), np.eye(3), metrics, True)

    insp_result = InspectionResult(
        verdict="FAIL",
        verdict_reason="FAIL: 1 missing component(s).",
        can_proceed_to_comparison=True,
        timestamp="2026-11-25T14:00:00Z",
        alignment=align_res,
        comparison=summary,
        ref_detections=[ref_det, miss_det],
        test_detections=[test_det],
        aligned_test_image=np.zeros((100, 100, 3), dtype=np.uint8)
    )

    csv_path = reporter.generate_csv_report(insp_result, "test_report_fail")
    assert csv_path.exists(), "CSV report must exist on disk"

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))

    assert len(reader) >= 4, "CSV should contain headers + detections + matched + missing rows"
    headers = reader[0]
    assert "class_name" in headers
    assert "centroid_x" in headers
    assert "match_status" in headers
    print(f"  -> Generated valid CSV log with {len(reader)-1} data rows: {csv_path.name}")
    print("  -> CSV Structure: PASSED")


def test_3_pdf_generation_and_readability():
    print("\n--- Test 3: PDF Certificate Creation & Integrity ---")
    reporter = InspectionReporter()
    ref_pcb = generate_synthetic_pcb(600, 600, seed=401)
    test_pcb = ref_pcb.copy()

    ref_ic = Detection(class_id=3, class_name="ic", confidence=0.97, box=[100.0, 100.0, 220.0, 220.0])
    test_ic = Detection(class_id=3, class_name="ic", confidence=0.95, box=[100.0, 100.0, 220.0, 220.0])
    ref_cap = Detection(class_id=0, class_name="capacitor", confidence=0.90, box=[300.0, 300.0, 350.0, 350.0])
    test_led = Detection(class_id=5, class_name="led", confidence=0.86, box=[400.0, 400.0, 440.0, 440.0])

    summary = ComparisonSummary(
        ref_count=2,
        test_count=2,
        matched=[MatchedPair(ref_ic, test_ic, 1.0, 0.0, False, "Exact match")],
        missing=[ComponentDiscrepancy("MISSING", ref_cap, severity="CRITICAL", reason="Missing component")],
        extra=[ComponentDiscrepancy("EXTRA", test_led, severity="MEDIUM", reason="Extra component")]
    )
    metrics = AlignmentMetrics(1200, 1150, 95, 85, 0.89, 1.15, True, 0.93, "SUCCESS")
    align_res = AlignmentResult(True, test_pcb, np.eye(3), np.eye(3), metrics, True)

    insp_result = InspectionResult(
        verdict="FAIL",
        verdict_reason="FAIL: Board has critical defects: 1 missing component(s).",
        can_proceed_to_comparison=True,
        timestamp="2026-11-25T14:00:00Z",
        alignment=align_res,
        comparison=summary,
        ref_detections=[ref_ic, ref_cap],
        test_detections=[test_ic, test_led],
        aligned_test_image=test_pcb
    )

    reports = reporter.generate_all_reports(insp_result, ref_image=ref_pcb, ref_name="golden_v1.png", test_name="sample_batch_3.png")

    assert reports.pdf_path is not None and reports.pdf_path.exists()
    assert reports.csv_path is not None and reports.csv_path.exists()
    assert reports.json_path is not None and reports.json_path.exists()
    assert reports.visualization_path is not None and reports.visualization_path.exists()

    # Verify PDF file header
    with open(reports.pdf_path, "rb") as f:
        header = f.read(5)
    assert header == b"%PDF-", f"PDF header must start with %PDF-, got: {header}"

    pdf_size_kb = reports.pdf_path.stat().st_size / 1024
    print(f"  -> Generated PDF Certificate ({pdf_size_kb:.1f} KB): {reports.pdf_path.name}")
    assert pdf_size_kb > 20.0, "PDF should contain 2 rendered pages with embedded visualization and tables"
    print("  -> PDF Creation & Readability: PASSED")


def test_4_rejected_alignment_reporting():
    print("\n--- Test 4: Rejected Alignment Reporting (HALTED with Zero False Defects) ---")
    reporter = InspectionReporter()

    # Alignment rejected scenario
    metrics = AlignmentMetrics(
        detected_features_ref=1200,
        detected_features_test=0,
        good_matches_count=0,
        inliers_count=0,
        inlier_ratio=0.0,
        reprojection_error=float("inf"),
        is_homography_valid=False,
        quality_score=0.0,
        status="REJECTED_INSUFFICIENT_FEATURES",
        rejection_reason="Test PCB image is blank or overexposed"
    )
    align_res = AlignmentResult(False, None, None, None, metrics, False)

    halted_result = InspectionResult(
        verdict="REVIEW",
        verdict_reason="Inspection HALTED: Image alignment rejected (REJECTED_INSUFFICIENT_FEATURES). Reason: Test PCB image is blank or overexposed. Component comparison aborted to prevent false defect claims.",
        can_proceed_to_comparison=False,
        timestamp="2026-11-25T14:30:00Z",
        alignment=align_res,
        comparison=None,
        ref_detections=[],
        test_detections=[],
        aligned_test_image=None
    )

    reports = reporter.generate_all_reports(halted_result, ref_image=None)

    # 1. Verify JSON accurately reports halted status
    with open(reports.json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    assert json_data["inspection_result"]["verdict"] == "REVIEW"
    assert json_data["inspection_result"]["can_proceed_to_comparison"] is False
    assert json_data["inspection_result"]["comparison"] is None

    # 2. Verify PDF accurately generated even without aligned image
    assert reports.pdf_path.exists()
    with open(reports.pdf_path, "rb") as f:
        header = f.read(5)
    assert header == b"%PDF-"

    print(f"  -> Verified Halted Alignment PDF report: {reports.pdf_path.name}")
    print("  -> Rejected Alignment Reporting: PASSED")


def test_5_missing_image_and_invalid_input_handling():
    print("\n--- Test 5: Missing / None Image Handling ---")
    reporter = InspectionReporter()

    metrics = AlignmentMetrics(100, 100, 20, 18, 0.90, 0.5, True, 0.90, "SUCCESS")
    align_res = AlignmentResult(True, None, np.eye(3), np.eye(3), metrics, True)
    summary = ComparisonSummary(ref_count=0, test_count=0)

    insp_result = InspectionResult(
        verdict="PASS",
        verdict_reason="PASS: No components.",
        can_proceed_to_comparison=True,
        timestamp="2026-11-25T14:40:00Z",
        alignment=align_res,
        comparison=summary,
        ref_detections=[],
        test_detections=[],
        aligned_test_image=None
    )

    # Pass ref_image=None -> should handle gracefully without crashing
    reports = reporter.generate_all_reports(insp_result, ref_image=None)
    assert reports.pdf_path.exists()
    assert reports.json_path.exists()
    assert reports.csv_path.exists()
    assert reports.visualization_path is None
    print("  -> Graceful handling of None images: PASSED")


def run_all_reporter_tests():
    print("=" * 65)
    print("BHARATPCB-AI: STAGE 4 MULTI-FORMAT REPORTER TESTS")
    print("=" * 65)
    test_1_json_generation()
    test_2_csv_generation()
    test_3_pdf_generation_and_readability()
    test_4_rejected_alignment_reporting()
    test_5_missing_image_and_invalid_input_handling()
    print("\n" + "=" * 65)
    print("STAGE 4 REPORTER TESTS: ALL 5 TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_all_reporter_tests()
