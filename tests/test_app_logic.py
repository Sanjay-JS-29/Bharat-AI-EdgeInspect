"""
Stage 5 Verification Suite: Application Workflow Logic Tests.
Tests:
  1. Reference vs. Test inspection workflow & download payload preparation.
  2. Single PCB standalone component detection workflow.
  3. Batch PCB inspection workflow with aggregate metrics computation.
  4. App-level alignment rejection safety gate enforcement.
"""

import sys
from pathlib import Path
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import PCBDetector
from src.inspector import PCBInspector, InspectionResult
from src.visualizer import PCBVisualizer
from src.reporter import InspectionReporter
from src.comparator import ComparisonConfig
from tests.test_alignment import generate_synthetic_pcb


def test_1_ref_vs_test_app_flow():
    print("\n--- Test 1: Reference vs. Test Inspection Flow & Download Payloads ---")
    inspector = PCBInspector()
    reporter = InspectionReporter()
    visualizer = PCBVisualizer()

    ref_pcb = generate_synthetic_pcb(640, 640, seed=501)
    test_pcb = ref_pcb.copy()

    # 1. Run inspection
    result: InspectionResult = inspector.inspect(ref_pcb, test_pcb)
    assert result.can_proceed_to_comparison is True
    assert result.verdict in {"PASS", "FAIL", "REVIEW"}

    # 2. Render visualization
    vis_img = visualizer.create_side_by_side_comparison(ref_pcb, result)
    assert vis_img is not None
    assert vis_img.shape[0] == 640 + 70

    # 3. Generate report files
    reports = reporter.generate_all_reports(
        result,
        ref_image=ref_pcb,
        ref_name="golden_board_v1.png",
        test_name="batch_01_unit_04.png"
    )

    # 4. Verify download bytes preparation (as required by st.download_button)
    with open(reports.pdf_path, "rb") as f:
        pdf_bytes = f.read()
    with open(reports.csv_path, "rb") as f:
        csv_bytes = f.read()
    with open(reports.json_path, "rb") as f:
        json_bytes = f.read()

    assert len(pdf_bytes) > 20000, "PDF download payload must be valid"
    assert len(csv_bytes) > 50, "CSV download payload must be valid"
    assert len(json_bytes) > 100, "JSON download payload must be valid"

    print("  -> Ref vs Test inspection and download payload preparation: PASSED")


def test_2_single_pcb_detection_flow():
    print("\n--- Test 2: Single PCB Standalone Detection Flow ---")
    detector = PCBDetector()
    visualizer = PCBVisualizer()

    test_pcb = generate_synthetic_pcb(640, 640, seed=502)
    detections = detector.predict(test_pcb, conf=0.10)

    # Class breakdown
    counts = {}
    for d in detections:
        counts[d.class_name] = counts.get(d.class_name, 0) + 1

    annotated = visualizer.draw_detections(test_pcb, detections, title="STANDALONE DETECTION")
    assert annotated.shape == test_pcb.shape
    print(f"  -> Single board detections count: {len(detections)}")
    print(f"  -> Class counts: {counts}")
    print("  -> Single PCB detection workflow: PASSED")


def test_3_batch_inspection_flow():
    print("\n--- Test 3: Batch PCB Inspection Workflow ---")
    inspector = PCBInspector()
    ref_pcb = generate_synthetic_pcb(600, 600, seed=503)

    # 3 sample test boards:
    # Board 1: Identical (PASS)
    test_1 = ref_pcb.copy()
    # Board 2: Shifted slightly (PASS / Shift guard)
    test_2 = ref_pcb.copy()
    test_2 = np.roll(test_2, shift=5, axis=0)
    # Board 3: Blank board (Rejected alignment -> REVIEW)
    test_3 = np.full((600, 600, 3), (25, 95, 25), dtype=np.uint8)

    batch_items = [
        ("PCB_SN_001", test_1),
        ("PCB_SN_002", test_2),
        ("PCB_SN_003", test_3)
    ]

    batch_results = []
    verdict_counts = {"PASS": 0, "FAIL": 0, "REVIEW": 0}

    for sn, img in batch_items:
        res = inspector.inspect(ref_pcb, img)
        verdict_counts[res.verdict] = verdict_counts.get(res.verdict, 0) + 1
        batch_results.append({
            "serial_number": sn,
            "verdict": res.verdict,
            "can_proceed": res.can_proceed_to_comparison,
            "defects_count": res.total_defects,
            "reason": res.verdict_reason
        })

    assert len(batch_results) == 3
    assert batch_results[2]["verdict"] == "REVIEW", "Blank board must be REVIEW"
    assert batch_results[2]["defects_count"] == 0, "Unaligned board must have 0 defects"

    total = len(batch_results)
    pass_rate = (verdict_counts["PASS"] / total) * 100
    print(f"  -> Batch Summary: Total={total}, PASS={verdict_counts['PASS']}, FAIL={verdict_counts['FAIL']}, REVIEW={verdict_counts['REVIEW']}")
    print(f"  -> Pass Rate: {pass_rate:.1f}%")
    print("  -> Batch inspection logic: PASSED")


def test_4_app_alignment_rejection_safety():
    print("\n--- Test 4: App Alignment Rejection Safety ---")
    inspector = PCBInspector()
    ref_pcb = generate_synthetic_pcb(600, 600, seed=504)
    corrupted_test = np.random.randint(0, 255, (600, 600, 3), dtype=np.uint8)

    res = inspector.inspect(ref_pcb, corrupted_test)
    assert res.can_proceed_to_comparison is False
    assert res.verdict == "REVIEW"
    assert res.comparison is None
    assert "HALTED" in res.verdict_reason
    print("  -> App-level safety gate strictly blocks false defect generation: PASSED")


def run_all_app_logic_tests():
    print("=" * 65)
    print("BHARATPCB-AI: STAGE 5 APP LOGIC TESTS")
    print("=" * 65)
    test_1_ref_vs_test_app_flow()
    test_2_single_pcb_detection_flow()
    test_3_batch_inspection_flow()
    test_4_app_alignment_rejection_safety()
    print("\n" + "=" * 65)
    print("STAGE 5 APP LOGIC TESTS: ALL 4 TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_all_app_logic_tests()
