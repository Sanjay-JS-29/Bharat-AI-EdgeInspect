"""
Verification Suite for PCBAdvisor, Structural Heatmap, Defect Crops & public run_inspection.
Tests:
  1. PCBAdvisor initialization, status detection, and ExecuTorch/Llama absence reporting.
  2. Root-cause inspection advice generation across verdicts (PASS, FAIL, REVIEW).
  3. Interactive technician Q&A without hallucination or fake LLM claims.
  4. Public run_inspection() entry point verification.
  5. Structural difference heatmap generation.
  6. Defect crop region generation.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import UltralyticsDetector
from src.inspector import run_inspection, extract_defect_records, get_board_region, InspectionResult
from src.visualizer import PCBVisualizer
from src.advisor import PCBAdvisor
from tests.test_alignment import generate_synthetic_pcb


def test_1_advisor_status_and_detection():
    print("\n--- Test 1: Advisor Status & Runtime Detection ---")
    advisor = PCBAdvisor()
    status = advisor.status_details
    print(f"  Advisor Badge: {advisor.status_badge}")
    print(f"  Has ExecuTorch: {status['has_executorch_runtime']}")
    print(f"  Has Model Weights: {status['has_model_weights']}")
    print(f"  Active Engine: {status['engine']}")

    # Verify transparent reporting (no fake LLM)
    assert status["is_llama_active"] is False, "Llama must not be reported active when weights/runtime missing"
    assert "Rules" in status["engine"] or "Deterministic" in status["engine"]
    print("  -> Transparent advisor status detection: PASSED")


def test_2_advisor_root_cause_generation():
    print("\n--- Test 2: Advisor Root Cause & Rework Advice ---")
    advisor = PCBAdvisor()
    ref_pcb = generate_synthetic_pcb(600, 600, seed=801)
    test_pcb = ref_pcb.copy()

    # 1. Inspect identical board (PASS)
    res_pass = run_inspection(ref_pcb, test_pcb)
    advice_pass = advisor.generate_inspection_advice(res_pass)
    print(f"  PASS Headline: {advice_pass['headline']}")
    assert "Zero Discrepancies" in advice_pass["headline"] or "PASS" in res_pass.verdict

    # 2. Inspect blank board (HALTED / REVIEW)
    res_blank = run_inspection(ref_pcb, np.zeros((600, 600, 3), dtype=np.uint8))
    advice_blank = advisor.generate_inspection_advice(res_blank)
    print(f"  HALTED Headline: {advice_blank['headline']}")
    assert "Alignment Failure" in advice_blank["headline"]

    print("  -> Root-cause advice generation: PASSED")


def test_3_advisor_technician_qa():
    print("\n--- Test 3: Interactive Technician Q&A ---")
    advisor = PCBAdvisor()
    ref_pcb = generate_synthetic_pcb(600, 600, seed=802)
    test_pcb = ref_pcb.copy()
    res = run_inspection(ref_pcb, test_pcb)

    # Ask about Llama
    ans_llama = advisor.answer_question("Is Llama connected?", res)
    print(f"  Q: Is Llama connected? -> {ans_llama[:80]}...")
    assert "NOT installed" in ans_llama

    # Ask about rework
    ans_rework = advisor.answer_question("What rework steps should I take?", res)
    assert len(ans_rework) > 10

    print("  -> Technician Q&A: PASSED")


def test_4_public_run_inspection_and_heatmap():
    print("\n--- Test 4: Public run_inspection() & Structural Heatmap ---")
    visualizer = PCBVisualizer()
    ref_pcb = generate_synthetic_pcb(600, 600, seed=803)
    test_pcb = ref_pcb.copy()

    result = run_inspection(ref_pcb, test_pcb, backend="yolo")
    assert isinstance(result, InspectionResult)
    assert result.can_proceed_to_comparison is True

    heatmap = visualizer.compute_structural_difference_heatmap(ref_pcb, result.aligned_test_image)
    assert heatmap is not None
    assert heatmap.shape == ref_pcb.shape

    crops = visualizer.crop_defect_regions(ref_pcb, result.aligned_test_image, result)
    assert isinstance(crops, list)

    records = extract_defect_records(result, ref_pcb.shape[1], ref_pcb.shape[0])
    assert isinstance(records, list)

    print("  -> run_inspection(), heatmap & crop utilities: PASSED")


def run_all_advisor_tests():
    print("=" * 65)
    print("BHARATPCB-AI / EDGEINSPECT AI: ADVISOR & UTILITY TESTS")
    print("=" * 65)
    test_1_advisor_status_and_detection()
    test_2_advisor_root_cause_generation()
    test_3_advisor_technician_qa()
    test_4_public_run_inspection_and_heatmap()
    print("\n" + "=" * 65)
    print("ALL ADVISOR & UTILITY TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_all_advisor_tests()
