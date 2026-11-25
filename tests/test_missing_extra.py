"""
Stage 3 Verification Suite: Missing, Extra & Alignment Safety Tests.
Tests:
  1. Missing component detection.
  2. Extra uncataloged component detection.
  3. Mixed board inspection (Matched + Missing + Extra + Misplaced).
  4. CRITICAL SAFETY GATE: Inspection HALTED on rejected alignment.
     Verifies ZERO false missing/extra defect claims when alignment fails.
"""

import sys
from pathlib import Path
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import Detection, PCBDetector
from src.aligner import PCBAligner
from src.comparator import ComponentComparator, ComparisonConfig
from src.inspector import PCBInspector, InspectionResult


def test_1_missing_component():
    print("\n--- Test 1: Missing Component Detection ---")
    comparator = ComponentComparator()
    ref_dets = [
        Detection(class_id=3, class_name="ic", confidence=0.96, box=[100.0, 100.0, 200.0, 200.0]),
        Detection(class_id=0, class_name="capacitor", confidence=0.89, box=[300.0, 300.0, 350.0, 350.0])
    ]
    # Test board only has the IC; the capacitor is missing!
    test_dets = [
        Detection(class_id=3, class_name="ic", confidence=0.95, box=[100.0, 100.0, 200.0, 200.0])
    ]

    summary = comparator.compare(ref_dets, test_dets)
    print(f"  Matched: {len(summary.matched)}")
    print(f"  Missing: {len(summary.missing)}")
    if summary.missing:
        print(f"  Missing Reason: {summary.missing[0].reason}")
        print(f"  Missing Severity: {summary.missing[0].severity}")

    assert len(summary.matched) == 1, "IC must be matched"
    assert len(summary.missing) == 1, "Must detect 1 missing capacitor"
    assert summary.missing[0].component.class_name == "capacitor"
    assert summary.missing[0].severity == "CRITICAL"
    print("  -> TEST 1 PASSED")


def test_2_extra_component():
    print("\n--- Test 2: Extra Component Detection ---")
    comparator = ComponentComparator()
    ref_dets = [
        Detection(class_id=3, class_name="ic", confidence=0.96, box=[100.0, 100.0, 200.0, 200.0])
    ]
    # Test board has the IC PLUS an extra uncataloged LED
    test_dets = [
        Detection(class_id=3, class_name="ic", confidence=0.95, box=[100.0, 100.0, 200.0, 200.0]),
        Detection(class_id=5, class_name="led", confidence=0.87, box=[500.0, 500.0, 540.0, 540.0])
    ]

    summary = comparator.compare(ref_dets, test_dets)
    print(f"  Matched: {len(summary.matched)}")
    print(f"  Extra: {len(summary.extra)}")
    if summary.extra:
        print(f"  Extra Reason: {summary.extra[0].reason}")

    assert len(summary.matched) == 1
    assert len(summary.extra) == 1, "Must detect 1 extra LED"
    assert summary.extra[0].component.class_name == "led"
    print("  -> TEST 2 PASSED")


def test_3_mixed_board_scenario():
    print("\n--- Test 3: Mixed Board Scenario ---")
    comparator = ComponentComparator(ComparisonConfig(max_centroid_drift=25.0, spatial_search_radius=70.0))

    ref_dets = [
        Detection(class_id=3, class_name="ic", confidence=0.97, box=[100.0, 100.0, 200.0, 200.0]),       # Expected Match
        Detection(class_id=6, class_name="resistor", confidence=0.92, box=[300.0, 100.0, 340.0, 120.0]), # Expected Match
        Detection(class_id=0, class_name="capacitor", confidence=0.91, box=[400.0, 400.0, 450.0, 450.0]),# Expected Missing
        Detection(class_id=4, class_name="inductor", confidence=0.89, box=[500.0, 200.0, 560.0, 260.0])  # Expected Misplaced
    ]

    test_dets = [
        Detection(class_id=3, class_name="ic", confidence=0.96, box=[100.0, 100.0, 200.0, 200.0]),       # Matched IC
        Detection(class_id=6, class_name="resistor", confidence=0.90, box=[302.0, 101.0, 342.0, 121.0]), # Matched Resistor (jitter)
        Detection(class_id=4, class_name="inductor", confidence=0.88, box=[545.0, 200.0, 605.0, 260.0]), # Misplaced Inductor (shift 45px)
        Detection(class_id=7, class_name="switch", confidence=0.86, box=[50.0, 500.0, 110.0, 560.0])     # Extra Switch
    ]

    summary = comparator.compare(ref_dets, test_dets)
    print(f"  Matched: {len(summary.matched)}")
    print(f"  Missing: {len(summary.missing)}")
    print(f"  Misplaced: {len(summary.misplaced)}")
    print(f"  Extra: {len(summary.extra)}")
    print(f"  Total Discrepancies: {summary.total_discrepancies}")

    assert len(summary.matched) == 2, "Should match IC and Resistor"
    assert len(summary.missing) == 1, "Capacitor should be missing"
    assert len(summary.misplaced) == 1, "Inductor should be misplaced"
    assert len(summary.extra) == 1, "Switch should be extra"
    assert summary.total_discrepancies == 3

    # Check inspector verdict logic
    inspector = PCBInspector(comparator=comparator)
    verdict, reason = inspector._determine_verdict(summary)
    print(f"  Inspector Verdict: {verdict}")
    print(f"  Inspector Reason: {reason}")
    assert verdict == "FAIL", "Critical defects (missing + misplaced) must cause FAIL verdict"
    print("  -> TEST 3 PASSED")


def test_4_safety_gate_on_rejected_alignment():
    print("\n--- Test 4: CRITICAL SAFETY GATE: Rejected Alignment Halts Inspection ---")
    inspector = PCBInspector()

    # Generate synthetic reference board
    from tests.test_alignment import generate_synthetic_pcb
    ref_pcb = generate_synthetic_pcb(640, 640, seed=200)

    # Completely blank test image (unalignable, zero features)
    blank_test_pcb = np.full((640, 640, 3), (25, 95, 25), dtype=np.uint8)

    # Run inspection
    result: InspectionResult = inspector.inspect(ref_pcb, blank_test_pcb)

    print(f"  Verdict: {result.verdict}")
    print(f"  Verdict Reason: {result.verdict_reason}")
    print(f"  Can Proceed To Comparison: {result.can_proceed_to_comparison}")
    print(f"  Total Defects Reported: {result.total_defects}")

    assert result.can_proceed_to_comparison is False, "Must NOT proceed to comparison"
    assert result.comparison is None, "Comparison summary must be None"
    assert result.total_defects == 0, "CRITICAL: Must report ZERO defects on unaligned board"
    assert result.verdict == "REVIEW", "Must set verdict to REVIEW"
    assert "HALTED" in result.verdict_reason, "Reason must explain inspection was halted"
    print("  -> TEST 4 PASSED")


def run_all_missing_extra_tests():
    print("=" * 65)
    print("BHARATPCB-AI: STAGE 3 MISSING, EXTRA & SAFETY GATE TESTS")
    print("=" * 65)
    test_1_missing_component()
    test_2_extra_component()
    test_3_mixed_board_scenario()
    test_4_safety_gate_on_rejected_alignment()
    print("\n" + "=" * 65)
    print("STAGE 3 MISSING/EXTRA TESTS: ALL 4 TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_all_missing_extra_tests()
