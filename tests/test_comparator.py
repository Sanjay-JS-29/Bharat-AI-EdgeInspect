"""
Stage 3 Verification Suite: Component Comparator Tests.
Tests:
  1. Identical / exact component matching.
  2. Small detection shift / jitter tolerance (not classified as missing/extra).
  3. Misplaced component detection (large shift exceeding tolerance).
  4. Class mismatch detection (wrong component in place).
  5. Low-confidence detection auditing (flagged without false missing verdict).
  6. Custom spatial tolerance configuration.
"""

import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import Detection
from src.comparator import (
    ComponentComparator,
    ComparisonConfig,
    compute_iou,
    compute_centroid_distance
)


def test_1_exact_matching():
    print("\n--- Test 1: Exact Component Matching ---")
    comparator = ComponentComparator()
    ref_dets = [
        Detection(class_id=3, class_name="ic", confidence=0.95, box=[100.0, 100.0, 200.0, 200.0]),
        Detection(class_id=0, class_name="capacitor", confidence=0.88, box=[300.0, 300.0, 350.0, 350.0])
    ]
    test_dets = [
        Detection(class_id=3, class_name="ic", confidence=0.94, box=[100.0, 100.0, 200.0, 200.0]),
        Detection(class_id=0, class_name="capacitor", confidence=0.85, box=[300.0, 300.0, 350.0, 350.0])
    ]

    summary = comparator.compare(ref_dets, test_dets)
    print(f"  Matched: {len(summary.matched)}, Missing: {len(summary.missing)}, Extra: {len(summary.extra)}")
    assert len(summary.matched) == 2, "Both components should be matched"
    assert len(summary.missing) == 0, "No missing components expected"
    assert len(summary.extra) == 0, "No extra components expected"
    assert summary.matched[0].iou == 1.0, "Exact match IoU should be 1.0"
    print("  -> TEST 1 PASSED")


def test_2_shift_tolerance_guard():
    print("\n--- Test 2: Shift Tolerance Guard (Small Jitter Is NOT Missing/Extra) ---")
    # Jitter of 8 pixels (well within default 35.0px tolerance)
    comparator = ComponentComparator()
    ref_dets = [
        Detection(class_id=6, class_name="resistor", confidence=0.91, box=[150.0, 150.0, 190.0, 170.0])
    ]
    test_dets = [
        # Shifted by dx=6, dy=5 (~7.8px centroid drift)
        Detection(class_id=6, class_name="resistor", confidence=0.89, box=[156.0, 155.0, 196.0, 175.0])
    ]

    summary = comparator.compare(ref_dets, test_dets)
    print(f"  Centroid drift: {summary.matched[0].centroid_distance:.2f}px")
    print(f"  Matched: {len(summary.matched)}, Missing: {len(summary.missing)}, Extra: {len(summary.extra)}")

    assert len(summary.matched) == 1, "Slightly shifted component must match"
    assert not summary.matched[0].is_misplaced, "Shift within tolerance must not be flagged misplaced"
    assert len(summary.missing) == 0, "Shift must NOT be reported as missing"
    assert len(summary.extra) == 0, "Shift must NOT be reported as extra"
    print("  -> TEST 2 PASSED")


def test_3_misplaced_component_detection():
    print("\n--- Test 3: Misplaced Component (Shift Exceeds Tolerance) ---")
    config = ComparisonConfig(max_centroid_drift=20.0, spatial_search_radius=60.0)
    comparator = ComponentComparator(config)
    ref_dets = [
        Detection(class_id=4, class_name="inductor", confidence=0.92, box=[200.0, 200.0, 260.0, 260.0])
    ]
    # Shifted by 45 pixels (> 20.0px max drift, but within 60.0px search radius)
    test_dets = [
        Detection(class_id=4, class_name="inductor", confidence=0.90, box=[245.0, 200.0, 305.0, 260.0])
    ]

    summary = comparator.compare(ref_dets, test_dets)
    print(f"  Misplaced count: {len(summary.misplaced)}")
    if summary.misplaced:
        print(f"  Reason: {summary.misplaced[0].reason}")
        print(f"  Centroid drift: {summary.misplaced[0].centroid_drift:.2f}px")

    assert len(summary.misplaced) == 1, "Must detect 1 misplaced component"
    assert len(summary.missing) == 0, "Must not be classified as missing"
    assert len(summary.extra) == 0, "Must not be classified as extra"
    assert summary.misplaced[0].discrepancy_type == "MISPLACED"
    print("  -> TEST 3 PASSED")


def test_4_class_mismatch():
    print("\n--- Test 4: Component Class Mismatch Detection ---")
    comparator = ComponentComparator()
    ref_dets = [
        # Expected Capacitor at [400, 400]
        Detection(class_id=0, class_name="capacitor", confidence=0.93, box=[400.0, 400.0, 460.0, 460.0])
    ]
    test_dets = [
        # Detected Resistor at same location
        Detection(class_id=6, class_name="resistor", confidence=0.91, box=[402.0, 398.0, 462.0, 458.0])
    ]

    summary = comparator.compare(ref_dets, test_dets)
    print(f"  Class mismatches: {len(summary.class_mismatches)}")
    if summary.class_mismatches:
        print(f"  Reason: {summary.class_mismatches[0].reason}")

    assert len(summary.class_mismatches) == 1, "Must detect 1 class mismatch"
    assert summary.class_mismatches[0].severity == "CRITICAL"
    assert len(summary.missing) == 0
    assert len(summary.extra) == 0
    print("  -> TEST 4 PASSED")


def test_5_low_confidence_handling():
    print("\n--- Test 5: Low-Confidence Detection Auditing ---")
    config = ComparisonConfig(min_confidence_threshold=0.50)
    comparator = ComponentComparator(config)
    ref_dets = [
        Detection(class_id=2, class_name="diode", confidence=0.90, box=[100.0, 100.0, 130.0, 130.0])
    ]
    test_dets = [
        # Detected but with borderline confidence 0.38 (< 0.50)
        Detection(class_id=2, class_name="diode", confidence=0.38, box=[100.0, 100.0, 130.0, 130.0])
    ]

    summary = comparator.compare(ref_dets, test_dets)
    print(f"  Matched: {len(summary.matched)}")
    print(f"  Low confidence warnings: {len(summary.low_confidence_warnings)}")

    assert len(summary.matched) == 1, "Low confidence detection is still spatially matched"
    assert len(summary.low_confidence_warnings) == 1, "Must be flagged in low_confidence_warnings"
    assert len(summary.missing) == 0, "CRITICAL: Low confidence detection must NOT be claimed as definitely missing!"
    print("  -> TEST 5 PASSED")


def run_all_comparator_tests():
    print("=" * 65)
    print("BHARATPCB-AI: STAGE 3 COMPONENT COMPARATOR TESTS")
    print("=" * 65)
    test_1_exact_matching()
    test_2_shift_tolerance_guard()
    test_3_misplaced_component_detection()
    test_4_class_mismatch()
    test_5_low_confidence_handling()
    print("\n" + "=" * 65)
    print("STAGE 3 COMPARATOR TESTS: ALL 5 TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_all_comparator_tests()
