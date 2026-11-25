"""
Stage 2 Verification Suite: OpenCV Image Alignment Engine.
Tests:
  1. Identical PCB image alignment.
  2. Rotated PCB image alignment.
  3. Perspective-transformed PCB image alignment.
  4. Insufficient feature rejection.
  5. Degenerate homography rejection & comparison blocking.
  6. Quality threshold enforcement.
  7. Manual fiducial alignment fallback.

NOTE: Uses high-entropy synthetic PCB images with solder pads, traces, and IC chips.
Clearly distinguishes synthetic test success from real PCB validation.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aligner import PCBAligner, AlignmentResult, AlignmentMetrics


def generate_synthetic_pcb(width: int = 800, height: int = 800, seed: int = 42) -> np.ndarray:
    """Generate a realistic high-entropy synthetic PCB board with traces, pads, and ICs."""
    np.random.seed(seed)
    # Green solder mask background
    img = np.full((height, width, 3), (25, 95, 25), dtype=np.uint8)

    # 1. Copper traces (lines)
    for _ in range(120):
        pt1 = (np.random.randint(20, width - 20), np.random.randint(20, height - 20))
        # 45-degree or orthogonal routing
        dx = np.random.choice([-100, -50, 0, 50, 100])
        dy = np.random.choice([-100, -50, 0, 50, 100])
        pt2 = (np.clip(pt1[0] + dx, 10, width - 10), np.clip(pt1[1] + dy, 10, height - 10))
        color = (15, 140, 45)  # Copper under mask
        cv2.line(img, pt1, pt2, color, thickness=np.random.choice([2, 3]))

    # 2. Solder pads (gold/silver circles and rectangles)
    for _ in range(250):
        cx = np.random.randint(30, width - 30)
        cy = np.random.randint(30, height - 30)
        color = (192, 192, 192) if np.random.rand() > 0.3 else (0, 215, 255)
        if np.random.rand() > 0.5:
            cv2.circle(img, (cx, cy), np.random.randint(3, 8), color, -1)
        else:
            w, h = np.random.randint(6, 16), np.random.randint(6, 16)
            cv2.rectangle(img, (cx - w//2, cy - h//2), (cx + w//2, cy + h//2), color, -1)

    # 3. IC Chips (Black packages with white silkscreen and silver pins)
    ic_positions = [(180, 180, 120, 120), (520, 220, 140, 100), (320, 500, 160, 140)]
    for (ix, iy, iw, ih) in ic_positions:
        # Silkscreen outline
        cv2.rectangle(img, (ix - 5, iy - 5), (ix + iw + 5, iy + ih + 5), (240, 240, 240), 1)
        # IC Body
        cv2.rectangle(img, (ix, iy), (ix + iw, iy + ih), (20, 20, 20), -1)
        # IC Pin 1 indicator notch
        cv2.circle(img, (ix + 12, iy + 12), 4, (100, 100, 100), -1)
        # Pins on sides
        pin_spacing = 10
        for px in range(ix + 10, ix + iw - 10, pin_spacing):
            cv2.line(img, (px, iy - 6), (px, iy), (210, 210, 210), 2)
            cv2.line(img, (px, iy + ih), (px, iy + ih + 6), (210, 210, 210), 2)
        for py in range(iy + 10, iy + ih - 10, pin_spacing):
            cv2.line(img, (ix - 6, py), (ix, py), (210, 210, 210), 2)
            cv2.line(img, (ix + iw, py), (ix + iw + 6, py), (210, 210, 210), 2)

    # 4. Silkscreen reference designators (text)
    labels = ["U1", "U2", "U3", "C1", "C2", "R1", "R2", "D1", "L1", "GND", "VCC", "BHARAT-PCB"]
    for i, lbl in enumerate(labels):
        tx = 60 + (i * 55) % (width - 120)
        ty = 50 + (i * 65) % (height - 80)
        cv2.putText(img, lbl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def test_1_identical_images():
    print("\n--- Test 1: Identical Images Alignment ---")
    aligner = PCBAligner(min_features=300, min_good_matches=15)
    ref = generate_synthetic_pcb(800, 800, seed=100)
    test = ref.copy()

    result = aligner.align(ref, test)
    print(f"  Status: {result.metrics.status}")
    print(f"  Inliers: {result.metrics.inliers_count}, Inlier Ratio: {result.metrics.inlier_ratio:.2%}")
    print(f"  Reprojection Error: {result.metrics.reprojection_error:.3f} px")
    print(f"  Can proceed to comparison: {result.can_proceed_to_comparison}")

    assert result.success, "Identical images alignment must succeed"
    assert result.can_proceed_to_comparison, "Must allow comparison"
    assert result.metrics.reprojection_error < 1.0, "Reprojection error should be near 0"
    print("  -> TEST 1 PASSED")


def test_2_rotated_images():
    print("\n--- Test 2: Rotated Test PCB (12 degrees) ---")
    aligner = PCBAligner(min_features=300, min_good_matches=15)
    ref = generate_synthetic_pcb(800, 800, seed=101)

    # Rotate test image by 12 degrees
    center = (400, 400)
    rot_mat = cv2.getRotationMatrix2D(center, 12.0, 1.0)
    test_rotated = cv2.warpAffine(ref, rot_mat, (800, 800), borderValue=(25, 95, 25))

    result = aligner.align(ref, test_rotated)
    print(f"  Status: {result.metrics.status}")
    print(f"  Inliers: {result.metrics.inliers_count}, Ratio: {result.metrics.inlier_ratio:.2%}")
    print(f"  Reprojection Error: {result.metrics.reprojection_error:.3f} px")
    print(f"  Quality Score: {result.metrics.quality_score:.3f}")
    print(f"  Can proceed to comparison: {result.can_proceed_to_comparison}")

    assert result.success, "Rotated image alignment must succeed"
    assert result.can_proceed_to_comparison, "Must allow comparison"
    assert result.metrics.is_homography_valid, "Homography must be valid"
    print("  -> TEST 2 PASSED")


def test_3_perspective_transformed_images():
    print("\n--- Test 3: Perspective-Distorted Test PCB ---")
    aligner = PCBAligner(min_features=300, min_good_matches=15)
    ref = generate_synthetic_pcb(800, 800, seed=102)

    # Perspective warp: tilt top edge slightly
    src_pts = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]])
    dst_pts = np.float32([[40, 20], [760, 25], [790, 785], [15, 780]])
    persp_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    test_persp = cv2.warpPerspective(ref, persp_mat, (800, 800), borderValue=(25, 95, 25))

    result = aligner.align(ref, test_persp)
    print(f"  Status: {result.metrics.status}")
    print(f"  Inliers: {result.metrics.inliers_count}, Ratio: {result.metrics.inlier_ratio:.2%}")
    print(f"  Reprojection Error: {result.metrics.reprojection_error:.3f} px")
    print(f"  Can proceed to comparison: {result.can_proceed_to_comparison}")

    assert result.success, "Perspective distorted alignment must succeed"
    assert result.can_proceed_to_comparison, "Must allow comparison"
    print("  -> TEST 3 PASSED")


def test_4_insufficient_features():
    print("\n--- Test 4: Insufficient Features Rejection ---")
    aligner = PCBAligner(min_features=500)
    ref = generate_synthetic_pcb(800, 800, seed=103)
    blank_test = np.full((800, 800, 3), (25, 95, 25), dtype=np.uint8)  # Flat blank green board

    result = aligner.align(ref, blank_test)
    print(f"  Status: {result.metrics.status}")
    print(f"  Rejection Reason: {result.metrics.rejection_reason}")
    print(f"  Can proceed to comparison: {result.can_proceed_to_comparison}")

    assert not result.success, "Blank board must be rejected"
    assert not result.can_proceed_to_comparison, "CRITICAL: Must block comparison on insufficient features"
    assert "INSUFFICIENT" in result.metrics.status
    print("  -> TEST 4 PASSED")


def test_5_invalid_homography_rejection():
    print("\n--- Test 5: Degenerate / Unrelated Image Rejection ---")
    aligner = PCBAligner(min_features=200, min_good_matches=15)
    ref = generate_synthetic_pcb(800, 800, seed=104)
    # Generate unrelated noise image
    noise_test = np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8)

    result = aligner.align(ref, noise_test)
    print(f"  Status: {result.metrics.status}")
    print(f"  Rejection Reason: {result.metrics.rejection_reason}")
    print(f"  Can proceed to comparison: {result.can_proceed_to_comparison}")

    assert not result.success, "Unrelated noisy image must be rejected"
    assert not result.can_proceed_to_comparison, "CRITICAL: Must block comparison on invalid homography"
    print("  -> TEST 5 PASSED")


def test_6_alignment_quality_thresholds():
    print("\n--- Test 6: Strict Quality Threshold Enforcement ---")
    # Setting unrealistically strict inlier ratio (95%)
    strict_aligner = PCBAligner(min_features=200, min_inlier_ratio=0.95)
    ref = generate_synthetic_pcb(800, 800, seed=105)
    rot_mat = cv2.getRotationMatrix2D((400, 400), 15.0, 1.0)
    test_rotated = cv2.warpAffine(ref, rot_mat, (800, 800), borderValue=(25, 95, 25))

    result = strict_aligner.align(ref, test_rotated)
    print(f"  Status: {result.metrics.status}")
    print(f"  Rejection Reason: {result.metrics.rejection_reason}")
    print(f"  Can proceed to comparison: {result.can_proceed_to_comparison}")

    assert not result.success, "Strict threshold violation must reject alignment"
    assert not result.can_proceed_to_comparison, "Must block comparison when quality threshold fails"
    print("  -> TEST 6 PASSED")


def test_7_manual_alignment_fallback():
    print("\n--- Test 7: Manual Alignment Fallback Interface ---")
    aligner = PCBAligner()
    ref = generate_synthetic_pcb(800, 800, seed=106)

    # Transform test image
    src_pts = np.float32([[50, 50], [750, 50], [750, 750], [50, 750]])
    dst_pts = np.float32([[70, 90], [730, 40], [770, 720], [30, 760]])
    persp_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    test_img = cv2.warpPerspective(ref, persp_mat, (800, 800))

    # User identifies corresponding fiducial coordinates
    ref_fiducials = [(50.0, 50.0), (750.0, 50.0), (750.0, 750.0), (50.0, 750.0)]
    test_fiducials = [(70.0, 90.0), (730.0, 40.0), (770.0, 720.0), (30.0, 760.0)]

    result = aligner.align_manual(ref, test_img, ref_fiducials, test_fiducials)
    print(f"  Status: {result.metrics.status}")
    print(f"  Quality Score: {result.metrics.quality_score}")
    print(f"  Can proceed to comparison: {result.can_proceed_to_comparison}")
    print(f"  Aligned image produced: {result.aligned_image is not None}")

    assert result.success, "Manual alignment with 4 fiducials must succeed"
    assert result.can_proceed_to_comparison, "Manual alignment must allow comparison"
    assert result.aligned_image is not None, "Must produce warped aligned image"
    print("  -> TEST 7 PASSED")


def run_all_stage2_tests():
    print("=" * 65)
    print("BHARATPCB-AI: STAGE 2 OPENCV IMAGE ALIGNMENT VERIFICATION")
    print("=" * 65)
    print("[NOTE] Automated tests use high-entropy synthetic PCB boards with")
    print("       copper traces, solder pads, IC packages, and silkscreen labels.")

    test_1_identical_images()
    test_2_rotated_images()
    test_3_perspective_transformed_images()
    test_4_insufficient_features()
    test_5_invalid_homography_rejection()
    test_6_alignment_quality_thresholds()
    test_7_manual_alignment_fallback()

    print("\n" + "=" * 65)
    print("STAGE 2 VERIFICATION RESULT: ALL 7 TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_all_stage2_tests()
