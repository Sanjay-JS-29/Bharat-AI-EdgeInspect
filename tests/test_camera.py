"""
Stage 5 Verification Suite: Camera & Image Input Helper Tests.
Tests:
  1. Image bytes decoding (PNG, JPEG, corrupt bytes, empty bytes).
  2. Physical dimension and channel validation.
  3. BGR to RGB color conversion for display.
  4. Camera capture error handling & fallback robustness.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera import (
    decode_image_bytes,
    validate_image,
    bgr_to_rgb,
    capture_frame_cv2,
    MIN_DIMENSION
)


def test_1_decode_image_bytes():
    print("\n--- Test 1: Image Bytes Decoding ---")
    # 1. Encode a synthetic image to PNG bytes
    test_img = np.full((300, 400, 3), (50, 150, 200), dtype=np.uint8)
    success, png_bytes = cv2.imencode(".png", test_img)
    assert success, "Encoding to PNG must succeed"

    decoded_img = decode_image_bytes(png_bytes.tobytes())
    assert decoded_img is not None, "Decoding valid PNG bytes must succeed"
    assert decoded_img.shape == (300, 400, 3)

    # 2. Encode to JPEG bytes
    success, jpg_bytes = cv2.imencode(".jpg", test_img)
    assert success
    decoded_jpg = decode_image_bytes(jpg_bytes.tobytes())
    assert decoded_jpg is not None
    assert decoded_jpg.shape == (300, 400, 3)

    # 3. Corrupt / empty bytes
    assert decode_image_bytes(b"") is None
    assert decode_image_bytes(b"corrupted_random_byte_data_not_an_image") is None
    print("  -> Image bytes decoding: PASSED")


def test_2_validate_image():
    print("\n--- Test 2: Image Dimension & Channel Validation ---")
    # Valid 640x640x3
    valid_img = np.full((640, 640, 3), 100, dtype=np.uint8)
    valid, msg = validate_image(valid_img)
    assert valid, f"Valid image should pass: {msg}"

    # None input
    valid, msg = validate_image(None)
    assert not valid
    assert "could not be decoded" in msg.lower()

    # Too small (< MIN_DIMENSION)
    tiny_img = np.full((MIN_DIMENSION - 20, MIN_DIMENSION - 20, 3), 100, dtype=np.uint8)
    valid, msg = validate_image(tiny_img)
    assert not valid
    assert "too small" in msg.lower()

    # Grayscale / 2-channel
    gray_img = np.full((400, 400), 100, dtype=np.uint8)
    valid, msg = validate_image(gray_img)
    assert not valid
    assert "3-channel" in msg.lower()

    print("  -> Image validation checks: PASSED")


def test_3_bgr_to_rgb():
    print("\n--- Test 3: BGR to RGB Color Conversion ---")
    # Pure Blue in BGR is (255, 0, 0)
    bgr_pixel = np.zeros((10, 10, 3), dtype=np.uint8)
    bgr_pixel[:, :] = (255, 0, 0)

    rgb_pixel = bgr_to_rgb(bgr_pixel)
    # Pure Blue in RGB is (0, 0, 255)
    assert rgb_pixel[0, 0, 0] == 0
    assert rgb_pixel[0, 0, 1] == 0
    assert rgb_pixel[0, 0, 2] == 255
    print("  -> BGR to RGB conversion: PASSED")


def test_4_camera_capture_fallback():
    print("\n--- Test 4: Camera Capture Fallback ---")
    # Camera index 99 (non-existent device) to test graceful failure
    success, frame, msg = capture_frame_cv2(camera_index=99)
    assert not success, "Non-existent camera index should fail gracefully"
    assert frame is None
    assert len(msg) > 0
    print(f"  -> Handled camera fallback gracefully: '{msg}'")
    print("  -> Camera capture error handling: PASSED")


def run_all_camera_tests():
    print("=" * 65)
    print("BHARATPCB-AI: STAGE 5 CAMERA HELPER TESTS")
    print("=" * 65)
    test_1_decode_image_bytes()
    test_2_validate_image()
    test_3_bgr_to_rgb()
    test_4_camera_capture_fallback()
    print("\n" + "=" * 65)
    print("STAGE 5 CAMERA TESTS: ALL 4 TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_all_camera_tests()
