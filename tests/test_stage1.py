"""
Stage 1 verification test: Verify PCBDetector initialization, device selection, and inference.
"""

import sys
import time
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_PATH, CLASSES
from src.detector import PCBDetector, Detection


def run_stage1_tests():
    print("=== STAGE 1 VERIFICATION: CORE INFERENCE ENGINE ===")
    print(f"Loading detector with model: {MODEL_PATH}")

    start_time = time.time()
    detector = PCBDetector()
    load_time = time.time() - start_time
    print(f"Detector successfully loaded in {load_time:.3f}s on device: '{detector.device}'")

    # Verify model classes match expected classes
    model_classes = list(detector.classes.values())
    print(f"Model classes ({len(model_classes)}): {model_classes}")
    for c in CLASSES:
        assert c in model_classes, f"Expected class '{c}' not found in model classes!"
    print("Class mapping check: PASSED")

    # Create a synthetic PCB image canvas (640x640x3) to test prediction pipeline
    synthetic_pcb = np.full((640, 640, 3), (34, 139, 34), dtype=np.uint8)  # Green PCB solder mask base
    # Draw some mock traces and pads
    synthetic_pcb[100:150, 100:150] = (20, 20, 20)
    synthetic_pcb[200:300, 200:400] = (10, 10, 10)

    # Run inference test
    t0 = time.time()
    detections = detector.predict(synthetic_pcb, conf=0.1)
    infer_time = (time.time() - t0) * 1000
    print(f"Synthetic inference completed in {infer_time:.2f} ms")
    print(f"Detections count: {len(detections)}")

    # Verify Detection dataclass functionality
    mock_det = Detection(class_id=3, class_name="ic", confidence=0.92, box=[50.0, 50.0, 150.0, 150.0])
    assert mock_det.width == 100.0, "Width calculation incorrect"
    assert mock_det.height == 100.0, "Height calculation incorrect"
    assert mock_det.area == 10000.0, "Area calculation incorrect"
    det_dict = mock_det.to_dict()
    assert det_dict["class_name"] == "ic"
    assert det_dict["confidence"] == 0.92
    print("Detection dataclass methods: PASSED")

    print("\nSTAGE 1 VERIFICATION RESULT: ALL TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_stage1_tests()
