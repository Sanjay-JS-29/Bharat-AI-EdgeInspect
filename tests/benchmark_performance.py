"""
Stage 6 Performance Benchmarking Suite for BharatPCB-AI.
Measures actual:
  - Model initialization latency.
  - Cold inference latency.
  - Warm inference latency (CPU vs MPS).
  - OpenCV image alignment latency.
  - End-to-end inspection pipeline latency.
  - System memory usage (RSS).
  - Batch inspection throughput.
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np
import psutil
import torch

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import PCBDetector
from src.aligner import PCBAligner
from src.inspector import PCBInspector
from src.reporter import InspectionReporter
from tests.test_alignment import generate_synthetic_pcb


def run_performance_benchmarks():
    print("=" * 65)
    print("BHARATPCB-AI: PERFORMANCE BENCHMARK & HARDWARE PROFILING")
    print("=" * 65)

    process = psutil.Process()
    mem_before_mb = process.memory_info().rss / (1024 * 1024)
    print(f"Initial Memory Usage (RSS): {mem_before_mb:.2f} MB")

    # 1. Model Initialization Latency
    print("\n[1/6] Measuring Model Initialization Latency...")
    t0 = time.perf_counter()
    detector = PCBDetector()
    init_time_s = time.perf_counter() - t0
    print(f"  -> Model Load & Initialization Time: {init_time_s:.4f} s")

    # 2. Cold vs Warm Inference Latency (CPU)
    print("\n[2/6] Measuring Inference Latency on Device: 'cpu'...")
    test_canvas = np.full((640, 640, 3), 40, dtype=np.uint8)

    # Cold run
    t0 = time.perf_counter()
    detector.predict(test_canvas, conf=0.25)
    cold_latency_ms = (time.perf_counter() - t0) * 1000
    print(f"  -> Cold Inference Latency: {cold_latency_ms:.2f} ms")

    # Warm runs (10 runs)
    warm_latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        detector.predict(test_canvas, conf=0.25)
        warm_latencies.append((time.perf_counter() - t0) * 1000)

    avg_warm_ms = float(np.mean(warm_latencies))
    std_warm_ms = float(np.std(warm_latencies))
    min_warm_ms = float(np.min(warm_latencies))
    max_warm_ms = float(np.max(warm_latencies))
    print(f"  -> Warm Inference Latency (Mean): {avg_warm_ms:.2f} ± {std_warm_ms:.2f} ms")
    print(f"  -> Min: {min_warm_ms:.2f} ms, Max: {max_warm_ms:.2f} ms")
    print(f"  -> Throughput: {1000.0 / avg_warm_ms:.1f} FPS")

    # 3. MPS (Metal Performance Shaders) Hardware Availability & Measurement
    print("\n[3/6] Checking Apple Silicon MPS (Metal) Support...")
    mps_built = torch.backends.mps.is_built()
    mps_avail = torch.backends.mps.is_available()
    print(f"  -> PyTorch MPS Built: {mps_built}")
    print(f"  -> PyTorch MPS Available: {mps_avail}")

    if mps_avail:
        try:
            detector_mps = PCBDetector(device="mps")
            t0 = time.perf_counter()
            detector_mps.predict(test_canvas, conf=0.25)
            mps_cold = (time.perf_counter() - t0) * 1000
            mps_warm = []
            for _ in range(10):
                t0 = time.perf_counter()
                detector_mps.predict(test_canvas, conf=0.25)
                mps_warm.append((time.perf_counter() - t0) * 1000)
            print(f"  -> MPS Cold Latency: {mps_cold:.2f} ms")
            print(f"  -> MPS Warm Latency: {np.mean(mps_warm):.2f} ms")
        except Exception as e:
            print(f"  -> MPS evaluation note: {str(e)}")
    else:
        print("  -> MPS device not available in current execution context; CPU inference active.")

    # 4. Image Alignment Latency
    print("\n[4/6] Measuring OpenCV ORB + RANSAC Alignment Latency...")
    aligner = PCBAligner(n_features=3000)
    ref_pcb = generate_synthetic_pcb(800, 800, seed=701)
    rot_mat = cv2.getRotationMatrix2D((400, 400), 8.0, 1.0)
    test_pcb = cv2.warpAffine(ref_pcb, rot_mat, (800, 800))

    align_latencies = []
    for _ in range(5):
        t0 = time.perf_counter()
        aligner.align(ref_pcb, test_pcb)
        align_latencies.append((time.perf_counter() - t0) * 1000)

    avg_align_ms = float(np.mean(align_latencies))
    print(f"  -> ORB Detection + Descriptor Matching + RANSAC: {avg_align_ms:.2f} ms (Mean over 5 runs)")

    # 5. Complete End-to-End Inspection Latency
    print("\n[5/6] Measuring End-to-End Inspection Pipeline Latency...")
    inspector = PCBInspector()
    e2e_latencies = []
    for _ in range(3):
        t0 = time.perf_counter()
        inspector.inspect(ref_pcb, test_pcb)
        e2e_latencies.append((time.perf_counter() - t0) * 1000)

    avg_e2e_ms = float(np.mean(e2e_latencies))
    print(f"  -> Full Pipeline (Alignment + 2x YOLO Inferences + Discrepancy Engine): {avg_e2e_ms:.2f} ms")

    # 6. Memory Footprint and Batch Throughput
    print("\n[6/6] Measuring Batch Processing Throughput & Memory...")
    batch_boards = [test_pcb.copy() for _ in range(5)]
    t0 = time.perf_counter()
    for b in batch_boards:
        inspector.inspect(ref_pcb, b)
    batch_total_s = time.perf_counter() - t0
    batch_fps = len(batch_boards) / batch_total_s

    mem_after_mb = process.memory_info().rss / (1024 * 1024)
    mem_delta_mb = mem_after_mb - mem_before_mb
    print(f"  -> 5-Board Batch Time: {batch_total_s:.3f} s ({batch_fps:.2f} boards/sec)")
    print(f"  -> Peak Resident Memory: {mem_after_mb:.2f} MB (Delta: +{mem_delta_mb:.2f} MB)")

    print("\n" + "=" * 65)
    print("PERFORMANCE BENCHMARKING COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    run_performance_benchmarks()
