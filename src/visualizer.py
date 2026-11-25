"""
Visualization Engine for BharatPCB-AI.
Renders high-visibility component bounding boxes, defect callouts,
and side-by-side reference vs test PCB comparison images.
"""

from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
import cv2
import numpy as np

from src.config import (
    CLASS_COLORS,
    DEFECT_COLORS,
    DEFAULT_COLOR,
    OUTPUT_DIR
)
from src.detector import Detection
from src.inspector import InspectionResult
from src.comparator import ComponentDiscrepancy, MatchedPair

VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)


class PCBVisualizer:
    """
    Renders high-visibility annotations, defect callouts,
    and side-by-side inspection comparison views.
    """

    def __init__(self):
        self.class_colors = CLASS_COLORS
        self.defect_colors = DEFECT_COLORS
        self.default_color = DEFAULT_COLOR

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
        title: Optional[str] = None
    ) -> np.ndarray:
        """Draw standard YOLO component detections on an image."""
        if image is None:
            raise ValueError("Input image cannot be None")

        canvas = image.copy()
        for det in detections:
            color = self.class_colors.get(det.class_name.lower(), self.default_color)
            x1, y1, x2, y2 = [int(round(c)) for c in det.box]

            # Bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # Label banner
            label = f"{det.class_name} {det.confidence:.0%}"
            self._draw_label_box(canvas, label, (x1, max(18, y1 - 4)), color, text_color=(0, 0, 0))

        if title:
            self._draw_header_bar(canvas, title, bg_color=(40, 40, 40))

        return canvas

    def render_inspection_views(
        self,
        ref_image: np.ndarray,
        inspection_result: InspectionResult
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render distinct annotated reference and test images:
        - Reference view shows verified matches and missing components.
        - Test view shows verified matches, misplaced components, extra components,
          class mismatches, and low-confidence warnings.
        """
        if ref_image is None:
            raise ValueError("Reference image cannot be None")

        ref_canvas = ref_image.copy()
        test_img = inspection_result.aligned_test_image
        if test_img is None:
            # When alignment failed, show placeholder or raw test image
            h, w = ref_image.shape[:2]
            test_canvas = np.full((h, w, 3), (30, 30, 30), dtype=np.uint8)
            cv2.putText(
                test_canvas,
                "ALIGNMENT REJECTED - NO ALIGNED IMAGE",
                (w // 8, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
            return ref_canvas, test_canvas

        test_canvas = test_img.copy()

        # If comparison was skipped due to safety gate
        if inspection_result.comparison is None:
            return ref_canvas, test_canvas

        summary = inspection_result.comparison

        # 1. Render Matched Components (Green outline on both boards)
        matched_color = (0, 200, 0)
        for pair in summary.matched:
            # On Ref
            rx1, ry1, rx2, ry2 = [int(round(c)) for c in pair.reference_component.box]
            cv2.rectangle(ref_canvas, (rx1, ry1), (rx2, ry2), matched_color, 2)
            self._draw_label_box(ref_canvas, pair.reference_component.class_name, (rx1, max(16, ry1 - 3)), matched_color)

            # On Test
            tx1, ty1, tx2, ty2 = [int(round(c)) for c in pair.test_component.box]
            cv2.rectangle(test_canvas, (tx1, ty1), (tx2, ty2), matched_color, 2)
            self._draw_label_box(test_canvas, f"{pair.test_component.class_name} {pair.test_component.confidence:.0%}", (tx1, max(16, ty1 - 3)), matched_color)

        # 2. Render Missing Components (Red on Reference board)
        missing_color = (0, 0, 255)
        for disc in summary.missing:
            x1, y1, x2, y2 = [int(round(c)) for c in disc.component.box]
            self._draw_dashed_rect(ref_canvas, (x1, y1), (x2, y2), missing_color, thickness=2)
            self._draw_label_box(ref_canvas, f"MISSING: {disc.component.class_name}", (x1, max(18, y1 - 4)), missing_color, (255, 255, 255))

        # 3. Render Extra Components (Magenta on Test board)
        extra_color = self.defect_colors.get("extra", (255, 0, 128))
        for disc in summary.extra:
            x1, y1, x2, y2 = [int(round(c)) for c in disc.component.box]
            self._draw_dashed_rect(test_canvas, (x1, y1), (x2, y2), extra_color, thickness=2)
            self._draw_label_box(test_canvas, f"EXTRA: {disc.component.class_name}", (x1, max(18, y1 - 4)), extra_color, (255, 255, 255))

        # 4. Render Misplaced Components (Orange on Test board with shift line)
        misplaced_color = self.defect_colors.get("misplaced", (0, 165, 255))
        for disc in summary.misplaced:
            x1, y1, x2, y2 = [int(round(c)) for c in disc.component.box]
            cv2.rectangle(test_canvas, (x1, y1), (x2, y2), misplaced_color, 2)
            lbl = f"SHIFTED: {disc.component.class_name} ({disc.centroid_drift:.0f}px)"
            self._draw_label_box(test_canvas, lbl, (x1, max(18, y1 - 4)), misplaced_color, (0, 0, 0))

            # Draw drift indicator arrow if paired component exists
            if disc.paired_component:
                pt_ref = (int(round(disc.paired_component.centroid[0])), int(round(disc.paired_component.centroid[1])))
                pt_test = (int(round(disc.component.centroid[0])), int(round(disc.component.centroid[1])))
                cv2.arrowedLine(test_canvas, pt_ref, pt_test, misplaced_color, 2, tipLength=0.25)

        # 5. Render Class Mismatches (Red/Orange on Test board)
        mismatch_color = (0, 69, 255)
        for disc in summary.class_mismatches:
            x1, y1, x2, y2 = [int(round(c)) for c in disc.component.box]
            cv2.rectangle(test_canvas, (x1, y1), (x2, y2), mismatch_color, 3)
            ref_cls = disc.paired_component.class_name if disc.paired_component else "unknown"
            lbl = f"WRONG CLASS: {disc.component.class_name} (Exp: {ref_cls})"
            self._draw_label_box(test_canvas, lbl, (x1, max(18, y1 - 4)), mismatch_color, (255, 255, 255))

        # 6. Render Low Confidence Warnings (Yellow highlight on Test board)
        low_conf_color = self.defect_colors.get("low_confidence", (0, 255, 255))
        for det in summary.low_confidence_warnings:
            x1, y1, x2, y2 = [int(round(c)) for c in det.box]
            # Small yellow corner ticks
            cv2.drawMarker(test_canvas, (x1, y1), low_conf_color, cv2.MARKER_TRIANGLE_UP, 10, 2)
            self._draw_label_box(test_canvas, f"LOW CONF ({det.confidence:.0%})", (x1, min(test_canvas.shape[0] - 8, y2 + 14)), low_conf_color, (0, 0, 0))

        return ref_canvas, test_canvas

    def create_side_by_side_comparison(
        self,
        ref_image: np.ndarray,
        inspection_result: InspectionResult
    ) -> np.ndarray:
        """
        Generate a side-by-side inspection view:
        Left: Reference PCB (Golden) with missing defect tags
        Right: Test PCB (Inspected) with extra/misplaced/mismatch defect tags
        Top: Color-coded Verdict Banner (PASS=Green, FAIL=Red, REVIEW=Yellow)
        """
        ref_annotated, test_annotated = self.render_inspection_views(ref_image, inspection_result)

        h_ref, w_ref = ref_annotated.shape[:2]
        h_test, w_test = test_annotated.shape[:2]

        # Match heights
        target_h = max(h_ref, h_test)
        if h_ref != target_h:
            ref_annotated = cv2.resize(ref_annotated, (int(w_ref * target_h / h_ref), target_h))
        if h_test != target_h:
            test_annotated = cv2.resize(test_annotated, (int(w_test * target_h / h_test), target_h))

        h_ref, w_ref = ref_annotated.shape[:2]
        h_test, w_test = test_annotated.shape[:2]

        # Section Subheaders
        self._draw_header_bar(ref_annotated, "GOLDEN REFERENCE PCB", bg_color=(45, 45, 45))
        self._draw_header_bar(test_annotated, "INSPECTED TEST PCB", bg_color=(45, 45, 45))

        # Stitch side-by-side
        combined_w = w_ref + w_test
        banner_h = 70
        full_canvas = np.zeros((target_h + banner_h, combined_w, 3), dtype=np.uint8)

        # Draw Verdict Banner
        verdict = inspection_result.verdict
        if verdict == "PASS":
            banner_bg = (34, 139, 34)    # Forest Green
            text_color = (255, 255, 255)
        elif verdict == "FAIL":
            banner_bg = (0, 0, 180)      # Crimson Red
            text_color = (255, 255, 255)
        else:  # REVIEW or HALTED
            banner_bg = (0, 140, 255)    # Dark Orange/Amber
            text_color = (255, 255, 255)

        full_canvas[0:banner_h, 0:combined_w] = banner_bg

        # Banner Text
        cv2.putText(
            full_canvas,
            f"BHARATPCB-AI VERDICT: {verdict}",
            (20, 32),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            text_color,
            2,
            cv2.LINE_AA
        )

        sub_text = inspection_result.verdict_reason
        if len(sub_text) > 110:
            sub_text = sub_text[:107] + "..."
        cv2.putText(
            full_canvas,
            sub_text,
            (20, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (240, 240, 240),
            1,
            cv2.LINE_AA
        )

        # Place images
        full_canvas[banner_h:banner_h + target_h, 0:w_ref] = ref_annotated
        full_canvas[banner_h:banner_h + target_h, w_ref:combined_w] = test_annotated

        # Divider line
        cv2.line(full_canvas, (w_ref, banner_h), (w_ref, banner_h + target_h), (200, 200, 200), 2)

        return full_canvas

    def save_visualization(
        self,
        image: np.ndarray,
        filename_prefix: str = "pcb_inspection"
    ) -> Path:
        """Save visualization image with unique timestamped filename."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:19]
        filename = f"{filename_prefix}_{ts}.png"
        filepath = VISUALIZATIONS_DIR / filename
        cv2.imwrite(str(filepath), image)
        return filepath

    def _draw_label_box(
        self,
        img: np.ndarray,
        text: str,
        pos: Tuple[int, int],
        bg_color: Tuple[int, int, int],
        text_color: Tuple[int, int, int] = (0, 0, 0)
    ):
        """Draw filled label background box with crisp text."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.42
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = pos
        x = max(2, min(img.shape[1] - tw - 4, x))
        y = max(th + 4, min(img.shape[0] - 4, y))

        cv2.rectangle(img, (x, y - th - 3), (x + tw + 4, y + baseline), bg_color, -1)
        cv2.putText(img, text, (x + 2, y - 2), font, scale, text_color, thickness, cv2.LINE_AA)

    def _draw_header_bar(
        self,
        img: np.ndarray,
        text: str,
        bg_color: Tuple[int, int, int]
    ):
        """Draw title header band across top of image."""
        w = img.shape[1]
        cv2.rectangle(img, (0, 0), (w, 26), bg_color, -1)
        cv2.putText(img, text, (10, 18), cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_dashed_rect(
        self,
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 2,
        dash_len: int = 8
    ):
        """Draw dashed rectangular boundary for defect callouts."""
        x1, y1 = pt1
        x2, y2 = pt2

        # Horizontal top and bottom
        for x in range(x1, x2, dash_len * 2):
            xe = min(x + dash_len, x2)
            cv2.line(img, (x, y1), (xe, y1), color, thickness)
            cv2.line(img, (x, y2), (xe, y2), color, thickness)

        # Vertical left and right
        for y in range(y1, y2, dash_len * 2):
            ye = min(y + dash_len, y2)
            cv2.line(img, (x1, y), (x1, ye), color, thickness)
            cv2.line(img, (x2, y), (x2, ye), color, thickness)

    def compute_structural_difference_heatmap(
        self,
        ref_image: np.ndarray,
        aligned_test_image: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Compute an illumination-normalized structural difference heatmap
        between the Golden Reference PCB and Aligned Test PCB.
        """
        if ref_image is None or aligned_test_image is None:
            # Fallback placeholder if no aligned image
            h = 400 if ref_image is None else ref_image.shape[0]
            w = 600 if ref_image is None else ref_image.shape[1]
            placeholder = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(
                placeholder,
                "HEATMAP UNAVAILABLE (NO ALIGNED IMAGE)",
                (w // 8, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
            return placeholder

        # Resize if dimensions differ slightly
        h_ref, w_ref = ref_image.shape[:2]
        test_aligned = aligned_test_image
        if test_aligned.shape[:2] != (h_ref, w_ref):
            test_aligned = cv2.resize(test_aligned, (w_ref, h_ref))

        gray_ref = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY) if len(ref_image.shape) == 3 else ref_image
        gray_test = cv2.cvtColor(test_aligned, cv2.COLOR_BGR2GRAY) if len(test_aligned.shape) == 3 else test_aligned

        # Illumination normalization via CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        norm_ref = clahe.apply(gray_ref)
        norm_test = clahe.apply(gray_test)

        # Smooth high-frequency subpixel noise
        blur_ref = cv2.GaussianBlur(norm_ref, (5, 5), 0)
        blur_test = cv2.GaussianBlur(norm_test, (5, 5), 0)

        # Absolute difference
        diff = cv2.absdiff(blur_ref, blur_test)

        # Apply threshold to suppress background grain
        _, diff_thresh = cv2.threshold(diff, 28, 255, cv2.THRESH_TOZERO)

        # Apply ColorMap JET (Blue = low difference, Red = high difference)
        heatmap = cv2.applyColorMap(diff_thresh, cv2.COLORMAP_JET)

        # Blend with grayscale background for physical board context
        gray_3ch = cv2.cvtColor(gray_test, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(gray_3ch, 0.45, heatmap, 0.55, 0)

        self._draw_header_bar(blended, "STRUCTURAL DIFFERENCE HEATMAP (JET COLORMAP)", bg_color=(20, 20, 20))
        return blended

    def crop_defect_regions(
        self,
        ref_image: np.ndarray,
        aligned_test_image: Optional[np.ndarray],
        inspection_result: InspectionResult,
        pad: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Crop high-resolution visual thumbnails for each identified defect
        for the Defect Crop Panel.
        """
        crops = []
        if inspection_result.comparison is None:
            return crops

        summary = inspection_result.comparison
        h_ref, w_ref = ref_image.shape[:2]
        h_test = aligned_test_image.shape[0] if aligned_test_image is not None else h_ref
        w_test = aligned_test_image.shape[1] if aligned_test_image is not None else w_ref

        idx = 1
        all_discs = (
            [("MISSING", d, "reference", ref_image, (w_ref, h_ref)) for d in summary.missing] +
            [("MISPLACED", d, "test", aligned_test_image, (w_test, h_test)) for d in summary.misplaced] +
            [("CLASS_MISMATCH", d, "test", aligned_test_image, (w_test, h_test)) for d in summary.class_mismatches] +
            [("EXTRA", d, "test", aligned_test_image, (w_test, h_test)) for d in summary.extra]
        )

        for disc_type, disc, source_name, img_src, (max_w, max_h) in all_discs:
            if img_src is None:
                continue

            x1, y1, x2, y2 = [int(round(c)) for c in disc.component.box]
            cx, cy = disc.component.centroid

            # Apply padding
            crop_x1 = max(0, x1 - pad)
            crop_y1 = max(0, y1 - pad)
            crop_x2 = min(max_w, x2 + pad)
            crop_y2 = min(max_h, y2 + pad)

            if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                crop_img = img_src[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                # Draw local bounding box on the crop
                local_x1 = x1 - crop_x1
                local_y1 = y1 - crop_y1
                local_x2 = x2 - crop_x1
                local_y2 = y2 - crop_y1

                box_color = (0, 0, 255) if disc_type in {"MISSING", "CLASS_MISMATCH"} else (
                    (0, 165, 255) if disc_type == "MISPLACED" else (255, 0, 128)
                )
                cv2.rectangle(crop_img, (local_x1, local_y1), (local_x2, local_y2), box_color, 2)

                crops.append({
                    "defect_id": f"DEF-{idx:03d}",
                    "defect_type": disc_type,
                    "component_name": disc.component.class_name.upper(),
                    "confidence": disc.component.confidence,
                    "location": f"({cx:.0f}, {cy:.0f})",
                    "explanation": disc.reason,
                    "crop_image": crop_img,
                    "source": f"{source_name.title()} Board"
                })
                idx += 1

        return crops

