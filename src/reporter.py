"""
Multi-Format Report Generator for BharatPCB-AI.
Generates comprehensive PDF inspection certificates, CSV component logs,
and structured JSON audit records with unique timestamped filenames.
"""

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import cv2
import numpy as np

# Ensure writable matplotlib cache
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

# Use Agg backend for headless PDF generation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches

from src.config import OUTPUT_DIR
from src.inspector import InspectionResult
from src.visualizer import PCBVisualizer

REPORTS_DIR = OUTPUT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ReportPaths:
    """Paths to all generated inspection report files."""
    pdf_path: Optional[Path]
    csv_path: Optional[Path]
    json_path: Optional[Path]
    visualization_path: Optional[Path]

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "pdf": str(self.pdf_path) if self.pdf_path else None,
            "csv": str(self.csv_path) if self.csv_path else None,
            "json": str(self.json_path) if self.json_path else None,
            "visualization": str(self.visualization_path) if self.visualization_path else None
        }


class InspectionReporter:
    """
    Produces PDF certificates, CSV audit tables, and JSON audit files
    from an InspectionResult without inventing or distorting defect conclusions.
    """

    def __init__(self, reports_dir: Path = REPORTS_DIR):
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.visualizer = PCBVisualizer()

    def generate_all_reports(
        self,
        inspection_result: InspectionResult,
        ref_image: Optional[np.ndarray] = None,
        ref_name: str = "reference_pcb.png",
        test_name: str = "test_pcb.png"
    ) -> ReportPaths:
        """Generate PDF, CSV, and JSON reports with matching timestamped unique IDs."""
        uid = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:19]
        base_name = f"bharatpcb_report_{uid}"

        # 1. Visualization
        vis_path = None
        vis_image = None
        if ref_image is not None:
            vis_image = self.visualizer.create_side_by_side_comparison(ref_image, inspection_result)
            vis_path = self.visualizer.save_visualization(vis_image, filename_prefix=f"vis_{uid}")

        # 2. JSON Report
        json_path = self.generate_json_report(inspection_result, base_name, ref_name, test_name)

        # 3. CSV Report
        csv_path = self.generate_csv_report(inspection_result, base_name)

        # 4. PDF Certificate Report
        pdf_path = self.generate_pdf_report(
            inspection_result,
            base_name,
            ref_name,
            test_name,
            comparison_image=vis_image
        )

        return ReportPaths(
            pdf_path=pdf_path,
            csv_path=csv_path,
            json_path=json_path,
            visualization_path=vis_path
        )

    def generate_json_report(
        self,
        inspection_result: InspectionResult,
        base_name: str,
        ref_name: str = "reference.png",
        test_name: str = "test.png"
    ) -> Path:
        """Generate formatted JSON inspection data."""
        filepath = self.reports_dir / f"{base_name}.json"
        data = {
            "project": "BharatPCB-AI",
            "system_version": "1.0.0",
            "inspection_time_utc": inspection_result.timestamp,
            "reference_image": ref_name,
            "test_image": test_name,
            "inspection_result": inspection_result.to_dict()
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return filepath

    def generate_csv_report(
        self,
        inspection_result: InspectionResult,
        base_name: str
    ) -> Path:
        """Generate granular CSV log with component and discrepancy records."""
        filepath = self.reports_dir / f"{base_name}.csv"
        headers = [
            "record_type",
            "board",
            "component_id",
            "class_name",
            "confidence",
            "box_x1",
            "box_y1",
            "box_x2",
            "box_y2",
            "centroid_x",
            "centroid_y",
            "match_status",
            "shift_distance_px",
            "discrepancy_reason"
        ]

        rows = []
        # Reference detections
        for idx, det in enumerate(inspection_result.ref_detections):
            rows.append([
                "DETECTION",
                "REFERENCE",
                f"REF_{idx+1}",
                det.class_name,
                f"{det.confidence:.4f}",
                round(det.box[0], 1),
                round(det.box[1], 1),
                round(det.box[2], 1),
                round(det.box[3], 1),
                det.centroid[0],
                det.centroid[1],
                "N/A",
                "0.0",
                "Baseline Reference Component"
            ])

        # Test detections
        for idx, det in enumerate(inspection_result.test_detections):
            rows.append([
                "DETECTION",
                "TEST",
                f"TEST_{idx+1}",
                det.class_name,
                f"{det.confidence:.4f}",
                round(det.box[0], 1),
                round(det.box[1], 1),
                round(det.box[2], 1),
                round(det.box[3], 1),
                det.centroid[0],
                det.centroid[1],
                "N/A",
                "0.0",
                "Aligned Test Board Detection"
            ])

        # If comparison was executed, add discrepancies
        summary = inspection_result.comparison
        if summary is not None:
            # Matched pairs
            for idx, pair in enumerate(summary.matched):
                rows.append([
                    "COMPARISON",
                    "MATCHED_PAIR",
                    f"MATCH_{idx+1}",
                    pair.reference_component.class_name,
                    f"{pair.test_component.confidence:.4f}",
                    round(pair.test_component.box[0], 1),
                    round(pair.test_component.box[1], 1),
                    round(pair.test_component.box[2], 1),
                    round(pair.test_component.box[3], 1),
                    pair.test_component.centroid[0],
                    pair.test_component.centroid[1],
                    "MATCHED",
                    f"{pair.centroid_distance:.2f}",
                    pair.notes
                ])

            # Discrepancies
            all_discs = (
                [("MISSING", d) for d in summary.missing] +
                [("EXTRA", d) for d in summary.extra] +
                [("MISPLACED", d) for d in summary.misplaced] +
                [("CLASS_MISMATCH", d) for d in summary.class_mismatches]
            )

            for idx, (disc_type, disc) in enumerate(all_discs):
                rows.append([
                    "DISCREPANCY",
                    "TEST" if disc_type != "MISSING" else "REFERENCE",
                    f"DISC_{idx+1}",
                    disc.component.class_name,
                    f"{disc.component.confidence:.4f}",
                    round(disc.component.box[0], 1),
                    round(disc.component.box[1], 1),
                    round(disc.component.box[2], 1),
                    round(disc.component.box[3], 1),
                    disc.component.centroid[0],
                    disc.component.centroid[1],
                    disc_type,
                    f"{disc.centroid_drift:.2f}" if disc.centroid_drift is not None else "N/A",
                    disc.reason
                ])

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

        return filepath

    def generate_pdf_report(
        self,
        inspection_result: InspectionResult,
        base_name: str,
        ref_name: str = "reference.png",
        test_name: str = "test.png",
        comparison_image: Optional[np.ndarray] = None
    ) -> Path:
        """
        Generate multi-page PDF Inspection Certificate.
        Uses matplotlib.backends.backend_pdf (zero external dependencies).
        """
        pdf_path = self.reports_dir / f"{base_name}.pdf"

        with PdfPages(str(pdf_path)) as pdf:
            # ---------------- PAGE 1: EXECUTIVE SUMMARY ----------------
            fig1 = plt.figure(figsize=(8.5, 11), dpi=150)
            fig1.patch.set_facecolor("#FAFAFA")

            # Title Header Band
            ax_header = fig1.add_axes([0.05, 0.90, 0.90, 0.08])
            ax_header.set_facecolor("#1A237E")  # Dark Blue
            ax_header.text(0.03, 0.60, "BharatPCB-AI: Automated Inspection Certificate",
                           fontsize=15, fontweight="bold", color="white", va="center")
            ax_header.text(0.03, 0.25, "Quality Control & Automated Defect Audit Report",
                           fontsize=9.5, color="#BBDEFB", va="center")
            ax_header.axis("off")

            # Verdict Card
            verdict = inspection_result.verdict
            v_color = "#2E7D32" if verdict == "PASS" else ("#C62828" if verdict == "FAIL" else "#EF6C00")
            v_bg = "#E8F5E9" if verdict == "PASS" else ("#FFEBEE" if verdict == "FAIL" else "#FFF3E0")

            ax_verdict = fig1.add_axes([0.05, 0.77, 0.90, 0.11])
            ax_verdict.set_facecolor(v_bg)
            rect = patches.Rectangle((0, 0), 1, 1, transform=ax_verdict.transAxes,
                                     fill=False, edgecolor=v_color, linewidth=2.0)
            ax_verdict.add_patch(rect)
            ax_verdict.text(0.04, 0.65, f"OVERALL VERDICT: {verdict}",
                            fontsize=16, fontweight="bold", color=v_color, va="center")
            ax_verdict.text(0.04, 0.28, inspection_result.verdict_reason,
                            fontsize=9.5, color="#212121", va="center", wrap=True)
            ax_verdict.axis("off")

            # Metadata & Alignment Table Area
            ax_meta = fig1.add_axes([0.05, 0.54, 0.43, 0.21])
            ax_meta.set_facecolor("#FFFFFF")
            ax_meta.axis("off")
            ax_meta.text(0.0, 0.95, "Inspection Metadata", fontsize=11, fontweight="bold", color="#1A237E")

            meta_data = [
                ["Inspection Date", inspection_result.timestamp[:19].replace("T", " ")],
                ["Reference Image", ref_name],
                ["Test Board Image", test_name],
                ["Processing Mode", "100% Local Inference"],
                ["Hardware", "Apple Silicon M5 (16 GB)"],
                ["Model Engine", "YOLOv8 Component Detector"]
            ]
            t_meta = ax_meta.table(cellText=meta_data, colWidths=[0.42, 0.58], loc="center", cellLoc="left")
            t_meta.auto_set_font_size(False)
            t_meta.set_fontsize(8.5)
            t_meta.scale(1.0, 1.35)

            # Alignment Quality Table
            ax_align = fig1.add_axes([0.52, 0.54, 0.43, 0.21])
            ax_align.set_facecolor("#FFFFFF")
            ax_align.axis("off")
            ax_align.text(0.0, 0.95, "Alignment Quality Metrics", fontsize=11, fontweight="bold", color="#1A237E")

            align_m = inspection_result.alignment.metrics
            align_data = [
                ["Alignment Status", align_m.status],
                ["RANSAC Inliers", str(align_m.inliers_count)],
                ["Inlier Ratio", f"{align_m.inlier_ratio:.1%}"],
                ["Reprojection Error", f"{align_m.reprojection_error:.2f} px" if align_m.reprojection_error < 999 else "N/A"],
                ["Homography Valid", "YES" if align_m.is_homography_valid else "NO"],
                ["Quality Score", f"{align_m.quality_score:.2f} / 1.00"]
            ]
            t_align = ax_align.table(cellText=align_data, colWidths=[0.50, 0.50], loc="center", cellLoc="left")
            t_align.auto_set_font_size(False)
            t_align.set_fontsize(8.5)
            t_align.scale(1.0, 1.35)

            # Component Discrepancy Breakdown Table
            ax_comp = fig1.add_axes([0.05, 0.22, 0.90, 0.29])
            ax_comp.set_facecolor("#FFFFFF")
            ax_comp.axis("off")
            ax_comp.text(0.0, 0.96, "Component Audit Summary", fontsize=11, fontweight="bold", color="#1A237E")

            summary = inspection_result.comparison
            if summary is not None:
                summary_rows = [
                    ["Reference Components (Expected)", str(summary.ref_count), "Components identified on golden board"],
                    ["Test Components Detected", str(summary.test_count), "Components identified on inspected board"],
                    ["Matched Components", str(len(summary.matched)), "Verified within spatial & class tolerances"],
                    ["Missing Components", str(len(summary.missing)), "CRITICAL: Present on ref, absent on test"],
                    ["Misplaced Components", str(len(summary.misplaced)), "Position shifted beyond allowable threshold"],
                    ["Extra Components", str(len(summary.extra)), "Uncataloged components detected on test board"],
                    ["Class Mismatches", str(len(summary.class_mismatches)), "CRITICAL: Incorrect component type installed"],
                    ["Low Confidence Warnings", str(len(summary.low_confidence_warnings)), "Borderline confidence requiring manual check"]
                ]
            else:
                summary_rows = [
                    ["Inspection Status", "HALTED", "Comparison aborted to prevent false defect claims"],
                    ["Reason", align_m.status, align_m.rejection_reason or "Alignment rejected"]
                ]

            t_comp = ax_comp.table(
                cellText=summary_rows,
                colLabels=["Metric", "Count", "Description"],
                colWidths=[0.36, 0.12, 0.52],
                loc="center",
                cellLoc="left"
            )
            t_comp.auto_set_font_size(False)
            t_comp.set_fontsize(8.5)
            t_comp.scale(1.0, 1.3)

            # Header row styling
            for (r, c), cell in t_comp.get_celld().items():
                if r == 0:
                    cell.set_facecolor("#E0E0E0")
                    cell.set_text_props(weight="bold")

            # Limitations Footer Note
            ax_limit = fig1.add_axes([0.05, 0.05, 0.90, 0.14])
            ax_limit.set_facecolor("#FFF9C4")  # Soft Yellow
            r_lim = patches.Rectangle((0, 0), 1, 1, transform=ax_limit.transAxes,
                                      fill=False, edgecolor="#FBC02D", linewidth=1.5)
            ax_limit.add_patch(r_lim)
            ax_limit.text(0.02, 0.82, "SYSTEM LIMITATIONS & SCOPE DISCLOSURE",
                          fontsize=9, fontweight="bold", color="#E65100")
            lim_text = (
                "1. The local YOLO model is trained solely to detect 9 electronic component packages "
                "(capacitor, connector, diode, ic, inductor, led, resistor, switch, transistor).\n"
                "2. The model DOES NOT detect solder bridging, cold joints, flux residue, or broken copper traces.\n"
                "3. Defect classifications (missing, extra, misplaced) are mathematically derived by comparing test detections "
                "against the aligned reference PCB. Alignment quality gating strictly prevents false defect conclusions.\n"
                "4. Final quality approval for flagged boards must be confirmed by a certified QA engineer."
            )
            ax_limit.text(0.02, 0.40, lim_text, fontsize=7.8, color="#37474F", va="center")
            ax_limit.axis("off")

            pdf.savefig(fig1)
            plt.close(fig1)

            # ---------------- PAGE 2: VISUAL COMPARISON & DEFECT DETAILS ----------------
            fig2 = plt.figure(figsize=(8.5, 11), dpi=150)
            fig2.patch.set_facecolor("#FAFAFA")

            ax_p2_hdr = fig2.add_axes([0.05, 0.92, 0.90, 0.06])
            ax_p2_hdr.set_facecolor("#1A237E")
            ax_p2_hdr.text(0.03, 0.5, "Visual Inspection & Discrepancy Breakdown",
                           fontsize=13, fontweight="bold", color="white", va="center")
            ax_p2_hdr.axis("off")

            # Render comparison image if available
            if comparison_image is not None:
                ax_img = fig2.add_axes([0.05, 0.50, 0.90, 0.40])
                # Convert BGR to RGB
                rgb_img = cv2.cvtColor(comparison_image, cv2.COLOR_BGR2RGB) if len(comparison_image.shape) == 3 else comparison_image
                ax_img.imshow(rgb_img)
                ax_img.axis("off")
                ax_img.set_title("Side-by-Side Comparison: Golden Reference vs Aligned Inspected PCB", fontsize=9.5, pad=5)
            else:
                ax_img = fig2.add_axes([0.05, 0.50, 0.90, 0.40])
                ax_img.text(0.5, 0.5, "No comparison image available (Alignment was rejected)",
                            ha="center", va="center", color="#757575", fontsize=11)
                ax_img.axis("off")

            # Discrepancy Table
            ax_disc_table = fig2.add_axes([0.05, 0.08, 0.90, 0.38])
            ax_disc_table.set_facecolor("#FFFFFF")
            ax_disc_table.axis("off")
            ax_disc_table.text(0.0, 0.96, "Itemized Discrepancy Log", fontsize=11, fontweight="bold", color="#1A237E")

            disc_rows = []
            if summary is not None:
                for d in summary.missing:
                    disc_rows.append(["MISSING", d.component.class_name, f"{d.component.centroid}", "CRITICAL", d.reason[:60]])
                for d in summary.class_mismatches:
                    disc_rows.append(["CLASS_MISMATCH", d.component.class_name, f"{d.component.centroid}", "CRITICAL", d.reason[:60]])
                for d in summary.misplaced:
                    disc_rows.append(["MISPLACED", d.component.class_name, f"{d.component.centroid}", "HIGH", d.reason[:60]])
                for d in summary.extra:
                    disc_rows.append(["EXTRA", d.component.class_name, f"{d.component.centroid}", "MEDIUM", d.reason[:60]])

            if not disc_rows:
                disc_rows = [["NONE", "N/A", "N/A", "INFO", "No component discrepancies detected."]]

            t_disc = ax_disc_table.table(
                cellText=disc_rows[:12],  # show top 12
                colLabels=["Type", "Class", "Centroid (x,y)", "Severity", "Description"],
                colWidths=[0.18, 0.15, 0.16, 0.12, 0.39],
                loc="center",
                cellLoc="left"
            )
            t_disc.auto_set_font_size(False)
            t_disc.set_fontsize(8.0)
            t_disc.scale(1.0, 1.25)

            for (r, c), cell in t_disc.get_celld().items():
                if r == 0:
                    cell.set_facecolor("#E0E0E0")
                    cell.set_text_props(weight="bold")

            pdf.savefig(fig2)
            plt.close(fig2)

        return pdf_path
