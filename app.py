"""
EdgeInspect AI — Industrial AI-Powered PCB Inspection System
(BharatPCB-AI Core Engine)

Features:
  - Dark industrial inspection interface
  - Reference PCB upload & Test PCB upload (PNG, JPG, JPEG, TIFF, BMP)
  - MacBook built-in camera capture via st.camera_input
  - Image previews with dimensional sanity checks
  - "Inspect PCB" action with live progress logs
  - Aligned Test image & Final annotated defect overlay
  - Defect Crop Panel showing individual cropped discrepancies
  - Illumination-normalized Structural Difference Heatmap
  - Defect list table (Defect ID, Type, Component, Confidence, Region, Positions, Explanation)
  - Separate counts: Missing, Misplaced, Extra, Mismatch, Low-Conf
  - Real safety-gated inspection verdicts (Never invent PASS on rejected alignment)
  - Local AI QC Advisor with root cause & interactive technician Q&A
  - Downloadable multi-format reports (PDF, CSV, JSON, Annotated PNG)
  - Inspection History archive
"""

from pathlib import Path
from datetime import datetime, timezone
import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Core Package Imports
from src.config import (
    MODEL_PATH,
    CLASSES,
    CLASS_COLORS,
    DEFECT_COLORS,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    OUTPUT_DIR,
    REPORTS_DIR
)
from src.camera import decode_image_bytes, validate_image, bgr_to_rgb
from src.detector import PCBDetector, UltralyticsDetector, Detection
from src.aligner import PCBAligner, AlignmentResult, AlignmentMetrics
from src.comparator import ComponentComparator, ComparisonConfig, ComparisonSummary
from src.inspector import PCBInspector, InspectionResult, run_inspection, extract_defect_records, get_board_region
from src.visualizer import PCBVisualizer, VISUALIZATIONS_DIR
from src.reporter import InspectionReporter, ReportPaths
from src.advisor import PCBAdvisor
from tests.test_alignment import generate_synthetic_pcb

# ----------------- STREAMLIT PAGE CONFIG -----------------
st.set_page_config(
    page_title="EdgeInspect AI | Industrial PCB Inspection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- DARK INDUSTRIAL THEME CSS -----------------
st.markdown("""
<style>
    /* Dark industrial palette */
    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
    }
    
    /* Top title & branding */
    .brand-title {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
        font-size: 28px;
        font-weight: 800;
        letter-spacing: 0.5px;
        color: #F8FAFC;
        margin-bottom: 2px;
    }
    .brand-subtitle {
        font-size: 13px;
        color: #00E5FF;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 18px;
    }

    /* Industrial card containers */
    .industrial-card {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }

    /* Metric cards */
    .metric-box {
        background-color: #111827;
        border-left: 4px solid #3B82F6;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .metric-num {
        font-size: 26px;
        font-weight: 800;
        color: #F8FAFC;
        font-family: monospace;
    }
    .metric-label {
        font-size: 11px;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Status Badges */
    .badge-pass {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%);
        color: #FFFFFF;
        padding: 10px 24px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 20px;
        letter-spacing: 1px;
        display: inline-block;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35);
    }
    .badge-fail {
        background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%);
        color: #FFFFFF;
        padding: 10px 24px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 20px;
        letter-spacing: 1px;
        display: inline-block;
        box-shadow: 0 4px 14px rgba(239, 68, 68, 0.35);
    }
    .badge-review {
        background: linear-gradient(135deg, #D97706 0%, #F59E0B 100%);
        color: #FFFFFF;
        padding: 10px 24px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 20px;
        letter-spacing: 1px;
        display: inline-block;
        box-shadow: 0 4px 14px rgba(245, 158, 11, 0.35);
    }

    /* Advisor container */
    .advisor-box {
        background-color: #0F172A;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ----------------- CACHED CORE ENGINES -----------------
@st.cache_resource
def load_core_systems():
    inspector_inst = PCBInspector()
    visualizer_inst = PCBVisualizer()
    reporter_inst = InspectionReporter()
    advisor_inst = PCBAdvisor()
    return inspector_inst, visualizer_inst, reporter_inst, advisor_inst


inspector, visualizer, reporter, advisor = load_core_systems()


# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.markdown("### ⚡ **EdgeInspect AI**")
    st.markdown("<span style='color:#00E5FF;font-size:12px;font-weight:bold;'>INDUSTRIAL PCB INSPECTOR</span>", unsafe_allow_html=True)
    st.markdown("---")

    nav_selection = st.radio(
        "Navigation",
        [
            "🔍 Inspect PCB (Ref vs Test)",
            "⚡ Standalone YOLO Detection",
            "📦 Batch PCB Inspection",
            "📜 Historical Audit Reports"
        ]
    )

    st.markdown("---")
    st.markdown("#### **QC Thresholds**")

    sidebar_conf = st.slider(
        "Model Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=DEFAULT_CONF_THRESHOLD,
        step=0.05,
        help="YOLO detection confidence cutoff."
    )

    sidebar_shift_tol = st.slider(
        "Max Centroid Shift Tolerance (px)",
        min_value=10.0,
        max_value=80.0,
        value=35.0,
        step=5.0,
        help="Displacement threshold before flagging a part as misplaced."
    )

    sidebar_match_iou = st.slider(
        "Component Pairing IoU",
        min_value=0.10,
        max_value=0.80,
        value=0.30,
        step=0.05,
        help="Spatial IoU threshold for matching reference to test components."
    )

    st.markdown("---")
    st.markdown("#### **Lot Information**")
    meta_board = st.text_input("Board Serial / Model", value="PCB-MAIN-REV3")
    meta_batch = st.text_input("Production Lot ID", value="LOT-2026-N48")
    meta_tech = st.text_input("QC Operator ID", value="OP-7041")

    st.markdown("---")
    st.markdown("""
    **Core Engine Details**:
    - **App Name**: EdgeInspect AI
    - **Engine**: BharatPCB-AI Local
    - **Model**: YOLOv8 (9 Classes)
    - **Device**: Apple Silicon Local (CPU)
    - **Model Hash**: `3bed9c01...5552`
    """)


# Helper function to generate clean sample reference PCB if user has no files
def get_sample_golden_board() -> np.ndarray:
    return generate_synthetic_pcb(800, 800, seed=42)


# ==============================================================================
# SECTION 1: REFERENCE VS TEST INSPECTION (MAIN PIPELINE)
# ==============================================================================
if nav_selection == "🔍 Inspect PCB (Ref vs Test)":
    st.markdown('<div class="brand-title">EdgeInspect AI — PCB Defect Inspection</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Automated Geometric Alignment, Multi-Class Component Detection & Defect Isolation</div>', unsafe_allow_html=True)

    # Dual Upload Columns
    c_ref, c_test = st.columns(2)

    with c_ref:
        st.markdown("#### 1. Reference PCB (Golden Board)")
        ref_mode = st.radio("Reference Input Mode", ["Upload Reference File", "Use Standard Golden Board"], horizontal=True, key="ref_mode")

        ref_img = None
        ref_filename = "golden_reference.png"

        if ref_mode == "Upload Reference File":
            up_ref = st.file_uploader(
                "Select Golden PCB (PNG, JPG)",
                type=["png", "jpg", "jpeg", "bmp", "tiff", "tif"],
                key="up_ref"
            )
            if up_ref:
                ref_img = decode_image_bytes(up_ref.read())
                ref_filename = up_ref.name
        else:
            ref_img = get_sample_golden_board()
            ref_filename = "standard_golden_board.png"
            st.caption("ℹ️ Using standard verified Golden Reference PCB.")

        if ref_img is not None:
            valid, msg = validate_image(ref_img)
            if valid:
                st.image(bgr_to_rgb(ref_img), caption=f"Golden Reference: {ref_filename} ({ref_img.shape[1]}x{ref_img.shape[0]} px)", use_container_width=True)
            else:
                st.error(f"Reference validation failed: {msg}")
                ref_img = None

    with c_test:
        st.markdown("#### 2. Test PCB (Inspected Board)")
        test_mode = st.radio("Test Input Mode", ["Upload Test File", "MacBook Camera Input"], horizontal=True, key="test_mode")

        test_img = None
        test_filename = "test_pcb.png"

        if test_mode == "Upload Test File":
            up_test = st.file_uploader(
                "Select Test PCB (PNG, JPG)",
                type=["png", "jpg", "jpeg", "bmp", "tiff", "tif"],
                key="up_test"
            )
            if up_test:
                test_img = decode_image_bytes(up_test.read())
                test_filename = up_test.name
        else:
            cam_feed = st.camera_input("Capture Test PCB via MacBook Camera", key="cam_feed")
            if cam_feed:
                test_img = decode_image_bytes(cam_feed.read())
                test_filename = f"macbook_capture_{datetime.now(timezone.utc).strftime('%H%M%S')}.png"

        if test_img is not None:
            valid, msg = validate_image(test_img)
            if valid:
                st.image(bgr_to_rgb(test_img), caption=f"Test PCB: {test_filename} ({test_img.shape[1]}x{test_img.shape[0]} px)", use_container_width=True)
            else:
                st.error(f"Test PCB validation failed: {msg}")
                test_img = None

    st.markdown("---")

    # Action Button
    ready_to_inspect = (ref_img is not None and test_img is not None)
    if not ready_to_inspect:
        st.info("💡 Please load both a Reference PCB image and a Test PCB image above to enable inspection.")

    inspect_btn = st.button("🔬 Inspect PCB", disabled=not ready_to_inspect, type="primary", use_container_width=True)

    if inspect_btn and ready_to_inspect:
        with st.status("Executing Industrial PCB Inspection Pipeline...", expanded=True) as status_log:
            st.write("1️⃣ Validating dimensional integrity and image channels...")
            
            st.write("2️⃣ Extracting ORB features and computing RANSAC homography alignment...")
            comp_cfg = ComparisonConfig(
                match_iou_threshold=sidebar_match_iou,
                max_centroid_drift=sidebar_shift_tol,
                min_confidence_threshold=sidebar_conf
            )

            st.write("3️⃣ Running dual-board YOLOv8 component detection using `models/best.pt`...")
            # Run inspection via public inspection function
            result = run_inspection(
                ref_img,
                test_img,
                backend="yolo",
                conf=sidebar_conf,
                comparison_config=comp_cfg
            )

            st.write("4️⃣ Analyzing spatial discrepancies (Missing, Misplaced, Extra, Mismatch)...")
            # Side-by-side visualization
            sbs_vis = visualizer.create_side_by_side_comparison(ref_img, result)

            st.write("5️⃣ Generating illumination-normalized structural difference heatmap...")
            heatmap_vis = visualizer.compute_structural_difference_heatmap(ref_img, result.aligned_test_image)

            st.write("6️⃣ Cropping localized defect regions for Defect Crop Panel...")
            defect_crops = visualizer.crop_defect_regions(ref_img, result.aligned_test_image, result)

            st.write("7️⃣ Compiling audit reports (PDF, CSV, JSON)...")
            report_paths = reporter.generate_all_reports(
                result,
                ref_image=ref_img,
                ref_name=ref_filename,
                test_name=test_filename
            )

            # Extract structured defect table records
            defect_records = extract_defect_records(
                result,
                img_w=ref_img.shape[1],
                img_h=ref_img.shape[0]
            )

            # Store in session state
            st.session_state["active_inspection"] = {
                "result": result,
                "sbs_vis": sbs_vis,
                "heatmap_vis": heatmap_vis,
                "defect_crops": defect_crops,
                "defect_records": defect_records,
                "report_paths": report_paths,
                "ref_img": ref_img,
                "test_img": test_img,
                "ref_name": ref_filename,
                "test_name": test_filename
            }

            status_log.update(label="✅ PCB Inspection Pipeline Completed!", state="complete", expanded=False)

    # ----------------- DISPLAY INSPECTION RESULTS -----------------
    if "active_inspection" in st.session_state:
        data = st.session_state["active_inspection"]
        res: InspectionResult = data["result"]
        summary = res.comparison

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### **Inspection Results Dashboard**")

        # 1. Main Verdict Header
        v = res.verdict
        if v == "PASS":
            st.markdown(f'<div class="badge-pass">PASS — ZERO CRITICAL DEFECTS</div> &nbsp;&nbsp;&nbsp; <span style="font-size:16px;">{res.verdict_reason}</span>', unsafe_allow_html=True)
        elif v == "FAIL":
            st.markdown(f'<div class="badge-fail">FAIL — DEFECTS DETECTED</div> &nbsp;&nbsp;&nbsp; <span style="font-size:16px;">{res.verdict_reason}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="badge-review">REVIEW REQUIRED</div> &nbsp;&nbsp;&nbsp; <span style="font-size:16px;">{res.verdict_reason}</span>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 2. Defect Counters
        cnt_missing = len(summary.missing) if summary else 0
        cnt_misplaced = len(summary.misplaced) if summary else 0
        cnt_extra = len(summary.extra) if summary else 0
        cnt_mismatch = len(summary.class_mismatches) if summary else 0
        cnt_matched = len(summary.matched) if summary else 0
        cnt_low_conf = len(summary.low_confidence_warnings) if summary else 0

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("❌ Missing", cnt_missing)
        m2.metric("⚠️ Misplaced", cnt_misplaced)
        m3.metric("🟣 Extra", cnt_extra)
        m4.metric("🛑 Mismatched", cnt_mismatch)
        m5.metric("✅ Matched", cnt_matched)
        m6.metric("🔍 Low Confidence", cnt_low_conf)

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Image Views Tabs
        tab_comp, tab_heat, tab_crops, tab_raw = st.tabs([
            "🖼️ Annotated Comparison",
            "🔥 Structural Difference Heatmap",
            "🔎 Defect Crop Panel",
            "📐 Aligned & Raw Boards"
        ])

        with tab_comp:
            st.image(bgr_to_rgb(data["sbs_vis"]), caption="Left: Golden Reference (Missing Callouts) | Right: Aligned Test PCB (Defect Overlays)", use_container_width=True)

        with tab_heat:
            st.image(bgr_to_rgb(data["heatmap_vis"]), caption="Structural Optical Differences (CLAHE Normalized Difference with JET Colormap)", use_container_width=True)
            st.caption("Warm regions (Red/Yellow) highlight areas of physical variance, component displacement, or surface anomalies.")

        with tab_crops:
            crops = data["defect_crops"]
            if crops:
                st.markdown(f"#### Localized Defect Crops ({len(crops)} Discrepancies)")
                # Grid display (4 per row)
                cols_per_row = 4
                for row_idx in range(0, len(crops), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for col_idx, crop_data in enumerate(crops[row_idx:row_idx + cols_per_row]):
                        with cols[col_idx]:
                            st.image(bgr_to_rgb(crop_data["crop_image"]), use_container_width=True)
                            badge_color = "#EF4444" if crop_data["defect_type"] == "MISSING" else ("#F59E0B" if crop_data["defect_type"] == "MISPLACED" else "#8B5CF6")
                            st.markdown(f"<span style='color:{badge_color};font-weight:bold;'>{crop_data['defect_id']} — {crop_data['defect_type']}</span>", unsafe_allow_html=True)
                            st.caption(f"{crop_data['component_name']} at {crop_data['location']}")
            else:
                st.success("✨ Zero defects to crop! All components conform to reference specification.")

        with tab_raw:
            raw_c1, raw_c2, raw_c3 = st.columns(3)
            with raw_c1:
                st.image(bgr_to_rgb(data["ref_img"]), caption="Original Golden Reference", use_container_width=True)
            with raw_c2:
                st.image(bgr_to_rgb(data["test_img"]), caption="Original Test Image", use_container_width=True)
            with raw_c3:
                if res.aligned_test_image is not None:
                    st.image(bgr_to_rgb(res.aligned_test_image), caption="Warped Aligned Test PCB", use_container_width=True)
                else:
                    st.warning("Alignment was rejected; no warped test image generated.")

        # 4. Itemized Defect Table
        st.markdown("---")
        st.markdown("#### **Itemized Defect List**")
        rec = data["defect_records"]
        if rec:
            df_defects = pd.DataFrame(rec)
            # Display cleanly formatted table
            st.dataframe(
                df_defects[["defect_id", "defect_type", "component_name", "confidence", "region", "expected_pos", "actual_pos", "explanation"]],
                use_container_width=True,
                hide_index=True
            )
        else:
            if res.can_proceed_to_comparison:
                st.success("✨ No component discrepancies found on this board.")
            else:
                st.warning("⚠️ Comparison was aborted due to rejected alignment. Zero false defects reported.")

        # 5. AI Advisor Section
        st.markdown("---")
        st.markdown("### 🤖 **AI Quality & Rework Advisor**")
        st.markdown(f"<div style='font-size:12px;color:#94A3B8;margin-bottom:8px;'>Engine Status: <b>{advisor.status_badge}</b></div>", unsafe_allow_html=True)

        advice = advisor.generate_inspection_advice(res)
        
        with st.container():
            st.markdown(f"**Risk Evaluation**: {advice['risk_assessment']}")
            
            c_rc, c_act = st.columns(2)
            with c_rc:
                st.markdown("##### 🔬 Root-Cause Analysis")
                for rc in advice["root_cause_analysis"]:
                    st.markdown(f"- {rc}")
            with c_act:
                st.markdown("##### 🛠️ Recommended Line Checks & Rework")
                for act in advice["recommended_actions"]:
                    st.markdown(f"- {act}")

        # Interactive Technician Q&A
        st.markdown("##### 💬 Ask the QC Advisor")
        user_q = st.text_input("Ask a question about this inspection (e.g., 'What caused the failure?', 'Which components are missing?', 'What rework should I perform?'):", key="advisor_q")
        if user_q:
            answer = advisor.answer_question(user_q, res)
            st.info(f"**Advisor Response:**\n\n{answer}")

        # 6. Reports & Downloads
        st.markdown("---")
        st.markdown("#### **Export Inspection Certificates & Audit Data**")
        d_paths: ReportPaths = data["report_paths"]
        d1, d2, d3, d4 = st.columns(4)

        if d_paths.pdf_path and d_paths.pdf_path.exists():
            with open(d_paths.pdf_path, "rb") as f:
                d1.download_button("📄 Download PDF Certificate", f.read(), file_name=d_paths.pdf_path.name, mime="application/pdf", use_container_width=True)

        if d_paths.csv_path and d_paths.csv_path.exists():
            with open(d_paths.csv_path, "rb") as f:
                d2.download_button("📊 Download CSV Audit Log", f.read(), file_name=d_paths.csv_path.name, mime="text/csv", use_container_width=True)

        if d_paths.json_path and d_paths.json_path.exists():
            with open(d_paths.json_path, "rb") as f:
                d3.download_button("📋 Download JSON Record", f.read(), file_name=d_paths.json_path.name, mime="application/json", use_container_width=True)

        # PNG Image download
        _, png_enc = cv2.imencode(".png", data["sbs_vis"])
        d4.download_button("🖼️ Download Annotated PNG", png_enc.tobytes(), file_name=f"inspection_{data['test_name']}.png", mime="image/png", use_container_width=True)


# ==============================================================================
# SECTION 2: STANDALONE YOLO DETECTION
# ==============================================================================
elif nav_selection == "⚡ Standalone YOLO Detection":
    st.markdown('<div class="brand-title">Standalone YOLO Component Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">High-Speed Local Component Identification on Single PCB Image (No Reference Board)</div>', unsafe_allow_html=True)

    s_source = st.radio("Image Input Mode", ["Upload PCB Image", "MacBook Camera Input", "Load Standard PCB Canvas"], horizontal=True)
    s_img = None
    s_filename = "pcb_target.png"

    if s_source == "Upload PCB Image":
        s_file = st.file_uploader("Upload Image (PNG, JPG)", type=["png", "jpg", "jpeg", "bmp"])
        if s_file:
            s_img = decode_image_bytes(s_file.read())
            s_filename = s_file.name
    elif s_source == "MacBook Camera Input":
        s_cam = st.camera_input("Capture Frame")
        if s_cam:
            s_img = decode_image_bytes(s_cam.read())
            s_filename = "camera_single.png"
    else:
        s_img = get_sample_golden_board()
        s_filename = "sample_canvas.png"

    if s_img is not None:
        valid, msg = validate_image(s_img)
        if valid:
            col_orig, col_annot = st.columns(2)
            with col_orig:
                st.image(bgr_to_rgb(s_img), caption=f"Input Image ({s_img.shape[1]}x{s_img.shape[0]} px)", use_container_width=True)

            with st.spinner("Detecting components with local YOLO model..."):
                dets = inspector.detector.predict(s_img, conf=sidebar_conf)
                annot_img = visualizer.draw_detections(s_img, dets, title=f"{len(dets)} Components Detected")

            with col_annot:
                st.image(bgr_to_rgb(annot_img), caption=f"Detections ({len(dets)} components)", use_container_width=True)

            st.markdown(f"#### Detection Summary ({len(dets)} components)")
            if dets:
                df_single = pd.DataFrame([d.to_dict() for d in dets])
                st.dataframe(df_single, use_container_width=True)
            else:
                st.info("No components detected above selected confidence cutoff.")
        else:
            st.error(f"Image error: {msg}")


# ==============================================================================
# SECTION 3: BATCH PCB INSPECTION
# ==============================================================================
elif nav_selection == "📦 Batch PCB Inspection":
    st.markdown('<div class="brand-title">Batch PCB Inspection</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Automated High-Throughput Inspection Against Single Golden Reference</div>', unsafe_allow_html=True)

    b_ref_mode = st.radio("Reference Board", ["Standard Golden PCB", "Upload Reference File"], horizontal=True)
    batch_ref_img = None
    if b_ref_mode == "Standard Golden PCB":
        batch_ref_img = get_sample_golden_board()
    else:
        b_up = st.file_uploader("Upload Golden Reference", type=["png", "jpg", "jpeg", "bmp"])
        if b_up:
            batch_ref_img = decode_image_bytes(b_up.read())

    batch_test_files = st.file_uploader("Upload Multiple Test PCB Images", type=["png", "jpg", "jpeg", "bmp"], accept_multiple_files=True)

    if st.button("🚀 Execute Batch Inspection", disabled=(batch_ref_img is None or not batch_test_files), type="primary"):
        p_bar = st.progress(0.0)
        p_status = st.empty()

        results_data = []
        counts = {"PASS": 0, "FAIL": 0, "REVIEW": 0}

        for idx, f in enumerate(batch_test_files):
            p_status.text(f"Inspecting ({idx+1}/{len(batch_test_files)}): {f.name}")
            t_bytes = f.read()
            t_image = decode_image_bytes(t_bytes)
            if t_image is None:
                continue

            cfg = ComparisonConfig(
                match_iou_threshold=sidebar_match_iou,
                max_centroid_drift=sidebar_shift_tol,
                min_confidence_threshold=sidebar_conf
            )
            res = run_inspection(batch_ref_img, t_image, conf=sidebar_conf, comparison_config=cfg)
            counts[res.verdict] = counts.get(res.verdict, 0) + 1

            results_data.append({
                "Board Name": f.name,
                "Verdict": res.verdict,
                "Missing": len(res.comparison.missing) if res.comparison else 0,
                "Misplaced": len(res.comparison.misplaced) if res.comparison else 0,
                "Extra": len(res.comparison.extra) if res.comparison else 0,
                "Matched": len(res.comparison.matched) if res.comparison else 0,
                "Alignment": res.alignment.metrics.status,
                "Reason": res.verdict_reason
            })

            p_bar.progress((idx + 1) / len(batch_test_files))

        p_status.text("Batch processing completed!")

        st.markdown("### **Batch QC Analytics**")
        tot = len(results_data)
        yield_rate = (counts["PASS"] / tot * 100) if tot > 0 else 0

        bc1, bc2, bc3, bc4, bc5 = st.columns(5)
        bc1.metric("Total Inspected", tot)
        bc2.metric("✅ Passed", counts["PASS"])
        bc3.metric("❌ Failed", counts["FAIL"])
        bc4.metric("⚠️ Review Required", counts["REVIEW"])
        bc5.metric("Yield Rate", f"{yield_rate:.1f}%")

        df_b = pd.DataFrame(results_data)
        st.dataframe(df_b, use_container_width=True)

        st.download_button(
            "📊 Download Batch Inspection Summary (CSV)",
            data=df_b.to_csv(index=False).encode("utf-8"),
            file_name=f"batch_summary_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


# ==============================================================================
# SECTION 4: INSPECTION HISTORY & REPORTS
# ==============================================================================
elif nav_selection == "📜 Historical Audit Reports":
    st.markdown('<div class="brand-title">Inspection History & Report Archive</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Persistent Local Audit Records and Certificates (Zero Cloud Storage)</div>', unsafe_allow_html=True)

    report_list = []
    if REPORTS_DIR.exists():
        for p in sorted(REPORTS_DIR.glob("*.*"), key=lambda x: x.stat().st_mtime, reverse=True):
            if p.suffix.lower() in {".pdf", ".csv", ".json"}:
                report_list.append({
                    "Filename": p.name,
                    "Format": p.suffix.upper().replace(".", ""),
                    "Size (KB)": round(p.stat().st_size / 1024, 1),
                    "Timestamp": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    "Filepath": p
                })

    if report_list:
        df_r = pd.DataFrame(report_list)
        st.dataframe(df_r[["Filename", "Format", "Size (KB)", "Timestamp"]], use_container_width=True, hide_index=True)

        sel_name = st.selectbox("Select Report to Export", df_r["Filename"].tolist())
        target = next(r["Filepath"] for r in report_list if r["Filename"] == sel_name)

        with open(target, "rb") as f:
            data_bytes = f.read()

        mtype = "application/pdf" if target.suffix == ".pdf" else ("text/csv" if target.suffix == ".csv" else "application/json")
        st.download_button(
            f"⬇️ Download {sel_name}",
            data=data_bytes,
            file_name=target.name,
            mime=mtype,
            type="primary"
        )
    else:
        st.info("No prior reports located in outputs/reports/.")
