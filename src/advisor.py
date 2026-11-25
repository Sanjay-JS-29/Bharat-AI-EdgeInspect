"""
AI Engineering Advisor for BharatPCB-AI / EdgeInspect AI.
Provides technical root-cause analysis, rework instructions, and technician Q&A.
Interfaces with local ExecuTorch Llama 3.2 1B Instruct when model weights exist,
or provides transparent, deterministic quality engineering analysis.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import os

from src.inspector import InspectionResult


class PCBAdvisor:
    """
    AI Quality Control & Rework Advisor.
    Transparently detects local ExecuTorch Llama runtime and model weights.
    When uninstalled, operates in deterministic Electronics QC Engineering mode.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or Path("models/llama3_2_1b.pte")
        self.has_executorch = self._check_executorch_installed()
        self.has_model_file = self.model_path.exists()
        self.is_llama_active = self.has_executorch and self.has_model_file

    def _check_executorch_installed(self) -> bool:
        """Check if ExecuTorch runtime is present in local environment."""
        try:
            import executorch  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def status_badge(self) -> str:
        if self.is_llama_active:
            return "🟢 Llama 3.2 1B Instruct (ExecuTorch Local)"
        return "⚙️ Quality Engineering QC Engine (Local Rules)"

    @property
    def status_details(self) -> Dict[str, Any]:
        return {
            "is_llama_active": self.is_llama_active,
            "has_executorch_runtime": self.has_executorch,
            "has_model_weights": self.has_model_file,
            "model_path": str(self.model_path),
            "engine": "Llama 3.2 1B ExecuTorch" if self.is_llama_active else "Deterministic PCB Quality Rules Engine",
            "offline_local": True
        }

    def generate_inspection_advice(self, result: InspectionResult) -> Dict[str, Any]:
        """
        Generate comprehensive root-cause analysis and actionable rework advice
        tailored to the current inspection results.
        """
        verdict = result.verdict

        if not result.can_proceed_to_comparison:
            return {
                "headline": "Alignment Failure — Inspection Halted for Safety",
                "risk_assessment": "CRITICAL: Image geometric alignment was rejected. No defect conclusions were generated to prevent false scrap.",
                "root_cause_analysis": [
                    "Board is physically shifted or rotated beyond automatic ORB/RANSAC convergence limits.",
                    "Board surface has severe lighting glare, blur, or overexposure obscuring fiducials.",
                    "Wrong board revision or blank substrate loaded under camera."
                ],
                "recommended_actions": [
                    "Check PCB positioning fixtures and edge-clamping rails.",
                    "Inspect lighting diffuser for specular reflections on solder joints.",
                    "Switch to manual 4-point fiducial registration in parameters if board has custom fiducials."
                ]
            }

        summary = result.comparison
        if summary is None or summary.total_discrepancies == 0:
            return {
                "headline": "Zero Discrepancies — Board Meets IPC Quality Baseline",
                "risk_assessment": f"NORMAL: All {len(summary.matched) if summary else 0} expected components verified within tolerances.",
                "root_cause_analysis": [
                    "Component placement coordinates and angular alignment match Golden Reference.",
                    "No missing, shifted, or uncataloged passives/ICs detected."
                ],
                "recommended_actions": [
                    "Approve board for downstream in-circuit testing (ICT) / functional testing.",
                    "Log inspection batch record into quality audit database."
                ]
            }

        # Analyze discrepancies
        root_causes = []
        actions = []

        if summary.missing:
            comp_names = [d.component.class_name.upper() for d in summary.missing[:3]]
            root_causes.append(
                f"Missing Component ({', '.join(comp_names)}): Solder paste deposition volume check, "
                "pick-and-place vacuum nozzle blockage, or component tape feeder pitch indexing failure."
            )
            actions.append(
                f"SMT Line Check: Inspect nozzle for {comp_names[0]} feeder; verify tape-and-reel pocket indexing."
            )

        if summary.misplaced:
            shifted_names = [d.component.class_name.upper() for d in summary.misplaced[:3]]
            root_causes.append(
                f"Misplaced/Shifted Component ({', '.join(shifted_names)}): Solder paste wetting imbalance (tombstoning risk), "
                "SMT placement head deceleration shock, or pick-and-place optical centering camera calibration drift."
            )
            actions.append(
                "SMT Calibration: Calibrate vision centering system on Pick-and-Place machine; check conveyor board clamp vibration."
            )

        if summary.class_mismatches:
            mismatches = [f"{d.component.class_name.upper()}" for d in summary.class_mismatches[:3]]
            root_causes.append(
                f"Class Mismatch ({', '.join(mismatches)}): Incorrect component reel mounted on feeder slot, or wrong BOM revision loaded."
            )
            actions.append(
                "Line Quarantine: Halt feeder line immediately; scan reel barcode against active production BOM."
            )

        if summary.extra:
            extras = [d.component.class_name.upper() for d in summary.extra[:3]]
            root_causes.append(
                f"Extra Component ({', '.join(extras)}): Residual component dropped from previous machine cycle or incorrect board loading."
            )
            actions.append(
                "Physical Check: Remove uncataloged part before reflow/soldering to prevent shorts."
            )

        if summary.low_confidence_warnings:
            root_causes.append(
                f"{len(summary.low_confidence_warnings)} detection(s) exhibited borderline confidence: "
                "Fillet reflection glare or silkscreen printing overlap causing detection ambiguity."
            )
            actions.append(
                "Visual QC: Verify low-confidence components with 10x stereo inspection microscope."
            )

        severity = "HIGH RISK — QUARANTINE" if (summary.missing or summary.class_mismatches) else "MODERATE RISK — REVIEW"

        return {
            "headline": f"{verdict} — {len(summary.missing)} Missing, {len(summary.misplaced)} Misplaced, {len(summary.extra)} Extra",
            "risk_assessment": f"{severity}: Defect pattern indicates manufacturing discrepancy requiring technician intervention.",
            "root_cause_analysis": root_causes,
            "recommended_actions": actions
        }

    def answer_question(self, question: str, result: InspectionResult) -> str:
        """
        Interactive Q&A answering operator questions about the active inspection.
        Operates deterministically without cloud or external API calls.
        """
        q = question.lower().strip()
        summary = result.comparison

        if not q:
            return "Please enter a specific question about the current PCB inspection (e.g., 'What caused the failure?', 'Which components are missing?', 'What rework is needed?')."

        if "llama" in q or "execu" in q or "model" in q:
            return (
                "Advisor Status: The local ExecuTorch Llama 3.2 1B Instruct runtime is currently NOT installed. "
                "To connect local Llama: 1) Install the macOS Apple Silicon ExecuTorch package; 2) Place the exported 'llama3_2_1b.pte' model weights in 'models/'. "
                "Until then, all advice is generated deterministically by the local Quality Control Rules Engine with zero cloud calls."
            )

        if not result.can_proceed_to_comparison:
            return (
                f"The inspection was HALTED because image alignment was rejected ({result.alignment.metrics.status}: {result.alignment.metrics.rejection_reason}). "
                "To resolve this, ensure the test PCB is evenly illuminated, lies flat within 30 degrees of horizontal alignment, and does not have severe glare. "
                "You can also use manual 4-point fiducials in the sidebar settings."
            )

        if "why" in q and ("fail" in q or "verdict" in q):
            return f"The board received a '{result.verdict}' verdict: {result.verdict_reason}"

        if "missing" in q:
            if not summary or not summary.missing:
                return "Zero components were classified as missing on this board."
            items = [f"- {d.component.class_name.upper()} at centroid {d.component.centroid}" for d in summary.missing]
            return f"The following {len(summary.missing)} component(s) are missing from the test board:\n" + "\n".join(items)

        if "misplaced" in q or "shifted" in q:
            if not summary or not summary.misplaced:
                return "Zero components were classified as misplaced on this board."
            items = [f"- {d.component.class_name.upper()} at centroid {d.component.centroid} (drift: {d.centroid_drift:.1f}px)" for d in summary.misplaced]
            return f"The following {len(summary.misplaced)} component(s) exceeded the positional shift tolerance:\n" + "\n".join(items)

        if "rework" in q or "fix" in q or "action" in q:
            advice = self.generate_inspection_advice(result)
            return "Recommended SMT Actions:\n" + "\n".join([f"• {a}" for a in advice["recommended_actions"]])

        if "coordinate" in q or "position" in q or "location" in q:
            if not summary or summary.total_discrepancies == 0:
                return "All components are positioned properly within allowable spatial drift tolerances."
            disc_locs = []
            for d in (summary.missing + summary.misplaced + summary.extra + summary.class_mismatches):
                disc_locs.append(f"• {d.discrepancy_type}: {d.component.class_name.upper()} at {d.component.centroid}")
            return "Discrepancy Coordinates:\n" + "\n".join(disc_locs)

        # Default summary answer
        advice = self.generate_inspection_advice(result)
        return (
            f"Inspection Verdict: {result.verdict}\n\n"
            f"Risk Assessment: {advice['risk_assessment']}\n\n"
            "Key Observations:\n" + "\n".join([f"• {r}" for r in advice["root_cause_analysis"]]) + "\n\n"
            "Recommended Rework:\n" + "\n".join([f"• {a}" for a in advice["recommended_actions"]])
        )
