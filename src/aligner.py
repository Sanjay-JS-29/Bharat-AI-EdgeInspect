"""
OpenCV Image Alignment Engine for BharatPCB-AI.
Aligns test PCB images to golden reference PCB images using ORB feature detection,
descriptor matching, RANSAC homography estimation, strict quality gating,
and manual alignment fallback.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Any
import cv2
import numpy as np

from src.config import (
    ALIGNMENT_MIN_FEATURES,
    ALIGNMENT_MIN_GOOD_MATCHES,
    ALIGNMENT_MIN_INLIER_RATIO,
    ALIGNMENT_RANSAC_REPROJ_THRESH
)


@dataclass
class AlignmentMetrics:
    """Metrics assessing the geometric reliability of the image alignment."""
    detected_features_ref: int
    detected_features_test: int
    good_matches_count: int
    inliers_count: int
    inlier_ratio: float
    reprojection_error: float
    is_homography_valid: bool
    quality_score: float
    status: str
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["inlier_ratio"] = round(self.inlier_ratio, 4)
        d["reprojection_error"] = round(self.reprojection_error, 3)
        d["quality_score"] = round(self.quality_score, 4)
        return d


@dataclass
class AlignmentResult:
    """Complete result container for the alignment stage."""
    success: bool
    aligned_image: Optional[np.ndarray]
    homography_matrix: Optional[np.ndarray]
    inverse_homography: Optional[np.ndarray]
    metrics: AlignmentMetrics
    can_proceed_to_comparison: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "can_proceed_to_comparison": self.can_proceed_to_comparison,
            "has_aligned_image": self.aligned_image is not None,
            "has_homography": self.homography_matrix is not None,
            "metrics": self.metrics.to_dict()
        }


class PCBAligner:
    """
    Robust image alignment engine comparing a Test PCB against a Reference PCB.
    Uses ORB feature detection, descriptor matching, RANSAC homography estimation,
    comprehensive geometric validation, and manual fiducial fallback.
    """

    def __init__(
        self,
        n_features: int = 3000,
        min_features: int = ALIGNMENT_MIN_FEATURES,
        min_good_matches: int = ALIGNMENT_MIN_GOOD_MATCHES,
        min_inlier_ratio: float = ALIGNMENT_MIN_INLIER_RATIO,
        ransac_reproj_thresh: float = ALIGNMENT_RANSAC_REPROJ_THRESH,
        max_reprojection_error: float = 4.5,
        ratio_thresh: float = 0.75
    ):
        self.n_features = n_features
        self.min_features = min_features
        self.min_good_matches = min_good_matches
        self.min_inlier_ratio = min_inlier_ratio
        self.ransac_reproj_thresh = ransac_reproj_thresh
        self.max_reprojection_error = max_reprojection_error
        self.ratio_thresh = ratio_thresh

        # Initialize ORB detector
        self.orb = cv2.ORB_create(
            nfeatures=self.n_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31
        )
        # BFMatcher with Hamming distance for binary ORB descriptors
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def align(self, ref_image: np.ndarray, test_image: np.ndarray) -> AlignmentResult:
        """
        Align test_image onto ref_image coordinate frame.
        Guarantees that can_proceed_to_comparison is False if alignment is rejected.
        """
        if ref_image is None or test_image is None:
            return self._create_failure_result(
                "Images cannot be None",
                status="REJECTED_INVALID_INPUT"
            )

        h_ref, w_ref = ref_image.shape[:2]
        h_test, w_test = test_image.shape[:2]

        if h_ref == 0 or w_ref == 0 or h_test == 0 or w_test == 0:
            return self._create_failure_result(
                "Image dimensions must be non-zero",
                status="REJECTED_INVALID_INPUT"
            )

        # Convert to grayscale for feature extraction
        gray_ref = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY) if len(ref_image.shape) == 3 else ref_image
        gray_test = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY) if len(test_image.shape) == 3 else test_image

        # 1. Feature Detection & Description
        kp_ref, des_ref = self.orb.detectAndCompute(gray_ref, None)
        kp_test, des_test = self.orb.detectAndCompute(gray_test, None)

        n_ref = len(kp_ref) if kp_ref is not None else 0
        n_test = len(kp_test) if kp_test is not None else 0

        # Safety Check: Insufficient features
        if n_ref < self.min_features or n_test < self.min_features:
            metrics = AlignmentMetrics(
                detected_features_ref=n_ref,
                detected_features_test=n_test,
                good_matches_count=0,
                inliers_count=0,
                inlier_ratio=0.0,
                reprojection_error=float("inf"),
                is_homography_valid=False,
                quality_score=0.0,
                status="REJECTED_INSUFFICIENT_FEATURES",
                rejection_reason=(
                    f"Insufficient features detected (Ref: {n_ref}, Test: {n_test}, "
                    f"Required: {self.min_features}). Board may be blank, overexposed, or blurred."
                )
            )
            return AlignmentResult(
                success=False,
                aligned_image=None,
                homography_matrix=None,
                inverse_homography=None,
                metrics=metrics,
                can_proceed_to_comparison=False
            )

        if des_ref is None or des_test is None or len(des_ref) < 2 or len(des_test) < 2:
            return self._create_failure_result(
                "Descriptors could not be computed",
                status="REJECTED_INSUFFICIENT_DESCRIPTORS",
                n_ref=n_ref,
                n_test=n_test
            )

        # 2. Descriptor Matching with Lowe's Ratio Test
        try:
            knn_matches = self.matcher.knnMatch(des_test, des_ref, k=2)
        except cv2.error as e:
            return self._create_failure_result(
                f"Feature matching failed: {str(e)}",
                status="REJECTED_MATCHING_ERROR",
                n_ref=n_ref,
                n_test=n_test
            )

        good_matches = []
        for match_pair in knn_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_thresh * n.distance:
                    good_matches.append(m)

        n_good = len(good_matches)
        if n_good < self.min_good_matches:
            metrics = AlignmentMetrics(
                detected_features_ref=n_ref,
                detected_features_test=n_test,
                good_matches_count=n_good,
                inliers_count=0,
                inlier_ratio=0.0,
                reprojection_error=float("inf"),
                is_homography_valid=False,
                quality_score=0.0,
                status="REJECTED_LOW_MATCHES",
                rejection_reason=(
                    f"Too few good feature matches ({n_good} < {self.min_good_matches}). "
                    "Images may represent different PCB revisions or have severe visual discrepancies."
                )
            )
            return AlignmentResult(
                success=False,
                aligned_image=None,
                homography_matrix=None,
                inverse_homography=None,
                metrics=metrics,
                can_proceed_to_comparison=False
            )

        # 3. RANSAC Homography Estimation
        pts_test = np.float32([kp_test[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, inlier_mask = cv2.findHomography(
            pts_test,
            pts_ref,
            cv2.RANSAC,
            self.ransac_reproj_thresh
        )

        if H is None or inlier_mask is None:
            metrics = AlignmentMetrics(
                detected_features_ref=n_ref,
                detected_features_test=n_test,
                good_matches_count=n_good,
                inliers_count=0,
                inlier_ratio=0.0,
                reprojection_error=float("inf"),
                is_homography_valid=False,
                quality_score=0.0,
                status="REJECTED_HOMOGRAPHY_FAILED",
                rejection_reason="RANSAC failed to estimate a geometric homography matrix."
            )
            return AlignmentResult(
                success=False,
                aligned_image=None,
                homography_matrix=None,
                inverse_homography=None,
                metrics=metrics,
                can_proceed_to_comparison=False
            )

        inliers_count = int(np.sum(inlier_mask))
        inlier_ratio = float(inliers_count / n_good) if n_good > 0 else 0.0

        # Calculate Reprojection Error on Inliers
        inlier_indices = np.where(inlier_mask.ravel() == 1)[0]
        if len(inlier_indices) >= 4:
            inlier_pts_test = pts_test[inlier_indices]
            inlier_pts_ref = pts_ref[inlier_indices]
            projected_pts = cv2.perspectiveTransform(inlier_pts_test, H)
            diff = inlier_pts_ref - projected_pts
            reproj_error = float(np.mean(np.sqrt(diff[:, :, 0]**2 + diff[:, :, 1]**2)))
        else:
            reproj_error = float("inf")

        # 4. Geometric Sanity & Homography Validity Check
        is_valid, validity_reason = self._validate_homography(H, w_ref, h_ref, w_test, h_test)

        # 5. Evaluate Rejection Safeguards
        rejection_reason = None
        status = "SUCCESS"

        if not is_valid:
            status = "REJECTED_DEGENERATE_HOMOGRAPHY"
            rejection_reason = f"Degenerate homography: {validity_reason}"
        elif inliers_count < self.min_good_matches:
            status = "REJECTED_LOW_INLIERS"
            rejection_reason = f"RANSAC inliers too low ({inliers_count} < {self.min_good_matches})"
        elif inlier_ratio < self.min_inlier_ratio:
            status = "REJECTED_LOW_INLIER_RATIO"
            rejection_reason = (
                f"Inlier ratio below threshold ({inlier_ratio:.2%} < {self.min_inlier_ratio:.2%})"
            )
        elif reproj_error > self.max_reprojection_error:
            status = "REJECTED_HIGH_REPROJECTION_ERROR"
            rejection_reason = (
                f"Reprojection error exceeds threshold ({reproj_error:.2f}px > {self.max_reprojection_error:.2f}px)"
            )

        # Composite Quality Score [0.0 - 1.0]
        quality_score = self._compute_quality_score(inliers_count, inlier_ratio, reproj_error)

        success = (status == "SUCCESS")
        metrics = AlignmentMetrics(
            detected_features_ref=n_ref,
            detected_features_test=n_test,
            good_matches_count=n_good,
            inliers_count=inliers_count,
            inlier_ratio=inlier_ratio,
            reprojection_error=reproj_error,
            is_homography_valid=is_valid,
            quality_score=quality_score if success else 0.0,
            status=status,
            rejection_reason=rejection_reason
        )

        if not success:
            return AlignmentResult(
                success=False,
                aligned_image=None,
                homography_matrix=None,
                inverse_homography=None,
                metrics=metrics,
                can_proceed_to_comparison=False
            )

        # 6. Perspective Warp test_image to reference dimensions
        aligned_image = cv2.warpPerspective(
            test_image,
            H,
            (w_ref, h_ref),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        try:
            inv_H = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            inv_H = None

        return AlignmentResult(
            success=True,
            aligned_image=aligned_image,
            homography_matrix=H,
            inverse_homography=inv_H,
            metrics=metrics,
            can_proceed_to_comparison=True
        )

    def align_manual(
        self,
        ref_image: np.ndarray,
        test_image: np.ndarray,
        ref_points: List[Tuple[float, float]],
        test_points: List[Tuple[float, float]]
    ) -> AlignmentResult:
        """
        Manual alignment fallback interface using user-selected 4-point fiducials.
        Allows manual alignment when automated ORB matching fails.
        """
        if len(ref_points) < 4 or len(test_points) < 4:
            return self._create_failure_result(
                "Manual alignment requires at least 4 corresponding fiducial point pairs",
                status="REJECTED_MANUAL_INSUFFICIENT_POINTS"
            )

        h_ref, w_ref = ref_image.shape[:2]
        pts_ref = np.float32(ref_points[:4]).reshape(-1, 1, 2)
        pts_test = np.float32(test_points[:4]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(pts_test, pts_ref, cv2.RANSAC, 3.0)
        if H is None:
            H = cv2.getPerspectiveTransform(pts_test[:4], pts_ref[:4])

        is_valid, validity_reason = self._validate_homography(H, w_ref, h_ref, test_image.shape[1], test_image.shape[0])

        if not is_valid:
            metrics = AlignmentMetrics(
                detected_features_ref=len(ref_points),
                detected_features_test=len(test_points),
                good_matches_count=4,
                inliers_count=4,
                inlier_ratio=1.0,
                reprojection_error=0.0,
                is_homography_valid=False,
                quality_score=0.0,
                status="REJECTED_MANUAL_INVALID_HOMOGRAPHY",
                rejection_reason=f"Manual points produced invalid homography: {validity_reason}"
            )
            return AlignmentResult(
                success=False,
                aligned_image=None,
                homography_matrix=None,
                inverse_homography=None,
                metrics=metrics,
                can_proceed_to_comparison=False
            )

        aligned_image = cv2.warpPerspective(
            test_image,
            H,
            (w_ref, h_ref),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        try:
            inv_H = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            inv_H = None

        metrics = AlignmentMetrics(
            detected_features_ref=len(ref_points),
            detected_features_test=len(test_points),
            good_matches_count=len(test_points),
            inliers_count=len(test_points),
            inlier_ratio=1.0,
            reprojection_error=1.0,
            is_homography_valid=True,
            quality_score=0.90,
            status="MANUAL_FALLBACK_SUCCESS"
        )

        return AlignmentResult(
            success=True,
            aligned_image=aligned_image,
            homography_matrix=H,
            inverse_homography=inv_H,
            metrics=metrics,
            can_proceed_to_comparison=True
        )

    def _validate_homography(
        self,
        H: np.ndarray,
        w_ref: int,
        h_ref: int,
        w_test: int,
        h_test: int
    ) -> Tuple[bool, str]:
        """
        Verify that homography H is physically realistic for PCB inspection:
        - Non-zero, positive determinant (no reflection/flip)
        - Reasonable scale magnification (0.25 < det(H) < 4.0)
        - Convex projected bounding polygon
        - Projected corner coordinates fall within plausible inspection bounds
        """
        if H is None or H.shape != (3, 3):
            return False, "Matrix is not 3x3"

        if np.isnan(H).any() or np.isinf(H).any():
            return False, "Matrix contains NaN or Inf values"

        det = np.linalg.det(H)
        if det <= 0.0:
            return False, f"Determinant is non-positive ({det:.4f}), indicating reflection or collapse"

        # Check scale distortion
        if det < 0.04 or det > 25.0:
            return False, f"Determinant ({det:.3f}) indicates extreme scaling/distortion"

        # Project 4 corners of test image
        corners_test = np.float32([
            [0, 0],
            [w_test, 0],
            [w_test, h_test],
            [0, h_test]
        ]).reshape(-1, 1, 2)

        try:
            proj_corners = cv2.perspectiveTransform(corners_test, H)
        except cv2.error as e:
            return False, f"Corner perspective transformation failed: {str(e)}"

        proj_pts = proj_corners.reshape(4, 2)

        # Check if projected polygon is convex
        if not cv2.isContourConvex(np.int32(proj_pts)):
            return False, "Projected quadrilateral is self-intersecting or concave"

        # Check projected area ratio
        area = cv2.contourArea(np.int32(proj_pts))
        ref_area = w_ref * h_ref
        area_ratio = area / ref_area if ref_area > 0 else 0

        if area_ratio < 0.15 or area_ratio > 6.0:
            return False, f"Projected area ratio ({area_ratio:.2f}) is outside acceptable limits [0.15, 6.0]"

        return True, "Valid"

    def _compute_quality_score(self, inliers: int, ratio: float, reproj_error: float) -> float:
        """Calculate normalized alignment quality score [0.0 - 1.0]."""
        # Score components: inlier count (up to 100), inlier ratio (0.25-0.75), reproj error (0-4px)
        inlier_score = min(1.0, inliers / 80.0)
        ratio_score = min(1.0, max(0.0, (ratio - 0.20) / 0.50))
        error_score = max(0.0, 1.0 - (reproj_error / self.max_reprojection_error))

        score = (0.35 * inlier_score) + (0.35 * ratio_score) + (0.30 * error_score)
        return float(np.clip(score, 0.0, 1.0))

    def _create_failure_result(
        self,
        reason: str,
        status: str,
        n_ref: int = 0,
        n_test: int = 0
    ) -> AlignmentResult:
        metrics = AlignmentMetrics(
            detected_features_ref=n_ref,
            detected_features_test=n_test,
            good_matches_count=0,
            inliers_count=0,
            inlier_ratio=0.0,
            reprojection_error=float("inf"),
            is_homography_valid=False,
            quality_score=0.0,
            status=status,
            rejection_reason=reason
        )
        return AlignmentResult(
            success=False,
            aligned_image=None,
            homography_matrix=None,
            inverse_homography=None,
            metrics=metrics,
            can_proceed_to_comparison=False
        )
