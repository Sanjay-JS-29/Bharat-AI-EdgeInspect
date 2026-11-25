"""
YOLO Object Detection Engine for BharatPCB-AI.
Safely loads models/best.pt in read-only mode and executes PCB component detection.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
import numpy as np
import torch
from ultralytics import YOLO

from src.config import (
    MODEL_PATH,
    CLASS_MAP,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_IMG_SIZE
)


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box: List[float]  # [x1, y1, x2, y2] in pixel coordinates

    @property
    def width(self) -> float:
        return max(0.0, self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return max(0.0, self.box[3] - self.box[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centroid(self) -> Tuple[float, float]:
        cx = (self.box[0] + self.box[2]) / 2.0
        cy = (self.box[1] + self.box[3]) / 2.0
        return (round(cx, 2), round(cy, 2))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["centroid"] = self.centroid
        d["width"] = round(self.width, 2)
        d["height"] = round(self.height, 2)
        d["area"] = round(self.area, 2)
        d["confidence"] = round(self.confidence, 4)
        return d


class PCBDetector:
    """Encapsulates YOLOv8 inference for PCB component detection."""

    def __init__(self, model_path: Union[str, Path] = MODEL_PATH, device: Optional[str] = None):
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Trained YOLO model not found at: {self.model_path}")

        self.device = self._resolve_device(device)
        # Load model strictly for read-only inference
        self.model = YOLO(str(self.model_path))
        self.classes: Dict[int, str] = self.model.names if hasattr(self.model, "names") else CLASS_MAP

    def _resolve_device(self, preferred_device: Optional[str] = None) -> str:
        """Resolve device for Apple Silicon macOS (MPS or CPU fallback)."""
        if preferred_device:
            return preferred_device
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        return "cpu"

    def predict(
        self,
        source: Union[str, Path, np.ndarray],
        conf: float = DEFAULT_CONF_THRESHOLD,
        iou: float = DEFAULT_IOU_THRESHOLD,
        imgsz: int = DEFAULT_IMG_SIZE
    ) -> List[Detection]:
        """
        Run inference on a single image (file path or numpy array).
        Returns a sorted list of Detection objects (highest confidence first).
        """
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            verbose=False
        )

        detections: List[Detection] = []
        if not results:
            return detections

        first_res = results[0]
        boxes = first_res.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        coords = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for coord, score, cls_id in zip(coords, confs, classes):
            cls_name = self.classes.get(cls_id, f"class_{cls_id}")
            detections.append(
                Detection(
                    class_id=int(cls_id),
                    class_name=str(cls_name),
                    confidence=float(score),
                    box=[float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3])]
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def predict_batch(
        self,
        sources: List[Union[str, Path, np.ndarray]],
        conf: float = DEFAULT_CONF_THRESHOLD,
        iou: float = DEFAULT_IOU_THRESHOLD,
        imgsz: int = DEFAULT_IMG_SIZE
    ) -> List[List[Detection]]:
        """Run inference across a batch of images."""
        return [self.predict(src, conf=conf, iou=iou, imgsz=imgsz) for src in sources]


# Alias for compatibility with external references
UltralyticsDetector = PCBDetector

