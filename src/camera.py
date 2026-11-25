"""
Camera and Image Input Helper for BharatPCB-AI.
Safely decodes image uploads, manages MacBook webcam input,
validates image integrity, and handles BGR/RGB color conversions.
"""

from typing import Optional, Tuple, Union
import io
import cv2
import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
MIN_DIMENSION = 100
MAX_DIMENSION = 10000


def decode_image_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Safely decode raw image bytes (from st.file_uploader or st.camera_input)
    into an OpenCV BGR numpy array.
    """
    if not image_bytes or len(image_bytes) < 8:
        return None

    # Check common image signatures to avoid triggering unwanted external format handlers
    is_png = image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    is_jpg = image_bytes.startswith(b"\xff\xd8\xff")
    is_bmp = image_bytes.startswith(b"BM")
    is_tiff = image_bytes.startswith(b"II*\x00") or image_bytes.startswith(b"MM\x00*")

    if not (is_png or is_jpg or is_bmp or is_tiff):
        return None

    try:
        # First attempt fast OpenCV imdecode
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None and img.size > 0:
            return img
    except Exception:
        pass

    # Only attempt Pillow if it's a TIFF format that OpenCV couldn't decode
    if is_tiff:
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img = pil_img.convert("RGB")
            rgb_arr = np.array(pil_img)
            return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    return None


def validate_image(image: Optional[np.ndarray]) -> Tuple[bool, str]:
    """
    Validate that an image array is physically valid for PCB inspection:
    - Non-None
    - 3-channel BGR
    - Height and Width between MIN_DIMENSION and MAX_DIMENSION
    """
    if image is None:
        return False, "Image could not be decoded. File may be empty or corrupted."

    if not isinstance(image, np.ndarray):
        return False, "Invalid image data format. Expected numpy ndarray."

    if len(image.shape) != 3 or image.shape[2] != 3:
        return False, f"Image must be a 3-channel color image (got shape {image.shape})."

    h, w = image.shape[:2]
    if h < MIN_DIMENSION or w < MIN_DIMENSION:
        return False, f"Image dimensions ({w}x{h}) are too small. Minimum required: {MIN_DIMENSION}x{MIN_DIMENSION} px."

    if h > MAX_DIMENSION or w > MAX_DIMENSION:
        return False, f"Image dimensions ({w}x{h}) exceed maximum allowable size ({MAX_DIMENSION}x{MAX_DIMENSION} px)."

    return True, "Valid"


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR image to RGB for Streamlit/Matplotlib display."""
    if image is None:
        return image
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def capture_frame_cv2(camera_index: int = 0) -> Tuple[bool, Optional[np.ndarray], str]:
    """
    Direct OpenCV camera capture fallback for headless or CLI environments.
    """
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return False, None, f"Cannot open camera device index {camera_index}."

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return False, None, "Failed to capture frame from camera."

        valid, msg = validate_image(frame)
        if not valid:
            return False, None, f"Captured frame invalid: {msg}"

        return True, frame, "Frame captured successfully."
    except Exception as e:
        return False, None, f"Camera capture error: {str(e)}"
