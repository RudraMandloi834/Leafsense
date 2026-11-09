"""
LeafSense Utility Module
Provides disease labels, color palettes, validation helpers, and formatting utilities.
"""

import io
import base64
from typing import Tuple, List, Dict, Optional
from PIL import Image

# ----------------- 17 SUPPORTED DISEASE CLASSES -----------------
LABEL_COLS: List[str] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Potato___Early_blight",
    "Potato___Late_blight",
]

NUM_CLASSES = len(LABEL_COLS)

# ----------------- SUPPORTED LANGUAGES -----------------
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ml": "Malayalam",
    "kn": "Kannada",
    "or": "Odia",
}

# ----------------- MULTI-DISEASE COLOR PALETTE -----------------
# High-contrast, vibrant RGB colors for Grad-CAM overlays
DISEASE_COLORS = [
    ("red", (255, 50, 50)),
    ("yellow", (255, 215, 0)),
    ("blue", (30, 144, 255)),
    ("green", (46, 204, 113)),
    ("purple", (155, 89, 182)),
    ("orange", (255, 140, 0)),
    ("cyan", (0, 206, 209)),
    ("pink", (255, 105, 180)),
]

# ----------------- SECURITY & VALIDATION CONSTANTS -----------------
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/jpg",
    "application/octet-stream",  # Fallback for some mobile camera blobs
}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def format_disease_name(raw_label: str) -> str:
    """Format raw label (e.g. 'Apple___Black_rot') into user-friendly title."""
    return raw_label.replace("___", " - ").replace("_", " ").strip()


def validate_image_payload(
    file_bytes: bytes,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
) -> Tuple[bool, Optional[str], Optional[Image.Image]]:
    """
    Validate image file size, format, and loadability.

    Returns:
        (is_valid, error_message, pil_image)
    """
    # Check payload size
    if not file_bytes:
        return False, "Uploaded image file is empty", None

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        return (
            False,
            f"Image exceeds maximum permitted size of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB",
            None,
        )

    # Validate image decoding
    try:
        image = Image.open(io.BytesIO(file_bytes))
        # Verify image integrity without loading all pixels
        image.verify()

        # Re-open for actual processing (verify closes stream state)
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return True, None, image
    except Exception as e:
        return False, f"Invalid or corrupted image format: {str(e)}", None


def pil_to_base64_data_uri(pil_image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL image to data URI string."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"
