"""
LeafSense Image Quality Checker
Validates image resolution, sharpness (blur), illumination, and color variation.
"""

from typing import Dict, Any, Tuple
import cv2
import numpy as np
from PIL import Image
import io


class ImageQualityChecker:
    """Pre-flight quality validation for leaf disease images."""

    def __init__(self):
        self.min_resolution: Tuple[int, int] = (224, 224)
        self.optimal_resolution: Tuple[int, int] = (512, 512)
        self.blur_threshold: float = 80.0  # Laplacian variance threshold
        self.brightness_range: Tuple[float, float] = (30.0, 225.0)

    def check_quality(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Evaluate image suitability for deep learning inference.

        Returns structured metrics, warnings, suggestions, and 0-100 score.
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))

            # Handle RGBA/transparency safely
            if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
                image = image.convert("RGB")

            img_array = np.array(image)

            # Convert to OpenCV BGR format
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_cv = img_array

            results: Dict[str, Any] = {
                "is_valid": True,
                "warnings": [],
                "suggestions": [],
                "metrics": {},
            }

            # 1. Resolution Check
            res_check = self._check_resolution(image)
            results["metrics"]["resolution"] = res_check
            if not res_check["is_acceptable"]:
                results["warnings"].append(res_check["message"])
                if res_check["suggestion"]:
                    results["suggestions"].append(res_check["suggestion"])
                if res_check.get("is_critical"):
                    results["is_valid"] = False

            # 2. Sharpness / Blur Detection
            blur_check = self._check_blur(img_cv)
            results["metrics"]["blur"] = blur_check
            if not blur_check["is_acceptable"]:
                results["warnings"].append(blur_check["message"])
                if blur_check["suggestion"]:
                    results["suggestions"].append(blur_check["suggestion"])

            # 3. Illumination / Brightness Check
            bright_check = self._check_brightness(img_cv)
            results["metrics"]["brightness"] = bright_check
            if not bright_check["is_acceptable"]:
                results["warnings"].append(bright_check["message"])
                if bright_check["suggestion"]:
                    results["suggestions"].append(bright_check["suggestion"])

            # 4. Color Distribution Check
            color_check = self._check_color_distribution(img_cv)
            results["metrics"]["color"] = color_check
            if not color_check["is_acceptable"]:
                results["warnings"].append(color_check["message"])
                if color_check["suggestion"]:
                    results["suggestions"].append(color_check["suggestion"])

            # Calculate composite quality score (0 - 100)
            results["quality_score"] = self._calculate_quality_score(results["metrics"])
            return results

        except Exception as e:
            return {
                "is_valid": False,
                "warnings": [f"Image inspection error: {str(e)}"],
                "suggestions": ["Please upload a standard RGB leaf image (JPG, PNG, WebP)"],
                "metrics": {},
                "quality_score": 0,
            }

    def _check_resolution(self, image: Image.Image) -> Dict[str, Any]:
        width, height = image.size
        is_critical = width < self.min_resolution[0] or height < self.min_resolution[1]
        is_optimal = width >= self.optimal_resolution[0] and height >= self.optimal_resolution[1]

        if is_critical:
            return {
                "width": width,
                "height": height,
                "is_acceptable": False,
                "is_critical": True,
                "message": f"Image resolution too low ({width}x{height}px)",
                "suggestion": f"Use an image with at least {self.min_resolution[0]}x{self.min_resolution[1]} pixels",
            }
        elif not is_optimal:
            return {
                "width": width,
                "height": height,
                "is_acceptable": True,
                "is_critical": False,
                "message": f"Resolution is acceptable ({width}x{height}px)",
                "suggestion": f"For optimal accuracy, use images with {self.optimal_resolution[0]}x{self.optimal_resolution[1]}px or higher",
            }
        return {
            "width": width,
            "height": height,
            "is_acceptable": True,
            "is_critical": False,
            "message": "Resolution is optimal",
            "suggestion": None,
        }

    def _check_blur(self, img_cv: np.ndarray) -> Dict[str, Any]:
        if len(img_cv.shape) == 3:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_cv

        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_acceptable = laplacian_var >= self.blur_threshold

        if not is_acceptable:
            return {
                "variance": round(laplacian_var, 2),
                "is_acceptable": False,
                "message": "Image appears blurry or out of focus",
                "suggestion": "Hold camera steady and ensure the leaf surface is sharp and in focus",
            }
        return {
            "variance": round(laplacian_var, 2),
            "is_acceptable": True,
            "message": "Image sharpness is good",
            "suggestion": None,
        }

    def _check_brightness(self, img_cv: np.ndarray) -> Dict[str, Any]:
        if len(img_cv.shape) == 3:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_cv

        avg_brightness = float(np.mean(gray))
        min_bright, max_bright = self.brightness_range

        if avg_brightness < min_bright:
            return {
                "average": round(avg_brightness, 2),
                "is_acceptable": False,
                "message": "Image is underexposed/too dark",
                "suggestion": "Take the photo in natural daylight or increase illumination",
            }
        elif avg_brightness > max_bright:
            return {
                "average": round(avg_brightness, 2),
                "is_acceptable": False,
                "message": "Image is overexposed/too bright",
                "suggestion": "Avoid direct camera flash or intense sun glare on leaf surface",
            }
        return {
            "average": round(avg_brightness, 2),
            "is_acceptable": True,
            "message": "Illumination is well balanced",
            "suggestion": None,
        }

    def _check_color_distribution(self, img_cv: np.ndarray) -> Dict[str, Any]:
        if len(img_cv.shape) != 3:
            return {
                "is_acceptable": False,
                "message": "Image is grayscale",
                "suggestion": "Please provide a full-color leaf photograph",
            }

        std_dev = np.std(img_cv, axis=(0, 1))
        avg_std = float(np.mean(std_dev))

        if avg_std < 15.0:
            return {
                "std_deviation": round(avg_std, 2),
                "is_acceptable": False,
                "message": "Image lacks chromatic variation",
                "suggestion": "Ensure natural plant colors (greens, yellows, browns) are captured",
            }
        return {
            "std_deviation": round(avg_std, 2),
            "is_acceptable": True,
            "message": "Color balance is good",
            "suggestion": None,
        }

    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> int:
        score = 100

        if "resolution" in metrics:
            res = metrics["resolution"]
            if res.get("is_critical"):
                score -= 40
            elif not res.get("is_acceptable"):
                score -= 10

        if "blur" in metrics and not metrics["blur"].get("is_acceptable"):
            score -= 25

        if "brightness" in metrics and not metrics["brightness"].get("is_acceptable"):
            score -= 15

        if "color" in metrics and not metrics["color"].get("is_acceptable"):
            score -= 10

        return max(0, min(100, score))


quality_checker = ImageQualityChecker()
