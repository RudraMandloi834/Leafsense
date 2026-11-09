"""
Image Quality Checker for Plant Disease Detection
Validates image quality before prediction
"""

import cv2
import numpy as np
from PIL import Image
import io


class ImageQualityChecker:
    """Check image quality for optimal disease detection"""

    def __init__(self):
        self.min_resolution = (224, 224)  # Minimum required resolution
        self.optimal_resolution = (512, 512)  # Optimal resolution
        self.blur_threshold = 100  # Laplacian variance threshold
        self.brightness_range = (30, 225)  # Acceptable brightness range

    def check_quality(self, image_bytes):
        """
        Comprehensive image quality check

        Args:
            image_bytes: Image file bytes

        Returns:
            dict: Quality assessment results
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))

            # Convert to numpy array for OpenCV
            img_array = np.array(image)

            # Convert RGB to BGR for OpenCV
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_cv = img_array

            results = {
                "is_valid": True,
                "warnings": [],
                "suggestions": [],
                "metrics": {},
            }

            # Check 1: Resolution
            resolution_check = self._check_resolution(image)
            results["metrics"]["resolution"] = resolution_check
            if not resolution_check["is_acceptable"]:
                results["warnings"].append(resolution_check["message"])
                results["suggestions"].append(resolution_check["suggestion"])
                if resolution_check["is_critical"]:
                    results["is_valid"] = False

            # Check 2: Blur detection
            blur_check = self._check_blur(img_cv)
            results["metrics"]["blur"] = blur_check
            if not blur_check["is_acceptable"]:
                results["warnings"].append(blur_check["message"])
                results["suggestions"].append(blur_check["suggestion"])

            # Check 3: Brightness
            brightness_check = self._check_brightness(img_cv)
            results["metrics"]["brightness"] = brightness_check
            if not brightness_check["is_acceptable"]:
                results["warnings"].append(brightness_check["message"])
                results["suggestions"].append(brightness_check["suggestion"])

            # Check 4: Color distribution (detect if image is too monochrome)
            color_check = self._check_color_distribution(img_cv)
            results["metrics"]["color"] = color_check
            if not color_check["is_acceptable"]:
                results["warnings"].append(color_check["message"])
                results["suggestions"].append(color_check["suggestion"])

            # Overall quality score (0-100)
            results["quality_score"] = self._calculate_quality_score(results["metrics"])

            return results

        except Exception as e:
            return {
                "is_valid": False,
                "warnings": [f"Error analyzing image: {str(e)}"],
                "suggestions": ["Please upload a valid image file"],
                "metrics": {},
                "quality_score": 0,
            }

    def _check_resolution(self, image):
        """Check if image resolution is acceptable"""
        width, height = image.size

        is_critical = width < self.min_resolution[0] or height < self.min_resolution[1]
        is_optimal = (
            width >= self.optimal_resolution[0] and height >= self.optimal_resolution[1]
        )

        if is_critical:
            return {
                "width": width,
                "height": height,
                "is_acceptable": False,
                "is_critical": True,
                "message": f"Image resolution too low ({width}x{height})",
                "suggestion": f"Please use an image with at least {self.min_resolution[0]}x{self.min_resolution[1]} pixels",
            }
        elif not is_optimal:
            return {
                "width": width,
                "height": height,
                "is_acceptable": True,
                "is_critical": False,
                "message": f"Image resolution is acceptable but not optimal ({width}x{height})",
                "suggestion": f"For best results, use images with {self.optimal_resolution[0]}x{self.optimal_resolution[1]} pixels or higher",
            }
        else:
            return {
                "width": width,
                "height": height,
                "is_acceptable": True,
                "is_critical": False,
                "message": "Resolution is optimal",
                "suggestion": None,
            }

    def _check_blur(self, img_cv):
        """Detect if image is blurry using Laplacian variance"""
        # Convert to grayscale
        if len(img_cv.shape) == 3:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_cv

        # Calculate Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        is_acceptable = laplacian_var >= self.blur_threshold

        if not is_acceptable:
            return {
                "variance": float(laplacian_var),
                "is_acceptable": False,
                "message": "Image appears to be blurry",
                "suggestion": "Hold the camera steady and ensure the leaf is in focus",
            }
        else:
            return {
                "variance": float(laplacian_var),
                "is_acceptable": True,
                "message": "Image sharpness is good",
                "suggestion": None,
            }

    def _check_brightness(self, img_cv):
        """Check if image brightness is in acceptable range"""
        # Convert to grayscale
        if len(img_cv.shape) == 3:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_cv

        # Calculate average brightness
        avg_brightness = np.mean(gray)

        min_bright, max_bright = self.brightness_range

        if avg_brightness < min_bright:
            return {
                "average": float(avg_brightness),
                "is_acceptable": False,
                "message": "Image is too dark",
                "suggestion": "Take the photo in better lighting or increase brightness",
            }
        elif avg_brightness > max_bright:
            return {
                "average": float(avg_brightness),
                "is_acceptable": False,
                "message": "Image is too bright/overexposed",
                "suggestion": "Reduce lighting or avoid direct sunlight",
            }
        else:
            return {
                "average": float(avg_brightness),
                "is_acceptable": True,
                "message": "Brightness is optimal",
                "suggestion": None,
            }

    def _check_color_distribution(self, img_cv):
        """Check if image has good color distribution (not too monochrome)"""
        if len(img_cv.shape) != 3:
            return {
                "is_acceptable": False,
                "message": "Image is grayscale",
                "suggestion": "Please use a color image for better detection",
            }

        # Calculate color standard deviation
        std_dev = np.std(img_cv, axis=(0, 1))
        avg_std = np.mean(std_dev)

        # If standard deviation is too low, image is too monochrome
        if avg_std < 15:
            return {
                "std_deviation": float(avg_std),
                "is_acceptable": False,
                "message": "Image lacks color variation",
                "suggestion": "Ensure the leaf is clearly visible with natural colors",
            }
        else:
            return {
                "std_deviation": float(avg_std),
                "is_acceptable": True,
                "message": "Color distribution is good",
                "suggestion": None,
            }

    def _calculate_quality_score(self, metrics):
        """Calculate overall quality score (0-100)"""
        score = 100

        # Resolution penalty
        if "resolution" in metrics:
            res = metrics["resolution"]
            if res.get("is_critical"):
                score -= 50
            elif not res.get("is_acceptable"):
                score -= 10

        # Blur penalty
        if "blur" in metrics:
            if not metrics["blur"].get("is_acceptable"):
                score -= 25

        # Brightness penalty
        if "brightness" in metrics:
            if not metrics["brightness"].get("is_acceptable"):
                score -= 15

        # Color penalty
        if "color" in metrics:
            if not metrics["color"].get("is_acceptable"):
                score -= 10

        return max(0, score)


# Singleton instance
quality_checker = ImageQualityChecker()
