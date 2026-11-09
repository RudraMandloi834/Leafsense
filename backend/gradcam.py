"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for Explainable AI
Provides single-disease and multi-disease localization with color-coded overlays.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
from PIL import Image
import io
import base64

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    TORCH_AVAILABLE = False


class GradCAM:
    """Generate Grad-CAM heatmaps for CNN models with automatic hook cleanup."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.forward_handle = None
        self.backward_handle = None
        self._register_hooks()

    def _register_hooks(self):
        self.forward_handle = self.target_layer.register_forward_hook(self.save_activation)
        # Use register_full_backward_hook to avoid PyTorch deprecation warnings
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self.backward_handle = self.target_layer.register_full_backward_hook(
                self.save_gradient
            )
        else:
            self.backward_handle = self.target_layer.register_backward_hook(
                self.save_gradient
            )

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        if grad_output and len(grad_output) > 0 and grad_output[0] is not None:
            self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        """Clean up PyTorch forward/backward hooks to prevent memory leaks."""
        if self.forward_handle is not None:
            self.forward_handle.remove()
            self.forward_handle = None
        if self.backward_handle is not None:
            self.backward_handle.remove()
            self.backward_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def generate_cam(self, input_tensor, target_class=None):
        """Generate normalized 2D CAM activation map for target class."""
        if not TORCH_AVAILABLE:
            return None

        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        # Ensure gradient tracking for backpropagation
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        class_loss = output[0, target_class]
        class_loss.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return None

        gradients = self.gradients
        activations = self.activations

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=False)
        cam = F.relu(cam)
        cam = cam.cpu().numpy()[0]

        cam_min = np.min(cam)
        cam_max = np.max(cam)
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def get_target_layer(model, model_name: str):
    """Retrieve the target convolutional layer for Grad-CAM."""
    model_name = model_name.lower()

    if "densenet" in model_name:
        return model.features.norm5
    elif "resnet" in model_name:
        return model.layer4[-1]
    elif "ghost" in model_name:
        return model.blocks[-1]
    elif "agri" in model_name or "efficient" in model_name:
        return model.backbone.features[-1]
    else:
        raise ValueError(f"Unsupported model architecture for Grad-CAM: {model_name}")


def generate_colored_heatmap_overlay(
    original_image: Image.Image,
    cam: np.ndarray,
    color_rgb: Tuple[int, int, int],
    alpha: float = 0.45,
    threshold: float = 0.35,
) -> Image.Image:
    """Overlay a colored Grad-CAM attention heatmap onto the original leaf image."""
    # Convert image to numpy RGB
    orig_np = np.array(original_image)
    if len(orig_np.shape) == 2:
        orig_np = cv2.cvtColor(orig_np, cv2.COLOR_GRAY2RGB)
    elif orig_np.shape[2] == 4:
        orig_np = orig_np[:, :, :3]

    h, w = orig_np.shape[:2]

    # Resize CAM to image dimensions
    cam_resized = cv2.resize(cam, (w, h))

    # Apply threshold to focus on salient diseased lesions
    cam_thresholded = np.where(cam_resized > threshold, cam_resized, 0.0)
    c_max = cam_thresholded.max()
    c_min = cam_thresholded.min()
    if c_max - c_min > 1e-8:
        cam_norm = (cam_thresholded - c_min) / (c_max - c_min)
    else:
        cam_norm = np.zeros_like(cam_thresholded)

    # Colorize heatmap
    heatmap = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        heatmap[:, :, c] = color_rgb[c] * cam_norm

    # Blend original image and heatmap
    orig_float = orig_np.astype(np.float32)
    mask = (cam_norm[:, :, None] > 0).astype(np.float32)

    # Blend selectively over salient regions
    blended = orig_float * (1.0 - alpha * mask) + heatmap * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


def generate_multi_disease_visualization(
    model,
    model_name: str,
    image_tensor,
    original_image: Image.Image,
    disease_classes: List[int],
    color_map: Dict[int, Tuple[int, int, int]],
) -> Optional[Dict[str, Any]]:
    """
    Generate distinct individual disease heatmaps and a single unified combined overlay.
    Properly releases all PyTorch gradient hooks after execution.
    """
    if not TORCH_AVAILABLE:
        return None

    try:
        target_layer = get_target_layer(model, model_name)
    except Exception as e:
        print(f"Target layer lookup failed: {e}")
        return None

    orig_np = np.array(original_image)
    if len(orig_np.shape) == 2:
        orig_np = cv2.cvtColor(orig_np, cv2.COLOR_GRAY2RGB)
    elif orig_np.shape[2] == 4:
        orig_np = orig_np[:, :, :3]

    h, w = orig_np.shape[:2]

    individual_visualizations = {}
    accumulated_heatmap = np.zeros((h, w, 3), dtype=np.float32)
    accumulated_weights = np.zeros((h, w, 1), dtype=np.float32)

    # Safely manage hooks with context manager
    with GradCAM(model, target_layer) as gradcam:
        for disease_idx in disease_classes:
            cam = gradcam.generate_cam(image_tensor, target_class=disease_idx)
            if cam is None:
                continue

            color_rgb = color_map.get(disease_idx, (255, 50, 50))

            # Individual overlay
            indiv_img = generate_colored_heatmap_overlay(
                original_image, cam, color_rgb, alpha=0.50, threshold=0.35
            )
            individual_visualizations[disease_idx] = indiv_img

            # Accumulate into combined heatmap
            cam_resized = cv2.resize(cam, (w, h))
            cam_thresh = np.where(cam_resized > 0.35, cam_resized, 0.0)
            if cam_thresh.max() > 0:
                cam_norm = cam_thresh / cam_thresh.max()
                for c in range(3):
                    accumulated_heatmap[:, :, c] += color_rgb[c] * cam_norm
                accumulated_weights += cam_norm[:, :, None]

    # Normalize overlapping regions in combined overlay
    mask = (accumulated_weights > 0).astype(np.float32)
    weight_safe = np.where(accumulated_weights > 1.0, accumulated_weights, 1.0)
    normalized_heatmap = np.zeros_like(accumulated_heatmap)
    np.divide(accumulated_heatmap, weight_safe, out=normalized_heatmap, where=accumulated_weights > 0)

    # Single-pass balanced composite
    alpha = 0.55
    orig_float = orig_np.astype(np.float32)
    combined_np = orig_float * (1.0 - alpha * mask) + normalized_heatmap * alpha * mask
    combined_np = np.clip(combined_np, 0, 255).astype(np.uint8)
    combined_pil = Image.fromarray(combined_np)

    return {
        "individual": individual_visualizations,
        "combined": combined_pil,
    }


def pil_to_base64(pil_image: Image.Image) -> str:
    """Convert PIL image to Base64 data URI string."""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"
