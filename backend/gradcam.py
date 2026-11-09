"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for Explainable AI
Multi-Disease Localization with Color-Coded Overlays - SIMPLIFIED VERSION
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import io
import base64


class GradCAM:
    """Generate Grad-CAM heatmaps for CNN models"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        batch_size, channels, height, width = input_tensor.size()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()

        if self.gradients is None or self.activations is None:
            return None

        gradients = self.gradients
        activations = self.activations

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=False)
        cam = F.relu(cam)
        cam = cam.cpu().numpy()[0]

        cam = cam - np.min(cam)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)

        return cam


def get_target_layer(model, model_name):
    """Get the target layer for Grad-CAM"""
    model_name = model_name.lower()

    if "densenet" in model_name:
        return model.features.norm5
    elif "resnet" in model_name:
        return model.layer4[-1]
    elif "ghost" in model_name:
        return model.blocks[-1]
    elif "efficient" in model_name or "agri" in model_name:
        return model.backbone.features[-1]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def generate_colored_heatmap_overlay(
    original_image, cam, color_rgb, alpha=0.6, threshold=0.4
):
    """Create BRIGHT colored heatmap overlay"""

    # Convert to numpy
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)

    # Ensure RGB
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

    h, w = original_image.shape[:2]

    # Resize CAM
    cam_resized = cv2.resize(cam, (w, h))

    # Threshold
    cam_thresholded = np.where(cam_resized > threshold, cam_resized, 0)
    if cam_thresholded.max() > 0:
        cam_thresholded = (cam_thresholded - cam_thresholded.min()) / (
            cam_thresholded.max() - cam_thresholded.min()
        )

    # Create BRIGHT colored heatmap (BGR format for OpenCV)
    cam_uint8 = np.uint8(255 * cam_thresholded)
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)

    # Apply color with FULL intensity (RGB format since we use PIL/numpy)
    heatmap[:, :, 0] = np.uint8(color_rgb[0] * (cam_uint8 / 255.0))  # R
    heatmap[:, :, 1] = np.uint8(color_rgb[1] * (cam_uint8 / 255.0))  # G
    heatmap[:, :, 2] = np.uint8(color_rgb[2] * (cam_uint8 / 255.0))  # B

    # Simple addWeighted blending - Balance between leaf visibility and color highlighting
    # 50% original image, 50% colored heatmap = Good balance
    overlay = cv2.addWeighted(original_image, 0.65, heatmap, 0.35, 0)

    return Image.fromarray(overlay)


def generate_multi_disease_visualization(
    model, model_name, image_tensor, original_image, disease_classes, color_map
):
    """Generate visualizations for multiple diseases"""

    try:
        target_layer = get_target_layer(model, model_name)
        gradcam = GradCAM(model, target_layer)

        individual_visualizations = {}
        combined_overlay = np.array(original_image).astype(np.float32)

        print(f"Generating Grad-CAM for {len(disease_classes)} diseases...")

        for disease_idx in disease_classes:
            print(f"  Processing disease index {disease_idx}...")

            # Generate CAM
            cam = gradcam.generate_cam(image_tensor, disease_idx)

            if cam is None:
                print(f"  Failed to generate CAM for disease {disease_idx}")
                continue

            # Get color for this disease
            color_rgb = color_map.get(disease_idx, (255, 255, 255))

            # Generate colored overlay
            colored_overlay = generate_colored_heatmap_overlay(
                original_image, cam, color_rgb, alpha=0.6, threshold=0.4
            )

            individual_visualizations[disease_idx] = colored_overlay

            # Add to combined (accumulate colors)
            overlay_array = np.array(colored_overlay).astype(np.float32)
            combined_overlay = combined_overlay * 0.5 + overlay_array * 0.5

        # Convert combined to PIL
        combined_overlay = np.clip(combined_overlay, 0, 255).astype(np.uint8)
        combined_pil = Image.fromarray(combined_overlay)

        print(f"Successfully generated {len(individual_visualizations)} visualizations")

        return {"individual": individual_visualizations, "combined": combined_pil}

    except Exception as e:
        print(f"Error in generate_multi_disease_visualization: {e}")
        import traceback

        traceback.print_exc()
        return None


def pil_to_base64(pil_image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"
