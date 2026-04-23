"""
models/classifier.py
ResNet-50 based chest X-ray disease classifier.
Supports: Normal, Pneumonia, COVID-19, Tuberculosis, Pleural Effusion
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os


# ── Disease classes ───────────────────────────────────────────────────────────
CLASSES = [
    "Normal",
    "Pneumonia",
    "COVID-19",
    "Tuberculosis",
    "Pleural Effusion",
]

CLASS_DESCRIPTIONS = {
    "Normal": "No significant pathology detected in the chest X-ray.",
    "Pneumonia": "Bacterial or viral infection causing lung inflammation. Look for consolidation or infiltrates.",
    "COVID-19": "Viral pneumonia with characteristic bilateral ground-glass opacities.",
    "Tuberculosis": "Mycobacterial infection often showing upper-lobe infiltrates or cavitation.",
    "Pleural Effusion": "Fluid accumulation in the pleural space, causing opacity at lung bases.",
}

SEVERITY_MAP = {
    "Normal": "none",
    "Pneumonia": "moderate",
    "COVID-19": "high",
    "Tuberculosis": "high",
    "Pleural Effusion": "moderate",
}


# ── Model definition ──────────────────────────────────────────────────────────
class ChestXRayClassifier(nn.Module):
    """
    Fine-tuned ResNet-50 for multi-class chest X-ray classification.
    Uses ImageNet pretrained weights with a custom classification head.
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()

        # Load ResNet-50 backbone
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )

        # Freeze early layers (optional: unfreeze for full fine-tuning)
        for name, param in self.backbone.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False

        # Replace the classification head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

        # Grad-CAM hook storage
        self._gradients = None
        self._activations = None
        self._register_hooks()

    def _register_hooks(self):
        """Register forward/backward hooks for Grad-CAM visualization."""
        def save_activation(module, input, output):
            self._activations = output.detach()

        def save_gradient(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        target_layer = self.backbone.layer4[-1]
        target_layer.register_forward_hook(save_activation)
        target_layer.register_full_backward_hook(save_gradient)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_gradcam(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Generate Grad-CAM heatmap for the predicted class."""
        self.eval()
        input_tensor.requires_grad_(True)

        output = self(input_tensor)
        self.zero_grad()
        output[0, class_idx].backward()

        gradients = self._gradients[0]          # (C, H, W)
        activations = self._activations[0]      # (C, H, W)

        weights = gradients.mean(dim=(1, 2))    # Global average pooling
        cam = (weights[:, None, None] * activations).sum(dim=0)
        cam = torch.relu(cam).cpu().numpy()

        # Normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ── Image preprocessing ───────────────────────────────────────────────────────
def get_transforms(mode: str = "inference") -> transforms.Compose:
    """
    Returns image transforms.
    Training includes augmentation; inference uses only normalization.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if mode == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])


# ── OpenCV preprocessing pipeline ────────────────────────────────────────────
def preprocess_xray_opencv(image_path: str) -> tuple[np.ndarray, dict]:
    """
    OpenCV pipeline for X-ray image enhancement.
    Returns (enhanced_image_rgb, metrics_dict)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    original = img.copy()
    metrics = {}

    # ── CLAHE contrast enhancement ──
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    # ── Gaussian denoising ──
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # ── Lung field segmentation (Otsu threshold) ──
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ── Edge detection (Canny) ──
    edges = cv2.Canny(denoised, 50, 150)

    # ── Compute image metrics ──
    metrics["mean_intensity"]  = float(np.mean(img))
    metrics["std_intensity"]   = float(np.std(img))
    metrics["contrast_ratio"]  = float(np.std(enhanced) / (np.mean(enhanced) + 1e-8))
    metrics["edge_density"]    = float(np.sum(edges > 0) / edges.size)
    metrics["lung_area_ratio"] = float(np.sum(morphed > 0) / morphed.size)

    # Convert grayscale → RGB for model input
    enhanced_rgb = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
    return enhanced_rgb, metrics


# ── Inference engine ──────────────────────────────────────────────────────────
class MedicalImageAnalyzer:
    """
    End-to-end inference pipeline:
      1. OpenCV preprocessing
      2. ResNet-50 classification
      3. Grad-CAM visualization
    """

    def __init__(self, model_path: str | None = None, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = ChestXRayClassifier(num_classes=len(CLASSES), pretrained=True)

        if model_path and os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"[Analyzer] Loaded weights from {model_path}")
        else:
            print("[Analyzer] Using pretrained ImageNet weights (no fine-tuned weights found).")

        self.model.to(self.device)
        self.model.eval()
        self.transform = get_transforms("inference")

    def analyze(self, image_path: str) -> dict:
        """
        Full analysis pipeline.
        Returns a structured result dictionary.
        """
        # Step 1: OpenCV preprocessing
        enhanced_rgb, cv_metrics = preprocess_xray_opencv(image_path)

        pil_image = Image.fromarray(enhanced_rgb)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Step 2: Model inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        pred_idx   = int(np.argmax(probs))
        pred_class = CLASSES[pred_idx]
        confidence = float(probs[pred_idx])

        # Step 3: Grad-CAM
        cam = self.model.get_gradcam(input_tensor, pred_idx)
        cam_overlay = self._overlay_gradcam(image_path, cam)

        # Step 4: Build class probability dict
        class_probs = {cls: float(p) for cls, p in zip(CLASSES, probs)}

        return {
            "predicted_class": pred_class,
            "confidence":       confidence,
            "severity":         SEVERITY_MAP[pred_class],
            "class_probs":      class_probs,
            "cv_metrics":       cv_metrics,
            "gradcam_overlay":  cam_overlay,
            "class_description": CLASS_DESCRIPTIONS[pred_class],
        }

    def _overlay_gradcam(self, image_path: str, cam: np.ndarray) -> np.ndarray:
        """Overlay Grad-CAM heatmap on original image."""
        original = cv2.imread(image_path)
        if original is None:
            return np.zeros((224, 224, 3), dtype=np.uint8)

        original = cv2.resize(original, (224, 224))
        heatmap  = cv2.resize(cam, (224, 224))
        heatmap  = np.uint8(255 * heatmap)
        heatmap  = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay  = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
