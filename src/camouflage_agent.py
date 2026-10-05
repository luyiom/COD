"""
Camouflage assessment agent.
Uses CV-based analysis for offline operation.
Optionally supports CLIP when HuggingFace is accessible.
"""
import os
import json
import warnings
from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from config import CLASS_NAMES, ROOT

warnings.filterwarnings("ignore")


def read_image_unicode(path):
    """Read image with Unicode path support on Windows."""
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


@dataclass
class CamouflageResult:
    has_camouflage: bool
    camouflage_level: str
    level_confidence: float
    details: str


class CVCamouflageAgent:
    """CV-based camouflage assessment. No network required."""

    def __init__(self):
        print("CamouflageAgent: using CV-based analysis (offline mode)")

    def _compute_edge_density(self, gray, block_size=64):
        """Compute edge density in blocks of the image."""
        edges = cv2.Canny(gray, 50, 150)
        h, w = gray.shape
        densities = []
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                block = edges[y:y+block_size, x:x+block_size]
                densities.append(np.mean(block > 0))
        return edges, np.mean(densities), np.std(densities)

    def _compute_color_entropy(self, img_hsv, block_size=64):
        """Compute color entropy across image blocks."""
        h, w = img_hsv.shape[:2]
        entropies = []
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                block = img_hsv[y:y+block_size, x:x+block_size, 0]
                hist = cv2.calcHist([block], [0], None, [32], [0, 180])
                hist = hist / (hist.sum() + 1e-7)
                hist = hist[hist > 0]
                entropy = -np.sum(hist * np.log2(hist))
                entropies.append(entropy)
        return np.mean(entropies), np.std(entropies)

    def _compute_texture_complexity(self, gray):
        """Compute texture complexity using local binary pattern approximation."""
        # Use Laplacian variance as texture indicator
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return lap.var()

    def assess(self, image_path: str) -> CamouflageResult:
        """Analyze image and return camouflage assessment using CV features."""
        img = read_image_unicode(image_path)
        if img is None:
            return CamouflageResult(
                has_camouflage=False,
                camouflage_level="no_camouflage",
                level_confidence=0.5,
                details="无法读取图片",
            )

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Feature 1: Edge density (low = more camouflaged)
        edges, edge_mean, edge_std = self._compute_edge_density(gray)

        # Feature 2: Color entropy (high = more varied colors, potential camo pattern)
        color_ent, color_ent_std = self._compute_color_entropy(hsv)

        # Feature 3: Texture complexity (high variance = more texture = camo pattern)
        texture_complexity = self._compute_texture_complexity(gray)

        # Feature 4: Overall image contrast
        contrast = gray.std()

        # Normalize features for scoring
        # Edge density: typical range 0.01-0.3, lower means less edges (camouflaged blends in)
        edge_score = max(0, min(1, 1.0 - edge_mean / 0.2))

        # Color entropy: typical range 1.0-4.0, higher means more complex color (camo pattern)
        color_score = max(0, min(1, (color_ent - 1.5) / 3.0))

        # Texture: higher variance means more texture disruption (camo netting, grass)
        texture_score = max(0, min(1, texture_complexity / 2000))

        # Combined camouflage score (0-1)
        camo_score = 0.35 * edge_score + 0.35 * color_score + 0.30 * texture_score

        # Determine level
        if camo_score < 0.25:
            level = "no_camouflage"
            has_camo = False
        elif camo_score < 0.45:
            level = "light_camouflage"
            has_camo = True
        elif camo_score < 0.65:
            level = "medium_camouflage"
            has_camo = True
        else:
            level = "heavy_camouflage"
            has_camo = True

        level_names = {
            "no_camouflage": "无伪装",
            "light_camouflage": "轻度伪装",
            "medium_camouflage": "中度伪装",
            "heavy_camouflage": "重度伪装",
        }

        detail_lines = [
            f"伪装评分: {camo_score:.2%}",
            f"边缘密度得分: {edge_score:.2%} (原始: {edge_mean:.4f})",
            f"颜色熵得分: {color_score:.2%} (原始: {color_ent:.2f})",
            f"纹理复杂度得分: {texture_score:.2%} (原始: {texture_complexity:.1f})",
            f"判定: {level_names[level]}",
        ]

        return CamouflageResult(
            has_camouflage=has_camo,
            camouflage_level=level,
            level_confidence=round(camo_score, 4),
            details=" | ".join(detail_lines),
        )


# Try to use CLIP, fall back to CV
_agent: Optional[object] = None


def get_agent(use_clip=False):
    global _agent
    if _agent is None:
        if use_clip:
            import torch
            from transformers import CLIPProcessor, CLIPModel

            class CLIPCamouflageAgent:
                def __init__(self):
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                    print(f"Loading CLIP model on {self.device}...")
                    self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
                    self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                    self.model.eval()
                    print("CLIP model loaded.")

                def assess(self, image_path):
                    from PIL import Image
                    image = Image.open(image_path).convert("RGB")
                    camo_prompts = {
                        "no_camouflage": "a clearly visible military vehicle with no camouflage",
                        "light_camouflage": "a military vehicle with camouflage paint matching the environment",
                        "medium_camouflage": "a military vehicle partially covered with branches, grass, or nets",
                        "heavy_camouflage": "a military vehicle heavily covered and almost invisible",
                    }
                    labels = list(camo_prompts.keys())
                    texts = [camo_prompts[k] for k in labels]
                    inputs = self.processor(text=texts, images=image, return_tensors="pt", padding=True).to(self.device)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
                    best_idx = int(np.argmax(probs))
                    best_level = labels[best_idx]
                    has_camo = best_level != "no_camouflage"
                    level_names = {"no_camouflage": "无伪装", "light_camouflage": "轻度伪装",
                                   "medium_camouflage": "中度伪装", "heavy_camouflage": "重度伪装"}
                    details = " | ".join([f"{level_names[l]}: {probs[i]:.2%}" for i, l in enumerate(labels)])
                    return CamouflageResult(
                        has_camouflage=has_camo, camouflage_level=best_level,
                        level_confidence=float(probs[best_idx]), details=details,
                    )

            _agent = CLIPCamouflageAgent()
        else:
            _agent = CVCamouflageAgent()
    return _agent


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        agent = get_agent()
        result = agent.assess(sys.argv[1])
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    else:
        print("Usage: py camouflage_agent.py <image_path>")
