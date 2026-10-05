"""
Qwen3-VL camouflage assessment agent.
Classifies vehicle images into camouflage levels:
  - no_camouflage (无伪装)
  - moderate_camouflage (中度伪装)
  - heavy_camouflage (重度伪装)

Supports: DashScope API / OpenAI-compatible endpoints.
"""
import os
import sys
import json
import base64
import io
from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env from project root
_ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(_ENV_FILE, override=True)

QWEN_API_URL = os.environ.get("QWEN_API_URL", "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen3-vl-plus")


def read_image_unicode(path):
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def encode_image_base64(img):
    """Encode numpy BGR image to base64 JPEG string."""
    success, buf = cv2.imencode(".jpg", img)
    if success:
        return base64.b64encode(buf.tobytes()).decode("utf-8")
    return ""


@dataclass
class CamouflageResult:
    level: str
    confidence: float
    detail: str
    source: str = "cv"


CAMO_PROMPT = """Analyze this military vehicle image and determine its camouflage level.
Choose ONE of the following:

- no_camouflage: The vehicle is clearly visible, no camouflage applied, stands out from the environment.
- moderate_camouflage: The vehicle has some camouflage (paint matching environment, partial covering with branches/nets), but is still identifiable.
- heavy_camouflage: The vehicle is heavily camouflaged, nearly invisible, extensively covered or blended into the terrain.

Reply with ONLY a JSON object: {"level": "<level>", "confidence": <0.0-1.0>, "detail": "<brief reason in Chinese>"}"""


class QwenCamouflageAgent:
    """Camouflage level assessment using Qwen3-VL."""

    def __init__(self, api_url: str = "", api_key: str = "", model: str = ""):
        self.api_url = api_url or QWEN_API_URL
        self.api_key = api_key or QWEN_API_KEY
        self.model = model or QWEN_MODEL

        if self.api_url:
            print(f"QwenAgent: API ready ({self.model} @ {self.api_url})")
        else:
            print("QwenAgent: no API configured, using placeholder CV classification.")

    def _classify_api(self, img_b64: str) -> Optional[CamouflageResult]:
        """Send image to DashScope API (native format)."""
        try:
            import requests
            payload = {
                "model": self.model,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"image": f"data:image/jpeg;base64,{img_b64}"},
                                {"text": CAMO_PROMPT},
                            ],
                        }
                    ]
                },
                "parameters": {
                    "max_tokens": 256,
                    "temperature": 0.1,
                },
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            resp = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            if resp.status_code != 200:
                print(f"QwenAgent API error: {resp.status_code} {resp.text[:300]}")
                return None

            data = resp.json()
            # DashScope native: output.choices[0].message.content is list of {text: ...}
            content = data.get("output", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
            if isinstance(content, list):
                content = "".join([c.get("text", "") for c in content])
            # OpenAI-compatible fallback
            if not content:
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            if content:
                return self._parse_response(content)
            print(f"QwenAgent: unexpected response format: {json.dumps(data, ensure_ascii=False)[:300]}")
            return None
        except Exception as e:
            print(f"QwenAgent API request failed: {e}")
            return None

    def _parse_response(self, text: str) -> Optional[CamouflageResult]:
        """Parse model response JSON, with robust fallback."""
        try:
            text = text.strip()
            # Extract JSON from markdown code blocks
            if "```" in text:
                parts = text.split("```")
                text = parts[1] if len(parts) > 1 else text
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            # Try to find JSON object in text
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                text = text[start:end + 1]
            data = json.loads(text)
            level = data.get("level", "no_camouflage")
            if level not in ("no_camouflage", "moderate_camouflage", "heavy_camouflage"):
                # Fuzzy match
                if "moderate" in str(level).lower():
                    level = "moderate_camouflage"
                elif "heavy" in str(level).lower():
                    level = "heavy_camouflage"
                else:
                    level = "no_camouflage"
            return CamouflageResult(
                level=level,
                confidence=float(data.get("confidence", 0.5)),
                detail=str(data.get("detail", "")),
                source="ai",
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            # If all parsing fails, use keyword matching from raw text
            text_lower = text.lower()
            if "heavy" in text_lower:
                return CamouflageResult(level="heavy_camouflage", confidence=0.7, detail=text[:200], source="ai")
            elif "moderate" in text_lower:
                return CamouflageResult(level="moderate_camouflage", confidence=0.7, detail=text[:200], source="ai")
            return None

    def _classify_placeholder(self, img) -> CamouflageResult:
        """CV-based analysis with natural language output."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]

        # Edge analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.mean(edges > 0))

        # Color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean_sat = float(np.mean(hsv[:, :, 1]))
        mean_val = float(np.mean(hsv[:, :, 2]))
        hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist = hist / (hist.sum() + 1e-7)
        hist = hist[hist > 0]
        color_entropy = float(-np.sum(hist * np.log2(hist)))

        # Texture analysis
        texture_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Compute camouflage score
        edge_score = max(0, min(1, 1.0 - edge_density / 0.2))
        color_score = max(0, min(1, (color_entropy - 1.5) / 3.0))
        texture_score = max(0, min(1, texture_var / 2000))
        camo_score = float(0.35 * edge_score + 0.35 * color_score + 0.3 * texture_score)

        # Generate natural language detail
        parts = []
        if edge_density < 0.08:
            parts.append("车辆边缘模糊，与背景边界不清晰")
        elif edge_density < 0.15:
            parts.append("车辆轮廓部分可见，边缘过渡自然")
        else:
            parts.append("车辆边缘清晰，轮廓分明")

        if color_entropy > 3.0:
            parts.append("车身颜色与周围环境高度融合")
        elif color_entropy > 2.0:
            parts.append("车身颜色与环境有一定近似度")
        else:
            parts.append("车身颜色与环境差异明显")

        if texture_var > 2000:
            parts.append("表面纹理复杂，存在伪装图案或遮挡物")
        elif texture_var > 800:
            parts.append("表面有一定纹理变化")
        else:
            parts.append("表面纹理简单，无明显伪装处理")

        detail = "；".join(parts) + "。"
        detail += f"（CV分析，边缘密度{edge_density:.3f}，颜色熵{color_entropy:.2f}，纹理方差{texture_var:.0f}）"

        if camo_score < 0.3:
            level = "no_camouflage"
        elif camo_score < 0.55:
            level = "moderate_camouflage"
        else:
            level = "heavy_camouflage"

        return CamouflageResult(
            level=level,
            confidence=round(camo_score, 4),
            detail=detail,
            source="cv",
        )

    def assess(self, image_path_or_array) -> CamouflageResult:
        """Assess camouflage level of a vehicle image. Always returns a result."""
        try:
            if isinstance(image_path_or_array, str):
                img = read_image_unicode(image_path_or_array)
            else:
                img = image_path_or_array

            if img is None or img.size == 0:
                return CamouflageResult(level="no_camouflage", confidence=0, detail="图片数据为空", source="cv")

            # Ensure minimum size for processing
            h, w = img.shape[:2]
            if h < 10 or w < 10:
                return CamouflageResult(level="no_camouflage", confidence=0, detail="检测区域过小", source="cv")

            # Try API if configured
            if self.api_url:
                img_b64 = encode_image_base64(img)
                result = self._classify_api(img_b64)
                if result:
                    return result

            # Fallback: CV-based placeholder
            return self._classify_placeholder(img)

        except Exception as e:
            print(f"QwenAgent assess error: {e}")
            return CamouflageResult(level="no_camouflage", confidence=0, detail=f"评估异常: {str(e)[:100]}", source="cv")

    def scene_analyze(self, image_path_or_array) -> dict:
        """
        Analyze entire image scene when YOLO finds nothing.
        Asks Qwen: is there any military vehicle here?
        Returns dict with {has_military, content_description, vehicles_found}
        """
        try:
            if isinstance(image_path_or_array, str):
                img = read_image_unicode(image_path_or_array)
            else:
                img = image_path_or_array

            if img is None or img.size == 0:
                return {"has_military": False, "content_description": "无法读取图片", "vehicles_found": []}

            if not self.api_url:
                return {"has_military": False, "content_description": "场景分析未配置API", "vehicles_found": []}

            img_b64 = encode_image_base64(img)
            prompt = """Analyze this image carefully.
Is there any military vehicle (tank, armored vehicle, military truck, military jeep, self-propelled artillery) visible in this image?

Reply with ONLY a JSON object:
{
  "has_military": true or false,
  "content_description": "<briefly describe what this image actually shows, in Chinese>",
  "vehicles_found": [
    {
      "type": "<vehicle type, e.g. tank/armored vehicle/military truck>",
      "description": "<rough location and appearance in Chinese, e.g. 图片左下角有一辆绿色涂装的坦克>",
      "likely_military": true or false
    }
  ]
}

If no military vehicle at all, set has_military=false and vehicles_found=[]."""

            import requests
            payload = {
                "model": self.model,
                "input": {
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"image": f"data:image/jpeg;base64,{img_b64}"},
                            {"text": prompt},
                        ],
                    }]
                },
                "parameters": {"max_tokens": 512, "temperature": 0.1},
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            resp = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            if resp.status_code != 200:
                return {"has_military": False, "content_description": f"API请求失败: {resp.status_code}", "vehicles_found": []}

            data = resp.json()
            content = data.get("output", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
            if isinstance(content, list):
                content = "".join([c.get("text", "") for c in content])

            # Parse JSON response
            text = content.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            import json
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                text = text[start:end + 1]

            result = json.loads(text)
            return {
                "has_military": result.get("has_military", False),
                "content_description": result.get("content_description", ""),
                "vehicles_found": result.get("vehicles_found", []),
            }

        except Exception as e:
            print(f"QwenAgent scene_analyze error: {e}")
            return {"has_military": False, "content_description": f"分析异常: {str(e)[:100]}", "vehicles_found": []}


_agent: Optional[QwenCamouflageAgent] = None


def get_qwen_agent() -> QwenCamouflageAgent:
    global _agent
    if _agent is None:
        _agent = QwenCamouflageAgent()
    return _agent
