"""
FastAPI backend for camouflage military vehicle detection.
YOLO detection + Qwen3-VL camouflage level assessment.
Two-stage: low-conf sweep + Qwen visual confirmation for anti-missed-detection.
"""
import io
import os
import sys
import uuid
import json
import threading
import base64
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_DIR = os.path.dirname(_APP_DIR)
sys.path.insert(0, os.path.join(_PROJ_DIR, "src"))

from config import CLASS_NAMES, MODELS_DIR, ROOT

app = FastAPI(title="Camouflage Military Vehicle Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(_APP_DIR, "static")
os.makedirs(static_dir, exist_ok=True)
os.makedirs(os.path.join(static_dir, "uploads"), exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

templates_dir = os.path.join(_APP_DIR, "templates")

HISTORY_FILE = os.path.join(_APP_DIR, "history.json")
_history_lock = threading.Lock()

_yolo_model = None
_qwen_agent = None

LEVEL_COLORS = {
    "no_camouflage": (0, 255, 0),
    "moderate_camouflage": (0, 165, 255),
    "heavy_camouflage": (0, 0, 255),
}
LEVEL_LABELS = {
    "no_camouflage": "no_camo",
    "moderate_camouflage": "moderate",
    "heavy_camouflage": "heavy",
}


def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1]
    success, buf = cv2.imencode(ext, img)
    if success:
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return True
    return False


def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        model_path = os.path.join(MODELS_DIR, "yolo26s_v12.pt")
        if not os.path.exists(model_path):
            model_path = os.path.join(MODELS_DIR, "best_v10.pt")
        if os.path.exists(model_path):
            _yolo_model = YOLO(model_path)
        else:
            _yolo_model = YOLO(os.path.join(MODELS_DIR, "yolov8n.pt"))
    return _yolo_model


def get_qwen():
    global _qwen_agent
    if _qwen_agent is None:
        from qwen_agent import get_qwen_agent
        _qwen_agent = get_qwen_agent()
    return _qwen_agent


def draw_boxes(image, detections):
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        level = d.get("camouflage_level", "no_camouflage")
        color = LEVEL_COLORS.get(level, (255, 255, 255))
        label = f"{LEVEL_LABELS.get(level, level)} {d['confidence']:.2f}"
        if d.get("source") == "qwen_verified":
            label += " [AI]"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def crop_vehicle(img, bbox, margin=0.05):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    dw = int((x2 - x1) * margin)
    dh = int((y2 - y1) * margin)
    x1 = max(0, x1 - dw); y1 = max(0, y1 - dh)
    x2 = min(w, x2 + dw); y2 = min(h, y2 + dh)
    return img[y1:y2, x1:x2]


def encode_b64(img):
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def nms_boxes(boxes, iou_thresh=0.5):
    """Greedy NMS, boxes sorted by conf descending."""
    keep = []
    for box in boxes:
        too_close = False
        xa1, ya1, xa2, ya2 = box["xyxy"]
        area_a = (xa2 - xa1) * (ya2 - ya1)
        for k in keep:
            xb1, yb1, xb2, yb2 = k["xyxy"]
            xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
            xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
            if xi2 > xi1 and yi2 > yi1:
                inter = (xi2 - xi1) * (yi2 - yi1)
                area_b = (xb2 - xb1) * (yb2 - yb1)
                iou_val = inter / (area_a + area_b - inter)
                if iou_val > iou_thresh:
                    too_close = True
                    break
        if not too_close:
            keep.append(box)
    return keep


def qwen_verify_batch(qwen, crops, max_concurrent=3):
    """Verify multiple crops concurrently with Qwen (max 3 parallel)."""
    import concurrent.futures
    results = [None] * len(crops)

    def verify_one(idx, crop):
        if crop is None or crop.size == 0:
            return idx, {"is_military": False, "detail": "empty crop"}
        try:
            import requests
            img_b64 = encode_b64(crop)
            payload = {
                "model": qwen.model,
                "input": {"messages": [{"role": "user", "content": [
                    {"image": f"data:image/jpeg;base64,{img_b64}"},
                    {"text": "Is this image a military vehicle (tank, armored vehicle, etc)? Reply JSON only: {\"is_military\":true/false,\"detail\":\"<reason in Chinese, <15 words>\"}"}
                ]}]},
                "parameters": {"max_tokens": 80, "temperature": 0.05},
            }
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {qwen.api_key}"}
            resp = requests.post(qwen.api_url, json=payload, headers=headers, timeout=12)
            if resp.status_code != 200:
                return idx, {"is_military": False, "detail": f"API {resp.status_code}"}
            data = resp.json()
            content = data.get("output", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
            if isinstance(content, list):
                content = "".join([c.get("text", "") for c in content])
            text = content.strip()
            if "```" in text:
                text = text.split("```")[1].strip().removeprefix("json").strip()
            s = text.find("{"); e = text.rfind("}")
            if s >= 0 and e > s:
                rv = json.loads(text[s:e+1])
                return idx, {"is_military": bool(rv.get("is_military", False)), "detail": str(rv.get("detail", ""))}
            return idx, {"is_military": False, "detail": "parse error"}
        except Exception as ex:
            return idx, {"is_military": False, "detail": str(ex)[:60]}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = [executor.submit(verify_one, i, c) for i, c in enumerate(crops)]
        for f in concurrent.futures.as_completed(futures):
            idx, rv = f.result()
            results[idx] = rv
    return results


@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = Form(0.30), iou: float = Form(0.30)):
    import time as time_mod
    t0 = time_mod.time()

    contents = await file.read()
    img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    upload_id = str(uuid.uuid4())[:8]
    upload_path = os.path.join(static_dir, "uploads", f"{upload_id}.jpg")
    imwrite_unicode(upload_path, img_np)

    result = {
        "upload_id": upload_id,
        "filename": file.filename,
        "timestamp": datetime.now().isoformat(),
        "detections": [],
        "annotated_image": None,
    }

    try:
        model = get_yolo()
        qwen = get_qwen()

        # Run YOLO once with low threshold to catch all candidates
        yolo_conf = max(0.10, min(conf, 0.25))
        det_results = model(img_np, conf=yolo_conf, iou=iou, verbose=False)

        all_boxes = []
        if det_results[0].boxes is not None:
            boxes = det_results[0].boxes
            for i in range(len(boxes)):
                all_boxes.append({
                    "xyxy": boxes.xyxy[i].cpu().numpy(),
                    "conf": float(boxes.conf[i]),
                    "cls_id": int(boxes.cls[i]),
                })

        # Sort & NMS
        all_boxes.sort(key=lambda b: b["conf"], reverse=True)
        all_boxes = nms_boxes(all_boxes)

        # Split into high-conf and low-conf
        high_conf = [b for b in all_boxes if b["conf"] >= 0.30]
        low_conf = [b for b in all_boxes if 0.12 <= b["conf"] < 0.30]

        # Prepare crops for batch Qwen verification (low-conf only)
        low_crops = []
        for b in low_conf:
            try:
                low_crops.append(crop_vehicle(img_np, b["xyxy"]))
            except Exception:
                low_crops.append(None)

        # Batch verify low-conf boxes with Qwen
        verified = []
        if low_crops and qwen.api_url:
            verify_results = qwen_verify_batch(qwen, low_crops)
            for b, vr in zip(low_conf, verify_results):
                if vr and vr.get("is_military"):
                    b["verified"] = True
                    b["verify_detail"] = vr.get("detail", "")
                    verified.append(b)

        # Combine: high-conf trusted + Qwen-verified low-conf
        final_boxes = high_conf + verified

        # Build detections
        detections = []
        for b in final_boxes:
            xyxy = b["xyxy"]
            bbox = [round(float(v), 1) for v in xyxy]
            source = "qwen_verified" if b["conf"] < 0.30 else "yolo"

            # Camouflage assessment
            try:
                crop = crop_vehicle(img_np, xyxy)
                camo_result = qwen.assess(crop)
            except Exception:
                camo_result = type('obj', (object,), {
                    'level': 'no_camouflage', 'confidence': 0.0,
                    'detail': '', 'source': 'cv'
                })()

            detection = {
                "class": CLASS_NAMES.get(b["cls_id"], str(b["cls_id"])),
                "class_id": b["cls_id"],
                "confidence": round(b["conf"], 4),
                "bbox": bbox,
                "camouflage_level": camo_result.level,
                "camouflage_confidence": camo_result.confidence,
                "camouflage_detail": camo_result.detail,
                "camouflage_source": camo_result.source,
                "source": source,
            }
            if b.get("verify_detail"):
                detection["verify_detail"] = b["verify_detail"]
            detections.append(detection)

        result["detections"] = detections

        # Draw
        if detections:
            img_annotated = draw_boxes(img_np.copy(), detections)
            annotated_path = os.path.join(static_dir, "uploads", f"{upload_id}_annotated.jpg")
            imwrite_unicode(annotated_path, img_annotated)
            result["annotated_image"] = f"/static/uploads/{upload_id}_annotated.jpg"

        result["timing_ms"] = round((time_mod.time() - t0) * 1000)

    except Exception as e:
        result["error"] = str(e)

    # Scene analysis fallback
    if len(result["detections"]) == 0 and not result.get("error"):
        try:
            qwen = get_qwen()
            scene = qwen.scene_analyze(img_np)
            result["scene_analysis"] = scene
        except Exception as e:
            result["scene_analysis"] = {"has_military": False, "content_description": f"scene analysis failed: {str(e)[:80]}", "vehicles_found": []}

    # Save history
    try:
        summary = {
            "upload_id": result["upload_id"], "filename": result["filename"],
            "timestamp": result["timestamp"], "detection_count": len(result["detections"]),
            "original_image": f"/static/uploads/{result['upload_id']}.jpg",
            "annotated_image": result.get("annotated_image"),
            "scene_analysis": result.get("scene_analysis"),
            "detections": [
                {"class": d["class"], "confidence": d["confidence"],
                 "camouflage_level": d["camouflage_level"],
                 "camouflage_confidence": d["camouflage_confidence"],
                 "camouflage_detail": d["camouflage_detail"],
                 "bbox": d["bbox"], "source": d.get("source", "yolo")}
                for d in result["detections"]
            ],
        }
        save_history_entry(summary)
    except Exception:
        pass

    return JSONResponse(content=result)


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history_entry(entry):
    with _history_lock:
        history = load_history()
        history.insert(0, entry)
        if len(history) > 50:
            history = history[:50]
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)


@app.get("/history")
async def get_history():
    return JSONResponse(content={"history": load_history()})


@app.get("/history/{upload_id}")
async def get_history_detail(upload_id: str):
    for entry in load_history():
        if entry["upload_id"] == upload_id:
            return JSONResponse(content=entry)
    return JSONResponse(content={"error": "not found"}, status_code=404)


@app.delete("/history")
async def clear_history():
    with _history_lock:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
    return JSONResponse(content={"status": "ok"})


@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    message = body.get("message", "")
    image_path = body.get("image_path", "")
    context = body.get("context")

    if not message:
        return JSONResponse(content={"error": "no message"}, status_code=400)

    try:
        qwen = get_qwen()
        context_block = ""
        if context:
            dets = context.get("detections", [])
            if dets:
                context_block += "\n\n【Detection Context】\n"
                for i, d in enumerate(dets):
                    context_block += f"- Vehicle#{i+1}: class={d.get('class','?')}, camo={d.get('camouflage_level','?')}, detail={d.get('camouflage_detail','?')}\n"
            sa = context.get("scene_analysis")
            if sa:
                context_block += f"\n【Scene】military={sa.get('has_military',False)}, desc={sa.get('content_description','')}\n"

        prompt = f"""You are a military vehicle identification assistant.
{context_block if context_block else '(No pre-detection info)'}
User question: {message}
Answer in Chinese, concise, under 200 chars. Be professional about camouflage analysis and military vehicle identification."""

        import requests
        payload = {
            "model": qwen.model,
            "input": {"messages": [{"role": "user", "content": [{"text": prompt}]}]},
            "parameters": {"max_tokens": 1024, "temperature": 0.7},
        }
        if image_path:
            for ext in [image_path, image_path.replace("_annotated", "")]:
                fp = os.path.join(static_dir, ext.lstrip("/static/"))
                if os.path.exists(fp):
                    with open(fp, "rb") as f:
                        img_b64 = base64.b64encode(f.read()).decode("utf-8")
                    payload["input"]["messages"][0]["content"].insert(0, {"image": f"data:image/jpeg;base64,{img_b64}"})
                    break

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {qwen.api_key}"}
        resp = requests.post(qwen.api_url, json=payload, headers=headers, timeout=60)
        if resp.status_code != 200:
            return JSONResponse(content={"reply": f"API failed (HTTP {resp.status_code})"})

        data = resp.json()
        content = data.get("output", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            content = "".join([c.get("text", "") for c in content])
        return JSONResponse(content={"reply": content or "No response from AI"})

    except Exception as e:
        return JSONResponse(content={"reply": f"Error: {str(e)[:200]}"})


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def index():
    return FileResponse(os.path.join(templates_dir, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
