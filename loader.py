from __future__ import annotations
from typing import Tuple, Dict, Any, List
from pathlib import Path
import os, math
import numpy as np
from PIL import Image, ImageDraw

# Supported file names (match DWPose releases)
TS_YOLO = "yolox_l.torchscript.pt"
TS_POSE = "dw-ll_ucoco_384_bs5.torchscript.pt"
ONNX_YOLO = "yolox_l.onnx"
ONNX_POSE = "dw-ll_ucoco_384.onnx"

# -------------------- path helpers --------------------
def _find_comfy_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "main.py").exists() and (p / "models").exists():
            return p
    return here.parents[2]

def get_models_dir() -> Path:
    return (_find_comfy_root() / "models" / "checkpoints" / "DWPose").resolve()

def _have_onnx_pair(d: Path) -> bool:
    return (d / ONNX_YOLO).exists() and (d / ONNX_POSE).exists()

def _have_ts_pair(d: Path) -> bool:
    return (d / TS_YOLO).exists() and (d / TS_POSE).exists()

def _pick_backend(models_dir: Path, backend: str) -> str:
    if backend == "onnx":
        if not _have_onnx_pair(models_dir):
            raise RuntimeError(f"Missing ONNX weights in {models_dir}: {ONNX_YOLO}, {ONNX_POSE}")
        return "onnx"
    if backend == "torchscript":
        if not _have_ts_pair(models_dir):
            raise RuntimeError(f"Missing TorchScript weights in {models_dir}: {TS_YOLO}, {TS_POSE}")
        # NOTE: TorchScript internal runner arrives next commit.
        raise RuntimeError("TorchScript backend not implemented yet in standalone mode. "
                           "Set backend='onnx' for now.")
    # auto
    if _have_onnx_pair(models_dir):
        return "onnx"
    if _have_ts_pair(models_dir):
        raise RuntimeError("Only TorchScript weights found, but TS backend not implemented yet. "
                           "Add ONNX pair or wait for next commit.")
    raise RuntimeError(
        f"No DWPose weights found in {models_dir}. "
        f"Put either ONNX ({ONNX_YOLO}, {ONNX_POSE}) or TorchScript pair."
    )

def _prepare_env(models_dir: Path, offline_ok: bool):
    os.environ.setdefault("HF_HOME", str(models_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(models_dir))
    if offline_ok:
        os.environ["HF_HUB_OFFLINE"] = "1"

# -------------------- image utils --------------------
def _letterbox(img: np.ndarray, new_size=(640, 640), color=114) -> Tuple[np.ndarray, float, Tuple[int,int]]:
    """Resize with unchanged aspect ratio and pad to new_size. Returns img, scale, pad (dw, dh)."""
    h, w = img.shape[:2]
    r = min(new_size[1] / h, new_size[0] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = np.array(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    canvas = np.full((new_size[1], new_size[0], 3), color, dtype=resized.dtype)
    dw, dh = (new_size[0] - nw) // 2, (new_size[1] - nh) // 2
    canvas[dh:dh+nh, dw:dw+nw] = resized
    return canvas, r, (dw, dh)

def _nms(boxes: np.ndarray, scores: np.ndarray, thresh=0.5) -> List[int]:
    """Non-maximum suppression; boxes: [N,4] (x0,y0,x1,y1)."""
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

# -------------------- ONNX runtime sessions --------------------
_ORT = None  # lazy import
_YOLO_SESS = None
_POSE_SESS = None

def _ort() :
    global _ORT
    if _ORT is None:
        import onnxruntime as ort
        _ORT = ort
    return _ORT

def _providers():
    ort = _ort()
    # prefer CUDA if present, else CPU
    return ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]

def _ensure_onnx(models_dir: Path):
    global _YOLO_SESS, _POSE_SESS
    if _YOLO_SESS is None:
        _YOLO_SESS = _ort().InferenceSession(str(models_dir / ONNX_YOLO), providers=_providers())
    if _POSE_SESS is None:
        _POSE_SESS = _ort().InferenceSession(str(models_dir / ONNX_POSE), providers=_providers())

# -------------------- YOLOX (person) detection --------------------
def _yolox_detect_person(img_rgb: np.ndarray, conf_thres=0.25, nms_thres=0.5) -> List[np.ndarray]:
    """
    Returns a list of person boxes [x0,y0,x1,y1] in original image coordinates.
    """
    _ensure_onnx(get_models_dir())  # sessions
    h0, w0 = img_rgb.shape[:2]
    inp, r, (dw, dh) = _letterbox(img_rgb, (640, 640))
    # normalize
    x = inp.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]  # 1x3x640x640

    sess = _YOLO_SESS
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: x})  # expect (1, N, 85)
    pred = out[0]
    if isinstance(pred, list):  # just in case
        pred = pred[0]
    pred = np.squeeze(pred, axis=0)  # N x 85

    # YOLOX usually gives cx,cy,w,h,obj,80cls
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    obj = 1.0 / (1.0 + np.exp(-pred[:, 4]))  # sigmoid
    cls_scores = 1.0 / (1.0 + np.exp(-pred[:, 5:]))  # sigmoid
    person_scores = cls_scores[:, 0]  # class 0 = person (COCO)
    conf = obj * person_scores
    keep = conf > conf_thres
    if not np.any(keep):
        return []

    cx, cy, w, h, conf = cx[keep], cy[keep], w[keep], h[keep], conf[keep]
    # xyxy on letterboxed image
    x0 = cx - w / 2
    y0 = cy - h / 2
    x1 = cx + w / 2
    y1 = cy + h / 2
    boxes = np.stack([x0, y0, x1, y1], axis=1)

    # undo letterbox to original scale
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes /= r
    # clip
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w0 - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h0 - 1)

    # NMS
    keep_idx = _nms(boxes, conf, thresh=nms_thres)
    return [boxes[i] for i in keep_idx]

# -------------------- Pose estimator (DWpose ONNX) --------------------
# COCO-17 skeleton pairs for drawing
_COCO_PAIRS = [
    (0,1),(0,2),(1,3),(2,4),(5,7),(7,9),(6,8),(8,10),
    (5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
]

def _crop_expand(img: np.ndarray, box: np.ndarray, expand=1.25) -> Tuple[np.ndarray, Tuple[int,int]]:
    x0, y0, x1, y1 = box.astype(int)
    cx, cy = (x0+x1)/2, (y0+y1)/2
    w, h = (x1-x0), (y1-y0)
    s = int(max(w, h) * expand)
    nx0 = int(max(cx - s/2, 0)); ny0 = int(max(cy - s/2, 0))
    nx1 = int(min(cx + s/2, img.shape[1]-1)); ny1 = int(min(cy + s/2, img.shape[0]-1))
    crop = img[ny0:ny1, nx0:nx1]
    return crop, (nx0, ny0)

def _pose_infer(crop_rgb: np.ndarray) -> np.ndarray:
    """
    Returns keypoints array [17,3] with (x, y, score) in crop coordinates.
    The DWpose ONNX in practice outputs either keypoints directly (N,17,3) or heatmaps.
    We handle both: if heatmaps come, take argmax per joint.
    """
    _ensure_onnx(get_models_dir())
    # prepare 384x384
    size = 384
    pil = Image.fromarray(crop_rgb).resize((size, size), Image.BILINEAR)
    x = np.asarray(pil).astype(np.float32) / 255.0
    x = np.transpose(x, (2,0,1))[None, ...]  # 1x3x384x384

    sess = _POSE_SESS
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: x})
    y = out[0]

    # case A: direct keypoints (1,17,3)
    if y.ndim == 3 and y.shape[-2] == 17 and y.shape[-1] == 3:
        return y[0]

    # case B: heatmaps (1,17,H,W) -> argmax
    if y.ndim == 4 and y.shape[1] >= 17:
        hm = y[0, :17, :, :]  # 17xHxW
        H, W = hm.shape[-2], hm.shape[-1]
        flat_idx = hm.reshape(17, -1).argmax(axis=1)
        ys = (flat_idx // W).astype(np.float32)
        xs = (flat_idx %  W).astype(np.float32)
        scores = hm.max(axis=(1,2))
        # map to 384x384
        xs = xs * (size / W)
        ys = ys * (size / H)
        return np.stack([xs, ys, scores], axis=1)

    raise RuntimeError("Unexpected DWpose ONNX output shape: {}".format([a.shape for a in out]))

def _to_image_and_json(img_rgb: np.ndarray, box: np.ndarray, kp_crop: np.ndarray, crop_origin: Tuple[int,int]) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Convert crop keypoints to full-image coordinates, draw, and format OpenPose-style JSON.
    """
    size = 384
    # Map crop coords back
    ox, oy = crop_origin
    # If crop was resized to 384x384, scale back by crop size / 384
    crop_w = box[2] - box[0]
    crop_h = box[3] - box[1]
    scale_x = crop_w / size
    scale_y = crop_h / size

    pts = []
    for i in range(17):
        x = float(kp_crop[i,0] * scale_x + ox)
        y = float(kp_crop[i,1] * scale_y + oy)
        c = float(max(0.0, min(1.0, kp_crop[i,2])))
        pts.append((x,y,c))

    # draw
    canvas = Image.fromarray(img_rgb.copy())
    draw = ImageDraw.Draw(canvas)
    for (i,j) in _COCO_PAIRS:
        xi, yi, ci = pts[i]
        xj, yj, cj = pts[j]
        if ci > 0.1 and cj > 0.1:
            draw.line([(xi,yi),(xj,yj)], width=3)
    for (x,y,c) in pts:
        if c > 0.1:
            r = 3
            draw.ellipse((x-r,y-r,x+r,y+r), width=2)

    # OpenPose-like JSON (flat list x,y,c for 17 keypoints)
    people = [{
        "pose_keypoints_2d": sum(([p[0], p[1], p[2]] for p in pts), []),
        "score": float(np.mean([p[2] for p in pts])),
    }]
    return canvas, {"version": 1.0, "people": people}

# -------------------- public API --------------------
def run_dwpose_once(
    pil_image: Image.Image,
    backend: str,
    detect_resolution: int,
    include_body: bool,
    include_hands: bool,
    include_face: bool,
    models_dir: Path,
    offline_ok: bool,
    allow_download: bool,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Standalone ONNX implementation:
      - YOLOX (person) -> pick the largest person box
      - DWpose -> 17 keypoints
      - Returns (pose image PIL, OpenPose-like JSON dict)
    Notes:
      - Hands/Face flags are placeholders for now (future sub-models).
      - detect_resolution is used implicitly by fixed 640/384 sizes.
    """
    _prepare_env(models_dir, offline_ok)
    chosen = _pick_backend(models_dir, backend)
    if chosen != "onnx":
        # see _pick_backend note
        raise RuntimeError("Use backend='onnx' for the current standalone build.")

    img_rgb = np.asarray(pil_image.convert("RGB"))
    boxes = _yolox_detect_person(img_rgb, conf_thres=0.25, nms_thres=0.5)
    if not boxes:
        # return original + empty json
        return Image.fromarray(img_rgb), {"version": 1.0, "people": []}

    # choose largest person
    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    box = boxes[int(np.argmax(areas))]

    crop, origin = _crop_expand(img_rgb, box, expand=1.25)
    keypoints = _pose_infer(crop)
    pose_img, json_dict = _to_image_and_json(img_rgb, box, keypoints, origin)
    return pose_img, json_dict
