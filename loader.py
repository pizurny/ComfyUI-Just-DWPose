from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, Any
import os

from PIL import Image

# Filenames we support (match Aux conventions)
TS_YOLO = "yolox_l.torchscript.pt"
TS_POSE = "dw-ll_ucoco_384_bs5.torchscript.pt"
ONNX_YOLO = "yolox_l.onnx"
ONNX_POSE = "dw-ll_ucoco_384.onnx"

# -------------------- Comfy paths --------------------
def _find_comfy_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "main.py").exists() and (p / "models").exists():
            return p
    return here.parents[2]

def get_models_dir() -> Path:
    return (_find_comfy_root() / "models" / "checkpoints" / "DWPose").resolve()

def _have_ts_pair(d: Path) -> bool:
    return (d / TS_YOLO).exists() and (d / TS_POSE).exists()

def _have_onnx_pair(d: Path) -> bool:
    return (d / ONNX_YOLO).exists() and (d / ONNX_POSE).exists()

def _pick_backend(models_dir: Path, backend: str) -> str:
    if backend == "torchscript":
        if not _have_ts_pair(models_dir):
            raise RuntimeError(f"Missing TorchScript weights in {models_dir}: {TS_YOLO}, {TS_POSE}")
        return "torchscript"
    if backend == "onnx":
        if not _have_onnx_pair(models_dir):
            raise RuntimeError(f"Missing ONNX weights in {models_dir}: {ONNX_YOLO}, {ONNX_POSE}")
        return "onnx"
    # auto
    if _have_ts_pair(models_dir):
        return "torchscript"
    if _have_onnx_pair(models_dir):
        return "onnx"
    raise RuntimeError(
        f"No DWPose weights found in {models_dir}.\n"
        f"Place TorchScript: {TS_YOLO}, {TS_POSE}\n"
        f"or ONNX: {ONNX_YOLO}, {ONNX_POSE}"
    )

def _torch_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

# -------------------- DWpose via controlnet-aux --------------------
def _run_with_aux(
    pil_image: Image.Image,
    backend: str,
    models_dir: Path,
    detect_resolution: int,
    include_body: bool,
    include_hands: bool,
    include_face: bool,
) -> Tuple[Image.Image, Dict[str, Any]]:
    try:
        from controlnet_aux.dwpose import DWposeDetector
    except Exception as e:
        raise RuntimeError(
            "controlnet-aux package is not installed. Install it in Comfy's Python:\n"
            "  python -m pip install controlnet-aux\n"
            f"Original import error: {e}"
        )

    device = "cuda" if _torch_cuda() else "cpu"

    if backend == "torchscript":
        yolo_path = str(models_dir / TS_YOLO)
        pose_path = str(models_dir / TS_POSE)
    else:
        yolo_path = str(models_dir / ONNX_YOLO)
        pose_path = str(models_dir / ONNX_POSE)

    # Make sure HF hub never writes outside (and offline stays offline)
    os.environ.setdefault("HF_HOME", str(models_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(models_dir))

    det = DWposeDetector(
        device=device,
        backend=backend,
        yolo_model_path=yolo_path,
        pose_model_path=pose_path,
    )

    # Returns: (pose_img, json_dict, src_img)
    pose_img, json_dict, _ = det(
        pil_image,
        include_hand=include_hands,
        include_face=include_face,
        include_body=include_body,
        detect_resolution=int(detect_resolution),
        image_and_json=True,
    )
    return pose_img, json_dict

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
    Aux-style runner: use DWposeDetector from controlnet-aux, but force local weights.
    """
    chosen = _pick_backend(models_dir, backend)
    return _run_with_aux(
        pil_image,
        backend=chosen,
        models_dir=models_dir,
        detect_resolution=detect_resolution,
        include_body=include_body,
        include_hands=include_hands,
        include_face=include_face,
    )
