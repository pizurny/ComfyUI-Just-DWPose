from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple, Dict, Any
from PIL import Image

# Filenames supported (ControlNet-Aux conventions)
TS_YOLO = "yolox_l.torchscript.pt"
TS_POSE = "dw-ll_ucoco_384_bs5.torchscript.pt"
ONNX_YOLO = "yolox_l.onnx"
ONNX_POSE = "dw-ll_ucoco_384.onnx"

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
    if _have_ts_pair(models_dir): return "torchscript"
    if _have_onnx_pair(models_dir): return "onnx"
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

def _prepare_env(models_dir: Path, offline_ok: bool):
    os.environ.setdefault("HF_HOME", str(models_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(models_dir))
    if offline_ok:
        os.environ["HF_HUB_OFFLINE"] = "1"

def _run_with_controlnet_aux(
    pil_image: Image.Image,
    backend: str,
    models_dir: Path,
    detect_resolution: int,
    include_body: bool,
    include_hands: bool,
    include_face: bool,
) -> Tuple[Image.Image, Dict[str, Any]]:
    try:
        from controlnet_aux.dwpose import DWposeDetector  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "controlnet-aux is not installed. Install it for the built-in runner: pip install controlnet-aux"
        ) from e

    device = "cuda" if _torch_cuda() else "cpu"
    if backend == "torchscript":
        detector = DWposeDetector(
            device=device, backend="torchscript",
            yolo_model_path=str(models_dir / TS_YOLO),
            pose_model_path=str(models_dir / TS_POSE),
        )
    else:
        detector = DWposeDetector(
            device=device, backend="onnx",
            yolo_model_path=str(models_dir / ONNX_YOLO),
            pose_model_path=str(models_dir / ONNX_POSE),
        )

    pose_img, json_dict, _ = detector(
        pil_image,
        include_hand=include_hands,
        include_face=include_face,
        include_body=include_body,
        detect_resolution=int(detect_resolution),
        image_and_json=True,
    )
    return pose_img, json_dict

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
):
    _prepare_env(models_dir, offline_ok)
    chosen = _pick_backend(models_dir, backend)
    # Prefer proven ControlNet-Aux implementation
    return _run_with_controlnet_aux(
        pil_image,
        backend=chosen,
        models_dir=models_dir,
        detect_resolution=detect_resolution,
        include_body=include_body,
        include_hands=include_hands,
        include_face=include_face,
    )
