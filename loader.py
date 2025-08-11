from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, List

# ---- Vendored import (no external comfyui_controlnet_aux) ----
try:
    from .vendor.dwpose.detector import DwposeDetector
except Exception as e:
    raise ImportError(
        "Failed to import vendored DwposeDetector. "
        "Ensure vendor/dwpose/__init__.py and vendor/dwpose/detector.py exist. "
        f"Original error: {e}"
    )

import inspect as _inspect
print("[Just-DWPose] Using DwposeDetector from:", _inspect.getsourcefile(DwposeDetector))

from PIL import Image

# -------------------- Filenames we support --------------------
# TorchScript (your files + a couple common aliases)
TS_DET_CANDIDATES  : List[str] = ["yolox_l.torchscript.pt", "yolox_x.torchscript.pt", "yolox_s.torchscript.pt"]
TS_POSE_CANDIDATES : List[str] = ["dw-ll_ucoco_384_bs5.torchscript.pt", "dwpose.torchscript.pt"]

# ONNX (primary + common alias)
ONNX_DET_CANDIDATES  : List[str] = ["yolox_l.onnx", "yolox_x.onnx", "yolox_s.onnx"]
ONNX_POSE_CANDIDATES : List[str] = ["dw-ll_ucoco_384.onnx", "dw-ll_ucoco_384_bs5.onnx", "dwpose.onnx"]

# -------------------- Helpers --------------------
def _find_comfy_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "main.py").exists() and (p / "models").exists():
            return p
    # Fallback: typical custom_nodes/…/file.py → comfy_root at parents[2]
    return here.parents[2]

def get_models_dir() -> Path:
    return (_find_comfy_root() / "models" / "checkpoints" / "DWPose").resolve()

def _first_existing(d: Path, candidates: List[str]) -> Path | None:
    for name in candidates:
        p = d / name
        if p.is_file():
            return p
    return None

def _looks_like_torchscript_zip(p: Path) -> bool:
    # TorchScript archives are ZIP files (magic "PK")
    try:
        with open(p, "rb") as f:
            return f.read(2) == b"PK"
    except Exception:
        return False

def _torch_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

@dataclass
class _Resolved:
    backend: str           # "torchscript" | "onnx"
    det_path: Path
    pose_path: Path
    device: str            # "cuda" | "cpu" (used for torchscript)

_LOGGED_ONCE = False

def _resolve_backend_and_paths(models_dir: Path, backend: str) -> _Resolved:
    # normalize backend choice
    chosen: str
    if backend in ("torchscript", "onnx"):
        chosen = backend
    else:
        # auto
        chosen = "torchscript" if _first_existing(models_dir, TS_DET_CANDIDATES) and _first_existing(models_dir, TS_POSE_CANDIDATES) \
            else "onnx" if _first_existing(models_dir, ONNX_DET_CANDIDATES) and _first_existing(models_dir, ONNX_POSE_CANDIDATES) \
            else "none"

    if chosen == "torchscript":
        det = _first_existing(models_dir, TS_DET_CANDIDATES)
        pose = _first_existing(models_dir, TS_POSE_CANDIDATES)
        if not (det and pose):
            raise RuntimeError(
                f"Missing TorchScript weights in {models_dir}.\n"
                f"Expected detector one of: {TS_DET_CANDIDATES}\n"
                f"Expected pose     one of: {TS_POSE_CANDIDATES}"
            )
        # Fail early if files aren’t real TorchScript zips
        bad = [p.name for p in (det, pose) if not _looks_like_torchscript_zip(p)]
        if bad:
            raise RuntimeError(
                "These files are not valid TorchScript archives (expected ZIP header 'PK'): "
                + ", ".join(bad)
                + "\nIf you see HTML/JSON in the file when opened in a text editor, re-download the correct TorchScript .pt."
            )
        device = "cuda" if _torch_cuda_available() else "cpu"
        return _Resolved("torchscript", det, pose, device)

    if chosen == "onnx":
        det = _first_existing(models_dir, ONNX_DET_CANDIDATES)
        pose = _first_existing(models_dir, ONNX_POSE_CANDIDATES)
        if not (det and pose):
            raise RuntimeError(
                f"Missing ONNX weights in {models_dir}.\n"
                f"Expected detector one of: {ONNX_DET_CANDIDATES}\n"
                f"Expected pose     one of: {ONNX_POSE_CANDIDATES}"
            )
        return _Resolved("onnx", det, pose, device="cpu")  # device not used by ORT here

    # none
    raise RuntimeError(
        f"No DWPose weights found in {models_dir}.\n"
        f"Place TorchScript: {TS_DET_CANDIDATES[0]} + {TS_POSE_CANDIDATES[0]}\n"
        f"or ONNX: {ONNX_DET_CANDIDATES[0]} + {ONNX_POSE_CANDIDATES[0]}"
    )

# -------------------- DWpose (vendored) --------------------
def _run_with_vendored_aux(
    pil_image: Image.Image,
    resolved: _Resolved,
    models_dir: Path,
    detect_resolution: int,
    include_body: bool,
    include_hands: bool,
    include_face: bool,
) -> Tuple[Image.Image, Dict[str, Any]]:
    # Hard offline: force local-only behavior for any downstream libs
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    local_dir = str(models_dir.resolve())  # directory with files

    # One-time startup log
    global _LOGGED_ONCE
    if not _LOGGED_ONCE:
        print(
            f"[Just-DWPose] backend={resolved.backend} device={resolved.device} "
            f"det={resolved.det_path.name} pose={resolved.pose_path.name} dir={models_dir}"
        )
        _LOGGED_ONCE = True

    if resolved.backend == "torchscript":
        det = DwposeDetector.from_pretrained(
            pretrained_model_or_path=local_dir,
            pose_model_or_path=local_dir,
            det_filename=resolved.det_path.name,
            pose_filename=resolved.pose_path.name,
            torchscript_device=resolved.device,
            local_files_only=True,
        )
    else:  # onnx
        # Prefer CUDA if ORT-GPU is installed; still fine if CPU-only
        os.environ.setdefault("AUX_ORT_PROVIDERS", "CUDAExecutionProvider;CPUExecutionProvider")
        det = DwposeDetector.from_pretrained(
            pretrained_model_or_path=local_dir,
            pose_model_or_path=local_dir,
            det_filename=resolved.det_path.name,
            pose_filename=resolved.pose_path.name,
            local_files_only=True,
        )

    pose_img, json_dict = det(
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
    offline_ok: bool,     # kept for signature compatibility (currently unused)
    allow_download: bool, # kept for signature compatibility (currently unused)
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Run DWPose once and return (overlay_image, pose_json_dict).

    backend: "torchscript", "onnx", or anything else for auto-pick.
    models_dir: path to ComfyUI/models/checkpoints/DWPose with local files.
    """
    resolved = _resolve_backend_and_paths(models_dir, backend)
    return _run_with_vendored_aux(
        pil_image=pil_image,
        resolved=resolved,
        models_dir=models_dir,
        detect_resolution=detect_resolution,
        include_body=include_body,
        include_hands=include_hands,
        include_face=include_face,
    )
