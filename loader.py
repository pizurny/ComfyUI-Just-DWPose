from __future__ import annotations

import sys, inspect
from pathlib import Path

# Point Python at: vendor/  (which contains custom_controlnet_aux/)
_THIS_DIR = Path(__file__).resolve().parent
_VENDOR_ROOT = _THIS_DIR / "vendor"
_CUSTOM_AUX = _VENDOR_ROOT / "custom_controlnet_aux"

if not _CUSTOM_AUX.is_dir():
    raise ImportError("Missing folder: vendor/custom_controlnet_aux (with dwpose/*.py and util.py)")

if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

# Import the detector through the package root so relative imports (..util) work
from custom_controlnet_aux.dwpose import DwposeDetector
print("[Just-DWPose] Using DwposeDetector from:",
      inspect.getsourcefile(DwposeDetector), "(impl=mini-vendor)")

# --- rest of imports ---
import os
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
from PIL import Image
import importlib

# -------------------- Filenames we support --------------------
TS_DET_CANDIDATES  : List[str] = ["yolox_l.torchscript.pt", "yolox_x.torchscript.pt", "yolox_s.torchscript.pt"]
TS_POSE_CANDIDATES : List[str] = ["dw-ll_ucoco_384_bs5.torchscript.pt", "dwpose.torchscript.pt"]

ONNX_DET_CANDIDATES  : List[str] = ["yolox_l.onnx", "yolox_x.onnx", "yolox_s.onnx"]
ONNX_POSE_CANDIDATES : List[str] = ["dw-ll_ucoco_384.onnx", "dw-ll_ucoco_384_bs5.onnx", "dwpose.onnx"]

# -------------------- Helpers --------------------
def _find_comfy_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "main.py").exists() and (p / "models").exists():
            return p
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
    try:
        with open(p, "rb") as f:
            return f.read(2) == b"PK"  # TorchScript archive is ZIP
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
    device: str            # "cuda" | "cpu" (torchscript only)

_LOGGED_ONCE = False

def _resolve_backend_and_paths(models_dir: Path, backend: str) -> _Resolved:
    # explicit choice or auto-pick
    if backend in ("torchscript", "onnx"):
        chosen = backend
    else:
        chosen = "torchscript" if (_first_existing(models_dir, TS_DET_CANDIDATES) and _first_existing(models_dir, TS_POSE_CANDIDATES)) \
                 else "onnx" if (_first_existing(models_dir, ONNX_DET_CANDIDATES) and _first_existing(models_dir, ONNX_POSE_CANDIDATES)) \
                 else "none"

    if chosen == "torchscript":
        det = _first_existing(models_dir, TS_DET_CANDIDATES)
        pose = _first_existing(models_dir, TS_POSE_CANDIDATES)
        if not (det and pose):
            raise RuntimeError(
                f"Missing TorchScript weights in {models_dir}.\n"
                f"Detector one of: {TS_DET_CANDIDATES}\nPose one of: {TS_POSE_CANDIDATES}"
            )
        # Validate TS archives
        bad = [p.name for p in (det, pose) if not _looks_like_torchscript_zip(p)]
        if bad:
            # Try graceful ONNX fallback if present
            od, op = _first_existing(models_dir, ONNX_DET_CANDIDATES), _first_existing(models_dir, ONNX_POSE_CANDIDATES)
            if od and op:
                print("[Just-DWPose] Invalid TorchScript files:", ", ".join(bad), "– falling back to ONNX.")
                return _Resolved("onnx", od, op, device="cpu")
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
                f"Detector one of: {ONNX_DET_CANDIDATES}\nPose one of: {ONNX_POSE_CANDIDATES}"
            )
        return _Resolved("onnx", det, pose, device="cpu")

    raise RuntimeError(
        f"No DWPose weights found in {models_dir}.\n"
        f"Place TorchScript: {TS_DET_CANDIDATES[0]} + {TS_POSE_CANDIDATES[0]}\n"
        f"or ONNX: {ONNX_DET_CANDIDATES[0]} + {ONNX_POSE_CANDIDATES[0]}"
    )

# -------------------- Robust loader for Aux API differences --------------------
def _load_dwpose(local_dir: str, det_name: str, pose_name: str, backend: str, device: str):
    """
    Call DwposeDetector.from_pretrained(...) but adapt to different Aux versions.
    We detect accepted kwargs via inspect and filter accordingly.
    """
    ts_dev = device if backend == "torchscript" else None
    candidates = {
        "pretrained_model_or_path": local_dir,
        "pose_model_or_path": local_dir,
        "det_filename": det_name,
        "pose_filename": pose_name,
        "torchscript_device": ts_dev,
        "local_files_only": True,
    }

    sig = inspect.signature(DwposeDetector.from_pretrained)
    accepted = set(sig.parameters.keys())

    # If this API uses "device" instead of "torchscript_device"
    if "torchscript_device" not in accepted and ts_dev is not None and "device" in accepted:
        candidates["device"] = ts_dev
    # Remove unaccepted kwargs
    kwargs = {k: v for k, v in candidates.items() if k in accepted and v is not None}

    try:
        return DwposeDetector.from_pretrained(**kwargs)
    except TypeError:
        # Very old positional style? Try minimal positional + filenames as kwargs.
        try:
            return DwposeDetector.from_pretrained(local_dir, det_filename=det_name, pose_filename=pose_name)
        except TypeError:
            # Last resort: positional only
            return DwposeDetector.from_pretrained(local_dir)

# -------------------- DWpose run --------------------
def _run_with_vendored_aux(
    pil_image: Image.Image,
    resolved: _Resolved,
    models_dir: Path,
    detect_resolution: int,
    include_body: bool,
    include_hands: bool,
    include_face: bool,
) -> Tuple[Image.Image, Dict[str, Any]]:
    # Force offline behavior
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    global _LOGGED_ONCE
    if not _LOGGED_ONCE:
        print(
            f"[Just-DWPose] backend={resolved.backend} device={resolved.device} "
            f"det={resolved.det_path.name} pose={resolved.pose_path.name} dir={models_dir}"
        )
        _LOGGED_ONCE = True

    if resolved.backend == "onnx":
        os.environ.setdefault("AUX_ORT_PROVIDERS", "CUDAExecutionProvider;CPUExecutionProvider")

    det = _load_dwpose(
        local_dir=str(models_dir.resolve()),
        det_name=resolved.det_path.name,
        pose_name=resolved.pose_path.name,
        backend=resolved.backend,
        device=resolved.device,
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
    offline_ok: bool,     # unused
    allow_download: bool, # unused
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Run DWPose once and return (overlay_image, pose_json_dict).
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
