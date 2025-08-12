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
    detection_threshold: float = 0.3,
    nms_threshold: float = 0.45,
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

    # Get pose results with custom threshold handling
    pose_img, json_dict = _run_dwpose_with_thresholds(
        det,
        pil_image,
        include_hands=include_hands,
        include_face=include_face,
        include_body=include_body,
        detect_resolution=int(detect_resolution),
        detection_threshold=detection_threshold,
        nms_threshold=nms_threshold,
    )
    return pose_img, json_dict

def _run_dwpose_with_thresholds(
    det,
    pil_image,
    include_hands: bool,
    include_face: bool,
    include_body: bool,
    detect_resolution: int,
    detection_threshold: float,
    nms_threshold: float,
):
    """Run DWPose detection with custom thresholds by post-processing results."""
    import json
    import numpy as np
    from custom_controlnet_aux.dwpose import draw_poses
    from custom_controlnet_aux.dwpose.types import PoseResult, BodyResult, HandResult, FaceResult, Keypoint
    
    # If using default threshold (0.3), use original method for compatibility
    if abs(detection_threshold - 0.3) < 0.001:
        return det(
            pil_image,
            include_hand=include_hands,
            include_face=include_face,
            include_body=include_body,
            detect_resolution=detect_resolution,
            image_and_json=True,
        )
    
    # For custom thresholds, use our filtering approach
    input_image = np.array(pil_image)
    
    # Get raw poses by calling detect_poses directly
    poses = det.detect_poses(input_image)
    
    # Debug output removed - we now understand the structure
    # Apply custom thresholds to filter keypoints
    filtered_poses = []
    for pose in poses:
        try:
            # Filter body keypoints
            body_result = None
            if hasattr(pose, 'body') and pose.body and hasattr(pose.body, 'keypoints') and pose.body.keypoints:
                body_keypoints = [
                    Keypoint(kp.x, kp.y, kp.score, kp.id) if kp and kp.score >= detection_threshold else None
                    for kp in pose.body.keypoints
                ]
                # Only keep body if at least one keypoint remains
                if any(kp is not None for kp in body_keypoints):
                    body_result = BodyResult(body_keypoints, pose.body.total_score, pose.body.total_parts)
            else:
                body_result = getattr(pose, 'body', None)
            
            # Filter hand keypoints
            left_hand = None
            right_hand = None
            if hasattr(pose, 'left_hand') and pose.left_hand and hasattr(pose.left_hand, 'keypoints') and pose.left_hand.keypoints:
                left_keypoints = [
                    Keypoint(kp.x, kp.y, kp.score, kp.id) if kp and kp.score >= detection_threshold else None
                    for kp in pose.left_hand.keypoints
                ]
                if any(kp is not None for kp in left_keypoints):
                    left_hand = HandResult(left_keypoints, pose.left_hand.total_score)
            
            if hasattr(pose, 'right_hand') and pose.right_hand and hasattr(pose.right_hand, 'keypoints') and pose.right_hand.keypoints:
                right_keypoints = [
                    Keypoint(kp.x, kp.y, kp.score, kp.id) if kp and kp.score >= detection_threshold else None
                    for kp in pose.right_hand.keypoints
                ]
                if any(kp is not None for kp in right_keypoints):
                    right_hand = HandResult(right_keypoints, pose.right_hand.total_score)
            
            # Filter face keypoints
            face_result = None
            if hasattr(pose, 'face') and pose.face and hasattr(pose.face, 'keypoints') and pose.face.keypoints:
                face_keypoints = [
                    Keypoint(kp.x, kp.y, kp.score, kp.id) if kp and kp.score >= detection_threshold else None
                    for kp in pose.face.keypoints
                ]
                if any(kp is not None for kp in face_keypoints):
                    face_result = FaceResult(face_keypoints, pose.face.total_score)
            
            # Create filtered pose result
            filtered_pose = PoseResult(body_result, left_hand, right_hand, face_result)
            filtered_poses.append(filtered_pose)
            
        except Exception as e:
            print(f"[DEBUG] Error processing pose: {e}")
            # Fallback: just add the original pose
            filtered_poses.append(pose)
    
    # Generate pose image from filtered results
    canvas = draw_poses(
        filtered_poses, 
        input_image.shape[0], 
        input_image.shape[1], 
        draw_body=include_body, 
        draw_hand=include_hands, 
        draw_face=include_face
    )
    
    # Resize to target resolution
    from custom_controlnet_aux.util import resize_image_with_pad, HWC3
    canvas, remove_pad = resize_image_with_pad(canvas, detect_resolution, "INTER_CUBIC")
    detected_map = HWC3(remove_pad(canvas))
    
    pose_img = Image.fromarray(detected_map, 'RGB')
    
    # Generate JSON dict similar to the original format
    json_dict = {
        "version": "ap10k",
        "people": []
    }
    
    for pose in filtered_poses:
        person_dict = {}
        
        if pose.body and pose.body.keypoints:
            # Convert body keypoints to OpenPose format: [x1, y1, conf1, x2, y2, conf2, ...]
            pose_keypoints_2d = []
            for kp in pose.body.keypoints:
                if kp is not None:
                    pose_keypoints_2d.extend([kp.x, kp.y, kp.score])
                else:
                    pose_keypoints_2d.extend([0.0, 0.0, 0.0])
            person_dict["pose_keypoints_2d"] = pose_keypoints_2d
        
        if pose.left_hand and pose.left_hand.keypoints:
            hand_keypoints = []
            for kp in pose.left_hand.keypoints:
                if kp is not None:
                    hand_keypoints.extend([kp.x, kp.y, kp.score])
                else:
                    hand_keypoints.extend([0.0, 0.0, 0.0])
            person_dict["hand_left_keypoints_2d"] = hand_keypoints
        
        if pose.right_hand and pose.right_hand.keypoints:
            hand_keypoints = []
            for kp in pose.right_hand.keypoints:
                if kp is not None:
                    hand_keypoints.extend([kp.x, kp.y, kp.score])
                else:
                    hand_keypoints.extend([0.0, 0.0, 0.0])
            person_dict["hand_right_keypoints_2d"] = hand_keypoints
        
        if pose.face and pose.face.keypoints:
            face_keypoints = []
            for kp in pose.face.keypoints:
                if kp is not None:
                    face_keypoints.extend([kp.x, kp.y, kp.score])
                else:
                    face_keypoints.extend([0.0, 0.0, 0.0])
            person_dict["face_keypoints_2d"] = face_keypoints
        
        json_dict["people"].append(person_dict)
    
    json_dict["canvas_height"] = input_image.shape[0]
    json_dict["canvas_width"] = input_image.shape[1]
    
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
    detection_threshold: float = 0.3,
    nms_threshold: float = 0.45,
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
        detection_threshold=detection_threshold,
        nms_threshold=nms_threshold,
    )
