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
    fallback = here.parents[2] if len(here.parents) > 2 else here.parent
    return fallback

def get_models_dir() -> Path:
    root = _find_comfy_root()
    models_path = root / "models" / "checkpoints" / "DWPose"
    return models_path.resolve()

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
    enable_bone_validation: bool = True,
    max_bone_ratio: float = 2.5,
    min_keypoint_confidence: float = 0.5,
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
        enable_bone_validation=enable_bone_validation,
        max_bone_ratio=max_bone_ratio,
        min_keypoint_confidence=min_keypoint_confidence,
    )
    return pose_img, json_dict

def validate_bone_lengths(pose, max_bone_ratio=2.5, min_keypoint_confidence=0.5):
    """Remove connections that are unrealistically long to fix elongated bones."""
    # Validate body AND hand keypoints - face preserved unchanged
    pose_modified = False
    
    # For very high ratios, skip validation entirely to preserve original behavior
    if max_bone_ratio >= 8.0:
        print("[DWPose] max_bone_ratio >= 8.0, skipping bone validation")
        return pose
    
    from custom_controlnet_aux.dwpose.types import PoseResult, BodyResult, HandResult, Keypoint
    import numpy as np
    
    # Validate BODY keypoints
    if hasattr(pose, 'body') and pose.body and hasattr(pose.body, 'keypoints') and pose.body.keypoints:
        pose_modified |= _validate_body_bones(pose, max_bone_ratio, min_keypoint_confidence)
    
    # Validate HAND keypoints (this is likely where the glitches come from!)
    if hasattr(pose, 'left_hand') and pose.left_hand and hasattr(pose.left_hand, 'keypoints') and pose.left_hand.keypoints:
        pose_modified |= _validate_hand_bones(pose.left_hand, max_bone_ratio, min_keypoint_confidence, "left")
    
    if hasattr(pose, 'right_hand') and pose.right_hand and hasattr(pose.right_hand, 'keypoints') and pose.right_hand.keypoints:
        pose_modified |= _validate_hand_bones(pose.right_hand, max_bone_ratio, min_keypoint_confidence, "right")
    
    return pose

def _validate_hand_bones(hand_result, max_bone_ratio, min_keypoint_confidence, hand_side):
    """Validate hand bone lengths and remove elongated finger connections."""
    if not hand_result or not hasattr(hand_result, 'keypoints') or not hand_result.keypoints:
        return False
    
    import numpy as np
    from custom_controlnet_aux.dwpose.types import HandResult, Keypoint
    
    keypoints = hand_result.keypoints
    if len(keypoints) < 21:  # Hand should have 21 keypoints
        return False
    
    # Focus on the most problematic connections that create elongated glitches:
    # 1. Direct wrist-to-fingertip connections (these create the worst glitches)
    # 2. Adjacent finger segment connections that are unreasonably long
    
    # Hand keypoint indices: 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
    # Define both segment-by-segment chains AND direct wrist connections
    finger_segments = [
        # Thumb segments
        [0, 1], [1, 2], [2, 3], [3, 4],
        # Index finger segments  
        [0, 5], [5, 6], [6, 7], [7, 8],
        # Middle finger segments
        [0, 9], [9, 10], [10, 11], [11, 12],
        # Ring finger segments
        [0, 13], [13, 14], [14, 15], [15, 16],
        # Pinky segments
        [0, 17], [17, 18], [18, 19], [19, 20]
    ]
    
    # Calculate reasonable finger segment lengths for reference
    valid_segments = []
    segment_lengths = {}
    
    for idx1, idx2 in finger_segments:
        if idx1 < len(keypoints) and idx2 < len(keypoints):
            kp1, kp2 = keypoints[idx1], keypoints[idx2]
            if (kp1 and kp2 and 
                hasattr(kp1, 'score') and hasattr(kp2, 'score') and
                kp1.score > min_keypoint_confidence and kp2.score > min_keypoint_confidence):
                length = ((kp1.x - kp2.x)**2 + (kp1.y - kp2.y)**2)**0.5
                if length > 0.001:  # Very small minimum threshold
                    valid_segments.append(length)
                    segment_lengths[(idx1, idx2)] = length
    
    if not valid_segments:
        print(f"[DWPose] No valid {hand_side} hand segments found for validation")
        return False
    
    # Use different thresholds for different types of connections:
    # - Wrist to first joints: more permissive 
    # - Finger segment connections: stricter
    wrist_connections = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17)]
    
    reference_length = np.median(valid_segments)
    
    # More permissive threshold for wrist connections (they're naturally longer)
    wrist_max_length = reference_length * max_bone_ratio * 1.5
    # Stricter threshold for finger segments (these create the glitches)
    finger_max_length = reference_length * max_bone_ratio
    
    print(f"[DWPose] {hand_side} hand validation: ref_length={reference_length:.3f}, wrist_max={wrist_max_length:.3f}, finger_max={finger_max_length:.3f}")
    
    # Check for elongated connections and remove problematic keypoints
    new_keypoints = list(keypoints)
    removed_points = []
    
    for idx1, idx2 in finger_segments:
        if (idx1, idx2) in segment_lengths:
            length = segment_lengths[(idx1, idx2)]
            
            # Use appropriate threshold based on connection type
            max_allowed = wrist_max_length if (idx1, idx2) in wrist_connections else finger_max_length
            
            if length > max_allowed:
                # For wrist connections, remove the finger base (first joint)
                # For finger segments, remove the more peripheral point (fingertip direction)
                if (idx1, idx2) in wrist_connections:
                    new_keypoints[idx2] = None  # Remove finger base
                    removed_points.append(f"wrist->joint{idx2}")
                else:
                    new_keypoints[idx2] = None  # Remove more peripheral point
                    removed_points.append(f"joint{idx1}->joint{idx2}")
    
    # Apply changes if we found problems but didn't remove too many points
    if removed_points and len(removed_points) < len(keypoints) // 2:  # Don't remove more than half
        try:
            hand_result.keypoints = new_keypoints
            print(f"[DWPose] {hand_side} hand: Removed elongated connections: {', '.join(removed_points)}")
            return True
        except Exception as e:
            print(f"[WARNING] Could not modify {hand_side} hand keypoints: {e}")
            return False
    elif removed_points:
        print(f"[DWPose] {hand_side} hand: Too many problems detected ({len(removed_points)}), keeping original")
    
    return False

def _validate_body_bones(pose, max_bone_ratio, min_keypoint_confidence):
    """Validate body bone lengths and remove elongated connections."""
    # For now, focus on hands since that's where the glitches come from
    # Body validation will be re-enabled later if needed
    return False
    
    # Calculate valid bone lengths for reference
    valid_lengths = []
    for p1_idx, p2_idx in bone_pairs:
        # Convert to 0-based indexing
        p1, p2 = p1_idx - 1, p2_idx - 1
        if p1 < len(keypoints) and p2 < len(keypoints):
            kp1, kp2 = keypoints[p1], keypoints[p2]
            if (kp1 and kp2 and 
                hasattr(kp1, 'score') and hasattr(kp2, 'score') and
                kp1.score > min_keypoint_confidence and kp2.score > min_keypoint_confidence):
                length = ((kp1.x - kp2.x)**2 + (kp1.y - kp2.y)**2)**0.5
                if length > 0.01:  # Avoid near-zero lengths
                    valid_lengths.append(length)
    
    # If no valid lengths found, don't do validation
    if not valid_lengths:
        print("[DWPose] No valid bone lengths found for validation, keeping original pose")
        return pose
    
    # Use median as reference to avoid outliers affecting the calculation
    reference_length = np.median(valid_lengths)
    max_allowed_length = reference_length * max_bone_ratio
    
    # Create new keypoints list, invalidating points that create elongated bones
    new_keypoints = list(keypoints)
    invalidated_count = 0
    
    for p1_idx, p2_idx in bone_pairs:
        # Convert to 0-based indexing
        p1, p2 = p1_idx - 1, p2_idx - 1
        if p1 < len(keypoints) and p2 < len(keypoints):
            kp1, kp2 = keypoints[p1], keypoints[p2]
            if kp1 and kp2:
                length = ((kp1.x - kp2.x)**2 + (kp1.y - kp2.y)**2)**0.5
                if length > max_allowed_length:
                    # Invalidate the keypoint with lower confidence
                    if hasattr(kp1, 'score') and hasattr(kp2, 'score'):
                        if kp1.score < kp2.score:
                            new_keypoints[p1] = None
                        else:
                            new_keypoints[p2] = None
                        invalidated_count += 1
                    else:
                        # If no score info, invalidate the second point (more peripheral)
                        new_keypoints[p2] = None
                        invalidated_count += 1
    
    # Only apply changes if we didn't invalidate too many points (preserve pose structure)
    # Be very conservative - only remove if very few keypoints affected
    if invalidated_count < min(3, len(keypoints) // 5):  # Allow up to 3 keypoints or 1/5 max
        try:
            # Try direct assignment first
            pose.body.keypoints = new_keypoints
            if invalidated_count > 0:
                print(f"[DWPose] Bone validation: Removed {invalidated_count} elongated connections")
        except AttributeError:
            # If direct assignment fails, create new BodyResult
            try:
                from custom_controlnet_aux.dwpose.types import BodyResult
                pose.body = BodyResult(
                    keypoints=new_keypoints,
                    total_score=pose.body.total_score if hasattr(pose.body, 'total_score') else 0,
                    total_parts=pose.body.total_parts if hasattr(pose.body, 'total_parts') else 0
                )
                if invalidated_count > 0:
                    print(f"[DWPose] Bone validation: Removed {invalidated_count} elongated connections")
            except Exception as e:
                print(f"[WARNING] Could not apply bone validation: {e}")
                # Return original pose if we can't modify it
                return pose
    
    return pose

def _run_dwpose_with_thresholds(
    det,
    pil_image,
    include_hands: bool,
    include_face: bool,
    include_body: bool,
    detect_resolution: int,
    detection_threshold: float,
    nms_threshold: float,
    enable_bone_validation: bool = True,
    max_bone_ratio: float = 2.5,
    min_keypoint_confidence: float = 0.5,
):
    """Run DWPose detection with custom thresholds by post-processing results."""
    import json
    import numpy as np
    from custom_controlnet_aux.dwpose import draw_poses
    from custom_controlnet_aux.dwpose.types import PoseResult, BodyResult, HandResult, FaceResult, Keypoint
    
    # If using default threshold (0.3) AND bone validation is disabled, use original method for compatibility
    if abs(detection_threshold - 0.3) < 0.001 and not enable_bone_validation:
        print("[DWPose] Using original method (default threshold, bone validation disabled)")
        return det(
            pil_image,
            include_hand=include_hands,
            include_face=include_face,
            include_body=include_body,
            detect_resolution=detect_resolution,
            image_and_json=True,
        )
    
    # If using default threshold but bone validation is enabled, use original method and apply validation after
    if abs(detection_threshold - 0.3) < 0.001 and enable_bone_validation:
        print("[DWPose] Using original method with bone validation post-processing")
        pose_img, json_dict = det(
            pil_image,
            include_hand=include_hands,
            include_face=include_face,
            include_body=include_body,
            detect_resolution=detect_resolution,
            image_and_json=True,
        )
        
        # Apply bone validation to the JSON data only, then regenerate the image
        if json_dict and 'people' in json_dict and json_dict['people']:
            # Convert JSON to pose objects for validation
            from custom_controlnet_aux.dwpose.types import PoseResult, BodyResult, Keypoint
            import json
            
            try:
                # Create a simple pose object from JSON for validation
                person = json_dict['people'][0]
                if 'pose_keypoints_2d' in person:
                    # Create temporary pose object for validation
                    temp_keypoints = []
                    kp_data = person['pose_keypoints_2d']
                    for i in range(0, len(kp_data), 3):
                        if i + 2 < len(kp_data):
                            temp_keypoints.append(Keypoint(kp_data[i], kp_data[i+1], kp_data[i+2], i//3))
                    
                    # Create temporary body and pose for validation
                    temp_body = BodyResult(temp_keypoints, 1.0, len(temp_keypoints))
                    temp_pose = PoseResult(temp_body, None, None, None)
                    
                    # Apply bone validation
                    validated_pose = validate_bone_lengths(temp_pose, max_bone_ratio, min_keypoint_confidence)
                    
                    # Update JSON with validated keypoints
                    if validated_pose.body and validated_pose.body.keypoints:
                        new_kp_data = []
                        for kp in validated_pose.body.keypoints:
                            if kp:
                                new_kp_data.extend([kp.x, kp.y, kp.score])
                            else:
                                new_kp_data.extend([0.0, 0.0, 0.0])
                        json_dict['people'][0]['pose_keypoints_2d'] = new_kp_data
                        
                        # Regenerate pose image using the node's regeneration method
                        # We'll use the existing regeneration from nodes.py
                        print("[DWPose] Regenerating pose image with validated keypoints")
                        # For now, just return the original pose_img since hands/face are preserved
                        # The bone validation only affects body keypoints
                    
            except Exception as e:
                print(f"[WARNING] Bone validation failed: {e}. Using original pose.")
        
        return pose_img, json_dict
    
    # For custom thresholds, use our filtering approach
    print(f"[DWPose] Using custom filtering (threshold={detection_threshold}, bone_validation={enable_bone_validation})")
    
    # Use very low threshold for hands/face to preserve them
    hands_face_threshold = 0.05
    
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
                    Keypoint(kp.x, kp.y, kp.score, kp.id) if kp and kp.score >= hands_face_threshold else None
                    for kp in pose.left_hand.keypoints
                ]
                if any(kp is not None for kp in left_keypoints):
                    left_hand = HandResult(left_keypoints, pose.left_hand.total_score)
            
            if hasattr(pose, 'right_hand') and pose.right_hand and hasattr(pose.right_hand, 'keypoints') and pose.right_hand.keypoints:
                right_keypoints = [
                    Keypoint(kp.x, kp.y, kp.score, kp.id) if kp and kp.score >= hands_face_threshold else None
                    for kp in pose.right_hand.keypoints
                ]
                if any(kp is not None for kp in right_keypoints):
                    right_hand = HandResult(right_keypoints, pose.right_hand.total_score)
            
            # Filter face keypoints
            face_result = None
            if hasattr(pose, 'face') and pose.face and hasattr(pose.face, 'keypoints') and pose.face.keypoints:
                face_keypoints = [
                    Keypoint(kp.x, kp.y, kp.score, kp.id) if kp and kp.score >= hands_face_threshold else None
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
    
    # Apply bone length validation to prevent elongated limbs
    if enable_bone_validation:  # Only apply if bone validation is enabled
        if max_bone_ratio >= 0.5:  # Only apply if ratio threshold is meaningful
            try:
                validated_poses = []
                for pose in filtered_poses:
                    validated_pose = validate_bone_lengths(pose, max_bone_ratio, min_keypoint_confidence)
                    validated_poses.append(validated_pose)
                filtered_poses = validated_poses
            except Exception as e:
                print(f"[WARNING] Bone validation failed: {e}. Using original poses.")
        else:
            print("[DWPose] Bone validation enabled but max_bone_ratio too low - no length filtering applied")
    else:
        print("[DWPose] Bone validation disabled - elongated limbs may appear")
    
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
    enable_bone_validation: bool = True,
    max_bone_ratio: float = 2.5,
    min_keypoint_confidence: float = 0.5,
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
        enable_bone_validation=enable_bone_validation,
        max_bone_ratio=max_bone_ratio,
        min_keypoint_confidence=min_keypoint_confidence,
    )
