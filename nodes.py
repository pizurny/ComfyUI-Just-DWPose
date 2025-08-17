from typing import Tuple
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
import gc
import os
from .loader import run_dwpose_once, get_models_dir
from .enhanced_loader import run_enhanced_dwpose

# Import ComfyUI folder management if available
try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False

try:
    from .dwpose_kalman_filter import DWPoseKalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False

# Import pose drawing utilities for regenerating filtered poses
import sys
_vendor_path = str(Path(__file__).parent / "vendor")
if _vendor_path not in sys.path:
    sys.path.insert(0, _vendor_path)

try:
    from custom_controlnet_aux.dwpose import draw_poses
    from custom_controlnet_aux.dwpose.types import PoseResult, BodyResult, Keypoint
    from custom_controlnet_aux.util import HWC3
    POSE_DRAWING_AVAILABLE = True
except ImportError:
    POSE_DRAWING_AVAILABLE = False

def _discover_available_models():
    """Dynamically discover available DWPose models in the models directory."""
    try:
        models_dir = get_models_dir()
        if not models_dir or not models_dir.exists():
            # Fallback to hardcoded options if models dir not found
            return {
                "bbox_detectors": ["auto", "yolox_l.torchscript.pt", "yolox_l.onnx"],
                "pose_estimators": ["auto", "dw-ll_ucoco_384_bs5.torchscript.pt", "dw-ll_ucoco_384.onnx"]
            }
        
        bbox_detectors = ["auto"]
        pose_estimators = ["auto"]
        
        # Scan for detection models
        for pattern in ["yolox*.torchscript.pt", "yolox*.onnx"]:
            for file_path in models_dir.glob(pattern):
                if file_path.is_file():
                    bbox_detectors.append(file_path.name)
        
        # Scan for pose estimation models  
        for pattern in ["dw-*.torchscript.pt", "dw-*.onnx", "dwpose*.torchscript.pt", "dwpose*.onnx"]:
            for file_path in models_dir.glob(pattern):
                if file_path.is_file():
                    pose_estimators.append(file_path.name)
        
        # Remove duplicates and sort (keeping "auto" first)
        bbox_detectors = ["auto"] + sorted(list(set(bbox_detectors[1:])))
        pose_estimators = ["auto"] + sorted(list(set(pose_estimators[1:])))
        
        return {
            "bbox_detectors": bbox_detectors,
            "pose_estimators": pose_estimators
        }
        
    except Exception as e:
        print(f"[DWPose] Error discovering models: {e}, using defaults")
        # Fallback to hardcoded options
        return {
            "bbox_detectors": ["auto", "yolox_l.torchscript.pt", "yolox_l.onnx"],
            "pose_estimators": ["auto", "dw-ll_ucoco_384_bs5.torchscript.pt", "dw-ll_ucoco_384.onnx"]
        }

class DWPoseAnnotator:
    CATEGORY = "annotators/dwpose"
    
    def __init__(self):
        self.kalman_filter = None

    @classmethod
    def INPUT_TYPES(cls):
        # Dynamically discover available models
        available_models = _discover_available_models()
        
        return {
            "required": {
                "image": ("IMAGE",),
                "backend": (["auto", "torchscript", "onnx"], {"default": "auto"}),
                "bbox_detector": (available_models["bbox_detectors"], {"default": "auto"}),
                "pose_estimator": (available_models["pose_estimators"], {"default": "auto"}),
                "detect_resolution": ("INT", {"default": 768, "min": 128, "max": 2048, "step": 32}),
                "include_body": ("BOOLEAN", {"default": True}),
                "include_hands": ("BOOLEAN", {"default": True}),
                "include_face": ("BOOLEAN", {"default": True}),
                "offline_ok": ("BOOLEAN", {"default": True}),
                "allow_download": ("BOOLEAN", {"default": False}),
                "detection_threshold": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 0.9, "step": 0.05}),
                "nms_threshold": ("FLOAT", {"default": 0.45, "min": 0.1, "max": 0.9, "step": 0.05}),
                "use_kalman": ("BOOLEAN", {"default": False}),
                "kalman_process_noise": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 1.0, "step": 0.001}),
                "kalman_measurement_noise": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "kalman_confidence_threshold": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 1.0, "step": 0.05}),
                # Bone validation parameters (prevents elongated/glitchy limbs)
                "enable_bone_validation": ("BOOLEAN", {"default": True}),
                "max_bone_ratio": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 10.0, "step": 0.1}),
                "min_keypoint_confidence": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05}),
                # Multi-scale detection parameters
                "enable_multiscale": ("BOOLEAN", {"default": False}),
                "multiscale_scales": ("STRING", {"default": "0.5,0.75,1.0,1.25,1.5"}),
                "multiscale_fusion": (["weighted_average", "max_confidence", "voting"], {"default": "weighted_average"}),
                # Multi-model ensemble parameters
                "enable_multimodel": ("BOOLEAN", {"default": False}),
                "multimodel_backends": ("STRING", {"default": "torchscript,onnx"}),
                "multimodel_fusion": (["weighted_average", "voting", "best_confidence"], {"default": "weighted_average"}),
            },
            "optional": {
                "model_dir_override": ("STRING", {"default": ""}),
                # Multi-person selection
                "person_index": ("STRING", {"default": "0"}),
                "process_all_persons": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("pose_image", "keypoints_json", "proof")
    FUNCTION = "run"

    def _tensor_to_pil(self, t: torch.Tensor) -> Image.Image:
        arr = (t.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr, "RGB")

    def _pil_to_tensor(self, img: Image.Image, target_size: tuple = None) -> torch.Tensor:
        # Resize image to target size if specified
        if target_size is not None:
            img = img.resize((target_size[1], target_size[0]), Image.LANCZOS)  # PIL uses (width, height)
        arr = np.asarray(img, dtype=np.uint8).astype(np.float32) / 255.0
        return torch.from_numpy(arr)

    def _create_proof_overlay(self, original_pil: Image.Image, pose_pil: Image.Image, blend_alpha: float = 0.7) -> Image.Image:
        """Create proof image by overlaying skeleton on original input frame."""
        try:
            # Ensure both images are the same size
            if original_pil.size != pose_pil.size:
                pose_pil = pose_pil.resize(original_pil.size, Image.LANCZOS)
            
            # Convert pose image to RGBA to use as overlay mask
            pose_rgba = pose_pil.convert("RGBA")
            
            # Create a mask based on the non-black pixels in the pose image
            # Black pixels (0,0,0) will be transparent, colored pixels will be visible
            pose_array = np.array(pose_rgba)
            
            # Create alpha channel: transparent where pose is black, opaque where pose has skeleton
            mask = (pose_array[:, :, 0] > 10) | (pose_array[:, :, 1] > 10) | (pose_array[:, :, 2] > 10)
            pose_array[:, :, 3] = mask.astype(np.uint8) * int(255 * blend_alpha)
            
            # Create the overlay image
            pose_overlay = Image.fromarray(pose_array, "RGBA")
            
            # Composite the skeleton over the original image
            original_rgba = original_pil.convert("RGBA")
            proof_image = Image.alpha_composite(original_rgba, pose_overlay)
            
            # Convert back to RGB
            return proof_image.convert("RGB")
            
        except Exception as e:
            print(f"[WARNING] Failed to create proof overlay: {e}. Using original image.")
            return original_pil

    def _convert_pose_to_kalman_format(self, pose_dict, person_index=0):
        """Convert pose dictionary to numpy format for Kalman filtering."""
        if 'people' not in pose_dict or not pose_dict['people']:
            return None
            
        # Check if requested person index exists
        if person_index >= len(pose_dict['people']):
            print(f"[WARNING] person_index {person_index} >= number of detected people ({len(pose_dict['people'])}), using person 0")
            person_index = 0
            
        person = pose_dict['people'][person_index]
        
        # Combine all keypoints: body + hands + face
        all_keypoints = []
        
        # Body keypoints (18 joints)
        if 'pose_keypoints_2d' in person:
            body_kp = person['pose_keypoints_2d']
            for i in range(0, len(body_kp), 3):
                if i + 2 < len(body_kp):
                    all_keypoints.extend([body_kp[i], body_kp[i+1], body_kp[i+2]])
        
        # Left hand keypoints (21 joints)
        if 'hand_left_keypoints_2d' in person:
            hand_kp = person['hand_left_keypoints_2d']
            for i in range(0, len(hand_kp), 3):
                if i + 2 < len(hand_kp):
                    all_keypoints.extend([hand_kp[i], hand_kp[i+1], hand_kp[i+2]])
        else:
            # Add 21 empty hand keypoints if not present
            all_keypoints.extend([0.0, 0.0, 0.0] * 21)
            
        # Right hand keypoints (21 joints)  
        if 'hand_right_keypoints_2d' in person:
            hand_kp = person['hand_right_keypoints_2d']
            for i in range(0, len(hand_kp), 3):
                if i + 2 < len(hand_kp):
                    all_keypoints.extend([hand_kp[i], hand_kp[i+1], hand_kp[i+2]])
        else:
            # Add 21 empty hand keypoints if not present
            all_keypoints.extend([0.0, 0.0, 0.0] * 21)
            
        # Face keypoints (70 joints)
        if 'face_keypoints_2d' in person:
            face_kp = person['face_keypoints_2d']
            for i in range(0, len(face_kp), 3):
                if i + 2 < len(face_kp):
                    all_keypoints.extend([face_kp[i], face_kp[i+1], face_kp[i+2]])
        else:
            # Add 70 empty face keypoints if not present
            all_keypoints.extend([0.0, 0.0, 0.0] * 70)
        
        if not all_keypoints:
            return None
            
        # Convert to (N, 3) array - total: 18 + 21 + 21 + 70 = 130 joints
        num_joints = len(all_keypoints) // 3
        pose_array = np.array(all_keypoints).reshape(num_joints, 3)
        return pose_array
    
    def _convert_kalman_to_pose_format(self, pose_array, original_dict, person_index=0):
        """Convert filtered numpy pose back to original dictionary format."""
        if pose_array is None:
            return original_dict
            
        # Update the pose keypoints in the dictionary
        filtered_dict = json.loads(json.dumps(original_dict))  # Deep copy
        if 'people' in filtered_dict and filtered_dict['people']:
            # Ensure person_index is valid
            if person_index >= len(filtered_dict['people']):
                person_index = 0
            # Split the filtered array back into body, hands, and face
            flattened = pose_array.flatten().tolist()
            
            # Body keypoints: first 18 joints (54 values)
            body_end = 18 * 3
            filtered_dict['people'][person_index]['pose_keypoints_2d'] = flattened[:body_end]
            
            # Left hand keypoints: next 21 joints (63 values)
            left_hand_start = body_end
            left_hand_end = left_hand_start + 21 * 3
            left_hand_kp = flattened[left_hand_start:left_hand_end]
            # Only add if not all zeros (meaning hand was detected)
            if any(abs(x) > 0.001 for x in left_hand_kp[2::3]):  # Check confidence values
                filtered_dict['people'][person_index]['hand_left_keypoints_2d'] = left_hand_kp
                
            # Right hand keypoints: next 21 joints (63 values)  
            right_hand_start = left_hand_end
            right_hand_end = right_hand_start + 21 * 3
            right_hand_kp = flattened[right_hand_start:right_hand_end]
            # Only add if not all zeros (meaning hand was detected)
            if any(abs(x) > 0.001 for x in right_hand_kp[2::3]):  # Check confidence values
                filtered_dict['people'][person_index]['hand_right_keypoints_2d'] = right_hand_kp
                
            # Face keypoints: remaining 70 joints (210 values)
            face_start = right_hand_end
            face_kp = flattened[face_start:]
            # Only add if not all zeros (meaning face was detected)
            if any(abs(x) > 0.001 for x in face_kp[2::3]):  # Check confidence values
                filtered_dict['people'][person_index]['face_keypoints_2d'] = face_kp
            
        return filtered_dict

    def _regenerate_pose_image(self, filtered_keypoints_dict, original_image_pil, detect_resolution):
        """Regenerate pose overlay image from filtered keypoints."""
        if not POSE_DRAWING_AVAILABLE:
            print("[WARNING] Pose drawing utilities not available, using original image")
            return original_image_pil
            
            
        try:
            # Convert filtered keypoints to PoseResult format
            pose_results = []
            
            if 'people' in filtered_keypoints_dict and filtered_keypoints_dict['people']:
                for person in filtered_keypoints_dict['people']:
                    # Process body keypoints
                    body_result = None
                    if 'pose_keypoints_2d' in person:
                        keypoints_2d = person['pose_keypoints_2d']
                        body_keypoints = []
                        for i in range(0, len(keypoints_2d), 3):
                            if i + 2 < len(keypoints_2d):
                                x, y, conf = keypoints_2d[i], keypoints_2d[i+1], keypoints_2d[i+2]
                                if conf > 0.1:
                                    body_keypoints.append(Keypoint(x, y, conf, len(body_keypoints)))
                                else:
                                    body_keypoints.append(None)
                        body_result = BodyResult(body_keypoints, 1.0, len(body_keypoints))
                    
                    # Process left hand keypoints
                    left_hand = None
                    if 'hand_left_keypoints_2d' in person:
                        hand_kp = person['hand_left_keypoints_2d']
                        hand_keypoints = []
                        for i in range(0, len(hand_kp), 3):
                            if i + 2 < len(hand_kp):
                                x, y, conf = hand_kp[i], hand_kp[i+1], hand_kp[i+2]
                                if conf > 0.1:
                                    hand_keypoints.append(Keypoint(x, y, conf, len(hand_keypoints)))
                                else:
                                    hand_keypoints.append(None)
                        if any(kp is not None for kp in hand_keypoints):
                            left_hand = hand_keypoints  # HandResult is just List[Keypoint]
                    
                    # Process right hand keypoints
                    right_hand = None
                    if 'hand_right_keypoints_2d' in person:
                        hand_kp = person['hand_right_keypoints_2d']
                        hand_keypoints = []
                        for i in range(0, len(hand_kp), 3):
                            if i + 2 < len(hand_kp):
                                x, y, conf = hand_kp[i], hand_kp[i+1], hand_kp[i+2]
                                if conf > 0.1:
                                    hand_keypoints.append(Keypoint(x, y, conf, len(hand_keypoints)))
                                else:
                                    hand_keypoints.append(None)
                        if any(kp is not None for kp in hand_keypoints):
                            right_hand = hand_keypoints  # HandResult is just List[Keypoint]
                    
                    # Process face keypoints
                    face_result = None
                    if 'face_keypoints_2d' in person:
                        face_kp = person['face_keypoints_2d']
                        face_keypoints = []
                        for i in range(0, len(face_kp), 3):
                            if i + 2 < len(face_kp):
                                x, y, conf = face_kp[i], face_kp[i+1], face_kp[i+2]
                                if conf > 0.1:
                                    face_keypoints.append(Keypoint(x, y, conf, len(face_keypoints)))
                                else:
                                    face_keypoints.append(None)
                        if any(kp is not None for kp in face_keypoints):
                            face_result = face_keypoints  # FaceResult is just List[Keypoint]
                        
                    # Create complete PoseResult with body, hands, and face
                    pose_result = PoseResult(body_result, left_hand, right_hand, face_result)
                    pose_results.append(pose_result)
            
            if not pose_results:
                return original_image_pil
                
                
            # Get original image dimensions
            img_array = np.array(original_image_pil)
            height, width = img_array.shape[:2]
            
            # Draw poses on black canvas
            canvas = draw_poses(pose_results, height, width, 
                              draw_body=True, draw_hand=True, draw_face=True)
            
            # Convert back to PIL Image
            canvas = HWC3(canvas)
            return Image.fromarray(canvas, 'RGB')
            
        except Exception as e:
            print(f"[WARNING] Failed to regenerate pose image: {e}")
            import traceback
            traceback.print_exc()
            return original_image_pil

    def run(
        self,
        image: torch.Tensor,
        backend: str,
        bbox_detector: str,
        pose_estimator: str,
        detect_resolution: int,
        include_body: bool,
        include_hands: bool,
        include_face: bool,
        offline_ok: bool,
        allow_download: bool,
        detection_threshold: float,
        nms_threshold: float,
        use_kalman: bool,
        kalman_process_noise: float,
        kalman_measurement_noise: float,
        kalman_confidence_threshold: float,
        enable_bone_validation: bool,
        max_bone_ratio: float,
        min_keypoint_confidence: float,
        # Enhanced parameters
        enable_multiscale: bool,
        multiscale_scales: str,
        multiscale_fusion: str,
        enable_multimodel: bool,
        multimodel_backends: str,
        multimodel_fusion: str,
        model_dir_override: str = "",
        person_index: str = "0", 
        process_all_persons: bool = False,
    ) -> Tuple[torch.Tensor, str, torch.Tensor]:
        
        try:
            # Input validation
            if image is None or not hasattr(image, 'shape'):
                raise ValueError("Invalid input image tensor")
            
            print(f"[Just-DWPose] Input tensor shape: {image.shape}, dtype: {image.dtype}")
            
            # Device management - determine device consistently
            self.device = 'cuda' if torch.cuda.is_available() and backend != 'onnx' else 'cpu'
            
            # Ensure input tensor is on correct device
            if image.is_cuda and self.device == 'cpu':
                image = image.cpu()
                print(f"[Just-DWPose] Moved tensor from CUDA to CPU")
            elif not image.is_cuda and self.device == 'cuda':
                image = image.cuda()
                print(f"[Just-DWPose] Moved tensor from CPU to CUDA")
            
            # Check for reasonable batch size
            batch_size = image.shape[0]
            if batch_size > 500:
                raise ValueError(f"Batch size {batch_size} is too large. Maximum recommended: 500")
            
            # Comprehensive parameter validation
            if not isinstance(image, torch.Tensor):
                raise TypeError(f"image must be torch.Tensor, got {type(image)}")
            
            if image.dim() != 4:
                raise ValueError(f"image must be 4D tensor [B,H,W,C], got shape {image.shape}")
            
            if not (128 <= detect_resolution <= 2048):
                raise ValueError(f"detect_resolution must be in [128, 2048], got {detect_resolution}")
            
            if not (0.05 <= detection_threshold <= 0.9):
                raise ValueError(f"detection_threshold must be in [0.05, 0.9], got {detection_threshold}")
            
            if not (0.1 <= nms_threshold <= 0.9):
                raise ValueError(f"nms_threshold must be in [0.1, 0.9], got {nms_threshold}")
            
            # Handle person selection mode
            if process_all_persons:
                print("[Just-DWPose] Process all persons mode enabled")
                person_index = None  # Special value indicating all persons
            else:
                # Validate and fix person_index for single person mode
                try:
                    person_index = int(person_index) if isinstance(person_index, str) else person_index
                    if person_index < 0:
                        person_index = 0
                except (ValueError, TypeError):
                    print(f"[WARNING] Invalid person_index '{person_index}', using 0")
                    person_index = 0
            
            # Validate bone validation parameters
            if max_bone_ratio <= 0:
                max_bone_ratio = 2.5
                print(f"[WARNING] Invalid max_bone_ratio, using default: {max_bone_ratio}")
            if min_keypoint_confidence <= 0 or min_keypoint_confidence > 1.0:
                min_keypoint_confidence = 0.5
                print(f"[WARNING] Invalid min_keypoint_confidence, using default: {min_keypoint_confidence}")
            
            # Parse multi-scale parameters
            if enable_multiscale:
                try:
                    scales = [float(s.strip()) for s in multiscale_scales.split(',') if s.strip()]
                    print(f"[Enhanced] Multi-scale enabled with scales: {scales}")
                except:
                    scales = [0.5, 0.75, 1.0, 1.25, 1.5]
                    print(f"[Enhanced] Using default scales: {scales}")
            else:
                scales = None
            
            # Parse multi-model configurations
            multimodel_configs = None
            if enable_multimodel:
                backends = [b.strip() for b in multimodel_backends.split(',') if b.strip()]
                multimodel_configs = []
                for idx, b in enumerate(backends):
                    multimodel_configs.append({
                        'name': f'{b}_model_{idx}',
                        'backend': b,
                        'models_dir': model_dir_override if model_dir_override else str(get_models_dir())
                    })
                print(f"[Enhanced] Multi-model enabled with {len(multimodel_configs)} models")
            
            # person_index validation already done above
            
            # Ensure models_dir is always a Path object
            try:
                if model_dir_override and model_dir_override.strip():
                    models_dir = Path(model_dir_override).expanduser().resolve()
                else:
                    models_dir = get_models_dir()
                
                # Better validation and fallback
                if not models_dir.exists() or "True" in str(models_dir):
                    print(f"[ERROR] Invalid models_dir detected: {models_dir}")
                    # Use ComfyUI's folder system
                    if FOLDER_PATHS_AVAILABLE:
                        try:
                            base_path = folder_paths.get_folder_paths("checkpoints")[0]
                            models_dir = Path(base_path) / "DWPose"
                            models_dir.mkdir(parents=True, exist_ok=True)
                            print(f"[DEBUG] Using ComfyUI folder system: {models_dir}")
                        except Exception as e:
                            print(f"[WARNING] ComfyUI folder system failed: {e}")
                            # Last resort - relative to current file
                            models_dir = Path(__file__).parent / "models" / "DWPose"
                            models_dir.mkdir(parents=True, exist_ok=True)
                            print(f"[DEBUG] Using relative path: {models_dir}")
                    else:
                        # Last resort - relative to current file
                        models_dir = Path(__file__).parent / "models" / "DWPose"
                        models_dir.mkdir(parents=True, exist_ok=True)
                        print(f"[DEBUG] Using relative path: {models_dir}")
                
                # Force conversion to ensure it's a Path object
                models_dir = Path(models_dir)
            except Exception as e:
                print(f"[ERROR] Failed to resolve models_dir: {e}")
                # Use relative fallback as last resort
                models_dir = Path(__file__).parent / "models" / "DWPose"
                models_dir.mkdir(parents=True, exist_ok=True)
                print(f"[DEBUG] Emergency fallback models_dir: {models_dir}")

            # Handle batch processing - image tensor shape is [Batch, Height, Width, Channels]
            print(f"[Just-DWPose] Processing batch of {batch_size} images...")
            
            # Determine target size from input images (use first image dimensions)
            input_height, input_width = image.shape[1], image.shape[2]
            target_size = (input_height, input_width)
            print(f"[Just-DWPose] Target output size: {target_size}")
            
            pose_images = []
            proof_images = []
            all_keypoints = []
            
            # Initialize Kalman filter if enabled and filterpy is available
            if use_kalman and KALMAN_AVAILABLE and batch_size > 1:
                # Reset filter for new batch/sequence
                self.kalman_filter = DWPoseKalmanFilter(
                    num_joints=130,  # Total: 18 body + 21 left hand + 21 right hand + 70 face
                    process_noise=kalman_process_noise,
                    measurement_noise=kalman_measurement_noise
                )
            
            # Add memory management for large batches
            BATCH_CHUNK_SIZE = 10  # Process in chunks
            
            for chunk_start in range(0, batch_size, BATCH_CHUNK_SIZE):
                chunk_end = min(chunk_start + BATCH_CHUNK_SIZE, batch_size)
                
                # Process chunk
                for i in range(chunk_start, chunk_end):
                    # Progress reporting for large batches
                    if batch_size > 10 and i % 10 == 0:
                        print(f"[Just-DWPose] Processing frame {i+1}/{batch_size}")
                        
                    try:
                        # Process each image in the batch
                        single_image = image[i]  # Shape: [Height, Width, Channels]
                        original_pil = self._tensor_to_pil(single_image)
                        
                        # Process with no_grad to save memory
                        with torch.no_grad():
                            pose_pil, kp_dict = run_enhanced_dwpose(
                                original_pil,
                                backend=backend,
                                bbox_detector=bbox_detector,
                                pose_estimator=pose_estimator,
                                detect_resolution=int(detect_resolution),
                                include_body=include_body,
                                include_hands=include_hands,
                                include_face=include_face,
                                models_dir=models_dir,
                                offline_ok=offline_ok,
                                allow_download=allow_download,
                                detection_threshold=detection_threshold,
                                nms_threshold=nms_threshold,
                                enable_bone_validation=enable_bone_validation,
                                max_bone_ratio=max_bone_ratio,
                                min_keypoint_confidence=min_keypoint_confidence,
                                person_index=person_index,
                                # Enhanced parameters
                                enable_multiscale=enable_multiscale,
                                multiscale_scales=scales,
                                multiscale_fusion_method=multiscale_fusion,
                                enable_multimodel=enable_multimodel,
                                multimodel_configs=multimodel_configs,
                                multimodel_fusion_method=multimodel_fusion,
                            )
                        
                            # Apply Kalman filtering if enabled and available (only for single person mode)
                            if use_kalman and KALMAN_AVAILABLE and self.kalman_filter is not None and batch_size > 1 and person_index is not None:
                                try:
                                    # Convert to Kalman format and filter
                                    pose_array = self._convert_pose_to_kalman_format(kp_dict, person_index)
                                    if pose_array is not None:
                                        filtered_pose_array = self.kalman_filter.update(
                                            pose_array, 
                                            confidence_threshold=kalman_confidence_threshold
                                        )
                                        # Convert back to dictionary format
                                        kp_dict = self._convert_kalman_to_pose_format(filtered_pose_array, kp_dict, person_index)
                                        
                                        # Regenerate pose image from filtered keypoints
                                        original_image_pil = self._tensor_to_pil(single_image)
                                        filtered_pose_pil = self._regenerate_pose_image(
                                            kp_dict, original_image_pil, detect_resolution
                                        )
                                        # Only use filtered image if it's different from original (meaning regeneration worked)
                                        if filtered_pose_pil != original_image_pil:
                                            pose_pil = filtered_pose_pil
                                        # else: keep the original pose_pil from dwpose detection
                                        
                                except Exception as e:
                                    print(f"[WARNING] Kalman filtering failed: {e}. Using original pose.")
                            elif use_kalman and not KALMAN_AVAILABLE:
                                print("[WARNING] Kalman filtering requested but filterpy not available. Install with: pip install filterpy")
                        
                            # Create proof image (original + skeleton overlay)
                            proof_pil = self._create_proof_overlay(original_pil, pose_pil, blend_alpha=0.6)
                                
                            pose_images.append(self._pil_to_tensor(pose_pil, target_size))
                            proof_images.append(self._pil_to_tensor(proof_pil, target_size))
                            all_keypoints.append(kp_dict)
                        
                    except Exception as e:
                        print(f"[ERROR] Failed to process frame {i+1}/{batch_size}: {e}")
                        # Add fallback image
                        fallback_img = Image.new('RGB', (target_size[1], target_size[0]), (0, 0, 0))
                        pose_images.append(self._pil_to_tensor(fallback_img))
                        proof_images.append(self._pil_to_tensor(fallback_img))
                        all_keypoints.append({"version": "ap10k", "people": []})
                
                # Clear GPU cache after each chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Force garbage collection for large batches
                if batch_size > 50 and chunk_end % 50 == 0:
                    gc.collect()
            
            # Final memory cleanup for large batches
            if batch_size > 50:
                gc.collect()
                
            print(f"[Just-DWPose] Batch processing complete: {batch_size} frames")
            
            # Stack all processed images back into batch format [Batch, Height, Width, Channels]
            batch_pose_tensor = torch.stack(pose_images, dim=0)
            batch_proof_tensor = torch.stack(proof_images, dim=0)
            
            # Combine all keypoints into a single JSON string
            combined_keypoints = json.dumps(all_keypoints, ensure_ascii=False)
            
            return batch_pose_tensor, combined_keypoints, batch_proof_tensor
            
        except Exception as e:
            print(f"[ERROR] DWPose node execution failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Return fallback result with proper dimensions
            try:
                # Try to get dimensions from input
                if hasattr(image, 'shape') and len(image.shape) >= 3:
                    h, w = image.shape[1], image.shape[2]
                    fallback_img = Image.new('RGB', (w, h), (0, 0, 0))
                else:
                    fallback_img = Image.new('RGB', (768, 768), (0, 0, 0))
                fallback_tensor = self._pil_to_tensor(fallback_img).unsqueeze(0)  # Add batch dimension
            except:
                # Last resort fallback
                fallback_img = Image.new('RGB', (768, 768), (0, 0, 0))
                fallback_tensor = self._pil_to_tensor(fallback_img).unsqueeze(0)
            
            fallback_json = json.dumps([{"version": "ap10k", "people": []}])
            return fallback_tensor, fallback_json, fallback_tensor  # Use same fallback for proof


def json_keypoints_to_pose_images(
    keypoints_json_str: str, 
    width: int, 
    height: int, 
    draw_body: bool = True, 
    draw_hands: bool = True, 
    draw_face: bool = True,
    frame_index: int = -1,  # -1 means process all frames
    point_size: int = 4,
    bone_thickness: int = 4,
    hand_line_thickness: int = 2
) -> list:
    """Convert DWPose/OpenPose JSON keypoints to pose visualization images.
    
    Args:
        keypoints_json_str: JSON string with DWPose/OpenPose keypoint data
        width: Output image width
        height: Output image height  
        draw_body: Whether to draw body skeleton
        draw_hands: Whether to draw hand skeletons
        draw_face: Whether to draw face keypoints
        frame_index: Frame to draw (-1 = all frames, 0+ = specific frame)
        point_size: Size of keypoint circles
        bone_thickness: Thickness of skeleton bones
        hand_line_thickness: Thickness of hand connection lines
        
    Returns:
        List of PIL Images with pose visualizations
    """
    if not POSE_DRAWING_AVAILABLE:
        return [Image.new('RGB', (width, height), (0, 0, 0))]
    
    try:
        # Parse JSON keypoints
        keypoints_data = json.loads(keypoints_json_str)
        print(f"[DWPose Helper] Parsing JSON data: {type(keypoints_data)}")
        
        # Handle different JSON formats:
        # 1. Direct OpenPose format: {"people": [...]}
        # 2. DWPose batch format: [{"people": [...]}, {"people": [...]}]  
        
        frames_to_process = []
        
        if isinstance(keypoints_data, list):
            # Batch format - list of dictionaries
            if keypoints_data and isinstance(keypoints_data[0], dict):
                if frame_index == -1:
                    # Process all frames
                    frames_to_process = keypoints_data
                    print(f"[DWPose Helper] Processing all {len(keypoints_data)} frames")
                elif frame_index < len(keypoints_data):
                    # Process specific frame
                    frames_to_process = [keypoints_data[frame_index]]
                    print(f"[DWPose Helper] Processing frame {frame_index}")
                else:
                    # Fallback to first frame
                    frames_to_process = [keypoints_data[0]]
                    print(f"[DWPose Helper] Frame {frame_index} out of range, using frame 0")
            else:
                # Flat array format - not supported
                print("[DWPose Helper] Unsupported flat array format")
                return [Image.new('RGB', (width, height), (0, 0, 0))]
        else:
            # Single dictionary format
            frames_to_process = [keypoints_data]
            print("[DWPose Helper] Processing single frame")
        
        # Process all frames
        result_images = []
        
        for frame_idx, keypoints_dict in enumerate(frames_to_process):
            # Ensure we have the expected OpenPose format
            if 'people' not in keypoints_dict:
                print(f"[DWPose Helper] Frame {frame_idx}: No 'people' key")
                result_images.append(Image.new('RGB', (width, height), (0, 0, 0)))
                continue
            
            if not keypoints_dict['people'] or len(keypoints_dict['people']) == 0:
                print(f"[DWPose Helper] Frame {frame_idx}: People array is empty")
                result_images.append(Image.new('RGB', (width, height), (0, 0, 0)))
                continue
            
            people = keypoints_dict['people']
            # Always use the first person in the frame
            person = people[0]
            
            # Convert to PoseResult format for this frame
            pose_results = []
            
            # Process body keypoints
            body_result = None
            if 'pose_keypoints_2d' in person and draw_body:
                keypoints_2d = person['pose_keypoints_2d']
                body_keypoints = []
                for i in range(0, len(keypoints_2d), 3):
                    if i + 2 < len(keypoints_2d):
                        x, y, conf = keypoints_2d[i], keypoints_2d[i+1], keypoints_2d[i+2]
                        if conf > 0.1:
                            body_keypoints.append(Keypoint(x, y, conf, len(body_keypoints)))
                        else:
                            body_keypoints.append(None)
                body_result = BodyResult(body_keypoints, 1.0, len(body_keypoints))
            
            # Process hand keypoints
            left_hand = None
            right_hand = None
            if draw_hands:
                # Left hand
                if 'hand_left_keypoints_2d' in person:
                    hand_kp = person['hand_left_keypoints_2d']
                    hand_keypoints = []
                    for i in range(0, len(hand_kp), 3):
                        if i + 2 < len(hand_kp):
                            x, y, conf = hand_kp[i], hand_kp[i+1], hand_kp[i+2]
                            if conf > 0.1:
                                hand_keypoints.append(Keypoint(x, y, conf, len(hand_keypoints)))
                            else:
                                hand_keypoints.append(None)
                    if any(kp is not None for kp in hand_keypoints):
                        left_hand = hand_keypoints
                
                # Right hand
                if 'hand_right_keypoints_2d' in person:
                    hand_kp = person['hand_right_keypoints_2d']
                    hand_keypoints = []
                    for i in range(0, len(hand_kp), 3):
                        if i + 2 < len(hand_kp):
                            x, y, conf = hand_kp[i], hand_kp[i+1], hand_kp[i+2]
                            if conf > 0.1:
                                hand_keypoints.append(Keypoint(x, y, conf, len(hand_keypoints)))
                            else:
                                hand_keypoints.append(None)
                    if any(kp is not None for kp in hand_keypoints):
                        right_hand = hand_keypoints
            
            # Process face keypoints
            face_result = None
            if 'face_keypoints_2d' in person and draw_face:
                face_kp = person['face_keypoints_2d']
                face_keypoints = []
                for i in range(0, len(face_kp), 3):
                    if i + 2 < len(face_kp):
                        x, y, conf = face_kp[i], face_kp[i+1], face_kp[i+2]
                        if conf > 0.1:
                            face_keypoints.append(Keypoint(x, y, conf, len(face_keypoints)))
                        else:
                            face_keypoints.append(None)
                if any(kp is not None for kp in face_keypoints):
                    face_result = face_keypoints
            
            # Create PoseResult for this frame
            pose_result = PoseResult(body_result, left_hand, right_hand, face_result)
            pose_results.append(pose_result)
            
            if not pose_results:
                result_images.append(Image.new('RGB', (width, height), (0, 0, 0)))
                continue
            
            # Generate pose image for this frame with custom drawing parameters
            canvas = draw_poses_with_custom_params(
                pose_results, height, width, 
                draw_body=draw_body, 
                draw_hand=draw_hands, 
                draw_face=draw_face,
                point_size=point_size,
                bone_thickness=bone_thickness,
                hand_line_thickness=hand_line_thickness
            )
            
            # Convert to PIL Image and add to results
            canvas = HWC3(canvas)
            frame_image = Image.fromarray(canvas, 'RGB')
            result_images.append(frame_image)
        
        return result_images
        
    except Exception as e:
        print(f"[WARNING] Failed to convert JSON keypoints to pose images: {e}")
        return [Image.new('RGB', (width, height), (0, 0, 0))]


def draw_poses_with_custom_params(poses, H, W, draw_body=True, draw_hand=True, draw_face=True, 
                                 point_size=4, bone_thickness=4, hand_line_thickness=2):
    """Custom draw_poses function with configurable drawing parameters."""
    if not POSE_DRAWING_AVAILABLE:
        return np.zeros((H, W, 3), dtype=np.uint8)
    
    try:
        # Import drawing utilities
        from custom_controlnet_aux.dwpose import util
        import cv2
        import numpy as np
        
        # Create blank canvas
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Draw each pose
        for pose in poses:
            if draw_body and pose.body:
                canvas = draw_body_with_custom_params(canvas, pose.body.keypoints, bone_thickness, point_size)
            
            if draw_hand:
                if pose.left_hand:
                    canvas = draw_hand_with_custom_params(canvas, pose.left_hand, hand_line_thickness, point_size)
                if pose.right_hand:
                    canvas = draw_hand_with_custom_params(canvas, pose.right_hand, hand_line_thickness, point_size)
            
            if draw_face and pose.face:
                canvas = draw_face_with_custom_params(canvas, pose.face, point_size)
        
        return canvas
        
    except Exception as e:
        print(f"[WARNING] Custom drawing failed: {e}, falling back to default")
        # Fallback to original drawing
        from custom_controlnet_aux.dwpose import draw_poses
        return draw_poses(poses, H, W, draw_body=draw_body, draw_hand=draw_hand, draw_face=draw_face)


def draw_body_with_custom_params(canvas, keypoints, bone_thickness=4, point_size=4):
    """Draw body pose with custom bone thickness and point size."""
    import cv2
    import numpy as np
    import matplotlib.colors
    
    if not keypoints:
        return canvas
    
    H, W, _ = canvas.shape
    
    # Body connections (OpenPose 18-point format)
    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], 
        [6, 7], [7, 8], [2, 9], [9, 10], 
        [10, 11], [2, 12], [12, 13], [13, 14], 
        [2, 1], [1, 15], [15, 17], [1, 16], 
        [16, 18]
    ]
    
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    
    # Draw bones
    for i, (start_idx, end_idx) in enumerate(limbSeq):
        start_idx -= 1  # Convert to 0-based
        end_idx -= 1
        
        if (start_idx < len(keypoints) and end_idx < len(keypoints) and 
            keypoints[start_idx] is not None and keypoints[end_idx] is not None):
            
            start_point = keypoints[start_idx]
            end_point = keypoints[end_idx]
            
            if start_point.score > 0.1 and end_point.score > 0.1:
                x1, y1 = int(start_point.x), int(start_point.y)
                x2, y2 = int(end_point.x), int(end_point.y)
                
                color = colors[i % len(colors)]
                cv2.line(canvas, (x1, y1), (x2, y2), color, thickness=bone_thickness)
    
    # Draw keypoints
    for kp in keypoints:
        if kp is not None and kp.score > 0.1:
            x, y = int(kp.x), int(kp.y)
            cv2.circle(canvas, (x, y), point_size, (0, 0, 255), thickness=-1)
    
    return canvas


def draw_hand_with_custom_params(canvas, keypoints, line_thickness=2, point_size=4):
    """Draw hand pose with custom line thickness and point size."""
    import cv2
    import numpy as np
    import matplotlib.colors
    
    if not keypoints:
        return canvas
    
    # Hand connections (21-point format)
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 4],  # thumb
        [0, 5], [5, 6], [6, 7], [7, 8],  # index finger
        [0, 9], [9, 10], [10, 11], [11, 12],  # middle finger
        [0, 13], [13, 14], [14, 15], [15, 16],  # ring finger
        [0, 17], [17, 18], [18, 19], [19, 20]   # pinky finger
    ]
    
    # Draw connections
    for ie, (start_idx, end_idx) in enumerate(edges):
        if (start_idx < len(keypoints) and end_idx < len(keypoints) and 
            keypoints[start_idx] is not None and keypoints[end_idx] is not None):
            
            start_point = keypoints[start_idx]
            end_point = keypoints[end_idx]
            
            if start_point.score > 0.1 and end_point.score > 0.1:
                x1, y1 = int(start_point.x), int(start_point.y)
                x2, y2 = int(end_point.x), int(end_point.y)
                
                color = matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255
                cv2.line(canvas, (x1, y1), (x2, y2), color.astype(int).tolist(), thickness=line_thickness)
    
    # Draw keypoints
    for kp in keypoints:
        if kp is not None and kp.score > 0.1:
            x, y = int(kp.x), int(kp.y)
            cv2.circle(canvas, (x, y), point_size, (255, 255, 255), thickness=-1)
    
    return canvas


def draw_face_with_custom_params(canvas, keypoints, point_size=3):
    """Draw face keypoints with custom point size."""
    import cv2
    
    if not keypoints:
        return canvas
    
    # Draw face keypoints
    for kp in keypoints:
        if kp is not None and kp.score > 0.1:
            x, y = int(kp.x), int(kp.y)
            cv2.circle(canvas, (x, y), point_size, (255, 255, 255), thickness=-1)
    
    return canvas


class DWPoseJSONToImage:
    """Helper node to convert DWPose JSON keypoints to pose visualization images."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keypoints_json": ("STRING", {"multiline": True, "placeholder": "Paste OpenPose JSON keypoints here..."}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "draw_body": ("BOOLEAN", {"default": True}),
                "draw_hands": ("BOOLEAN", {"default": True}),
                "draw_face": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "point_size": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1}),
                "bone_thickness": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1}),
                "hand_line_thickness": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pose_image",)
    FUNCTION = "convert_json_to_image"
    CATEGORY = "DWPose"
    DESCRIPTION = "Convert DWPose JSON keypoints to pose visualization images. Automatically processes all frames in batch data with customizable drawing parameters."

    def convert_json_to_image(self, keypoints_json, width, height, draw_body, draw_hands, draw_face,
                            point_size=4, bone_thickness=4, hand_line_thickness=2):
        """Convert JSON keypoints to pose image tensor(s). Always processes all frames."""
        
        # Always process all frames
        target_frame_index = -1
        
        # Generate pose images
        pose_pil_list = json_keypoints_to_pose_images(
            keypoints_json, width, height, draw_body, draw_hands, draw_face, 
            target_frame_index, point_size, bone_thickness, hand_line_thickness
        )
        
        # Convert PIL list to batch tensor
        pose_tensors = []
        for pose_pil in pose_pil_list:
            pose_tensor = self._pil_to_tensor(pose_pil)
            pose_tensors.append(pose_tensor)
        
        # Stack into batch tensor [Batch, Height, Width, Channels]
        batch_tensor = torch.stack(pose_tensors, dim=0)
        return (batch_tensor,)
    
    def _pil_to_tensor(self, pil_image):
        """Convert PIL Image to tensor."""
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0)
