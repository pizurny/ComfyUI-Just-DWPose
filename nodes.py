from typing import Tuple
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
import gc
from .loader import run_dwpose_once, get_models_dir

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

class DWPoseAnnotator:
    CATEGORY = "annotators/dwpose"
    
    def __init__(self):
        self.kalman_filter = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "backend": (["auto", "torchscript", "onnx"], {"default": "auto"}),
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
            },
            "optional": {"model_dir_override": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("pose_image", "keypoints_json")
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

    def _convert_pose_to_kalman_format(self, pose_dict):
        """Convert pose dictionary to numpy format for Kalman filtering."""
        if 'people' not in pose_dict or not pose_dict['people']:
            return None
            
        person = pose_dict['people'][0]
        
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
    
    def _convert_kalman_to_pose_format(self, pose_array, original_dict):
        """Convert filtered numpy pose back to original dictionary format."""
        if pose_array is None:
            return original_dict
            
        # Update the pose keypoints in the dictionary
        filtered_dict = json.loads(json.dumps(original_dict))  # Deep copy
        if 'people' in filtered_dict and filtered_dict['people']:
            # Split the filtered array back into body, hands, and face
            flattened = pose_array.flatten().tolist()
            
            # Body keypoints: first 18 joints (54 values)
            body_end = 18 * 3
            filtered_dict['people'][0]['pose_keypoints_2d'] = flattened[:body_end]
            
            # Left hand keypoints: next 21 joints (63 values)
            left_hand_start = body_end
            left_hand_end = left_hand_start + 21 * 3
            left_hand_kp = flattened[left_hand_start:left_hand_end]
            # Only add if not all zeros (meaning hand was detected)
            if any(abs(x) > 0.001 for x in left_hand_kp[2::3]):  # Check confidence values
                filtered_dict['people'][0]['hand_left_keypoints_2d'] = left_hand_kp
                
            # Right hand keypoints: next 21 joints (63 values)  
            right_hand_start = left_hand_end
            right_hand_end = right_hand_start + 21 * 3
            right_hand_kp = flattened[right_hand_start:right_hand_end]
            # Only add if not all zeros (meaning hand was detected)
            if any(abs(x) > 0.001 for x in right_hand_kp[2::3]):  # Check confidence values
                filtered_dict['people'][0]['hand_right_keypoints_2d'] = right_hand_kp
                
            # Face keypoints: remaining 70 joints (210 values)
            face_start = right_hand_end
            face_kp = flattened[face_start:]
            # Only add if not all zeros (meaning face was detected)
            if any(abs(x) > 0.001 for x in face_kp[2::3]):  # Check confidence values
                filtered_dict['people'][0]['face_keypoints_2d'] = face_kp
            
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
        model_dir_override: str = "",
    ) -> Tuple[torch.Tensor, str]:
        
        try:
            # Input validation
            if image is None or not hasattr(image, 'shape'):
                raise ValueError("Invalid input image tensor")
            
            print(f"[Just-DWPose] Input tensor shape: {image.shape}, dtype: {image.dtype}")
            
            # Check for reasonable batch size
            batch_size = image.shape[0]
            if batch_size > 500:
                raise ValueError(f"Batch size {batch_size} is too large. Maximum recommended: 500")
            
            # Validate bone validation parameters
            if max_bone_ratio <= 0:
                max_bone_ratio = 2.5
                print(f"[WARNING] Invalid max_bone_ratio, using default: {max_bone_ratio}")
            if min_keypoint_confidence <= 0 or min_keypoint_confidence > 1.0:
                min_keypoint_confidence = 0.5
                print(f"[WARNING] Invalid min_keypoint_confidence, using default: {min_keypoint_confidence}")
            
            # Ensure models_dir is always a Path object
            try:
                models_dir = (
                    Path(model_dir_override).expanduser().resolve()
                    if model_dir_override.strip()
                    else get_models_dir()
                )
                
                # Check if the path looks wrong (contains "True")
                if "True" in str(models_dir):
                    print(f"[ERROR] Invalid models_dir detected: {models_dir}")
                    # Use hardcoded fallback path
                    models_dir = Path("X:/ai/Comfy_Dev/ComfyUI_windows_portable/ComfyUI/models/checkpoints/DWPose")
                    print(f"[DEBUG] Using hardcoded fallback: {models_dir}")
                
                # Force conversion to ensure it's a Path object
                models_dir = Path(models_dir)
            except Exception as e:
                print(f"[ERROR] Failed to resolve models_dir: {e}")
                # Use hardcoded fallback as last resort
                models_dir = Path("X:/ai/Comfy_Dev/ComfyUI_windows_portable/ComfyUI/models/checkpoints/DWPose")
                print(f"[DEBUG] Emergency fallback models_dir: {models_dir}")

            # Handle batch processing - image tensor shape is [Batch, Height, Width, Channels]
            print(f"[Just-DWPose] Processing batch of {batch_size} images...")
            
            # Determine target size from input images (use first image dimensions)
            input_height, input_width = image.shape[1], image.shape[2]
            target_size = (input_height, input_width)
            print(f"[Just-DWPose] Target output size: {target_size}")
            
            pose_images = []
            all_keypoints = []
            
            # Initialize Kalman filter if enabled and filterpy is available
            if use_kalman and KALMAN_AVAILABLE and batch_size > 1:
                # Reset filter for new batch/sequence
                self.kalman_filter = DWPoseKalmanFilter(
                    num_joints=130,  # Total: 18 body + 21 left hand + 21 right hand + 70 face
                    process_noise=kalman_process_noise,
                    measurement_noise=kalman_measurement_noise
                )
            
            for i in range(batch_size):
                # Progress reporting for large batches
                if batch_size > 10 and i % 10 == 0:
                    print(f"[Just-DWPose] Processing frame {i+1}/{batch_size}")
                    
                try:
                    # Process each image in the batch
                    single_image = image[i]  # Shape: [Height, Width, Channels]
                    pose_pil, kp_dict = run_dwpose_once(
                        self._tensor_to_pil(single_image),
                        backend=backend,
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
                    )
                    
                    # Apply Kalman filtering if enabled and available
                    if use_kalman and KALMAN_AVAILABLE and self.kalman_filter is not None and batch_size > 1:
                        try:
                            # Convert to Kalman format and filter
                            pose_array = self._convert_pose_to_kalman_format(kp_dict)
                            if pose_array is not None:
                                filtered_pose_array = self.kalman_filter.update(
                                    pose_array, 
                                    confidence_threshold=kalman_confidence_threshold
                                )
                                # Convert back to dictionary format
                                kp_dict = self._convert_kalman_to_pose_format(filtered_pose_array, kp_dict)
                                
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
                        
                    pose_images.append(self._pil_to_tensor(pose_pil, target_size))
                    all_keypoints.append(kp_dict)
                    
                    # Memory cleanup for large batches
                    if batch_size > 50 and i % 25 == 0:
                        gc.collect()
                    
                except Exception as e:
                    print(f"[ERROR] Failed to process frame {i+1}/{batch_size}: {e}")
                    # Create a black image as fallback with correct target size
                    fallback_img = Image.new('RGB', (target_size[1], target_size[0]), (0, 0, 0))  # PIL uses (width, height)
                    pose_images.append(self._pil_to_tensor(fallback_img))
                    all_keypoints.append({"version": "ap10k", "people": []})
            
            # Final memory cleanup for large batches
            if batch_size > 50:
                gc.collect()
                
            print(f"[Just-DWPose] Batch processing complete: {batch_size} frames")
            
            # Stack all processed images back into batch format [Batch, Height, Width, Channels]
            batch_pose_tensor = torch.stack(pose_images, dim=0)
            
            # Combine all keypoints into a single JSON string
            combined_keypoints = json.dumps(all_keypoints, ensure_ascii=False)
            
            return batch_pose_tensor, combined_keypoints
            
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
            return fallback_tensor, fallback_json
