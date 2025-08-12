from typing import Tuple
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
from .loader import run_dwpose_once, get_models_dir

class DWPoseAnnotator:
    CATEGORY = "annotators/dwpose"

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
            },
            "optional": {"model_dir_override": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("pose_image", "keypoints_json")
    FUNCTION = "run"

    def _tensor_to_pil(self, t: torch.Tensor) -> Image.Image:
        arr = (t.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr, "RGB")

    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img, dtype=np.uint8).astype(np.float32) / 255.0
        return torch.from_numpy(arr)

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
        model_dir_override: str = "",
    ) -> Tuple[torch.Tensor, str]:

        models_dir = (
            Path(model_dir_override).expanduser().resolve()
            if model_dir_override.strip()
            else get_models_dir()
        )

        # Handle batch processing - image tensor shape is [Batch, Height, Width, Channels]
        batch_size = image.shape[0]
        pose_images = []
        all_keypoints = []
        
        for i in range(batch_size):
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
            )
            pose_images.append(self._pil_to_tensor(pose_pil))
            all_keypoints.append(kp_dict)
        
        # Stack all processed images back into batch format [Batch, Height, Width, Channels]
        batch_pose_tensor = torch.stack(pose_images, dim=0)
        
        # Combine all keypoints into a single JSON string
        combined_keypoints = json.dumps(all_keypoints, ensure_ascii=False)
        
        return batch_pose_tensor, combined_keypoints
