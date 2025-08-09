from typing import Tuple
import numpy as np
import torch
from PIL import Image

class DWPoseAnnotator:
    CATEGORY = "annotators/dwpose"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("pose_image", "keypoints_json")
    FUNCTION = "run"

    def _tensor_to_pil(self, t: torch.Tensor) -> Image.Image:
        arr = (t[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr, "RGB")

    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img, dtype=np.uint8).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def run(self, image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        # placeholder: pass-through image + empty JSON
        return image, "{}"
