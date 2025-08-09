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
            },
            "optional": {"model_dir_override": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("pose_image", "keypoints_json")
    FUNCTION = "run"

    def _tensor_to_pil(self, t: torch.Tensor) -> Image.Image:
        arr = (t[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr, "RGB")

    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img, dtype=np.uint8).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

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
        model_dir_override: str = "",
    ) -> Tuple[torch.Tensor, str]:

        models_dir = (
            Path(model_dir_override).expanduser().resolve()
            if model_dir_override.strip()
            else get_models_dir()
        )

        pose_pil, kp_dict = run_dwpose_once(
            self._tensor_to_pil(image),
            backend=backend,
            detect_resolution=int(detect_resolution),
            include_body=include_body,
            include_hands=include_hands,
            include_face=include_face,
            models_dir=models_dir,
            offline_ok=offline_ok,
            allow_download=allow_download,
        )
        return self._pil_to_tensor(pose_pil), json.dumps(kp_dict, ensure_ascii=False)
