# vendor/custom_controlnet_aux/util.py
from __future__ import annotations
import os
import numpy as np
from PIL import Image

def HWC3(x):
    """Ensure uint8 HxWx3 numpy image."""
    if isinstance(x, Image.Image):
        x = np.array(x)
    x = np.asarray(x)
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=2)
    if x.shape[2] == 4:
        x = x[:, :, :3]
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    return x

def resize_image_with_pad(img: np.ndarray, resolution: int, upscale_method="INTER_CUBIC"):
    """Keep aspect ratio, pad with black to target resolution. If resolution=0, return original."""
    import cv2
    
    # If resolution is 0, return original image with identity transform
    if resolution == 0:
        def identity_remove_pad(x):
            return x
        return img, identity_remove_pad
    
    ih, iw = img.shape[:2]
    # Make it square with the given resolution
    w = h = resolution
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    
    # Get interpolation method
    interp_map = {
        "INTER_CUBIC": cv2.INTER_CUBIC,
        "INTER_LINEAR": cv2.INTER_LINEAR,
        "INTER_NEAREST": cv2.INTER_NEAREST
    }
    interp = interp_map.get(upscale_method, cv2.INTER_CUBIC)
    
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    top, left = (h - nh) // 2, (w - nw) // 2
    out[top:top+nh, left:left+nw] = resized
    
    # Return function to remove padding
    def remove_pad(canvas):
        return canvas[top:top+nh, left:left+nw]
    
    return out, remove_pad

def common_input_validate(image, output_type="pil", **kwargs):
    """Convert to HWC3 np.uint8 numpy array expected by dwpose."""
    if isinstance(image, Image.Image):
        image = np.array(image)
    image = HWC3(image)  # Ensure proper format
    return image, output_type

def custom_hf_download(repo_id_or_dir: str, filename: str, local_files_only: bool = True):
    """
    Minimal stub: we only support local directories.
    """
    base = os.path.abspath(repo_id_or_dir)
    p = os.path.join(base, filename)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Expected local file: {p}")
    return p

def get_logger():
    class _L:
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def error(self, *a, **k): pass
    return _L()
