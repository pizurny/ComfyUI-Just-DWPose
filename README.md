# ComfyUI-Just-DWPose

A clean **DWPose** annotator for **ComfyUI** with **TorchScript** and **ONNX** backends.  
- Single model location: `ComfyUI/models/checkpoints/DWPose/`
- No downloads at startup; lazy-load on first execution.
- Outputs: pose overlay (IMAGE) + OpenPose-style keypoints (JSON string).

## Install
1) Clone into your ComfyUI custom nodes:
