# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ComfyUI-Just-DWPose is a clean DWPose (pose estimation) annotator for ComfyUI that supports both TorchScript and ONNX backends. The project provides pose detection capabilities with lazy-loading of models and offline operation.

## Architecture

### Core Components

- **`nodes.py`**: Defines the main `DWPoseAnnotator` ComfyUI node class with input/output specifications
- **`loader.py`**: Contains the core pose detection logic, model loading, and backend selection
- **`__init__.py`**: ComfyUI node registration and exports
- **`vendor/custom_controlnet_aux/`**: Vendored DWPose detection library with ONNX and TorchScript implementations

### Backend Architecture

The system supports two inference backends:
- **TorchScript**: Uses `.torchscript.pt` files with CUDA/CPU device selection
- **ONNX**: Uses `.onnx` files with ONNX Runtime providers

Model resolution follows this priority:
1. User-specified backend (if available)
2. Auto-detection: TorchScript if models exist, fallback to ONNX
3. Graceful fallback from invalid TorchScript to ONNX if available

### Key Functions

- **`run_dwpose_once()`** (`loader.py:205`): Main entry point for pose detection
- **`_resolve_backend_and_paths()`** (`loader.py:77`): Backend selection and model path resolution
- **`_load_dwpose()`** (`loader.py:127`): Adaptive model loading with API compatibility detection
- **`get_models_dir()`** (`loader.py:44`): Locates ComfyUI models directory

## Development Commands

### Model Management
```bash
# Download all models (TorchScript + ONNX)
get_models.bat

# Download only TorchScript models  
get_models.bat ts

# Download only ONNX models
get_models.bat onnx

# Force re-download existing models
get_models.bat --force
```

### Testing Integration
Since this is a ComfyUI custom node, testing requires running within ComfyUI:
1. Install in ComfyUI's `custom_nodes/` directory
2. Run `get_models.bat` to download required models
3. Test through ComfyUI interface with the "DWPose Annotator (Just_DWPose)" node

## Dependencies

Core Python dependencies (from `requirements.txt`):
- Pillow (image processing)
- numpy (array operations)
- onnxruntime/onnxruntime-gpu (ONNX inference)

The vendored `custom_controlnet_aux` library handles the actual pose detection algorithms.

## Model Storage

- **Location**: `ComfyUI/models/checkpoints/DWPose/`
- **TorchScript files**: `yolox_l.torchscript.pt`, `dw-ll_ucoco_384_bs5.torchscript.pt`
- **ONNX files**: `yolox_l.onnx`, `dw-ll_ucoco_384.onnx`

Models are lazy-loaded on first execution rather than at startup.

## Error Handling

The loader includes robust error handling:
- Invalid TorchScript archive detection with automatic ONNX fallback
- Missing model file detection with helpful error messages
- API compatibility detection for different controlnet-aux versions
- Offline mode enforcement to prevent unexpected downloads