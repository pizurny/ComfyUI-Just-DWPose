# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ComfyUI-Just-DWPose is an advanced DWPose (pose estimation) annotator for ComfyUI that supports both TorchScript and ONNX backends. The project provides comprehensive pose detection capabilities including:

- **Single image and batch/sequence processing**
- **Configurable detection thresholds** for fine-tuned keypoint filtering
- **Kalman filtering for temporal smoothing** in video sequences
- **Lazy-loading of models** and offline operation support
- **Robust error handling** with graceful fallbacks

## Architecture

### Core Components

- **`nodes.py`**: Defines the main `DWPoseAnnotator` ComfyUI node class with advanced parameters:
  - Batch/sequence processing support
  - Detection and NMS threshold controls
  - Kalman filtering parameters
  - Tensor format conversion utilities
- **`loader.py`**: Contains the core pose detection logic, model loading, and backend selection:
  - Custom threshold filtering pipeline
  - Backend resolution and model loading
  - Robust error handling with fallbacks
- **`dwpose_kalman_filter.py`**: Kalman filtering implementation for temporal pose smoothing:
  - Per-joint 2D position + velocity tracking
  - Configurable process and measurement noise
  - Sequential pose filtering for video sequences
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

- **`run_dwpose_once()`** (`loader.py:352`): Main entry point for pose detection with threshold support
- **`_run_dwpose_with_thresholds()`** (`loader.py:209`): Custom threshold filtering pipeline
- **`_resolve_backend_and_paths()`** (`loader.py:77`): Backend selection and model path resolution
- **`_load_dwpose()`** (`loader.py:127`): Adaptive model loading with API compatibility detection
- **`get_models_dir()`** (`loader.py:44`): Locates ComfyUI models directory
- **`DWPoseKalmanFilter.update()`** (`dwpose_kalman_filter.py:64`): Applies temporal smoothing to pose sequences

### Processing Pipeline

The node supports multiple processing modes:

1. **Single Image Processing**: Standard pose detection on individual images
2. **Batch Processing**: Processes sequences of images with shape `[Batch, Height, Width, Channels]`
3. **Threshold Filtering**: Custom confidence filtering (when threshold ≠ 0.3)
4. **Kalman Smoothing**: Temporal filtering for smooth pose sequences (batch_size > 1)

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

## Node Parameters

### Basic Parameters
- **`image`**: Input image tensor (supports both single images and batches)
- **`backend`**: Model backend selection ("auto", "torchscript", "onnx")
- **`detect_resolution`**: Detection resolution (128-2048, default: 768)
- **`include_body/hands/face`**: Enable/disable specific pose components
- **`offline_ok/allow_download`**: Model loading behavior

### Advanced Threshold Parameters
- **`detection_threshold`** (0.05-0.9, default: 0.3): 
  - Controls keypoint confidence filtering
  - Lower values = more keypoints (including uncertain ones)
  - Higher values = fewer keypoints (only high-confidence ones)
  - When set to 0.3, uses original DWPose filtering for compatibility
- **`nms_threshold`** (0.1-0.9, default: 0.45):
  - Non-maximum suppression threshold for duplicate detection reduction
  - Currently implemented but limited effect in most single-person scenarios

### Kalman Filtering Parameters
- **`use_kalman`** (Boolean, default: False): Enable temporal smoothing
- **`kalman_process_noise`** (0.001-1.0, default: 0.01):
  - Motion model uncertainty - lower = smoother but less responsive
- **`kalman_measurement_noise`** (0.1-50.0, default: 5.0):
  - Measurement trust level - higher = trust detections less, rely more on prediction
- **`kalman_confidence_threshold`** (0.1-1.0, default: 0.3):
  - Minimum confidence to update filter with new measurements

## Dependencies

Core Python dependencies (from `requirements.txt`):
- **Pillow**: Image processing operations
- **numpy**: Array operations and numerical computations  
- **filterpy**: Kalman filtering library for temporal smoothing
- **onnxruntime/onnxruntime-gpu**: ONNX model inference

The vendored `custom_controlnet_aux` library handles the actual pose detection algorithms.

## Model Storage

- **Location**: `ComfyUI/models/checkpoints/DWPose/`
- **TorchScript files**: `yolox_l.torchscript.pt`, `dw-ll_ucoco_384_bs5.torchscript.pt`
- **ONNX files**: `yolox_l.onnx`, `dw-ll_ucoco_384.onnx`

Models are lazy-loaded on first execution rather than at startup.

## Advanced Features

### Batch/Sequence Processing
The node automatically handles batched input tensors with shape `[Batch, Height, Width, Channels]`:
- **Single images**: Batch size = 1, standard pose detection
- **Image sequences**: Batch size > 1, enables temporal processing capabilities
- **Tensor handling**: Automatic conversion between PIL Images and PyTorch tensors

### Custom Threshold Filtering  
When `detection_threshold ≠ 0.3`, the node uses a custom filtering pipeline:
1. **Raw pose detection**: Gets unfiltered keypoints from DWPose models
2. **Confidence filtering**: Applies custom threshold to individual keypoints
3. **Format conversion**: Handles OpenPose JSON ↔ internal pose formats
4. **Image regeneration**: Creates filtered pose overlay from smoothed keypoints

### Kalman Filtering for Temporal Smoothing
For video sequences (batch_size > 1), Kalman filtering provides:
- **Per-joint tracking**: Independent 2D Kalman filters for each body keypoint
- **Velocity estimation**: Tracks position + velocity for smooth prediction
- **Missing data handling**: Continues tracking when keypoints temporarily disappear
- **Confidence-based updates**: Only updates filter when detection confidence is sufficient

### Graceful Fallbacks
- **Missing filterpy**: Kalman filtering disabled with warning message
- **Filtering errors**: Falls back to original pose detection
- **Backend failures**: Automatic TorchScript → ONNX fallback
- **Model loading issues**: Clear error messages with troubleshooting guidance

## Usage Examples

### Basic Single Image Processing
```python
# Default settings - single image pose detection
node = DWPoseAnnotator()
pose_image, keypoints_json = node.run(
    image=single_image_tensor,  # Shape: [1, H, W, 3]
    backend="auto",
    detect_resolution=768,
    include_body=True,
    include_hands=True,
    include_face=True,
    # All other parameters use defaults
)
```

### Batch Processing with Custom Thresholds  
```python
# Process image sequence with custom confidence filtering
pose_images, keypoints_json = node.run(
    image=batch_tensor,  # Shape: [N, H, W, 3] where N > 1
    detection_threshold=0.2,  # Show more keypoints (lower confidence)
    nms_threshold=0.3,        # More aggressive duplicate suppression
    # Other parameters...
)
```

### Temporal Smoothing with Kalman Filtering
```python
# Smooth pose sequence for video processing
smooth_poses, smooth_keypoints = node.run(
    image=video_frames_tensor,    # Shape: [N, H, W, 3] where N > 1
    use_kalman=True,              # Enable temporal smoothing
    kalman_process_noise=0.005,   # Smooth motion (lower = smoother)
    kalman_measurement_noise=8.0, # Trust level for detections
    kalman_confidence_threshold=0.25,  # Update threshold
    # Other parameters...
)
```

### Parameter Tuning Guidelines

#### Detection Threshold Tuning
- **0.05-0.15**: Very permissive - shows uncertain keypoints, good for difficult poses
- **0.2-0.3**: Standard range - balanced accuracy vs completeness
- **0.4-0.6**: Conservative - only high-confidence keypoints, good for clean poses
- **0.7+**: Very strict - minimal keypoints, may miss important pose information

#### Kalman Filter Tuning
- **Smooth motion** (dance, slow movement): `process_noise=0.001-0.005`
- **Normal motion** (walking, gesturing): `process_noise=0.01-0.02` 
- **Fast motion** (sports, jumping): `process_noise=0.05-0.1`
- **Noisy detections**: Increase `measurement_noise` to 10-30
- **Clean detections**: Decrease `measurement_noise` to 1-5

## Error Handling

The loader includes robust error handling:
- Invalid TorchScript archive detection with automatic ONNX fallback
- Missing model file detection with helpful error messages
- API compatibility detection for different controlnet-aux versions
- Offline mode enforcement to prevent unexpected downloads
- Kalman filtering graceful degradation when filterpy unavailable
- Custom threshold filtering with fallback to original methods

## Performance Considerations

### Memory Usage
- **Batch processing**: Memory scales with batch size - larger sequences use more RAM
- **Kalman filtering**: Minimal memory overhead (per-joint state vectors)
- **Custom thresholds**: Small overhead for format conversions

### Processing Speed
- **Single images**: Fastest mode, no temporal processing
- **Batch sequences**: Moderate overhead for tensor batching
- **Kalman filtering**: Minimal computational overhead (~1-5ms per frame)
- **Custom thresholds**: Small overhead for keypoint filtering and image regeneration

### Recommended Settings
- **Real-time applications**: Use default thresholds, disable Kalman filtering
- **Video processing**: Enable Kalman filtering with moderate process noise (0.01)  
- **High-quality results**: Use lower detection thresholds (0.2) with Kalman smoothing

## Troubleshooting

### Common Issues

#### "filterpy not available" warning
- **Cause**: Kalman filtering enabled but filterpy not installed
- **Solution**: `pip install filterpy` or disable Kalman filtering

#### Black/empty pose images with custom thresholds
- **Cause**: Detection threshold too high, filtering out all keypoints
- **Solution**: Lower detection_threshold to 0.1-0.3 range

#### Kalman filtering has no visual effect
- **Cause**: Only enabled for batch_size > 1, or process noise too high
- **Solution**: Ensure image sequence input, lower kalman_process_noise

#### Model loading failures
- **Cause**: Missing model files or corrupted downloads
- **Solution**: Re-run `get_models.bat` or check model directory

### Debug Information
- **Threshold filtering**: Check console for "[DEBUG]" messages when using custom thresholds
- **Kalman filtering**: "[WARNING]" messages indicate filtering issues or fallbacks  
- **Model loading**: Error messages include specific file paths and troubleshooting steps