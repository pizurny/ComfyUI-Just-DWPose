# ComfyUI-Just-DWPose

An advanced **DWPose** annotator for **ComfyUI** with **TorchScript** and **ONNX** backends, featuring comprehensive pose detection, bone validation, temporal smoothing, and custom visualization tools.

## ✨ Features

### 🎯 **Core Capabilities**
- **Dual Backend Support**: TorchScript (CUDA/CPU) and ONNX Runtime with automatic fallback
- **Single/Batch Processing**: Handle individual images or full video sequences  
- **Multi-Person Detection**: Select specific people in multi-person scenes
- **Lazy Model Loading**: Models load on first use, not at startup
- **Offline Operation**: Works completely offline once models are downloaded

### 🩻 **Advanced Pose Processing**
- **Bone Validation**: Automatic removal of elongated finger artifacts and pose glitches
- **Kalman Filtering**: Temporal smoothing for video sequences with configurable parameters
- **Custom Thresholds**: Fine-tune detection sensitivity and NMS filtering
- **Component Control**: Enable/disable body, hands, and face detection independently

### 🖼️ **Triple Output System**
1. **Pose Image**: Clean skeleton visualization on black background
2. **Keypoints JSON**: Complete OpenPose-compatible keypoint data
3. **Proof Output**: ✨ Original frames with skeleton overlay for verification

### 🎨 **Helper Node: DWPose JSON to Image**
- **Custom Visualization**: Convert any OpenPose JSON to pose images with custom drawing parameters
- **Drawing Controls**: Adjustable point size, bone thickness, and hand line thickness
- **Batch Support**: Automatically processes entire sequences
- **Workflow Flexibility**: Use with any JSON keypoint source, not just DWPose

## 🚀 Quick Start

### Installation
```bash
# Clone into your ComfyUI custom nodes directory
cd ComfyUI/custom_nodes
git clone https://github.com/pizurny/ComfyUI-Just-DWPose.git
cd ComfyUI-Just-DWPose

# Install Python dependencies
pip install -r requirements.txt

# Download models (choose one)
get_models.bat           # Download all models (TorchScript + ONNX)
get_models.bat ts        # TorchScript only (recommended for CUDA)
get_models.bat onnx      # ONNX only (CPU/compatibility)
```

### Basic Usage
1. **Add Node**: Search for "DWPose Annotator (Just_DWPose)" in ComfyUI
2. **Connect Image**: Input your image or image sequence 
3. **Run**: Get pose image, JSON keypoints, and proof overlay
4. **Optional**: Use "DWPose JSON to Image" helper node for custom visualization

## 📊 Node Reference

### DWPose Annotator (Just_DWPose)

#### Basic Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | STRING | "auto" | Model backend: "auto", "torchscript", "onnx" |
| `detect_resolution` | INT | 768 | Detection resolution (128-2048) |
| `include_body` | BOOLEAN | True | Enable body skeleton detection |
| `include_hands` | BOOLEAN | True | Enable hand keypoint detection |
| `include_face` | BOOLEAN | True | Enable facial landmark detection |

#### Advanced Parameters
| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `detection_threshold` | FLOAT | 0.3 | 0.05-0.9 | Keypoint confidence threshold |
| `nms_threshold` | FLOAT | 0.45 | 0.1-0.9 | Non-maximum suppression threshold |
| `person_index` | STRING | "0" | - | Which person to process (0, 1, 2...) |

#### Bone Validation
| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `enable_bone_validation` | BOOLEAN | True | - | Enable elongated bone removal |
| `max_bone_ratio` | FLOAT | 2.5 | 0.5-10.0 | Maximum bone length ratio |
| `min_keypoint_confidence` | FLOAT | 0.5 | 0.1-0.9 | Minimum confidence for validation |

#### Kalman Filtering (Video Sequences)
| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `use_kalman` | BOOLEAN | False | - | Enable temporal smoothing |
| `kalman_process_noise` | FLOAT | 0.01 | 0.001-1.0 | Motion model uncertainty |
| `kalman_measurement_noise` | FLOAT | 5.0 | 0.1-50.0 | Measurement trust level |
| `kalman_confidence_threshold` | FLOAT | 0.3 | 0.1-1.0 | Update confidence threshold |

#### Outputs
- **`pose_image`** (IMAGE): Skeleton visualization on black background
- **`keypoints_json`** (STRING): OpenPose-compatible JSON keypoint data  
- **`proof`** (IMAGE): Original frames with skeleton overlay (60% opacity)

### DWPose JSON to Image

Convert JSON keypoints to custom pose visualizations.

#### Parameters
| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `keypoints_json` | STRING | - | - | JSON keypoint data (from DWPose or other source) |
| `width` | INT | 512 | 64-2048 | Output image width |
| `height` | INT | 512 | 64-2048 | Output image height |
| `draw_body` | BOOLEAN | True | - | Draw body skeleton |
| `draw_hands` | BOOLEAN | True | - | Draw hand keypoints |
| `draw_face` | BOOLEAN | True | - | Draw facial landmarks |

#### Drawing Controls
| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `point_size` | INT | 4 | 1-20 | Keypoint circle size |
| `bone_thickness` | INT | 4 | 1-20 | Skeleton bone line thickness |
| `hand_line_thickness` | INT | 2 | 1-10 | Hand connection line thickness |

## 🎛️ Usage Examples

### Single Image Processing
```
Basic pose detection with default settings:
[Image] → [DWPose Annotator] → [Preview Image] (pose_image output)
```

### Video Sequence with Temporal Smoothing
```
Batch processing with Kalman filtering for smooth pose sequences:
[Image Sequence] → [DWPose Annotator] 
  ├─ use_kalman: True
  ├─ kalman_process_noise: 0.01
  └─ kalman_measurement_noise: 5.0
→ [Preview Image] (smooth pose sequence)
```

### Custom Visualization Pipeline
```
Generate custom pose visualizations with different drawing styles:
[DWPose Annotator] → keypoints_json → [DWPose JSON to Image]
  ├─ point_size: 8 (large keypoints)
  ├─ bone_thickness: 6 (thick skeleton)
  └─ hand_line_thickness: 4 (thick finger lines)
→ [Preview Image] (custom styled poses)
```

### Multi-Person Scene Processing
```
Process specific people in crowded scenes:
[Multi-Person Image] → [DWPose Annotator]
  └─ person_index: "1" (select second person)
→ [Preview Image] (pose for person 1 only)
```

## 🔧 Parameter Tuning Guide

### Detection Thresholds
- **0.1-0.2**: Very permissive, shows uncertain keypoints
- **0.3**: Default, balanced accuracy vs completeness  
- **0.5-0.7**: Conservative, only high-confidence keypoints

### Bone Validation
- **max_bone_ratio 1.5-2.0**: Strict validation, removes more connections
- **max_bone_ratio 2.5-3.5**: Moderate validation (recommended)
- **max_bone_ratio 5.0+**: Permissive, allows longer connections

### Kalman Filtering
- **Smooth motion** (dance, slow): `process_noise=0.001-0.005`
- **Normal motion** (walking): `process_noise=0.01-0.02`
- **Fast motion** (sports): `process_noise=0.05-0.1`
- **Noisy detections**: Increase `measurement_noise` to 10-30
- **Clean detections**: Decrease `measurement_noise` to 1-5

## 🗂️ Model Storage

**Location**: `ComfyUI/models/checkpoints/DWPose/`

**TorchScript Models** (recommended for CUDA):
- `yolox_l.torchscript.pt` (person detection)
- `dw-ll_ucoco_384_bs5.torchscript.pt` (pose estimation)

**ONNX Models** (CPU/compatibility):
- `yolox_l.onnx` (person detection)  
- `dw-ll_ucoco_384.onnx` (pose estimation)

## 🐛 Troubleshooting

### Common Issues

**"No models found"**
- Run `get_models.bat` to download required models
- Check `ComfyUI/models/checkpoints/DWPose/` directory exists

**"filterpy not available" warning**  
- Install with: `pip install filterpy`
- Or disable Kalman filtering in node parameters

**Black pose images with custom JSON**
- Ensure correct width/height matching keypoint coordinates
- Check detection_threshold isn't too high (try 0.1-0.3)

**Custom drawing parameters not working**
- Use the "DWPose JSON to Image" helper node
- Original node uses fixed drawing parameters

### Performance Tips
- **Single images**: Disable Kalman filtering for best speed
- **Video sequences**: Enable Kalman filtering for smooth results
- **Large batches**: Consider memory usage with sequences >50 frames
- **CUDA acceleration**: Use TorchScript backend with GPU

## 🤝 Contributing

Issues and pull requests welcome! Please check existing issues before creating new ones.

## 📄 License

This project follows the same license as the original DWPose implementation.

## 🙏 Credits

Based on the excellent DWPose work and adapted for ComfyUI with enhanced features for professional pose detection workflows.
