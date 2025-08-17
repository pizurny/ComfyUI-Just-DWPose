# Enhanced loader with graceful fallback
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
from pathlib import Path

# Import existing loader functions
try:
    from .loader import run_dwpose_once, get_models_dir, _resolve_backend_and_paths
    LOADER_AVAILABLE = True
except ImportError:
    LOADER_AVAILABLE = False

# Optional enhanced dependencies
try:
    import numpy as np
    import scipy.optimize
    ENHANCED_DEPS_AVAILABLE = True
    print("[Enhanced] Full enhanced features available (numpy, scipy)")
except ImportError:
    ENHANCED_DEPS_AVAILABLE = False
    print("[Enhanced] Enhanced features will use standard detection (missing numpy/scipy)")

def run_enhanced_dwpose(
    pil_image: Image.Image,
    backend: str,
    bbox_detector: str,
    pose_estimator: str,
    detect_resolution: int,
    include_body: bool,
    include_hands: bool,
    include_face: bool,
    models_dir: Path,
    # Multi-scale parameters
    enable_multiscale: bool = False,
    multiscale_scales: List[float] = None,
    multiscale_fusion_method: str = "weighted_average",
    # Multi-model parameters
    enable_multimodel: bool = False,
    multimodel_configs: List[Dict[str, str]] = None,
    multimodel_fusion_method: str = "weighted_average",
    # Other parameters
    **kwargs
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Enhanced DWPose with multi-scale and multi-model support."""
    
    if not LOADER_AVAILABLE:
        raise ImportError("Base loader not available")
    
    # Check if enhanced features are requested but not available
    if (enable_multiscale or enable_multimodel) and not ENHANCED_DEPS_AVAILABLE:
        print("[Enhanced] Multi-scale/multi-model requested but dependencies not available.")
        print("[Enhanced] Using standard detection. Install with: pip install scipy numpy")
        enable_multiscale = False
        enable_multimodel = False
    
    # If enhanced features are available and requested, use them
    if ENHANCED_DEPS_AVAILABLE and (enable_multiscale or enable_multimodel):
        return _run_enhanced_detection(
            pil_image, backend, bbox_detector, pose_estimator, detect_resolution, include_body, include_hands, include_face,
            models_dir, enable_multiscale, multiscale_scales, multiscale_fusion_method,
            enable_multimodel, multimodel_configs, multimodel_fusion_method, **kwargs
        )
    
    # Fall back to standard detection
    return run_dwpose_once(
        pil_image,
        backend=backend,
        bbox_detector=bbox_detector,
        pose_estimator=pose_estimator,
        include_body=include_body,
        include_hands=include_hands,
        include_face=include_face,
        models_dir=models_dir,
        detect_resolution=detect_resolution,
        **kwargs
    )

def _run_enhanced_detection(
    pil_image, backend, bbox_detector, pose_estimator, detect_resolution, include_body, include_hands, include_face,
    models_dir, enable_multiscale, multiscale_scales, multiscale_fusion_method,
    enable_multimodel, multimodel_configs, multimodel_fusion_method, **kwargs
):
    """Run enhanced detection with multi-scale and/or multi-model."""
    
    # Base detector function
    def base_detector(img, **det_kwargs):
        return run_dwpose_once(
            img,
            backend=det_kwargs.get('backend', backend),
            bbox_detector=det_kwargs.get('bbox_detector', bbox_detector),
            pose_estimator=det_kwargs.get('pose_estimator', pose_estimator),
            include_body=include_body,
            include_hands=include_hands,
            include_face=include_face,
            models_dir=models_dir,
            detect_resolution=det_kwargs.get('detect_resolution', detect_resolution),
            **{k: v for k, v in kwargs.items() if k not in ['backend', 'bbox_detector', 'pose_estimator', 'detect_resolution']}
        )
    
    # Multi-model ensemble
    if enable_multimodel and multimodel_configs:
        print(f"[Enhanced] Multi-model ensemble enabled with {len(multimodel_configs)} models")
        
        all_results = []
        
        for model_config in multimodel_configs:
            try:
                model_backend = model_config.get('backend', 'auto')
                print(f"[Enhanced] Running detection with backend: {model_backend}")
                
                pose_img, keypoints_dict = base_detector(
                    pil_image,
                    backend=model_backend,
                    detect_resolution=detect_resolution
                )
                
                if keypoints_dict and 'people' in keypoints_dict and keypoints_dict['people']:
                    all_results.append({
                        'backend': model_backend,
                        'keypoints': keypoints_dict,
                        'pose_img': pose_img
                    })
                    print(f"[Enhanced] {model_backend} detected {len(keypoints_dict['people'])} people")
                else:
                    print(f"[Enhanced] {model_backend} detected no people")
                    
            except Exception as e:
                print(f"[Enhanced] Backend {model_config.get('backend', 'unknown')} failed: {e}")
                continue
        
        if not all_results:
            print("[Enhanced] No successful model predictions, using standard detection")
            return base_detector(pil_image, detect_resolution=detect_resolution)
        
        # Simple fusion: use the result with most detected people
        best_result = max(all_results, key=lambda x: len(x['keypoints']['people']))
        print(f"[Enhanced] Selected {best_result['backend']} with {len(best_result['keypoints']['people'])} people")
        
        return best_result['pose_img'], best_result['keypoints']
    
    # Multi-scale detection
    if enable_multiscale:
        scales = multiscale_scales if multiscale_scales else [0.5, 0.75, 1.0, 1.25, 1.5]
        print(f"[Enhanced] Multi-scale detection enabled with scales: {scales}")
        
        all_detections = []
        
        for scale in scales:
            scaled_resolution = int(detect_resolution * scale)
            if scaled_resolution < 128:
                continue
                
            print(f"[Enhanced] Processing scale {scale:.2f} (resolution: {scaled_resolution})")
            
            try:
                pose_img, keypoints_dict = base_detector(
                    pil_image,
                    detect_resolution=scaled_resolution
                )
                
                if keypoints_dict and 'people' in keypoints_dict and keypoints_dict['people']:
                    all_detections.append({
                        'scale': scale,
                        'keypoints': keypoints_dict,
                        'pose_img': pose_img,
                        'people_count': len(keypoints_dict['people'])
                    })
                    
            except Exception as e:
                print(f"[Enhanced] Scale {scale} failed: {e}")
                continue
        
        if not all_detections:
            print("[Enhanced] No successful multi-scale detections, using standard detection")
            return base_detector(pil_image, detect_resolution=detect_resolution)
        
        # Simple fusion: use the scale with most detected people
        best_detection = max(all_detections, key=lambda x: x['people_count'])
        print(f"[Enhanced] Selected scale {best_detection['scale']} with {best_detection['people_count']} people")
        
        return best_detection['pose_img'], best_detection['keypoints']
    
    # Fallback to standard detection
    return base_detector(pil_image, detect_resolution=detect_resolution)