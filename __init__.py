from .nodes import DWPoseAnnotator

NODE_CLASS_MAPPINGS = {"DWPoseAnnotator": DWPoseAnnotator}
NODE_DISPLAY_NAME_MAPPINGS = {"DWPoseAnnotator": "DWPose Annotator (Just_DWPose)"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
