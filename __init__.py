from .nodes import DWPoseAnnotator, DWPoseJSONToImage

NODE_CLASS_MAPPINGS = {
    "DWPoseAnnotator": DWPoseAnnotator,
    "DWPoseJSONToImage": DWPoseJSONToImage
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DWPoseAnnotator": "DWPose Annotator (Just_DWPose)",
    "DWPoseJSONToImage": "DWPose JSON to Image"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
