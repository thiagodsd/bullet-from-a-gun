"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from .pipelines.detectron2 import create_pipeline as detectron2_pipeline
from .pipelines.mask_rcnn import create_pipeline as mask_rcnn_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    detectron2 = detectron2_pipeline()
    mask_rcnn = mask_rcnn_pipeline()
    return {
        "__default__"      : detectron2 + mask_rcnn,
        "detectron2" : detectron2,
        "mask_rcnn"  : mask_rcnn,
    }
