"""
This is a boilerplate pipeline 'fine_tuning'
generated using Kedro 0.19.5
"""

import logging
from typing import Dict

from PIL import Image
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fine_tune_yolo_model(
        train_images: Dict[str, Image.Image],
        train_annotations: Dict[str, str],
        train_config: dict
    ):
    """
    Fine-tune a pre-trained YOLOv5 model on custom data.
    """
    model = YOLO(train_config["model_name"])
    model.train(
        name=train_config["experiment_name"],
        data=train_config["model_config"]["data"],
        epochs=train_config["model_config"]["epochs"],
        batch=train_config["model_config"]["batch"],
        imgsz=train_config["model_config"]["img_size"],
        # plots=False,
        # save=False,
        # val=True,
        exist_ok=True,
        seed=0,
        cache=True,
        # single_cls=True,
        # iterations=5
    )


    return "Model trained and saved successfully"
