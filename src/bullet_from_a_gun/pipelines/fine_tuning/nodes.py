"""
This is a boilerplate pipeline 'fine_tuning'
generated using Kedro 0.19.5
"""

import logging
from typing import Dict

import torch
from PIL import Image
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fine_tune_yolo_model(
        train_config: dict
    ):
    """
    Fine-tune a pre-trained YOLOv5 model on custom data.
    """
    # check my cuda
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    model = YOLO(train_config["model_name"])
    model.to("cuda")
    logger.info(torch.cuda.is_available())
    logger.info(torch.cuda.current_device())
    logger.info(torch.cuda.get_device_name())
    logger.info(torch.cuda.memory_allocated())
    logger.info(torch.cuda.memory_reserved())
    logger.info(torch.cuda.memory_summary())
    model.tune(
        name=train_config["experiment_name"],
        data=train_config["model_config"]["data"],
        epochs=train_config["model_config"]["epochs"],
        batch=train_config["model_config"]["batch"],
        imgsz=train_config["model_config"]["img_size"],
        plots=True,
        save=False,
        val=True,
        exist_ok=True,
        seed=0,
        cache=True,
        # single_cls=True,
        iterations=3,
        device=0
    )
    model.train(
        name=train_config["experiment_name"],
        data=train_config["model_config"]["data"],
        epochs=train_config["model_config"]["epochs"],
        batch=train_config["model_config"]["batch"],
        imgsz=train_config["model_config"]["img_size"],
        plots=True,
        save=True,
        val=True,
        exist_ok=True,
        seed=0,
        cache=True,
        # single_cls=True,
        # iterations=3,
        device=0
    )


    return "Model trained and saved successfully"
