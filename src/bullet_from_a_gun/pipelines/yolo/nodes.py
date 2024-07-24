"""
This is a boilerplate pipeline 'yolo'
generated using Kedro 0.19.5
"""
import copy
import io
import logging
import os
import random
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

# from coco_eval import CocoEvaluator
# from coco_utils import get_coco_api_from_dataset
# from engine import evaluate
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.metrics import (  # noqa: F401
    average_precision_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    SSD300_VGG16_Weights,  # noqa: F401
    fasterrcnn_resnet50_fpn_v2,
    ssd300_vgg16,  # noqa: F401
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics import YOLO

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def fine_tune_yolo(
    dataprep_params: dict,
    fine_tuning_params: dict,
) -> dict:
    """
    `todo` documentation.
    """
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    np.set_printoptions(precision=5)

    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    model = YOLO(fine_tuning_params["model_name"].replace(".pt", ".yaml"))
    model = YOLO(fine_tuning_params["model_name"])
    model.to("cuda")
    logger.debug(torch.cuda.is_available())
    logger.debug(torch.cuda.current_device())
    logger.debug(torch.cuda.get_device_name())
    logger.debug(torch.cuda.memory_allocated())
    logger.debug(torch.cuda.memory_reserved())
    logger.debug(torch.cuda.memory_summary())

    results = dict()

    _experiment_id_ = dataprep_params['experiment_id']
    _output_path_ = os.path.join(*fine_tuning_params["path"])
    _yolo_data_ = os.path.join(*dataprep_params['yolo_data']['path'])
    _yolo_conf_data_ = os.path.join(_yolo_data_, 'data.yaml')

    logger.debug(_experiment_id_)
    logger.debug(_yolo_data_)
    logger.debug(_output_path_)

    model.train(
        data = _yolo_conf_data_,
        epochs = fine_tuning_params["model_config"]["epochs"],
        batch = fine_tuning_params["model_config"]["batch"],
        imgsz = fine_tuning_params["model_config"]["img_size"],
        project = _output_path_,
        name = _experiment_id_,
        device = 0,
        exist_ok = True,
        save = True,
        val = True,
        cache = True,
        single_cls = True,
        optimizer="SGD",
        lr0 = 0.0001
    )

    return results
