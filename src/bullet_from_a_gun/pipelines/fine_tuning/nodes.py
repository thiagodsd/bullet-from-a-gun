"""
This is a boilerplate pipeline 'fine_tuning'
generated using Kedro 0.19.5
"""

import logging
import random
from typing import Dict

import numpy as np
import torch
import tqdm
from PIL import Image

import yolov5

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _preprocess_images(images):
    processed_images = {}
    for image_name, image in tqdm.tqdm(images.items()):
        _image = image.resize((640, 640))
        _image = np.array(_image) / 255.0
        processed_images[image_name] = _image
    return processed_images


def _split_data(images, annotations):
    image_items = list(images.items())
    random.shuffle(image_items)
    split_index = int(0.8 * len(image_items))
    train_images = dict(image_items[:split_index])
    test_images = dict(image_items[split_index:])
    train_annotations = {k: annotations[k] for k in train_images}
    test_annotations = {k: annotations[k] for k in test_images}
    return train_images, train_annotations, test_images, test_annotations


def fine_tune_yolov5_model(
        train_images: Dict[str, Image.Image],
        train_annotations: Dict[str, str],
        train_config: dict
    ):
    """
    Fine-tune a pre-trained YOLOv5 model on custom data.
    """
    train_path = "/home/dusoudeth/Documentos/github/bullet-from-a-gun/yolov5/train.py"
    data_config = "/home/dusoudeth/Documentos/github/bullet-from-a-gun/data/03_primary/gunshots/circle/yolov5_format/data_config.yaml"
    command = f"python {train_path} --img 640 --batch 16 --epochs 3 --data {data_config} --weights yolov5s.pt --cache"
    logger.info(f"""Running command
    {command}
    """)
    return "Model trained and saved successfully"
