my kedro project has the following structure catalog

# catalog.yml
primary_circle_yolov5_train_images:
    type: partitions.PartitionedDataset
    path: data/03_primary/gunshots/circle/yolov5_format/images/train/
    dataset:
        type: pillow.ImageDataset
    filename_suffix: '.jpg'
    metadata:
        kedro-viz:
            layer: primary

primary_circle_yolov5_train_annotations:
    type: partitions.PartitionedDataset
    path: data/03_primary/gunshots/circle/yolov5_format/labels/train/
    dataset:
        type: text.TextDataset
    filename_suffix: '.txt'
    metadata:
        kedro-viz:
            layer: primary

primary_circle_yolov5_train_config:
    type: json.JSONDataset
    filepath: data/03_primary/gunshots/circle/yolov5_format/data_config.json
    metadata:
        kedro-viz:
            layer: primary

write/fill the function "fine_tune_yolov5_model" node that fine-tune a yolov5 pre-trainned model to customize it to object detection of gunshots in images, based on primary_circle_yolov5_train_images and primary_circle_yolov5_train_annotations datasets

"""
This is a boilerplate pipeline 'fine_tuning'
generated using Kedro 0.19.5
"""

import logging
import random
from typing import Dict

import numpy as np
import torch
from PIL import Image

import yolov5

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _preprocess_images(images):
    processed_images = {}
    for image_name, image in images.items():
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
    # todo
    return "Model trained and saved successfully"


knowing that

config:
{'colors': [240, 230, 92], 'names': ['gunshot_hole'], 'nc': 1, 'train': '../298631_bullet-from-a-gun/images/train', 'val': '../298631_bullet-from-a-gun/images/val'}

train:
{'941548_dataset 2024-05-08 15:56:28_1': <bound method AbstractVersionedDataset.load of <kedro_datasets.pillow.image_dataset.ImageDataset object at 0x75b7e658ba90>>, ..., '941548_dataset 2024-05-08 15:56:28_9': <bound method AbstractVersionedDataset.load of                                    
<kedro_datasets.pillow.image_dataset.ImageDataset object at 0x75b7e62b56c0>>}                                                                                                                                                                                   
annotations:
{'941548_dataset 2024-05-08 15:56:28_1': <bound method AbstractVersionedDataset.load of <kedro_datasets.text.text_dataset.TextDataset object at 0x75b7e62b5540>>, ..., '941548_dataset 2024-05-08 15:56:28_9': <bound method AbstractVersionedDataset.load of <kedro_datasets.text.text_dataset.TextDataset object at                        
0x75b7e62b6500>>} 



import logging
import random
from typing import Dict

import numpy as np
import torch
from PIL import Image

from yolov5 import train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _preprocess_images(images):
    processed_images = {}
    for image_name, image in images.items():
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
    # Preprocess the images
    processed_images = _preprocess_images(train_images)

    # Split the data into train and test sets
    train_images, train_annotations, test_images, test_annotations = _split_data(processed_images, train_annotations)

    # Create a data configuration file
    data_config = {
        "train": train_config["train"],
        "val": train_config["val"],
        "nc": train_config["nc"],
        "names": train_config["names"]
    }

    # Save the data configuration to a YAML file
    with open("data.yaml", "w") as f:
        yaml.dump(data_config, f)

    # Set up the YOLOv5 model for training
    model = yolov5.load('yolov5s.pt')  # Load a pre-trained YOLOv5 model
    model.nc = data_config["nc"]  # Update the number of classes
    model.names = data_config["names"]  # Update the class names
    model.hyp["box"] *= 0.5  # Scale the box loss
    model.hyp["cls"] *= 0.5  # Scale the class loss

    # Fine-tune the model
    train.run(
        data='data.yaml',
        imgsz=640,
        weights='yolov5s.pt',
        epochs=50,
        batch_size=16,
        name='gunshot_detection'
    )

    return "Model trained and saved successfully"