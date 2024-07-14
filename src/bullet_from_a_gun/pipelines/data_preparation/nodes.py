"""
This is a boilerplate pipeline 'data_preparation'
generated using Kedro 0.19.5
"""
import json
import logging
import random
from typing import Dict

import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def load_images(image_dict: Dict[str, Image.Image]) -> Dict[str, Image.Image]:
    logger.info(image_dict)
    return image_dict


def _handle_json_annotations(coco_data_path: str):
    """
    `todo` add documenation
    """
    dataset_dicts = list()

    coco_data = json.load(open(coco_data_path))

    images = {image['id']: image for image in coco_data['images']}
    annotations = coco_data['annotations']

    for image_id, image_info in images.items():
        record = {}

        filename = f"{coco_data_path.replace('_annotations.coco.json', '')}/{image_info['file_name']}"
        height, width = image_info['height'], image_info['width']

        record["file_name"] = filename
        record["image_id"] = image_id
        record["height"] = height
        record["width"] = width

        objs = []
        for anno in annotations:
            if anno['image_id'] == image_id:
                bbox = anno['bbox']
                # COCO format is [x, y, width, height] for bounding boxes
                px = [bbox[0], bbox[0] + bbox[2]]
                py = [bbox[1], bbox[1] + bbox[3]]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                obj = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": anno['segmentation'],
                    "category_id": anno['category_id'],
                    "iscrowd": anno.get('iscrowd', 0)
                }
                objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def read_coco_data(
        dataprep_params: dict,
    ):
    """
    `todo` add documenation
    """
    _experiment_id_ = dataprep_params['experiment_id']
    _coco_path_ = dataprep_params['coco_data']['path']

    logger.debug(dataprep_params)

    for _dataset_ in dataprep_params['coco_data']['datasets']:
        logger.debug(f"reading {_dataset_} data")
        _annotations_ = _handle_json_annotations(f"{_coco_path_}/{_dataset_}/_annotations.coco.json")
        DatasetCatalog.register(f"{_experiment_id_}_{_dataset_}", lambda: _annotations_)
        MetadataCatalog.get(f"{_experiment_id_}_{_dataset_}").set(thing_classes=["none", "bullets"])

    # # sanity check (train)
    # train_metadata = MetadataCatalog.get(f"{_experiment_id_}_train")
    # dataset_dicts = _handle_json_annotations(f"{_coco_path_}/train/_annotations.coco.json")
    # for _img_ in random.sample(dataset_dicts, 3):
    #     logger.debug(_img_)
    #     im = cv2.imread(_img_["file_name"])
    #     visualizer = Visualizer(im[:, :, ::-1], metadata=train_metadata, scale=0.5)
    #     out = visualizer.draw_dataset_dict(_img_)
    #     Image.fromarray(out.get_image()[:, :, ::-1]).save(f"data/08_reporting/{_experiment_id_}_train_sample_{_img_['image_id']}.png")

    logger.debug(DatasetCatalog)

    return DatasetCatalog
