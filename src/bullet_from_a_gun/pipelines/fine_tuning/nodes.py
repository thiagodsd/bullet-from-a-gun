"""
This is a boilerplate pipeline 'fine_tuning'
generated using Kedro 0.19.5
"""
import logging
import os
import json
from detectron2.structures import BoxMode
import requests
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
import cv2
import random
import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from PIL import Image
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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

        filename = f"{coco_data_path.replace('/_annotations.coco.json', '')}/{image_info['file_name']}"
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


def fine_tune_detectron2(
        dataprep_params: dict,
        fine_tuning_params: dict,
    ):
    """
    `todo` documentation.
    """
    _experiment_id_ = dataprep_params['experiment_id']
    _coco_path_ = dataprep_params['coco_data']['path']

    logger.debug(dataprep_params)
    logger.debug(fine_tuning_params)

    for _dataset_ in dataprep_params['coco_data']['datasets']:
        logger.debug(f"reading {_dataset_} data")
        _annotations_ = _handle_json_annotations(f"{_coco_path_}/{_dataset_}/_annotations.coco.json")
        DatasetCatalog.register(f"{_experiment_id_}_{_dataset_}", lambda: _annotations_)
        MetadataCatalog.get(f"{_experiment_id_}_{_dataset_}").set(thing_classes=["none", "bullets"])


    cfg = get_cfg()
    config_url = model_zoo.get_config_file(fine_tuning_params["pretrained_model_config"])
    cfg.merge_from_file(config_url)

    cfg.DATASETS.TRAIN = (f"{_experiment_id_}_train",)
    cfg.DATASETS.TEST = (f"{_experiment_id_}_valid",)
    cfg.DATALOADER.NUM_WORKERS = fine_tuning_params["num_workers"]
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(fine_tuning_params["pretrained_model_weights"])
    cfg.SOLVER.IMS_PER_BATCH = fine_tuning_params["ims_per_batch"]
    cfg.SOLVER.BASE_LR = fine_tuning_params["base_lr"]
    cfg.SOLVER.MAX_ITER = fine_tuning_params["max_iter"]
    cfg.SOLVER.STEPS = fine_tuning_params["steps"]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = fine_tuning_params["batch_size_per_image"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = fine_tuning_params["num_classes"]

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = fine_tuning_params["score_thresh_test"]
    cfg.DATASETS.TEST = (f"{_experiment_id_}_test",)
    predictor = DefaultPredictor(cfg)

    # sanity check (test)
    train_metadata = MetadataCatalog.get(f"{_experiment_id_}_train")
    dataset_dicts = _handle_json_annotations(f"{dataprep_params['coco_data']['path']}/test/_annotations.coco.json")
    for _img_ in random.sample(dataset_dicts, 3):
        logger.debug(_img_)
        im = cv2.imread(_img_["file_name"])
        outputs = predictor(im)
        logger.debug(outputs)
        v = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        Image.fromarray(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)).save(f"data/08_reporting/{_experiment_id_}_test_sample_{_img_['image_id']}.png")

    return list()
