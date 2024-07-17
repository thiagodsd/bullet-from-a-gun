"""
This is a boilerplate pipeline 'detectron2'
generated using Kedro 0.19.5
"""
import logging
import os
import random

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode, Visualizer
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def fine_tune_detectron2(
        dataprep_params: dict,
        fine_tuning_params: dict,
    ):
    """
    `todo` documentation.
    """
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    np.set_printoptions(precision=5)

    results = dict()

    _experiment_id_ = dataprep_params['experiment_id']

    _coco_path_ = os.path.join(*dataprep_params['coco_data']['path'])
    _output_path_ = os.path.join(*fine_tuning_params["path"])

    results[_experiment_id_] = dataprep_params.copy()
    results[_experiment_id_].update(fine_tuning_params)

    # registering datasets
    logger.info("registering datasets...")
    for _dataset_ in ["train", "valid"]:
        logger.debug(f"reading {_dataset_} data")
        register_coco_instances(
            f"{_experiment_id_}_{_dataset_}",
            {},
            os.path.join(_coco_path_, _dataset_, "_annotations.coco.json"),
            os.path.join(_coco_path_, _dataset_)
        )

    # setting up configuration
    logger.info("setting up configuration...")
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
    cfg.OUTPUT_DIR = os.path.join(_output_path_, _experiment_id_)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    results[_experiment_id_]["cfg"] = dict(cfg)

    logger.debug(results)

    # training
    logger.info("training...")
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # saving
    logger.info("saving...")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, f"{_experiment_id_}_model_final.pth")

    return results


def evaluate_detectron2(
        dataprep_params: dict,
        fine_tuning_params: dict,
        model_name:dict
    ):
    """
    `todo` documentation.
    """
    logger.debug(dataprep_params)
    logger.debug(fine_tuning_params)
    logger.debug(model_name)
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = fine_tuning_params["score_thresh_test"]
    # predictor = DefaultPredictor(cfg)

    # # sanity check (train)
    # sample_counter = 0
    # dataset_dicts = load_coco_json(
    #     os.path.join(_coco_path_, "train", "_annotations.coco.json"),
    #     os.path.join(_coco_path_, "train")
    # )
    # metadata = MetadataCatalog.get(f"{_experiment_id_}_train")
    # for _img_ in random.sample(dataset_dicts, 3):
    #     logger.debug(_img_)
    #     im = cv2.imread(_img_["file_name"])
    #     outputs = predictor(im)
    #     logger.debug(outputs)
    #     v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.2)
    #     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #     Image.fromarray(
    #         cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
    #     ).save(f"data/08_reporting/{_experiment_id_}_train_prediction_sample_{sample_counter}.png")
    #     sample_counter += 1

    # # sanity check (valid)
    # sample_counter = 0
    # dataset_dicts = load_coco_json(
    #     os.path.join(_coco_path_, "valid", "_annotations.coco.json"),
    #     os.path.join(_coco_path_, "valid")
    # )
    # metadata = MetadataCatalog.get(f"{_experiment_id_}_valid")
    # for _img_ in random.sample(dataset_dicts, 3):
    #     logger.debug(_img_)
    #     im = cv2.imread(_img_["file_name"])
    #     outputs = predictor(im)
    #     logger.debug(outputs)
    #     v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.2)
    #     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #     Image.fromarray(
    #         cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
    #     ).save(f"data/08_reporting/{_experiment_id_}_valid_prediction_sample_{sample_counter}.png")
    #     sample_counter += 1

    # evaluator = COCOEvaluator(
    #     f"{_experiment_id_}_valid",
    #     cfg,
    #     False,
    #     output_dir=cfg.OUTPUT_DIR
    # )
    # val_loader = build_detection_test_loader(cfg, f"{_experiment_id_}_valid")
    return list()
