"""
This is a boilerplate pipeline 'detectron2'
generated using Kedro 0.19.5
"""
import logging
import os
import random
import re
import zipfile
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
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
    for _dataset_ in ["train", "valid", "test"]:
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

    torch.save(
        trainer.model.state_dict(),
        os.path.join(cfg.OUTPUT_DIR, f"{_experiment_id_}_model_final.pth")
    )

    return results


def evaluate_detectron2(
        dataprep_params: dict,
        fine_tuning_params: dict,
        fine_tuning_results:dict
    ) -> Union[dict, dict]:
    """
    `todo` documentation.
    """
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    np.set_printoptions(precision=5)

    plots = dict()
    results = dict()

    _experiment_id_ = dataprep_params['experiment_id']
    _coco_path_ = os.path.join(*dataprep_params['coco_data']['path'])
    _output_path_ = os.path.join(*fine_tuning_params["path"])

    if _experiment_id_ not in results:
        results[_experiment_id_] = dict()
    if _experiment_id_ not in plots:
        plots[_experiment_id_] = dict()

    # registering datasets
    logger.info("registering datasets...")
    for _dataset_ in ["train", "valid", "test"]:
        if f"{_experiment_id_}_{_dataset_}" not in MetadataCatalog.list():
            logger.debug(f"reading {_dataset_} data")
            register_coco_instances(
                f"{_experiment_id_}_{_dataset_}",
                {},
                os.path.join(_coco_path_, _dataset_, "_annotations.coco.json"),
                os.path.join(_coco_path_, _dataset_)
            )

    # loading model
    logger.info("loading model...")
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(fine_tuning_params["pretrained_model_config"]))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = fine_tuning_params["num_classes"]
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = fine_tuning_params["score_thresh_test"]
    cfg.OUTPUT_DIR = os.path.join(_output_path_, _experiment_id_)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    predictor = DefaultPredictor(cfg)

    for _SET_ in ["train", "valid", "test"]:
        logger.debug(f"evaluating {_SET_} set...")
        os.makedirs(f"data/08_reporting/{_experiment_id_}/{_SET_}", exist_ok=True)
        # registering datasets
        datasets_dicts = load_coco_json(
            os.path.join(_coco_path_, _SET_, "_annotations.coco.json"),
            os.path.join(_coco_path_, _SET_)
        )
        # generating 5 random samples
        sample_counter = 0
        for _img_ in random.sample(datasets_dicts, 5):
            # logger.debug(_img_)
            im = cv2.imread(_img_["file_name"])
            outputs = predictor(im)
            # logger.debug(outputs)
            v = Visualizer(
                im[:, :, ::-1],
                metadata=MetadataCatalog.get(f"{_experiment_id_}_{_SET_}"),
                scale=1.2,
            )
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            Image.fromarray(
                cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
            ).save(f"data/08_reporting/{_experiment_id_}/{_SET_}/prediction_sample_{sample_counter}.png")
            sample_counter += 1
        # evaluating
        evaluator = COCOEvaluator(
            f"{_experiment_id_}_{_SET_}",
            ("bbox",),
            False,
            output_dir=cfg.OUTPUT_DIR
        )
        loader = build_detection_test_loader(
            cfg,
            f"{_experiment_id_}_{_SET_}"
        )
        _results_ = inference_on_dataset(predictor.model, loader, evaluator)
        # logger.debug(_results_)
        results[_experiment_id_][_SET_] = _results_
        # confusion matrix
        _dataset_ = sv.DetectionDataset.from_coco(
            os.path.join(_coco_path_, _SET_),
            os.path.join(_coco_path_, _SET_, "_annotations.coco.json"),
        )
        def _confusion_matrix_callback(image:np.ndarray) -> sv.Detections:
            return sv.Detections.from_detectron2(
                predictor(image)
            )
        confusion_matrix = sv.ConfusionMatrix.benchmark(
            _dataset_,
            _confusion_matrix_callback
        )
        _confusion_matrix_ = confusion_matrix.matrix
        # logger.debug(_confusion_matrix_)
        results[_experiment_id_][_SET_]["confusion_matrix"] = _confusion_matrix_.tolist()
        # plotting confusion matrix
        plots[_experiment_id_][f"{_SET_}_confusion_matrix.png"] = confusion_matrix.plot()

    return results, plots


def compress_results(
        dataprep_params: dict,
        fine_tuning_params: dict,
    ) -> None:
    """
    `todo` documentation.

    important folders:
        + dataset metadata i.e. data/05_model_input/gunshots/coco/v1/README.roboflow.txt
        + tuning metadata i.e. data/06_models/tuned/detectron2_rccn_101_conf2_v1.json
        + results metadata i.e. data/06_models/output/detectron2_rccn_101_conf2_v1
        + evaluation metadata i.e. data/06_models/eval/detectron2_rccn_101_conf2_v1.json
        + evaluation plots i.e. data/08_reporting/detectron2_rccn_101_conf2_v1 AND data/08_reporting/plots/detectron2_rccn_101_conf2_v1
    """
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    np.set_printoptions(precision=5)

    _experiment_id_ = dataprep_params['experiment_id']
    _coco_path_ = os.path.join(*dataprep_params['coco_data']['path'])
    _output_path_ = os.path.join(*fine_tuning_params["path"])

    _metadata_ = {
        "experiment_id": _experiment_id_,
        "coco_path": _coco_path_,
        "output_path": _output_path_,
        "files" : {
            "dataset": os.path.join("data/05_model_input/gunshots/coco/v1", "README.roboflow.txt"),
            "tuning": os.path.join(f"data/06_models/tuned/{_experiment_id_}.json"),
            "evaluation": os.path.join(f"data/06_models/eval/{_experiment_id_}.json"),
        },
        "folders" : {
            "results": os.path.join("data/06_models/output", _experiment_id_),
            "evaluation": os.path.join("data/06_models/eval", _experiment_id_),
            "evaluation_plots": os.path.join("data/08_reporting", _experiment_id_),
        }
    }

    # get text from README.roboflow.txt
    with open(_metadata_["files"]["dataset"]) as f:
        _dataset_metadata_ = f.read()
    exclude_section = re.compile(
        r'Roboflow is an end-to-end computer vision platform that helps you\n'
        r'\* collaborate with your team on computer vision projects\n'
        r'\* collect & organize images\n'
        r'\* understand and search unstructured image data\n'
        r'\* annotate, and create datasets\n'
        r'\* export, train, and deploy computer vision models\n'
        r'\* use active learning to improve your dataset over time\n\n'
        r'For state of the art Computer Vision training notebooks you can use with this dataset,\n'
        r'visit https:\/\/github\.com\/roboflow\/notebooks\n\n'
        r'To find over 100k other datasets and pre-trained models, visit https:\/\/universe\.roboflow\.com\n'
    )
    _dataset_metadata_ = exclude_section.sub("", _dataset_metadata_)

    with zipfile.ZipFile(f"{_experiment_id_}.zip", "w") as zipf:
        # add files, reconstructing subfolders structure, containing _experiment_id_ and excluding *.pth and *tfevents*
        for folder in [
            os.path.join("data", "05_model_input"),
            os.path.join("data", "06_models"),
            os.path.join("data", "08_reporting"),
        ]:
            for root, _, files in os.walk(folder):
                for file in files:
                    if np.all([
                        not file.endswith(".pth"),
                        "tfevents" not in file,
                        "last_checkpoint" not in file,
                    ]):
                        if _experiment_id_ in root or _experiment_id_ in file:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, os.path.dirname(folder))
                            zipf.write(file_path, arcname)

        readme_content = f"""Experiment
= = = = = = = = = =
{_experiment_id_}

{_dataset_metadata_}

Tuning
= = = = = = = = = =
+ {str(_metadata_["files"]["tuning"])}, a JSON file containing model details and hyperparameters used in setting up the model

Results
= = = = = = = = = =
+ {str(_metadata_["folders"]["results"])}, a folder containing the model training output

Evaluation
= = = = = = = = = =
+ {str(_metadata_["files"]["evaluation"])}, a JSON file containing the evaluation metrics
+ {str(_metadata_["folders"]["evaluation"])}, a folder containing sample predictions in each dataset split
+ {str(_metadata_["folders"]["evaluation_plots"])}, a folder containing plots generated during evaluation
        """

        zipf.writestr("README.txt", readme_content)

