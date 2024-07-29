"""
This is a boilerplate pipeline 'yolo'
generated using Kedro 0.19.5
"""
import logging
import os
import random
import re
import zipfile
from typing import Union

import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def hyperparameter_tuning_yolo(
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

    results = dict()

    model = YOLO(
        fine_tuning_params["model_name"].replace(".pt", ".yaml")
    ).load(
        fine_tuning_params["model_name"]
    )

    model.to("cuda")
    logger.debug(torch.cuda.is_available())
    logger.debug(torch.cuda.current_device())
    logger.debug(torch.cuda.get_device_name())
    logger.debug(torch.cuda.memory_allocated())
    logger.debug(torch.cuda.memory_reserved())
    logger.debug(torch.cuda.memory_summary())

    _experiment_id_ = dataprep_params['experiment_id']
    _output_path_ = os.path.join(*fine_tuning_params["path"])
    _yolo_data_ = os.path.join(*dataprep_params['yolo_data']['path'])
    _yolo_conf_data_ = os.path.join(_yolo_data_, 'data.yaml')

    logger.debug(_experiment_id_)
    logger.debug(_yolo_data_)
    logger.debug(_output_path_)

    model.tune(
        data = _yolo_conf_data_,
        project = os.path.join(_output_path_, _experiment_id_),
        epochs = 25,
        iterations = 50,
        optimizer='AdamW',
        batch = 32,
        plots = True,
        save = True,
        val = True,
        device = 0,
        exist_ok = True,
        single_cls = True,
        amp = False
    )

    return results


def fine_tune_yolo(
    dataprep_params: dict,
    fine_tuning_params: dict,
    hyperparameter_tuning_results: dict
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

    model = YOLO(
        fine_tuning_params["model_name"].replace(".pt", ".yaml")
    ).load(
        fine_tuning_params["model_name"]
    )

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
        optimizer = fine_tuning_params["model_config"]["optimizer"],
        lr0 = fine_tuning_params["model_config"]["lr0"],
        lrf = fine_tuning_params["model_config"]["lrf"],
        # rect = fine_tuning_params["model_config"]["rect"],
        # iou = fine_tuning_params["model_config"]["iou"],
        device = 0,
        exist_ok = True,
        save = True,
        val = True,
        cache = True,
        single_cls = True,
        amp = False,
    )

    return results


def evaluate_yolo(
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
    _output_path_ = os.path.join(*fine_tuning_params["path"])
    _yolo_data_ = os.path.join(*dataprep_params['yolo_data']['path'])
    _yolo_conf_data_ = os.path.join(_yolo_data_, 'data.yaml')

    torch.cuda.set_device(0)
    torch.cuda.empty_cache()

    model = YOLO(os.path.join(_output_path_, _experiment_id_, "weights", "best.pt"))

    model.to("cuda")
    logger.debug(torch.cuda.is_available())
    logger.debug(torch.cuda.current_device())
    logger.debug(torch.cuda.get_device_name())
    logger.debug(torch.cuda.memory_allocated())
    logger.debug(torch.cuda.memory_reserved())
    logger.debug(torch.cuda.memory_summary())

    logger.debug(_experiment_id_)
    logger.debug(_yolo_data_)
    logger.debug(_output_path_)

    for _SET_ in ["train", "val", "test"]:
        model.val(
            data = _yolo_conf_data_,
            batch = fine_tuning_params["model_config"]["batch"],
            imgsz = fine_tuning_params["model_config"]["img_size"],
            iou = fine_tuning_params["model_config"]["iou"],
            project = _output_path_,
            name = _experiment_id_,
            save_json = True,
            exist_ok = True,
            plots = True,
            split = _SET_,
        )

    return results, plots


def compress_results(
        dataprep_params: dict,
        fine_tuning_params: dict,
        evaluation_results: dict,
        # evaluation_plots: dict,
    ) -> None:
    """
    `todo` documentation.
    """
    _ = evaluation_results
    # _ = evaluation_plots
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    np.set_printoptions(precision=5)

    _experiment_id_ = dataprep_params['experiment_id']
    _output_path_ = os.path.join(*fine_tuning_params["path"])
    _yolo_data_ = os.path.join(*dataprep_params['yolo_data']['path'])
    _yolo_conf_data_ = os.path.join(_yolo_data_, 'data.yaml')

    _metadata_ = {
        "experiment_id": _experiment_id_,
        "yolo_data": _yolo_data_,
        "output_path": _output_path_,
        "files" : {
            "dataset": os.path.join("data/05_model_input/gunshots/coco/v1", "README.roboflow.txt"),
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

    with open(_yolo_conf_data_) as f:
        _yolo_conf_data_ = f.read()

    with zipfile.ZipFile(f"{_experiment_id_}.zip", "w") as zipf:
        for folder in [
            os.path.join("data", "06_models"),
        ]:
            for root, _, files in os.walk(folder):
                for file in files:
                    if np.all([
                        not file.endswith(".pth"),
                        not file.endswith(".pt"),
                        "tfevents" not in file,
                        "last_checkpoint" not in file,
                    ]):
                        if _experiment_id_ in root or _experiment_id_ in file:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, os.path.dirname(folder))
                            zipf.write(file_path, arcname)

        readme_content = f"""Experiment
==============================
{_experiment_id_}

{_dataset_metadata_}

{_yolo_conf_data_}

Tuning, Results & Reports
==============================
+ 06_models/output/{_experiment_id_}, contains all information about the fine-tuned model and the results of the evaluation
        """
        zipf.writestr("README.txt", readme_content)
