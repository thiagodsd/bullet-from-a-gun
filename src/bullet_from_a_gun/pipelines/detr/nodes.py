"""
This is a boilerplate pipeline 'detr'
generated using Kedro 0.19.5
"""
import io
import logging
import os
import random
import sys
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv  # noqa: F401
import torch
import torchvision
import tqdm
from coco_eval import CocoEvaluator
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CustomCocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        image_directory_path: str,
        image_processor,
        train: bool = True
    ):
        ANNOTATION_FILE_NAME = "_annotations.coco.json"
        annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
        super().__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super().__getitem__(idx)
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target


class Detr(LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels = 2,
            ignore_mismatched_sizes=True
        )
        self.image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss
        loss_dict = outputs.loss_dict
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())
        return loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad], "lr": self.lr_backbone},
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)


def fine_tune_detr(  # noqa: PLR0915
        dataprep_params: dict,
        fine_tuning_params: dict
    ) -> dict:
    """
    `todo` documentation.
    """
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    np.set_printoptions(precision=5)

    results = dict()

    _device_ = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    _experiment_id_ = dataprep_params['experiment_id']
    _coco_path_ = os.path.join(*dataprep_params['coco_data']['path'])
    _output_path_ = os.path.join(*fine_tuning_params["path"])

    results[_experiment_id_] = dataprep_params.copy()
    results[_experiment_id_].update(fine_tuning_params)

    torch.cuda.empty_cache()

    image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    def collate_fn(batch):
        """
        `todo` documentation.
        """
        pixel_values = [item[0] for item in batch]
        encoding = image_processor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        return {
            'pixel_values': encoding['pixel_values'],
            'pixel_mask': encoding['pixel_mask'],
            'labels': labels
        }

    # create the datasets
    train_dataset = CustomCocoDetection(
        os.path.join(_coco_path_, "train"),
        image_processor,
        train=True
    )
    val_dataset = CustomCocoDetection(
        os.path.join(_coco_path_, "valid"),
        image_processor,
        train=False
    )

    # create the data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size = fine_tuning_params["batch_size"],
        shuffle = True,
        num_workers = fine_tuning_params["num_workers"],
        collate_fn = collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size = fine_tuning_params["batch_size"],
        shuffle = False,
        num_workers = fine_tuning_params["num_workers"],
        collate_fn = collate_fn
    )

    # create the model
    model = Detr(
        lr = fine_tuning_params["lr"],
        lr_backbone = fine_tuning_params["lr_backbone"],
        weight_decay = fine_tuning_params["weight_decay"]
    )
    model.to(_device_)

    checkpoint_callback = ModelCheckpoint(
        dirpath = os.path.join(_output_path_, _experiment_id_, "checkpoints"),
        filename = "{epoch:02d}-{validation_loss:.2f}",
        save_top_k = 3,
        monitor = "validation_loss",
        mode = "min",
    )

    logger_tensorboard = TensorBoardLogger(
        save_dir = os.path.join(_output_path_, _experiment_id_, "logs"),
        name = _experiment_id_,
    )

    logger.debug(torch.cuda.is_available())
    logger.debug(torch.cuda.current_device())
    logger.debug(torch.cuda.get_device_name())
    logger.debug(torch.cuda.memory_allocated())
    logger.debug(torch.cuda.memory_reserved())
    logger.debug(torch.cuda.memory_summary())

    trainer = Trainer(
        devices = 1,
        accelerator = "gpu",
        max_epochs = fine_tuning_params["epochs"],
        gradient_clip_val = 0.1,
        accumulate_grad_batches = 8,
        log_every_n_steps = 5,
        default_root_dir = os.path.join(_output_path_, _experiment_id_),
        callbacks = [checkpoint_callback],
        logger = logger_tensorboard,
    )

    trainer.fit(model, train_loader, val_loader)

    # save the model
    os.makedirs(_output_path_, exist_ok=True)
    torch.save(
        model.state_dict(),
        os.path.join(_output_path_, _experiment_id_, "state_dict.pth")
    )
    model.model.save_pretrained(os.path.join(_output_path_, _experiment_id_))

    return results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results


def evaluate_detr(  # noqa: PLR0915
        dataprep_params: dict,
        fine_tuning_params: dict,
        fine_tuning_results: dict
    ) -> Union[dict, dict]:
    """
    `todo` documentation.
    """
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    np.set_printoptions(precision=5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    plots = dict()
    results = dict()

    _experiment_id_ = dataprep_params['experiment_id']
    _coco_path_ = os.path.join(*dataprep_params['coco_data']['path'])
    _output_path_ = os.path.join(*fine_tuning_params["path"])

    if _experiment_id_ not in results:
        results[_experiment_id_] = dict()
    if _experiment_id_ not in plots:
        plots[_experiment_id_] = dict()

    torch.cuda.empty_cache()

    # loading image processor
    image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    def collate_fn(batch):
        """
        `todo` documentation.
        """
        pixel_values = [item[0] for item in batch]
        encoding = image_processor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        return {
            'pixel_values': encoding['pixel_values'],
            'pixel_mask': encoding['pixel_mask'],
            'labels': labels
        }

    # loading model
    model = DetrForObjectDetection.from_pretrained(
        os.path.join(_output_path_, _experiment_id_),
    )
    model.to(device)
    model.eval()

    logger.debug(torch.cuda.is_available())
    logger.debug(torch.cuda.current_device())
    logger.debug(torch.cuda.get_device_name())
    logger.debug(torch.cuda.memory_allocated())
    logger.debug(torch.cuda.memory_reserved())
    logger.debug(torch.cuda.memory_summary())

    for _SET_ in ["train", "valid", "test"]:
        logger.debug(f"Running evaluation on {_SET_} set")
        os.makedirs(f"data/08_reporting/{_experiment_id_}/{_SET_}", exist_ok=True)
        # create the datasets
        _dataset_ = CustomCocoDetection(
            os.path.join(_coco_path_, _SET_),
            image_processor,
            train=True
        )
        # create the data loaders
        _loader_ = DataLoader(
            _dataset_,
            batch_size = fine_tuning_params["batch_size"],
            shuffle = False,
            num_workers = fine_tuning_params["num_workers"],
            collate_fn = collate_fn
        )
        # generating 5 random samples
        _images_ids_ = _dataset_.coco.getImgIds()
        # _categories_ = _dataset_.coco.cats
        # _category_ids_ = {k: v["id"] for k, v in _categories_.items()}
        # _box_annotator_ = sv.BoxAnnotator()
        for sampler_counter in range(5):
            _id_ = random.choice(_images_ids_)
            logger.debug(f"image id: {_id_}")
            _image_ = _dataset_.coco.loadImgs(_id_)[0]
            # _image_annotations_ = _dataset_.coco.imgToAnns[_id_]
            _image_path_ = os.path.join(_dataset_.root, _image_["file_name"])
            _im_ = cv2.imread(_image_path_)
            # _detections_ = sv.Detections.from_coco_annotations(_image_annotations_) # bug
            # _labels_ = [f"{_categories_[class_id]}" for _,_,class_id,_ in _detections_]
            # _frame_ = _box_annotator_.annotate(
            #     scene=_im_.copy(),
            #     detections=_detections_,
            #     labels=_labels_,
            # )
            # Image.fromarray(
            #     cv2.cvtColor(_frame_, cv2.COLOR_BGR2RGB)
            # ).save(f"data/08_reporting/{_experiment_id_}/{_SET_}/prediction_sample_{sampler_counter}.png")
            with torch.no_grad():
                _inputs_ = image_processor(images=_im_, return_tensors="pt").to(device)
                _outputs_ = model(pixel_values=_inputs_.pixel_values, pixel_mask=None)
                _target_sizes_ = torch.tensor([_im_.shape[:2]]).to(device)
                _results_ = image_processor.post_process_object_detection(
                    _outputs_,
                    target_sizes = _target_sizes_,
                    threshold=0.33
                )[0]
            # logger.debug(_results_)
            # _detections_ = sv.Detections.from_transformers(_results_).with_nms(threshold=0.5)
            def save_results(
                    image,
                    score,
                    label,
                    box
                ):
                # logger.debug(score)
                # logger.debug(label)
                # logger.debug(box)
                _, ax = plt.subplots(1, 1, figsize=(10, 10))
                ax.imshow(image)
                for score_, label_, (x0, y0, x1, y1) in zip(score.tolist(), label.tolist(), box.tolist()):
                    ax.add_patch(
                        plt.Rectangle(
                            (x0, y0),
                            x1 - x0,
                            y1 - y0,
                            fill = False,
                            edgecolor='red',
                            lw=3
                        )
                    )
                    # ax.text(x0, y0, f"{_categories_[label_]}: {score_:.2f}", color='red', fontsize=15)
                    ax.text(x0, y0, f"{label_}: {score_:.2f}", color='red', fontsize=15)
                plt.axis('off')
                plt.savefig(f"data/08_reporting/{_experiment_id_}/{_SET_}/prediction_sample_{sampler_counter}.png")
            # save_results(
            #     image = _im_,
            #     score = _results_.confidence,
            #     label = _results_.class_id,
            #     box = _results_.xyxy
            # )
            save_results(
                image = _im_,
                score = _results_['scores'],
                label = _results_['labels'],
                box = _results_['boxes']
            )


        # create the evaluator
        evaluator = CocoEvaluator(
            coco_gt=_dataset_.coco,
            iou_types=["bbox"],
        )

        for _, batch in enumerate(tqdm.tqdm(_loader_)):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            with torch.no_grad():
                outputs = model(
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                )
            original_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
            _results_ = image_processor.post_process_object_detection(
                outputs,
                target_sizes = original_target_sizes,
                threshold=0
            )

            predictions = {
                target["image_id"].item(): output for target, output in zip(labels, _results_)
            }
            predictions = prepare_for_coco_detection(predictions)
            evaluator.update(predictions)

        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        # evaluator.summarize()

        # handling summary output
        _output_ = io.StringIO()
        sys.stdout = _output_
        for _, _eval_ in evaluator.coco_eval.items():
            _eval_.summarize()
        sys.stdout = sys.__stdout__
        results[_experiment_id_][_SET_] = f"""{_output_.getvalue()}"""
        logger.debug(f"""{_output_.getvalue()}""")

        # # confusion matrix
        # TODO

    return results, plots
