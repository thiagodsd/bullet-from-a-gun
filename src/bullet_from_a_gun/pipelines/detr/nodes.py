"""
This is a boilerplate pipeline 'detr'
generated using Kedro 0.19.5
"""
import logging
import os
import random
import sys
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor
from transformers import DetrForObjectDetection, DetrImageProcessor
from pytorch_lightning.loggers import TensorBoardLogger


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
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
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
        default_root_dir = os.path.join(_output_path_, _experiment_id_, "logs"),
        callbacks = [checkpoint_callback],
        logger = logger_tensorboard,
    )

    trainer.fit(model, train_loader, val_loader)

    # save the model
    os.makedirs(_output_path_, exist_ok=True)
    torch.save(
        model.state_dict(),
        os.path.join(_output_path_, _experiment_id_, "model_final.pth")
    )

    return results
