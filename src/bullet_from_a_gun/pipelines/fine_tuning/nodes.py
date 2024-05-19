"""
This is a boilerplate pipeline 'fine_tuning'
generated using Kedro 0.19.5
"""

import logging
import os

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from sklearn.metrics import average_precision_score, precision_score, recall_score
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from torchvision.io.image import read_image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    SSD300_VGG16_Weights,
    fasterrcnn_resnet50_fpn_v2,
    ssd300_vgg16,
)
from torchvision.transforms import functional as F
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomCocoDataset(VisionDataset):
    """
    Custom dataset class for COCO dataset.
    """
    def __init__(
            self,
            root,
            annFile,
            transform=None,
            target_transform=None
        ):
        super(CustomCocoDataset, self).__init__(  # noqa: UP008
            root,
            transform=transform,
            target_transform=target_transform
        )
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        num_objs = len(anns)

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for i in range(num_objs):
            xmin = anns[i]['bbox'][0]
            ymin = anns[i]['bbox'][1]
            xmax = xmin + anns[i]['bbox'][2]
            ymax = ymin + anns[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(anns[i]['category_id'])
            areas.append(anns[i]['area'])
            iscrowd.append(anns[i]['iscrowd'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])
        target["area"] = areas
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)


def run_yolo_model(
        train_config: dict
    ):
    """
    Fine-tune a pre-trained YOLOv5 model on custom data.
    """
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    model = YOLO(train_config["model_name"])
    model.to("cuda")
    logger.info(torch.cuda.is_available())
    logger.info(torch.cuda.current_device())
    logger.info(torch.cuda.get_device_name())
    logger.info(torch.cuda.memory_allocated())
    logger.info(torch.cuda.memory_reserved())
    logger.info(torch.cuda.memory_summary())

    model.train(
        name=train_config["experiment_name"],
        data=train_config["model_config"]["data"],
        epochs=train_config["model_config"]["epochs"],
        batch=train_config["model_config"]["batch"],
        imgsz=train_config["model_config"]["img_size"],
        plots=True,
        save=True,
        val=True,
        exist_ok=True,
        seed=0,
        cache=True,
        # single_cls=True,
        # iterations=3,
        device=0,
        cfg=train_config["model_config"]["cfg"],
    )
    return "Model trained and saved successfully"


def run_torchvision_model(
        train_config: dict,
        train_images: str,
        train_annotations: str
    ):
    """
    Fine-tune a pre-trained torchvision model on custom data.
    """
    # check my cuda
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    logger.info(torch.cuda.is_available())
    logger.info(torch.cuda.current_device())
    logger.info(torch.cuda.get_device_name())
    logger.info(torch.cuda.memory_allocated())
    logger.info(torch.cuda.memory_reserved())
    logger.info(torch.cuda.memory_summary())

    # adjusting keys
    _train_images_ = dict()
    _train_annotations_ = dict()
    for key in train_images:
        _train_images_[key.split("_")[-1]] = train_images[key]
    for key in train_annotations:
        _train_annotations_[key.split("_")[-1]] = train_annotations[key]

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])

    dataset = CustomCocoDataset(
        root=train_config["model_config"]["coco_images"],
        annFile=train_config["model_config"]["coco_annotations"],
        # transform=F.to_tensor
        transform=transform
    )

    train_indices = list(range(0, 20))
    val_indices = list(range(20, 25))
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["model_config"]["batch"],
        shuffle=True,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )

    if "rcnn" in train_config["experiment_name"].lower():
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(
            weights=weights,
            box_score_thresh=0.9
        )
        model.eval()
        model.to("cuda")

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=train_config["model_config"]["learning_rate"], momentum=0.9, weight_decay=0.0005)
        num_epochs = train_config["model_config"]["epochs"]

        for epoch in range(num_epochs):
            model.train()
            i = 0
            for images, targets in train_loader:
                _images_ = list(image.to("cuda") for image in images)
                _targets_ = [{k: v.to("cuda") for k, v in t.items()} for t in targets]
                loss_dict = model(_images_, _targets_)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                torch.cuda.empty_cache()

                i += 1
                if i % 10 == 0:
                    logger.info(f"Epoch {epoch}/{num_epochs}, Step {i}, Loss: {losses.item()}")

    return "Model trained and saved successfully"
