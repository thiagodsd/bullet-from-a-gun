"""
This is a boilerplate pipeline 'fine_tuning'
generated using Kedro 0.19.5
"""
import sys  # noqa: I001
sys.path.append("src/bullet_from_a_gun/pipelines/fine_tuning")

import copy  # noqa: I001
import io
import logging
import os
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
# from coco_eval import CocoEvaluator
# from coco_utils import get_coco_api_from_dataset
# from engine import evaluate
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.metrics import average_precision_score, precision_score, recall_score  # noqa: F401
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    SSD300_VGG16_Weights,  # noqa: F401
    fasterrcnn_resnet50_fpn_v2,
    ssd300_vgg16,  # noqa: F401
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomCocoDataset(torch.utils.data.Dataset):
    """
    Custom dataset class for COCO dataset.
    """
    def __init__(self, root, annFile, transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for ann in anns:
            xmin, ymin, width, height = ann['bbox']
            xmax = xmin + width
            ymax = ymin + height
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(ann['category_id'])
            areas.append(width * height)
            iscrowd.append(ann.get('iscrowd', 0))

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

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):  # noqa: PLR0913
    """
    Train the model for one epoch.
    """
    model.train()
    for i, (images, targets) in enumerate(data_loader):
        _images_ = list(image.to(device) for image in images)
        _targets_ = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(_images_, _targets_)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if i % print_freq == 0:
            logger.info(f"Epoch: [{epoch}], Step: [{i}/{len(data_loader)}], Loss: {losses.item()}")


def get_coco_api_from_dataset(dataset):
    """
    Create a COCO API object from a dataset.
    """
    coco_ds = COCO()
    ann_id = 1
    dataset_dict = {"images": [], "categories": [], "annotations": []}
    for img_idx in range(len(dataset)):
        img, targets = dataset[img_idx]
        image_id = int(targets["image_id"].item())
        img_dict = {
            "id": image_id,
            "height": img.shape[1],
            "width": img.shape[2],
        }
        dataset_dict["images"].append(img_dict)
        bboxes = targets["boxes"]
        labels = targets["labels"]
        areas = targets["area"]
        iscrowd = targets["iscrowd"]
        for bbox, label, area, iscrowd_flag in zip(bboxes, labels, areas, iscrowd):
            ann = {
                "id": ann_id,
                "image_id": image_id,
                "bbox": bbox.tolist(),
                "category_id": int(label.item()),
                "area": float(area.item()),
                "iscrowd": int(iscrowd_flag.item())
            }
            dataset_dict["annotations"].append(ann)
            ann_id += 1
    coco_ds.dataset = dataset_dict
    coco_ds.createIndex()
    return coco_ds


class CocoEvaluator:
    def __init__(self, coco_gt, iou_types):
        if not isinstance(iou_types, (list, tuple)):
            raise TypeError(f"This constructor expects iou_types of type list or tuple, instead got {type(iou_types)}")
        self.coco_gt = copy.deepcopy(coco_gt)
        self.iou_types = iou_types
        self.coco_eval = {iou_type: COCOeval(coco_gt, iouType=iou_type) for iou_type in iou_types}
        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            with redirect_stdout(io.StringIO()):
                coco_dt = self.coco_gt.loadRes(results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]
            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(set(self.img_ids) & set(self.coco_gt.getImgIds()))
            coco_eval.evaluate()
            self.eval_imgs[iou_type].append(coco_eval.evalImgs)

    def synchronize_between_processes(self):
        pass

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        raise ValueError(f"Unknown iou type {iou_type}")

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction["boxes"]) == 0:
                continue

            boxes = prediction["boxes"]
            scores = prediction["scores"]
            labels = prediction["labels"]

            boxes = boxes.tolist()
            scores = scores.tolist()
            labels = labels.tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results


@torch.inference_mode()
def evaluate_model(model, data_loader, device):
    """
    Evaluate the model on the dataset.
    """
    model.eval()
    cpu_device = torch.device("cpu")
    coco = get_coco_api_from_dataset(data_loader.dataset)
    coco_evaluator = CocoEvaluator(coco, ["bbox"])

    for images, targets in data_loader:
        _images_ = list(img.to(device) for img in images)
        _targets_ = [{k: v.to(cpu_device) for k, v in t.items()} for t in targets]
        outputs = model(_images_)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        res = {target["image_id"].item(): output for target, output in zip(_targets_, outputs)}
        coco_evaluator.update(res)

    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator

def run_torchvision_model(train_config: dict):
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((640, 640))
    ])

    dataset = CustomCocoDataset(
        root=train_config["model_config"]["coco_images"],
        annFile=train_config["model_config"]["coco_annotations"],
        transform=transform
    )

    train_indices = list(range(20))
    eval_indices = list(range(20, 25))

    train_dataset = Subset(dataset, train_indices)
    eval_dataset = Subset(dataset, eval_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["model_config"]["batch"],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, train_config["model_config"]["num_classes"])
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=train_config["model_config"]["learning_rate"],
        momentum=0.9,
        weight_decay=0.0005
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    num_epochs = train_config["model_config"]["epochs"]

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        lr_scheduler.step()

    logger.info("Evaluating on training set...")
    evaluate_model(model, train_loader, device)
    logger.info("Evaluating on validation set...")
    evaluate_model(model, eval_loader, device)

    run_path = Path("runs") / train_config["experiment_name"]
    run_path.mkdir(parents=True, exist_ok=True)

    # torch.save(model.state_dict(), run_path / "model.pth")

    return "Model trained and evaluated successfully"


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

