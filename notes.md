# Development Notes

## Notes on Object Detection Models

- Region Proposal Methods
    - Selective Search
    - Edge Boxes
    - Region Proposal Networks (RPN)
    - Superpixels

<br/>

- Dataset Preparation
- Neural Network Architecture Selection
  - Propósal (Two-Stage)
    - RCNN
    - Fast RCNN
    - Faster RCNN
    - Mask RCNN
    - RFCN
  - Proposal-Free (One-Stage)
    - YOLO
    - SSD
- Model Training
- Inference
- Evaluation
- Results

<br/>

YOLO vs RCNN
- YOLO is faster than RCNN
- RCNN is more accurate than YOLO
- YOLO is better for real-time applications

## Notes on Object Detection Metrics

+ IoU, intersection over union, area of overlap divided by area of union
+ AP, average precision: varying different thresholds for the IoU
+ AP50, average precision at 50% IoU
+ AP75, average precision at 75% IoU, it is more strict
+ APs, APm, APl, average precision for small, medium, large objects

In recent years, the most frequently used evaluation for detection is "Average Precision (AP)", which was originally introduced in VOC2007. AP is defined as the average detection precision under different recalls and is usually evaluated in a category-specific manner. The mean AP (mAP) averaged over all categories is typically used as the final metric of performance. To measure object localization accuracy, the Intersection over Union (IoU) between the predicted box and the ground truth is used to verify whether it is greater than a predefined threshold, such as 0.5. If it is, the object is identified as "detected"; otherwise, it is considered "missed". The 0.5-IoU mAP has then become the de facto metric for object detection​(nodes)​.

Citing/reference: `arXiv:1905.05055v3 [cs.CV] 18 Jan 2023`

## Development References
- [https://arxiv.org/abs/1807.05511](https://arxiv.org/abs/1807.05511)
- [https://www.sciencedirect.com/science/article/abs/pii/S1051200422004298](https://www.sciencedirect.com/science/article/abs/pii/S1051200422004298)
- [Object Detection in 20 Years: A Survey](https://arxiv.org/abs/1905.05055)
-  https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb
-  https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETA/Fine_tuning_DETA_on_a_custom_dataset_(balloon).ipynb
-  https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-huggingface-detr-on-custom-dataset.ipynb?ref=blog.roboflow.com#scrollTo=jbzTzHJW22up
