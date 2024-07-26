# Gunshot Detection in Targets: An Object Detection Benchmark

This repository is dedicated to comparing and benchmarking state-of-the-art Convolutional Neural Networks (CNNs) for detecting gunshot holes in various surfaces and materials. Although many algorithms were implemented, for the purposes of this paper, only YOLOv8 was selected for further experiments.

<div align="center">

![repository cover](./docs/repository-cover-2.png)

<a href="https://app.roboflow.com/bulletfromagun/bullet-holes-other-things/overview">
    <img src="https://app.roboflow.com/images/download-dataset-badge.svg"></img>
</a>

</div>


## Objective
Our primary goal is to systematically evaluate and identify the most effective CNN models for gunshot hole detection, considering aspects such as accuracy, speed, and computational efficiency. Through rigorous testing across different datasets, including various target materials, bullet calibers, and shooting distances, we aim to provide comprehensive insights that can guide researchers, hobbyists, and professionals in selecting or developing optimized models for similar applications.

# Software and Hardware
This project was developed on a 64-bit *Ubuntu Linux 22.04* operating system, equipped with 16 GB of RAM and an 8-core *AMD Ryzen 7 3700X* processor. The graphics processing unit used was a *NVIDIA GeForce GTX 1660 Ti*. All code was implemented and executed using the *Python* programming language, version 3.10.12. Most of the deep learning models were developed using the *PyTorch* library, version 2.0.0+cu117.

## Methodology
In this section the steps taken to prepare data, train models, and evaluate the results are described.

### Data Preparation
The dataset used in this project consists of images with the presence or absence of gunshot holes, mostly sourced from YouTube videos.

The first step in the data preparation process was to annotate the gunshot holes in the images. This was done using the [Roboflow](https://roboflow.com/) platform, which allows for the creation of custom object detection datasets. During this process, it was necessary to remove some images due to the following reasons:

* Bullet holes were not visible because the destruction from gunshots was too severe.
* Removing brand names, logos, or human faces caused the image to lose visible bullet holes.
* Presence of grass, as many images were taken from a video of a shooting club, and the grass would introduce bias in the model.

Additionally, it was necessary to standardize images to generalize the dataset for a broader range of algorithms and architecture possibilities. Therefore, images were cropped in batches of 50 using the web tool [Bulk Image Crop](https://bulkimagecrop.com/), by uploading them to the tool and setting the target aspect ratio to 1:1. Then, all images were previewed, and those not cropped correctly were manually cropped using the same tool.

After this, the images were grouped into a project at Roboflow, which in this case is published as [Bullet Holes & Other Things Project](https://app.roboflow.com/bulletfromagun/bullet-holes-other-things/overview). The annotated dataset was then exported in both YOLOv8 and COCO formats to facilitate the training of different algorithms.

# Development Notes

## Experimentation
As a [Kedro](https://kedro.org/) project, this repository is structured in a modular and reproducible way. A tl;dr manner to execute an experiment is to run the following command:

```sh
kedro run --to-nodes=detectron2.rccn_101_conf1_v1.evaluate_detectron2
```

This command will execute the following steps, under the hood:

### 1. Choose an Experiment
The file `/conf/base/parameters.yml` contains the configuration for the experiments. As an example:

```yaml
detectron2:
    rccn_101_conf1_v1:
        dataprep_params:
            experiment_id: "detectron2_rccn_101_conf1_v1"
            coco_data:
                path:
                    - data
                    - 05_model_input
                    - gunshots
                    - coco
                    - v1
                datasets:
                    - train
                    - valid
                    - test
        fine_tuning_params:
            path:
                - data
                - 06_models
                - output
            pretrained_model_config: COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml
            num_workers: 2
            pretrained_model_weights: COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml
            ims_per_batch: 2
            base_lr: 0.00125
            max_iter: 256
            steps: []
            batch_size_per_image: 512
            num_classes: 2
            score_thresh_test: 0.5
```

Here, `detectron2` is the object detection framework, `rccn_101_conf1_v1` is the experiment ID, in which
* `rcnn_101` identifies the model
* `conf1` is the configuration version
* `v1` is the dataset version

The `dataprep_params` section contains the parameters for the dataset preparation, and the `fine_tuning_params` section contains the parameters for the model fine-tuning.

### 2. Run the Experiment
To run the experiment, execute the following command:

```sh
kedro run -n detectron2.rccn_101_conf1_v1.fine_tune_detectron2
```

The data, generally at the `data/05_model_input` folder, will be prepared and the model will be fine-tuned. The results will be saved at the `data/06_models/output/experiment_id` folder.

### 3. Evaluate the Experiment

There is the option to visualize the results of the experiment via tensorboard:

```sh
tensorboard --logdir="data/06_models/output/detectron2_rccn_101_conf1_v1"
```

or to evaluate the model using the `detectron2` evaluation script:

```sh
kedro run -n detectron2.rccn_101_conf1_v1.fine_tune_detectron2
```

The evaluation results will be saved at the `data/06_models/output/experiment_id/evaluation` and `data/07_model_output/experiment_id` folders.

## Notes on Object Detection Models

- Region Proposal Methods
    - Selective Search
    - Edge Boxes
    - Region Proposal Networks (RPN)
    - Superpixels

---

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

# References
- [https://arxiv.org/abs/1807.05511](https://arxiv.org/abs/1807.05511)
- [https://www.sciencedirect.com/science/article/abs/pii/S1051200422004298](https://www.sciencedirect.com/science/article/abs/pii/S1051200422004298)
- [Object Detection in 20 Years: A Survey](https://arxiv.org/abs/1905.05055)
-  https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb
-  https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETA/Fine_tuning_DETA_on_a_custom_dataset_(balloon).ipynb
-  https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-huggingface-detr-on-custom-dataset.ipynb?ref=blog.roboflow.com#scrollTo=jbzTzHJW22up

# License
This project is licensed under the MIT License - see the LICENSE file for details.