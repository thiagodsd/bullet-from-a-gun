<div align="center">

![repository cover](./docs/repository-cover.jpg)

<a href="https://universe.roboflow.com/bulletfromagun/bullets-holes-and-other-things">
    <img src="https://app.roboflow.com/images/download-dataset-badge.svg"></img>
</a>

</div>

<br/>

# Gunshot Detection in Targets: A Convolutional Neural Network Benchmark

This repository is dedicated to the comparison and benchmarking of state-of-the-art Convolutional Neural Networks (CNNs) for the purpose of detecting gunshot holes in targets. As an emerging area of interest within the fields of computer vision and object detection, the accurate identification of gunshot impacts on various targets presents unique challenges and applications, ranging from sports shooting analysis to law enforcement training enhancements.


# Objective
Our primary goal is to systematically evaluate and identify the most effective CNN models for gunshot hole detection, considering aspects such as accuracy, speed, and computational efficiency. Through rigorous testing across different datasets, including varying target materials, bullet calibers, and shooting distances, we aim to provide comprehensive insights that can guide researchers, hobbyists, and professionals in selecting or developing optimized models for similar applications.

# Methodology
`todo`

## Data Preparation
The dataset used in this project consists of images of targets with gunshot holes, captured under mainly two different conditions: in with the target is composed by concentric circles (reffered as "circle" in the current work) and the other with the target is composed by concentric rounded-corner like squares (reffered as "vertical" in the current work).

The first step in the data preparation process was to annotate the gunshot holes in the images. This was done using the [Supervisely](https://supervise.ly/) platform, which allows for the creation of custom object detection datasets. The annotated dataset was then exported in both formats, YOLOv5 and COCO, to facilitate the training of different CNN models.

# Development Notes


# References
- [https://arxiv.org/abs/1807.05511](https://arxiv.org/abs/1807.05511)
- [https://www.sciencedirect.com/science/article/abs/pii/S1051200422004298](https://www.sciencedirect.com/science/article/abs/pii/S1051200422004298)

# License
This project is licensed under the MIT License - see the LICENSE file for details.