"""
This is a boilerplate pipeline 'data_preparation'
generated using Kedro 0.19.5
"""
from kedro.pipeline import Pipeline, node

from .nodes import load_images, load_json_annotations


def create_pipeline(**kwargs):
    _ = kwargs
    return Pipeline(
        [
            node(
                func=load_images,
                inputs="raw_gunshots_images",
                outputs="processed_images",
                name="process_raw_images"
            ),
            node(
                func=load_images,
                inputs="intermediate_circle_images",
                outputs="processed_circle_images",
                name="process_intermediate_circle_images"
            ),
            node(
                func=load_images,
                inputs="intermediate_vertical_images",
                outputs="processed_vertical_images",
                name="process_intermediate_vertical_images"
            ),
            node(
                func=load_json_annotations,
                inputs="primary_circle_coco_annotations",
                outputs="processed_primary_circle_coco_annotations",
                name="process_primary_circle_coco_annotations"
            ),
            node(
                func=load_images,
                inputs="primary_circle_yolov5_train_images",
                outputs="processed_primary_circle_yolov5_train_images",
                name="process_primary_circle_yolov5_train_images"
            ),
            node(
                func=load_json_annotations,
                inputs="primary_circle_yolov5_train_annotations",
                outputs="processed_primary_circle_yolov5_train_annotations",
                name="process_primary_circle_yolov5_train_annotations"
            ),
            node(
                func=load_json_annotations,
                inputs="primary_circle_yolov5_train_config",
                outputs="processed_primary_circle_yolov5_train_config",
                name="process_primary_circle_yolov5_train_config"
            ),
        ]
    )
