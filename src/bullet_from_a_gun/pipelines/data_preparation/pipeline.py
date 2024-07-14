"""
This is a boilerplate pipeline 'data_preparation'
generated using Kedro 0.19.5
"""
from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import read_coco_data


def create_pipeline(**kwargs) -> Pipeline:
    """
    `todo` documentation.
    """
    _ = kwargs
    # return Pipeline(
    #     [
    #         node(
    #             func=load_images,
    #             inputs="raw_gunshots_images",
    #             outputs="processed_images",
    #             name="process_raw_images"
    #         ),
    #         node(
    #             func=load_images,
    #             inputs="intermediate_circle_images",
    #             outputs="processed_circle_images",
    #             name="process_intermediate_circle_images"
    #         ),
    #         node(
    #             func=load_images,
    #             inputs="intermediate_vertical_images",
    #             outputs="processed_vertical_images",
    #             name="process_intermediate_vertical_images"
    #         ),
    #         node(
    #             func=load_json_annotations,
    #             inputs="primary_circle_coco_annotations",
    #             outputs="processed_primary_circle_coco_annotations",
    #             name="process_primary_circle_coco_annotations"
    #         ),
    #         node(
    #             func=load_images,
    #             inputs="primary_circle_yolov5_train_images",
    #             outputs="processed_primary_circle_yolov5_train_images",
    #             name="process_primary_circle_yolov5_train_images"
    #         ),
    #         node(
    #             func=load_json_annotations,
    #             inputs="primary_circle_yolov5_train_annotations",
    #             outputs="processed_primary_circle_yolov5_train_annotations",
    #             name="process_primary_circle_yolov5_train_annotations"
    #         ),
    #         node(
    #             func=load_json_annotations,
    #             inputs="primary_circle_yolov5_train_config",
    #             outputs="processed_primary_circle_yolov5_train_config",
    #             name="process_primary_circle_yolov5_train_config"
    #         ),
    #     ]
    # )
    template_data_preparation = pipeline([
        node(
            func=read_coco_data,
            inputs="params:dataprep_params",
            outputs="processed_images",
            name="read_coco_data",
        ),
    ])

    # gunshot
    gunshot_data_preparation = pipeline(
        pipe=template_data_preparation,
        namespace="gunshot",
    )
    # kedro run --pipeline "data_preparation" -n "gunshot.read_coco_data"

    return gunshot_data_preparation
