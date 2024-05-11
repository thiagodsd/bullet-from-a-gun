"""
This is a boilerplate pipeline 'fine_tuning'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, node

from .nodes import fine_tune_yolov5_model


def create_pipeline(**kwargs):
    _ = kwargs
    return Pipeline(
        [
            node(
                func=fine_tune_yolov5_model,
                inputs=[
                    "primary_circle_yolov5_train_images",
                    "primary_circle_yolov5_train_annotations",
                    "params:circle_yolov5_train_config"
                ],
                outputs="status",
                name="fine_tune_yolov5_model"
            )
        ]
    )
