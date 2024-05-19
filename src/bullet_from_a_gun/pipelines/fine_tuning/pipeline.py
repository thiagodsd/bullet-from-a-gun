"""
This is a boilerplate pipeline 'fine_tuning'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, node

from .nodes import run_yolo_model, run_torchvision_model


def create_pipeline(**kwargs):
    _ = kwargs
    return Pipeline(
        [
            node(
                func=run_yolo_model,
                inputs=[
                    "params:circle_yolo_exp_1_1"
                ],
                outputs="status_yolo",
                name="run_yolo_model"
            ),
            node(
                func=run_torchvision_model,
                inputs=[
                    "params:circle_torch_exp_1_1",
                    "primary_circle_yolov5_train_images",
                    "primary_circle_yolov5_train_annotations"
                ],
                outputs="status_torch",
                name="run_torchvision_model"
            )
        ]
    )
