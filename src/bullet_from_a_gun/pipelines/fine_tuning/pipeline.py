"""
This is a boilerplate pipeline 'fine_tuning'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, node

from .nodes import fine_tune_yolo_model


def create_pipeline(**kwargs):
    _ = kwargs
    return Pipeline(
        [
            node(
                func=fine_tune_yolo_model,
                inputs=[
                    "params:circle_yolo_exp_1"
                ],
                outputs="status",
                name="fine_tune_yolo_model"
            )
        ]
    )
