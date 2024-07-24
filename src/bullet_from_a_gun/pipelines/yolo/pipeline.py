"""
This is a boilerplate pipeline 'yolo'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import compress_results, evaluate_yolo, fine_tune_yolo


def create_pipeline(**kwargs) -> Pipeline:
    """
    `todo` documentation.
    """
    _ = kwargs
    template_fine_tuning = pipeline([
        node(
            func=fine_tune_yolo,
            inputs=[
                "params:dataprep_params",
                "params:fine_tuning_params",
            ],
            outputs="fine_tuning_results",
            name="fine_tune_yolo",
        ),
        node(
            func=evaluate_yolo,
            inputs=[
                "params:dataprep_params",
                "params:fine_tuning_params",
                "fine_tuning_results",
            ],
            outputs=[
                "evaluation_results",
                "evaluation_plots",
            ],
            name="evaluate_yolo",
        ),
        node(
            func=compress_results,
            inputs=[
                "params:dataprep_params",
                "params:fine_tuning_params",
            ],
            outputs=None,
            name="compress_results_yolo",
        ),
    ])

    # gunshot :: yolov8 :: rccn_101_conf1_v1
    yolo_rccn_101_conf1_v1 = pipeline(
        pipe=template_fine_tuning,
        namespace="yolo.yolov8_conf1_v1",
    )
    # kedro run -n yolo.yolov8_conf1_v1.fine_tune_yolo
    # kedro run -n yolo.yolov8_conf1_v1.evaluate_yolo
    # kedro run -n yolo.yolov8_conf1_v1.compress_results_yolo

    return yolo_rccn_101_conf1_v1
