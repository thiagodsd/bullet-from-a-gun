"""
This is a boilerplate pipeline 'yolo'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
    compress_results,
    evaluate_yolo,
    fine_tune_yolo,
    hyperparameter_tuning_yolo,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    `todo` documentation.
    """
    _ = kwargs
    template_fine_tuning = pipeline([
        node(
            func=hyperparameter_tuning_yolo,
            inputs=[
                "params:dataprep_params",
                "params:fine_tuning_params",
            ],
            outputs="hyperparameter_tuning_results",
            name="hyperparameter_tuning_yolo",
        ),
        node(
            func=fine_tune_yolo,
            inputs=[
                "params:dataprep_params",
                "params:fine_tuning_params",
                "hyperparameter_tuning_results",
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
                "evaluation_results",
                # "evaluation_plots",
            ],
            outputs=None,
            name="compress_results_yolo",
        ),
    ])

    # gunshot :: yolov8 :: yolov8_conf1_v1
    yolov8_conf1_v1 = pipeline(
        pipe=template_fine_tuning,
        namespace="yolo.yolov8_conf1_v1",
    )
    # kedro run -n yolo.yolov8_conf1_v1.hyperparameter_tuning_yolo
    # kedro run -n yolo.yolov8_conf1_v1.fine_tune_yolo
    # kedro run -n yolo.yolov8_conf1_v1.evaluate_yolo
    # kedro run -n yolo.yolov8_conf1_v1.compress_results_yolo
    # kedro run -n yolo.yolov8_conf1_v1.fine_tune_yolo,yolo.yolov8_conf1_v1.evaluate_yolo

    # gunshot :: yolov8 :: yolov8_conf2_v1
    yolov8_conf2_v1 = pipeline(
        pipe=template_fine_tuning,
        namespace="yolo.yolov8_conf2_v1",
    )
    # kedro run -n yolo.yolov8_conf2_v1.fine_tune_yolo
    # kedro run -n yolo.yolov8_conf2_v1.evaluate_yolo
    # kedro run -n yolo.yolov8_conf2_v1.compress_results_yolo
    # kedro run -n yolo.yolov8_conf2_v1.fine_tune_yolo,yolo.yolov8_conf2_v1.evaluate_yolo

    # gunshot :: yolov8 :: yolov8_conf3_v1
    yolov8_conf3_v1 = pipeline(
        pipe=template_fine_tuning,
        namespace="yolo.yolov8_conf3_v1",
    )
    # kedro run -n yolo.yolov8_conf3_v1.fine_tune_yolo
    # kedro run -n yolo.yolov8_conf3_v1.evaluate_yolo
    # kedro run -n yolo.yolov8_conf3_v1.compress_results_yolo
    # kedro run -n yolo.yolov8_conf3_v1.fine_tune_yolo,yolo.yolov8_conf3_v1.evaluate_yolo

    # gunshot :: yolov8 :: yolov8_conf4_v1
    yolov8_conf4_v1 = pipeline(
        pipe=template_fine_tuning,
        namespace="yolo.yolov8_conf4_v1",
    )
    # kedro run -n yolo.yolov8_conf4_v1.fine_tune_yolo
    # kedro run -n yolo.yolov8_conf4_v1.evaluate_yolo
    # kedro run -n yolo.yolov8_conf4_v1.compress_results_yolo
    # kedro run -n yolo.yolov8_conf4_v1.fine_tune_yolo,yolo.yolov8_conf4_v1.evaluate_yolo

    # gunshot :: yolov8 :: yolov8_conf5_v1
    yolov8_conf5_v1 = pipeline(
        pipe=template_fine_tuning,
        namespace="yolo.yolov8_conf5_v1",
    )
    # kedro run -n yolo.yolov8_conf5_v1.fine_tune_yolo
    # kedro run -n yolo.yolov8_conf5_v1.evaluate_yolo
    # kedro run -n yolo.yolov8_conf5_v1.compress_results_yolo
    # kedro run -n yolo.yolov8_conf5_v1.fine_tune_yolo,yolo.yolov8_conf5_v1.evaluate_yolo

    # gunshot :: yolov8 :: yolov8_conf6_v1
    yolov8_conf6_v1 = pipeline(
        pipe=template_fine_tuning,
        namespace="yolo.yolov8_conf6_v1",
    )
    # kedro run -n yolo.yolov8_conf6_v1.fine_tune_yolo
    # kedro run -n yolo.yolov8_conf6_v1.evaluate_yolo
    # kedro run -n yolo.yolov8_conf6_v1.compress_results_yolo
    # kedro run -n yolo.yolov8_conf6_v1.fine_tune_yolo,yolo.yolov8_conf6_v1.evaluate_yolo

    # gunshot :: yolov8 :: yolov8_conf7_v1
    yolov8_conf7_v1 = pipeline(
        pipe=template_fine_tuning,
        namespace="yolo.yolov8_conf7_v1",
    )
    # kedro run -n yolo.yolov8_conf7_v1.fine_tune_yolo
    # kedro run -n yolo.yolov8_conf7_v1.evaluate_yolo
    # kedro run -n yolo.yolov8_conf7_v1.compress_results_yolo
    # kedro run -n yolo.yolov8_conf7_v1.fine_tune_yolo,yolo.yolov8_conf7_v1.evaluate_yolo

    # gunshot :: yolov8 :: yolov8_conf8_v1
    yolov8_conf8_v1 = pipeline(
        pipe=template_fine_tuning,
        namespace="yolo.yolov8_conf8_v1",
    )
    # kedro run -n yolo.yolov8_conf8_v1.hyperparameter_tuning_yolo
    # kedro run -n yolo.yolov8_conf8_v1.fine_tune_yolo
    # kedro run -n yolo.yolov8_conf8_v1.evaluate_yolo
    # kedro run -n yolo.yolov8_conf8_v1.compress_results_yolo
    # kedro run -n yolo.yolov8_conf8_v1.fine_tune_yolo,yolo.yolov8_conf8_v1.evaluate_yolo

    return yolov8_conf1_v1 \
        + yolov8_conf2_v1 \
        + yolov8_conf3_v1 \
        + yolov8_conf4_v1 \
        + yolov8_conf5_v1 \
        + yolov8_conf6_v1 \
        + yolov8_conf7_v1 \
        + yolov8_conf8_v1
