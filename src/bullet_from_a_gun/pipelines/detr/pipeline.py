"""
This is a boilerplate pipeline 'detr'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import fine_tune_detr


def create_pipeline(**kwargs) -> Pipeline:
    """
    `todo` documentation.
    """
    _ = kwargs
    template_fine_tuning = pipeline([
        node(
            func=fine_tune_detr,
            inputs=[
                "params:dataprep_params",
                "params:fine_tuning_params",
            ],
            outputs="fine_tuning_results",
            name="fine_tune_detr",
        ),
    ])

    # gunshot :: detr :: rccn_101_conf1_v1
    detr_rccn_101_conf1_v1 = pipeline(
        pipe=template_fine_tuning,
        namespace="detr.detr_resnet_50_conf1_v1",
    )
    # kedro run -n detr.detr_resnet_50_conf1_v1.fine_tune_detr

    return detr_rccn_101_conf1_v1

