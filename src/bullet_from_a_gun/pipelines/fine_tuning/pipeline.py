"""
This is a boilerplate pipeline 'fine_tuning'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import fine_tune_detectron2


def create_pipeline(**kwargs) -> Pipeline:
    """
    `todo` documentation.
    """
    _ = kwargs
    template_fine_tuning = pipeline([
        node(
            func=fine_tune_detectron2,
            inputs=[
                "params:dataprep_params",
                "params:fine_tuning_params",
            ],
            outputs="results",
            name="fine_tune_detectron2",
        ),
    ])

    # gunshot
    gunshot_fine_tuning = pipeline(
        pipe=template_fine_tuning,
        namespace="gunshot",
    )
    # kedro run --to-nodes=gunshot.fine_tune_detectron2

    return gunshot_fine_tuning
