"""
This is a boilerplate pipeline 'evaluation'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    """
    `todo` documentation.
    """
    _ = kwargs
    template_evaluation = pipeline([
        node(
            func=evaluate_model,
            inputs=[
                "params:evaluation_params",
                "raw_images"
            ],
            outputs=[
                "object_detected_metadata",
                "object_detected_images",
            ],
            name="evaluate_model",
        ),
    ])

    # proof-of-concept
    generic_evaluation = pipeline(
        pipe=template_evaluation,
        namespace="generic",
    )

    return generic_evaluation
