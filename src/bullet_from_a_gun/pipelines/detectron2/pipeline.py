"""
This is a boilerplate pipeline 'detectron2'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import compress_results, evaluate_detectron2, fine_tune_detectron2


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
            outputs="fine_tuning_results",
            name="fine_tune_detectron2",
        ),
        node(
            func=evaluate_detectron2,
            inputs=[
                "params:dataprep_params",
                "params:fine_tuning_params",
                "fine_tuning_results",
            ],
            outputs=[
                "evaluation_results",
                "evaluation_plots",
            ],
            name="evaluate_detectron2",
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
            name="compress_results_detectron2",
        ),
    ])

    # gunshot :: detectron2 :: rccn_101_conf1_v1
    detectron2_rccn_101_conf1_v1 = pipeline(
        pipe=template_fine_tuning,
        namespace="detectron2.rccn_101_conf1_v1",
    )
    # kedro run -n detectron2.rccn_101_conf1_v1.fine_tune_detectron2
    # kedro run -n detectron2.rccn_101_conf1_v1.evaluate_detectron2
    # kedro run -n detectron2.rccn_101_conf1_v1.compress_results_detectron2

    # gunshot :: detectron2 :: rccn_101_conf2_v1
    detectron2_rccn_101_conf2_v1 = pipeline(
        pipe=template_fine_tuning,
        namespace="detectron2.rccn_101_conf2_v1",
    )
    # kedro run -n detectron2.rccn_101_conf2_v1.fine_tune_detectron2
    # kedro run -n detectron2.rccn_101_conf2_v1.evaluate_detectron2
    # kedro run -n detectron2.rccn_101_conf2_v1.compress_results_detectron2

    # gunshot :: detectron2 :: rccn_101_conf3_v1
    detectron2_rccn_101_conf3_v1 = pipeline(
        pipe=template_fine_tuning,
        namespace="detectron2.rccn_101_conf3_v1",
    )
    # kedro run -n detectron2.rccn_101_conf3_v1.fine_tune_detectron2
    # kedro run -n detectron2.rccn_101_conf3_v1.evaluate_detectron2
    # kedro run -n detectron2.rccn_101_conf3_v1.compress_results_detectron2

    # gunshot :: detectron2 :: rccn_101_conf4_v1
    detectron2_rccn_101_conf4_v1 = pipeline(
        pipe=template_fine_tuning,
        namespace="detectron2.rccn_101_conf4_v1",
    )
    # kedro run -n detectron2.rccn_101_conf4_v1.fine_tune_detectron2
    # kedro run -n detectron2.rccn_101_conf4_v1.evaluate_detectron2
    # kedro run -n detectron2.rccn_101_conf4_v1.compress_results_detectron2

    # gunshot :: detectron2 :: rccn_101_conf5_v1
    detectron2_rccn_101_conf5_v1 = pipeline(
        pipe=template_fine_tuning,
        namespace="detectron2.rccn_101_conf5_v1",
    )
    # kedro run -n detectron2.rccn_101_conf5_v1.fine_tune_detectron2
    # kedro run -n detectron2.rccn_101_conf5_v1.evaluate_detectron2
    # kedro run -n detectron2.rccn_101_conf5_v1.compress_results_detectron2

    # gunshot :: detectron2 :: rccn_101_conf6_v1
    detectron2_rccn_101_conf6_v1 = pipeline(
        pipe=template_fine_tuning,
        namespace="detectron2.rccn_101_conf6_v1",
    )
    # kedro run -n detectron2.rccn_101_conf6_v1.fine_tune_detectron2
    # kedro run -n detectron2.rccn_101_conf6_v1.evaluate_detectron2
    # kedro run -n detectron2.rccn_101_conf6_v1.compress_results_detectron2

    # gunshot :: detectron2 :: rccn_101_conf2_v1
    detectron2_mask_rccn_50_conf1_v1 = pipeline(
        pipe=template_fine_tuning,
        namespace="detectron2.mask_rccn_50_conf1_v1",
    )
    # kedro run -n detectron2.mask_rccn_50_conf1_v1.fine_tune_detectron2
    # kedro run -n detectron2.mask_rccn_50_conf1_v1.evaluate_detectron2
    # kedro run -n detectron2.mask_rccn_50_conf1_v1.compress_results_detectron2

    # gunshot :: detectron2 :: rccn_101_conf1_v2
    detectron2_rccn_101_conf1_v2 = pipeline(
        pipe=template_fine_tuning,
        namespace="detectron2.rccn_101_conf1_v2",
    )
    # kedro run -n detectron2.rccn_101_conf1_v2.fine_tune_detectron2
    # kedro run -n detectron2.rccn_101_conf1_v2.evaluate_detectron2
    # kedro run -n detectron2.rccn_101_conf1_v2.compress_results_detectron2

    return detectron2_rccn_101_conf1_v1 \
        + detectron2_rccn_101_conf2_v1 \
        + detectron2_rccn_101_conf3_v1 \
        + detectron2_rccn_101_conf4_v1 \
        + detectron2_rccn_101_conf5_v1 \
        + detectron2_rccn_101_conf6_v1 \
        + detectron2_mask_rccn_50_conf1_v1 \
        + detectron2_rccn_101_conf1_v2
