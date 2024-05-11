"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from .pipelines.data_preparation import create_pipeline as data_preparation_pipeline
from .pipelines.fine_tuning import create_pipeline as fine_tuning_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_pipeline = data_preparation_pipeline()
    fine_tuning = fine_tuning_pipeline()
    return {
        "__default__"      : data_pipeline + fine_tuning,
        "data_preparation" : data_pipeline,
        "fine_tuning"      : fine_tuning,
    }
