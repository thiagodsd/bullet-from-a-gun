"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from .pipelines.data_preparation import create_pipeline as data_preparation_pipeline
from .pipelines.fine_tuning import create_pipeline as fine_tuning_pipeline
from .pipelines.evaluation import create_pipeline as evaluation_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_preparation = data_preparation_pipeline()
    fine_tuning = fine_tuning_pipeline()
    evaluation = evaluation_pipeline()
    return {
        "__default__"      : data_preparation + fine_tuning + evaluation,
        "data_preparation" : data_preparation,
        "fine_tuning"      : fine_tuning,
        "evaluation"       : evaluation,
    }
