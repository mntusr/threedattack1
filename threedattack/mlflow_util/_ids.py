from tkinter.font import BOLD
from typing import NamedTuple


class ExperimentId(NamedTuple):
    """
    A typed alternative of the Mlflow experiment-id strings.
    """

    experiment_id: str


class RunId(NamedTuple):
    """
    A typed alternative of the Mlflow run-id strings.
    """

    run_id: str
