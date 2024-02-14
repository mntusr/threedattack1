from dataclasses import dataclass
from typing import Any

from ._ids import ExperimentId


@dataclass
class RunConfig:
    """
    The configuration of a new run in a specified existing experiment.
    """

    experiment_id: ExperimentId
    """
    The experiment id.
    """

    run_params: dict[str, Any]
    """
    The mlflow params of the new run.
    """

    run_repro_command: list[str]
    """
    The CLI command to reproduce the run.
    """
