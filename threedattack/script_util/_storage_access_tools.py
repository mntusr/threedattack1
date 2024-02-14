import argparse
import re
from multiprocessing import Value
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import optuna
import PIL.Image as Image
from mlflow import MlflowClient

from threedattack import SceneConfig

from .._scene import Scene
from ..mlflow_util import RUN_DESCRIPTION_TAG, ExperimentId, RunId
from ._script_util import (
    AttackDetails,
    add_attack_details_args,
    attack_details_2_dict,
    get_attack_details_from_parsed_args,
)


def set_mlflow_run_name_as_optuna_attribute(trial: optuna.Trial, run_name: str) -> None:
    """
    Set the name of the corresponsing Mlflow run as a user attribute.

    Parameters
    ----------
    trial
        The trial to modify.
    run_name
        The name of the corresponding Mlflow run.
    """
    trial.set_user_attr("run_name", run_name)


class MlflowRunLoader:
    """
    A class to load information from the logged Mlflow runs. Useful for evaluation.

    Parameters
    ----------
    client
        The Mlflow client to use for communication.
    experiment_name
        The name of the experiment to look up the runs.

    Raises
    ------
    ValueError
        If the experiment does not exist.
    """

    def __init__(self, client: MlflowClient, experiment_name: str):
        self.__client = client
        self.__experiment_name = experiment_name
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment is None:
            raise ValueError(f'There is no experiment called "{experiment_name}".')
        self.__experiment_id = ExperimentId(experiment.experiment_id)

    @property
    def experiment_name(self) -> str:
        """
        The name of the experiment to look up the runs.
        """
        return self.__experiment_name

    def get_modified_attack_details(
        self, run_name: str, changes_dict: dict[str, Any]
    ) -> AttackDetails:
        """
        Load the specified experiment and modify it according to the dict.

        If the description of the attack does not specify the number of testing samples (this is the case for older runs) then the function treats the trial as if it was set to 200.

        The function does not check if the types of the values in the dict are valid.

        Parameters
        ----------
        run_name
            The name of the run to load modified.
        changes_dict
            The dict that specifes the changes on the loaded attack details.

        Returns
        -------
        v
            The modified attack details.

        Raises
        ------
        ValueError
            If the run does not exist or its description does not contain the proper command.
        """
        run_id = self._get_run_id_by_name(run_name)
        original_attack_details = self._get_attack_details(run_id=run_id)
        attack_details_dict = attack_details_2_dict(original_attack_details)
        for param_name, new_value in changes_dict.items():
            if param_name not in attack_details_dict.keys():
                raise ValueError(f'There is no parameter name called "{param_name}".')

            attack_details_dict[param_name] = new_value

        return AttackDetails(**attack_details_dict)

    def get_run_best_renders(self, run_name: str) -> tuple[str, Image.Image]:
        run_id = self._get_run_id_by_name(run_name)
        all_artifacts = self.__client.list_artifacts(
            run_id=run_id.run_id,
        )
        artifact_paths: list[str] = [
            artifact.path for artifact in all_artifacts if artifact
        ]
        best_rgb_artifact_path = sorted(
            [
                artifact_path
                for artifact_path in artifact_paths
                if artifact_path.startswith("best_rgb")
            ]
        )[-1]
        with TemporaryDirectory() as td:
            jpeg_path = self.__client.download_artifacts(
                run_id=run_id.run_id,
                path=best_rgb_artifact_path,
                dst_path=td,
            )
            jpeg_path = Path(jpeg_path)
            loaded_im = Image.open(jpeg_path)
            loaded_im.load()
        return (best_rgb_artifact_path, loaded_im)

    def _get_attack_details(self, run_id: RunId) -> AttackDetails:
        """
        Get the attack details for the specified Mlflow run.

        Parameters
        ----------
        run_id
            The Mlflow id of the run.

        Returns
        -------
        v
            The loaded attack details.

        Raises
        ------
        ValueError
            If the description does not contain the command.
        """
        run = self.__client.get_run(run_id.run_id)
        description: str = run.data.tags[RUN_DESCRIPTION_TAG]
        attack_details = self._description_2_attack_details(description)
        return attack_details

    def get_run_result_scene_config(self, run_name: str) -> SceneConfig:
        run_id = self._get_run_id_by_name(run_name)
        with TemporaryDirectory() as td:
            scene_path = self.__client.download_artifacts(
                run_id=run_id.run_id,
                path="best_solution.scene",
                dst_path=td,
            )
            scene_path = Path(scene_path)
            return SceneConfig.from_json(scene_path)

    def get_metrics_for_run(self, run_name: str) -> dict[str, float]:
        run_id = self._get_run_id_by_name(run_name)
        run = self.__client.get_run(run_id.run_id)
        metrics_copy: dict[str, float] = run.data.metrics.copy()
        return metrics_copy

    def _get_run_id_by_name(self, run_name: str) -> RunId:
        r"""
        Get the run id for the specified run.

        Parameters
        ----------
        run_name
            The name of the searched run.

        Returns
        -------
        v
            The Mlflow run id.

        Raises
        ------
        ValueError
            If the run was not found.

            If the run does not match the pattern "[a-zA-Z\-_0-9]+".
        """
        run_pattern = r"[a-zA-Z\-_0-9]+"
        if not re.match(run_pattern, run_name):
            raise ValueError(
                f'The run name "{run_name}" does not match the run pattern "{run_pattern}".'
            )

        runs = self.__client.search_runs(
            experiment_ids=[self.__experiment_id.experiment_id],
            filter_string=f'attributes.run_name="{run_name}"',
            max_results=1,
        )
        if len(runs) == 0:
            raise ValueError(f'The run "{run_name}" was not found.')

        return RunId(runs[0].info.run_id)

    def _description_2_attack_details(self, description: str) -> AttackDetails:
        """
        Convert the description of an Mlflow run to an attack details object.

        This function basically extracts the command from the description, then executes it. It assumes that the command starts with the word 'python' and the 'python' word and each argument and flag starts and ends with '.

        Parameters
        ----------
        description
            The description to parse.

        Returns
        -------
        v
            The parsed attack details.
        """
        args: list[str] = []

        command_start_idx = description.find("'python'")

        if command_start_idx == -1:
            raise ValueError("The description does not contain the expected command.")

        outside = True
        acc = ""
        for i in range(command_start_idx, len(description)):
            c = description[i]
            if c == "'":
                if not outside:
                    args.append(acc)
                    acc = ""
                outside = not outside
            elif c == "\n":
                break
            else:
                if not outside:
                    acc += c
        args = args[2:]

        if "--n-test-viewpoints" not in args:
            args += ["--n-test-viewpoints", "200"]

        parser = _NonExitingArgumentParser()
        add_attack_details_args(parser)
        parsed = parser.parse_args(args)
        attack_details = get_attack_details_from_parsed_args(parsed)
        return attack_details


class _NonExitingArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(
            "Invalid description. Could not parse the arguments. Argparse message: "
            + message
        )


class OptunaTrialLoader:
    """
    A class to load information from the logged Optuna trials and the corresponding Mlflow runs. Useful for evaluation.

    Parameters
    ----------
    mlflow_client
        The Mlflow client to use for communication.
    mlflow_experiment_name
        The name of the experiment to look up the runs.
    optuna_storage
        The storage of the Optuna studies.
    optuna_study_name
        The name of the Optuna study.

    Raises
    ------
    ValueError
        If the experiment does not exist.
    """

    def __init__(
        self,
        mlflow_client: MlflowClient,
        mlflow_experiment_name: str,
        optuna_study: optuna.Study,
    ):
        self.__run_loader = MlflowRunLoader(
            client=mlflow_client, experiment_name=mlflow_experiment_name
        )
        self.__study = optuna_study

    @property
    def run_loader(self) -> MlflowRunLoader:
        return self.__run_loader

    @property
    def study(self) -> optuna.Study:
        return self.__study

    def get_trial_result_scene_config(self, trial_number: int) -> SceneConfig:
        """
        Get the resulting scene of the trial at the specified number.

        Parameters
        ----------
        trial_number
            The number of the trial to load.

        Returns
        -------
        v
            The loaded scene.

        Raises
        ------
        TBD
        """
        run_name = self._get_run_name_for_trial(trial_number)

        scene = self.__run_loader.get_run_result_scene_config(run_name=run_name)
        return scene

    def get_modified_attack_details(
        self, trial_number: int, changes_dict: dict[str, Any]
    ) -> AttackDetails:
        run_name = self._get_run_name_for_trial(trial_number)
        attack_details = self.__run_loader.get_modified_attack_details(
            run_name=run_name, changes_dict=changes_dict
        )
        return attack_details

    def _get_run_name_for_trial(self, trial_number: int) -> str:
        relevant_trial_list = [
            trial for trial in self.__study.trials if trial.number == trial_number
        ]
        if len(relevant_trial_list) != 1:
            raise ValueError(
                f"The trial at number {trial_number} was not found or more than one trials exist with the same number."
            )
        trial = relevant_trial_list[0]
        run_name: str = trial.user_attrs["run_name"]
        return run_name
