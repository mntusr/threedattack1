import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Sequence, cast

import matplotlib.pyplot as plt
import mlflow.entities
import numpy as np
from matplotlib.axes import Axes
from mlflow import MlflowClient

from ..dataset_model import CamProjSpec, DepthsWithMasks, RGBsWithDepthsAndMasks
from ..rendering_core import depthmaps_2_point_cloud_fig, imshow
from ..tensor_types.idx import *
from ..tensor_types.npy import *
from ._ids import ExperimentId, RunId
from ._run_config import RunConfig


class CustomMlflowClient:
    """
    A higher-level client for Mlflow.

    Parameters
    ----------
    client
        The mlflow client to use.
    """

    def __init__(self, client: MlflowClient):
        self._client = client

    def set_terminated(self, run_id: RunId):
        self._client.set_terminated(run_id=run_id.run_id)

    def log_metrics(
        self,
        run_id: RunId,
        metrics: dict[str, float],
        generation_idx: int | None,
    ) -> None:
        """
        Asynchronously log the specified metrics.

        The constraints imposed by Mlflow still apply to the names of the metrics.

        Parameters
        ----------
        run_id
            The run to which the metrics belong.
        metrics
            The dictionary of the metrics.
        generation_idx
            The index of the generation of the metrics. If it is none, then the metrics will have no specified iteration.
        """

        for metric_name, metric_value in metrics.items():
            self._client.log_metric(
                run_id=run_id.run_id,
                key=metric_name,
                step=generation_idx,
                value=metric_value,
            )

    def get_run_name(self, run_id: RunId) -> str:
        """
        Get the name of the specified run.

        If the name is None, then the function returns with a placeholder string.
        """
        run = self._client.get_run(run_id=run_id.run_id)

        run_name = run.info.run_name
        if run_name is None:
            run_name = "<not set name>"
        return run_name

    def log_npz(self, name: str, run_id: RunId, data: dict[str, np.ndarray]) -> None:
        """
        Log the specified dict of numpy arrays to an npz file artifact.

        Parameters
        ----------
        name
            The name of the npz file. Its extension should be npz. The stem can only contain English letters, digits and "-" and "_".
        run_id
            The id of the run.
        data
            The numpy data to store. There is no restriction for the format of the arrays.

        Raises
        ------
        ValueError
            If the name of the npz file is not valid.
        """
        STEM_PATTERN = r"^[a-zA-Z0-9_-]+\.npz$"

        if not re.match(STEM_PATTERN, name):
            raise ValueError(
                'The name of the artifact should only contain English letters, digits and "_" and "-"'
            )

        with TemporaryDirectory() as td:
            td_path = Path(td)
            local_path = td_path / name
            np.savez(local_path, **data)

            self._client.log_artifact(run_id=run_id.run_id, local_path=str(local_path))

    def log_text(self, name: str, run_id: RunId, data: str) -> None:
        """
        Log the specified text as an artifact.

        Parameters
        ----------
        name
            The name of the artifact file. The name can only contain English letters, exactly one dot, digits and "-" and "_".
        run_id
            The id of the run.
        data
            The string to log.

        Raises
        ------
        ValueError
            If the name of the artifact is not valid.
        """
        name_pattern = r"^[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+$"

        if not re.match(name_pattern, name):
            raise ValueError(
                'The name of the artifact should only contain English letters, digits and "_" and "-"'
            )

        self._client.log_text(run_id=run_id.run_id, artifact_file=name, text=data)

    def log_gt_depths_and_preds(
        self,
        run_id: RunId,
        rgbds: RGBsWithDepthsAndMasks,
        pred_depths_aligned: np.ndarray,
        cam_proj_spec: CamProjSpec,
        generation_idx: int,
    ):
        """
        Asynchronously log the ground truth depths and depth predictions as Plotly figure artifacts.

        This function has no effect if the number of images is 0.

        If more than depth is specified, than this function assumes that they are separate samples in the same iteration. The function identifies them with ther index. This index is contained by the name of the artifact too.

        Parameters
        ----------
        run_id
            The run to which the depth data belongs.
        rgbds
            The original rgbd images.
        pred_depths_aligned
            The predicted aligned depths. Format: ``Im::DepthMaps``
        cam_proj_spec
            The description of the projection of the depth data.
        generation_idx
            The index of the generation of the prediction and the ground truth data.

        Notes
        -----
        The artifact name formatting will be incorrect if the generation index is greater than or equal 10,000,000. The artifact name formatting will be incorrect if the number of samples in a single generation is greater than or equal 100.
        """
        n_samples = rgbds.depths.shape[DIM_IM_N]

        for sample_idx in range(n_samples):
            mask = idx_im_depthmaps(rgbds.masks, n=[sample_idx])
            pred_depth = idx_im_depthmaps(pred_depths_aligned, n=[sample_idx])
            gt_depth = idx_im_depthmaps(rgbds.depths, n=[sample_idx])
            pc_fig = depthmaps_2_point_cloud_fig(
                cam_proj_spec=cam_proj_spec,
                depths={
                    "Pred depth": DepthsWithMasks(
                        depths=pred_depth,
                        masks=mask,
                    ),
                    "GT depth": DepthsWithMasks(
                        depths=gt_depth,
                        masks=mask,
                    ),
                },
            )

            self._client.log_figure(
                run_id=run_id.run_id,
                figure=pc_fig,
                artifact_file=f"best_pointcloud_gen{generation_idx:07d}_sample{sample_idx:02d}.html",
            )

    def log_rgbs(self, run_id: RunId, rgbs: np.ndarray, generation_idx: int) -> None:
        """
        Asynchronously log the specifed RGB images into a single image artifact.

        This function has no effect if the number of images is 0.

        Parameters
        ----------
        run_id
            The run to which the depth data belongs.
        rgbds
            The original rgbd images. Format: ``Im::RGBs``
        generation_idx
            The index of the generation of the images.

        Notes
        -----
        The artifact name formatting will be incorrect if the generation index is greater than or equal 10,000,000.
        """
        n_samples = rgbs.shape[DIM_IM_N]

        if n_samples == 0:
            return

        fig, axs = plt.subplots(ncols=n_samples, nrows=1, squeeze=False)
        fig.set_size_inches(h=12, w=n_samples * 12)
        for sample_idx in range(n_samples):
            rgb = idx_im_rgbs(rgbs, n=[sample_idx])
            imshow(rgb, on=axs[0, sample_idx], show=False)
            axs[0, sample_idx].set_title("RGB " + str(sample_idx))

        self._client.log_figure(
            run_id=run_id.run_id,
            figure=fig,
            artifact_file=f"best_rgb_gen{generation_idx:06d}.png",
        )
        plt.close(fig)

    def get_experiment_id_by_name(self, experiment_name: str) -> ExperimentId:
        experiment = self._client.get_experiment_by_name(name=experiment_name)

        if experiment is None:
            raise ValueError(f'The experiment "{experiment_name}" was not found.')

        return ExperimentId(experiment.experiment_id)

    def create_run(self, run_config: RunConfig) -> RunId:
        """
        Create a new run with the specified configuration.

        The description of the new run contains the command to start this run.

        This function is synchronous.

        Parameters
        ----------
        client
            The mlflow client to use to create the new run.
        run_config
            The new run.

        Returns
        -------
        v
            The id of the new run.

        Raises
        ------
        ValueError
            If the experiment does not exist.
        mlflow.exceptions.MlflowException
            If the configuration of the new run or the client is not valid. This includes the params too.
        """
        experiment = self._client.get_experiment(run_config.experiment_id.experiment_id)
        if experiment is None:
            raise ValueError("The experiment does not exist.")

        run = self._client.create_run(
            experiment_id=experiment.experiment_id,
            tags={
                RUN_DESCRIPTION_TAG: f"Command: {_get_command_str(run_config.run_repro_command)}"
            },
        )
        for name, value in run_config.run_params.items():
            self._client.log_param(
                run_id=run.info.run_id, key=name, value=value, synchronous=True
            )
        return RunId(run.info.run_id)


def _get_command_str(command_parts: list[str]) -> str:
    return " ".join([f"'{command_part}'" for command_part in command_parts])


def _get_experiment_description(experiment: mlflow.entities.Experiment) -> str:
    """
    Get the description text of the experiment.

    Parameters
    ----------
    experiment
        The retrieved experiment.

    Returns
    -------
    v
        The description text.
    """
    return experiment.tags[_EXPERIMENT_DESCRIPTION_TAG]


def _calculate_experiment_description_text(data: dict[str, str]) -> str:
    """
    Calculate the description text of an experiment based on its configuration.

    The description text does not depend on the order of the configuration options and contains all values of the configuration options.

    Parameters
    ----------
    data
        The configuration of the experiment.

    Returns
    -------
    v
        The calculated description text.
    """
    acc = ""
    for key in sorted(data.keys()):
        value = data[key]
        acc += key + ": " + value + "\n"
    return acc


_EXPERIMENT_DESCRIPTION_TAG = "mlflow.note.content"
"""
The name of the tag that sets the description of an experiment.
"""

RUN_DESCRIPTION_TAG = "mlflow.note.content"
"""
The name of the tag that sets the description of a run.
"""
