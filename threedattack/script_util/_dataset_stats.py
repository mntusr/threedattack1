from dataclasses import dataclass
from typing import NamedTuple, TypeVar

import numpy as np

from ..dataset_model import DatasetLike, SamplesBase, SampleType, SampleTypeError

T = TypeVar("T", bound=SamplesBase, covariant=True)


def calculate_dataset_depth_stats(
    dataset: DatasetLike[T], sample_type: SampleType
) -> "DepthStats":
    """
    Get the mean and the median of the following statistics of the individual depth maps in the dataset:

    * Minimum depth
    * Median depth
    * Mean depth
    * Maximum depth

    Parameters
    ----------
    dataset
        The dataset on which these statistics should be calculated.
    sample_type
        The type of the samples in the dataset on which these statistics should be calculated.

    Returns
    -------
    v
        The calculated statistics.

    Raises
    ------
    SampleTypeError
        If the dataset does not contain any sample for the specified type.
    """
    sample_count = dataset.get_n_samples().get_n_samples_by_type(sample_type)

    min_depth_list: list[float] = []
    max_depth_list: list[float] = []
    median_depth_list: list[float] = []
    mean_depth_list: list[float] = []

    for sample_idx in range(sample_count):
        sample = dataset.get_sample(sample_idx, sample_type)
        valid_depths = sample.rgbds.depths[sample.rgbds.masks]

        min_depth_list.append(float(valid_depths.min()))
        max_depth_list.append(float(valid_depths.max()))
        median_depth_list.append(float(np.median(valid_depths)))
        mean_depth_list.append(float(np.mean(valid_depths)))

    return DepthStats(
        min_depths=min_depth_list,
        max_depths=max_depth_list,
        mean_depths=mean_depth_list,
        median_depths=median_depth_list,
    )


@dataclass
class DepthStats:
    min_depths: list[float]
    max_depths: list[float]
    mean_depths: list[float]
    median_depths: list[float]
