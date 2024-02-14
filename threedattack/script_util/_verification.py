import math
from typing import TypeVar

import numpy as np

from ..dataset_model import DatasetLike, SamplesBase, SampleType, SampleTypeError
from ..losses import RawLossFn, calculate_loss_values, concat_losses, get_loss_val_mean
from ..target_model import AsyncDepthPredictor, predict_aligned

T = TypeVar("T", bound=SamplesBase, covariant=True)


def calculate_mean_losses_of_predictor_on_dataset(
    predictor: AsyncDepthPredictor,
    dataset: DatasetLike[T],
    batch_size: int,
    sample_types_and_fns: set[tuple[SampleType, RawLossFn]],
) -> dict[tuple[SampleType, RawLossFn], float]:
    """
    Calculate the mean of different losses of the specified predictor on the specified dataset.

    This function reports the calculation progress to the console.

    Parameters
    ----------
    predictor
        The depth predictor to evaluate.
    dataset
        The dataset on which the losses should be calculated.
    batch_size
        The batch size during loss calculation.
    sample_types_and_fns
        The types of the samles and the loss functions to calculate.

    Returns
    -------
    v
        The calculated means for each loss function and sample type.

    Raises
    ------
    SampleTypeError
        If the function should calculate the loss functions for sample types not contained by the dataset.
    ValueError
        If the batch size is non-positive.
    """
    mean_losses: dict[tuple[SampleType, RawLossFn], float] = dict()
    for sample_type in SampleType:
        loss_fns = {
            curr_loss_fn
            for curr_sample_type, curr_loss_fn in sample_types_and_fns
            if curr_sample_type == sample_type
        }
        if len(loss_fns) > 0:
            mean_losses_for_sample_type = (
                _calculate_mean_losses_of_predictor_on_dataset_for_sample_type(
                    batch_size=batch_size,
                    dataset=dataset,
                    predictor=predictor,
                    sample_type=sample_type,
                    loss_functions=loss_fns,
                )
            )
            mean_losses = mean_losses | mean_losses_for_sample_type

    return mean_losses


def _calculate_mean_losses_of_predictor_on_dataset_for_sample_type(
    predictor: AsyncDepthPredictor,
    dataset: DatasetLike[T],
    batch_size: int,
    sample_type: SampleType,
    loss_functions: set[RawLossFn],
) -> dict[tuple[SampleType, RawLossFn], float]:
    """
    Calculate the mean of different losses of the specified predictor on the specified samples in the specified dataset.

    This function reports the calculation progress to the console.

    Parameters
    ----------
    predictor
        The depth predictor to evaluate.
    dataset
        The dataset on which the losses should be calculated.
    batch_size
        The batch size during loss calculation.
    sample_type
        The type of the samples on which the losses should be calculated.
    loss_functions
        The loss functions to calculate.

    Returns
    -------
    v
        The calculated means for each loss function and sample type.

    Raises
    ------
    SampleTypeError
        If the function should calculate the loss functions for sample types not contained by the dataset.
    ValueError
        If the batch size is non-positive.
    """
    n_samples = dataset.get_n_samples().get_n_samples_by_type(sample_type)
    if n_samples == 0:
        raise SampleTypeError(
            f'The dataset does not contain any sample for sample type "{sample_type.public_name}".'
        )

    if batch_size <= 0:
        raise ValueError(f"The batch size ({batch_size}) is non-positive.")

    losses: dict[RawLossFn, np.ndarray] | None = None
    n_batches = math.ceil(n_samples / batch_size)
    for batch_idx in range(n_batches):
        print(
            f'Processing batch {batch_idx+1}/{n_batches}, sample type: "{sample_type.public_name}"'
        )
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx + 1) * batch_size
        batch = dataset.get_samples(slice(batch_start, batch_end), sample_type)
        pred = predict_aligned(
            depth_cap=dataset.get_depth_cap(),
            images=batch.rgbds,
            predictor=predictor,
        )

        losses_for_batch = calculate_loss_values(
            pred_depths=pred,
            gt=batch.rgbds.get_depths_with_masks(),
            loss_fns=loss_functions,
            target_obj_areas=None,
        )

        if losses is None:
            losses = losses_for_batch
        else:
            losses = concat_losses(losses, losses_for_batch)

    assert losses is not None
    mean_losses = get_loss_val_mean(losses)

    mean_losses_with_type = {
        (sample_type, loss_fn): mean_loss_val
        for loss_fn, mean_loss_val in mean_losses.items()
    }

    return mean_losses_with_type
