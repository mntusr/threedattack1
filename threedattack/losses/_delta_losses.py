import itertools
from dataclasses import fields
from enum import Enum, auto

import numpy as np

from ..dataset_model import DepthsWithMasks, SampleType
from ..rendering_core import TwoDAreas
from ._aggregation import (
    calculate_loss_values,
    divide_losses,
    get_loss_val_mean,
    get_loss_val_median,
    get_loss_val_min,
    subtract_losses,
)
from ._loss_functions import RawLossFn


def get_aggr_delta_loss_dict_from_preds(
    gt: DepthsWithMasks,
    aligned_depth_preds: np.ndarray,
    orig_losses: dict[RawLossFn, np.ndarray],
    loss_precision: "LossPrecision",
    viewpt_type: SampleType,
    loss_fns: set[RawLossFn],
    target_obj_areas: TwoDAreas,
) -> "dict[tuple[LossDerivationMethod, LossPrecision, SampleType, RawLossFn], float]":
    """
    Calculate the different losses specified in `LossValuesForSamples`, then return with their different aggreagations.

    Parameters
    ----------
    gt
        The current ground truth depth data.
    aligned_depth_preds
        The current aligned depth predictions.
    orig_losses
        The reference loss values. Format of the values: ``Scalars::Float``
    loss_precision
        How are the losses calculated? Is the calculation exact (all samples are used) or an estimation.
    viewpt_type
        The type of viewpoints on which the losses are calculated.
    loss_fns
        The loss functions to calculate.
    target_obj_areas
        The areas of the target object on the rendered images.

    Returns
    -------
    v
        The created dictionary.

    Raises
    ------
    ValueError
        If the original losses are not subset of the losses to calculate.
    """
    orig_loss_fns = orig_losses.keys()

    if not loss_fns.issubset(orig_loss_fns):
        raise ValueError(
            f"The original losses ({orig_loss_fns}) are not subset of the losses to calculate ({loss_fns})."
        )

    new_losses = calculate_loss_values(
        pred_depths=aligned_depth_preds,
        gt=gt,
        loss_fns=loss_fns,
        target_obj_areas=target_obj_areas,
    )

    relevant_orig_losses = {
        loss_fn: loss_val
        for loss_fn, loss_val in orig_losses.items()
        if loss_fn in loss_fns
    }
    aggr_delta_losses_dict = get_aggr_delta_loss_dict_from_losses(
        new_losses=new_losses,
        orig_losses=relevant_orig_losses,
        loss_precision=loss_precision,
        viewpt_type=viewpt_type,
    )
    return aggr_delta_losses_dict


def get_aggr_delta_loss_dict_from_losses(
    new_losses: dict[RawLossFn, np.ndarray],
    orig_losses: dict[RawLossFn, np.ndarray],
    loss_precision: "LossPrecision",
    viewpt_type: SampleType,
) -> "dict[tuple[LossDerivationMethod, LossPrecision, SampleType, RawLossFn], float]":
    """
    Calculate different derivated losses from the losses ``new_losses-orig_losses`` and return it as a dictionary.

    Parameters
    ----------
    new_losses
        The new raw loss values. Format of the values: ``Scalars::Float``
    orig_losses
        The reference raw loss values. Format of the values: ``Scalars::Float``
    loss_precision
        How are the losses calculated? Is the calculation exact (all samples are used) or an estimation.
    viewpt_type
        The type of viewpoints on which the losses are calculated.

    Returns
    -------
    v
        The created dictionary.

    Raises
    ------
    ValueError
        If the set of the original loss functions is not equal to the set of the new loss functions.
    """
    delta_losses_dict = subtract_losses(new_losses, orig_losses)
    reldelta_losses_dict = divide_losses(delta_losses_dict, orig_losses)

    mean_delta_losses = get_loss_val_mean(delta_losses_dict)
    median_delta_losses = get_loss_val_median(delta_losses_dict)
    min_delta_losses = get_loss_val_min(delta_losses_dict)
    mean_reldelta_losses = get_loss_val_mean(reldelta_losses_dict)
    median_reldelta_losses = get_loss_val_median(reldelta_losses_dict)
    min_reldelta_losses = get_loss_val_min(reldelta_losses_dict)

    def update_key(
        losses_dict: dict[RawLossFn, float], deriv_method: LossDerivationMethod
    ) -> dict[tuple[LossDerivationMethod, LossPrecision, SampleType, RawLossFn], float]:
        return {
            (
                deriv_method,
                loss_precision,
                viewpt_type,
                loss_fn,
            ): losses_dict[loss_fn]
            for loss_fn in losses_dict.keys()
        }

    all_losses = (
        update_key(mean_delta_losses, LossDerivationMethod.MeanDelta)
        | update_key(median_delta_losses, LossDerivationMethod.MedianDelta)
        | update_key(min_delta_losses, LossDerivationMethod.MinDelta)
        | update_key(mean_reldelta_losses, LossDerivationMethod.MeanReldelta)
        | update_key(median_reldelta_losses, LossDerivationMethod.MedianReldelta)
        | update_key(min_reldelta_losses, LossDerivationMethod.MinReldelta)
    )

    return all_losses


def derived_loss_dict_2_str_float_dict(
    derived_loss_dict: "dict[tuple[LossDerivationMethod, LossPrecision, SampleType, RawLossFn], float]",
) -> dict[str, float]:
    """
    Convert the keys of the derived loss dictionary to string.

    Key format: ``"{estim}{agg_method}_{viewpt_type}_{loss_name}"``. Where:

    * ``estim`` is ``"estim_"`` if the losses are only estimations, otherwise empty
    * ``agg_method is the public name of the aggregation method
    * ``viewpt_type`` is the public name of the viewpoint type
    * ``loss_name`` is the public name of the loss function

    Parameters
    ----------
    derived_loss_dict
        The dictionary of the derived losses.

    Returns
    -------
    v
        The converted dictionary.
    """
    return {
        _delta_agg_loss_key_2_str(key): value
        for key, value in derived_loss_dict.items()
    }


def _delta_agg_loss_key_2_str(
    delta_agg_loss_key: "tuple[LossDerivationMethod, LossPrecision, SampleType, RawLossFn]",
) -> str:
    """
    Convert the derived loss keys to string.

    Format: ``"{estim}_{agg_method}_{viewpt_type}_{loss_name}"``. Where:

    * ``estim`` is the public name of the precision of the loss
    * ``agg_method is the public name of the aggregation method
    * ``viewpt_type`` is the public name of the viewpoint type
    * ``loss_name`` is the public name of the loss function

    Parameters
    ----------
    delta_agg_loss_key
        The aggregated delta loss key to convert.
    """
    estim = delta_agg_loss_key[1].public_name
    agg_method = delta_agg_loss_key[0].public_name
    viewpt_type = delta_agg_loss_key[2].public_name
    loss_name = delta_agg_loss_key[3].public_name

    return f"{estim}_{agg_method}_{viewpt_type}_{loss_name}"


def get_derived_loss_by_name(name: str) -> "tuple[LossDerivationMethod, RawLossFn]":
    """
    Get the derived loss by its name.

    The name format is ``{agg_method}_{loss_name}``, where:

    * ``agg_method``: The public name of any of the values of `LossDerivationMethod`
    * ``loss_name``: The public name of any of the loss functions of ``RawLossFn``

    Parameters
    ----------
    name
        The name of the derived loss.

    Returns
    -------
    v
        The derived loss.

    Raises
    ------
    ValueError
        If the derived loss was not found.
    """
    for deriv_method, loss_fn in itertools.product(LossDerivationMethod, RawLossFn):
        variant_name = deriv_method.public_name + "_" + loss_fn.public_name

        if variant_name == name:
            return (deriv_method, loss_fn)
    else:
        raise ValueError(f'Unknown loss function "{name}".')


def get_derived_loss_name(
    deriv_method: "LossDerivationMethod", loss_fn: RawLossFn
) -> str:
    """Get the name of the derived loss."""
    return f"{deriv_method.public_name}_{loss_fn.public_name}"


def get_all_derived_loss_name_list() -> list[str]:
    """
    Get the names of all derived losses a in an alphabetically sorted list.
    """
    loss_names = [
        get_derived_loss_name(deriv_method, loss_fn)
        for deriv_method, loss_fn in itertools.product(LossDerivationMethod, RawLossFn)
    ]
    loss_names = sorted(loss_names)
    return loss_names


class LossDerivationMethod(Enum):
    MeanDelta = ("mean_delta",)
    MinDelta = ("min_delta",)
    MedianDelta = ("median_delta",)
    MeanReldelta = ("mean_reldelta",)
    MinReldelta = ("min_reldelta",)
    MedianReldelta = ("median_reldelta",)

    def __init__(self, public_name: str):
        self.public_name = public_name


class LossPrecision(Enum):
    Exact = ("exact",)
    PossiblyEstim = ("estim",)

    def __init__(self, public_name: str):
        self.public_name = public_name
