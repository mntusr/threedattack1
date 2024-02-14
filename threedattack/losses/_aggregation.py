from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Literal, NamedTuple

import numpy as np

from ..dataset_model import DepthsWithMasks, SampleType
from ..rendering_core import TwoDAreas
from ..tensor_types.idx import *
from ..tensor_types.npy import *
from ._loss_functions import (
    RawLossFn,
    cropped_d1_loss,
    cropped_log10_loss,
    cropped_rmse_loss,
    d1_loss,
    log10_loss,
    rmse_loss,
)


def calculate_loss_values(
    pred_depths: np.ndarray,
    gt: DepthsWithMasks,
    loss_fns: "set[RawLossFn]",
    target_obj_areas: TwoDAreas | None,
) -> "dict[RawLossFn, np.ndarray]":
    """
    Calculate the different loss values for the specified prediction and ground truth data.

    Parameters
    ----------
    pred_depths
        The predicted depths. Format: ``Scalars::Float``
    gt
        The ground truth depth maps and masks.
    loss_fns
        The loss functions to calculate.
    target_obj_area
        The area of the target object on the rendered images.

    Returns
    -------
    v
        A dictionary that contains the loss values for each sample and loss function. Format of the values: ``Scalars::Float``

    Raises
    ------
    ValueError
        If the areas of the target objects on the images are not specified and a cropped loss is calculated.
    """

    losses_dict: "dict[RawLossFn, np.ndarray]" = dict()

    if RawLossFn.RMSE in loss_fns:
        losses_dict[RawLossFn.RMSE] = rmse_loss(pred_depths=pred_depths, gt=gt)
    if RawLossFn.CroppedRMSE in loss_fns:
        if target_obj_areas is None:
            raise ValueError(
                "It not possible to calculate the cropped RMSE loss. The areas of the target object on the screen are not specified."
            )
        losses_dict[RawLossFn.CroppedRMSE] = cropped_rmse_loss(
            pred_depths=pred_depths, gt=gt, target_obj_areas=target_obj_areas
        )
    if RawLossFn.D1 in loss_fns:
        losses_dict[RawLossFn.D1] = d1_loss(pred_depths=pred_depths, gt=gt)
    if RawLossFn.CroppedD1 in loss_fns:
        if target_obj_areas is None:
            raise ValueError(
                "It not possible to calculate the cropped d1 loss. The areas of the target object on the screen are not specified."
            )
        losses_dict[RawLossFn.CroppedD1] = cropped_d1_loss(
            pred_depths=pred_depths, gt=gt, target_obj_areas=target_obj_areas
        )
    if RawLossFn.Log10 in loss_fns:
        losses_dict[RawLossFn.Log10] = log10_loss(pred_depths=pred_depths, gt=gt)
    if RawLossFn.CroppedLog10 in loss_fns:
        if target_obj_areas is None:
            raise ValueError(
                "It not possible to calculate the cropped log10 loss. The areas of the target object on the screen are not specified."
            )
        losses_dict[RawLossFn.CroppedLog10] = cropped_log10_loss(
            pred_depths=pred_depths, gt=gt, target_obj_areas=target_obj_areas
        )

    return losses_dict


def get_loss_val_min(
    loss_dict: "dict[RawLossFn, np.ndarray]",
) -> "dict[RawLossFn, float]":
    """
    Get the minimum of the specified values for each loss function.

    Parameters
    ----------
    loss_dict
        The loss values for each loss function. Format of the values: ``Scalars::Float``

    Returns
    -------
    v
        The minimal value for each loss function.
    """
    return {
        loss_fn: float(np.min(loss_vals)) for loss_fn, loss_vals in loss_dict.items()
    }


def get_loss_val_median(
    loss_dict: "dict[RawLossFn, np.ndarray]",
) -> "dict[RawLossFn, float]":
    """
    Get the median of the specified values for each loss function.

    Parameters
    ----------
    loss_dict
        The loss values for each loss function. Format of the values: ``Scalars::Float``

    Returns
    -------
    v
        The minimal value for each loss function.
    """
    return {
        loss_fn: float(np.median(loss_vals)) for loss_fn, loss_vals in loss_dict.items()
    }


def get_loss_val_mean(
    loss_dict: "dict[RawLossFn, np.ndarray]",
) -> "dict[RawLossFn, float]":
    """
    Get the mean of the specified values for each loss function.

    Parameters
    ----------
    loss_dict
        The loss values for each loss function. Format of the values: ``Scalars::Float``

    Returns
    -------
    v
        The minimal value for each loss function.
    """
    return {
        loss_fn: float(np.mean(loss_vals)) for loss_fn, loss_vals in loss_dict.items()
    }


def subtract_losses(
    left_dict: "dict[RawLossFn, np.ndarray]", right_dict: "dict[RawLossFn, np.ndarray]"
) -> "dict[RawLossFn, np.ndarray]":
    """
    Elementwise subtract the values of the loss functions.

    Parameters
    ----------
    left_dict
        The loss values on the left side. Format of the values: ``Scalars::Float``
    right_dict
        The loss values on the right side. Format of the values: ``Scalars::Float``

    Returns
    -------
    v
        The result of the substraction. Format of the values: ``Scalars::Float``

    Raises
    ------
    ValueError
        If the set of the keys of the dicts on the left and right side are different.
    """
    if left_dict.keys() != right_dict.keys():
        raise ValueError(
            f"The two loss dictionaries contain different keys. Keys of the first dict: {left_dict.keys()}, keys of the second dict: {right_dict.keys()}"
        )

    return {
        loss_fn: left_dict[loss_fn] - right_dict[loss_fn]
        for loss_fn in left_dict.keys()
    }


def divide_losses(
    left_dict: "dict[RawLossFn, np.ndarray]",
    right_dict: "dict[RawLossFn, np.ndarray]",
    epsilon: float = 1e-9,
) -> "dict[RawLossFn, np.ndarray]":
    """
    Elementwise divide the values of the loss functions.

    Parameters
    ----------
    left_dict
        The loss values on the left side. Format of the values: ``Scalars::Float``
    right_dict
        The loss values on the right side. Format of the values: ``Scalars::Float``

    Returns
    -------
    v
        The result of the substraction. Format of the values: ``Scalars::Float``

    Raises
    ------
    ValueError
        If the set of the keys of the dicts on the left and right side are different.
    """
    if left_dict.keys() != right_dict.keys():
        raise ValueError(
            f"The two loss dictionaries contain different keys. Keys of the first dict: {left_dict.keys()}, keys of the second dict: {right_dict.keys()}"
        )

    return {
        loss_fn: left_dict[loss_fn] / (right_dict[loss_fn] + epsilon)
        for loss_fn in left_dict.keys()
    }


def idx_losses(
    loss_dict: "dict[RawLossFn, np.ndarray]", idx: Any
) -> "dict[RawLossFn, np.ndarray]":
    """
    Index the values in the dictionary of the losses.

    Parameters
    ----------
    loss_dict
        The dict in which the values should be indexed. Format of the values: ``Scalars::Float``

    Returns
    -------
    v
        The new dictionary that contains the indexed values. Format of the values: ``Scalars::Float``
    """
    return {loss_fn: loss_dict[loss_fn][idx] for loss_fn in loss_dict.keys()}


def concat_losses(
    first_losses: "dict[RawLossFn, np.ndarray]",
    second_losses: "dict[RawLossFn, np.ndarray]",
) -> "dict[RawLossFn, np.ndarray]":
    """
    Concatenate the values of the loss functions.

    Parameters
    ----------
    first_losses
        The first loss values. Format of the values: ``Scalars::Float``
    second_losses
        The second loss values. Format of the values: ``Scalars::Float``

    Returns
    -------
    v
        The result of the substraction. Format of the values: ``Scalars::Float``

    Raises
    ------
    ValueError
        If the set of the keys of the dicts on the left and right side are different.
    """
    if first_losses.keys() != second_losses.keys():
        raise ValueError(
            f"The two loss dictionaries contain different keys. Keys of the first dict: {first_losses}, keys of the second dict: {second_losses}"
        )

    return {
        loss_fn: np.concatenate(
            [first_losses[loss_fn], second_losses[loss_fn]], axis=DIM_SCALARS_N
        )
        for loss_fn in first_losses.keys()
    }


def collect_raw_losses_with_viewpt_type(
    losses: dict[tuple[SampleType, RawLossFn], np.ndarray],
    viewpt_type: SampleType,
) -> dict[RawLossFn, np.ndarray]:
    """
    Select the loss values for the specified viewpoint type.

    Parameters
    ----------
    losses
        The loss values for different viewpoint types. Format of the values: ``Scalars::Float``
    viewpt_type
        The relevant viewpoint type.

    Returns
    -------
    v
        The collected loss values. Format of the values: ``Scalars::Float``
    """
    return {
        loss_fn: loss_vals
        for (loss_viewpt_type, loss_fn), loss_vals in losses.items()
        if loss_viewpt_type == viewpt_type
    }
