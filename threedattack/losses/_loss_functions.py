from enum import Enum

import numpy as np

from ..dataset_model import DepthsWithMasks
from ..rendering_core import TwoDAreas, TwoDSize, get_twod_area_masks
from ..tensor_types.idx import *
from ..tensor_types.npy import *


def cropped_se_map(
    pred_depths: np.ndarray, gt: DepthsWithMasks, target_obj_areas: TwoDAreas
) -> np.ndarray:
    """
    The maps that specify the masked squared error values in the cropped RMSE loss for each pixel.

    Parameters
    ----------
    pred_depths
        The aligned predicted depths. Format: ``Im::DepthMaps``
    gt
        The ground truth depths and masks.
    target_obj_areas
        The areas of the target objects.

    Returns
    -------
    v
        The calculated map. Format: ``Im::FloatMap``
    """
    target_obj_area_masks = get_twod_area_masks(
        target_obj_areas, _get_im_size(gt.masks)
    )
    return se_map(
        pred_depths=pred_depths,
        gt=DepthsWithMasks(depths=gt.depths, masks=gt.masks & target_obj_area_masks),
    )


def cropped_rmse_loss(
    pred_depths: np.ndarray, gt: DepthsWithMasks, target_obj_areas: TwoDAreas
) -> np.ndarray:
    """
    Calculate the RMSE loss for the area of the target object for each individual depth prediction.

    Parameters
    ----------
    pred_depths
        The aligned predicted depths. Format: ``Im::DepthMaps``
    gt
        The ground truth depths and masks.
    target_obj_areas
        The areas of the target objects.

    Returns
    -------
    v
        The loss values for each sample. Format: ``Scalars::Float``
    """
    target_obj_area_masks = get_twod_area_masks(
        target_obj_areas, _get_im_size(gt.masks)
    )

    cropped_losses = rmse_loss(
        pred_depths=pred_depths,
        gt=DepthsWithMasks(depths=gt.depths, masks=gt.masks & target_obj_area_masks),
    )
    return cropped_losses


def se_map(pred_depths: np.ndarray, gt: DepthsWithMasks) -> np.ndarray:
    """
    The maps that specify the masked squared error values in the RMSE loss for each pixel.

    Parameters
    ----------
    pred_depths
        The aligned predicted depths. Format: ``Im::DepthMaps``
    gt
        The ground truth depths and masks.
    target_obj_areas
        The areas of the target objects.

    Returns
    -------
    v
        The calculated map. Format: ``Im::FloatMap``
    """
    perpixel_mse = np.zeros_like(gt.depths, dtype=np.float32)
    perpixel_mse[gt.masks] = (gt.depths[gt.masks] - pred_depths[gt.masks]) ** 2
    return perpixel_mse


def rmse_loss(pred_depths: np.ndarray, gt: DepthsWithMasks) -> np.ndarray:
    """
    Calculate the RMSE loss for the individual depth predictions.

    Parameters
    ----------
    pred_depths
        The aligned predicted depths. Format: ``Im::DepthMaps``
    gt
        The ground truth depths and masks.

    Returns
    -------
    v
        The loss values for each sample. Format: ``Scalars::Float``
    """
    perpixel_mse = np.zeros_like(gt.depths, dtype=np.float32)
    perpixel_mse[gt.masks] = (gt.depths[gt.masks] - pred_depths[gt.masks]) ** 2
    rmse_val = np.sqrt(_masked_samplewise_mean(ims=perpixel_mse, masks=gt.masks))
    return rmse_val


def cropped_log10_loss(
    pred_depths: np.ndarray, gt: DepthsWithMasks, target_obj_areas: TwoDAreas
) -> np.ndarray:
    """
    Calculate the log10 loss for the area of the target object for each individual depth prediction.

    Parameters
    ----------
    pred_depths
        The aligned predicted depths. Format: ``Im::DepthMaps``
    gt
        The ground truth depths and masks.
    target_obj_areas
        The areas of the target objects.

    Returns
    -------
    v
        The loss values for each sample. Format: ``Scalars::Float``
    """
    target_obj_area_masks = get_twod_area_masks(
        target_obj_areas, _get_im_size(gt.masks)
    )

    cropped_losses = log10_loss(
        pred_depths=pred_depths,
        gt=DepthsWithMasks(depths=gt.depths, masks=gt.masks & target_obj_area_masks),
    )
    return cropped_losses


def cropped_log10_map(
    pred_depths: np.ndarray, gt: DepthsWithMasks, target_obj_areas: TwoDAreas
) -> np.ndarray:
    """
    The maps that specify the masked log10 values in the cropped log10 loss for each pixel.

    Parameters
    ----------
    pred_depths
        The aligned predicted depths. Format: ``Im::DepthMaps``
    gt
        The ground truth depths and masks.
    target_obj_areas
        The areas of the target objects.

    Returns
    -------
    v
        The calculated map. Format: ``Im::FloatMap``
    """
    target_obj_area_masks = get_twod_area_masks(
        target_obj_areas, _get_im_size(gt.masks)
    )
    cropped_map = log10_map(
        pred_depths=pred_depths,
        gt=DepthsWithMasks(depths=gt.depths, masks=gt.masks & target_obj_area_masks),
    )
    return cropped_map


def log10_loss(pred_depths: np.ndarray, gt: DepthsWithMasks) -> np.ndarray:
    """
    Calculate the log10 loss for the individual depth predictions.

    Parameters
    ----------
    pred_depths
        The aligned predicted depths. Format: ``Im::DepthMaps``
    gt
        The ground truth depths and masks.

    Returns
    -------
    v
        The loss values for each sample. Format: ``Scalars::Float``
    """
    log_10 = np.zeros_like(pred_depths, dtype=np.float32)
    log_10[gt.masks] = np.abs(
        np.log10(gt.depths[gt.masks]) - np.log10(pred_depths[gt.masks])
    )
    p = _masked_samplewise_mean(ims=log_10, masks=gt.masks)
    return p


def log10_map(pred_depths: np.ndarray, gt: DepthsWithMasks) -> np.ndarray:
    """
    The maps that specify the masked log10 values in the log10 loss for each pixel.

    Parameters
    ----------
    pred_depths
        The aligned predicted depths. Format: ``Im::DepthMaps``
    gt
        The ground truth depths and masks.
    target_obj_areas
        The areas of the target objects.

    Returns
    -------
    v
        The calculated map. Format: ``Im::FloatMap``
    """
    log_10 = np.zeros_like(pred_depths, dtype=np.float32)
    log_10[gt.masks] = np.abs(
        np.log10(gt.depths[gt.masks]) - np.log10(pred_depths[gt.masks])
    )
    return log_10


def cropped_d1_map(
    pred_depths: np.ndarray, gt: DepthsWithMasks, target_obj_areas: TwoDAreas
) -> np.ndarray:
    """
    The maps that specify the masked binary error values in the d1 loss for each pixel.

    Parameters
    ----------
    pred_depths
        The aligned predicted depths. Format: ``Im::DepthMaps``
    gt
        The ground truth depths and masks.
    target_obj_areas
        The areas of the target objects.

    Returns
    -------
    v
        The calculated map. Format: ``Im::FloatMap``
    """
    target_obj_area_masks = get_twod_area_masks(
        target_obj_areas, _get_im_size(gt.masks)
    )

    cropped_map = d1_map(
        pred_depths=pred_depths,
        gt=DepthsWithMasks(depths=gt.depths, masks=gt.masks & target_obj_area_masks),
    )
    return cropped_map


def cropped_d1_loss(
    pred_depths: np.ndarray, gt: DepthsWithMasks, target_obj_areas: TwoDAreas
) -> np.ndarray:
    """
    Calculate the d1 loss for the individual depth predictions.

    Parameters
    ----------
    pred_depths
        The aligned predicted depths. Format: ``Im::DepthMaps``
    gt
        The ground truth depths and masks.

    Returns
    -------
    v
        The loss values for each sample. Format: ``Scalars::Float``
    """
    target_obj_area_masks = get_twod_area_masks(
        target_obj_areas, _get_im_size(gt.masks)
    )

    cropped_losses = d1_loss(
        pred_depths=pred_depths,
        gt=DepthsWithMasks(depths=gt.depths, masks=gt.masks & target_obj_area_masks),
    )
    return cropped_losses


def d1_map(pred_depths: np.ndarray, gt: DepthsWithMasks) -> np.ndarray:
    """
    The maps that specify the masked binary error values in the d1 loss for each pixel.

    Parameters
    ----------
    pred_depths
        The aligned predicted depths. Format: ``Im::DepthMaps``
    gt
        The ground truth depths and masks.
    target_obj_areas
        The areas of the target objects.

    Returns
    -------
    v
        The calculated map. Format: ``Im::FloatMap``
    """
    threshold = 1.25
    err = np.zeros_like(pred_depths, dtype=np.float32)
    err[gt.masks == 1] = np.maximum(
        pred_depths[gt.masks == 1] / gt.depths[gt.masks == 1],
        gt.depths[gt.masks == 1] / pred_depths[gt.masks == 1],
    )

    err[gt.masks == 1] = (err[gt.masks == 1] < threshold).astype(np.float32)
    return err


def d1_loss(pred_depths: np.ndarray, gt: DepthsWithMasks) -> np.ndarray:
    """
    Calculate the log10 loss for the individual depth predictions.

    Parameters
    ----------
    pred_depths
        The aligned predicted depths. Format: ``Im::DepthMaps``
    gt
        The ground truth depths and masks.

    Returns
    -------
    v
        The loss values for each sample. Format: ``Scalars::Float``
    """
    threshold = 1.25
    err = np.zeros_like(pred_depths, dtype=np.float32)
    err[gt.masks == 1] = np.maximum(
        pred_depths[gt.masks == 1] / gt.depths[gt.masks == 1],
        gt.depths[gt.masks == 1] / pred_depths[gt.masks == 1],
    )

    err[gt.masks == 1] = (err[gt.masks == 1] < threshold).astype(np.float32)

    p = _masked_samplewise_mean(ims=err, masks=gt.masks)
    return p


def _masked_samplewise_mean(ims: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """
    Calculate the masked mean for all samples individually.

    Parameters
    ----------
    ims
        The images. Format: ``Im::*``
    masks
        The masks. Format: ``Im::DepthMasks``

    Returns
    -------
    v
        The elementwise masked means. Format: ``Scalars::Float``
    """

    return np.sum(ims, axis=(DIM_IM_C, DIM_IM_H, DIM_IM_W)).astype(np.float32) / np.sum(
        masks, axis=(DIM_IM_C, DIM_IM_H, DIM_IM_W)
    )


def _get_im_size(im: np.ndarray) -> TwoDSize:
    return TwoDSize(x_size=im.shape[DIM_IM_W], y_size=im.shape[DIM_IM_H])


class RawLossFn(Enum):
    RMSE = ("rmse",)
    CroppedRMSE = ("cropped_rmse",)
    Log10 = ("log10",)
    CroppedLog10 = ("cropped_log10",)
    D1 = ("d1",)
    CroppedD1 = ("cropped_d1",)

    def __init__(self, public_name: str):
        self.public_name = public_name


NON_CROPEED_RAW_LOSS_FNS = {RawLossFn.RMSE, RawLossFn.Log10, RawLossFn.D1}
