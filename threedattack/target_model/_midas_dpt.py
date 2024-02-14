from typing import TYPE_CHECKING

import numpy as np

from .._typing import type_instance
from ..tensor_types.npy import *
from ._comm_controller import RemoteDepthEstController
from ._protocols import AlignmentFunction, AsyncDepthPredictor, DepthFuture


class DptBeit384Predictor:
    MODEL_NAME = "dpt_beit_384"

    def __init__(self):
        self.alignment_function = MidasAlignmentFunction()

        self.rem_proc = RemoteDepthEstController(
            shared_mem_stem="dptbeit384",
        )

    def get_name(self) -> str:
        return self.MODEL_NAME

    def predict_native(self, rgbs: np.ndarray) -> DepthFuture:
        return self.rem_proc.process_async(rgbs)


class MidasSmallPredictor:
    MODEL_NAME = "midas_small"

    def __init__(self):
        self.alignment_function = MidasAlignmentFunction()

        self.rem_proc = RemoteDepthEstController(
            shared_mem_stem="midassmall",
        )

    def get_name(self) -> str:
        return self.MODEL_NAME

    def predict_native(self, rgbs: np.ndarray) -> DepthFuture:
        return self.rem_proc.process_async(rgbs)


class MidasLargePredictor:
    MODEL_NAME = "midas_large"

    def __init__(self):
        self.alignment_function = MidasAlignmentFunction()

        self.rem_proc = RemoteDepthEstController(
            shared_mem_stem="midaslarge",
        )

    def get_name(self) -> str:
        return self.MODEL_NAME

    def predict_native(self, rgbs: np.ndarray) -> DepthFuture:
        return self.rem_proc.process_async(rgbs)


class MidasAlignmentFunction:
    def __call__(
        self,
        native_preds: np.ndarray,
        gt_depths: np.ndarray,
        masks: np.ndarray,
        depth_cap: float,
    ) -> np.ndarray:
        """
        Convert the unaligned disparity map predictions provided by the MiDaS models to aligned depth maps.

        Parameters
        ----------
        pred_disps
            The unaligned disparity map predictions. Format: ``Im::DispMaps``
        gt_depths
            The ground truth depth maps. Format: ``Im::DepthMaps``
        depth_cap
            The maximal depth. The depths are clapped to range ``[0, depth_cap]`` in the final depth maps.

        Returns
        -------
        v
            The aligned depth maps. Format: ``Im::DepthMaps``
        """
        gt_disp = _gt_depth_2_disp(gt_depths=gt_depths, masks=masks)
        scale, shift = _compute_scale_and_shift(
            pred_disp=native_preds, target_depth=gt_disp, masks=masks
        )

        pred_disp_aligned = scale.reshape(-1, 1, 1, 1) * native_preds + shift.reshape(
            -1, 1, 1, 1
        )
        pred_depth_aligned = _aligned_disp_pred_2_depth(
            disps=pred_disp_aligned, depth_cap=depth_cap
        )

        return pred_depth_aligned


def _gt_depth_2_disp(gt_depths: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """
    Convert the ground truth depth maps to disparity maps.

    Parameters
    ----------
    gt_depths
        The ground truth depth maps. Format: ``Im::DepthMaps``
    masks
        The masks for the gt depth maps. Format: ``Im::DepthMasks``

    Returns
    -------
    v
        The ground truth depth maps as disparity maps. Format: ``Im::DispMaps``
    """
    disp = np.zeros_like(gt_depths)
    disp[masks == 1] = 1.0 / gt_depths[masks == 1]
    return disp


def _aligned_disp_pred_2_depth(disps: np.ndarray, depth_cap: float) -> np.ndarray:
    """
    Convert the already aligned disparity maps prediction to depth maps. The depths are clopped to range ``[0, depth_cap]``

    Parameters
    ----------
    disps
        The aligned disparity maps.
    depth_cap
        The maximal depths.
    """
    disparity_cap = 1.0 / depth_cap
    too_far_pixel_est = disps < disparity_cap
    disps = too_far_pixel_est * disparity_cap + (~too_far_pixel_est) * disps

    depth = 1 / disps
    return depth


def _compute_scale_and_shift(
    pred_disp: np.ndarray, target_depth: np.ndarray, masks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the scale and shift values for each prediction.

    Parameters
    ----------
    pred_disp
        The unaligned non-negative disparity maps prediction. Format: ``DisparityMaps``
    target_depth
        The ground truth depth maps. Format: ``Im::DepthMaps``
    mask
        The masks. Format: ``Masks``

    Returns
    -------
    x0
        The scales. Format: ``Scalars::Float``
    x1
        The shifts. Format: ``Scalars::Float``
    """
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = np.sum(masks * pred_disp * pred_disp, (2, 3))
    a_01 = np.sum(masks * pred_disp, (2, 3))
    a_11 = np.sum(masks, (2, 3))

    # right hand side: b = [b_0, b_1]
    b_0 = np.sum(masks * pred_disp * target_depth, (2, 3))
    b_1 = np.sum(masks * target_depth, (2, 3))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = np.zeros_like(b_0)
    x_1 = np.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


if TYPE_CHECKING:
    v1: AlignmentFunction = type_instance(MidasAlignmentFunction)
    v2: AsyncDepthPredictor[MidasAlignmentFunction] = type_instance(MidasSmallPredictor)
