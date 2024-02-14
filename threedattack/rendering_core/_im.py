import math
from typing import Any, Literal, Union

import cv2
import matplotlib.axes as axes
import matplotlib.pyplot as plt
import numpy as np

from ..dataset_model import DepthsWithMasks
from ..tensor_types.idx import *
from ..tensor_types.npy import *


def scale_depth_maps_with_masks(
    depths_with_masks: DepthsWithMasks, new_size_rel: float
) -> DepthsWithMasks:
    """
    Scale the specified masked depth maps to the new size.

    The function uses nearest neighbor interpolation.

    The sizes of the new depth maps and masks:

    * width: ``floor(original_width*new_size_rel)``
    * height: ``floor(original_height*new_size_rel)``

    Parameters
    ----------
    depths_with_masks
        The depth maps and corresponding masks to resize.
    new_size_rel
        The new relative size.

    Returns
    -------
    v
        The resized depth maps and masks.

    Raises
    ------
    ValueError
        If the width or height of the scaled image would be smaller than 1.
    """

    width = math.floor(depths_with_masks.depths.shape[DIM_IM_W] * new_size_rel)
    height = math.floor(depths_with_masks.depths.shape[DIM_IM_H] * new_size_rel)

    if min(width, height) < 1:
        raise ValueError(
            "The width or height of the new scaled image would be smaller than 1."
        )
    scaled_depths_depth_data = cv_resize_multiple_images(
        images=depths_with_masks.depths,
        interpolation_method=cv2.INTER_NEAREST,
        width=width,
        height=height,
    )
    scaled_depths_mask_data = cv_resize_multiple_images(
        images=depths_with_masks.masks.astype(np.int8),
        interpolation_method=cv2.INTER_NEAREST,
        width=width,
        height=height,
    ).astype(np.bool_)

    return DepthsWithMasks(
        depths=scaled_depths_depth_data, masks=scaled_depths_mask_data
    )


def cv_resize_multiple_images(
    images: np.ndarray, interpolation_method: Any, width: int, height: int
) -> np.ndarray:
    """
    Resize multiple images simultaneously using OpenCV.

    Parameters
    ----------
    images
        The images. Format: ``Im::*``
    interpolation_method
        The interpolation method to use.
    width
        The target width.
    height
        The target height.

    Returns
    -------
    v
        The resized images. Format: same as the format of the input images.
    """
    orig_width: int = images.shape[DIM_IM_W]
    orig_height: int = images.shape[DIM_IM_H]
    n_channels = images.shape[DIM_IM_C]
    n_samples = images.shape[DIM_IM_N]

    resized: list[np.ndarray] = []
    for i in range(n_samples):
        cv2_im = (
            images[i]
            .reshape((n_channels, orig_height, orig_width))
            .transpose([1, 2, 0])
        )
        cv2_im = cv2.resize(
            cv2_im,
            (width, height),
            interpolation=interpolation_method,
        )
        if n_channels == 1:
            cv2_im = cv2_im.reshape((width, height, 1))
        resized.append(
            cv2_im.transpose([2, 0, 1]).reshape((1, n_channels, height, width))
        )

    return np.concatenate(resized, axis=0)


def imshow(
    im: np.ndarray,
    on: Union[Literal["newfig"], axes.Axes] = "newfig",
    show: bool = True,
) -> None:
    """
    Show the specified image.

    Parameters
    ----------
    im
        Format: ``Im::*[Single]``
    on
        The axes on which the figure should be shown. It is either an axis, or "newfig". "newfig" means that the image should be shown on a new figure.
    show
        If true, then `matplotlib.pyplot.show` will be invoked too.

    Raises
    ------
    ValueError
        If ``im`` is not ``Im::*[Single]``.
    """
    if not dmatch_im(im, kinds={"single"}):
        raise ValueError('The argument "im" is not "Im::*[Single]".')

    if on == "newfig":
        fig = plt.figure()
        target_axes = fig.gca()
    else:
        target_axes = on

    target_axes.imshow(idx_im(im, n=0).transpose([1, 2, 0]))

    if show:
        plt.show()
