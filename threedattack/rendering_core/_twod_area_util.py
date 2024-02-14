import numpy as np

from ..tensor_types.idx import *
from ..tensor_types.npy import *
from ._data import TwoDAreas, TwoDSize


def get_twod_area_masks(areas: TwoDAreas, im_shape: TwoDSize) -> np.ndarray:
    """
    Get the masks that selects the two dimensional areas on images.

    Parameters
    ----------
    areas
        The two dimensional areas.
    im_shape
        The shape of the images.

    Returns
    -------
    v
        The masks. Format: ``Im::DepthMasks``

    Raises
    ------
    ValueError
        If there is no specified area or at least one dimension of the image is not positive.
    """
    if areas.x_maxes_excluding.shape[DIM_SCALARS_N] == 0:
        raise ValueError("There is no specified area.")

    if not im_shape.is_positive():
        raise ValueError(
            f"At least one dimension of the specified image is not positive. Size: {im_shape}"
        )

    n_ims = areas.x_maxes_excluding.shape[DIM_SCALARS_N]

    masks = np.zeros(
        newshape_im_depthmasks(n=n_ims, h=im_shape.y_size, w=im_shape.x_size),
        dtype=np.bool_,
    )
    for im_idx in range(n_ims):
        x_max_excluding = idx_scalars_float(areas.x_maxes_excluding, n=im_idx)
        x_min_including = idx_scalars_float(areas.x_mins_including, n=im_idx)
        y_max_excluding = idx_scalars_float(areas.y_maxes_excluding, n=im_idx)
        y_min_including = idx_scalars_float(areas.y_mins_including, n=im_idx)

        assert x_min_including < x_max_excluding
        assert y_min_including < y_max_excluding

        if (x_max_excluding <= 0) or (y_max_excluding <= 0):
            continue
        if (x_min_including >= im_shape.x_size) or (y_min_including >= im_shape.y_size):
            continue

        x_max_excluding = np.clip(x_max_excluding, 0, im_shape.x_size)
        x_min_including = np.clip(x_min_including, 0, im_shape.x_size)
        y_max_excluding = np.clip(y_max_excluding, 0, im_shape.y_size)
        y_min_including = np.clip(y_min_including, 0, im_shape.y_size)

        upd_im_depthmasks(
            masks,
            n=im_idx,
            w=slice(x_min_including, x_max_excluding),
            h=slice(y_min_including, y_max_excluding),
            value_=1,
        )

    return masks
