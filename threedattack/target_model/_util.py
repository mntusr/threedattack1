from dataclasses import dataclass

import numpy as np

from ..rendering_core import RGBsWithDepthsAndMasks
from ._protocols import AsyncDepthPredictor


def predict_aligned(
    predictor: AsyncDepthPredictor,
    images: RGBsWithDepthsAndMasks,
    depth_cap: float,
) -> np.ndarray:
    """
    Do the depth prediction, then return with the aligned depth map.

    The algorithm uses the alignment algorithm provided by the specified predictor.

    Parameters
    ----------
    predictor
        The predictor to do the depth prediction.
    images
        The images for which the depth should be predicted. The ground truth depth maps and the corresponding masks are required by the alignment step.
    depth_cap
        The maximal predictable depth. The depth will be clopped to the ``[0, depth_cap]`` range. This value is typically provided by the source of the RGBD images.

    Returns
    -------
    v
        The predicted depth maps. They have the same masks as the ground truth depth maps.
    """
    native_preds = predictor.predict_native(images.rgbs).get()
    aligned_pred_depths = predictor.alignment_function(
        native_preds=native_preds,
        gt_depths=images.depths,
        masks=images.masks,
        depth_cap=depth_cap,
    )
    return aligned_pred_depths
