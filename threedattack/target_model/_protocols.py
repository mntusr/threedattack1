from typing import Protocol, TypeVar

import numpy as np

T = TypeVar("T", bound="AlignmentFunction")


class AsyncDepthPredictor(Protocol[T]):
    """
    The common protocol of the depth predictors.
    """

    alignment_function: T

    def get_name(self) -> str:
        ...

    def predict_native(self, rgbs: np.ndarray) -> "DepthFuture":
        """
        Start the depth prediction in the "native" prediction format of the model.


        Parameters
        ----------
        rgbs
            The input images. Format: ``Im::RGBs``

        Returns
        -------
        v
            A promise-like object that will provide the predictions when they are done.
        """
        ...


class DepthFuture(Protocol):
    def get(self) -> np.ndarray:
        """
        Get the predicted depth-like data. The resulting tensor does not necessarily contain depth maps. The exact format is implementation detail.

        This function may return immediately if the prediction is done. If the prediction is not done, then this function waits until it is done, then returns with the results.

        Returns
        -------
        v
            The predictions. Format: ``Im::*``

        Notes
        -----
        The predictions are not necessarily invariant for the batch size.

        The predictions are invariant for the previous calls of this function.

        The predictions are invariant for the order of the RGB images in the same batch, as long as the batch size is the same.

        """
        ...


class AlignmentFunction(Protocol):
    """
    Convert the native prediction of a model to an exact depth prediction.

    Notes
    -----
    The alignment function might make extra assumptions about the native prediction format of the model.
    """

    def __call__(
        self,
        native_preds: np.ndarray,
        gt_depths: np.ndarray,
        masks: np.ndarray,
        depth_cap: float,
    ) -> np.ndarray:
        """
        Convert the native predictions of a depth predictor to exact depth maps using an alignment operation.

        Parameters
        ----------
        native_preds
            The native predictions. Format: ``ArbSamples::*``
        gt_depths
            The ground truth depth maps. Format: ``Im::DepthMaps``
        masks
            The masks for the ground truth depth maps. Format: ``Im::DepthMasks``
        depth_cap
            The maximum predictable depths. The greate values should be clipped to the ``[0, depth_cap]`` range.

        Returns
        -------
        v
            The aligned depth predictions. Format: ``Im::DepthMaps``
        """
        ...
