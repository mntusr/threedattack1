from typing import TYPE_CHECKING

import numpy as np

from .._typing import type_instance
from ..tensor_types.npy import *
from ._comm_controller import RemoteDepthEstController
from ._protocols import AlignmentFunction, AsyncDepthPredictor, DepthFuture


class ZoeDepthPredictor:
    MODEL_NAME = "zoedepth_nk"

    def __init__(self):
        self.alignment_function = IdentityAlignmentFunction()

        self.rem_proc = RemoteDepthEstController(
            shared_mem_stem="zoedepth",
        )

    def get_name(self) -> str:
        return self.MODEL_NAME

    def predict_native(self, rgbs: np.ndarray) -> DepthFuture:
        return self.rem_proc.process_async(rgbs)


class IdentityAlignmentFunction:
    def __call__(
        self,
        native_preds: np.ndarray,
        gt_depths: np.ndarray,
        masks: np.ndarray,
        depth_cap: float,
    ) -> np.ndarray:
        return native_preds


if TYPE_CHECKING:
    v1: AlignmentFunction = type_instance(IdentityAlignmentFunction)
    v2: AsyncDepthPredictor = type_instance(ZoeDepthPredictor)
