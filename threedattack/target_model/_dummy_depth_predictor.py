from typing import TYPE_CHECKING

import numpy as np

from .._typing import type_instance
from ..tensor_types.idx import *
from ..tensor_types.npy import *
from ._comm_controller import RemoteDepthEstController
from ._protocols import AlignmentFunction, AsyncDepthPredictor, DepthFuture


class DummyDepthPredictor:
    DUMMY_DEPTH_NAMES = ["dummy_depth1", "dummy_depth2"]

    @staticmethod
    def is_dummy_name(name: str) -> bool:
        return name in DummyDepthPredictor.DUMMY_DEPTH_NAMES

    def __init__(self, name: str):
        if not self.is_dummy_name(name):
            raise ValueError("This name does not belong to a dummy depth predictor.")

        self.alignment_function = DummyAlignmentFunction()
        self.model_name = name

    def get_name(self) -> str:
        return self.model_name

    def predict_native(self, rgbs: np.ndarray) -> DepthFuture:
        raise NotImplementedError(
            "This depth predictor is not intended for actual depth prediction."
        )


class DummyAlignmentFunction:
    def __call__(
        self,
        native_preds: np.ndarray,
        gt_depths: np.ndarray,
        masks: np.ndarray,
        depth_cap: float,
    ) -> np.ndarray:
        raise NotImplementedError(
            "This alignment function does not intended for actual usage."
        )


if TYPE_CHECKING:
    v1: AlignmentFunction = type_instance(DummyAlignmentFunction)
    v2: AsyncDepthPredictor = type_instance(DummyDepthPredictor)
