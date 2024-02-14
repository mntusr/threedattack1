from enum import Enum, auto
from typing import Callable

from ._dummy_depth_predictor import DummyDepthPredictor
from ._midas_dpt import DptBeit384Predictor, MidasLargePredictor, MidasSmallPredictor
from ._protocols import AsyncDepthPredictor
from ._zoedepth import ZoeDepthPredictor


def load_target_model_by_name(model_name) -> AsyncDepthPredictor:
    if DummyDepthPredictor.is_dummy_name(model_name):
        return DummyDepthPredictor(model_name)
    else:
        return _MODELS_BY_NAME[model_name]()


def get_supported_models() -> list[str]:
    return sorted(list(_MODELS_BY_NAME.keys()) + DummyDepthPredictor.DUMMY_DEPTH_NAMES)


_MODELS_BY_NAME: dict[str, Callable[[], AsyncDepthPredictor]] = {
    MidasSmallPredictor.MODEL_NAME: MidasSmallPredictor,
    ZoeDepthPredictor.MODEL_NAME: ZoeDepthPredictor,
    DptBeit384Predictor.MODEL_NAME: DptBeit384Predictor,
    MidasLargePredictor.MODEL_NAME: MidasLargePredictor,
}
