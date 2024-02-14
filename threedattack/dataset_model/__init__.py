"""
This package contains the basic model of an RGBD dataset.
"""

from ._dataset_model import (
    CamProjSpec,
    DatasetLike,
    DepthsWithMasks,
    ExactSampleCounts,
    RGBsWithDepthsAndMasks,
    SamplesBase,
    SampleType,
    SampleTypeError,
)

__all__ = [
    "DatasetLike",
    "DepthsWithMasks",
    "RGBsWithDepthsAndMasks",
    "SamplesBase",
    "SampleType",
    "ExactSampleCounts",
    "CamProjSpec",
    "SampleTypeError",
]
