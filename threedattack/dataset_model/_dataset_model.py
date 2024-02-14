from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, NamedTuple, Protocol, Sequence, TypeVar

import numpy as np


class CamProjSpec(NamedTuple):
    """
    A class that describes the invertable affine linear projection of a camera to the image plane.

    The source coordinate system is a Z-up right handed coordinate system.

    The described projection:

    * Input: ``[[x], [y], [z], [1]]``
    * Output: ``[[x_image], [y_image], [w_image]]``
    """

    im_left_x_val: float
    """
    The image x coordinate corresponding to ``x=0`` in the depth map.
    """

    im_right_x_val: float
    """
    The image x coordinate corresponding to ``x=-1`` in the depth map.
    """

    im_top_y_val: float
    """
    The image y coordinate corresponding to ``y=0`` in the depth map.
    """

    im_bottom_y_val: float
    """
    The image y coordinate corresponding to ``y=-1`` in the depth map.
    """

    proj_mat: np.ndarray
    """
    The projection matrix. B is invertable, where ``proj_mat=:[B|l]``.

    Format: ``Mat::Float[F3x4]``
    """


class SamplesBase:
    """
    The base class of one or more samples in RGBD dataset-like objects.

    Parameters
    ----------
    rgbds
        The RGBD images in the sample.
    """

    rgbds: "RGBsWithDepthsAndMasks"

    def __init__(self, rgbds: "RGBsWithDepthsAndMasks"):
        self.rgbds = rgbds


@dataclass
class RGBsWithDepthsAndMasks:
    """
    A data structure that gropus the RGB images, the corresponding depth maps and masks.
    """

    rgbs: np.ndarray
    """
    The RGB images.

    Format: ``Im::RGBs``
    """
    depths: np.ndarray
    """
    The corresponding depth maps. The amount of depth maps is the same as the amount of the RGB images.

    The size of the masks is the same as the size of the depth maps.

    Format: ``Im::DepthMaps``
    """
    masks: np.ndarray
    """
    The masksk for the depth maps. The amount of the masks is the same as the amount of the depth maps.

    The sizes of the masks are the same as the sizes of the depth maps.

    Format: ``Im::DepthMasks``
    """

    def get_depths_with_masks(self) -> "DepthsWithMasks":
        return DepthsWithMasks(depths=self.depths, masks=self.masks)

    def __getitem__(self, item: Any) -> "RGBsWithDepthsAndMasks":
        return RGBsWithDepthsAndMasks(
            rgbs=self.rgbs[item], depths=self.depths[item], masks=self.masks[item]
        )


@dataclass
class DepthsWithMasks:
    """
    A data structure that groups the depth maps and the corresponding masks.
    """

    depths: np.ndarray
    """
    Format: ``Im::DepthMaps``
    """
    masks: np.ndarray
    """
    Format: ``Im::DepthMasks``
    """

    def __getitem__(self, item: Any) -> "DepthsWithMasks":
        return DepthsWithMasks(depths=self.depths[item], masks=self.masks[item])


class SampleType(Enum):
    """
    An enum that specifies the types (training, validation, testing) of the samples.
    """

    Train = ("train",)
    Test = ("test",)
    Val = ("val",)

    def __init__(self, public_name: str):
        self.public_name = public_name


class ExactSampleCounts:
    """
    A class that specifies the exact number of samples with different types in a dataset-like object.

    Parameters
    ----------
    n_train_samples
        The number of training samples.
    n_test_samples
        The number of testing samples.
    n_val_samples
        The number of validation samples.

    Raises
    ------
    ValueError
        If any of the sample counts is negative.
    """

    def __init__(self, n_train_samples: int, n_test_samples: int, n_val_samples: int):
        if n_train_samples < 0:
            raise ValueError(
                f"The number of training samples should be at least 0. Current value: {n_train_samples}"
            )
        if n_test_samples < 0:
            raise ValueError(
                f"The number of testing samples should be at least 0. Current value: {n_test_samples}"
            )
        if n_val_samples < 0:
            raise ValueError(
                f"The number of validation samples should be at least 0. Current value: {n_val_samples}"
            )

        self.__n_train_samples = n_train_samples
        self.__n_test_samples = n_test_samples
        self.__n_val_samples = n_val_samples

    def __eq__(self, other: "ExactSampleCounts") -> bool:
        return (
            (self.__n_train_samples == other.__n_train_samples)
            and (self.__n_test_samples == other.__n_test_samples)
            and (self.__n_val_samples == other.__n_val_samples)
        )

    def __repr__(self) -> str:
        return f"ExactSampleCounts(n_train_samples={self.__n_train_samples}, n_test_samples={self.__n_test_samples}, n_val_samples={self.__n_val_samples})"

    def __str__(self) -> str:
        return self.__repr__()

    def __ne__(self, other: "ExactSampleCounts") -> bool:
        return not self.__eq__(other)

    @property
    def n_train_samples(self) -> int:
        """
        The number of training samples.
        """
        return self.__n_train_samples

    @property
    def n_test_samples(self) -> int:
        """
        The number of testing samples.
        """
        return self.__n_test_samples

    @property
    def n_val_samples(self) -> int:
        """
        The number of validation samples.
        """
        return self.__n_val_samples

    def sum(self) -> int:
        """
        Get the sum of the sample counts.
        """
        return self.n_train_samples + self.n_test_samples + self.n_val_samples

    def is_all_smaller_or_equal(self, other: "ExactSampleCounts") -> bool:
        """
        Return true if all sample counts are smaller than or equal to the viewpoint counts specified in the other object.
        """
        if self.__n_train_samples > other.__n_train_samples:
            return False
        if self.__n_test_samples > other.__n_test_samples:
            return False
        if self.__n_val_samples > other.__n_val_samples:
            return False

        return True

    def get_n_samples_by_type(self, sample_type: SampleType) -> int:
        """
        Get the number of samples for the specified type.
        """
        match sample_type:
            case SampleType.Train:
                return self.__n_train_samples
            case SampleType.Test:
                return self.__n_test_samples
            case SampleType.Val:
                return self.__n_val_samples


T = TypeVar("T", bound=SamplesBase, covariant=True)


class DatasetLike(Protocol[T]):
    """
    The protocol of all dataset-like objects. These objects behave like an RGBD dataset.
    """

    def get_n_samples(self) -> "ExactSampleCounts":
        """Get the number of samples for each type."""
        ...

    def get_sample(self, idx: int, sample_type: "SampleType") -> T:
        """
        Get the sample at the specified index.

        The index might be negative too.

        Parameters
        ----------
        idx
            The index of the sample.
        sample_type
            The type of the sample.

        Returns
        -------
        v
            The sample at the specified index.

        Raises
        ------
        IndexError
            If the specified index is not valid.
        SampleTypeError
            If the dataset does not contain any sample with the specified type.
        """
        ...

    def get_samples(self, idxs: Sequence[int] | slice, sample_type: SampleType) -> T:
        """
        Get the samples at the specified indices.

        The index might be negative too.

        Parameters
        ----------
        idxs
            The indices of the samples.
        sample_type
            The type of the samples.

        Returns
        -------
        v
            The samples at the specified indices.

        Raises
        ------
        IndexError
            If the specified indices are not valid.
        SampleTypeError
            If the dataset does not contain any sample with the specified type.
        """
        ...

    def get_depth_cap(self) -> float:
        """
        Get the depth value above which the depth values are masked out.

        The returned value might be greater than the actual maximal depth of the samples in the dataset.
        """
        ...

    def get_cam_proj_spec(self) -> "CamProjSpec":
        """
        Get the projection properties of the camera used to create the samples in the dataset.
        """
        ...


class SampleTypeError(Exception):
    """An error that is raised when the specified sample type is not valid."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
