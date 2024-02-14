import math
from dataclasses import dataclass
from email.mime import base
from enum import Enum, auto
from pathlib import Path
from pickle import UnpicklingError
from typing import Callable, NamedTuple

import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import NodePath

from ..dataset_model import ExactSampleCounts, SampleType, SampleTypeError
from ..tensor_types.idx import *
from ..tensor_types.npy import *
from ._areas import ScaledStandingAreas
from ._custom_show_base import Panda3dShowBase
from ._data import ThreeDPoint, ThreeDSize
from ._errors import Panda3dAssumptionViolation
from ._scene_util import (
    get_ob_size_from_vertices,
    get_vertex_count,
    get_vertex_positions_copy,
)


class ViewpointBasedCamController:
    """
    A class that controls the camera of a standard scene using the predefined viewpoints. This class can exclude the viewpoints in which the center of the near plane of the camera may be inside of the target area.

    The class does not assume exclusive control over the controlled camera.

    This class acquires the size of the target object, the distance of the near plane of the camera and the vertexes that specify the viewpoint once. This means that the operations of this class are not affected by the subsequent changes of the scene.

    The constructor only initializes the class itself. It does not change the camera position automatically.

    Parameters
    ----------
    base
        The controlled showbase.
    viewpt_split
        The viewpoint split to use. This class stores a deep copy of this argument.

    Raises
    ------
    ValueError
        If the scene does not match to the standard scene format or there is no viewpoint outside of the area of the target object.
    Panda3dAssumptionViolation
        If the near plane distance is negative.
    """

    def __init__(
        self,
        base: Panda3dShowBase,
        viewpt_split: "ViewpointSplit",
    ) -> None:
        scene_format_errors = base.get_standard_scene_format_errors()
        if len(scene_format_errors) > 0:
            raise ValueError(
                f"The specified scene does not match to the standard scene format. Errors: {scene_format_errors}"
            )

        self.__total_viewpts = viewpt_split.copy()

        target_obj = base.get_target_obj_mesh_path()

        near_plane_distance = base.cam.node().getLens().near
        if near_plane_distance < 0:
            raise Panda3dAssumptionViolation(
                f"The near plane distance ({near_plane_distance}) is negative."
            )
        self.near_plane_distance = near_plane_distance

        target_pos_vec = base.get_target_obj_mesh_path().getParent().getPos()
        self.__target_position = ThreeDPoint(
            x=target_pos_vec.x, y=target_pos_vec.y, z=target_pos_vec.z
        )
        self.base = base

        vertex_positions = get_vertex_positions_copy(target_obj)
        self.__original_target_obj_size = get_ob_size_from_vertices(vertex_positions)

        self.update_target_area(1)

    def get_original_target_obj_size(self) -> ThreeDSize:
        return self.__original_target_obj_size

    def get_target_position(self) -> ThreeDPoint:
        return self.__target_position

    def get_extra_offset_due_to_near_plane(self) -> float:
        return 2 * self.near_plane_distance

    def update_target_area(self, size_multiplier: float) -> ScaledStandingAreas:
        """
        Change the area around the target object that should be excluded.

        This function does not modify the currently set camera position. In other words, it is possible that the camera may be placed inside the excluded area around the target object after this function.

        Parameters
        ----------
        size_multiplier
            The new size multiplier for the standing area around the target object.

        Returns
        -------
        v
            The calculated scaled standing area.

        Raises
        ------
        ValueError
            If the size multiplier is less than 1.
        """
        target_area = ScaledStandingAreas(
            original_size=self.__original_target_obj_size,
            size_multiplier=size_multiplier,
            extra_offset_after_size_mult=self.get_extra_offset_due_to_near_plane(),
        )
        filtered_viewpts = self.__total_viewpts.transform_or_filter(
            lambda a: target_area.get_full_area(
                origin_of_obj=self.__target_position, include_outside_offset=True
            ).select_points_outside(points=a)
        )

        self.__filtered_viewpts = filtered_viewpts

        return target_area

    def select_viewpoint(
        self, filtered_viewpoint_index: int, viewpt_type: "SampleType"
    ) -> None:
        """
        Place the camera to the specified viewpoint. This function also modifies the camera rotations to look at the target object.

        Parameters
        ----------
        filtered_viewpoint_index
            The index of the viewpoint among the viewpoints that are outside of the scaled area around the target object. All index values in range ``{0, 1, ..., n_filtered_viewpoints}`` are valid. Negative indexes are supported.

        Raises
        ------
        IndexError
            If the index is out of range.
        ValueError
            If there is no selectable viewpoint for the specified type.
        """
        if self.__filtered_viewpts.get_viewpt_count(viewpt_type) == 0:
            raise ValueError(
                f'There is no selectable viewpoint for type "{viewpt_type.public_name}".'
            )
        filtered_viewpoints = self.__filtered_viewpts.get_viewpts_by_type(viewpt_type)
        raw_viewpoint = filtered_viewpoints[filtered_viewpoint_index].astype(float)
        viewpoint = ThreeDPoint(
            x=float(raw_viewpoint[CAT_POINTS_SPACE_DATA_X]),
            y=float(raw_viewpoint[CAT_POINTS_SPACE_DATA_Y]),
            z=float(raw_viewpoint[CAT_POINTS_SPACE_DATA_Z]),
        )
        self.base.set_cam_pos_and_look_at(
            new_cam_pos=viewpoint, look_at=self.__target_position
        )

    def get_filtered_viewpoint_count(self, viewpt_type: "SampleType") -> int:
        """Get the number of the viewpoints that are outside of the scaled area around the target object for the specified type."""
        return self.__filtered_viewpts.get_viewpt_count(viewpt_type)

    def get_filtered_viewpoint_counts(self) -> "ExactSampleCounts":
        """Get the number of the viewpoints that are outside of the scaled area around the target object for each type."""
        return self.__filtered_viewpts.get_viewpt_counts()


class DesiredViewpointCounts:
    """
    A named tuple that specifies the required viewpoint counts for each viewpoint type.

    This requirement is exact if the value is not None. If the value is none, then there is no required viewpoint count specified.
    """

    def __init__(
        self,
        n_train_samples: int | None,
        n_val_samples: int | None,
        n_test_samples: int | None,
    ):
        if n_train_samples is not None:
            if n_train_samples < 0:
                raise ValueError(
                    f"The number of desired training samples ({n_train_samples}) is negative."
                )
        if n_test_samples is not None:
            if n_test_samples < 0:
                raise ValueError(
                    f"The number of desired testing samples ({n_test_samples}) is negative."
                )
        if n_val_samples is not None:
            if n_val_samples < 0:
                raise ValueError(
                    f"The number of desired validation samples ({n_val_samples}) is negative."
                )

        self.__n_train_samples = n_train_samples
        self.__n_val_samples = n_val_samples
        self.__n_test_samples = n_test_samples

    @property
    def n_train_samples(self) -> int | None:
        return self.__n_train_samples

    @property
    def n_val_samples(self) -> int | None:
        return self.__n_val_samples

    @property
    def n_test_samples(self) -> int | None:
        return self.__n_test_samples

    def to_exact(
        self, total_viewpoint_counts: "ExactSampleCounts"
    ) -> "ExactSampleCounts":
        """
        Substitute the non-specified viewpoint counts from the specified total viewpoint counts.

        Parameters
        ----------
        total_viewpoint_counts
            The total viewpoint counts.
        """
        return ExactSampleCounts(
            n_train_samples=self.n_train_samples
            if self.n_train_samples is not None
            else total_viewpoint_counts.n_train_samples,
            n_test_samples=self.n_test_samples
            if self.n_test_samples is not None
            else total_viewpoint_counts.n_test_samples,
            n_val_samples=self.n_val_samples
            if self.n_val_samples is not None
            else total_viewpoint_counts.n_val_samples,
        )

    def __str__(self) -> str:
        return f"DesiredViewpointCounts(n_train_samples={self.n_train_samples}, n_test_samples={self.n_test_samples}, n_val_samples={self.n_val_samples})"

    def __repr__(self) -> str:
        return self.__str__()


class ViewpointSplit:
    """
    A split of the viewpoints in a scene.

    The splis are saved to npz files. The keys are the fields of this class. The values are the points with data type `numpy.float32`.

    Parameters
    ----------
    train_viewpoints
        The viewpoints for training. Format: ``Points::Space``
    test_viewpoints
        The viewpoints for testing. Format: ``Points::Space``
    val_viewpoints
        The viewpoints for validation. Format: ``Points::Space``
    """

    VIEWPOINT_POS_SERIALIZATION_TYPE = np.float32

    def __init__(
        self,
        train_viewpoints: np.ndarray,
        test_viewpoints: np.ndarray,
        val_viewpoints: np.ndarray,
    ):
        self.train_viewpoints = train_viewpoints
        """
        The viewpoints for training. Format: ``Points::Space``
        """
        self.test_viewpoints = test_viewpoints
        """
        The viewpoints for testing. Format: ``Points::Space``
        """
        self.val_viewpoints = val_viewpoints
        """
        The viewpoints for validation. Format: ``Points::Space``
        """

    def copy(self) -> "ViewpointSplit":
        """Create a deep copy of this split."""
        return ViewpointSplit(
            train_viewpoints=self.train_viewpoints.copy(),
            test_viewpoints=self.test_viewpoints.copy(),
            val_viewpoints=self.val_viewpoints.copy(),
        )

    def transform_or_filter(
        self, fn: Callable[[np.ndarray], np.ndarray]
    ) -> "ViewpointSplit":
        return ViewpointSplit(
            train_viewpoints=fn(self.train_viewpoints),
            test_viewpoints=fn(self.test_viewpoints),
            val_viewpoints=fn(self.val_viewpoints),
        )

    def select_n(
        self,
        viewpt_counts: DesiredViewpointCounts,
    ) -> "ViewpointSplit":
        """
        Select the first n viewpoints.

        This function does not change this instance, it returns with a new instance instead.

        Parameters
        ----------
        viewpt_counts
            The number of training, testing and validation viewpoints.

        Returns
        -------
        v
            The selected viewpoints.

        Raises
        ------
        ValueError
            If the number of viewpoints to select is greater than the current number of viewpoints.
        """
        select_n_errors = self.get_select_n_errors(viewpt_counts)
        if len(select_n_errors) > 0:
            raise ValueError(
                f"The number of viewpoints to select is not valid. Errors: {select_n_errors}"
            )

        exact_viewpt_counts = viewpt_counts.to_exact(self.get_viewpt_counts())

        return ViewpointSplit(
            train_viewpoints=idx_points_space(
                self.train_viewpoints, n=slice(exact_viewpt_counts.n_train_samples)
            ),
            test_viewpoints=idx_points_space(
                self.test_viewpoints, n=slice(exact_viewpt_counts.n_test_samples)
            ),
            val_viewpoints=idx_points_space(
                self.val_viewpoints, n=slice(exact_viewpt_counts.n_val_samples)
            ),
        )

    def get_select_n_errors(self, viewpt_counts: DesiredViewpointCounts) -> list[str]:
        """
        Check whether it is possible to select the specified number of viewpoints in the viewpoint split.

        The selection is not possible if the number of viewpoints to select is greater than the current number of viewpoints.

        Parameters
        ----------
        viewpt_counts
            The number of training, testing and validation viewpoints.

        Returns
        -------
        v
            The errors. If it has length 0, then the selection is possible.

        See Also
        --------

        Notes
        -----
        """
        current_viewpt_counts = self.get_viewpt_counts()
        exact_viewpt_counts = viewpt_counts.to_exact(current_viewpt_counts)

        errors: list[str] = []

        if not exact_viewpt_counts.is_all_smaller_or_equal(current_viewpt_counts):
            errors += [
                f"At least one desired viewpoint count is greater than the original number of viewpoints. Original number of viewpoints: {current_viewpt_counts}, desired number of viewpoints: {viewpt_counts}"
            ]

        return errors

    @staticmethod
    def load_npz(file_path: Path) -> "ViewpointSplit":
        """
        Load the viewpoint split from an npz file.

        Parameters
        ----------
        file_path
            The path of the npz file.

        Returns
        -------
        v
            The loaded split.

        Raises
        ------
        SplitFormatError
            If it was not possible to load the split.
        """

        try:
            with np.load(file_path) as data:

                def get_points_array(key: str) -> np.ndarray:
                    """
                    Returns
                    -------
                    v
                        Format: ``Points::Space``

                    Raises
                    ------
                    SplitFormatError
                        If the specified key does not exist or the array does not contain points.
                    """
                    if key not in data.keys():
                        raise SplitFormatError(
                            f'The file does not contain the key "{key}"'
                        )

                    array_candidate = data[key]

                    if not isinstance(array_candidate, np.ndarray):
                        raise SplitFormatError(
                            f'The value for key "{key}" is not an array.'
                        )

                    if not match_points_space(
                        array_candidate,
                        dtype=ViewpointSplit.VIEWPOINT_POS_SERIALIZATION_TYPE,
                    ):
                        raise SplitFormatError(
                            f'The value for key "{array_candidate}" is an array, but it does not contain points.'
                        )

                    return array_candidate

                train_viewpoints = get_points_array("train_viewpoints")
                test_viewpoints = get_points_array("test_viewpoints")
                val_viewpoints = get_points_array("val_viewpoints")
        except OSError as e:
            raise SplitFormatError(
                f'Failed to load the viewpoint split from "{file_path}"'
            ) from e
        except UnpicklingError as e:
            raise SplitFormatError(
                f'Failed to load the viewpoint split from "{file_path}"'
            ) from e
        except ValueError as e:
            raise SplitFormatError(
                f'Failed to load the viewpoint split from "{file_path}"'
            ) from e

        return ViewpointSplit(
            train_viewpoints=train_viewpoints,
            test_viewpoints=test_viewpoints,
            val_viewpoints=val_viewpoints,
        )

    def save_npz(self, file_path: Path) -> None:
        """
        Save the split to an ``.npz`` file.

        Parameters
        ----------
        file_path
            The path of the file.
        """

        data = {
            "train_viewpoints": self.train_viewpoints,
            "test_viewpoints": self.test_viewpoints,
            "val_viewpoints": self.val_viewpoints,
        }
        with file_path.open("+wb") as f:
            np.savez(f, **data)

    @staticmethod
    def get_generate_errors(
        base: Panda3dShowBase, viewpt_counts: ExactSampleCounts
    ) -> list[str]:
        """
        Return a list of errors that prevent the viewpoint split generation.

        If the list is empty, then the wiewpoint generation is possible.

        Possible errors:

        * The scene does not follow the standard scene format.
        * The viewpoints object is empty.
        * The sum of the number of the viewpoints with different types is not equal the number of viewpoints in the scene.
        """
        errors = []

        scene_format_errors = base.get_standard_scene_format_errors()
        if len(scene_format_errors) > 0:
            for scene_format_error in scene_format_errors:
                errors.append(
                    f"The scene does not follow the standard scene format. Error: {scene_format_error}"
                )

        viewpoints_mesh = base.get_viewpoints_obj_mesh_path()
        n_viewpoints = get_vertex_count(viewpoints_mesh)

        if viewpt_counts.sum() != n_viewpoints:
            errors.append(
                f"The sum of the number of the viewpoints in the splits is not equal to the total number of viewpoints in the scene. Viewpoint counts: (train.: {viewpt_counts.n_train_samples}, test: {viewpt_counts.n_test_samples}, val.: {viewpt_counts.n_val_samples}); total number of viewpoints in the scene: {n_viewpoints}"
            )

        return errors

    @staticmethod
    def generate(
        base: Panda3dShowBase,
        viewpt_counts: ExactSampleCounts,
        seed: int | None,
    ) -> "ViewpointSplit":
        """
        Generate a viewpoint split for the specified scene.

        This function shuffles the order of the viewpoints.

        Parameters
        ----------
        base
            The scene to which the viewpoint split should be generated.
        viewpt_counts
            The number of viewpoints to generate. The sum of the number of the viewpoints in the splits should be equal to the total number of viewpoints in the scene.
        seed
            The seed to use for the shuffling operation.

        Returns
        -------
        v
            The created split. ``Points::Space``

        Raises
        ------
        ValueError
            If the scene does not follow the standard scene format or the sum of the number of the viewpoints in the splits is not the total number of viewpoints in the scene.
        """
        generate_errors = ViewpointSplit.get_generate_errors(base, viewpt_counts)
        if len(generate_errors) > 0:
            raise ValueError(
                f"It is not possible to generate a viewpoint split with this configuration. Errors: {generate_errors}"
            )

        viewpoints_mesh = base.get_viewpoints_obj_mesh_path()
        viewpoints = get_vertex_positions_copy(viewpoints_mesh)
        n_viewpoints = viewpoints.shape[DIM_POINTS_N]
        rng = np.random.default_rng(seed)
        viewpoint_indices = np.arange(n_viewpoints)
        rng.shuffle(viewpoint_indices, axis=0)

        train_viewpoint_indices = viewpoint_indices[0 : viewpt_counts.n_train_samples]
        test_viewpoint_indces = viewpoint_indices[
            viewpt_counts.n_train_samples : viewpt_counts.n_train_samples
            + viewpt_counts.n_test_samples
        ]
        val_viewpoint_indces = viewpoint_indices[
            viewpt_counts.n_train_samples
            + viewpt_counts.n_test_samples : viewpt_counts.n_train_samples
            + viewpt_counts.n_test_samples
            + viewpt_counts.n_val_samples
        ]

        train_viewpoints = idx_points_space(viewpoints, n=train_viewpoint_indices)
        test_viewpoints = idx_points_space(viewpoints, n=test_viewpoint_indces)
        val_viewpoints = idx_points_space(viewpoints, n=val_viewpoint_indces)

        return ViewpointSplit(
            train_viewpoints=train_viewpoints,
            test_viewpoints=test_viewpoints,
            val_viewpoints=val_viewpoints,
        )

    def get_viewpt_counts(self) -> ExactSampleCounts:
        """
        Get the number of viewpoints for all viewpoint types.
        """
        return ExactSampleCounts(
            n_train_samples=self.train_viewpoints.shape[DIM_POINTS_N],
            n_test_samples=self.test_viewpoints.shape[DIM_POINTS_N],
            n_val_samples=self.val_viewpoints.shape[DIM_POINTS_N],
        )

    def get_viewpt_count(self, viewpt_type: SampleType) -> int:
        return self.get_viewpts_by_type(viewpt_type).shape[DIM_POINTS_N]

    def get_viewpts_by_type(self, viewpt_type: SampleType) -> np.ndarray:
        """
        Get the viewpoints for the specified viewpoint type.

        This function does not copy the original arrays.

        Parameters
        ----------
        viewpt_type
            The type of the viewpoints to get.

        Returns
        -------
        v
            The points. Format: ``Points::Space``
        """
        match viewpt_type:
            case SampleType.Train:
                return self.train_viewpoints
            case SampleType.Test:
                return self.test_viewpoints
            case SampleType.Val:
                return self.val_viewpoints


class SplitFormatError(Exception):
    """An error that specifies that the file of the viewpoint split is not valid."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def get_viewpoints_path_for_world(glb_path: Path) -> Path:
    return glb_path.with_stem(glb_path.stem + "_viewpoints").with_suffix(".npz")
