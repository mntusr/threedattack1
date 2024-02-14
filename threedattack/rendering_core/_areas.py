from typing import Optional

import numpy as np

from ..tensor_types.idx import (
    CAT_POINTS_SPACE_DATA_X,
    CAT_POINTS_SPACE_DATA_Y,
    CAT_POINTS_SPACE_DATA_Z,
)
from ..tensor_types.npy import *
from ..tensor_types.npy import idx_points, upd_points
from ._data import ThreeDPoint, ThreeDSize


class ScaledStandingAreas:
    """
    The class that handles the numerous bounding boxes of objects that stand on the floor.

    These objects are special, since when you scale their bounding boxes, these bounding boxes should not be scaled into the ground. This means that the vertical scaling occures upwards twice, whilst the horizontal scaling is bidierenctional.

    The scaling of the bounding box does not imply scaling in the local coordinate system of the object. In other worlds, this class assumes that the world coordinates of the points of the object remain the same regardless of the scaling of the bounding box.

    This class assumes that neither the obejct nor its ancestors has any rotation or scaling in the world coordinate system.

    The origin of the object is assumed to be at the center of the UNSCALED bounding box. The class does not store the position of the object origin in the world coordinate system. The reason is that this is not always important, since some calculations might happen in the local coordinate system of the object.

    Bounding boxes:

    * The original bounding box of the object in the local coordinate system of the object.
    * The properly scaled bounding box of the object in the local coordinate system of the object.
    * The properly scaled bounding box of the object in the local coordinate system of the object with the extra offset added.
    * The bounding boxes above in the global coordinate system.

    Parameters
    ----------
    original_size
        The original size of the bounding box of the object.
    size_multiplier
        The scale of the bounding box.
    extra_offset_after_size_mult
        An additional absolute offset around the scaled area. This offset is not contained by the scaled area, but it is excluded too when the points outside of the scale area are filtered.

    Raises
    ------
    ValueError
        If the size multiplier is less than 1.
    """

    def __init__(
        self,
        original_size: ThreeDSize,
        size_multiplier: float,
        extra_offset_after_size_mult: float,
    ) -> None:
        if size_multiplier < 1:
            raise ValueError("The size multiplier should be greater than 1.")

        self.__original_size = original_size
        self.__size_multiplier = size_multiplier
        self.__extra_offset_after_size_mult = extra_offset_after_size_mult

    def get_original_size(self) -> ThreeDSize:
        return self.__original_size

    def get_size_multiplier(self) -> float:
        return self.__size_multiplier

    def get_extra_offset_after_size_mult(self) -> float:
        return self.__extra_offset_after_size_mult

    def _get_x_min_max_full(
        self, include_outside_offset: bool, origin_of_obj: "Optional[ThreeDPoint]"
    ) -> tuple[float, float]:
        """
        Get the minimum and maximum x coordinates inside of the bounding box in the specified coordinate system. If the origin of the object is specified, then the world coordinate system will be used. Otherwise the local coordinate system will be used.

        Parameters
        ----------
        origin_of_obj
            The origin of the object in the world coordinate system.
        include_outside_offset
            True if the extra offset outside of the area should be contained by the ``[y_min, y_max]`` range.

        Returns
        -------
        x_min
            The minimum x coordinate.
        x_max
            The maximum x coordinate.
        """
        obj_orig_x = origin_of_obj.x if origin_of_obj is not None else 0

        dx = self.__original_size.x_size / 2 * self.__size_multiplier

        if include_outside_offset:
            dx += self.__extra_offset_after_size_mult
        return obj_orig_x - dx, obj_orig_x + dx

    def _get_y_min_max_full(
        self, include_outside_offset: bool, origin_of_obj: "Optional[ThreeDPoint]"
    ) -> tuple[float, float]:
        """
        Get the minimum and maximum Y coordinates inside of the bounding box in the specified coordinate system. If the origin of the object is specified, then the world coordinate system will be used. Otherwise the local coordinate system will be used.

        Parameters
        ----------
        origin_of_obj
            The origin of the object in the world coordinate system.
        include_outside_offset
            True if the extra offset outside of the area should be contained by the ``[y_min, y_max]`` range.

        Returns
        -------
        y_min
            The minimum Y coordinate.
        y_max
            The maximum Y coordinate.
        """
        obj_orig_y = origin_of_obj.y if origin_of_obj is not None else 0

        dy = self.__original_size.y_size / 2 * self.__size_multiplier

        if include_outside_offset:
            dy += self.__extra_offset_after_size_mult

        return obj_orig_y - dy, obj_orig_y + dy

    def _get_z_min_max_full(
        self, include_outside_offset: bool, origin_of_obj: "Optional[ThreeDPoint]"
    ) -> tuple[float, float]:
        """
        Get the minimum and maximum Z coordinates inside of the bounding box in the specified coordinate system. If the origin of the object is specified, then the world coordinate system will be used. Otherwise the local coordinate system will be used.

        Parameters
        ----------
        origin_of_obj
            The origin of the object in the world coordinate system.
        include_outside_offset
            True if the extra offset outside of the area should be contained by the ``[z_min, z_max]`` range.

        Returns
        -------
        z_min
            The minimum Z coordinate.
        z_max
            The maximum Z coordinate.
        """
        obj_orig_z = origin_of_obj.z if origin_of_obj is not None else 0

        dz_up = self.__original_size.z_size / 2 * (1 + 2 * (self.__size_multiplier - 1))
        dz_down = self.__original_size.z_size / 2

        if include_outside_offset:
            dz_up += self.__extra_offset_after_size_mult
            dz_down += self.__extra_offset_after_size_mult

        return obj_orig_z - dz_down, obj_orig_z + dz_up

    def get_full_area(
        self, origin_of_obj: ThreeDPoint | None, include_outside_offset: bool
    ) -> "ThreeDArea":
        x_min, x_max = self._get_x_min_max_full(
            include_outside_offset=include_outside_offset, origin_of_obj=origin_of_obj
        )
        y_min, y_max = self._get_y_min_max_full(
            include_outside_offset=include_outside_offset, origin_of_obj=origin_of_obj
        )
        z_min, z_max = self._get_z_min_max_full(
            include_outside_offset=include_outside_offset, origin_of_obj=origin_of_obj
        )

        return ThreeDArea(
            x_bounds=(x_min, x_max),
            y_bounds=(y_min, y_max),
            z_bounds=(z_min, z_max),
        )

    def get_original_area(self, origin_of_obj: ThreeDPoint | None) -> "ThreeDArea":
        if origin_of_obj is None:
            origin_of_obj = ThreeDPoint(0, 0, 0)
        nonscaled_x_min = -self.__original_size.x_size / 2 + origin_of_obj.x
        nonscaled_x_max = self.__original_size.x_size / 2 + origin_of_obj.x
        nonscaled_y_min = -self.__original_size.y_size / 2 + origin_of_obj.y
        nonscaled_y_max = self.__original_size.y_size / 2 + origin_of_obj.y
        nonscaled_z_min = -self.__original_size.z_size / 2 + origin_of_obj.z
        nonscaled_z_max = self.__original_size.z_size / 2 + origin_of_obj.z

        return ThreeDArea(
            x_bounds=(nonscaled_x_min, nonscaled_x_max),
            y_bounds=(nonscaled_y_min, nonscaled_y_max),
            z_bounds=(nonscaled_z_min, nonscaled_z_max),
        )


class ThreeDArea:
    """
    A coordinate system independent implementation of 3D areas.

    Parameters
    ----------
    x_bounds
        The bounds of the area on the X axis.
    y_bounds
        The bounds of the area on the Y axis.
    z_bounds
        The bounds of the area on the Z axis.

    Raises
    ------
    ValueError
        If any of the lower bounds is greater than the corresponding upper bound.
    """

    def __init__(
        self,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        z_bounds: tuple[float, float],
    ):
        if x_bounds[1] < x_bounds[0]:
            raise ValueError(
                f"The lower bound ({x_bounds[0]}) on the X axis is greater than the upper bound ({x_bounds[1]})"
            )
        if y_bounds[1] < y_bounds[0]:
            raise ValueError(
                f"The lower bound ({y_bounds[0]}) on the Y axis is greater than the upper bound ({y_bounds[1]})"
            )
        if z_bounds[1] < z_bounds[0]:
            raise ValueError(
                f"The lower bound ({z_bounds[0]}) on the Z axis is greater than the upper bound ({z_bounds[1]})"
            )

        self.__x_bounds = x_bounds
        self.__y_bounds = y_bounds
        self.__z_bounds = z_bounds

    def get_x_bounds(self) -> tuple[float, float]:
        """
        Get the lower and upper bounds on the X axis.
        """
        return self.__x_bounds

    def get_y_bounds(self) -> tuple[float, float]:
        """
        Get the lower and upper bounds on the X axis.
        """
        return self.__y_bounds

    def get_z_bounds(self) -> tuple[float, float]:
        """
        Get the lower and upper bounds on the X axis.
        """
        return self.__z_bounds

    def clip_points_to_inside(
        self,
        abs_points: np.ndarray,
    ) -> np.ndarray:
        """
        Clip the absolute points to be inside of the area.

        Parameters
        ----------
        abs_points
            The points to clip in the local or world coordinate system depending on the specified origin. Format: ``Points::Space``


        Returns
        -------
        v
            The clipped points.
        """
        x_min, x_max = self.__x_bounds
        y_min, y_max = self.__y_bounds
        z_min, z_max = self.__z_bounds

        result = np.zeros_like(abs_points)
        upd_points_space(
            result,
            data="x",
            value_=np.clip(idx_points_space(abs_points, data="x"), x_min, x_max),
        )
        upd_points_space(
            result,
            data="y",
            value_=np.clip(idx_points_space(abs_points, data="y"), y_min, y_max),
        )
        upd_points_space(
            result,
            data="z",
            value_=np.clip(idx_points_space(abs_points, data="z"), z_min, z_max),
        )
        return result

    def select_points_outside(self, points: np.ndarray) -> np.ndarray:
        """
        Select the points that are outside of the scaled area and the extra area around it.

        Parameters
        ----------
        points
            The points to filter. Format: ``Points::Space``

        Returns
        -------
        v
            The filtered points. Format: ``Points::Space``
        """
        x_min, x_max = self.__x_bounds
        y_min, y_max = self.__y_bounds
        z_min, z_max = self.__z_bounds

        return points[
            ~(
                (idx_points(points, data=CAT_POINTS_SPACE_DATA_X) >= x_min)
                & (idx_points(points, data=CAT_POINTS_SPACE_DATA_X) <= x_max)
                & (idx_points(points, data=CAT_POINTS_SPACE_DATA_Y) >= y_min)
                & (idx_points(points, data=CAT_POINTS_SPACE_DATA_Y) <= y_max)
                & (idx_points(points, data=CAT_POINTS_SPACE_DATA_Z) >= z_min)
                & (idx_points(points, data=CAT_POINTS_SPACE_DATA_Z) <= z_max)
            )
        ]

    def make_rel_points_absolute(self, rel_points: np.ndarray) -> np.ndarray:
        """
        Calculate the positions of the "relative" points.

        The pseudocode of the formula:

        * ``x_abs := x_min+x_rel*(x_max-x_min)``
        * ``y_abs := y_min+y_rel*(y_max-y_min)``
        * ``z_abs := z_min+z_rel*(z_max-z_min)``

        Parameters
        ----------
        rel_points
            The "relative" points. Format: ``points::space``
        origin_of_obj
            The origin of the object in the world coordinate system.


        Returns
        -------
        v
            The exact coordinates of the points. Format: ``points::space``
        """
        abs_points = np.zeros_like(rel_points)

        min_x, max_x = self.__x_bounds
        min_y, max_y = self.__y_bounds
        min_z, max_z = self.__z_bounds

        upd_points(
            abs_points,
            data=CAT_POINTS_SPACE_DATA_X,
            value_=min_x
            + idx_points(rel_points, data=CAT_POINTS_SPACE_DATA_X) * (max_x - min_x),
        )
        upd_points(
            abs_points,
            data=CAT_POINTS_SPACE_DATA_Y,
            value_=min_y
            + idx_points(rel_points, data=CAT_POINTS_SPACE_DATA_Y) * (max_y - min_y),
        )
        upd_points(
            abs_points,
            data=CAT_POINTS_SPACE_DATA_Z,
            value_=min_z
            + idx_points(rel_points, data=CAT_POINTS_SPACE_DATA_Z) * (max_z - min_z),
        )

        return abs_points

    def get_corners(self) -> np.ndarray:
        """
        Get the corner points of the area.

        Returns
        -------
        v
            The corner points of the area. Format: ``Points::Space``
        """

        x_min, x_max = self.__x_bounds
        y_min, y_max = self.__y_bounds
        z_min, z_max = self.__z_bounds

        return np.array(
            [
                [x_min, y_min, z_min],
                [x_min, y_min, z_max],
                [x_min, y_max, z_min],
                [x_min, y_max, z_max],
                [x_max, y_min, z_min],
                [x_max, y_min, z_max],
                [x_max, y_max, z_min],
                [x_max, y_max, z_max],
            ],
            dtype=np.float32,
        )

    def get_min_size(self) -> float:
        """
        Get the minimum of the sizes of the object alongside the dimensions.
        """
        x_range = self.__x_bounds[1] - self.__x_bounds[0]
        y_range = self.__y_bounds[1] - self.__y_bounds[0]
        z_range = self.__z_bounds[1] - self.__z_bounds[0]

        return min(x_range, y_range, z_range)

    def __eq__(self, other: "ThreeDArea"):
        return (
            (self.__x_bounds == other.__x_bounds)
            and (self.__y_bounds == other.__y_bounds)
            and (self.__z_bounds == other.__z_bounds)
        )

    def __ne__(self, other: "ThreeDArea") -> bool:
        return not self.__eq__(other)
