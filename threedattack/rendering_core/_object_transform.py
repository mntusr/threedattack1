import copy
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from panda3d.core import GeomNode, NodePath
from scipy.interpolate import LinearNDInterpolator

from .._typing import type_instance
from ..tensor_types.npy import *
from ._areas import ScaledStandingAreas
from ._data import VertsAndFaces
from ._errors import Panda3dAssumptionViolation
from ._object_creation import verts_and_faces_2_obj
from ._occupacy_fields import (
    InterpolatedOccupacyField,
    OccupacyFieldSamples,
    WarpField,
    get_d_coord_from_obj_size_along_the_shortest_axis,
    occupacy_field_2_occupacy_field_samples,
    occupacy_field_samples_2_verts_and_faces,
    verts_and_faces_2_obj,
)
from ._scene_util import get_vertices_and_faces_copy, is_geom_node_obj


class ObjectTransform(Protocol):
    def transform_obj_new(
        self,
        vector_field_spec: "PointBasedVectorFieldSpec",
    ) -> "ObjectTransformResult":
        """
        Create a new transformed copy of the original object.

        The transform is controlled by an interpolated vector field. The interpolation method is linear interpolation.

        The exact effect of the vector field depends on the exact implementing class.

        The transformation is evaluated in the local coordinate system of the object.

        The name and the material of the created object is the same as the name of the original object.

        Parameters
        ----------
        vector_field_spec
            The specification of the vector field to use.

        Returns
        -------
        v
            The transformed copy of the object and the calculated statistics.

        Raises
        ------
        Panda3dAssumptionViolation
            The exact condition depends on the implementation.
        """
        ...

    def get_transform_type(self) -> "ObjectTransformType":
        """
        Get the enum that describes the transform type.
        """
        ...

    def get_initial_obj_deepcopy(self) -> NodePath:
        """
        Create a copy of the initial transformed object.
        """
        ...


class MeshBasedObjectTransform:
    """
    A class that transforms the specified Panda3d object purely using its mesh.


    Transformation algorithm: Add the vectors specified by the interpolated vector field to the positions of the vertices.

    Change amount score function: The maximal change of the position of the transformed vertexes alongside a single axis / the minimal size of the object along the X, Y and Z axes.

    Parameters
    ----------
    obj
        The object to transform.
    transformed_obj_areas
        The different bounding boxes of the target object.
    """

    def __init__(self, obj: NodePath, transformed_obj_areas: ScaledStandingAreas):
        self.__obj_copy = copy.deepcopy(obj)
        self.__transformed_obj_areas = transformed_obj_areas

    def get_initial_obj_deepcopy(self) -> NodePath:
        return copy.deepcopy(self.__obj_copy)

    def get_transform_type(self) -> "ObjectTransformType":
        return ObjectTransformType.MeshBased

    def transform_obj_new(
        self,
        vector_field_spec: "PointBasedVectorFieldSpec",
    ) -> "ObjectTransformResult":
        verts_and_faces = get_vertices_and_faces_copy(self.__obj_copy)

        def transform_fn(verts: np.ndarray) -> np.ndarray:
            interpolator = LinearNDInterpolator(
                points=vector_field_spec.control_points,
                fill_value=0,
                values=vector_field_spec.vectors,
            )
            changes = interpolator(verts)
            new_positions = verts + changes

            new_positions = self.__transformed_obj_areas.get_full_area(
                origin_of_obj=None, include_outside_offset=False
            ).clip_points_to_inside(new_positions)

            return new_positions

        new_verts = transform_fn(verts_and_faces.vertices)

        new_obj = verts_and_faces_2_obj(
            verts_and_faces=VertsAndFaces(
                vertices=new_verts, faces=verts_and_faces.faces
            ),
            name=self.__obj_copy.name,
            render_state_source=self.__obj_copy,
        )

        change_amount_score = self._change_amount_score_fn(
            vert_pos_changes=verts_and_faces.vertices - new_verts,
        )

        return ObjectTransformResult(
            new_obj=new_obj, n_bodies=1, change_amount_score=change_amount_score
        )

    def _change_amount_score_fn(self, vert_pos_changes: np.ndarray) -> float:
        """
        The calculation of the change amount score function of this class.

        Parameters
        ----------
        vert_pos_changes
            The differences between the original positions of the vertexes and their new positions. Format: ``Points::Space``

        Returns
        -------
        v
            The calculated change amount score.
        """
        original_min_size = self.__transformed_obj_areas.get_original_area(
            origin_of_obj=None
        ).get_min_size()
        return float(abs(vert_pos_changes).max()) / original_min_size


MIN_CONTROL_POINTS_FOR_WARP = 5
"""
The minimum number of control points to enable object warping.
"""


class VolumeBasedObjectTransform:
    """
    Transform the original object using its occupacy function instead of its mesh. The transformed occupacy function is then converted to a Panda3d object using the Marching Cubes algorithm.

    Transformation algorithm: ``occupacy_function_of_original_obj(points+vector_field(points))``

    Change amount score function: ``n_changed/n_in0``, where:

    * ``n_changed``: The number of sampled points that changed due to the application of the transformation.
    * ``n_in0``: The total number of sampled points inside of the non-transformed object.

    Parameters
    ----------
    obj
        The object to transform.
    field_cache
        The samples of the occupacy field of the object to transform.
    n_cubes_steps
        The number of steps alongside each axis at the application of the Marching Cubes algorithm.
    n_steps_along_shortest_axis
        The number of sampling steps alongside the smallest size alongside the dimensions of the original object.

    Raises
    ------
    Panda3dAssumptionViolation
        If the node of the object to transfor is not a GeomNode.
        If the object to transform has children.
        The object to transform has more than one geoms.
    ValueError
        If there is no object according to the specified occupacy field cache.
        If ``n_steps_along_shortest_axis<=0``
    """

    def __init__(
        self,
        obj: NodePath,
        field_cache: OccupacyFieldSamples,
        n_steps_along_shortest_axis: int,
        target_obj_areas: ScaledStandingAreas,
    ):
        if not is_geom_node_obj(obj):
            raise Panda3dAssumptionViolation(
                f'The node of the object to transform ("{obj.name}") is not a GeomNode.'
            )
        if obj.getNumChildren() > 0:
            raise Panda3dAssumptionViolation(
                f'The object to transform ("{obj.name}") has children.'
            )
        if obj.node().getNumGeoms() != 1:
            raise Panda3dAssumptionViolation(
                f'The object to transform ("{obj.name}") has more than one geoms.'
            )
        if field_cache.contains_no_object():
            raise ValueError(
                "The specified occupacy field does not describe any object."
            )
        # TODO validate that the specified bounds contain the full object
        if n_steps_along_shortest_axis <= 0:
            raise ValueError(
                f"The number of steps alongside the shortest dimension should be positive. Current value: {n_steps_along_shortest_axis}"
            )

        self.__obj_copy: "NodePath[GeomNode]" = copy.deepcopy(obj)
        self.__object_field = InterpolatedOccupacyField(field_cache)
        self.__n_steps_along_shortest_axis = n_steps_along_shortest_axis

        d_coord = get_d_coord_from_obj_size_along_the_shortest_axis(
            area=target_obj_areas.get_original_area(origin_of_obj=None),
            n_steps_along_shortest_axis=n_steps_along_shortest_axis,
        )

        self.__orig_field_samples = occupacy_field_2_occupacy_field_samples(
            occupacy_field=self.__object_field,
            relevant_area=target_obj_areas.get_full_area(
                origin_of_obj=None, include_outside_offset=False
            ),
            d_coord=d_coord,
        )
        self.__target_obj_areas = target_obj_areas
        self.__d_coord = d_coord

    def get_n_steps_along_shortest_axis(self) -> int:
        """
        Get the number of sampling steps alongside the smallest size alongside the dimensions of the original object.
        """
        return self.__n_steps_along_shortest_axis

    def get_orig_field_samples_deepcopy(self) -> OccupacyFieldSamples:
        """
        Get a deep copy of the samples of the occupacy field of the original object. The provided samples use the same spacing as the samples used to generate the transformed object.
        """
        return copy.deepcopy(self.__orig_field_samples)

    def get_d_coord(self) -> float:
        """
        Get the difference between the neighbor sampling points on a single axis.
        """
        return self.__d_coord

    def get_transform_type(self) -> "ObjectTransformType":
        return ObjectTransformType.VolumeBased

    def get_initial_obj_deepcopy(self) -> NodePath:
        return copy.deepcopy(self.__obj_copy)

    def transform_obj_new(
        self,
        vector_field_spec: "PointBasedVectorFieldSpec",
    ) -> "ObjectTransformResult":
        """

        The transformed occupacy function is then converted to a new mesh using the Marching Cubes algorithm.

        Change amount score: ``(number of changed sampled points)/(number of sampled points in the original object)``
        """
        occupacy_fn = WarpField(
            control_points=vector_field_spec.control_points,
            vectors=vector_field_spec.vectors,
            field_fn=self.__object_field,
        )
        occupacy_field_samples = occupacy_field_2_occupacy_field_samples(
            occupacy_field=occupacy_fn,
            relevant_area=self.__target_obj_areas.get_full_area(
                origin_of_obj=None, include_outside_offset=False
            ),
            d_coord=self.__d_coord,
        )
        verts_and_faces = occupacy_field_samples_2_verts_and_faces(
            occupacy_field_samples
        )
        n_bodies = verts_and_faces.get_n_bodies()
        change_amount_score = self._change_amount_score_fn(
            new_obj_occup_field_samples_grid=occupacy_field_samples.grid
        )
        new_obj = verts_and_faces_2_obj(
            verts_and_faces=verts_and_faces,
            name=self.__obj_copy.name,
            render_state_source=self.__obj_copy,
        )

        return ObjectTransformResult(
            new_obj=new_obj, n_bodies=n_bodies, change_amount_score=change_amount_score
        )

    def _change_amount_score_fn(
        self, new_obj_occup_field_samples_grid: np.ndarray
    ) -> float:
        """
        The calculation of the change amount score function of this class.

        Parameters
        ----------
        new_obj_occup_field_samples_grid
            The sample grid of the resulting occupacy field. Format: ``FieldGrid::OccupacyFieldGrid``.

        Returns
        -------
        v
            The calculated change amount score.
        """
        old_binary_grid = self.__orig_field_samples.grid >= 0
        new_binary_grid = new_obj_occup_field_samples_grid >= 0

        n_changed = (
            np.logical_xor(old_binary_grid, new_binary_grid).astype(np.int32).sum()
        )
        n_in0 = old_binary_grid.sum()

        return n_changed / n_in0


def get_object_transform_by_type(
    transform_type: "ObjectTransformType",
    original_obj: NodePath,
    field_cache: OccupacyFieldSamples,
    n_volume_sampling_steps_along_shortest_axis: int,
    transform_bounds: ScaledStandingAreas,
) -> ObjectTransform:
    """
    Get the object transform function based on the specified object transform type.

    Parameters
    ----------
    original_obj
        The original object to transform.
    field_cache
        The cache of the occupacy field of the original object. Ignored if the mesh of the object is transformed.
    n_volume_sampling_steps_along_shortest_axis
        The number of sampling steps alongside the axis alongside which the parts of the object inside of the relevant area the smallest are. Ignored if the mesh of the object is transformed.
    transform_bounds
        The scaled standing area of the object to transform.
    """
    match transform_type:
        case ObjectTransformType.MeshBased:
            return MeshBasedObjectTransform(
                obj=original_obj, transformed_obj_areas=transform_bounds
            )
        case ObjectTransformType.VolumeBased:
            return VolumeBasedObjectTransform(
                obj=original_obj,
                field_cache=field_cache,
                n_steps_along_shortest_axis=n_volume_sampling_steps_along_shortest_axis,
                target_obj_areas=transform_bounds,
            )


class ObjectTransformType(Enum):
    """
    An enum that describes the types of the object transforms.
    """

    VolumeBased = ("volume_based",)
    """
    The occupacy function of the object is transformed.

    Algorithm: ``occupacy_function_of_original_obj(points+vector_field(points))``

    Change amount score function: ``n_changed/n_in0``, where:

    * ``n_changed``: The number of sampled points that changed due to the application of the transformation.
    * ``n_in0``: The total number of sampled points inside of the non-transformed object.
    """

    MeshBased = ("mesh_based",)
    """
    The mesh of the target object is transformed.
    
    Transformation algorithm: Add the vectors specified by the interpolated vector field to the positions of the vertices.

    Change amount score function: The maximal change of the position of the transformed vertexes alongside a single axis.
    """

    def __init__(self, public_name: str):
        self.public_name = public_name
        """
        The user-facing name of the object transform type.
        """


def get_target_obj_field_cache_path_for_world(world_path: Path) -> Path:
    return world_path.with_name(world_path.stem + "_target_field_cache.npz")


def get_object_transform_type_by_name(name: str) -> ObjectTransformType:
    """Get an object transform type by its public name."""
    for transform_type in ObjectTransformType:
        if transform_type.public_name == name:
            return transform_type
    else:
        all_names = get_supported_object_transform_type_names()
        raise ValueError(
            f'Unknown object transform type "{name}". Supported object transform types: {all_names}'
        )


def get_supported_object_transform_type_names() -> list[str]:
    """Get the public names of all object transform types."""
    return [transform_type.public_name for transform_type in ObjectTransformType]


@dataclass
class PointBasedVectorFieldSpec:
    """
    A structure that specifies an interpolated vector field using control points and the corresponding vectors.
    """

    control_points: np.ndarray
    """
    control_points
        The control points of the vector field. Format: ``Points::Space``
    """

    vectors: np.ndarray
    """
    vectors
        The vectors at the control points of the vector field. ``Points::Space``
    """


@dataclass
class ObjectTransformResult:
    new_obj: NodePath
    """
    The result of the transformation.
    """

    n_bodies: int
    """
    The number of separate bodies generated by the transform.
    """

    change_amount_score: float
    """
    A transformation-dependent number that describes the amount of change of the transformed object due to the transformation. Its calculation is dependent on the transformation operation, but the smaller values always mean less change and its minimum is always 0.
    """


if TYPE_CHECKING:
    v1: ObjectTransform = type_instance(MeshBasedObjectTransform)
    v1: ObjectTransform = type_instance(VolumeBasedObjectTransform)
