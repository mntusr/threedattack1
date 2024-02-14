from dataclasses import dataclass
from multiprocessing import Value
from typing import Any, NamedTuple

import numpy as np
import trimesh

from ..tensor_types.idx import *


@dataclass
class RGBWithZbuf:
    """
    A data structure that groups the RGB images and the corresponding Panda3d Z-buffers.
    """

    rgbs: np.ndarray
    """
    The RGB images.

    Format: ``Im::RGBs``
    """
    zbufs: np.ndarray
    """
    The corresponding Z-buffers for the RGB images.

    The amount of the Z-buffers is the same as the number of the RGB images.

    The sizes of the Z-buffers are the same as the sizes of the RGB images.

    Format: ``Im::ZBuffers``
    """


class ThreeDSize(NamedTuple):
    x_size: float
    y_size: float
    z_size: float


class ThreeDPoint(NamedTuple):
    x: float
    y: float
    z: float

    def is_almost_equal(self, other: "ThreeDPoint", epsilon: float) -> bool:
        """
        Returns true if ``max({|self.$d-other.$d|, where $d in {x, y, z}})<epsilon``
        """
        return (
            (abs(self.x - other.x) < epsilon)
            and (abs(self.y - other.y) < epsilon)
            and (abs(self.z - other.z) < epsilon)
        )


class TwoDSize:
    def __init__(self, x_size: int, y_size: int):
        if x_size < 0:
            raise ValueError(f"The argument x_size is negative ({x_size})")
        if y_size < 0:
            raise ValueError(f"The argument y_size is negative ({y_size})")

        self.__x_size = x_size
        self.__y_size = y_size

    @property
    def x_size(self):
        return self.__x_size

    @property
    def y_size(self):
        return self.__y_size

    def is_positive(self) -> bool:
        return (self.x_size > 0) and (self.y_size > 0)


class TwoDAreas(NamedTuple):
    x_mins_including: np.ndarray
    """
    The (including) upper lower for the x coordinates inside of the areas.
     
    The values might be negative or greater than the width of the images, but they are always smaller than `TwoDAreas.x_maxes_excluding`.
    
    Format: ``Scalars::Int``
    """

    x_maxes_excluding: np.ndarray
    """
    The (excluding) upper bounds for the x coordinates inside of the areas.
     
    The values might be negative or greater than the width of the images, but they are always greater than `TwoDAreas.x_mins_including`.
    
    Format: ``Scalars::Int``
    """

    y_mins_including: np.ndarray
    """
    The (including) lower for the y coordinates inside of the areas.
     
    The values might be negative or greater than the height of the images, but they are always smaller than `TwoDAreas.y_maxes_excluding`.
    
    Format: ``Scalars::Int``
    """

    y_maxes_excluding: np.ndarray
    """
    The (excluding) upper bounds for the y coordinates inside of the areas.
     
    The values might be negative or greater than the height of the images, but they are always greater than `TwoDAreas.y_mins_including`.
    
    Format: ``Scalars::Int``
    """

    def idx_areas(self, indices: Any) -> "TwoDAreas":
        return TwoDAreas(
            x_maxes_excluding=self.x_maxes_excluding[indices],
            x_mins_including=self.x_mins_including[indices],
            y_maxes_excluding=self.y_maxes_excluding[indices],
            y_mins_including=self.y_mins_including[indices],
        )


@dataclass
class VertsAndFaces:
    """
    A class that contains the vertexes and the corresponding faces.
    """

    vertices: np.ndarray
    """
    Format: ``Points::Space``
    """

    faces: np.ndarray
    """
    The corresponding faces. Format: ``Faces::Faces[Triangles]``
    """

    def get_n_bodies(self) -> int:
        """
        Get the number of separate bodies in the mesh. This function only works for "nice" (closed, no duplicate vertices, the faces correctly specified) and probably gives garbage result for other meshes.

        Returns
        -------
        v
            The number of separate bodies in the mesh. It is 1 if the mesh does not contain any vertex.

        Notes
        -----
        This function solely relies on neighborhood-relations in the mesh, not distance. In other words, the result will not be correct if the mesh contains duplicate vertices.
        """
        if self.vertices.shape[DIM_POINTS_N] == 0:
            return 1

        trimesh_mesh = trimesh.Trimesh(
            vertices=self.vertices, faces=self.faces, validate=True
        )

        n_bodies = len(trimesh_mesh.split(only_watertight=False))
        return n_bodies
