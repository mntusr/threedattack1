import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import trimesh
from panda3d.core import GeomNode, NodePath
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from skimage import measure

from ..tensor_types.idx import *
from ..tensor_types.npy import *
from ._areas import ThreeDArea
from ._data import VertsAndFaces
from ._object_creation import verts_and_faces_2_obj
from ._scene_util import get_vertices_and_faces_copy


class OccupacyField(Protocol):
    def eval_at(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Evalate the occupacy function in the specified grid.

        Parameters
        ----------
        x
            The array of the X coordinates. It should contain float values, but there are no other restrictions on its shape except it should not be empty.
        y
            The array of the Y coordinates. It should contain float values. Its shape should be equal the shape of the array of the X coordinates.
        z
            The array of the Z coordinates. It should contain float values. Its shape should be equal the shape of the array of the X coordinates.

        Returns
        -------
        v
            The array that contains the corresponding occupacy values. Values greater than or equal 0 mean that the corresponding point is inside of the object. Values smaller than 0 mean that the corresponding point is outside of the object. The array has floating-like type. Its shape is equal the shape of the array of the x coordinates.

        Notes
        -----
        The function might be discrete or continuous depending to the implementing object.

        The function is implemented in a way that enables you to call with hundreds of thousands at points at once without significant memory usage.
        """
        ...


def _fail_if_pyembree_instaled() -> None:
    """
    This function raises an exception if pyembree is installed.

    This makes the implemented algorithms more reliable and reproducible, since this prevents Trimesh from using Embree for ray casting.

    It is also not necessarily trivial to install Embree on some plaforms, so it would not be a good idea to require it.

    Related issues:

    * `Inaccuracies in mesh.contains <https://github.com/mikedh/trimesh/issues/242>`.
    * `conda-forge installation in win10 env? <https://github.com/scopatz/pyembree/issues/14>`
    * `Apple Silicion Support <https://github.com/scopatz/pyembree/issues/33>`
    """
    try:
        import pyembree  # type: ignore

        pyembree_installed = True
    except ImportError:
        pyembree_installed = False

    if pyembree_installed:
        raise Exception(
            "In order to increase reproducibility and robustness, pyembree should not be installed to make sure trimesh does not use it."
        )


_fail_if_pyembree_instaled()


def occupacy_field_samples_2_interpolator(
    occupacy_field_samples: "OccupacyFieldSamples",
) -> RegularGridInterpolator:
    """
    Creates a grid-based interpolator based on the specified occupacy field samples.

    The interpolator uses linear interpolation. It returns with -1 for points outside of the grid.

    Parameters
    ----------
    occupacy_field_samples
        The object field to use.

    Returns
    -------
    v
        The created interpolator.
    """
    return RegularGridInterpolator(
        (
            occupacy_field_samples.get_x_coordinates(),
            occupacy_field_samples.get_y_coordinates(),
            occupacy_field_samples.get_z_coordinates(),
        ),
        occupacy_field_samples.grid,
        fill_value=-1,
        bounds_error=False,
        method="linear",
    )


class InterpolatedOccupacyField:
    """
    An occupacy field that works by interpolating the samples of an other occupacy field.

    The bounds of the field basically follow the original bounds, but they are larger than them, with 0.2%.

    Parameters
    ----------
    samples
        The interpolated samples.
    """

    def __init__(self, samples: "OccupacyFieldSamples"):
        self.__interp: RegularGridInterpolator = occupacy_field_samples_2_interpolator(
            samples
        )

    @staticmethod
    def _increase_bounds(bounds: tuple[float, float]) -> tuple[float, float]:
        """
        Increase the specified range using the following formula: ``(x, y) |-> (x-(y-x)/1000, y+(y-x)/1000)``
        """
        coord_range = bounds[1] - bounds[0]
        bound_change = coord_range / 1000
        return (bounds[0] - bound_change, bounds[1] + bound_change)

    def eval_at(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()

        points = np.stack([x_flat, y_flat, z_flat], axis=1)

        result_flat = self.__interp(points)

        result = result_flat.reshape(x.shape)
        return result


class WarpField:
    def __init__(
        self,
        control_points: np.ndarray,
        vectors: np.ndarray,
        field_fn: OccupacyField,
    ):
        self.interpolator = LinearNDInterpolator(
            points=control_points, values=vectors, fill_value=0
        )
        self.field_fn = field_fn
        self._vectors = vectors

    def eval_at(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()

        points = np.stack([x_flat, y_flat, z_flat], axis=1)

        vectors = self.interpolator(points)

        transformed_points = points + vectors

        x = transformed_points[:, 0].reshape(x.shape)
        y = transformed_points[:, 1].reshape(y.shape)
        z = transformed_points[:, 2].reshape(z.shape)

        field_grid = self.field_fn.eval_at(x, y, z)

        return field_grid


@dataclass
class OccupacyFieldSamples:
    """
    An occupacy field sampled in the points of regular a grid assuming that the values outside of this grid should be treated as if they were -1, regardless of their original values.

    All coordinates are in the local coordinate system of the object.
    """

    d_coord: float
    """
    The difference between the neighbor points of the grid alongside a single axis.
    """

    x_min: float
    """
    The X coordinate of the point that is described by the ``(0, 0, 0)`` point in the grid.
    """

    y_min: float
    """
    The Y coordinate of the point that is described by the ``(0, 0, 0)`` point in the grid.
    """

    z_min: float
    """
    The Z coordinate of the point that is described by the ``(0, 0, 0)`` point in the grid.
    """

    grid: np.ndarray
    """
    The sampled values of the occupacy field. Format: ``FieldGrid::OccupacyFieldGrid``
    """

    def save_npz(self, npz_path: Path) -> None:
        """
        Save the sampled occupacy field to an npz file.
        """
        np.savez(
            str(npz_path),
            grid=self.grid,
            d_coord=self.d_coord,
            x_min=self.x_min,
            y_min=self.y_min,
            z_min=self.z_min,
        )

    @staticmethod
    def load_npz(npz_path: Path) -> "OccupacyFieldSamples":
        """
        Load the the sampled occupacy field from an npz file.

        Raises
        ------
        OSError
            If the file was not loadable.
        """
        with np.load(str(npz_path)) as data:
            return OccupacyFieldSamples(
                grid=data["grid"],
                d_coord=data["d_coord"],
                x_min=data["x_min"],
                y_min=data["y_min"],
                z_min=data["z_min"],
            )

    def get_n_x_steps(self) -> int:
        """Get the number of sampling steps along the X axis."""
        return self.grid.shape[DIM_FIELDGRID_X]

    def get_n_y_steps(self) -> int:
        """Get the number of sampling steps along the Y axis."""
        return self.grid.shape[DIM_FIELDGRID_Y]

    def get_n_z_steps(self) -> int:
        """Get the number of sampling steps along the Z axis."""
        return self.grid.shape[DIM_FIELDGRID_Z]

    def get_x_coordinates(self) -> np.ndarray:
        """
        Get the monothonically increasing series of the X coordinates of the sampled points.

        Returns
        -------
        v
            The coordinates. Format: ``Coords::Float``
        """
        return np.arange(0, self.get_n_x_steps()) * self.d_coord + self.x_min

    def get_y_coordinates(self) -> np.ndarray:
        """
        Get the monothonically increasing series of the Y coordinates of the sampled points.

        Returns
        -------
        v
            The coordinates. Format: ``Coords::Float``
        """
        return np.arange(0, self.get_n_y_steps()) * self.d_coord + self.y_min

    def get_z_coordinates(self) -> np.ndarray:
        """
        Get the monothonically increasing series of the Z coordinates of the sampled points.

        Returns
        -------
        v
            The coordinates. Format: ``Coords::Float``
        """
        return np.arange(0, self.get_n_z_steps()) * self.d_coord + self.z_min

    def contains_no_object(self) -> bool:
        """True if the grid is empty or its maximal value is non-positive."""
        return (self.grid.size == 0) or (self.grid.max() < 0)


def occupacy_field_2_occupacy_field_samples(
    occupacy_field: OccupacyField, d_coord: float, relevant_area: ThreeDArea
) -> "OccupacyFieldSamples":
    """
    Get the samples of the occupacy field in the specified area

    This function treats the space outside of the relevant area as if it was emtpy.

    Parameters
    ----------
    occupacy_field
        The occupacy field to sample.
    d_coord
        The sampling step alongside a single dimension. It should be positive.
    relevant_area
        The relevant area.

    Returns
    -------
    v
        The occupacy field samples.
    """

    x_min, x_max = relevant_area.get_x_bounds()
    y_min, y_max = relevant_area.get_y_bounds()
    z_min, z_max = relevant_area.get_z_bounds()

    x_steps = np.arange(x_min, x_max + d_coord / 1000, d_coord)
    y_steps = np.arange(y_min, y_max + d_coord / 1000, d_coord)
    z_steps = np.arange(z_min, z_max + d_coord / 1000, d_coord)

    x_grid, y_grid, z_grid = np.meshgrid(x_steps, y_steps, z_steps)
    fn_grid = occupacy_field.eval_at(x_grid, y_grid, z_grid)

    return OccupacyFieldSamples(
        grid=fn_grid, d_coord=d_coord, x_min=x_min, y_min=y_min, z_min=z_min
    )


def get_d_coord_from_obj_size_along_the_shortest_axis(
    area: ThreeDArea,
    n_steps_along_shortest_axis: int,
) -> float:
    """
    Pseudocode:

    1. Calculate the minimum of the dimensions of the object (``dim_min``)
    2. ``return dim_min / (n_steps_along_shortest_axis-1)``
    """
    if n_steps_along_shortest_axis < 2:
        raise ValueError(
            f'Argument "n_steps_along_shortest_axis" should be at least 2. Current value: {n_steps_along_shortest_axis}'
        )

    d_coord = area.get_min_size() / (n_steps_along_shortest_axis - 1)
    return d_coord


def verts_and_faces_2_occupacy_field_samples(
    verts_and_faces: VertsAndFaces,
    n_steps_along_shortest_axis: int,
) -> "OccupacyFieldSamples":
    """
    Evaluate the occupacy field of the object in a specified grid, assuming that the value of the occupacy field of the object is -1 outside of the object and 1 inside of the object.

    This uses the following algorithm to calculate the occupacy field:

    1. ``d_coord := smallest_dimension_of_the_object/n_steps_along_shortest_axis``
    2. Create an evenly spaced grid in the bounding box of the object, where the distance between neighbor points alongside a single axis is ``d_coord``. This grid contains exactly ``n_steps_along_shortest_axis`` points alongside the axis alonside which the smallest dimension of the object is.
    3. Check whether the mesh contains each point.

    Parameters
    ----------
    verts_and_faces
        The mesh to sample.
    n_steps_along_shortest_axis
        The number of steps in the grid alongside the axis along which the object the shortest is.


    Returns
    -------
    v
        The specification of the occupacy field.

    Raises
    ------
    ValueError
        If ``n_steps_along_shortest_axis < 3``.
        If the number of vertexes or faces is equal to 0.

    Notes
    -----
    This function might take multiple minutes to run, depending on the exact mesh.

    This function uses the full area occupied by the object and does not have support to sample only a subset of it.
    """
    if n_steps_along_shortest_axis < 3:
        raise ValueError(
            f"The number of steps alongside the shortest dimension should be at least 3. Current value: {n_steps_along_shortest_axis}"
        )

    if verts_and_faces.vertices.shape[DIM_POINTS_N] == 0:
        raise ValueError(f"The number of vertexes is equal to 0.")

    if verts_and_faces.faces.shape[DIM_POINTS_N] == 0:
        raise ValueError(f"The number of faces is equal to 0.")

    trimesh_mesh = trimesh.Trimesh(
        vertices=verts_and_faces.vertices, faces=verts_and_faces.faces
    )

    verts_x = trimesh_mesh.vertices[:, 0]
    verts_y = trimesh_mesh.vertices[:, 1]
    verts_z = trimesh_mesh.vertices[:, 2]

    x_min = verts_x.min()
    x_max = verts_x.max()
    y_min = verts_y.min()
    y_max = verts_y.max()
    z_min = verts_z.min()
    z_max = verts_z.max()

    d_coord = get_d_coord_from_obj_size_along_the_shortest_axis(
        area=ThreeDArea(
            x_bounds=(x_min, x_max),
            y_bounds=(y_min, y_max),
            z_bounds=(z_min, z_max),
        ),
        n_steps_along_shortest_axis=n_steps_along_shortest_axis,
    )

    x_steps = np.arange(x_min, x_max + d_coord / 1000, d_coord)
    y_steps = np.arange(y_min, y_max + d_coord / 1000, d_coord)
    z_steps = np.arange(z_min, z_max + d_coord / 1000, d_coord)

    x_grid, y_grid, z_grid = np.meshgrid(x_steps, y_steps, z_steps, indexing="ij")

    points = np.stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()], axis=1)

    containment_mask_batches = []

    # reduce peak memory usage
    batch_size = 200
    n_batches = math.ceil(len(points) / batch_size)
    for batch_idx in range(n_batches):
        containment_mask_batch = trimesh_mesh.contains(
            points[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        )
        containment_mask_batches.append(containment_mask_batch)

    containment_mask = np.concatenate(containment_mask_batches)

    containment_vals_flat = np.full(fill_value=-1, shape=points.shape[0])
    containment_vals_flat[containment_mask] = 1

    containment_grid = containment_vals_flat.reshape(x_grid.shape)

    return OccupacyFieldSamples(
        grid=containment_grid,
        d_coord=d_coord,
        x_min=x_min,
        y_min=y_min,
        z_min=z_min,
    )


def occupacy_field_samples_2_verts_and_faces(
    occupacy_field_samples: OccupacyFieldSamples,
) -> VertsAndFaces:
    """
    A modification of the Scikit-image Marching Cubes algorithm that makes sure that the volume is closed if the most outer points in the volume grid are inside of the object and properly positions the created mesh.

    This function organizes the order of the corners of the faces to match the order used by panda3d-simplepbr

    Unlike ``skimage.measure.marching_cubes``, this function does not raise any exception if there is no object in the mesh.

    The gradient is assumed to be ``descent``.

    Parameters
    ----------
    occupacy_field_samples
        The occupacy field samples on which the algorithm should be run.

    Returns
    -------
    v
        The created mesh.

    Raises
    ------
    ValueError
        If the volume does not contain at least 3 values in each direction.
    """
    if (
        (occupacy_field_samples.get_n_x_steps() < 3)
        or (occupacy_field_samples.get_n_y_steps() < 3)
        or (occupacy_field_samples.get_n_z_steps() < 3)
    ):
        raise ValueError(
            f"The grid should contain at least 3 points in each dimension. Current volume shape: {occupacy_field_samples.grid.shape}"
        )

    if occupacy_field_samples.grid.min() < 0 < occupacy_field_samples.grid.max():
        volume_extended = np.full(
            fill_value=-1,
            shape=newshape_fieldgrid(
                x=occupacy_field_samples.get_n_x_steps() + 2,
                y=occupacy_field_samples.get_n_y_steps() + 2,
                z=occupacy_field_samples.get_n_z_steps() + 2,
            ),
            dtype=occupacy_field_samples.grid.dtype,
        )
        upd_fieldgrid_occupacyfieldgrid(
            volume_extended,
            x=slice(1, occupacy_field_samples.get_n_x_steps() + 1),
            y=slice(1, occupacy_field_samples.get_n_y_steps() + 1),
            z=slice(1, occupacy_field_samples.get_n_z_steps() + 1),
            value_=occupacy_field_samples.grid,
        )

        # the marching_cubes function uses a different dimension order
        d_coord = occupacy_field_samples.d_coord
        verts, faces, _, _ = measure.marching_cubes(
            volume=volume_extended.transpose(
                [DIM_FIELDGRID_Y, DIM_FIELDGRID_X, DIM_FIELDGRID_Z]
            ),
            level=0,
            spacing=(d_coord, d_coord, d_coord),
        )

        upd_points_space(
            verts,
            data="x",
            value_=lambda a: a
            + occupacy_field_samples.x_min
            - occupacy_field_samples.d_coord,
        )
        upd_points_space(
            verts,
            data="y",
            value_=lambda a: a
            + occupacy_field_samples.y_min
            - occupacy_field_samples.d_coord,
        )
        upd_points_space(
            verts,
            data="z",
            value_=lambda a: a
            + occupacy_field_samples.z_min
            - occupacy_field_samples.d_coord,
        )

        return VertsAndFaces(
            vertices=verts, faces=idx_faces_faces(faces, corner=[0, 2, 1])
        )
    else:
        return VertsAndFaces(
            vertices=np.zeros(newshape_points_space(n=0)),
            faces=np.zeros(newshape_faces_faces(face=0, corner=3)),
        )
