import copy
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, TypeGuard, cast

import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexArrayData,
    GeomVertexData,
    GeomVertexFormat,
    GraphicsOutput,
    GraphicsWindow,
    Lens,
    LoaderOptions,
    NodePath,
    PandaNode,
    PerspectiveLens,
    WindowProperties,
)

from ..dataset_model import CamProjSpec, DepthsWithMasks
from ..tensor_types.idx import *
from ..tensor_types.npy import *
from ._data import ThreeDSize, TwoDAreas, TwoDSize, VertsAndFaces
from ._errors import Panda3dAssumptionViolation


def get_obj_size(obj: NodePath) -> ThreeDSize:
    """
    Get the size of the tight bounding box of the specified object in the local coordinate system.

    Parameters
    ----------
    obj
        The object to measure.

    Returns
    -------
    v
        The calculated size.

    Raises
    ------
    ValueError
        If the object or its descendants do not contain any vertex.

    Notes
    -----
    This function uses `panda3d.core.NodePath.getTightBounds` to calculate the bounding box.
    """
    tight_bounds = obj.getTightBounds()

    if tight_bounds is None:
        raise ValueError(f'The object "{obj.name}" does not contain any point.')

    min_point, max_point = tight_bounds
    x_size = max_point.x - min_point.x
    y_size = max_point.y - min_point.y
    z_size = max_point.z - min_point.z

    return ThreeDSize(x_size=x_size, y_size=y_size, z_size=z_size)


def get_cam_proj_spec_for_showbase(base: ShowBase) -> CamProjSpec:
    """
    Get the projection properties of the lens of the default camera of a particular scene.

    This function assumes without checking that Panda3d uses the Z-up right handed coordinate system.

    Parameters
    ----------
    scene
        The scene.

    Returns
    -------
    v
        The projection.

    Raises
    ------
    Panda3dAssumptionViolation
        If the lens is not a `panda3d.core.PerspectiveLens` or its near or far plane is infinite or they are reversed.
    """
    return get_cam_proj_spec_for_lens(base.cam.node().getLens())


def get_cam_proj_spec_for_lens(lens: Lens) -> CamProjSpec:
    """
    Get the projection properties of a particular lens.

    This function assumes without checking that Panda3d uses the Z-up right handed coordinate system.

    Parameters
    ----------
    cam_lens
        The lens of the camera.

    Returns
    -------
    v
        The projection.

    Raises
    ------
    Panda3dAssumptionViolation
        If the lens is not a `panda3d.core.PerspectiveLens` or its near or far plane is infinite or they are reversed.
    """
    if not isinstance(lens, PerspectiveLens):
        raise Panda3dAssumptionViolation("Only PerspectiveLens is supported.")
    if _is_near_far_inf_or_swapped(lens):
        raise Panda3dAssumptionViolation(
            "Only lenses with finite near and far plane are supported."
        )

    """
    In panda3d, the projection matrix projects the image to the (x in [-1, 1], y in [-1, 1]) range, regardless of the film size.

    As an implementation detail, it does depend on the film size, but just to eliminate its effect.
    """
    proj_mat_col = get_projection_mat_col(lens)
    proj_mat_col = idx_mat(proj_mat_col, row=[0, 1, 3])

    return CamProjSpec(
        im_left_x_val=-1,
        im_right_x_val=1,
        im_top_y_val=1,
        im_bottom_y_val=-1,
        proj_mat=proj_mat_col,
    )


def _is_near_far_inf_or_swapped(lens: Lens) -> bool:
    """
    Return true if the near or the far plane of the lens is infinite or they are swapped.
    """
    near = lens.getNear()
    far = lens.getFar()

    if math.isinf(near):
        return True
    if math.isinf(far):
        return True

    if far < near:
        return True

    return False


def get_obj_copy(obj: NodePath) -> PandaNode:
    """
    Create a deepcopy of an object. This only copies the single node that is directly controlled by the NodePath.

    Parameters
    ----------
    obj
        The object to copy.

    Returns
    -------
    v
        A deep copy of the object.

    Raises
    ------
    ValueError
        If the specified NodePath is empty.
    """
    if obj.isEmpty():
        raise ValueError("The specified NodePath is empty.")
    return copy.deepcopy(obj.node())


def put_obj(path: NodePath, new_obj: PandaNode) -> None:
    """
    Replace the object at the specified node path.

    This function uses the `panda3d.core.NodePath.removeNode` to remove the node at the specified node path.

    Parameters
    ----------
    path
        The node path to remove.
    new_obj
        The object to add to the parent of `path`.

    Raises
    ------
    Panda3dAssumptionViolation
        If the object does not have any parent.
    """
    if not path.hasParent():
        raise Panda3dAssumptionViolation("The object does not have any parent.")
    parent: NodePath = path.getParent()
    path.removeNode()
    parent.attachNewNode(new_obj)


def zbuf_2_depth_and_mask(
    zbuf_data: np.ndarray, camera_lens: PerspectiveLens, max_depth: float
) -> DepthsWithMasks:
    """
    Convert the Z-buffer data to a depth map.

    This function substitutes 0 into the masked points.

    Parameters
    ----------
    zbuf_data
        The data in the Z-buffer. Format: ``Im::ZBuffers[Single]``
    camera_lens
        The lens of the camera that that captured the image.
    max_depth
        An additional maximum of the depth. When calculationg the mask, the MINIMUM of this and the sky mask limit is used.

    Raises
    ------
    Panda3dAssumptionViolation
        If the lens has infinite far or near plane.
    ValueError
        If the image does not have a matching shape.
        If ``max_depth`` is non-positive.
    """
    if not dmatch_im(zbuf_data, shape={"n": 1}):
        raise ValueError("The specified tensor does not have a valid zbuffer shape.")

    if max_depth <= 0:
        raise ValueError(
            f'The argument "max_depth" should be positive. Current value: {max_depth}'
        )

    P = get_projection_mat_col(camera_lens)

    b = P[2, 3]
    a = P[2, 1]
    true_depth_data: np.ndarray = -b / (a - 2 * zbuf_data + 1)

    sky_mask_limit = get_sky_mask_limit(camera_lens)
    _, far = get_near_far_planes_safe(camera_lens)

    mask = true_depth_data < min(sky_mask_limit, max_depth)
    true_depth_data[~mask] = 0

    return DepthsWithMasks(depths=true_depth_data, masks=mask)


def get_sky_mask_limit(lens: PerspectiveLens) -> float:
    """
    Get a depth limit value that enables the separation of the sky in the depth maps.

    Parameters
    ----------
    lens
        The lens of the camera that produces the depth maps.

    Returns
    -------
    v
       The depth limit value.

    Raises
    ------
    Panda3dAssumptionViolation
        If the lens has infinite far or near plane.
    """
    near, far = get_near_far_planes_safe(lens)

    return near + (far - near) * 0.99


def get_near_far_planes_safe(lens: Lens) -> tuple[float, float]:
    """
    Get the near and far planes of the specified lens.

    This function makes sure that it is "relatively safe" to use these numbers.

    Parameters
    ----------
    lens
        The lens on which these properties should be calculated.

    Returns
    -------
    near
        The calculated near plane.
    far
        The calculated far plane.

    Raises
    ------
    Panda3dAssumptionViolation
        If the lens has infinite far or near plane or the near and far planes are swapped.
    """
    near = lens.getNear()
    far = lens.getFar()

    if math.isinf(far):
        raise Panda3dAssumptionViolation(
            "The far plane should not be infinite for this function to wrok properly."
        )

    if math.isinf(near):
        raise Panda3dAssumptionViolation(
            "The far plane shoul not be infinite for this function to wrok properly."
        )

    if near < far:
        return near, far
    else:
        raise Panda3dAssumptionViolation(
            "The handling of lenses with swapped near and fare planes is not supported."
        )


def get_projection_mat_col(lens: Lens) -> np.ndarray:
    """
    Get the transposed projection matrix of the specified `Lens`.

    This function returns the transposed projection matrix regardless of the actual type of the lens. In other words, the proper handling of this projection matrix is the responsibility of the caller.

    Parameters
    ----------
    lens
        The lens.

    Returns
    -------
    v
        Format: ``Mat::Float``

    Notes
    -----
    This function returns the transposed projection matrix, since Panda3d uses row vectors instead of column vectors for projection.
    """
    P = lens.getProjectionMat()
    return np.asarray(P).T


def get_projection_inv_mat_col(lens: Lens) -> np.ndarray:
    """
    Get the transposed inverse projection matrix of the specified `Lens`.

    This function returns the transposed projection matrix regardless of the actual type of the lens. In other words, the proper handling of this projection matrix is the responsibility of the caller.

    Parameters
    ----------
    lens
        The lens.

    Returns
    -------
    v
        Format: ``Mat::Float``

    Notes
    -----
    This function returns the transposed inverse projection matrix, since Panda3d uses row vectors instead of column vectors for projection.
    """
    P = lens.getProjectionMatInv()
    return np.asarray(P).T


def find_node(root: NodePath, obj_path: str) -> Optional[NodePath]:
    """
    Find the specified node path. This function is similar to `panda3d.core.NodePath.findNode`, however it returns `None` if the node path was not found, instead of returning an empty `panda3d.core.NodePath`.

    Parameters
    ----------
    root
        The root node.
    obj_path
        The path expression to the node.

    Returns
    -------
    v
        The `panda3d.core.NodePath` if found, otherwise `None`.
    """
    search_result = root.find(obj_path)

    if search_result.isEmpty():
        return None
    else:
        return search_result


def get_properties_copy(buffer: GraphicsOutput) -> WindowProperties:
    """
    Get a copy of the `panda3d.core.WindowProperties` of the specified buffer.

    If the buffer is a `panda3d.core.GraphicsWindow`, then it uses the copy constructor of `panda3d.core.WindowProperties`. Otherwise, it creates a new `panda3d.core.WindowProperties` from the width and the height of the buffer.

    Parameters
    ----------
    buffer
        The buffer to use.

    Returns
    -------
    v
        The created `panda3d.core.WindowProperties`.
    """
    if isinstance(buffer, GraphicsWindow):
        return WindowProperties(buffer.getProperties())
    else:
        props = WindowProperties()
        props.setSize(x_size=buffer.getXSize(), y_size=buffer.getYSize())
        return props


def get_ob_size_from_vertices(vertex_positions: np.ndarray) -> ThreeDSize:
    """
    Get the size of the bounding box of the object with the specified vertexes in local coordinate system.

    Parameters
    ----------
    vertex_positions
        The vertexes of the object. Format: ``Points::Space``

    Raises
    ------
    ValueError
        If the array does not contain vertex data.
    """
    if not match_points_space(vertex_positions):
        raise ValueError("The specified array does not contain vertex data.")

    if vertex_positions.shape[DIM_POINTS_N] == 0:
        return ThreeDSize(0, 0, 0)

    x_range = float(
        idx_points(vertex_positions, data=CAT_POINTS_SPACE_DATA_X).max()
        - idx_points(vertex_positions, data=CAT_POINTS_SPACE_DATA_X).min()
    )
    y_range = float(
        idx_points(vertex_positions, data=CAT_POINTS_SPACE_DATA_Y).max()
        - idx_points(vertex_positions, data=CAT_POINTS_SPACE_DATA_Y).min()
    )
    z_range = float(
        idx_points(vertex_positions, data=CAT_POINTS_SPACE_DATA_Z).max()
        - idx_points(vertex_positions, data=CAT_POINTS_SPACE_DATA_Z).min()
    )
    return ThreeDSize(x_size=x_range, y_size=y_range, z_size=z_range)


def get_vertex_count(obj: NodePath) -> int:
    """
    Get the number of vertexes in the object.

    If the GeomNode of this object does not have any vertex, then the function returns with 0.

    Raises
    ------
    Panda3dAssumptionViolation
        If the NodePath does not have directly a geometry node.
    """
    geom_node = _get_the_sole_geometry_node(obj)
    vertex_count = _get_vert_count_of_geom_node(geom_node)
    return vertex_count


def is_geom_node_obj(obj: NodePath) -> "TypeGuard[NodePath[GeomNode]]":
    node = obj.node()
    return isinstance(node, GeomNode)


def get_vertices_and_faces_copy(obj: NodePath) -> VertsAndFaces:
    """
    Copies the vertices and faces of the specified object.

    Parameters
    ----------
    obj
        The initial object.

    Returns
    -------
    v
        The copied vertexes and matrices.

    Raises
    ------
    Panda3dAssumptionViolation
        If the NodePath does not directly have any node or that node is not a GeomNode.
        If the specified geom node does not have any vertex or some of the vertexes do not have specified positions or the vertexes of this array has any non-float data.
        If the specified geom node does not have at least one primitive or the specified primitive does not have type `panda3d.core.GeomTriangles`.
        If the object has at least one child.
        If at least one Geom of the node is non-indexed.
    """
    if obj.getNumChildren() > 0:
        raise Panda3dAssumptionViolation(
            f'The object "{obj.name}" has at least one children. Number of children: {obj.getNumChildren()}.'
        )
    geom_node = _get_the_sole_geometry_node(obj)
    vertex_positions = _get_vertex_positions_copy_from_geom_node(geom_node)
    faces = _get_faces_copy_from_geom_node(geom_node)

    return VertsAndFaces(vertices=vertex_positions, faces=faces)


def get_vertex_face_copy_most_common_errors(obj: NodePath) -> list[str]:
    """
    Generally, the face and vertex copying of an object might fail for numerous reasons.

    This function tries to detect the most common errors.


    Parameters
    ----------
    obj
        The object to check.

    Returns
    -------
    v
        The most common errors that might prevent the copying of the vertexes and faces of the object.
    """
    errors: list[str] = []

    if obj.getNumChildren() > 0:
        errors.append("The objecth has children.")

    if not is_geom_node_obj(obj):
        errors.append("The object is not a GeomNode.")
        return errors

    if obj.node().getNumGeoms() == 0:
        errors.append("The object does not have any Geom.")
        return errors

    for geom in obj.node().getGeoms():
        n_primitives = geom.getNumPrimitives()
        if n_primitives == 0:
            errors.append(
                "At least one geom of the object does not have any primitive."
            )

        for primitive in geom.getPrimitives():
            if not isinstance(primitive, GeomTriangles):
                errors.append(
                    "A primitive of a geom of the object is not GeomTriangles."
                )

    return errors


def get_vertex_positions_copy(obj: NodePath) -> np.ndarray:
    """
    Copy the vertex positions from the specified object to a numpy array.

    The returned array is a copy, i. e. its value is not affected by the subsequent changes of the vertexes of the original object.

    Parameters
    ----------
    obj
        The relevant object.

    Returns
    -------
    v
        The vertexes. Format: ``Points::Space``

    Raises
    ------
    Panda3dAssumptionViolation
        If the object does not directly have a GeomNode with at least one vertex or the vertexes of this object have any non-float data.
    """
    geom_node = _get_the_sole_geometry_node(obj)
    vertex_positions = _get_vertex_positions_copy_from_geom_node(geom_node)

    if vertex_positions is None:
        raise Panda3dAssumptionViolation("No GeomNodes were found in the object.")

    return vertex_positions


def _get_the_sole_geometry_node(model: NodePath) -> GeomNode:
    """
    Get the list of all geometry nodes in the specified model.

    Parameters
    ----------
    model
        The model.

    Returns
    -------
    v
        The geometry node.

    Raises
    ------
    Panda3dAssumptionViolation
        If the NodePath does not directly have any node or that node is not a GeomNode.
    """

    if not directly_contains_geometry_data(model):
        raise Panda3dAssumptionViolation("The object does not have a geometry node.")

    return model.node()


def directly_contains_geometry_data(obj: NodePath) -> "TypeGuard[NodePath[GeomNode]]":
    if obj.isEmpty():
        return False

    node = obj.node()

    return isinstance(node, GeomNode)


def _get_vert_count_of_geom_node(geom_node: GeomNode) -> int:
    """
    Get the number of the vertexes in the specified geom node.

    If the geom node does not have any vertex, then this function returns 0.
    """
    acc = 0
    for geom in geom_node.getGeoms():
        vertex_data = geom.getVertexData()
        acc += vertex_data.getNumRows()

    return acc


def _get_vertex_positions_copy_from_geom_node(geom_node: GeomNode) -> np.ndarray:
    """
    Get the copy of the vertex positions of specified geom node in the local coordinate system.

    Returns
    -------
    v
        The copy of the vertex positions. Format: ``Points::Space``

    Raises
    ------
    Panda3dAssumptionViolation
        If the specified geom node does not have any vertex or some of the vertexes do not have specified positions or the vertexes of this array has any non-float data.
    """
    total_points: Optional[np.ndarray] = None
    for i in range(geom_node.getNumGeoms()):
        geom = geom_node.modifyGeom(i)
        vertex_data = geom.modifyVertexData()
        vertex_data_points = get_col_copy_from_vertex_data(
            col_name=_VERTEX_POS_COL_NAME_IN_PANDA3D, vertex_data=vertex_data
        )
        if total_points is None:
            total_points = vertex_data_points
        else:
            total_points = np.concatenate([total_points, vertex_data_points], axis=0)

    if total_points is None:
        raise Panda3dAssumptionViolation("The GeomNode has no GeomVertexData objects.")

    return total_points


def _get_faces_copy_from_geom_node(geom_node: GeomNode) -> np.ndarray:
    """
    Get the copy of the faces of specified geom node in the local coordinate system.

    Returns
    -------
    v
        The copy of the vertex positions. Format: ``Faces::Faces[Triangles]``

    Raises
    ------
    Panda3dAssumptionViolation
        If the specified geom node does not have at least one primitive or the specified primitive does not have type `panda3d.core.GeomTriangles`.
        If at least one Geom of the node is non-indexed.
    """
    total_faces: Optional[np.ndarray] = None
    for i in range(geom_node.getNumGeoms()):
        geom = geom_node.modifyGeom(i)
        n_geom_primitives = geom.getNumPrimitives()
        if n_geom_primitives == 0:
            raise Panda3dAssumptionViolation(
                f"The object {geom_node.name} has more than one geom primitives. Number of geom primitives: {n_geom_primitives}"
            )
        for i_primitive in range(n_geom_primitives):
            geom_primitive = geom.getPrimitive(i_primitive)
            if not isinstance(geom_primitive, GeomTriangles):
                raise Panda3dAssumptionViolation(
                    f"The type of the geom primitives of {geom_node.name} is not GeomTriangles."
                )
            faces = _get_faces_copy_from_primitive(geom_primitive)
            offset = 0 if total_faces is None else total_faces.shape[DIM_FACES_FACE]
            faces = faces + offset

            if total_faces is None:
                total_faces = faces
            else:
                total_faces = np.concatenate([total_faces, faces], axis=DIM_FACES_FACE)

    if total_faces is None:
        raise Panda3dAssumptionViolation("The GeomNode has no GeomVertexData objects.")

    return total_faces


def _get_faces_copy_from_primitive(geom_primitive: GeomTriangles) -> np.ndarray:
    """
    Copies the faces from the specified primitive to a Numpy array.

    Parameters
    ----------
    geom_primitive
        The source primitive.

    Returns
    -------
    v
        The array of the copied faces. ``Faces::Faces[Triangles]``

    Raises
    ------
    Panda3dAssumptionViolation
        If the geom is non-indexed.
    """
    faces_native: GeomVertexArrayData | None = geom_primitive.getVertices()
    if faces_native is None:
        raise Panda3dAssumptionViolation(
            "The copying of the faces from a non-indexed GeomPrimitive is not implemented."
        )
    geom_primitive_view_raw = memoryview(faces_native)  # type: ignore
    geom_primitive_view_casted = geom_primitive_view_raw.cast("B").cast("H")
    faces_copy_array = np.array(geom_primitive_view_casted).reshape((-1, 3))
    return faces_copy_array


def set_col_in_vertex_data(
    vertex_data: GeomVertexData,
    col_name: str,
    new_values: np.ndarray,
) -> None:
    """
    Transform the values of the specfied column of a `panda3d.core.GeomVertexData`.

    This function assumes without checking that all columns use the float data type.

    Parameters
    ----------
    vertex_data
        The `panda3d.core.GeomVertexData` to modify.
    col_name
        The name of the column to modify.
    new_values
        The array of the new values. This is a two dimensional array.

    Raises
    ------
    Panda3dAssumptionViolation
        If the column does not exist or the data types of all columns in the specified geom vertex data are not float32.
    """

    col_info = _get_column_info(col_name, vertex_data)
    v_array = vertex_data.modifyArray(col_info.array_index_in_geom_vertex_data)
    float_view = _get_memoryview_for_pure_float_array_data(v_array)
    np_array = np.asarray(float_view).reshape((v_array.getNumRows(), -1))
    np_array[:, col_info.index_range_in_array] = new_values.astype(np.float32)
    float_view[:] = np_array.reshape((-1,))  # type: ignore


def get_col_copy_from_vertex_data(
    col_name: str, vertex_data: GeomVertexData
) -> np.ndarray:
    """
    Get the copy of the specified column in the specified geom vertex data.

    Parameters
    ----------
    col_name
        The name of the column to copy.
    vertex_data
        The geom vertex data to which the column belongs.

    Returns
    -------
    v
        The copy of the column. Format: ``Points::ArbData[NonEmpty]``

    Raises
    ------
    Panda3dAssumptionViolation
        If the column does not exist or the data types of all columns in the specified geom vertex data are not float32.
    """
    if vertex_data.getNumRows() == 0:
        raise Panda3dAssumptionViolation("The GeomVertexData has no rows.")

    col_info = _get_column_info(col_name, vertex_data)
    v_array = vertex_data.getArray(col_info.array_index_in_geom_vertex_data)
    float_view = _get_memoryview_for_pure_float_array_data(v_array)
    np_array = np.asarray(float_view).reshape((v_array.getNumRows(), -1))
    col_array = np_array[:, col_info.index_range_in_array].copy()
    return col_array


def get_all_vertex_arrays_copy_from_vertex_data(
    vertex_data: GeomVertexData,
) -> list[np.ndarray]:
    """
    Copy the full arrays as vertex data.

    This function is mostly useful for debugging purposes.

    Parameters
    ----------
    vertex_data
        The vertex data from which the arrays should be copied.

    Returns
    -------
    v
        The list of the raw arrays. Format: ``Table::Float``

    Raises
    ------
    Panda3dAssumptionViolation
        If the column does not exist or the data types of all columns in the specified geom vertex data are not float32.

    See Also
    --------

    Notes
    -----
    """
    arrays: list[np.ndarray] = []
    for array_idx in range(vertex_data.getNumArrays()):
        v_array = vertex_data.getArray(array_idx)
        float_view = _get_memoryview_for_pure_float_array_data(v_array)
        np_array = np.asarray(float_view).reshape((v_array.getNumRows(), -1))
        arrays.append(np_array)
    return arrays


def _get_column_info(col_name: str, geom_vertex_data: GeomVertexData) -> "_ColumnInfo":
    for i_arr, arr in enumerate(geom_vertex_data.getArrays()):
        arr_format = arr.getArrayFormat()
        columns = arr_format.getColumns()
        len_acc = 0
        for col in columns:
            if str(col.getName()) == col_name:
                array_idxs = list(np.arange(0, col.getNumValues()) + len_acc)
                return _ColumnInfo(
                    index_range_in_array=array_idxs,
                    array_index_in_geom_vertex_data=i_arr,
                )
            else:
                len_acc += col.getNumValues()
    else:
        raise Panda3dAssumptionViolation(f'The column "{col_name}" was not found.')


def _get_memoryview_for_pure_float_array_data(
    array_data: GeomVertexArrayData,
) -> memoryview:
    raw_view = memoryview(array_data)  # type: ignore
    total_len = raw_view.nbytes // 4
    float_view = raw_view.cast("B").cast("f", (total_len,))

    return float_view


def get_bounding_rectangle_on_screen(
    points: np.ndarray, base: ShowBase, rendering_resolution: TwoDSize
) -> TwoDAreas:
    """
    Get the bounding rectangle around the specified points on screen.

    This bounding rectangle might be partially outside of the screen if some of the projected points are outside of the screen.

    Parameters
    ----------
    points
        The points in the world coordinate system. Format: ``Points::Space``
    base
        The Panda3d show base.
    screen_resolution
        The resolution of the rendered image.

    Returns
    -------
    v
        The bounding rectangle.
    """
    projected_points = project_points_to_screen(
        base=base, points=points, rendering_resolution=rendering_resolution
    )

    return get_bounding_rectangle_2d(points=projected_points)


def get_bounding_rectangle_2d(points: np.ndarray) -> TwoDAreas:
    """
    Get the bounding rectangle of the rectangular area around the specified 2d points.

    This function rounds the non-integer values using the rules of `numpy.round`.

    Parameters
    ----------
    points
        The points that select the area. Format: ``Points::Plane``
    im_size
        The size of the image.

    Returns
    -------
    v
        The areas.

    Raises
    ------
    ValueError
        If no point is specified.
    """
    if points.shape[DIM_POINTS_N] == 0:
        raise ValueError("At least one point should be specified.")

    x_coordinates = idx_points_plane(points, data="x")
    y_coordinates = idx_points_plane(points, data="y")

    return TwoDAreas(
        x_maxes_excluding=np.array([np.round(np.max(x_coordinates)).astype(np.int32)])
        + 1,
        x_mins_including=np.array([np.round(np.min(x_coordinates)).astype(np.int32)]),
        y_maxes_excluding=np.array([np.round(np.max(y_coordinates)).astype(np.int32)])
        + 1,
        y_mins_including=np.array([np.round(np.min(y_coordinates)).astype(np.int32)]),
    )


def project_points_to_screen(
    points: np.ndarray, base: ShowBase, rendering_resolution: TwoDSize
) -> np.ndarray:
    """
    Project the specified points to the screen.

    Unlike the the `panda3d.core.Lens.project`, this function projects the points to the space of the captured image, instead of the ``x in (-1, 1), y in (-1, 1)`` space and it supports the projection outside of the visible area.

    Parameters
    ----------
    points
        The points in the world coordinate system. Format: ``Points::Space``
    base
        The Panda3d show base.
    screen_resolution
        The resolution of the rendered image.

    Returns
    -------
    v
        The projected points. Format: ``Points::Plane``

    Raises
    ------
    ValueError
        If any of the values in the rendered image size is not positive.
    """
    if not rendering_resolution.is_positive():
        raise ValueError(
            f"The screen resolution is not valid. Resolution: {rendering_resolution}"
        )

    camera = base.cam

    twod_points = np.zeros(
        newshape_points_plane(n=points.shape[DIM_POINTS_N]), dtype=np.float32
    )

    cam_proj_mat = get_projection_mat_col(camera.node().get_lens())
    cam_proj_mat = idx_mat_float(cam_proj_mat, row=[0, 1, 3])

    for point_idx in range(points.shape[DIM_POINTS_N]):
        x, y, z = idx_points_space(points, n=point_idx, data=["x", "y", "z"])
        relative_point = camera.getRelativePoint(base.render, (x, y, z))
        projected_point = cam_proj_mat @ np.array(
            [[relative_point.x], [relative_point.y], [relative_point.z], [1]],
            dtype=np.float32,
        )

        upd_mat_float(
            projected_point,
            row=0,
            value_=lambda v: v / idx_mat_float(projected_point, row=-1),
        )
        upd_mat_float(
            projected_point,
            row=1,
            value_=lambda v: v / idx_mat_float(projected_point, row=-1),
        )
        projected_point = idx_mat_float(projected_point, row=[0, 1])

        projected_x, projected_y = idx_mat_float(projected_point, row=[0, 1])

        upd_points_plane(twod_points, n=point_idx, data="x", value_=projected_x)
        upd_points_plane(twod_points, n=point_idx, data="y", value_=projected_y)

    upd_points_plane(
        twod_points,
        data="x",
        value_=lambda a: a * rendering_resolution.x_size / 2
        + rendering_resolution.x_size / 2,
    )
    upd_points_plane(
        twod_points,
        data="y",
        value_=lambda a: a * rendering_resolution.y_size / 2
        + rendering_resolution.y_size / 2,
    )

    return twod_points


def load_model_from_local_file(base: ShowBase, model_path: Path) -> NodePath | None:
    """
    A simple function to load a 3d model using Panda3d from a local model file.

    Unlike the original Panda3d model loader function, this function by default provides more information, disables model caching and supports Windows-style paths.

    This function keeps the hierarchy of the objects in the model.

    Parameters
    ----------
    base
        The `direct.showbase.ShowBase.ShowBase` to load the model.
    model_path
        The path of the model.

    Returns
    -------
    v
        The `panda3d.core.NodePath` of the loaded model if the model was loadable. Otherwise None.
    """
    options = LoaderOptions(LoaderOptions.LFReportErrors | LoaderOptions.LFNoCache)
    unix_style_path = _convert_path_to_unix_style(model_path)
    model = base.loader.loadModel(unix_style_path, loaderOptions=options)
    return model


def _convert_path_to_unix_style(path: Path) -> str:
    """
    Convert the path to the Unix-style style required by Panda3d on Windows. This function makes the specified path absolute.

    This function does not change the paths in all platforms, but Windows.

    Parameters
    ----------
    path
        The path to convert.

    Returns
    -------
    v
        The path as a string.
    """
    if sys.platform != "win32":
        return str(path)

    path_parts = list(path.resolve().parts)

    assert len(path_parts) > 0
    if path_parts[0].endswith(":/") or path_parts[0].endswith(":\\"):
        path_parts[0] = path_parts[0][:-2].lower()

    full_path = "/" + ("/".join(path_parts))
    return full_path


_VERTEX_POS_COL_NAME_IN_PANDA3D = "vertex"


@dataclass
class _ColumnInfo:
    index_range_in_array: list[int]
    array_index_in_geom_vertex_data: int


@dataclass
class ObjectSize:
    x: float
    y: float
    z: float
